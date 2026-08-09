"""LLM proposal pool ranked with HEBO's transformed GP and MACE scores."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..core import Observation, SearchSpace
from ..providers import FreshProposalProvider
from .pool_rank import LLMPoolRankArm, PoolRanking, PoolRanker


def nondominated_fronts(values: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Assign zero-based Pareto front numbers for minimization objectives."""

    remaining = set(range(len(values)))
    fronts = [-1] * len(values)
    front = 0
    while remaining:
        current = []
        for candidate in sorted(remaining):
            dominated = False
            for other in remaining:
                if other == candidate:
                    continue
                if all(
                    values[other][objective_index]
                    <= values[candidate][objective_index]
                    for objective_index in range(len(values[candidate]))
                ) and any(
                    values[other][objective_index]
                    < values[candidate][objective_index]
                    for objective_index in range(len(values[candidate]))
                ):
                    dominated = True
                    break
            if not dominated:
                current.append(candidate)
        for candidate in current:
            fronts[candidate] = front
            remaining.remove(candidate)
        front += 1
    return tuple(fronts)


def _stable_rank_sums(values: Sequence[Sequence[float]]) -> tuple[int, ...]:
    sums = [0] * len(values)
    if not values:
        return ()
    for objective_index in range(len(values[0])):
        order = sorted(
            range(len(values)),
            key=lambda index: (values[index][objective_index], index),
        )
        for rank, index in enumerate(order):
            sums[index] += rank
    return tuple(sums)


class HEBOMACERanker:
    """NumPy-2-compatible port of HEBO's transformed GP + MACE scoring.

    PyPI HEBO 0.3.6 pins NumPy below 1.25, while the GPU task requires NumPy 2.
    This keeps the benchmark in one task environment and directly implements
    the small public-code surface used here: HEBO numeric/log encoding, target
    power transform, a Matern-3/2 exact GP, and the three MACE objectives.
    """

    name = "hebo_mace"

    def rank(
        self,
        space: SearchSpace,
        observations: Sequence[Observation],
        pool: Sequence[Mapping[str, Any]],
        *,
        seed: int,
    ) -> PoolRanking:
        active = space.active_dimensions
        if not active:
            empty_scores = tuple(
                {
                    "mace_lcb": 0.0,
                    "mace_neg_log_ei": 0.0,
                    "mace_neg_log_pi": 0.0,
                    "pareto_front": 0,
                    "rank_sum": 0,
                }
                for _ in pool
            )
            return PoolRanking(order=tuple(range(len(pool))), member_scores=empty_scores)

        try:
            import gpytorch
            import numpy as np
            import torch
            from sklearn.preprocessing import power_transform
        except ImportError as exc:
            raise RuntimeError(
                "HEBO MACE ranking requires gpytorch, scikit-learn, numpy, and torch"
            ) from exc

        def encode(params: Mapping[str, Any]) -> list[float]:
            projected = space.project(params)
            values: list[float] = []
            for dimension in active:
                value = projected[dimension.name]
                if dimension.kind == "categorical":
                    values.extend(
                        1.0 if value == choice else 0.0
                        for choice in dimension.choices
                    )
                    continue
                assert dimension.low is not None and dimension.high is not None
                numeric = float(value)
                low, high = float(dimension.low), float(dimension.high)
                if dimension.log:
                    numeric, low, high = math.log10(numeric), math.log10(low), math.log10(high)
                values.append(-1.0 if low == high else -1.0 + 2.0 * (numeric - low) / (high - low))
            return values

        history_x = torch.as_tensor(
            [encode(observation.params) for observation in observations],
            dtype=torch.float32,
        )
        pool_x = torch.as_tensor(
            [encode(params) for params in pool], dtype=torch.float32
        )
        raw_y = np.asarray([float(item.score) for item in observations]).reshape(-1, 1)
        y = raw_y.copy()
        try:
            std = float(raw_y.std())
            if not math.isfinite(std) or std <= 0:
                raise ValueError("constant objective history")
            method = "yeo-johnson" if raw_y.min() <= 0 else "box-cox"
            y = power_transform(raw_y / std, method=method)
            if float(y.std()) < 0.5:
                y = power_transform(raw_y / std, method="yeo-johnson")
            if float(y.std()) < 0.5:
                raise ValueError("power transformation failed")
        except (ValueError, FloatingPointError):
            y = raw_y.copy()
        y_mean, y_std = float(y.mean()), float(y.std())
        if not math.isfinite(y_std) or y_std <= 1e-12:
            y_std = 1.0
        train_y = torch.as_tensor(
            ((y.reshape(-1) - y_mean) / y_std), dtype=torch.float32
        )

        class ExactModel(gpytorch.models.ExactGP):
            def __init__(self, x, target, likelihood):
                super().__init__(x, target, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(
                        nu=1.5, ard_num_dims=x.shape[1]
                    )
                )

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        iterations = max(1, len(observations))
        kappa = math.sqrt(
            0.5
            * 2.0
            * (
                (2.0 + len(active) / 2.0) * math.log(iterations)
                + math.log(3.0 * math.pi**2 / (3.0 * 0.01))
            )
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(8e-4)
            )
            model = ExactModel(history_x, train_y, likelihood)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(
                likelihood, model
            )
            for _ in range(100):
                optimizer.zero_grad()
                loss = -marginal_log_likelihood(model(history_x), train_y)
                loss.backward()
                optimizer.step()
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pool_prediction = model(pool_x)
                pool_mean = pool_prediction.mean.reshape(-1, 1)
                pool_variance = pool_prediction.variance.reshape(-1, 1)
                best_index = int(raw_y.argmin())
                best_prediction = model(history_x[[best_index]])
                tau = best_prediction.mean.squeeze()
                noise = math.sqrt(2.0) * likelihood.noise.sqrt()
                standard_deviation = pool_variance.sqrt().clamp(
                    min=torch.finfo(pool_variance.dtype).eps
                )
                lower_confidence_bound = (
                    pool_mean + noise * torch.randn(pool_mean.shape)
                ) - kappa * standard_deviation
                normalized = (
                    tau
                    - 1e-4
                    - pool_mean
                    - noise * torch.randn(pool_mean.shape)
                ) / standard_deviation
                normal = torch.distributions.Normal(0.0, 1.0)
                log_phi = normal.log_prob(normalized)
                probability = normal.cdf(normalized)
                expected_improvement = standard_deviation * (
                    probability * normalized + log_phi.exp()
                )
                approximate_log_ei = (
                    standard_deviation.log()
                    - 0.5 * normalized**2
                    - (normalized**2 - 1).log()
                )
                approximate_log_pi = (
                    -0.5 * normalized**2
                    - torch.log(-normalized)
                    - math.log(math.sqrt(2.0 * math.pi))
                )
                use_approximation = ~(
                    (normalized > -6)
                    & torch.isfinite(expected_improvement.log())
                    & torch.isfinite(probability.log())
                ).reshape(-1)
                values = torch.zeros(len(pool), 3)
                values[:, 0] = lower_confidence_bound.reshape(-1)
                values[:, 1][use_approximation] = -approximate_log_ei[
                    use_approximation
                ].reshape(-1)
                values[:, 2][use_approximation] = -approximate_log_pi[
                    use_approximation
                ].reshape(-1)
                values[:, 1][~use_approximation] = -expected_improvement[
                    ~use_approximation
                ].log().reshape(-1)
                values[:, 2][~use_approximation] = -probability[
                    ~use_approximation
                ].log().reshape(-1)
                values = values.cpu().numpy()
        objectives = [tuple(float(value) for value in row) for row in values]
        if any(not math.isfinite(value) for row in objectives for value in row):
            raise RuntimeError("HEBO MACE returned a non-finite pool score")

        fronts = nondominated_fronts(objectives)
        rank_sums = _stable_rank_sums(objectives)
        order = tuple(
            sorted(
                range(len(pool)),
                key=lambda index: (
                    fronts[index],
                    rank_sums[index],
                    objectives[index][0],
                    index,
                ),
            )
        )
        member_scores = tuple(
            {
                "mace_lcb": objectives[index][0],
                "mace_neg_log_ei": objectives[index][1],
                "mace_neg_log_pi": objectives[index][2],
                "pareto_front": fronts[index],
                "rank_sum": rank_sums[index],
                "implementation": "numpy2_compatible_hebo_mace_port",
                "gp_kernel": "matern_3_2",
                "gp_epochs": 100,
            }
            for index in range(len(pool))
        )
        return PoolRanking(order=order, member_scores=member_scores)


class LLMPoolHEBORankArm(LLMPoolRankArm):
    def __init__(
        self,
        provider: FreshProposalProvider,
        *,
        ranker: PoolRanker | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="llm_pool_hebo_rank",
            provider=provider,
            ranker=ranker or HEBOMACERanker(),
            **kwargs,
        )


def create_arm(
    *,
    provider: FreshProposalProvider,
    ranker: PoolRanker | None = None,
    **_: Any,
) -> LLMPoolHEBORankArm:
    return LLMPoolHEBORankArm(provider, ranker=ranker)
