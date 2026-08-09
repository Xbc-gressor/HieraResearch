"""LLM proposal pool ranked by Matern-5/2 Gaussian-process EI."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

from ..core import Observation, SearchSpace
from ..providers import FreshProposalProvider
from .pool_rank import LLMPoolRankArm, PoolRanker, PoolRanking


class NormalizedOneHotEncoder:
    """Encode numeric dimensions to [0, 1] and categoricals one-hot."""

    def __init__(self, space: SearchSpace):
        self.space = space
        self.feature_names = tuple(
            feature
            for dimension in space.dimensions
            for feature in (
                tuple(
                    f"{dimension.name}={choice!r}"
                    for choice in dimension.choices
                )
                if dimension.kind == "categorical"
                else (dimension.name,)
            )
        )

    def encode(self, params: Mapping[str, Any]) -> list[float]:
        projected = self.space.project(params)
        row: list[float] = []
        for dimension in self.space.dimensions:
            value = projected[dimension.name]
            if dimension.kind == "categorical":
                row.extend(
                    1.0 if value == choice else 0.0
                    for choice in dimension.choices
                )
            else:
                row.append(dimension.normalized(value))
        return row


class GPEIRanker:
    """Exact GP with a fixed Matern-5/2 kernel and expected improvement."""

    name = "gp_ei"

    def __init__(
        self,
        *,
        length_scale: float = 0.2,
        noise: float = 1e-6,
        xi: float = 0.0,
    ):
        if length_scale <= 0 or noise <= 0:
            raise ValueError("GP length_scale and noise must be positive")
        self.length_scale = length_scale
        self.noise = noise
        self.xi = xi

    def rank(
        self,
        space: SearchSpace,
        observations: Sequence[Observation],
        pool: Sequence[Mapping[str, Any]],
        *,
        seed: int,
    ) -> PoolRanking:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - root dev env has numpy
            raise RuntimeError("GP-EI ranking requires numpy") from exc

        encoder = NormalizedOneHotEncoder(space)
        history_x = np.asarray(
            [encoder.encode(item.params) for item in observations], dtype=float
        )
        pool_x = np.asarray([encoder.encode(params) for params in pool], dtype=float)
        scores = np.asarray(
            [float(item.score) for item in observations], dtype=float
        )
        score_mean = float(np.mean(scores))
        score_scale = float(np.std(scores))
        if score_scale <= 1e-12:
            score_scale = 1.0
        normalized_scores = (scores - score_mean) / score_scale

        covariance = self._kernel(np, history_x, history_x)
        covariance.flat[:: covariance.shape[0] + 1] += self.noise
        cholesky = np.linalg.cholesky(covariance)
        alpha = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(cholesky, normalized_scores),
        )
        cross_covariance = self._kernel(np, history_x, pool_x)
        predicted_means = score_mean + score_scale * (cross_covariance.T @ alpha)
        projected = np.linalg.solve(cholesky, cross_covariance)
        predicted_variances = score_scale**2 * np.maximum(
            0.0, 1.0 - np.sum(projected * projected, axis=0)
        )
        predicted_stds = np.sqrt(predicted_variances)
        best = float(np.min(scores))
        expected_improvements = [
            self._expected_improvement(best, float(mean), float(std))
            for mean, std in zip(predicted_means, predicted_stds, strict=True)
        ]
        order = tuple(
            sorted(
                range(len(pool)),
                key=lambda index: (-expected_improvements[index], index),
            )
        )
        ranks = [0] * len(pool)
        for rank, index in enumerate(order, start=1):
            ranks[index] = rank
        member_scores = tuple(
            {
                "surrogate": "gaussian_process",
                "kernel": "matern_5_2",
                "acquisition": "expected_improvement",
                "predicted_mean": float(predicted_means[index]),
                "predicted_std": float(predicted_stds[index]),
                "expected_improvement": expected_improvements[index],
                "acquisition_rank": ranks[index],
                "length_scale": self.length_scale,
                "noise": self.noise,
                "xi": self.xi,
                "numeric_encoding": "normalized",
                "categorical_encoding": "one_hot",
                "training_observations": len(observations),
                "seed": seed,
            }
            for index in range(len(pool))
        )
        return PoolRanking(order=order, member_scores=member_scores)

    def _kernel(self, np: Any, left: Any, right: Any) -> Any:
        differences = (
            left[:, np.newaxis, :] - right[np.newaxis, :, :]
        ) / self.length_scale
        distance = np.sqrt(np.sum(differences * differences, axis=2))
        scaled = math.sqrt(5.0) * distance
        return (1.0 + scaled + scaled * scaled / 3.0) * np.exp(-scaled)

    def _expected_improvement(
        self, best: float, mean: float, standard_deviation: float
    ) -> float:
        improvement = best - mean - self.xi
        if standard_deviation <= 1e-15:
            return max(0.0, improvement)
        z = improvement / standard_deviation
        normal = NormalDist()
        return improvement * normal.cdf(z) + standard_deviation * math.exp(
            -0.5 * z * z
        ) / math.sqrt(2.0 * math.pi)


class LLMPoolGPRankArm(LLMPoolRankArm):
    def __init__(
        self,
        provider: FreshProposalProvider,
        *,
        ranker: PoolRanker | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="llm_pool_gp_rank",
            provider=provider,
            ranker=ranker or GPEIRanker(),
            **kwargs,
        )


def create_arm(
    *,
    provider: FreshProposalProvider,
    ranker: PoolRanker | None = None,
    **_: Any,
) -> LLMPoolGPRankArm:
    return LLMPoolGPRankArm(provider, ranker=ranker)
