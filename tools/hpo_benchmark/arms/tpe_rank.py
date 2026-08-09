"""LLM proposal pool ranked by an explicit TPE density ratio."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..core import Observation, SearchDimension, SearchSpace
from ..providers import FreshProposalProvider
from .pool_rank import LLMPoolRankArm, PoolRanking


class TPEDensityRatioRanker:
    """Independent Parzen estimators for p(x|good) / p(x|bad).

    This scorer is intentionally public and self-contained: it does not depend
    on Optuna private APIs, so every externally proposed pool member can be
    scored and recorded.
    """

    name = "tpe_density_ratio"

    def __init__(
        self,
        *,
        good_fraction: float = 0.1,
        max_good: int = 25,
        min_bandwidth: float = 0.05,
        prior_weight: float = 0.1,
    ):
        if not 0.0 < good_fraction < 1.0:
            raise ValueError("good_fraction must be between zero and one")
        if max_good < 1 or min_bandwidth <= 0 or prior_weight <= 0:
            raise ValueError("invalid TPE ranker hyperparameters")
        self.good_fraction = good_fraction
        self.max_good = max_good
        self.min_bandwidth = min_bandwidth
        self.prior_weight = prior_weight

    def _numeric_log_density(
        self,
        dimension: SearchDimension,
        value: Any,
        observations: Sequence[Observation],
    ) -> float:
        samples = [dimension.normalized(item.params[dimension.name]) for item in observations]
        point = dimension.normalized(value)
        if len(samples) == 1:
            bandwidth = 0.2
        else:
            mean = sum(samples) / len(samples)
            variance = sum((sample - mean) ** 2 for sample in samples) / (
                len(samples) - 1
            )
            bandwidth = 1.06 * math.sqrt(variance) * len(samples) ** -0.2
        bandwidth = min(0.5, max(self.min_bandwidth, bandwidth))
        normalizer = bandwidth * math.sqrt(2.0 * math.pi)
        kernel_sum = sum(
            math.exp(-0.5 * ((point - sample) / bandwidth) ** 2) / normalizer
            for sample in samples
        )
        # Mix the same uniform-prior weight into both groups.  A pseudo-count
        # normalized by group size would give the smaller ``good`` group a
        # stronger prior and can make unsupported boundary points win solely
        # because good and bad contain different numbers of observations.
        density = (kernel_sum / len(samples) + self.prior_weight) / (
            1.0 + self.prior_weight
        )
        return math.log(max(density, 1e-300))

    def _categorical_log_density(
        self,
        dimension: SearchDimension,
        value: Any,
        observations: Sequence[Observation],
    ) -> float:
        matches = sum(item.params[dimension.name] == value for item in observations)
        denominator = len(observations) + self.prior_weight * len(dimension.choices)
        density = (matches + self.prior_weight) / denominator
        return math.log(max(density, 1e-300))

    def _log_density(
        self,
        space: SearchSpace,
        params: Mapping[str, Any],
        observations: Sequence[Observation],
    ) -> float:
        total = 0.0
        for dimension in space.active_dimensions:
            if dimension.kind == "categorical":
                total += self._categorical_log_density(
                    dimension, params[dimension.name], observations
                )
            else:
                total += self._numeric_log_density(
                    dimension, params[dimension.name], observations
                )
        return total

    def rank(
        self,
        space: SearchSpace,
        observations: Sequence[Observation],
        pool: Sequence[Mapping[str, Any]],
        *,
        seed: int,
    ) -> PoolRanking:
        del seed
        ordered_history = sorted(observations, key=lambda item: float(item.score))
        good_count = min(
            self.max_good,
            max(1, math.ceil(self.good_fraction * len(ordered_history))),
        )
        good = ordered_history[:good_count]
        bad = ordered_history[good_count:]
        if not bad:
            raise ValueError("TPE ranking requires at least one bad observation")

        member_scores = []
        ratios = []
        for params in pool:
            log_good = self._log_density(space, params, good)
            log_bad = self._log_density(space, params, bad)
            ratio = log_good - log_bad
            ratios.append(ratio)
            member_scores.append(
                {
                    "log_density_good": log_good,
                    "log_density_bad": log_bad,
                    "log_density_ratio": ratio,
                    "good_observations": len(good),
                    "bad_observations": len(bad),
                }
            )
        order = tuple(sorted(range(len(pool)), key=lambda index: (-ratios[index], index)))
        return PoolRanking(order=order, member_scores=tuple(member_scores))


class LLMPoolTPERankArm(LLMPoolRankArm):
    def __init__(
        self,
        provider: FreshProposalProvider,
        *,
        ranker: TPEDensityRatioRanker | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="llm_pool_tpe_rank",
            provider=provider,
            ranker=ranker or TPEDensityRatioRanker(),
            **kwargs,
        )


def create_arm(
    *,
    provider: FreshProposalProvider,
    ranker: TPEDensityRatioRanker | None = None,
    **_: Any,
) -> LLMPoolTPERankArm:
    return LLMPoolTPERankArm(provider, ranker=ranker)
