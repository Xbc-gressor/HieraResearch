"""LLM proposal pool ranked by SMAC random-forest expected improvement."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..core import Observation, SearchSpace
from ..providers import FreshProposalProvider
from .pool_rank import LLMPoolRankArm, PoolRanker, PoolRanking


class SMACRFEIRanker:
    """Fit SMAC's public RF model and score only the supplied LLM pool."""

    name = "smac_rf_ei"

    def __init__(
        self,
        *,
        n_trees: int = 10,
        ratio_features: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        xi: float = 0.0,
    ):
        self.n_trees = n_trees
        self.ratio_features = ratio_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
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
            from ConfigSpace import (
                Categorical,
                Configuration,
                ConfigurationSpace,
                Constant,
                Float,
                Integer,
            )
            from smac.acquisition.function.expected_improvement import EI
            from smac.model.random_forest.random_forest import RandomForest
        except ImportError as exc:  # pragma: no cover - optional task dependency
            raise RuntimeError(
                "SMAC RF-EI ranking requires compatible smac, ConfigSpace, "
                "scikit-learn, scipy, and numpy packages"
            ) from exc

        configspace = ConfigurationSpace(seed=seed)
        for dimension in space.dimensions:
            if dimension.kind == "categorical":
                if len(dimension.choices) == 1:
                    hyperparameter = Constant(
                        dimension.name, dimension.choices[0]
                    )
                else:
                    hyperparameter = Categorical(
                        dimension.name, list(dimension.choices)
                    )
            elif dimension.low == dimension.high:
                hyperparameter = Constant(dimension.name, dimension.low)
            elif dimension.kind == "float":
                hyperparameter = Float(
                    dimension.name,
                    (float(dimension.low), float(dimension.high)),
                    log=dimension.log,
                )
            else:
                hyperparameter = Integer(
                    dimension.name,
                    (int(dimension.low), int(dimension.high)),
                    log=dimension.log,
                )
            configspace.add(hyperparameter)

        history_configs = [
            Configuration(configspace, values=space.project(item.params))
            for item in observations
        ]
        pool_configs = [
            Configuration(configspace, values=space.project(params))
            for params in pool
        ]
        history_x = np.asarray(
            [configuration.get_array() for configuration in history_configs]
        )
        scores = np.asarray(
            [float(item.score) for item in observations], dtype=float
        ).reshape(-1, 1)

        model = RandomForest(
            configspace=configspace,
            n_trees=self.n_trees,
            ratio_features=self.ratio_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrapping=True,
            log_y=False,
            seed=seed,
        )
        model.train(history_x, scores)
        acquisition = EI(xi=self.xi, log=False)
        acquisition.update(model, eta=float(np.min(scores)))
        expected_improvements = np.asarray(
            acquisition(pool_configs), dtype=float
        ).reshape(-1)

        pool_x = np.asarray(
            [configuration.get_array() for configuration in pool_configs]
        )
        means, variances = model.predict_marginalized(pool_x)
        means = np.asarray(means, dtype=float).reshape(-1)
        variances = np.asarray(variances, dtype=float).reshape(-1)
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
                "surrogate": "smac_random_forest",
                "acquisition": "expected_improvement",
                "predicted_mean": float(means[index]),
                "predicted_std": math.sqrt(max(0.0, float(variances[index]))),
                "expected_improvement": float(expected_improvements[index]),
                "acquisition_rank": ranks[index],
                "n_trees": self.n_trees,
                "ratio_features": self.ratio_features,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "xi": self.xi,
                "training_observations": len(observations),
            }
            for index in range(len(pool))
        )
        return PoolRanking(order=order, member_scores=member_scores)


class LLMPoolSMACRankArm(LLMPoolRankArm):
    def __init__(
        self,
        provider: FreshProposalProvider,
        *,
        ranker: PoolRanker | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="llm_pool_smac_rank",
            provider=provider,
            ranker=ranker or SMACRFEIRanker(),
            **kwargs,
        )


def create_arm(
    *,
    provider: FreshProposalProvider,
    ranker: PoolRanker | None = None,
    **_: Any,
) -> LLMPoolSMACRankArm:
    return LLMPoolSMACRankArm(provider, ranker=ranker)
