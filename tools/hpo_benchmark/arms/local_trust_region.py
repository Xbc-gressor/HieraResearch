"""Seeded local random search with an adaptive normalized trust region."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from ..core import (
    BenchmarkContext,
    EvaluationOutcome,
    Observation,
    PolicyContractError,
    Proposal,
    ProposalBatch,
    SearchDimension,
    select_incumbent,
)


def _full_space_incumbent(
    context: BenchmarkContext, observations: Sequence[Observation]
) -> Observation:
    names = set(context.space.names)
    full_space = [
        observation
        for observation in observations
        if set(observation.params) == names
    ]
    try:
        return select_incumbent(full_space)
    except ValueError as exc:
        raise PolicyContractError(
            "local trust region requires a finite full-space incumbent"
        ) from exc


def _from_normalized(dimension: SearchDimension, value: float) -> Any:
    assert dimension.low is not None and dimension.high is not None
    value = min(1.0, max(0.0, value))
    if dimension.log:
        low = math.log(float(dimension.low))
        high = math.log(float(dimension.high))
        raw = math.exp(low + value * (high - low))
    else:
        raw = float(dimension.low) + value * (
            float(dimension.high) - float(dimension.low)
        )
    return dimension.project(raw)


class LocalTrustRegionArm:
    name = "local_trust_region"

    def __init__(
        self,
        *,
        initial_radius: float = 0.25,
        expansion_factor: float = 1.5,
        contraction_factor: float = 0.5,
        min_radius: float = 0.025,
        max_radius: float = 1.0,
    ):
        if not 0 < min_radius <= initial_radius <= max_radius <= 1:
            raise ValueError(
                "trust-region radii must satisfy 0 < min <= initial <= max <= 1"
            )
        if expansion_factor <= 1:
            raise ValueError("expansion_factor must be greater than 1")
        if not 0 < contraction_factor < 1:
            raise ValueError("contraction_factor must be between 0 and 1")
        self.initial_radius = initial_radius
        self.expansion_factor = expansion_factor
        self.contraction_factor = contraction_factor
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.context: BenchmarkContext | None = None
        self.observations: list[Observation] = []
        self.radius = initial_radius
        self.proposal_count = 0
        self.improvement_count = 0
        self._rng = random.Random()
        self._seen: set[str] = set()

    def initialize(self, context: BenchmarkContext) -> None:
        _full_space_incumbent(context, context.observations)
        if not context.space.active_dimensions:
            raise PolicyContractError(
                "local trust region requires at least one non-fixed dimension"
            )
        self.context = context
        self.observations = list(context.observations)
        self.radius = self.initial_radius
        self.proposal_count = 0
        self.improvement_count = 0
        self._rng = random.Random(context.seed)
        names = set(context.space.names)
        self._seen = {
            context.space.canonical(observation.params)
            for observation in self.observations
            if set(observation.params) == names
        }

    def ask(self, remaining_budget: int) -> ProposalBatch:
        if self.context is None:
            raise PolicyContractError("local trust region arm was not initialized")
        if remaining_budget < 1:
            raise PolicyContractError("local trust region requires remaining budget")

        incumbent = _full_space_incumbent(self.context, self.observations)
        params: dict[str, Any] | None = None
        for _ in range(128):
            candidate = self._sample(dict(incumbent.params))
            key = self.context.space.canonical(candidate)
            if key not in self._seen:
                params = candidate
                self._seen.add(key)
                break
        if params is None:
            raise PolicyContractError(
                "local trust region could not sample an unseen configuration"
            )

        self.proposal_count += 1
        metadata = {
            "proposal_index": self.proposal_count,
            "center_observation_id": incumbent.observation_id,
            "center_params": dict(incumbent.params),
            "radius": self.radius,
            "categorical_replace_probability": self.radius,
            "sampling": "independent_uniform_box",
        }
        return ProposalBatch(
            proposals=(Proposal(params=params, origin=self.name, metadata=metadata),),
            metadata={
                "center_observation_id": incumbent.observation_id,
                "radius": self.radius,
            },
        )

    def _sample(self, center: Mapping[str, Any]) -> dict[str, Any]:
        assert self.context is not None
        params: dict[str, Any] = {}
        for dimension in self.context.space.dimensions:
            current = center[dimension.name]
            if dimension.kind == "categorical":
                alternatives = [
                    choice for choice in dimension.choices if choice != current
                ]
                if alternatives and self._rng.random() < self.radius:
                    params[dimension.name] = self._rng.choice(alternatives)
                else:
                    params[dimension.name] = current
                continue
            if dimension.low == dimension.high:
                params[dimension.name] = current
                continue
            normalized = dimension.normalized(current)
            low = max(0.0, normalized - self.radius)
            high = min(1.0, normalized + self.radius)
            params[dimension.name] = _from_normalized(
                dimension, self._rng.uniform(low, high)
            )
        return params

    def tell(self, outcomes: tuple[EvaluationOutcome, ...]) -> None:
        if self.context is None:
            raise PolicyContractError("local trust region arm was not initialized")
        incumbent_before = _full_space_incumbent(self.context, self.observations)
        self.observations.extend(outcome.observation for outcome in outcomes)
        for outcome in outcomes:
            observation = outcome.observation
            if set(observation.params) == set(self.context.space.names):
                self._seen.add(self.context.space.canonical(observation.params))

        admitted = [
            outcome.observation
            for outcome in outcomes
            if outcome.observation.consumes_budget
        ]
        if not admitted:
            return
        incumbent_after = _full_space_incumbent(self.context, self.observations)
        improved = (
            incumbent_after.observation_id != incumbent_before.observation_id
            and float(incumbent_after.score) < float(incumbent_before.score)
        )
        if improved:
            self.improvement_count += 1
            self.radius = min(
                self.max_radius, self.radius * self.expansion_factor
            )
        else:
            self.radius = max(
                self.min_radius, self.radius * self.contraction_factor
            )

    def snapshot(self) -> Mapping[str, Any]:
        if self.context is None:
            raise PolicyContractError("local trust region arm was not initialized")
        incumbent = _full_space_incumbent(self.context, self.observations)
        return {
            "radius": self.radius,
            "proposal_count": self.proposal_count,
            "improvement_count": self.improvement_count,
            "incumbent_id": incumbent.observation_id,
            "incumbent_score": incumbent.score,
        }


def create_arm(**dependencies: Any) -> LocalTrustRegionArm:
    return LocalTrustRegionArm(
        **{
            key: dependencies[key]
            for key in (
                "initial_radius",
                "expansion_factor",
                "contraction_factor",
                "min_radius",
                "max_radius",
            )
            if key in dependencies
        }
    )
