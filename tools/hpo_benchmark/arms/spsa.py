"""Simultaneous perturbation stochastic approximation benchmark arm."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
        raise PolicyContractError("SPSA requires a finite full-space incumbent") from exc


def _from_normalized(dimension: SearchDimension, value: float) -> float:
    assert dimension.low is not None and dimension.high is not None
    value = min(1.0, max(0.0, value))
    if dimension.log:
        low = math.log(float(dimension.low))
        high = math.log(float(dimension.high))
        return math.exp(low + value * (high - low))
    return float(dimension.low) + value * (
        float(dimension.high) - float(dimension.low)
    )


@dataclass(frozen=True)
class _PendingPair:
    center_params: Mapping[str, Any]
    center_normalized: Mapping[str, float]
    plus_normalized: Mapping[str, float]
    minus_normalized: Mapping[str, float]
    perturbation: Mapping[str, int]
    step_size: float
    perturbation_size: float
    round_number: int


class SPSAArm:
    name = "spsa"

    def __init__(
        self,
        *,
        a0: float = 0.05,
        c0: float = 0.10,
        alpha: float = 0.602,
        gamma: float = 0.101,
    ):
        if min(a0, c0, alpha, gamma) <= 0:
            raise ValueError("SPSA schedule parameters must be positive")
        self.a0 = a0
        self.c0 = c0
        self.alpha = alpha
        self.gamma = gamma
        self.context: BenchmarkContext | None = None
        self.observations: list[Observation] = []
        self.center_params: dict[str, Any] = {}
        self.completed_rounds = 0
        self.failed_rounds = 0
        self.rejected_batches = 0
        self._pair_retries = 0
        self._rng = random.Random()
        self._active: tuple[SearchDimension, ...] = ()
        self._seen: set[str] = set()
        self._pending: _PendingPair | None = None

    def initialize(self, context: BenchmarkContext) -> None:
        if context.budget % 2:
            raise PolicyContractError("SPSA requires an even evaluation budget")
        self._active = tuple(
            dimension
            for dimension in context.space.dimensions
            if dimension.kind == "float" and dimension.low != dimension.high
        )
        if not self._active:
            raise PolicyContractError(
                "SPSA requires at least one non-fixed continuous dimension"
            )
        incumbent = _full_space_incumbent(context, context.observations)
        self.context = context
        self.observations = list(context.observations)
        self.center_params = context.space.project(incumbent.params)
        self.completed_rounds = 0
        self.failed_rounds = 0
        self.rejected_batches = 0
        self._pair_retries = 0
        self._rng = random.Random(context.seed)
        names = set(context.space.names)
        self._seen = {
            context.space.canonical(observation.params)
            for observation in self.observations
            if set(observation.params) == names
        }
        self._pending = None

    def ask(self, remaining_budget: int) -> ProposalBatch:
        if self.context is None:
            raise PolicyContractError("SPSA arm was not initialized")
        if self._pending is not None:
            raise PolicyContractError("SPSA received ask before the previous pair was told")
        if remaining_budget < 2:
            raise PolicyContractError(
                "SPSA cannot split a perturbation pair across the remaining budget"
            )

        round_number = self.completed_rounds + 1
        step_size = self.a0 / round_number**self.alpha
        perturbation_size = (
            self.c0 / round_number**self.gamma * 0.5**self._pair_retries
        )
        center_normalized = {
            dimension.name: dimension.normalized(self.center_params[dimension.name])
            for dimension in self._active
        }

        pair: tuple[dict[str, Any], dict[str, Any]] | None = None
        perturbation: dict[str, int] = {}
        plus_normalized: dict[str, float] = {}
        minus_normalized: dict[str, float] = {}
        for _ in range(128):
            perturbation = {
                dimension.name: self._rng.choice((-1, 1))
                for dimension in self._active
            }
            plus_normalized = {
                name: min(
                    1.0,
                    max(0.0, center_normalized[name] + perturbation_size * sign),
                )
                for name, sign in perturbation.items()
            }
            minus_normalized = {
                name: min(
                    1.0,
                    max(0.0, center_normalized[name] - perturbation_size * sign),
                )
                for name, sign in perturbation.items()
            }
            plus = self._configuration(plus_normalized)
            minus = self._configuration(minus_normalized)
            plus_key = self.context.space.canonical(plus)
            minus_key = self.context.space.canonical(minus)
            if (
                plus_key != minus_key
                and plus_key not in self._seen
                and minus_key not in self._seen
            ):
                pair = plus, minus
                self._seen.update((plus_key, minus_key))
                break
        if pair is None:
            raise PolicyContractError("SPSA could not construct an unseen perturbation pair")

        self._pending = _PendingPair(
            center_params=dict(self.center_params),
            center_normalized=center_normalized,
            plus_normalized=plus_normalized,
            minus_normalized=minus_normalized,
            perturbation=perturbation,
            step_size=step_size,
            perturbation_size=perturbation_size,
            round_number=round_number,
        )
        shared_metadata = {
            "round": round_number,
            "center_params": dict(self.center_params),
            "perturbation": dict(perturbation),
            "a_k": step_size,
            "c_k": perturbation_size,
            "retry": self._pair_retries,
        }
        return ProposalBatch(
            proposals=(
                Proposal(
                    params=pair[0],
                    origin=self.name,
                    metadata={**shared_metadata, "side": "plus"},
                ),
                Proposal(
                    params=pair[1],
                    origin=self.name,
                    metadata={**shared_metadata, "side": "minus"},
                ),
            ),
            atomic=True,
            metadata=shared_metadata,
        )

    def _configuration(self, normalized: Mapping[str, float]) -> dict[str, Any]:
        assert self.context is not None
        params = dict(self.center_params)
        for dimension in self._active:
            params[dimension.name] = _from_normalized(
                dimension, normalized[dimension.name]
            )
        return params

    def tell(self, outcomes: tuple[EvaluationOutcome, ...]) -> None:
        if self.context is None or self._pending is None:
            raise PolicyContractError("SPSA received outcomes without a pending pair")
        if len(outcomes) != 2:
            raise PolicyContractError("SPSA expects exactly two outcomes per pair")
        pending = self._pending
        self._pending = None
        self.observations.extend(outcome.observation for outcome in outcomes)
        names = set(self.context.space.names)
        for outcome in outcomes:
            observation = outcome.observation
            if set(observation.params) == names:
                self._seen.add(self.context.space.canonical(observation.params))

        admitted = [outcome.observation.consumes_budget for outcome in outcomes]
        if not any(admitted):
            self.rejected_batches += 1
            self._pair_retries += 1
            return
        if not all(admitted):
            raise PolicyContractError("SPSA perturbation pair was only partially admitted")

        self.completed_rounds += 1
        self._pair_retries = 0
        plus_score = outcomes[0].observation.score
        minus_score = outcomes[1].observation.score
        if (
            outcomes[0].observation.status != "ok"
            or outcomes[1].observation.status != "ok"
            or plus_score is None
            or minus_score is None
            or not math.isfinite(plus_score)
            or not math.isfinite(minus_score)
        ):
            self.failed_rounds += 1
            return

        score_difference = float(plus_score) - float(minus_score)
        updated = dict(pending.center_params)
        for dimension in self._active:
            name = dimension.name
            separation = (
                pending.plus_normalized[name] - pending.minus_normalized[name]
            )
            if separation == 0:
                raise PolicyContractError(
                    f"SPSA perturbation collapsed for dimension {name!r}"
                )
            gradient = score_difference / separation
            next_value = (
                pending.center_normalized[name] - pending.step_size * gradient
            )
            updated[name] = _from_normalized(dimension, next_value)
        self.center_params = self.context.space.project(updated)

    def snapshot(self) -> Mapping[str, Any]:
        if self.context is None:
            raise PolicyContractError("SPSA arm was not initialized")
        incumbent = _full_space_incumbent(self.context, self.observations)
        return {
            "completed_rounds": self.completed_rounds,
            "failed_rounds": self.failed_rounds,
            "rejected_batches": self.rejected_batches,
            "pending_pair_retries": self._pair_retries,
            "center_params": dict(self.center_params),
            "incumbent_id": incumbent.observation_id,
            "incumbent_score": incumbent.score,
        }


def create_arm(**dependencies: Any) -> SPSAArm:
    return SPSAArm(
        **{
            key: dependencies[key]
            for key in ("a0", "c0", "alpha", "gamma")
            if key in dependencies
        }
    )
