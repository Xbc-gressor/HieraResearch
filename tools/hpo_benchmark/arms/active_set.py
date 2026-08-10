"""Fresh-session LLM active-set arm with controller-built symmetric pairs."""

from __future__ import annotations

import hashlib
import json
import math
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
from ..providers import FreshProposalProvider
from ..summary import FocusedSummaryBuilder
from ._repair import complete_with_repair


ALLOWED_STEPS = (0.05, 0.10, 0.20, 0.40)
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["active_dimensions", "reason"],
    "properties": {
        "active_dimensions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {
                "type": "object",
                "required": ["name", "direction", "step"],
                "properties": {
                    "name": {"type": "string"},
                    "direction": {"type": "integer", "enum": [-1, 1]},
                    "step": {"type": "number", "enum": list(ALLOWED_STEPS)},
                },
                "additionalProperties": False,
            },
        },
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}


def _full_space_incumbent(
    context: BenchmarkContext, observations: Sequence[Observation]
) -> Observation:
    names = set(context.space.names)
    compatible = [
        observation
        for observation in observations
        if set(observation.params) == names
    ]
    try:
        return select_incumbent(compatible)
    except ValueError as exc:
        raise PolicyContractError(
            "LLM active-set requires a finite full-space incumbent"
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


@dataclass(frozen=True)
class _Move:
    dimension: SearchDimension
    direction: int
    step: float

    def record(self) -> dict[str, Any]:
        return {
            "name": self.dimension.name,
            "direction": self.direction,
            "step": self.step,
        }


class LLMActiveSetArm:
    name = "llm_active_set"

    def __init__(
        self,
        provider: FreshProposalProvider,
        *,
        summary_builder: FocusedSummaryBuilder | None = None,
        corrective_attempts: int = 3,
    ):
        if corrective_attempts < 0:
            raise ValueError("corrective_attempts must be non-negative")
        self.provider = provider
        self.summary_builder = summary_builder or FocusedSummaryBuilder()
        self.corrective_attempts = corrective_attempts
        self.context: BenchmarkContext | None = None
        self.observations: list[Observation] = []
        self.provider_calls = 0
        self.provider_attempts = 0
        self.corrective_calls = 0
        self.completed_rounds = 0
        self.rejected_batches = 0
        self.failed_rounds = 0
        self.degraded_calls = 0
        self._pending_group_id: str | None = None

    def initialize(self, context: BenchmarkContext) -> None:
        if context.budget % 2:
            raise PolicyContractError(
                "LLM active-set requires an even evaluation budget"
            )
        _full_space_incumbent(context, context.observations)
        numeric = [
            dimension
            for dimension in context.space.active_dimensions
            if dimension.kind in {"float", "int"}
        ]
        if not numeric:
            raise PolicyContractError(
                "LLM active-set requires a non-fixed numeric dimension"
            )
        self.context = context
        self.observations = list(context.observations)
        self.provider_calls = 0
        self.provider_attempts = 0
        self.corrective_calls = 0
        self.completed_rounds = 0
        self.rejected_batches = 0
        self.failed_rounds = 0
        self.degraded_calls = 0
        self._pending_group_id = None

    def ask(self, remaining_budget: int) -> ProposalBatch:
        if self.context is None:
            raise PolicyContractError("LLM active-set arm was not initialized")
        if self._pending_group_id is not None:
            raise PolicyContractError(
                "LLM active-set received ask before the previous pair was told"
            )
        if remaining_budget < 2:
            raise PolicyContractError(
                "LLM active-set cannot split a symmetric pair"
            )

        summary = self.summary_builder.build(
            self.context, self.observations, remaining_budget=remaining_budget
        )
        recent_rejections = [
            {
                "params": dict(observation.params),
                "failure": observation.failure,
            }
            for observation in reversed(self.observations)
            if observation.status == "rejected"
        ][:4]
        if recent_rejections:
            summary["recent_rejections"] = recent_rejections
        prompt = (
            "Choose one or two non-fixed numeric hyperparameter dimensions for "
            "a symmetric local experiment. For each dimension return a signed "
            "direction and one normalized step from 0.05, 0.10, 0.20, or 0.40. "
            "Do not select categorical or fixed dimensions. The controller will "
            "evaluate both +direction and -direction from the same incumbent. "
            "Scores are lower-is-better. Return JSON matching the supplied schema."
            "\n\n"
            + json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        )
        response, problems, final_prompt = complete_with_repair(
            self.provider,
            prompt,
            OUTPUT_SCHEMA,
            validate=self._problems,
            corrective_attempts=self.corrective_attempts,
        )
        self.provider_attempts += int(
            response.metadata["repair_provider_attempts"]
        )
        self.corrective_calls += int(
            response.metadata["repair_corrective_calls"]
        )
        center = dict(summary["incumbent"]["params"])
        if problems:
            self.degraded_calls += 1
            moves = self._default_move(center)
            reason = "degraded default move: " + "; ".join(problems)
        else:
            moves, reason = self._parse_output(response.output)
        (
            plus,
            minus,
            plus_targets,
            minus_targets,
            effective_steps,
        ) = self._symmetric_pair(center, moves)

        self.provider_calls += 1
        group_id = f"active-set-{self.provider_calls:04d}"
        self._pending_group_id = group_id
        prompt_hash = hashlib.sha256(final_prompt.encode("utf-8")).hexdigest()
        move_records = [move.record() for move in moves]
        shared = {
            "group_id": group_id,
            "round": self.completed_rounds + 1,
            "center_observation_id": summary["incumbent"]["observation_id"],
            "center_params": center,
            "active_dimensions": move_records,
            "effective_steps": effective_steps,
            "reason": reason,
            "repair_problems": list(problems),
            "degraded": bool(problems),
            "prompt": final_prompt,
            "prompt_hash": prompt_hash,
            "raw_output": response.raw_output,
            "model": response.model,
            "provider_metadata": dict(response.metadata),
        }
        return ProposalBatch(
            proposals=(
                Proposal(
                    params=plus,
                    origin=self.name,
                    metadata={
                        **shared,
                        "side": "plus",
                        "group_role": "plus",
                        "normalized_targets": plus_targets,
                    },
                ),
                Proposal(
                    params=minus,
                    origin=self.name,
                    metadata={
                        **shared,
                        "side": "minus",
                        "group_role": "minus",
                        "normalized_targets": minus_targets,
                    },
                ),
            ),
            atomic=True,
            metadata={
                "group_id": group_id,
                "fresh_provider_call": self.provider_calls,
                "center_observation_id": summary["incumbent"]["observation_id"],
                "active_dimensions": move_records,
            },
        )

    def _problems(self, output: Mapping[str, Any]) -> list[str]:
        try:
            moves, _ = self._parse_output(output)
            assert self.context is not None
            center = dict(
                _full_space_incumbent(
                    self.context, self.observations
                ).params
            )
            plus, minus, _, _, _ = self._symmetric_pair(center, moves)
            plus_key = self.context.space.canonical(plus)
            minus_key = self.context.space.canonical(minus)
            if plus_key == minus_key:
                return [
                    "the symmetric pair collapses after projection; choose a "
                    "dimension away from its boundary"
                ]
            history = {
                self.context.space.canonical(observation.params)
                for observation in self.observations
                if set(observation.params) == set(self.context.space.names)
            }
            if plus_key in history or minus_key in history:
                return [
                    "the symmetric pair revisits a configuration in history; "
                    "choose different active dimensions or steps"
                ]
        except (PolicyContractError, TypeError, ValueError) as exc:
            return [str(exc)]
        return []

    def _default_move(self, center: Mapping[str, Any]) -> tuple[_Move, ...]:
        assert self.context is not None
        for dimension in self.context.space.active_dimensions:
            if dimension.kind not in {"float", "int"}:
                continue
            normalized = dimension.normalized(center[dimension.name])
            direction = 1 if normalized <= 0.5 else -1
            moves = (_Move(dimension, direction, 0.10),)
            plus, minus, _, _, _ = self._symmetric_pair(center, moves)
            plus_key = self.context.space.canonical(plus)
            minus_key = self.context.space.canonical(minus)
            history = {
                self.context.space.canonical(observation.params)
                for observation in self.observations
                if set(observation.params) == set(self.context.space.names)
            }
            if (
                plus_key != minus_key
                and plus_key not in history
                and minus_key not in history
            ):
                return moves
        raise PolicyContractError(
            "LLM active-set cannot build a distinct unseen symmetric pair"
        )

    def _parse_output(
        self, output: Mapping[str, Any]
    ) -> tuple[tuple[_Move, ...], str]:
        assert self.context is not None
        raw_moves = output.get("active_dimensions")
        reason = output.get("reason")
        if (
            not isinstance(raw_moves, list)
            or not 1 <= len(raw_moves) <= 2
        ):
            raise PolicyContractError(
                "LLM active-set must return one or two active dimensions"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise PolicyContractError(
                "LLM active-set must return a non-empty reason"
            )
        dimensions = {
            dimension.name: dimension
            for dimension in self.context.space.active_dimensions
            if dimension.kind in {"float", "int"}
        }
        moves: list[_Move] = []
        names: set[str] = set()
        for raw in raw_moves:
            if not isinstance(raw, Mapping):
                raise PolicyContractError(
                    "LLM active-set dimension entries must be objects"
                )
            name = raw.get("name")
            direction = raw.get("direction")
            step = raw.get("step")
            if not isinstance(name, str) or name not in dimensions:
                raise PolicyContractError(
                    f"LLM active-set selected non-numeric or fixed dimension {name!r}"
                )
            if name in names:
                raise PolicyContractError(
                    f"LLM active-set selected dimension {name!r} more than once"
                )
            if isinstance(direction, bool) or direction not in {-1, 1}:
                raise PolicyContractError(
                    "LLM active-set direction must be -1 or 1"
                )
            if isinstance(step, bool) or not isinstance(step, (int, float)):
                raise PolicyContractError(
                    "LLM active-set step must be numeric"
                )
            allowed_step = next(
                (
                    candidate
                    for candidate in ALLOWED_STEPS
                    if math.isclose(float(step), candidate, abs_tol=1e-12)
                ),
                None,
            )
            if allowed_step is None:
                raise PolicyContractError(
                    f"LLM active-set step must be one of {ALLOWED_STEPS}"
                )
            names.add(name)
            moves.append(
                _Move(
                    dimension=dimensions[name],
                    direction=int(direction),
                    step=allowed_step,
                )
            )
        return tuple(moves), reason.strip()

    def _symmetric_pair(
        self, center: Mapping[str, Any], moves: Sequence[_Move]
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ]:
        assert self.context is not None
        plus, minus = dict(center), dict(center)
        plus_targets: dict[str, float] = {}
        minus_targets: dict[str, float] = {}
        effective_steps: dict[str, float] = {}
        for move in moves:
            name = move.dimension.name
            normalized = move.dimension.normalized(center[name])
            effective_step = min(move.step, normalized, 1.0 - normalized)
            plus_target = normalized + move.direction * effective_step
            minus_target = normalized - move.direction * effective_step
            plus[name] = _from_normalized(move.dimension, plus_target)
            minus[name] = _from_normalized(move.dimension, minus_target)
            plus_targets[name] = plus_target
            minus_targets[name] = minus_target
            effective_steps[name] = effective_step
        return (
            self.context.space.project(plus),
            self.context.space.project(minus),
            plus_targets,
            minus_targets,
            effective_steps,
        )

    def tell(self, outcomes: tuple[EvaluationOutcome, ...]) -> None:
        if self.context is None or self._pending_group_id is None:
            raise PolicyContractError(
                "LLM active-set received outcomes without a pending pair"
            )
        if len(outcomes) != 2:
            raise PolicyContractError(
                "LLM active-set expects exactly two outcomes per pair"
            )
        self._pending_group_id = None
        self.observations.extend(outcome.observation for outcome in outcomes)
        admitted = [outcome.observation.consumes_budget for outcome in outcomes]
        if not any(admitted):
            self.rejected_batches += 1
            return
        if not all(admitted):
            raise PolicyContractError(
                "LLM active-set symmetric pair was only partially admitted"
            )
        self.completed_rounds += 1
        if any(outcome.observation.status != "ok" for outcome in outcomes):
            self.failed_rounds += 1

    def snapshot(self) -> Mapping[str, Any]:
        if self.context is None:
            raise PolicyContractError("LLM active-set arm was not initialized")
        incumbent = _full_space_incumbent(self.context, self.observations)
        return {
            "provider_calls": self.provider_calls,
            "provider_attempts": self.provider_attempts,
            "corrective_calls": self.corrective_calls,
            "completed_rounds": self.completed_rounds,
            "rejected_batches": self.rejected_batches,
            "failed_rounds": self.failed_rounds,
            "degraded_calls": self.degraded_calls,
            "incumbent_id": incumbent.observation_id,
            "incumbent_score": incumbent.score,
        }


def create_arm(
    *, provider: FreshProposalProvider, **_: Any
) -> LLMActiveSetArm:
    return LLMActiveSetArm(provider)
