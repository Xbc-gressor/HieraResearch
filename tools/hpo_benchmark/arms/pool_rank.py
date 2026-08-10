"""Shared fresh-session LLM proposal-pool policy.

Ranker arms deliberately share this module so their LLM prompt, focused
summary, pool validation, and retry behavior are identical.  Only the numeric
ranking object differs between benchmark cells.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..core import (
    BenchmarkContext,
    EvaluationOutcome,
    Observation,
    PolicyContractError,
    Proposal,
    ProposalBatch,
    SearchSpace,
    select_incumbent,
)
from ..providers import FreshProposalProvider
from ..summary import FocusedSummaryBuilder
from ._repair import complete_with_repair, random_config


POOL_SIZE = 8
MIN_RANKER_HISTORY = 8

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["proposals"],
    "properties": {
        "proposals": {
            "type": "array",
            "minItems": POOL_SIZE,
            "maxItems": POOL_SIZE,
            "items": {
                "type": "object",
                "required": ["params", "reason"],
                "properties": {
                    "params": {"type": "object"},
                    "reason": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}


@dataclass(frozen=True)
class PoolRanking:
    """Complete deterministic ordering and per-member acquisition evidence."""

    order: tuple[int, ...]
    member_scores: tuple[Mapping[str, Any], ...]


class PoolRanker(Protocol):
    name: str

    def rank(
        self,
        space: SearchSpace,
        observations: Sequence[Observation],
        pool: Sequence[Mapping[str, Any]],
        *,
        seed: int,
    ) -> PoolRanking: ...


def effective_observations(
    space: SearchSpace, observations: Sequence[Observation]
) -> tuple[Observation, ...]:
    """Return finite full-space observations that can train a numeric ranker."""

    return tuple(
        observation
        for observation in observations
        if observation.status == "ok"
        and observation.score is not None
        and math.isfinite(observation.score)
        and set(observation.params) == set(space.names)
    )


class LLMPoolRankArm:
    def __init__(
        self,
        *,
        name: str,
        provider: FreshProposalProvider,
        ranker: PoolRanker,
        summary_builder: FocusedSummaryBuilder | None = None,
        min_ranker_history: int = MIN_RANKER_HISTORY,
        corrective_attempts: int = 3,
    ):
        if min_ranker_history < 1:
            raise ValueError("min_ranker_history must be positive")
        if corrective_attempts < 0:
            raise ValueError("corrective_attempts must be non-negative")
        self.name = name
        self.provider = provider
        self.ranker = ranker
        self.summary_builder = summary_builder or FocusedSummaryBuilder()
        self.min_ranker_history = min_ranker_history
        self.corrective_attempts = corrective_attempts
        self.context: BenchmarkContext | None = None
        self.observations: list[Observation] = []
        self.provider_calls = 0
        self.provider_attempts = 0
        self.corrective_calls = 0
        self.ranker_calls = 0
        self.degraded_calls = 0
        self._pool: list[dict[str, Any]] | None = None
        self._order: list[int] = []
        self._selection_mode: str | None = None
        self._prompt: str | None = None
        self._raw_output: str | None = None
        self._model: str | None = None
        self._provider_metadata: dict[str, Any] = {}
        self._repair_problems: list[str] = []
        self._last_selected_index: int | None = None
        self._rejected_indices: list[int] = []

    def initialize(self, context: BenchmarkContext) -> None:
        self.context = context
        self.observations = list(context.observations)
        self.provider_calls = 0
        self.provider_attempts = 0
        self.corrective_calls = 0
        self.ranker_calls = 0
        self.degraded_calls = 0
        self._clear_pool()

    def _clear_pool(self) -> None:
        self._pool = None
        self._order = []
        self._selection_mode = None
        self._prompt = None
        self._raw_output = None
        self._model = None
        self._provider_metadata = {}
        self._repair_problems = []
        self._last_selected_index = None
        self._rejected_indices = []

    def _pool_problems(self, output: Mapping[str, Any]) -> list[str]:
        assert self.context is not None
        raw_pool = output.get("proposals")
        if not isinstance(raw_pool, list):
            return [f"proposals must be a list of exactly {POOL_SIZE} objects"]
        if len(raw_pool) != POOL_SIZE:
            return [
                f"proposals must contain exactly {POOL_SIZE} configurations"
            ]
        problems: list[str] = []
        names = set(self.context.space.names)
        history = {
            self.context.space.canonical(observation.params)
            for observation in self.observations
            if set(observation.params) == names
        }
        proposed: set[str] = set()
        for index, item in enumerate(raw_pool):
            if not isinstance(item, Mapping):
                problems.append(f"proposal {index + 1} is not an object")
                continue
            params = item.get("params")
            reason = item.get("reason")
            if not isinstance(params, Mapping):
                problems.append(f"proposal {index + 1} must contain a params object")
            else:
                try:
                    projected = self.context.space.project(params)
                except (TypeError, ValueError) as exc:
                    problems.append(f"proposal {index + 1} params are invalid: {exc}")
                else:
                    key = self.context.space.canonical(projected)
                    if key in history:
                        problems.append(
                            f"proposal {index + 1} revisits a configuration in history"
                        )
                    if key in proposed:
                        problems.append(
                            f"proposal {index + 1} duplicates another proposal"
                        )
                    proposed.add(key)
            if not isinstance(reason, str) or not reason.strip():
                problems.append(
                    f"proposal {index + 1} must contain a non-empty reason"
                )
        return problems

    def _fallback_pool(self) -> list[dict[str, Any]]:
        assert self.context is not None
        rng = random.Random(self.context.seed * 10007 + self.provider_calls * 31)
        space = self.context.space
        names = set(space.names)
        seen = {
            space.canonical(observation.params)
            for observation in self.observations
            if set(observation.params) == names
        }
        pool: list[dict[str, Any]] = []
        draws = 0
        while len(pool) < POOL_SIZE:
            draws += 1
            if draws > POOL_SIZE * 256:
                raise PolicyContractError(
                    f"cannot build a fallback pool of {POOL_SIZE} distinct "
                    "configurations in this search space"
                )
            params = random_config(space, rng)
            key = space.canonical(params)
            if key in seen:
                continue
            seen.add(key)
            pool.append(
                {
                    "llm_rank": len(pool) + 1,
                    "params": params,
                    "reason": "degraded fallback pool",
                    "ranker_scores": None,
                }
            )
        return pool

    def _new_pool(self, remaining_budget: int) -> None:
        assert self.context is not None
        summary = self.summary_builder.build(
            self.context, self.observations, remaining_budget=remaining_budget
        )
        prompt = (
            f"Propose exactly {POOL_SIZE} complete hyperparameter configurations, "
            "ordered from most to least promising according to your judgment. "
            "Scores are lower-is-better. Preserve useful interactions, diversify "
            "the pool, and return JSON matching the supplied schema.\n\n"
            + json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        )
        response, problems, final_prompt = complete_with_repair(
            self.provider,
            prompt,
            OUTPUT_SCHEMA,
            validate=self._pool_problems,
            corrective_attempts=self.corrective_attempts,
        )
        self.provider_calls += 1
        self.provider_attempts += int(
            response.metadata["repair_provider_attempts"]
        )
        self.corrective_calls += int(
            response.metadata["repair_corrective_calls"]
        )
        if problems:
            self.degraded_calls += 1
            pool = self._fallback_pool()
        else:
            pool = []
            for index, item in enumerate(response.output["proposals"]):
                projected = self.context.space.project(item["params"])
                pool.append(
                    {
                        "llm_rank": index + 1,
                        "params": projected,
                        "reason": item["reason"],
                        "ranker_scores": None,
                    }
                )

        history = effective_observations(self.context.space, self.observations)
        if len(history) < self.min_ranker_history:
            order = tuple(range(POOL_SIZE))
            selection_mode = "llm_order"
        else:
            ranking = self.ranker.rank(
                self.context.space,
                history,
                [member["params"] for member in pool],
                seed=self.context.seed + self.provider_calls,
            )
            if (
                len(ranking.member_scores) != POOL_SIZE
                or len(ranking.order) != POOL_SIZE
                or set(ranking.order) != set(range(POOL_SIZE))
            ):
                raise PolicyContractError(
                    f"ranker {self.ranker.name!r} did not rank the complete pool"
                )
            for member, scores in zip(pool, ranking.member_scores):
                member["ranker_scores"] = dict(scores)
            order = ranking.order
            selection_mode = self.ranker.name
            self.ranker_calls += 1

        self._pool = pool
        self._order = list(order)
        self._selection_mode = selection_mode
        self._prompt = final_prompt
        self._raw_output = response.raw_output
        self._model = response.model
        self._provider_metadata = dict(response.metadata)
        self._repair_problems = list(problems)
        self._rejected_indices = []

    def ask(self, remaining_budget: int) -> ProposalBatch:
        if self.context is None:
            raise PolicyContractError(f"arm {self.name!r} was not initialized")
        if self._pool is None or not self._order:
            self._new_pool(remaining_budget)
        assert self._pool is not None and self._selection_mode is not None

        selected_index = self._order.pop(0)
        self._last_selected_index = selected_index
        member = self._pool[selected_index]
        selection_rank = len(self._rejected_indices) + 1
        evidence = {
            "pool_id": f"pool-{self.provider_calls:04d}",
            "pool": [dict(item) for item in self._pool],
            "selected_pool_index": selected_index,
            "selected_rank": selection_rank,
            "selection_mode": self._selection_mode,
            "ranker": self.ranker.name,
            "effective_history_count": len(
                effective_observations(self.context.space, self.observations)
            ),
            "min_ranker_history": self.min_ranker_history,
            "previously_rejected_pool_indices": list(self._rejected_indices),
            "repair_problems": list(self._repair_problems),
            "degraded": bool(self._repair_problems),
            "prompt": self._prompt,
            "raw_output": self._raw_output,
            "model": self._model,
            "provider_metadata": dict(self._provider_metadata),
        }
        return ProposalBatch(
            proposals=(
                Proposal(
                    params=member["params"],
                    origin=self.name,
                    metadata={**evidence, "reason": member["reason"]},
                ),
            ),
            metadata={
                "pool_id": evidence["pool_id"],
                "pool_size": POOL_SIZE,
                "selected_pool_index": selected_index,
                "selected_rank": selection_rank,
                "selection_mode": self._selection_mode,
                "effective_history_count": evidence["effective_history_count"],
            },
        )

    def tell(self, outcomes: tuple[EvaluationOutcome, ...]) -> None:
        if len(outcomes) != 1 or self._last_selected_index is None:
            raise PolicyContractError(
                f"arm {self.name!r} expects one outcome for its selected pool member"
            )
        self.observations.append(outcomes[0].observation)
        if outcomes[0].observation.status == "rejected" and self._order:
            self._rejected_indices.append(self._last_selected_index)
            self._last_selected_index = None
            return
        self._clear_pool()

    def snapshot(self) -> Mapping[str, Any]:
        incumbent = select_incumbent(
            [
                observation
                for observation in self.observations
                if self.context is not None
                and set(observation.params) == set(self.context.space.names)
            ]
        )
        assert self.context is not None
        return {
            "provider_calls": self.provider_calls,
            "provider_attempts": self.provider_attempts,
            "corrective_calls": self.corrective_calls,
            "ranker_calls": self.ranker_calls,
            "ranker": self.ranker.name,
            "min_ranker_history": self.min_ranker_history,
            "degraded_calls": self.degraded_calls,
            "effective_history_count": len(
                effective_observations(self.context.space, self.observations)
            ),
            "incumbent_id": incumbent.observation_id,
            "incumbent_score": incumbent.score,
        }
