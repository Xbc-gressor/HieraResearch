"""Shared fresh-session LLM proposal-pool policy.

Ranker arms deliberately share this module so their LLM prompt, focused
summary, pool validation, and retry behavior are identical.  Only the numeric
ranking object differs between benchmark cells.
"""

from __future__ import annotations

import json
import math
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
    ):
        if min_ranker_history < 1:
            raise ValueError("min_ranker_history must be positive")
        self.name = name
        self.provider = provider
        self.ranker = ranker
        self.summary_builder = summary_builder or FocusedSummaryBuilder()
        self.min_ranker_history = min_ranker_history
        self.context: BenchmarkContext | None = None
        self.observations: list[Observation] = []
        self.provider_calls = 0
        self.ranker_calls = 0
        self._pool: list[dict[str, Any]] | None = None
        self._order: list[int] = []
        self._selection_mode: str | None = None
        self._prompt: str | None = None
        self._raw_output: str | None = None
        self._model: str | None = None
        self._provider_metadata: dict[str, Any] = {}
        self._last_selected_index: int | None = None
        self._rejected_indices: list[int] = []

    def initialize(self, context: BenchmarkContext) -> None:
        self.context = context
        self.observations = list(context.observations)
        self.provider_calls = 0
        self.ranker_calls = 0
        self._clear_pool()

    def _clear_pool(self) -> None:
        self._pool = None
        self._order = []
        self._selection_mode = None
        self._prompt = None
        self._raw_output = None
        self._model = None
        self._provider_metadata = {}
        self._last_selected_index = None
        self._rejected_indices = []

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
        response = self.provider.complete(prompt, output_schema=OUTPUT_SCHEMA)
        raw_pool = response.output.get("proposals")
        if not isinstance(raw_pool, list) or len(raw_pool) != POOL_SIZE:
            raise PolicyContractError(
                f"pool provider must return exactly {POOL_SIZE} proposals"
            )

        pool: list[dict[str, Any]] = []
        for index, item in enumerate(raw_pool):
            if not isinstance(item, Mapping):
                raise PolicyContractError(f"pool proposal {index + 1} is not an object")
            params = item.get("params")
            reason = item.get("reason")
            if not isinstance(params, Mapping):
                raise PolicyContractError(
                    f"pool proposal {index + 1} must contain params"
                )
            if not isinstance(reason, str) or not reason.strip():
                raise PolicyContractError(
                    f"pool proposal {index + 1} must contain a non-empty reason"
                )
            try:
                projected = self.context.space.project(params)
            except ValueError as exc:
                raise PolicyContractError(
                    f"pool proposal {index + 1} is invalid: {exc}"
                ) from exc
            pool.append(
                {
                    "llm_rank": index + 1,
                    "params": projected,
                    "reason": reason,
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

        self.provider_calls += 1
        self._pool = pool
        self._order = list(order)
        self._selection_mode = selection_mode
        self._prompt = prompt
        self._raw_output = response.raw_output
        self._model = response.model
        self._provider_metadata = dict(response.metadata)
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
            "ranker_calls": self.ranker_calls,
            "ranker": self.ranker.name,
            "min_ranker_history": self.min_ranker_history,
            "effective_history_count": len(
                effective_observations(self.context.space, self.observations)
            ),
            "incumbent_id": incumbent.observation_id,
            "incumbent_score": incumbent.score,
        }
