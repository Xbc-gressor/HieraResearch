"""Deterministic focused history for fresh LLM tuner calls."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .core import BenchmarkContext, Observation, select_incumbent


def _changes(params: Mapping[str, Any], incumbent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {"from": incumbent[name], "to": value}
        for name, value in params.items()
        if incumbent.get(name) != value
    }


def _row(observation: Observation, incumbent: Observation, reason: str) -> dict[str, Any]:
    score_delta = None
    if observation.score is not None and math.isfinite(observation.score):
        score_delta = observation.score - float(incumbent.score)
    return {
        "observation_id": observation.observation_id,
        "selected_as": reason,
        "changes_from_incumbent": _changes(observation.params, incumbent.params),
        "score": observation.score if observation.score is None or math.isfinite(observation.score) else "+inf",
        "score_delta_from_incumbent": score_delta,
        "status": observation.status,
        "origin": observation.origin,
        "failure": observation.failure,
    }


class FocusedSummaryBuilder:
    """Always include current state; cap selectively retrieved history at eight."""

    def build(
        self,
        context: BenchmarkContext,
        observations: Sequence[Observation],
        *,
        remaining_budget: int,
    ) -> dict[str, Any]:
        compatible = []
        for observation in observations:
            if set(observation.params) != set(context.space.names):
                continue
            try:
                context.space.project(observation.params)
            except ValueError:
                continue
            compatible.append(observation)
        incumbent = select_incumbent(compatible)
        attempted = [
            observation
            for observation in compatible
            if observation.status in {"ok", "crash", "rejected"}
        ]

        improvements: list[Observation] = []
        best = math.inf
        for observation in attempted:
            if (
                observation.status == "ok"
                and observation.eligible_incumbent
                and observation.score is not None
                and observation.score < best
            ):
                if best < math.inf:
                    improvements.append(observation)
                best = observation.score

        selected: dict[str, tuple[Observation, list[str]]] = {}

        def add(items: Sequence[Observation], reason: str) -> None:
            for observation in items:
                if observation.observation_id in selected:
                    selected[observation.observation_id][1].append(reason)
                else:
                    selected[observation.observation_id] = (observation, [reason])

        add(attempted[-4:], "recent")
        add(improvements[-2:], "incumbent_improvement")

        improvement_ids = {observation.observation_id for observation in improvements}
        nearby = [
            observation
            for observation in attempted
            if observation.observation_id != incumbent.observation_id
            and observation.observation_id not in improvement_ids
        ]
        nearby.sort(key=lambda observation: context.space.distance(observation.params, incumbent.params))
        add(nearby[:2], "nearest_non_improvement")

        rows = []
        for observation, reasons in selected.values():
            rows.append(_row(observation, incumbent, "+".join(reasons)))
        rows = rows[-8:]
        return {
            "checkpoint_id": context.checkpoint_id,
            "regime": context.regime,
            "remaining_budget": remaining_budget,
            "search_space": context.space.describe(),
            "incumbent": {
                "observation_id": incumbent.observation_id,
                "params": dict(incumbent.params),
                "score": incumbent.score,
            },
            "focused_history": rows,
        }
