"""Pure lifecycle decisions over durable coordinator and ledger summaries."""

from __future__ import annotations

from typing import Any

from .models import CoordinatorPhase, CoordinatorState, Transition


def next_transition(
    state: CoordinatorState,
    *,
    ledger_exists: bool,
    ledger_brief: dict[str, Any] | None,
    has_provided_baseline: bool,
) -> Transition:
    if state.phase in {CoordinatorPhase.BLOCKED, CoordinatorPhase.COMPLETED}:
        return Transition.STOP

    active = state.active_round
    if active is not None:
        if not active.admission_complete or any(not action.admitted for action in active.actions):
            return Transition.ADMIT_ROUND
        if any(not action.resolved for action in active.actions):
            action = next(action for action in active.actions if not action.resolved)
            if not action.materialized:
                return Transition.MATERIALIZE_CANDIDATE
            if not action.implemented:
                return Transition.MATERIALIZE_CANDIDATE
            if not action.contract_ready:
                return Transition.BUILD_TUNING_CONTRACT
            if not action.preflight_ready:
                return Transition.PREFLIGHT_CANDIDATE
            return Transition.EVALUATE_WARM_CONFIGS
        if not active.tuning_complete:
            return Transition.DEEP_TUNE

    if not ledger_exists:
        return Transition.ADMIT_BASELINE if has_provided_baseline else Transition.ADMIT_ROUND

    if ledger_brief is None:
        raise ValueError("ledger_brief is required when the ledger exists")
    if ledger_brief.get("experience_refresh_required"):
        return Transition.REFRESH_EXPERIENCE
    if ledger_brief.get("reached") or ledger_brief.get("phase") == "completed":
        return Transition.COMPLETE
    if ledger_brief.get("phase") == "blocked":
        return Transition.STOP
    return Transition.ADMIT_ROUND
