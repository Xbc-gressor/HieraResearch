"""Deterministic early-anchor / late-challenger scheduler.

This policy is intentionally free of a learned transition model.  It buys two
complete first bouts: one immediately after the reserved seed roots are
visible so tuned parameters can transfer into later improve/crossover nodes,
and one at the end of generation as a late challenger.  It then spends exactly
two later bouts.  The first goes to the better initialized candidate; the
second stays with a responder and switches to the other initialized candidate
after a zero-gain first later bout.

The policy is generic over the active inner tuner's three-bout cost schedule.
For the planned mixup policy that schedule is (24, 10, 10), hence the full
tournament reserve is 68 evaluations.  Existing regime policies can exercise
the same mechanics with their own real costs without pretending to run mixup.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contract import CandidateView, ResourceContract
from .policy import Decision
from .state import SchedulerState


POLICY_ID = "anchor_challenger_v1"
POLICY_VERSION = "scheduler-anchor-challenger-v1"
INITIALIZED_CANDIDATES = 2
LATER_BOUTS = 2


@dataclass(frozen=True)
class TournamentStatus:
    initialized: tuple[CandidateView, ...]
    uninitialized: tuple[CandidateView, ...]
    later_bouts_spent: int

    @property
    def phase(self) -> str:
        if not self.initialized:
            return "early_anchor"
        if self.later_bouts_spent:
            return "later_bouts" if self.later_bouts_spent < LATER_BOUTS else "complete"
        if len(self.initialized) < INITIALIZED_CANDIDATES:
            return "generation_then_challenger"
        return "later_bouts"


def status(state: SchedulerState) -> TournamentStatus:
    live = tuple(candidate for candidate in state.candidates if not candidate.crashed)
    initialized = tuple(candidate for candidate in live if candidate.bouts_used >= 1)
    return TournamentStatus(
        initialized=initialized,
        uninitialized=tuple(candidate for candidate in live if candidate.bouts_used == 0),
        later_bouts_spent=sum(
            max(0, candidate.bouts_used - 1) for candidate in initialized
        ),
    )


def full_tuning_reserve(contract: ResourceContract) -> int:
    """Two first bouts plus exactly two later bouts."""
    second_later = max(contract.bout_cost(1), contract.bout_cost(2))
    return (
        2 * contract.bout_cost(0)
        + contract.bout_cost(1)
        + second_later
    )


def generation_reserve(state: SchedulerState) -> int:
    """Budget that generation must leave untouched at this state."""
    current = status(state)
    if current.phase == "early_anchor":
        return full_tuning_reserve(state.contract)
    if current.phase == "generation_then_challenger":
        second_later = max(
            state.contract.bout_cost(1), state.contract.bout_cost(2)
        )
        return (
            state.contract.bout_cost(0)
            + state.contract.bout_cost(1)
            + second_later
        )
    if current.phase == "later_bouts":
        remaining_later = LATER_BOUTS - current.later_bouts_spent
        # A switched second later bout is still bout index 1 for the other
        # initialized candidate.  The planned policy has equal later costs;
        # max keeps the reserve sound for a generic three-cost schedule.
        later_cost = max(
            state.contract.bout_cost(1), state.contract.bout_cost(2)
        )
        return remaining_later * later_cost
    return 0


def generation_admission_cap(state: SchedulerState) -> int:
    """Maximum new candidates that preserve the tournament reserve.

    Generation ends permanently once the challenger has completed its first
    bout or any later bout has begun.  Before then, admission uses the same
    ``K_eval`` cost as got_select and leaves the appropriate hard reserve.
    """
    current = status(state)
    if current.phase in {"later_bouts", "complete"}:
        return 0
    spendable = max(0, state.remaining_budget - generation_reserve(state))
    return spendable // state.contract.k_eval


def _best(state: SchedulerState, candidates: tuple[CandidateView, ...]):
    eligible_ids = {candidate.run_id for candidate in state.eligible()}
    choices = [candidate for candidate in candidates if candidate.run_id in eligible_ids]
    if not choices:
        return None
    return min(choices, key=lambda candidate: (candidate.best_score, candidate.run_id))


def decide(state: SchedulerState) -> Decision:
    """Choose one deterministic tournament action."""
    current = status(state)
    mode = {
        "mode": "deterministic_tournament",
        "phase": current.phase,
        "initialized_run_ids": [candidate.run_id for candidate in current.initialized],
        "later_bouts_spent": current.later_bouts_spent,
        "generation_reserve": generation_reserve(state),
        "generation_admission_cap": generation_admission_cap(state),
        "n_roots": state.n_roots,
        "n_seed": state.n_seed,
    }

    if current.phase == "complete":
        return Decision("STOP", None, "tournament complete", evidence_mode=mode)

    if (
        state.n_roots is not None
        and state.n_seed > 0
        and state.n_roots < state.n_seed
        and generation_admission_cap(state) > 0
    ):
        return Decision(
            "DEFER",
            None,
            f"seed set incomplete ({state.n_roots}/{state.n_seed})",
            evidence_mode=mode,
        )

    if current.phase == "early_anchor":
        required = full_tuning_reserve(state.contract)
        if state.remaining_budget < required:
            return Decision(
                "STOP",
                None,
                f"remaining budget cannot fund full tournament reserve {required}",
                evidence_mode=mode,
            )
        target = _best(state, current.uninitialized)
        if target is None:
            return Decision("STOP", None, "no eligible early anchor", evidence_mode=mode)
        return Decision(
            "TUNE",
            target.run_id,
            "initialize best seed-set candidate as transferable early anchor",
            evidence_mode=mode,
        )

    if current.phase == "generation_then_challenger":
        if generation_admission_cap(state) > 0:
            return Decision(
                "DEFER",
                None,
                "continue generation while preserving challenger + later reserve",
                evidence_mode=mode,
            )
        target = _best(state, current.uninitialized)
        if target is None:
            # A short or failure-heavy run may never produce a second usable
            # candidate.  Preserve useful work by letting the anchor consume
            # the later budget rather than spinning forever.
            target = _best(state, current.initialized)
            reason = "no eligible late challenger; continue the early anchor"
        else:
            reason = "generation reserve reached; initialize best late challenger"
        if target is None:
            return Decision("STOP", None, "no eligible tournament candidate", evidence_mode=mode)
        return Decision("TUNE", target.run_id, reason, evidence_mode=mode)

    # Two candidates have first-bout evidence.  Spend exactly two later bouts.
    if current.later_bouts_spent == 0:
        target = _best(state, current.initialized)
        reason = "first later bout goes to the better initialized candidate"
    else:
        prior = next(
            (candidate for candidate in current.initialized if candidate.bouts_used > 1),
            None,
        )
        alternatives = tuple(
            candidate
            for candidate in current.initialized
            if prior is None or candidate.run_id != prior.run_id
        )
        if prior is not None and (prior.previous_gain or 0.0) > 0.0:
            target = _best(state, (prior,))
            reason = "first later bout improved; continue the responder"
            if target is None:
                # The responder can be ineligible for its own next bout (e.g.
                # an SPSA DEEP bout without a movable continuous dimension).
                # Switch rather than abandon the reserved second later bout.
                target = _best(state, alternatives)
                reason = (
                    "responder ineligible for its next bout; "
                    "switch initialized candidate"
                )
        else:
            target = _best(state, alternatives) or (
                _best(state, (prior,)) if prior is not None else None
            )
            reason = "first later bout had zero gain; switch initialized candidate"
    if target is None:
        return Decision("STOP", None, "no eligible later-bout candidate", evidence_mode=mode)
    return Decision("TUNE", target.run_id, reason, evidence_mode=mode)


def receipt(
    decision: Decision,
    *,
    state: SchedulerState,
    evidence_cursor: int,
    snapshot_id: str,
    decision_id: str,
) -> dict:
    """Persist the same stable scheduler receipt envelope without rollout fields."""
    return {
        "schema_version": 1,
        "kind": "scheduler_decision",
        "decision_id": decision_id,
        "state_snapshot_id": snapshot_id,
        "policy_version": POLICY_VERSION,
        "prior_id": None,
        "reference_policy_version": None,
        "evidence_cursor": evidence_cursor,
        "paired_scenarios": 0,
        "crn_scheme": None,
        "selected_action": decision.action,
        "selected_run_id": decision.run_id,
        "reason": decision.reason,
        "tie_broken": False,
        "coverage_spent": 0,
        "evidence_mode": decision.evidence_mode,
        "q_hat": [],
        "margin": None,
        "global_best": state.global_best,
        "remaining_budget": state.remaining_budget,
    }
