"""Deterministic early-anchor / late-challenger scheduler.

This policy is intentionally free of a learned transition model.  It buys two
complete INITIAL bouts: one immediately after the reserved seed roots are
visible so tuned parameters can transfer into later improve/crossover nodes,
and one at the end of generation as a late challenger.  It then spends exactly
two DEEP segments.  The first goes to the better initialized candidate; the
second stays with a responder and switches to the other initialized candidate
after a zero-gain first DEEP segment.

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
DEEP_SEGMENTS = 2


@dataclass(frozen=True)
class TournamentStatus:
    initialized: tuple[CandidateView, ...]
    uninitialized: tuple[CandidateView, ...]
    deep_segments_spent: int

    @property
    def phase(self) -> str:
        if not self.initialized:
            return "early_anchor"
        if self.deep_segments_spent:
            return "deep_segments" if self.deep_segments_spent < DEEP_SEGMENTS else "complete"
        if len(self.initialized) < INITIALIZED_CANDIDATES:
            return "generation_then_challenger"
        return "deep_segments"


def status(state: SchedulerState) -> TournamentStatus:
    live = tuple(candidate for candidate in state.candidates if not candidate.crashed)
    initialized = tuple(candidate for candidate in live if candidate.bouts_used >= 1)
    return TournamentStatus(
        initialized=initialized,
        uninitialized=tuple(candidate for candidate in live if candidate.bouts_used == 0),
        deep_segments_spent=sum(
            max(0, candidate.bouts_used - 1) for candidate in initialized
        ),
    )


def full_tuning_reserve(contract: ResourceContract) -> int:
    """Two INITIAL bouts plus exactly two DEEP segments."""
    second_deep = max(contract.bout_cost(1), contract.bout_cost(2))
    return (
        2 * contract.bout_cost(0)
        + contract.bout_cost(1)
        + second_deep
    )


def generation_reserve(state: SchedulerState) -> int:
    """Budget that generation must leave untouched at this state."""
    current = status(state)
    if current.phase == "early_anchor":
        return full_tuning_reserve(state.contract)
    if current.phase == "generation_then_challenger":
        second_deep = max(
            state.contract.bout_cost(1), state.contract.bout_cost(2)
        )
        return (
            state.contract.bout_cost(0)
            + state.contract.bout_cost(1)
            + second_deep
        )
    if current.phase == "deep_segments":
        remaining_deep = DEEP_SEGMENTS - current.deep_segments_spent
        # A switched second DEEP segment is still bout index 1 for the other
        # initialized candidate.  The planned policy has equal DEEP costs;
        # max keeps the reserve sound for a generic three-cost schedule.
        deep_cost = max(
            state.contract.bout_cost(1), state.contract.bout_cost(2)
        )
        return remaining_deep * deep_cost
    return 0


def generation_admission_cap(state: SchedulerState) -> int:
    """Maximum new candidates that preserve the tournament reserve.

    No generation may interleave with the DEEP segments. After both segments
    complete, terminal screening may spend the otherwise stranded two-call
    tail; those candidates cannot affect an already-complete tournament.
    """
    current = status(state)
    if current.phase == "deep_segments":
        return 0
    if current.phase == "complete":
        return state.contract.generation_admits(
            state.remaining_budget,
            allow_terminal_degrade=True,
        )
    spendable = max(0, state.remaining_budget - generation_reserve(state))
    return spendable // state.contract.k_eval


def generation_candidate_reservation(state: SchedulerState) -> int:
    """Per-candidate screening cost paired with generation_admission_cap."""
    if status(state).phase == "complete":
        return state.contract.screening_reservation(
            state.remaining_budget,
            allow_terminal_degrade=True,
        )
    return state.contract.k_eval


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
        "deep_segments_spent": current.deep_segments_spent,
        "generation_reserve": generation_reserve(state),
        "generation_admission_cap": generation_admission_cap(state),
        "n_roots": state.n_roots,
        "n_seed": state.n_seed,
    }

    if current.phase == "complete":
        if generation_admission_cap(state) > 0:
            return Decision(
                "DEFER",
                None,
                "tournament complete; spend terminal screening budget",
                evidence_mode=mode,
            )
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
                "continue generation while preserving challenger + DEEP reserve",
                evidence_mode=mode,
            )
        target = _best(state, current.uninitialized)
        if target is None:
            # A short or failure-heavy run may never produce a second usable
            # candidate.  Preserve useful work by letting the anchor consume
            # the DEEP budget rather than spinning forever.
            target = _best(state, current.initialized)
            reason = "no eligible late challenger; continue the early anchor"
        else:
            reason = "generation reserve reached; initialize best late challenger"
        if target is None:
            return Decision("STOP", None, "no eligible tournament candidate", evidence_mode=mode)
        return Decision("TUNE", target.run_id, reason, evidence_mode=mode)

    # Two candidates have INITIAL evidence. Spend exactly two DEEP segments.
    if current.deep_segments_spent == 0:
        target = _best(state, current.initialized)
        reason = "first DEEP segment goes to the better INITIAL candidate"
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
            reason = "first DEEP segment improved; continue the responder"
            if target is None:
                # The responder can be ineligible for its own next bout (e.g.
                # an SPSA DEEP bout without a movable continuous dimension).
                # Switch rather than abandon the reserved second DEEP segment.
                target = _best(state, alternatives)
                reason = (
                    "responder ineligible for its next bout; "
                    "switch initialized candidate"
                )
        else:
            target = _best(state, alternatives) or (
                _best(state, (prior,)) if prior is not None else None
            )
            reason = "first DEEP segment had zero gain; switch INITIAL candidate"
    if target is None:
        return Decision("STOP", None, "no eligible DEEP candidate", evidence_mode=mode)
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
