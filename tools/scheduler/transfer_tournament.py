"""Deterministic anchor + donor-transfer challenger scheduler.

This policy is the transfer sibling of ``tournament.py``'s early-anchor /
late-challenger tournament, paired with the candidate-aware inner policy
``hebo24-transfer10-hebo10`` (transfer-scheduler design §6). It is
intentionally free of a learned transition model. It buys one ordinary
24-eval INITIAL on the best seed candidate — the anchor, whose tuned
parameters the run-global donor snapshot then transfers into later
candidates — and exactly two 10-eval post-anchor segments. The first goes
to the best donor-initialized challenger (a ``global_donor`` candidate whose
mandatory donor row evaluated finite); with no eligible challenger it is a
DEEP continuation of the anchor instead of stranding the reserved budget.
The second stays with a positive-gain responder, switches to the best
untouched finite-donor challenger after a zero-gain first segment, and falls
back to the anchor when no untouched challenger exists.

Every phase and reserve below is recomputed from mechanical facts — the
anchor is the unique candidate with a completed ordinary INITIAL, and
post-anchor segments spent are ``max(0, anchor.bouts_used - 1) +
sum(global_donor_candidate.bouts_used)`` — so a kill/resume reconstructs
the same decision without any in-memory flag. Artifacts that show more than
one competing ordinary INITIAL fail closed: the policy refuses to guess
which one is the anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contract import (
    MIN_GENERATION_K_EVAL,
    CandidateView,
    ResourceContract,
)
from .policy import Decision
from .state import SchedulerState


POLICY_ID = "anchor_transfer_challenger_v1"
POLICY_VERSION = "scheduler-anchor-transfer-challenger-v1"
POST_ANCHOR_SEGMENTS = 2


class AmbiguousAnchorError(ValueError):
    """The artifacts show more than one completed ordinary INITIAL.

    A normal run has exactly one anchor. Refusing to decide — in decide and
    in replay alike — is the fail-closed answer; guessing which candidate
    anchored the donor frontier would silently rewrite the run's history.
    """


@dataclass(frozen=True)
class TransferStatus:
    anchor: CandidateView | None
    segments_spent: int
    #: The candidate whose bout accounted for the first post-anchor segment
    #: (set once that segment is spent): the tuned donor challenger, or the
    #: anchor itself when it consumed the segment as a DEEP continuation.
    first_segment_target: CandidateView | None

    @property
    def phase(self) -> str:
        if self.anchor is None:
            return "early_anchor"
        if self.segments_spent == 0:
            return "post_anchor_generation"
        if self.segments_spent < POST_ANCHOR_SEGMENTS:
            return "post_anchor_segments"
        return "complete"


def _anchor(state: SchedulerState) -> CandidateView | None:
    """The run's anchor: the unique candidate with a completed ordinary INITIAL.

    Derived from facts alone (mode stamp + completed bouts), so a crashed
    candidate still counts as historical fact — its INITIAL was spent. A
    competing second ordinary INITIAL means the artifacts disagree with the
    policy, which is a hard error rather than a choice.
    """
    anchors = tuple(
        candidate
        for candidate in state.candidates
        if candidate.initialization_mode == "ordinary" and candidate.bouts_used >= 1
    )
    if len(anchors) > 1:
        raise AmbiguousAnchorError(
            "multiple competing ordinary INITIAL candidates "
            f"{[candidate.run_id for candidate in anchors]}; the anchor is "
            "ambiguous and the policy does not guess"
        )
    return anchors[0] if anchors else None


def status(state: SchedulerState) -> TransferStatus:
    anchor = _anchor(state)
    donor_bouts = sum(
        candidate.bouts_used
        for candidate in state.candidates
        if candidate.initialization_mode == "global_donor"
    )
    spent = (
        (max(0, anchor.bouts_used - 1) if anchor is not None else 0)
        + donor_bouts
    )
    target = None
    if anchor is not None and spent > 0:
        # Exactly one candidate can account for a single spent segment: one
        # tuned donor challenger, or the anchor's own second bout.
        target = next(
            (
                candidate
                for candidate in state.candidates
                if candidate.initialization_mode == "global_donor"
                and candidate.bouts_used >= 1
            ),
            None,
        )
        if target is None and anchor.bouts_used >= 2:
            target = anchor
    return TransferStatus(
        anchor=anchor,
        segments_spent=spent,
        first_segment_target=target,
    )


def _segment_cost(contract: ResourceContract) -> int:
    """One post-anchor segment's cost under the candidate-aware contract.

    A segment is a challenger's TRANSFERRED first bout or an anchor/
    responder DEEP continuation; the max keeps the reserve sound for a
    generic schedule, as in the sibling tournament.
    """
    costs = [contract.bout_cost(1), contract.bout_cost(2)]
    if contract.transferred_first_bout_trials is not None:
        costs.append(contract.transferred_first_bout_trials)
    return max(costs)


def full_tuning_reserve(contract: ResourceContract) -> int:
    """One ordinary INITIAL plus exactly two post-anchor segments."""
    return contract.bout_cost(0) + POST_ANCHOR_SEGMENTS * _segment_cost(contract)


def generation_reserve(state: SchedulerState) -> int:
    """Budget that generation must leave untouched at this state."""
    current = status(state)
    if current.phase == "early_anchor":
        return full_tuning_reserve(state.contract)
    if current.phase == "post_anchor_generation":
        return POST_ANCHOR_SEGMENTS * _segment_cost(state.contract)
    if current.phase == "post_anchor_segments":
        remaining_segments = POST_ANCHOR_SEGMENTS - current.segments_spent
        return remaining_segments * _segment_cost(state.contract)
    return 0


def generation_admission_cap(state: SchedulerState) -> int:
    """Maximum new candidates that preserve the transfer reserve.

    No generation may interleave with a spent-but-unfinished pair of
    post-anchor segments. After both segments complete, terminal screening
    may spend the otherwise stranded two-call tail; those candidates cannot
    affect an already-complete tournament.
    """
    current = status(state)
    if current.phase == "post_anchor_segments":
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


def _donor_eligible_challengers(
    state: SchedulerState,
) -> tuple[CandidateView, ...]:
    """Untouched candidates with a finite mandatory donor observation.

    These are the only legal TRANSFERRED first-segment targets (design
    §6.1): stamped ``global_donor`` initialization, donor row evaluated
    finite, no completed bout.
    """
    return tuple(
        candidate
        for candidate in state.candidates
        if candidate.initialization_mode == "global_donor"
        and candidate.donor_finite
        and candidate.bouts_used == 0
    )


def _current_donor_id(state: SchedulerState) -> str | None:
    """The bound donor snapshot id when every donor candidate shares one."""
    ids = {
        candidate.donor_snapshot_id
        for candidate in state.candidates
        if candidate.initialization_mode == "global_donor"
        and candidate.donor_snapshot_id
    }
    return next(iter(ids)) if len(ids) == 1 else None


def _best(
    state: SchedulerState, candidates: tuple[CandidateView, ...]
) -> CandidateView | None:
    eligible_ids = {candidate.run_id for candidate in state.eligible()}
    choices = [
        candidate for candidate in candidates if candidate.run_id in eligible_ids
    ]
    if not choices:
        return None
    return min(choices, key=lambda candidate: (candidate.best_score, candidate.run_id))


def decide(state: SchedulerState) -> Decision:
    """Choose one deterministic transfer-tournament action."""
    current = status(state)
    challengers = _donor_eligible_challengers(state)
    mode = {
        "mode": "deterministic_transfer_tournament",
        "phase": current.phase,
        "anchor_run_id": (
            current.anchor.run_id if current.anchor is not None else None
        ),
        "first_post_anchor_target": (
            current.first_segment_target.run_id
            if current.first_segment_target is not None
            else None
        ),
        "post_anchor_segments_spent": current.segments_spent,
        "generation_reserve": generation_reserve(state),
        "generation_admission_cap": generation_admission_cap(state),
        "donor_eligible_challenger_ids": [
            candidate.run_id
            for candidate in sorted(
                challengers,
                key=lambda candidate: (candidate.best_score, candidate.run_id),
            )
        ],
        "donor_snapshot_id": _current_donor_id(state),
        "n_roots": state.n_roots,
        "n_seed": state.n_seed,
    }

    if current.phase == "complete":
        if generation_admission_cap(state) > 0:
            return Decision(
                "DEFER",
                None,
                "transfer tournament complete; spend terminal screening budget",
                evidence_mode=mode,
            )
        if state.remaining_budget > 0:
            return Decision(
                "STOP",
                None,
                "transfer tournament complete; remaining budget "
                f"{state.remaining_budget} is below the minimum screening "
                f"cost ({MIN_GENERATION_K_EVAL}) and stays unused",
                evidence_mode=mode,
            )
        return Decision(
            "STOP", None, "transfer tournament complete", evidence_mode=mode
        )

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
                f"remaining budget cannot fund full transfer tournament reserve {required}",
                evidence_mode=mode,
            )
        seeds = tuple(
            candidate
            for candidate in state.candidates
            if candidate.initialization_mode == "ordinary"
            and candidate.bouts_used == 0
        )
        target = _best(state, seeds)
        if target is None:
            return Decision(
                "STOP", None, "no eligible early anchor", evidence_mode=mode
            )
        return Decision(
            "TUNE",
            target.run_id,
            "initialize best ordinary seed candidate as the transfer anchor",
            evidence_mode=mode,
        )

    if current.phase == "post_anchor_generation":
        if generation_admission_cap(state) > 0:
            return Decision(
                "DEFER",
                None,
                "continue generation while preserving the two post-anchor "
                "segment reserve",
                evidence_mode=mode,
            )
        target = _best(state, challengers)
        if target is not None:
            return Decision(
                "TUNE",
                target.run_id,
                "generation reserve reached; run the best finite-donor "
                "challenger's TRANSFERRED segment",
                evidence_mode=mode,
            )
        target = (
            _best(state, (current.anchor,)) if current.anchor is not None else None
        )
        if target is not None:
            return Decision(
                "TUNE",
                target.run_id,
                "no eligible donor challenger; continue the anchor with a "
                "DEEP segment instead of stranding the reserve",
                evidence_mode=mode,
            )
        return Decision(
            "STOP",
            None,
            "no eligible challenger or anchor continuation",
            evidence_mode=mode,
        )

    # One post-anchor segment is spent. Spend exactly one more.
    target = current.first_segment_target
    gain = (target.previous_gain or 0.0) if target is not None else 0.0
    if target is not None and gain > 0.0:
        responder = _best(state, (target,))
        if responder is not None:
            return Decision(
                "TUNE",
                responder.run_id,
                "first post-anchor segment improved; continue the responder",
                evidence_mode=mode,
            )
        reason = "responder ineligible for its next bout"
    else:
        reason = "first post-anchor segment had zero gain"
    challenger = _best(state, challengers)
    if challenger is not None:
        return Decision(
            "TUNE",
            challenger.run_id,
            f"{reason}; switch to the best untouched finite-donor challenger",
            evidence_mode=mode,
        )
    if target is not None and current.anchor is not None and (
        target.run_id != current.anchor.run_id
    ):
        continuation = _best(state, (current.anchor,))
        if continuation is not None:
            return Decision(
                "TUNE",
                continuation.run_id,
                f"{reason}; no untouched challenger, continue the anchor",
                evidence_mode=mode,
            )
    return Decision(
        "STOP",
        None,
        "no eligible target for the second post-anchor segment",
        evidence_mode=mode,
    )


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
