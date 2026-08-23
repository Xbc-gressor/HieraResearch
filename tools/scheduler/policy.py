"""Historical v3.2 comparison scheduler: seed-set gate, then rollout (§6, §7).

New runs use ``anchor_challenger_v1`` in ``tournament.py``; this module does
not define the current scheduler or its INITIAL/DEEP taxonomy.

One decision runs three stages:

1. **Terminal check** — no admissible full-bout TUNE and no executable
   DEFER means there is nothing to decide.
2. **Seed-set gate** — while `n_roots < n_seed` and DEFER can still buy
   another generation round, defer. Those reserved fresh roots arrive
   whether we tune or not; a FIRST spent before the set is visible is
   a timing error, not a value comparison.
3. **Rollout** — paired full-remaining-budget comparison of every root
   action, then the frozen tie rule when the champion's margin is inside
   the estimator's resolution.

Predictive models start from the versioned design prior in `prior.py`.
Current-run records replace a class once that class has enough of its own
observations. The scheduler does not force a TUNE or DEFER to fill a
class count: spending run budget to feed `P(Z, D | q)` inverts the
objective.

`POLICY_VERSION` is part of every receipt. Rollout values are only
comparable across decisions that share a policy, a reference policy, and
the same prior id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .evidence import (
    PRIOR_ID,
    ArrivalModel,
    FIRST,
    LATER,
    TuningModel,
    bout_class,
)
from .rollout import (
    ActionValue,
    RolloutConfig,
    evaluate_actions,
    paired_difference,
)
from .state import SchedulerState

POLICY_VERSION = "scheduler-v3.2.2"


@dataclass(frozen=True)
class CoverageConfig:
    """Disabled sample-seeking coverage. Kept so receipts and configs
    still carry the field; `budget_cap=0` means the gate never fires.

    Earlier builds forced 3 FIRST + 3 LATER bouts (and arrival episodes)
    so the plug-in model would have exact-class support. That spent the
    run on the scheduler's own evidence instead of terminal raw best.
    """

    budget_cap: int = 0
    min_first: int = 0
    min_later: int = 0
    min_arrival_episodes: int = 0


@dataclass(frozen=True)
class PolicyConfig:
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)


@dataclass(frozen=True)
class Decision:
    """The chosen action plus everything needed to replay it (§8)."""

    action: str
    run_id: str | None
    reason: str
    values: tuple[ActionValue, ...] = ()
    margin: dict | None = None
    coverage_spent: int = 0
    tie_broken: bool = False
    #: Per-class estimation mode at decision time: "exact", "prior", or
    #: "unsupported". A Q value computed against the frozen design prior
    #: is not the same quantity as one computed against this run's own
    #: records, so the receipt has to say which one it is.
    evidence_mode: dict = field(default_factory=dict)

    def receipt(
        self,
        *,
        state: SchedulerState,
        evidence_cursor: int,
        config: PolicyConfig,
        snapshot_id: str | None = None,
        decision_id: str | None = None,
    ) -> dict:
        return {
            "schema_version": 1,
            "kind": "scheduler_decision",
            "decision_id": decision_id,
            "state_snapshot_id": snapshot_id,
            "policy_version": POLICY_VERSION,
            "prior_id": PRIOR_ID,
            "reference_policy_version": (
                config.rollout.reference_policy.version
            ),
            "evidence_cursor": evidence_cursor,
            "paired_scenarios": config.rollout.scenarios,
            "crn_scheme": "event_identity",
            "selected_action": self.action,
            "selected_run_id": self.run_id,
            "reason": self.reason,
            "tie_broken": self.tie_broken,
            "coverage_spent": self.coverage_spent,
            "evidence_mode": self.evidence_mode,
            "q_hat": [
                {
                    "action": value.action,
                    "run_id": value.run_id,
                    "q_hat": value.q_hat,
                }
                for value in self.values
            ],
            "margin": self.margin,
            "global_best": state.global_best,
            "remaining_budget": state.remaining_budget,
        }


def _models_usable(
    state: SchedulerState,
    tuning: TuningModel,
    arrival: ArrivalModel,
) -> bool:
    """Whether the rollout can simulate every action the state offers.

    A rollout that cannot simulate DEFER's arrivals, or cannot simulate the
    bout classes its candidates would run, does not produce a comparable Q
    — it produces a truncated one, biased toward whichever branch it could
    still expand.
    """
    eligible = state.eligible()
    if eligible:
        classes = {bout_class(candidate.bouts_used) for candidate in eligible}
        if not all(tuning.supported(klass) for klass in classes):
            return False
    if state.defer_available() and not arrival.supported():
        return False
    return bool(eligible) or state.defer_available()


def _fallback(state: SchedulerState, reason: str, mode: dict | None = None) -> Decision:
    """Frozen rule for when the rollout cannot run (§6).

    Prefer the eligible candidate with the smallest headroom; DEFER only
    when nothing is eligible. This matches `pi_ref`, so a fallback decision
    is at least consistent with the continuation policy every rollout
    assumes.
    """
    mode = mode or {}
    eligible = state.eligible()
    if eligible:
        target = min(
            eligible,
            key=lambda candidate: (
                state.headroom(candidate),
                candidate.bouts_used,
                candidate.run_id,
            ),
        )
        return Decision(
            action="TUNE",
            run_id=target.run_id,
            reason=f"fallback ({reason}): smallest-headroom eligible candidate",
            evidence_mode=mode,
        )
    if state.defer_available():
        return Decision(
            action="DEFER",
            run_id=None,
            reason=f"fallback ({reason}): no eligible candidate",
            evidence_mode=mode,
        )
    return Decision(
        action="STOP",
        run_id=None,
        reason="terminal state",
        evidence_mode=mode,
    )


def _tie_break(
    values: Sequence[ActionValue],
    state: SchedulerState,
) -> ActionValue:
    """Frozen deterministic tie rule (§7).

    Inside the estimator's resolution the Q ordering is noise, so the rule
    must not be "whatever sorted first by value". Exact ties are not rare
    here: whenever the remaining budget suffices to run every candidate to
    its bout cap, the terminal best is order-invariant and every TUNE
    action has genuinely equal value. Falling back to the smallest run_id
    would then pick an arbitrary candidate for the rest of the run.

    So the rule is: TUNE before DEFER — within a tie a bout yields evidence
    and a defer yields none — then the same ordering `pi_ref` uses
    (smallest headroom, fewest bouts, run_id). Tie, fallback, and rollout
    continuation therefore all prefer the same candidate, and run_id only
    settles what remains.
    """
    by_id = {candidate.run_id: candidate for candidate in state.candidates}

    def order(value: ActionValue) -> tuple:
        candidate = by_id.get(value.run_id) if value.run_id else None
        if candidate is None:  # DEFER
            return (1, 0.0, 0, "")
        return (
            0,
            state.headroom(candidate),
            candidate.bouts_used,
            candidate.run_id,
        )

    return min(values, key=order)


def _evidence_mode(
    tuning: TuningModel,
    arrival: ArrivalModel,
    state: SchedulerState | None = None,
) -> dict:
    mode = {
        FIRST: tuning.usage_mode(FIRST),
        LATER: tuning.usage_mode(LATER),
        "arrival": arrival.usage_mode(),
        "arrival_episodes": len(arrival.episodes),
        "prior_id": PRIOR_ID,
    }
    if state is not None:
        mode["n_roots"] = state.n_roots
        mode["n_seed"] = state.n_seed
        mode["seed_set_incomplete"] = state.seed_set_incomplete()
    return mode


def decide(
    state: SchedulerState,
    tuning: TuningModel,
    arrival: ArrivalModel,
    *,
    config: PolicyConfig | None = None,
    coverage_spent: int = 0,
) -> Decision:
    """Choose TUNE(i) or DEFER for one decision point."""
    config = config or PolicyConfig()
    mode = _evidence_mode(tuning, arrival, state)

    if state.terminal():
        return Decision(
            action="STOP",
            run_id=None,
            reason="terminal state",
            evidence_mode=mode,
        )

    if state.seed_set_incomplete():
        return Decision(
            action="DEFER",
            run_id=None,
            reason=(
                f"seed set incomplete ({state.n_roots}/{state.n_seed}); "
                "defer until the reserved fresh roots are visible"
            ),
            evidence_mode=mode,
        )

    if not _models_usable(state, tuning, arrival):
        return _fallback(state, "models have no support", mode)

    values = evaluate_actions(state, tuning, arrival, config.rollout)
    if not values:
        return _fallback(state, "no root action produced a value", mode)

    ordered = sorted(values, key=lambda value: value.q_hat, reverse=True)
    champion = ordered[0]
    if len(ordered) == 1:
        return Decision(
            action=champion.action,
            run_id=champion.run_id,
            reason="only admissible root action",
            values=tuple(values),
            coverage_spent=coverage_spent,
            evidence_mode=mode,
        )

    runner_up = ordered[1]
    margin = paired_difference(champion, runner_up)
    resolution = config.rollout.tie_band
    if margin["se"] is not None:
        resolution = max(resolution, margin["se"])
    tied = [
        value
        for value in ordered
        if champion.q_hat - value.q_hat <= resolution
    ]
    if len(tied) > 1:
        chosen = _tie_break(tied, state)
        return Decision(
            action=chosen.action,
            run_id=chosen.run_id,
            reason=(
                f"paired margin {margin['mean']:.6g} within estimator "
                f"resolution {resolution:.6g}; frozen tie rule"
            ),
            values=tuple(values),
            margin=margin,
            coverage_spent=coverage_spent,
            tie_broken=True,
            evidence_mode=mode,
        )
    return Decision(
        action=champion.action,
        run_id=champion.run_id,
        reason=(
            f"highest paired Q_hat {champion.q_hat:.6g} over "
            f"{config.rollout.scenarios} paired scenarios"
        ),
        values=tuple(values),
        margin=margin,
        coverage_spent=coverage_spent,
        evidence_mode=mode,
    )
