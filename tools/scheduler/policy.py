"""The scheduler decision: coverage, rollout comparison, frozen tie rule (§6, §7).

One decision runs three stages:

1. **Terminal check** — no admissible full-bout TUNE and no executable
   DEFER means there is nothing to decide.
2. **Coverage** — when the current-run evidence cannot support the models
   the rollout needs, the rollout would be sampling from a distribution
   that does not exist. The scheduler then takes a bounded mechanical
   bootstrap: collect the missing class of evidence directly. This is
   evidence-gated cold start, not an exploration bonus, and every forced
   collection draws on ONE run-level coverage budget. Once that cap is
   spent the policy falls back to a frozen rule rather than exploring
   forever.
3. **Rollout** — paired full-remaining-budget comparison of every root
   action, then the frozen tie rule when the champion's margin is inside
   the estimator's resolution.

`POLICY_VERSION` is part of every receipt. Rollout values are only
comparable across decisions that share a policy and a reference policy, so
a change to either must change the version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .contract import CandidateView
from .evidence import (
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

POLICY_VERSION = "scheduler-v3.2"


@dataclass(frozen=True)
class CoverageConfig:
    """Bounded evidence bootstrap (§6). Config, not frozen design.

    `budget_cap` is a single run-level total shared by every forced
    collection — the design requires one shared cap so cold start cannot
    consume the run one class at a time.

    `min_first` / `min_later` are how much evidence of a class's OWN kind
    coverage tries to collect before it stops forcing. They are what makes
    coverage do anything at all: without them the model's pooled fallback
    reports every class "supported" the moment three records of any class
    exist, and the FIRST/LATER split the model is built around would never
    be estimated from FIRST/LATER data.
    """

    budget_cap: int = 6
    min_first: int = 3
    min_later: int = 3
    min_arrival_episodes: int = 2


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
    #: Per-bout-class estimation mode at decision time: "exact", "pooled",
    #: or "unsupported". A Q value computed against pooled FIRST+LATER
    #: records is not the same quantity as one computed against the class's
    #: own records, so the receipt has to say which one it is.
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


def _coverage_target(
    state: SchedulerState,
    tuning: TuningModel,
    arrival: ArrivalModel,
    coverage: CoverageConfig,
    spent: int,
) -> tuple[str, str | None, str] | None:
    """The one action forced to collect missing evidence, if any.

    The gate is the *exact-class* count, not `supported()`. `supported()`
    accepts the pooled FIRST+LATER fallback, which is what the rollout uses
    when it must sample something; coverage is the mechanism that tries to
    make that fallback unnecessary, so it has to see the class counts
    themselves. Gating on `supported()` would end coverage the moment any
    three bouts existed and leave the FIRST/LATER distinction permanently
    estimated from pooled data.

    Returns None once both classes are covered as far as this state can
    reach, or the shared cap is spent — in both cases the caller proceeds to
    the normal path (rollout, or the frozen fallback when the models still
    cannot support one).
    """
    if spent >= coverage.budget_cap:
        return None

    eligible = state.eligible()
    first_pool = [c for c in eligible if c.is_first]
    later_pool = [c for c in eligible if not c.is_first]

    # A FIRST bout is only collectable while an untuned candidate exists,
    # and a LATER bout only after some candidate has been tuned once. The
    # order below asks for whichever class is both short and reachable.
    first_short = tuning.exact_count(FIRST) < coverage.min_first
    later_short = tuning.exact_count(LATER) < coverage.min_later
    if first_short and first_pool:
        return (
            "TUNE",
            _coverage_pick(first_pool, state),
            f"coverage: {tuning.exact_count(FIRST)}/{coverage.min_first} "
            "FIRST tuning records in this run",
        )
    if later_short and later_pool:
        return (
            "TUNE",
            _coverage_pick(later_pool, state),
            f"coverage: {tuning.exact_count(LATER)}/{coverage.min_later} "
            "LATER tuning records in this run",
        )
    if not arrival.supported() and state.defer_available():
        return (
            "DEFER",
            None,
            "coverage: fewer than "
            f"{coverage.min_arrival_episodes} observed arrival episodes",
        )
    return None


def _coverage_pick(pool: Sequence[CandidateView], state: SchedulerState) -> str:
    """Which candidate serves a coverage bout.

    Deterministic and evidence-oriented: the smallest headroom, so the
    forced bout is also the one most likely to be useful, with run_id
    breaking ties for replayability.
    """
    return min(
        pool,
        key=lambda candidate: (state.headroom(candidate), candidate.run_id),
    ).run_id


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


def _evidence_mode(tuning: TuningModel, arrival: ArrivalModel) -> dict:
    return {
        FIRST: tuning.usage_mode(FIRST),
        LATER: tuning.usage_mode(LATER),
        "arrival_episodes": len(arrival.episodes),
    }


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
    mode = _evidence_mode(tuning, arrival)

    if state.terminal():
        return Decision(
            action="STOP",
            run_id=None,
            reason="terminal state",
            evidence_mode=mode,
        )

    target = _coverage_target(
        state, tuning, arrival, config.coverage, coverage_spent
    )
    if target is not None:
        action, run_id, reason = target
        return Decision(
            action=action,
            run_id=run_id,
            reason=reason,
            coverage_spent=coverage_spent + 1,
            evidence_mode=mode,
        )

    if not _models_usable(state, tuning, arrival):
        return _fallback(state, "insufficient current-run evidence", mode)

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
