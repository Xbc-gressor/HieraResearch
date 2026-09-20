"""One scheduler decision against a live run directory.

Both entry points — the CLI and `select-candidate`'s v3.2 branch — go
through `decide_for_run`, so there is exactly one implementation of the
order that matters: reconcile evidence and close executed decisions, build
the exact state, reuse an open decision if this round already has one, and
only then decide and commit.

**Why reuse rather than re-decide.** A decision commits budget. The
orchestrator can reach this code more than once for the same round (a
corrective follow-up, a retried tool call, a crash between the call and the
bout), and a scheduler that answers every one of those with a fresh
committed decision fills the decision log with choices nobody executed.
The policy is a pure function of `(state snapshot, evidence cursor)`, so
an identical pair means the same round: the open decision is returned
unchanged.
"""

from __future__ import annotations

from pathlib import Path

from .contract import ResourceContract
from .evidence import ArrivalModel, TuningModel
from .policy import POLICY_VERSION, PolicyConfig, decide as decide_policy
from .reconcile import reconcile_path
from .rollout import RolloutConfig
from .state import SchedulerState, load_state
from .store import SchedulerStore
from . import round_policy
from . import tournament
from . import transfer_tournament


def config_from_scenarios(scenarios: int | None) -> PolicyConfig:
    """Policy config with an optional Monte-Carlo count override.

    `M` is experiment configuration (§13), not frozen design, so it is
    overridable per run — but it is recorded in every receipt, because two
    decisions taken at different `M` are not directly comparable.
    """
    base = PolicyConfig()
    if scenarios is None:
        return base
    return PolicyConfig(
        rollout=RolloutConfig(
            scenarios=int(scenarios),
            tie_band=base.rollout.tie_band,
            reference_policy=base.rollout.reference_policy,
        ),
        coverage=base.coverage,
    )


def contract_for(ledger_path: Path) -> ResourceContract:
    """The run's resource contract, honoring framework_cfg overrides.

    `run_cfg` has already rejected any override that the real admission
    layer could not honor (`_validate_scheduler_v3_2`), so what arrives here
    is a contract the driver can actually execute.
    """
    from run_cfg import load_run_cfg

    tuner = load_run_cfg(Path(ledger_path).parent, "tuner")
    bout_trials = int(tuner.get("bout_trials", ResourceContract.bout_trials))
    scheduler_policy = str(tuner.get("scheduler_policy", "v3_2"))
    if scheduler_policy in (tournament.POLICY_ID, round_policy.POLICY_ID):
        # Derive the resource schedule from the executable inner policy rather
        # than baking mixup's planned 24/10/10 into the scheduler.  This keeps
        # the scheduler mechanically testable with today's production arms and
        # makes a future mixup policy change the reserve through one contract.
        from tuners.inner_policy import POLICY_ID as DEFAULT_INNER_POLICY_ID
        from tuners.inner_policy import (
            expected_bout_trials,
            numeric_required_from_bout_index,
        )

        inner_policy_id = str(tuner.get("inner_policy", DEFAULT_INNER_POLICY_ID))
        schedule = tuple(
            expected_bout_trials(inner_policy_id, index, bout_trials)
            for index in range(3)
        )
        return ResourceContract(
            bout_trials=schedule[1],
            max_bouts=3,
            k_eval=max(2, int(tuner.get("K_eval", ResourceContract.k_eval))),
            first_bout_trials=schedule[0],
            bout_cost_schedule=schedule,
            numeric_required_from_bout_index=(
                numeric_required_from_bout_index(inner_policy_id)
            ),
        )
    if scheduler_policy == transfer_tournament.POLICY_ID:
        # Same inner-policy-derived schedule as the anchor/challenger branch,
        # plus the candidate-aware TRANSFERRED first-bout price so the
        # scheduler's accounting matches what the driver executes (§5.2).
        from tuners.inner_policy import (
            HEBO_TRANSFER_HEBO_POLICY_ID,
            INITIALIZATION_GLOBAL_DONOR,
            expected_bout_trials,
            numeric_required_from_bout_index,
        )

        inner_policy_id = str(
            tuner.get("inner_policy", HEBO_TRANSFER_HEBO_POLICY_ID)
        )
        schedule = tuple(
            expected_bout_trials(inner_policy_id, index, bout_trials)
            for index in range(3)
        )
        return ResourceContract(
            bout_trials=schedule[1],
            max_bouts=3,
            k_eval=max(2, int(tuner.get("K_eval", ResourceContract.k_eval))),
            first_bout_trials=schedule[0],
            bout_cost_schedule=schedule,
            transferred_first_bout_trials=expected_bout_trials(
                inner_policy_id, 0, bout_trials, INITIALIZATION_GLOBAL_DONOR
            ),
            numeric_required_from_bout_index=(
                numeric_required_from_bout_index(inner_policy_id)
            ),
        )
    # Under the legacy inner policy every bout costs bout_trials; historical
    # regime policies use B_FIRST for initialization when no exact schedule is supplied.
    legacy_inner = str(tuner.get("inner_policy", "")) == "legacy"
    return ResourceContract(
        bout_trials=bout_trials,
        max_bouts=int(
            tuner.get("max_bouts_per_candidate", ResourceContract.max_bouts)
        ),
        k_eval=max(2, int(tuner.get("K_eval", ResourceContract.k_eval))),
        first_bout_trials=(
            bout_trials if legacy_inner else ResourceContract.first_bout_trials
        ),
    )


def models_for(store: SchedulerStore, cursor: int | None = None):
    log = store.evidence
    tuning = TuningModel.from_records(log.tuning_records(cursor=cursor))
    arrival = ArrivalModel.from_records(log.arrival_records(cursor=cursor))
    return tuning, arrival


def decide_for_run(
    ledger_path: Path,
    *,
    scenarios: int | None = None,
    exclude: tuple[str, ...] | list[str] = (),
) -> dict:
    """Reconcile, decide (or reuse), commit. Returns the decision view.

    ``exclude`` (round_v1 tune only): candidates the driver has backed off
    from tuning; part of the decision identity, like rewrite channels'.
    """
    ledger_path = Path(ledger_path)
    run_dir = ledger_path.parent
    exclude = sorted(str(r) for r in exclude)
    store = SchedulerStore(run_dir)
    contract = contract_for(ledger_path)
    config = config_from_scenarios(scenarios)
    from run_cfg import load_run_cfg

    scheduler_policy = str(
        load_run_cfg(run_dir, "tuner").get("scheduler_policy", "v3_2")
    )


    # Evidence first: the models must see every bout and arrival the run's
    # artifacts already record, whether or not anything reported them. This
    # also closes the decision the previous round executed, which is what
    # lets the reuse check below distinguish "same round again" from "the
    # next round happens to look identical".
    reconciled = reconcile_path(ledger_path, k_eval=contract.k_eval)

    state = load_state(ledger_path, contract=contract)
    snapshot_id = store.put_snapshot(state.snapshot())
    cursor = store.evidence.cursor()

    existing = store.open_decision(snapshot_id, cursor)
    # A rewrite decision (round_v1) is closed by the driver, never by a bout;
    # an open one left by a crash must not stand in for this tune decision.
    # A round_v1 TUNE decision is never reused through this generic path
    # either: the cycle lives in round_state, not the ledger, so snapshot
    # identity cannot see the phase boundary and would resurrect an earlier
    # cycle's decision whenever the ledger did not move across it.
    if (
        existing is not None
        and existing.get("selected_action") != round_policy.REWRITE
        and not (
            scheduler_policy == round_policy.POLICY_ID
            and existing.get("selected_action") == "TUNE"
        )
    ):
        return _view(existing, state, reconciled, reused=True)

    if scheduler_policy == round_policy.POLICY_ID:
        # Snapshot identity cannot survive a partially executed bout (the
        # state has advanced), so within one optimization phase reuse is
        # keyed on the cycle instead: an interrupted bout's decision must be
        # re-issued, not re-decided against drifted state.
        resumed = round_policy.open_tune_decision(store, run_dir, exclude)
        if resumed is not None:
            return _view(resumed, state, reconciled, reused=True)

    tuning = arrival = None
    if scheduler_policy == tournament.POLICY_ID:
        decision = tournament.decide(state)
    elif scheduler_policy == transfer_tournament.POLICY_ID:
        decision = transfer_tournament.decide(state)
    elif scheduler_policy == round_policy.POLICY_ID:
        decision = round_policy.decide(state, run_dir, exclude=exclude)
    else:
        tuning, arrival = models_for(store)
        decision = decide_policy(
            state,
            tuning,
            arrival,
            config=config,
            coverage_spent=store.coverage_spent(),
        )
    decision_id = store.next_decision_id()
    if scheduler_policy == tournament.POLICY_ID:
        receipt = tournament.receipt(
            decision,
            state=state,
            evidence_cursor=cursor,
            snapshot_id=snapshot_id,
            decision_id=decision_id,
        )
    elif scheduler_policy == transfer_tournament.POLICY_ID:
        receipt = transfer_tournament.receipt(
            decision,
            state=state,
            evidence_cursor=cursor,
            snapshot_id=snapshot_id,
            decision_id=decision_id,
        )
    elif scheduler_policy == round_policy.POLICY_ID:
        receipt = round_policy.receipt(
            decision,
            state=state,
            evidence_cursor=cursor,
            snapshot_id=snapshot_id,
            decision_id=decision_id,
            exclude_run_ids=exclude,
        )
        receipt["round_cycle"] = int(
            round_policy.load_round_state(run_dir).get("cycle", 0)
        )
    else:
        receipt = decision.receipt(
            state=state,
            evidence_cursor=cursor,
            config=config,
            snapshot_id=snapshot_id,
            decision_id=decision_id,
        )
    store.append_decision(receipt)
    return _view(
        receipt,
        state,
        reconciled,
        reused=False,
        evidence=(
            {"tuning": tuning.summary(), "arrival": arrival.summary()}
            if tuning is not None and arrival is not None
            else None
        ),
    )


def peek_tune_for_run(ledger_path: Path) -> dict:
    """Select a round_v1 tune target without persisting scheduler state."""
    ledger_path = Path(ledger_path)
    run_dir = ledger_path.parent
    contract = contract_for(ledger_path)
    state = load_state(ledger_path, contract=contract)
    decision = round_policy.select_tune(state, run_dir)
    return {"action": decision.action, "run_id": decision.run_id,
            "reason": decision.reason,
            "evidence_mode": decision.evidence_mode,
            "reference": (decision.evidence_mode or {}).get("reference")}


def select_rewrite_for_run(ledger_path: Path,
                           exclude: tuple[str, ...] | list[str] = ()) -> dict:
    """round_v1: choose (or reuse) this phase's rewrite target and commit it.

    Same commit discipline as `decide_for_run`: reconcile, build the exact
    state, reuse an open REWRITE decision for the identical state, otherwise
    decide and append. The driver closes the decision with `record` once the
    bout has run. ``exclude`` names the candidates other concurrent channels
    are climbing; it is part of the decision's identity (STOP rows too), so
    channels with different exclude sets never share a decision.
    """
    ledger_path = Path(ledger_path)
    run_dir = ledger_path.parent
    exclude = sorted(str(r) for r in exclude)
    store = SchedulerStore(run_dir)
    contract = contract_for(ledger_path)
    reconciled = reconcile_path(ledger_path, k_eval=contract.k_eval)
    state = load_state(ledger_path, contract=contract)
    snapshot_id = store.put_snapshot(state.snapshot())
    cursor = store.evidence.cursor()
    existing = store.open_decision(snapshot_id, cursor, exclude)
    if existing is not None and existing.get("selected_action") in (
            round_policy.REWRITE, "STOP"):
        return _view(existing, state, reconciled, reused=True)
    decision = round_policy.select_rewrite(state, run_dir, exclude=exclude)
    receipt = round_policy.receipt(
        decision,
        state=state,
        evidence_cursor=cursor,
        snapshot_id=snapshot_id,
        decision_id=store.next_decision_id(),
        exclude_run_ids=exclude,
    )
    store.append_decision(receipt)
    return _view(receipt, state, reconciled, reused=False)


def _view(
    receipt: dict,
    state: SchedulerState,
    reconciled: dict,
    *,
    reused: bool,
    evidence: dict | None = None,
) -> dict:
    return {
        "decision_id": receipt.get("decision_id"),
        "action": receipt.get("selected_action"),
        "run_id": receipt.get("selected_run_id"),
        "reason": receipt.get("reason"),
        "state_snapshot_id": receipt.get("state_snapshot_id"),
        "policy_version": receipt.get("policy_version", POLICY_VERSION),
        "evidence_cursor": receipt.get("evidence_cursor"),
        "round_cycle": receipt.get("round_cycle"),
        "prior_id": receipt.get("prior_id"),
        "evidence_mode": receipt.get("evidence_mode", {}),
        "exclude_run_ids": receipt.get("exclude_run_ids") or [],
        "coverage_spent": receipt.get("coverage_spent", 0),
        "reused_open_decision": reused,
        "bout_trials": state.contract.bout_trials,
        "remaining_budget": state.remaining_budget,
        "eligible_run_ids": [c.run_id for c in state.eligible()],
        "defer_available": state.defer_available(),
        "reconciled": reconciled,
        "state": state,
        "evidence": evidence,
    }


__all__ = [
    "config_from_scenarios",
    "contract_for",
    "decide_for_run",
    "peek_tune_for_run",
    "models_for",
    "select_rewrite_for_run",
]
