"""Derive the scheduler's evidence from durable run artifacts.

The evidence log is what the predictive models are estimated from, so it
cannot depend on a role session remembering to report an outcome: a
forgotten call does not raise, it silently leaves the current-run log
empty so the models stay on the frozen design prior. The
artifacts already record everything both models need —

* a bout's `(Z, D, cost)` is in the candidate's `tune_report.json`
  (`derive_tuning_records`);
* an arrival episode's ordered warm gaps are in the ledger, anchored to the
  global best that held before the episode's candidates existed;

— so `reconcile()` recomputes both from those artifacts and appends only
what the log is missing. It runs before every decision, which makes the
evidence a *view of the run*, not a side effect of the loop.

Arrival episodes are grouped by ledger order rather than by a driver-side
round marker: the ledger has no round field, and adding one to a shared
schema for this consumer alone would be a wider change than the grouping
warrants. Consecutive untuned candidates that appeared after the last
recorded episode form the next episode, which is exactly what one
generation round contributes.
"""

from __future__ import annotations

import json
from pathlib import Path

from .contract import DEFAULT_K_EVAL
from .evidence import (
    EVIDENCE_FILENAME,
    INFRA_FAILURE,
    VALID,
    ArrivalRecord,
    EvidenceLog,
    TuningRecord,
    derive_tuning_records,
)
from .state import candidate_score, is_finite_score


def _recorded_bout_keys(log: EvidenceLog) -> set[tuple[str | None, int | None]]:
    return {(r.run_id, r.bout_index) for r in log.tuning_records()}


def _recorded_arrival_ids(log: EvidenceLog) -> set[str]:
    return {
        run_id for record in log.arrival_records() for run_id in record.run_ids
    }


def _arrival_episode(
    ledger: dict,
    pending: list[dict],
    k_eval: int,
) -> ArrivalRecord | None:
    """One episode from the candidates admitted after the recorded frontier.

    The anchor is the run's raw global best among the candidates that
    already existed when this episode was generated — the same pre-episode
    `g_0` equation (7) re-anchors a resampled episode to. Each prior
    candidate contributes its *current* best (`final_best_score` once
    tuned, else its warm score) via `candidate_score`, because that is
    what `g` means. Anchoring on warm scores alone would measure new
    arrivals against a global best the run had already beaten, and would
    report an arrival as improving whenever tuning had moved `g` past it.

    A run's first episode has no anchor: there is no "global best before"
    to measure its gaps from. It is still recorded, with no gaps, so the
    frontier advances — otherwise those candidates would be swallowed into
    every later episode forever. `ArrivalModel` drops gapless episodes, so
    the record marks what has been consumed without becoming evidence.
    """
    if not pending:
        return None
    pending_ids = {str(row.get("run_id")) for row in pending}
    prior = [
        score
        for record in ledger.get("records", [])
        if str(record.get("run_id")) not in pending_ids
        and record.get("status") != "crash"
        for score in [candidate_score(record)]
        if score is not None
    ]
    anchor = min(prior) if prior else None

    gaps: list[float | None] = []
    statuses: list[str] = []
    run_ids: list[str] = []
    for record in pending:
        run_ids.append(str(record.get("run_id")))
        if anchor is None:
            statuses.append("unanchored")
            continue
        # An arrival's contribution is its warm score: that is what a newly
        # generated candidate is worth before any bout runs on it. A later
        # tuned score belongs to the tuning evidence, not to arrival.
        warm = record.get("best_warm_score")
        if is_finite_score(warm) and record.get("status") != "crash":
            gaps.append(float(warm) - anchor)
            statuses.append("valid")
        else:
            # No usable candidate, but the slots were spent. `None` keeps
            # the cost in the episode without inventing a score for it.
            gaps.append(None)
            statuses.append("infra_failure")
    return ArrivalRecord(
        warm_gaps=tuple(gaps),
        planned_count=len(pending),
        per_candidate_cost=max(1, k_eval),
        statuses=tuple(statuses),
        run_ids=tuple(run_ids),
    )


def reconcile(
    run_dir: Path,
    ledger: dict,
    *,
    k_eval: int = DEFAULT_K_EVAL,
) -> dict:
    """Append missing tuning and arrival evidence; report what was added."""
    run_dir = Path(run_dir)
    log = EvidenceLog(run_dir / ".scheduler" / EVIDENCE_FILENAME)

    seen_bouts = _recorded_bout_keys(log)
    added_bouts: list[TuningRecord] = []
    for record in derive_tuning_records(run_dir, ledger):
        if (record.run_id, record.bout_index) in seen_bouts:
            continue
        log.append(record)
        added_bouts.append(record)

    seen_arrivals = _recorded_arrival_ids(log)
    pending = [
        record
        for record in ledger.get("records", [])
        if str(record.get("run_id")) not in seen_arrivals
        and record.get("status") not in ("pending", None)
    ]
    episode = _arrival_episode(ledger, pending, k_eval)
    if episode is not None:
        log.append(episode)

    bound = _bind_outcomes(run_dir, added_bouts, episode)
    return {
        "tuning_records_added": len(added_bouts),
        "arrival_episodes_added": 1 if episode is not None else 0,
        "outcomes_bound": bound,
        "cursor": log.cursor(),
    }


def _bind_outcomes(
    run_dir: Path,
    bouts: list[TuningRecord],
    episode: ArrivalRecord | None,
) -> list[str]:
    """Close open decisions against the execution the artifacts now show.

    This is what makes a decision a commitment rather than a query. The
    binding is derived, not reported: a role session that forgets to call
    `record` — or dies before it can — would otherwise leave every decision
    permanently open, and an open decision is what suppresses re-deciding.
    Reconciliation already knows exactly which bouts and episodes newly
    appeared, so it is the only place that can say "the decision that chose
    TUNE(017) is the one this new bout on 017 executed".

    Only the most recent open decision is closed per call: decisions are
    issued one per round and a round runs one action, so more than one open
    decision means the loop skipped a binding, and guessing which artifact
    belongs to which stale decision would fabricate provenance.
    """
    from .store import SchedulerStore

    store = SchedulerStore(run_dir)
    pending = store.unbound_decisions()
    if not pending:
        return []
    decision = pending[-1]
    action = decision.get("selected_action")
    decision_id = decision.get("decision_id")

    if action == "TUNE":
        run_id = decision.get("selected_run_id")
        match = next((r for r in bouts if r.run_id == run_id), None)
        if match is None:
            return []
        store.record_outcome(
            decision_id,
            executed_action="TUNE",
            executed_run_id=run_id,
            realized_gain=match.gain,
            consumed=match.cost,
            status=match.status,
        )
        return [decision_id]

    if action == "DEFER" and episode is not None:
        store.record_outcome(
            decision_id,
            executed_action="DEFER",
            executed_run_id=None,
            realized_gain=None,
            consumed=len(episode.warm_gaps) * max(1, episode.per_candidate_cost),
            status=(
                VALID if episode.usable_gaps else INFRA_FAILURE
            ),
        )
        return [decision_id]
    return []


def reconcile_path(
    ledger_path: Path,
    *,
    k_eval: int = DEFAULT_K_EVAL,
) -> dict:
    ledger_path = Path(ledger_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    return reconcile(ledger_path.parent, ledger, k_eval=k_eval)


__all__ = ["reconcile", "reconcile_path"]
