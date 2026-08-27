"""Read-only calibration for historical v3.2 scheduler artifacts (design §9).

    python tools/scheduler/calibrate.py --ledger runs/<task>/<tag>/ledger.json

This reads and computes; it never decides, never writes, and never touches
the run. Its job is falsification: to say where the decisions a run actually
took disagree with the artifacts that run produced, so a scheduler
experiment can be discarded on evidence rather than defended on argument.

What it can check, and what it cannot, is the design's validation boundary.
It checks:

* **execution closure** — was every decision executed, and executed on the
  exact candidate it named? An open decision or a mismatched target means
  the exact-target contract leaked.
* **budget accounting** — do the evaluations the decisions charged match the
  attempt log the objective actually reserved?
* **replay** — does re-deciding from the stored snapshot and evidence cursor
  reproduce the recorded action? A divergence means the receipt does not
  determine the decision, and no later comparison of runs means anything.
* **model calibration** — how far the FIRST/LATER predicted mean gain sits
  from the realized one, and whether the arrival gaps drift across episodes.
* **ranking stability** — leave-one-record-out over the tuning evidence: how
  often the champion action changes when one record is dropped. A ranking
  that flips on one record is not evidence of a policy difference.

It cannot say what a different policy would have achieved. A factual
trajectory contains no counterfactual, so scheduler efficacy is settled only
by fresh E2E arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "tuners"))

if __package__ in (None, ""):  # direct script invocation
    sys.path.insert(0, str(TOOLS.parent))
    from tools.scheduler import tournament  # noqa: E402
    from tools.scheduler import transfer_tournament  # noqa: E402
    from tools.scheduler.evidence import FIRST, LATER, TuningModel  # noqa: E402
    from tools.scheduler.policy import PolicyConfig, decide  # noqa: E402
    from tools.scheduler.rollout import RolloutConfig  # noqa: E402
    from tools.scheduler.session import models_for  # noqa: E402
    from tools.scheduler.state import state_from_snapshot  # noqa: E402
    from tools.scheduler.store import SchedulerStore  # noqa: E402
else:  # pragma: no cover - imported as a package
    from . import tournament
    from . import transfer_tournament
    from .evidence import FIRST, LATER, TuningModel
    from .policy import PolicyConfig, decide
    from .rollout import RolloutConfig
    from .session import models_for
    from .state import state_from_snapshot
    from .store import SchedulerStore


def _decisions(store: SchedulerStore) -> list[dict]:
    return [
        row for row in store.decisions() if row.get("kind") == "scheduler_decision"
    ]


def _outcomes(store: SchedulerStore) -> dict[str, dict]:
    return {
        row["decision_id"]: row
        for row in store.decisions()
        if row.get("kind") == "scheduler_outcome"
    }


def execution_closure(store: SchedulerStore) -> dict:
    """Did every committed decision execute, on the candidate it named?"""
    outcomes = _outcomes(store)
    open_ids, mismatched = [], []
    for row in _decisions(store):
        if row.get("selected_action") == "STOP":
            continue
        outcome = outcomes.get(row.get("decision_id"))
        if outcome is None:
            open_ids.append(row.get("decision_id"))
            continue
        if (
            outcome.get("executed_action") != row.get("selected_action")
            or outcome.get("executed_run_id") != row.get("selected_run_id")
        ):
            mismatched.append(
                {
                    "decision_id": row.get("decision_id"),
                    "chosen": [row.get("selected_action"), row.get("selected_run_id")],
                    "executed": [
                        outcome.get("executed_action"),
                        outcome.get("executed_run_id"),
                    ],
                }
            )
    return {
        "decisions": len(_decisions(store)),
        "bound": len(outcomes),
        "unbound_decision_ids": open_ids,
        "target_mismatches": mismatched,
        "closed": not open_ids and not mismatched,
    }


def budget_accounting(run_dir: Path, store: SchedulerStore) -> dict:
    """Do the charged evaluations match what the objective reserved?

    The attempt log is the authority — it is what `reserve_evaluation`
    writes immediately before `score_fn`. Decisions can only *account* for
    the slots they chose to spend; a gap means the run spent evaluations the
    scheduler never decided on (warm evaluation, screening, a legacy path),
    which is expected and reported rather than flagged. An excess is not
    expected: it would mean a decision claimed cost that never happened.
    """
    from evaluation_budget import budget_status

    status = budget_status(Path(run_dir))
    charged = sum(
        int(row.get("consumed_evaluations") or 0) for row in _outcomes(store).values()
    )
    observed = int(status.get("evaluations_done", 0))
    return {
        "attempt_log_evaluations": observed,
        "scheduler_charged": charged,
        "unaccounted": observed - charged,
        "overcharged": charged > observed,
        "budget": status.get("budget"),
        "remaining": status.get("remaining"),
    }


def replay_all(store: SchedulerStore) -> dict:
    """Re-decide every recorded decision from its own snapshot and cursor."""
    diverged = []
    for row in _decisions(store):
        snapshot_id = row.get("state_snapshot_id")
        if not snapshot_id:
            continue
        try:
            state = state_from_snapshot(store.get_snapshot(snapshot_id))
        except (OSError, KeyError, ValueError):
            diverged.append({"decision_id": row.get("decision_id"), "error": "snapshot"})
            continue
        if row.get("policy_version") == tournament.POLICY_VERSION:
            # The tournament is a pure function of the snapshot; replaying it
            # through the v3.2 rollout policy would report false divergences.
            replayed = tournament.decide(state)
        elif row.get("policy_version") == transfer_tournament.POLICY_VERSION:
            replayed = transfer_tournament.decide(state)
        else:
            tuning, arrival = models_for(store, cursor=row.get("evidence_cursor"))
            config = PolicyConfig(
                rollout=RolloutConfig(scenarios=int(row.get("paired_scenarios", 64)))
            )
            # The receipt's `coverage_spent` already includes this decision's
            # own charge, so replaying needs the value as of *before* it.
            spent = int(row.get("coverage_spent", 0)) - (
                1 if str(row.get("reason", "")).startswith("coverage:") else 0
            )
            replayed = decide(
                state, tuning, arrival, config=config, coverage_spent=spent
            )
        if (
            replayed.action != row.get("selected_action")
            or replayed.run_id != row.get("selected_run_id")
        ):
            diverged.append(
                {
                    "decision_id": row.get("decision_id"),
                    "recorded": [row.get("selected_action"), row.get("selected_run_id")],
                    "replayed": [replayed.action, replayed.run_id],
                }
            )
    return {
        "replayed": len(_decisions(store)),
        "divergences": diverged,
        "reproducible": not diverged,
    }


def transition_calibration(store: SchedulerStore) -> dict:
    """Predicted-vs-realized mean gain, per bout class.

    The prediction is the model estimated from every record *before* the
    bout; the realization is that bout's own `D`. Comparing a model against
    the records it was fitted on would report zero error by construction.
    """
    records = store.evidence.tuning_records()
    rows = []
    for index, record in enumerate(records):
        prior = TuningModel.from_records(records[:index])
        if not prior.supported(record.bout_class):
            continue
        support = prior.support(record.bout_class)
        predicted = sum(r.gain for r in support) / len(support)
        rows.append(
            {
                "bout_class": record.bout_class,
                "mode": prior.usage_mode(record.bout_class),
                "predicted": predicted,
                "realized": record.gain,
                "error": record.gain - predicted,
            }
        )

    def summarize(klass: str) -> dict:
        subset = [row for row in rows if row["bout_class"] == klass]
        if not subset:
            return {"n": 0}
        errors = [row["error"] for row in subset]
        return {
            "n": len(subset),
            "mean_error": sum(errors) / len(errors),
            "mean_absolute_error": sum(abs(e) for e in errors) / len(errors),
            "exact_mode_fraction": (
                sum(1 for row in subset if row["mode"] == "exact") / len(subset)
            ),
        }

    return {FIRST: summarize(FIRST), LATER: summarize(LATER), "points": rows}


def arrival_drift(store: SchedulerStore) -> dict:
    """Is the exchangeable arrival marginal drifting across the run?

    Exchangeability is the model's central assumption (§5): episodes are
    resampled whole on the premise that a later episode is as good a draw as
    an earlier one. Comparing the first and second halves' mean gap is a
    coarse falsification of that — not a test, a smell.
    """
    episodes = [e for e in store.evidence.arrival_records() if e.usable_gaps]
    if len(episodes) < 2:
        return {"episodes": len(episodes), "drift": None}
    half = len(episodes) // 2
    def mean_gap(subset):
        gaps = [gap for episode in subset for gap in episode.usable_gaps]
        return sum(gaps) / len(gaps) if gaps else None

    early, late = mean_gap(episodes[:half]), mean_gap(episodes[half:])
    return {
        "episodes": len(episodes),
        "early_mean_gap": early,
        "late_mean_gap": late,
        "drift": None if early is None or late is None else late - early,
        "failed_arrival_fraction": (
            sum(1 for e in episodes for g in e.warm_gaps if g is None)
            / sum(len(e.warm_gaps) for e in episodes)
        ),
    }


def ranking_stability(store: SchedulerStore, *, scenarios: int = 32) -> dict:
    """Leave-one-record-out: how often the champion changes.

    Only the most recent decision is probed. Re-running the rollout for
    every decision times every dropped record is quadratic in a run's
    history and buys nothing: instability is a property of how thin the
    evidence is, and the latest decision has the most of it.
    """
    decisions = _decisions(store)
    if not decisions:
        return {"decision_id": None, "flips": None}
    row = decisions[-1]
    snapshot_id = row.get("state_snapshot_id")
    if not snapshot_id:
        return {"decision_id": row.get("decision_id"), "flips": None}
    state = state_from_snapshot(store.get_snapshot(snapshot_id))
    cursor = row.get("evidence_cursor")
    records = store.evidence.tuning_records(cursor=cursor)
    _, arrival = models_for(store, cursor=cursor)
    config = PolicyConfig(rollout=RolloutConfig(scenarios=scenarios))
    baseline = (row.get("selected_action"), row.get("selected_run_id"))

    flips = 0
    for index in range(len(records)):
        held_out = records[:index] + records[index + 1 :]
        model = TuningModel.from_records(held_out)
        alternative = decide(
            state,
            model,
            arrival,
            config=config,
            coverage_spent=int(row.get("coverage_spent", 0)),
        )
        if (alternative.action, alternative.run_id) != baseline:
            flips += 1
    return {
        "decision_id": row.get("decision_id"),
        "records": len(records),
        "flips": flips,
        "flip_fraction": flips / len(records) if records else None,
    }


def calibrate(ledger_path: Path, *, scenarios: int = 32) -> dict:
    run_dir = Path(ledger_path).parent
    store = SchedulerStore(run_dir)
    return {
        "run_dir": str(run_dir),
        "execution_closure": execution_closure(store),
        "budget_accounting": budget_accounting(run_dir, store),
        "replay": replay_all(store),
        "transition_calibration": transition_calibration(store),
        "arrival_drift": arrival_drift(store),
        "ranking_stability": ranking_stability(store, scenarios=scenarios),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--scenarios", type=int, default=32)
    parser.add_argument(
        "--full-points",
        action="store_true",
        help="Include every per-bout calibration point, not just the summary.",
    )
    args = parser.parse_args()
    report = calibrate(args.ledger, scenarios=args.scenarios)
    if not args.full_points:
        report["transition_calibration"].pop("points", None)
    print(json.dumps(report, indent=2))
    closure = report["execution_closure"]
    return 0 if closure["closed"] and report["replay"]["reproducible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
