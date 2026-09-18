#!/usr/bin/env python3
"""Run-global scheduler CLI: decision, replay, and calibration views.

Decision/replay dispatch to the run's persisted policy: current
``anchor_challenger_v1`` or the historical ``v3_2`` comparison.

    python tools/scheduler/cli.py decide  --ledger <run_dir>/ledger.json
    python tools/scheduler/cli.py record  --ledger ... --decision-id ... \
                                          --status valid --gain 0.012
    python tools/scheduler/cli.py replay  --ledger ... --decision-id ...
    python tools/scheduler/cli.py evidence --ledger ...
    python tools/scheduler/cli.py round status|begin|end|select|overhead ...

`decide` is the tune command the orchestrator's `select-candidate` calls;
the `round` family is what the driver's round_v1 loop calls. It prints
``{action, run_id, reason, decision_id, ...}`` where `action` is `TUNE`,
`DEFER`, or `STOP`, and `run_id` names the exact candidate for `TUNE` —
the execution layer never re-selects.

Stdlib only, like `select-candidate`, so the loop can call it without a uv
environment.
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
    from tools.scheduler import evidence as evidence_mod  # noqa: E402
    from tools.scheduler import round_policy  # noqa: E402
    from tools.scheduler import tournament  # noqa: E402
    from tools.scheduler import transfer_tournament  # noqa: E402
    from tools.scheduler.policy import (  # noqa: E402
        PolicyConfig,
        decide as decide_policy,
    )
    from tools.scheduler.rollout import RolloutConfig  # noqa: E402
    from tools.scheduler.session import (  # noqa: E402
        decide_for_run,
        models_for,
        peek_tune_for_run,
        select_rewrite_for_run,
    )
    from tools.scheduler.state import state_from_snapshot  # noqa: E402
    from tools.scheduler.store import SchedulerStore  # noqa: E402
else:  # pragma: no cover - imported as a package
    from . import evidence as evidence_mod
    from . import round_policy
    from . import tournament
    from . import transfer_tournament
    from .policy import PolicyConfig, decide as decide_policy
    from .rollout import RolloutConfig
    from .session import decide_for_run, models_for, peek_tune_for_run, select_rewrite_for_run
    from .state import state_from_snapshot
    from .store import SchedulerStore


def cmd_decide(args) -> int:
    view = decide_for_run(Path(args.ledger), scenarios=args.scenarios)
    view.pop("state", None)
    print(json.dumps(view, indent=2))
    return 0


def cmd_record(args) -> int:
    """Bind a decision to what the execution layer actually did.

    Normally unnecessary: `reconcile` derives the binding from the run's own
    artifacts at decide time, so a forgotten call cannot starve the models
    or leave a decision open forever. This stays as the manual override for
    an execution the artifacts cannot show — a decision abandoned without a
    bout, or a shadow/replay harness closing a decision by hand.
    """
    run_dir = Path(args.ledger).parent
    store = SchedulerStore(run_dir)
    store.record_outcome(
        args.decision_id,
        executed_action=args.action,
        executed_run_id=args.run_id,
        realized_gain=(
            float(args.gain)
            if args.action != "DEFER" and args.gain is not None
            else None
        ),
        consumed=int(args.consumed),
        status=args.status,
    )
    print(json.dumps({"ok": True, "decision_id": args.decision_id}))
    return 0


def cmd_replay(args) -> int:
    """Re-decide from a stored snapshot and evidence cursor (§9).

    This proves the decision is reproducible — same snapshot, same cursor,
    same policy version, same action. It cannot show what an alternative
    policy would have achieved; only fresh E2E arms can.
    """
    run_dir = Path(args.ledger).parent
    store = SchedulerStore(run_dir)
    receipt = next(
        (
            row
            for row in store.decisions()
            if row.get("decision_id") == args.decision_id
            and row.get("kind") == "scheduler_decision"
        ),
        None,
    )
    if receipt is None:
        raise SystemExit(f"no decision {args.decision_id}")

    state = state_from_snapshot(store.get_snapshot(receipt["state_snapshot_id"]))
    # Replay must use the policy that produced the receipt: the deterministic
    # tournaments are pure functions of the snapshot, while v3.2 also needs
    # its models.
    if receipt.get("policy_version") == tournament.POLICY_VERSION:
        replayed = tournament.decide(state)
    elif receipt.get("policy_version") == transfer_tournament.POLICY_VERSION:
        replayed = transfer_tournament.decide(state)
    elif receipt.get("policy_version") == round_policy.POLICY_VERSION:
        # Round decisions also read the live phase state and eval timings,
        # so replay checks the recorded action kind against the snapshot only.
        replayed = (
            round_policy.select_rewrite(state, run_dir)
            if receipt["selected_action"] == round_policy.REWRITE
            else round_policy.decide(state, run_dir)
        )
    else:
        tuning, arrival = models_for(store, cursor=receipt.get("evidence_cursor"))
        config = PolicyConfig(
            rollout=RolloutConfig(scenarios=int(receipt.get("paired_scenarios", 64)))
        )
        replayed = decide_policy(
            state,
            tuning,
            arrival,
            config=config,
            coverage_spent=int(receipt.get("coverage_spent", 0)) - (
                1 if str(receipt.get("reason", "")).startswith("coverage:") else 0
            ),
        )
    matches = (
        replayed.action == receipt["selected_action"]
        and replayed.run_id == receipt["selected_run_id"]
    )
    print(
        json.dumps(
            {
                "decision_id": args.decision_id,
                "reproduced": matches,
                "recorded": {
                    "action": receipt["selected_action"],
                    "run_id": receipt["selected_run_id"],
                },
                "replayed": {"action": replayed.action, "run_id": replayed.run_id},
                "policy_version": receipt.get("policy_version"),
            },
            indent=2,
        )
    )
    return 0 if matches else 1


def cmd_evidence(args) -> int:
    """Show the current-run empirical models and their support."""
    store = SchedulerStore(Path(args.ledger).parent)
    tuning, arrival = models_for(store)
    print(
        json.dumps(
            {
                "scope": evidence_mod.EVIDENCE_SCOPE,
                "prior_id": evidence_mod.PRIOR_ID,
                "tuning": tuning.summary(),
                "arrival": arrival.summary(),
                "coverage_spent": store.coverage_spent(),
                "decisions": len(
                    [
                        row
                        for row in store.decisions()
                        if row.get("kind") == "scheduler_decision"
                    ]
                ),
            },
            indent=2,
        )
    )
    return 0


def _ledger_doc(ledger: Path) -> dict:
    if Path(ledger).is_file():
        doc = json.loads(Path(ledger).read_text(encoding="utf-8"))
        try:
            from .state import materialize_evaluation_records
        except ImportError:
            from scheduler.state import materialize_evaluation_records
        return materialize_evaluation_records(doc)
    return {"records": []}


def cmd_round(args) -> int:
    """round_v1 driver entry points; every subcommand prints one JSON view."""
    ledger = Path(args.ledger)
    run_dir = ledger.parent
    stage = getattr(args, "stage", None)
    fidelity = getattr(args, "fidelity", None)
    if (stage is None) != (fidelity is None):
        raise SystemExit("--stage and --fidelity must be supplied together")
    if stage == "official":
        raise SystemExit("official evaluations are reporting-only and cannot drive round scheduling")
    if args.round_command == "status":
        view = round_policy.round_status(
            run_dir, _ledger_doc(ledger), stage=stage, fidelity=fidelity
        )
    elif args.round_command == "begin":
        view = round_policy.begin_optimization(run_dir)
    elif args.round_command == "end":
        view = round_policy.end_optimization(
            run_dir, _ledger_doc(ledger), stage=stage, fidelity=fidelity
        )
    elif args.round_command == "overhead":
        view = round_policy.record_overhead(run_dir, args.kind, args.seconds)
    elif args.round_command == "select":
        if args.peek and args.kind == "tune":
            view = peek_tune_for_run(ledger)
        elif args.kind == "rewrite":
            exclude = [r for r in (args.exclude or "").split(",") if r]
            view = select_rewrite_for_run(ledger, exclude)
        else:
            view = decide_for_run(ledger)
        view.pop("state", None)
        view["reference"] = (view.get("evidence_mode") or {}).get("reference")
    else:  # pragma: no cover - argparse restricts the choices
        raise AssertionError(args.round_command)
    print(json.dumps(view, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dec = sub.add_parser("decide", help="Choose TUNE(i) or DEFER for this round.")
    dec.add_argument("--ledger", required=True, type=Path)
    dec.add_argument("--scenarios", type=int, default=None)
    dec.set_defaults(func=cmd_decide)

    rec = sub.add_parser("record", help="Bind a decision to what was executed.")
    rec.add_argument("--ledger", required=True, type=Path)
    rec.add_argument("--decision-id", required=True)
    rec.add_argument("--action", required=True,
                     choices=("TUNE", "DEFER", round_policy.REWRITE))
    rec.add_argument("--run-id", default=None)
    rec.add_argument("--gain", type=float, default=None)
    rec.add_argument("--consumed", type=int, required=True)
    rec.add_argument(
        "--status",
        default=evidence_mod.VALID,
        choices=(
            evidence_mod.VALID,
            evidence_mod.SCIENTIFIC_INVALID,
            evidence_mod.INFRA_FAILURE,
        ),
    )
    rec.set_defaults(func=cmd_record)

    rep = sub.add_parser("replay", help="Re-decide from a stored snapshot.")
    rep.add_argument("--ledger", required=True, type=Path)
    rep.add_argument("--decision-id", required=True)
    rep.set_defaults(func=cmd_replay)

    ev = sub.add_parser("evidence", help="Show current-run empirical models.")
    ev.add_argument("--ledger", required=True, type=Path)
    ev.set_defaults(func=cmd_evidence)

    rnd = sub.add_parser("round", help="round_v1 phase state and selections.")
    rnd.add_argument("--ledger", required=True, type=Path)
    rnd_sub = rnd.add_subparsers(dest="round_command", required=True)
    status = rnd_sub.add_parser("status", help="generate-or-optimize switch")
    status.add_argument("--stage", choices=("proxy", "protocol", "official"))
    status.add_argument("--fidelity")
    rnd_sub.add_parser("begin", help="open an optimization phase (sets the quota)")
    end = rnd_sub.add_parser("end", help="close the phase; next cycle counts from now")
    end.add_argument("--stage", choices=("proxy", "protocol", "official"))
    end.add_argument("--fidelity")
    sel = rnd_sub.add_parser("select", help="choose the next rewrite/tune target")
    sel.add_argument("--kind", required=True, choices=("rewrite", "tune"))
    sel.add_argument("--peek", action="store_true")
    sel.add_argument("--exclude", default="",
                     help="comma-separated run_ids another rewrite channel is "
                          "climbing; ineligible and part of the decision identity")
    ovh = rnd_sub.add_parser("overhead", help="record one bout's non-eval seconds")
    ovh.add_argument("--kind", required=True, choices=("rewrite", "tune"))
    ovh.add_argument("--seconds", required=True, type=float)
    rnd.set_defaults(func=cmd_round)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
