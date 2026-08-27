#!/usr/bin/env python3
"""Run-global scheduler CLI: decision, replay, and calibration views.

Decision/replay dispatch to the run's persisted policy: current
``anchor_challenger_v1`` or the historical ``v3_2`` comparison.

    python tools/scheduler/cli.py decide  --ledger <run_dir>/ledger.json
    python tools/scheduler/cli.py record  --ledger ... --decision-id ... \
                                          --status valid --gain 0.012
    python tools/scheduler/cli.py replay  --ledger ... --decision-id ...
    python tools/scheduler/cli.py evidence --ledger ...

`decide` is the only command the loop calls. It prints
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
    )
    from tools.scheduler.state import state_from_snapshot  # noqa: E402
    from tools.scheduler.store import SchedulerStore  # noqa: E402
else:  # pragma: no cover - imported as a package
    from . import evidence as evidence_mod
    from . import tournament
    from . import transfer_tournament
    from .policy import PolicyConfig, decide as decide_policy
    from .rollout import RolloutConfig
    from .session import decide_for_run, models_for
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
        realized_gain=float(args.gain) if args.action == "TUNE" else None,
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
    rec.add_argument("--action", required=True, choices=("TUNE", "DEFER"))
    rec.add_argument("--run-id", default=None)
    rec.add_argument("--gain", type=float, default=0.0)
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
