"""CLI: uv run python -m driver run <task> <tag> --loop experiment|hillclimb ..."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="driver")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one experiment or hillclimb loop")
    run.add_argument("task")
    run.add_argument("tag")
    run.add_argument("--loop", choices=["experiment", "hillclimb"], required=True)
    run.add_argument("--model", help="resolved model id; required for a new run, "
                                     "ignored on resume (run_metadata.json wins)")
    run.add_argument("--max-evaluations", type=int)
    run.add_argument("--timeout", type=float,
                     help="per-evaluation limit in seconds; pass-through alias for "
                          "init_run.py --per-runtime-limit, NOT a session watchdog")
    run.add_argument("--dimension-strategy", choices=["catalog_subset", "llm_induced"])
    run.add_argument("--llm-intelligence-score", type=float)
    run.add_argument("--cli-path", help="system claude CLI path; default is the "
                                        "SDK-bundled CLI (pinned via uv.lock)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        import json
        from pathlib import Path

        from driver.loops.hillclimb import run_hillclimb
        from driver.roles import REPO_ROOT
        from driver.session import SDKSessionRunner
        from driver.events import EventsLog

        run_dir = REPO_ROOT / "runs" / args.task / args.tag
        model = args.model
        metadata_path = run_dir / "run_metadata.json"
        if model is None and metadata_path.exists():
            model = json.loads(metadata_path.read_text(encoding="utf-8"))["model"]
        if model is None:
            print("error: --model is required for a new run", file=sys.stderr)
            return 2
        if args.loop == "hillclimb":
            runner = SDKSessionRunner(model=model, events=EventsLog(run_dir),
                                      cli_path=args.cli_path)
            status = run_hillclimb(
                args.task, args.tag, runner=runner, model=model,
                max_evaluations=args.max_evaluations, timeout=args.timeout,
                cli_path=args.cli_path,
            )
            print(json.dumps(status, indent=2, sort_keys=True))
            return 0
        from driver.loops.experiment import run_experiment
        runner = SDKSessionRunner(model=model, events=EventsLog(run_dir),
                                  cli_path=args.cli_path)
        status = run_experiment(
            args.task, args.tag, runner=runner, model=model,
            max_evaluations=args.max_evaluations, timeout=args.timeout,
            dimension_strategy=args.dimension_strategy,
            llm_intelligence_score=args.llm_intelligence_score,
            cli_path=args.cli_path,
        )
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
