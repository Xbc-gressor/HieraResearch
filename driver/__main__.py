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
        if not args.model:
            # New runs must pin a concrete model. Resumed runs reuse
            # run_metadata.json; the loops resolve this in Tasks 6-7.
            print("error: --model is required for a new run", file=sys.stderr)
            return 2
        print("error: loops not implemented yet", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
