"""CLI: uv run python -m driver run <task> <tag> --loop experiment|hillclimb ..."""

from __future__ import annotations

import argparse
import sys


SEMANTIC_POLICIES = (
    "coverage",
    "coverage_experience",
    "coverage_attempt",
    "coverage_carrier_attempt",
    "gain",
    "gain_uncertainty",
    "gain_uncertainty_nocost",
)


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
    run.add_argument(
        "--semantic-policy",
        choices=SEMANTIC_POLICIES,
        help="semantic acquisition policy; new runs default to coverage_attempt",
    )
    run.add_argument(
        "--scheduler-policy",
        choices=["legacy", "legacy_wide", "v3_2", "anchor_challenger_v1"],
        help="tuner scheduler policy; new runs default to v3_2",
    )
    run.add_argument(
        "--inner-tuner-policy",
        choices=[
            "deferred-random8-hebo10-spsa10-v1",
            "localtr8-hebo10-spsa10-v1",
            "localtr8-hebo10-hebo10-v1",
            "selfrank8-hebo10-hebo10",
            "mixup24-turbo20-v1",
            "hebo24-turbo20-v1",
            "legacy",
        ],
        help="regime-conditioned inner-tuner policy; new runs default to "
             "deferred-random8-hebo10-spsa10-v1",
    )
    run.add_argument(
        "--k-warm",
        type=int,
        help="how many warm configs are proposed per candidate at step 0+1 "
             "(minimum 2; template default 5). The K - K_eval leftovers are "
             "deferred to the promoted candidate's first tuning bout. "
             "Frozen once run artifacts exist",
    )
    run.add_argument(
        "--k-eval",
        type=int,
        help="how many proposed warm configs are evaluated at step 0+1 "
             "(minimum 2; template default 2). Frozen once run artifacts exist",
    )
    run.add_argument("--cli-path", help="system claude CLI path; default is the "
                                        "SDK-bundled CLI (pinned via uv.lock)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        import json

        from driver.loops.hillclimb import run_hillclimb
        from driver.metadata import resolve_model
        from driver.roles import REPO_ROOT
        from driver.session import SDKSessionRunner
        from driver.status import exit_code_for
        from driver.events import EventsLog

        run_dir = REPO_ROOT / "runs" / args.task / args.tag
        model, warning = resolve_model(args.model, run_dir)
        if warning:
            print(f"warning: {warning}", file=sys.stderr)
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
            return exit_code_for(status, args.loop)
        from driver.loops.experiment import run_experiment
        runner = SDKSessionRunner(model=model, events=EventsLog(run_dir),
                                  cli_path=args.cli_path)
        status = run_experiment(
            args.task, args.tag, runner=runner, model=model,
            max_evaluations=args.max_evaluations, timeout=args.timeout,
            dimension_strategy=args.dimension_strategy,
            llm_intelligence_score=args.llm_intelligence_score,
            semantic_policy=args.semantic_policy,
            scheduler_policy=args.scheduler_policy,
            inner_policy=args.inner_tuner_policy,
            k_warm=args.k_warm,
            k_eval=args.k_eval,
            cli_path=args.cli_path,
        )
        print(json.dumps(status, indent=2, sort_keys=True))
        return exit_code_for(status, args.loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
