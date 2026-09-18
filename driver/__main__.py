"""CLI: uv run python -m driver run <task> <tag> --loop experiment|hillclimb ..."""

from __future__ import annotations

import argparse
import os
import sys


SEMANTIC_POLICIES = (
    "coverage",
    "coverage_experience",
    "coverage_attempt",
    "coverage_carrier_attempt",
    "gain",
    "gain_uncertainty",
    "gain_uncertainty_nocost",
    "judged_slate",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="driver")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one experiment or hillclimb loop")
    run.add_argument("task")
    run.add_argument("tag")
    run.add_argument("--loop", choices=["experiment", "hillclimb",
                                         "baseline-tune", "rewrite"],
                     required=True)
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
        help="semantic point policy; new runs default to judged_slate (the "
             "judged-slate generation); the other values are acquisition-"
             "policy comparison arms",
    )
    run.add_argument(
        "--scheduler-policy",
        choices=["legacy", "legacy_wide", "v3_2", "anchor_challenger_v1",
                 "anchor_transfer_challenger_v1", "round_v1"],
        help="tuner scheduler policy; new experiment runs default to "
             "round_v1 (candidate-count-triggered rewrite+tune rounds under a "
             "wall-clock budget). Ignored by --loop baseline-tune, which runs "
             "without a scheduler",
    )
    run.add_argument(
        "--time-budget", type=float, metavar="SECONDS",
        help="run-level wall-clock budget from now (persisted as an absolute "
             "deadline); the round_v1 default budget unit",
    )
    run.add_argument("--deadline", type=float, metavar="EPOCH_SECONDS",
                     help="absolute run deadline instead of --time-budget")
    run.add_argument("--final-reserve", type=float, metavar="SECONDS",
                     help="tail of the time budget kept free for finalization")
    run.add_argument("--round-new-candidates", type=int, metavar="N",
                     help="round_v1: new finite candidates per optimization round")
    run.add_argument("--round-rewrite-bouts", type=int, metavar="R",
                     help="round_v1: rewrite climbs per optimization round "
                          "(one climb hillclimbs one candidate until it "
                          "stalls, caps out, or the round quota ends)")
    run.add_argument("--round-tune-bouts", type=int, metavar="T",
                     help="round_v1: tune bouts per optimization round")
    run.add_argument("--round-seconds", type=float, metavar="Q",
                     help="round_v1: wall-clock quota of one optimization round")
    run.add_argument("--round-rewrite-top-k", type=int, metavar="K",
                     help="round_v1: rewrite eligibility is the top-K by score")
    run.add_argument("--session-concurrency", type=int, metavar="S",
                     help="experiment loop: seats of one generation implemented "
                          "at once on the session channel (template default 2; "
                          "1 = serial)")
    run.add_argument("--rewrite-concurrency", type=int, metavar="R",
                     help="round_v1: rewrite climbs of one optimization round "
                          "run at once over different candidates (template "
                          "default 2; 1 = serial; needs --round-rewrite-bouts "
                          ">= R to matter)")
    run.add_argument(
        "--inner-tuner-policy",
        choices=[
            "deferred-random8-hebo10-spsa10-v1",
            "localtr8-hebo10-spsa10-v1",
            "localtr8-hebo10-hebo10-v1",
            "selfrank8-hebo10-hebo10",
            "mixup24-turbo20-v1",
            "hebo24-turbo20-v1",
            "hebo24-hebo20",
            "hebo24-transfer10-hebo10",
            "legacy",
        ],
        help="regime-conditioned inner-tuner policy; new experiment runs "
             "default to hebo24-hebo20. Ignored by --loop "
             "baseline-tune, which freezes baseline-hebo-full-v1",
    )
    run.add_argument(
        "--proposer-arm",
        choices=["pool_hebo_mace", "explicit_e3u2"],
        help="inner-benchmark arm that proposes and ranks configs inside "
             "HEBO bouts; new runs default to explicit_e3u2. Pairs with any "
             "inner tuner policy",
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
             "(minimum 2; template default 3). Frozen once run artifacts exist",
    )
    run.add_argument(
        "--noise-margin",
        type=float,
        default=None,
        help="rewrite bouts (rewrite loop and round_v1): a score must beat "
             "the candidate's reference by more than this margin to be kept "
             "(default 0 = strict improvement)",
    )
    run.add_argument(
        "--max-bouts",
        type=int,
        default=12,
        help="rewrite loop: per-candidate bout cap",
    )
    run.add_argument(
        "--stall-after",
        type=int,
        default=5,
        help="rewrite loop: a candidate stalls after this many consecutive "
             "non-kept bouts",
    )
    run.add_argument(
        "--context",
        default="full",
        help="rewrite loop: context.md section subset passed through to "
             "rewrite_context.py --sections (default 'full' = all sections)",
    )
    run.add_argument("--cli-path", help="system claude CLI path; default is the "
                                        "SDK-bundled CLI (pinned via uv.lock)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.noise_margin is not None and args.noise_margin < 0:
            parser.error("--noise-margin must be >= 0")
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
        if args.loop == "baseline-tune":
            from driver.loops.baseline_tune import run_baseline_tune
            runner = SDKSessionRunner(model=model, events=EventsLog(run_dir),
                                      cli_path=args.cli_path)
            status = run_baseline_tune(
                args.task, args.tag, runner=runner, model=model,
                max_evaluations=args.max_evaluations, timeout=args.timeout,
                k_warm=args.k_warm, k_eval=args.k_eval,
                proposer_arm=args.proposer_arm,
                cli_path=args.cli_path,
            )
            print(json.dumps(status, indent=2, sort_keys=True))
            return exit_code_for(status, args.loop)
        if args.loop == "rewrite":
            from driver.loops.rewrite import run_rewrite
            runner = SDKSessionRunner(model=model, events=EventsLog(run_dir),
                                      cli_path=args.cli_path)
            status = run_rewrite(
                args.task, args.tag, runner=runner, model=model,
                noise_margin=args.noise_margin or 0.0, max_bouts=args.max_bouts,
                stall_after=args.stall_after, context=args.context,
                max_evaluations=args.max_evaluations, timeout=args.timeout,
                cli_path=args.cli_path,
            )
            print(json.dumps(status, indent=2, sort_keys=True))
            return exit_code_for(status, args.loop)
        from driver.loops.experiment import run_experiment
        from driver.loops.rounds import DEFAULT_EVALUATION_DOMAIN
        # Selection consumes one evaluation domain: the task-native records
        # every production objective path indexes (legacy adapter -> stage
        # "proxy", fidelity "fast"). An operator override stays in force.
        os.environ.setdefault("EVALUATION_STAGE", DEFAULT_EVALUATION_DOMAIN[0])
        os.environ.setdefault("EVALUATION_FIDELITY", DEFAULT_EVALUATION_DOMAIN[1])
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
            proposer_arm=args.proposer_arm,
            time_budget=args.time_budget,
            deadline=args.deadline,
            final_reserve=args.final_reserve,
            round_options={
                "new_candidates": args.round_new_candidates,
                "rewrite_bouts": args.round_rewrite_bouts,
                "tune_bouts": args.round_tune_bouts,
                "round_seconds": args.round_seconds,
                "noise_margin": args.noise_margin,
                "rewrite_top_k": args.round_rewrite_top_k,
            },
            session_concurrency=args.session_concurrency,
            rewrite_concurrency=args.rewrite_concurrency,
            cli_path=args.cli_path,
        )
        print(json.dumps(status, indent=2, sort_keys=True))
        return exit_code_for(status, args.loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
