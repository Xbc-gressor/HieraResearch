"""Command-line entrypoint for the deterministic HieraResearch coordinator."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

from .artifacts import InvocationJournal
from .coordinator import ExperimentCoordinator, RunControls
from .llm import (
    AnthropicMessagesBackend,
    ClaudeAgentBackend,
    ModelGateway,
    RecordedBackend,
)
from .models import RunIdentity
from .process import ProcessRunner
from .toolchain import Toolchain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name")
    parser.add_argument("tag")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="HieraResearch repository root (default: current directory)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("HIERARESEARCH_MODEL"),
        help="Claude model id (or set HIERARESEARCH_MODEL)",
    )
    parser.add_argument(
        "--recordings",
        type=Path,
        help="replay purpose-keyed model outputs from this directory",
    )
    parser.add_argument(
        "--dimension-strategy",
        choices=["catalog_subset", "llm_induced"],
    )
    parser.add_argument("--llm-intelligence-score", type=float)
    parser.add_argument("--max-evaluations", type=int)
    parser.add_argument("--timeout", dest="per_runtime_limit", type=float)
    parser.add_argument("--worker-timeout", type=float, default=21600.0)
    parser.add_argument("--helper-timeout", type=float, default=120.0)
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "initialize the run and validate its task environment, then stop "
            "before background research, model calls, or objective evaluation"
        ),
    )
    parser.add_argument(
        "--resume-blocked",
        action="store_true",
        help="explicitly reopen a previously blocked coordinator/ledger state",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.model and not args.recordings and not args.preflight_only:
        parser.error("--model or HIERARESEARCH_MODEL is required")
    repo_root = args.repo_root.resolve()
    if not (repo_root / "tools/ledger.py").is_file() or not (repo_root / "tasks").is_dir():
        parser.error(f"not a HieraResearch repository root: {repo_root}")
    if args.worker_timeout <= 0 or args.helper_timeout <= 0:
        parser.error("process timeouts must be positive")
    if args.recordings and not args.recordings.is_dir():
        parser.error(f"recording directory does not exist: {args.recordings}")

    identity = RunIdentity(repo_root, args.task_name, args.tag)
    runner = ProcessRunner()
    toolchain = Toolchain(
        repo_root,
        runner,
        helper_timeout=args.helper_timeout,
        worker_timeout=args.worker_timeout,
    )
    if args.recordings:
        replay = RecordedBackend(args.recordings)
        structured_backend = replay
        edit_backend = replay
    else:
        structured_backend = AnthropicMessagesBackend()
        edit_backend = ClaudeAgentBackend()
    gateway = ModelGateway(
        model=args.model or ("recorded" if args.recordings else "preflight-only"),
        journal=InvocationJournal(identity.run_dir),
        structured_backend=structured_backend,
        edit_backend=edit_backend,
    )
    coordinator = ExperimentCoordinator(
        identity,
        toolchain,
        gateway,
        RunControls(
            dimension_strategy=args.dimension_strategy,
            llm_intelligence_score=args.llm_intelligence_score,
            max_evaluations=args.max_evaluations,
            per_runtime_limit=args.per_runtime_limit,
            sync_environment=not args.no_sync,
            prepare_task=not args.no_prepare,
            preflight_only=args.preflight_only,
            resume_blocked=args.resume_blocked,
        ),
    )

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_on_sigterm(signum: int, frame: object) -> None:
        del signum, frame
        raise KeyboardInterrupt("received SIGTERM")

    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    try:
        status = coordinator.run(max_transitions=args.max_transitions)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 2 if status.get("phase") == "blocked" else 0
