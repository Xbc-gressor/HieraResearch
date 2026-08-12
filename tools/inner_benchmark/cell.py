"""Cell wiring for the inner-tuner benchmark (PLAN §5.3/§八).

Connects one (checkpoint, arm, seed) cell to the real infrastructure: the
arm from the registry, the runner's objective path, and the bout session
factory backed by the SDK session layer. One process = one cell; the §八
sweep distributes cells across machines externally (one GPU runs one cell
at a time — enforced by the operator, not here).

    uv run python tools/inner_benchmark/cell.py \
        --arm pool_tpe --checkpoint /path/to/ckpt --seed 7 \
        --out <cell-out-dir> --model <pinned-model-id> [--machine a800-3]

``--model`` is required (§5.3: one pinned model id recorded in every
manifest; production likewise takes an explicit ``--model`` and exposes no
decoding knobs, so LLMConfig records ``decoding`` as null).

The bout-session run_dir is ``<out>/llm`` — exactly one cell per run_dir
(the invariant of llm.make_bout_session_factory: receipt invocation ids
are per-directory, so two cells sharing one could cross-resume
transcripts). It gathers the driver-layer artifacts (``receipts/``,
``driver_events.jsonl``) next to the runner's own manifest/events/result.

Editor confinement: ``bench-hillclimb-editor`` is the only bench role
with file tools. ``BenchSessionRunner`` appends a PreToolUse hook
confining its Read/Edit to the invocation's working copy (the absolute
path the arm puts in the ``working_copy`` extra of every editor ask), so
a frozen checkpoint — shared read-only by every cell of every arm —
stays unreachable from an editor session even under prompt-level
misbehavior. Fail-closed when the confinement path is missing. This is
minimal containment, not a sandbox — the same stance the production
capability hook documents for bash_patterns.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arms  # noqa: E402
import checkpoint as checkpoint_mod  # noqa: E402
import llm  # noqa: E402
import runner  # noqa: E402
from driver.events import EventsLog  # noqa: E402
from driver.session import SDKSessionRunner  # noqa: E402

# Driver-layer artifacts live under <out>/llm (see module docstring).
LLM_DIRNAME = "llm"
EDITOR_ROLE = "bench-hillclimb-editor"


def _working_copy_hook(allowed_path):
    """PreToolUse hook confining the editor role's Read/Edit to one file.

    ``allowed_path`` is the working copy's absolute path from the
    invocation's ``working_copy`` extra; None fails closed (a miswired
    invocation is not entitled to any file access).
    """
    allowed = None if allowed_path is None else Path(allowed_path).resolve()

    def deny(reason: str) -> dict:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    async def hook(input_data, tool_use_id, context):
        name = input_data.get("tool_name", "")
        if name not in ("Read", "Edit"):
            # Not a file tool: the base capability hook owns allow/deny.
            return {}
        if allowed is None:
            return deny(
                "editor invocation carries no working_copy path; Read/Edit "
                "stay fail-closed"
            )
        raw = (input_data.get("tool_input") or {}).get("file_path", "")
        try:
            resolved = Path(raw).resolve()
        except (OSError, RuntimeError, ValueError):
            return deny(f"unresolvable file_path {raw!r}")
        if resolved != allowed:
            return deny(
                f"{EDITOR_ROLE} may only Read/Edit its working copy "
                f"{allowed}; got {resolved}"
            )
        return {}

    return hook


class BenchSessionRunner(SDKSessionRunner):
    """SDKSessionRunner plus the benchmark's editor working-copy confinement
    (see module docstring). Every editor invocation carries the path in its
    extras, so the hook is built per invocation with the exact file."""

    def _build_options(self, role, ctx, server):
        options = super()._build_options(role, ctx, server)
        if role.name == EDITOR_ROLE:
            from claude_agent_sdk import HookMatcher

            allowed = (ctx.extra or {}).get(llm.WORKING_COPY_KEY)
            options.hooks["PreToolUse"].append(
                HookMatcher(matcher=None, hooks=[_working_copy_hook(allowed)])
            )
        return options


def _machine_record(machine: str | None) -> dict:
    """Manifest machine/GPU record (PLAN §八: 排除误配). Best-effort."""
    record = {
        "label": machine,
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        probe = None
    if probe is not None and probe.returncode == 0:
        record["gpus"] = [
            line.strip() for line in probe.stdout.splitlines() if line.strip()
        ]
    return record


def execute_cell(
    *,
    arm_name: str,
    checkpoint_dir,
    out_dir,
    seed: int,
    model: str,
    budget: int,
    machine: str | None = None,
    cli_path: str | None = None,
    session_runner=None,
) -> dict:
    """Wire one cell end to end and run it; return the result.json dict.

    ``session_runner`` is the sanctioned test seam (e.g. the driver's
    FakeSessionRunner); real cells leave it None and get the SDK-backed
    BenchSessionRunner.
    """
    arm = arms.load_arm(arm_name)
    checkpoint_dir = Path(checkpoint_dir)
    out_dir = Path(out_dir)
    frozen = checkpoint_mod.load_checkpoint(checkpoint_dir)
    project = frozen.task.project or ""
    task_name = project.rsplit("/", 1)[-1] if project else "unknown"
    tag = f"{frozen.checkpoint_id}--{arm_name}--s{seed}"

    run_dir = out_dir / LLM_DIRNAME
    events = EventsLog(run_dir)
    if session_runner is None:
        session_runner = BenchSessionRunner(
            model=model, events=events, cli_path=cli_path
        )
    factory = llm.make_bout_session_factory(
        runner=session_runner, run_dir=run_dir, task=task_name, tag=tag
    )
    return runner.run_cell(
        arm=arm,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        seed=seed,
        budget=budget,
        model=llm.LLMConfig(model=model).to_manifest(),
        manifest_extra={"machine": _machine_record(machine)},
        extras={"session_factory": factory},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inner_benchmark.cell")
    parser.add_argument("--arm", required=True, choices=sorted(arms.ARM_MODULES))
    parser.add_argument("--checkpoint", required=True, help="frozen checkpoint dir")
    parser.add_argument("--out", required=True, help="cell output dir (created)")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help=f"objective-evaluation budget (default B={runner.B})",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="pinned model id for every LLM call in the cell (PLAN §5.3)",
    )
    parser.add_argument(
        "--machine", default=None, help="operator machine label (manifest only)"
    )
    parser.add_argument(
        "--cli-path",
        default=None,
        help="system claude CLI path; default is the SDK-bundled CLI",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = execute_cell(
        arm_name=args.arm,
        checkpoint_dir=args.checkpoint,
        out_dir=args.out,
        seed=args.seed,
        model=args.model,
        budget=args.budget if args.budget is not None else runner.B,
        machine=args.machine,
        cli_path=args.cli_path,
    )
    # A completed cell — including arm_error/unsupported, which are recorded
    # experiment outcomes with a full result.json — exits 0. Setup failures
    # (bad args, missing checkpoint, unmeasurable remeasurement) raise above.
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
