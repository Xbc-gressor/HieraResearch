#!/usr/bin/env python3
"""Run MLE-bench final submission and grading with explicit timing semantics.

The submission command is part of the agent deadline.  The grader runs only
after a valid submission has been produced and is measured separately, like
the official evaluator phase.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

try:
    from .mlebench_official import OfficialPreflightError, validate_submission
    from .process_group import (arm_sigterm_forwarding, register_child,
                                terminate_group, unregister_child)
except ImportError:  # direct script execution from the repository checkout
    from mlebench_official import OfficialPreflightError, validate_submission
    from process_group import (arm_sigterm_forwarding, register_child,
                               terminate_group, unregister_child)


def _command(value: str, *, task: str, run_dir: Path, data_dir: Path) -> list[str]:
    rendered = value.format(task=task, run_dir=str(run_dir), data_dir=str(data_dir))
    return shlex.split(rendered)


def _run(command: list[str], cwd: Path, timeout: float) -> int:
    # Export commands commonly wrap uv/python and training subprocesses.
    # Stop the whole command tree at timeout, including its GPU workers.
    arm_sigterm_forwarding()
    proc = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    register_child(proc)
    try:
        return proc.wait(timeout=timeout)
    finally:
        if proc.poll() is None:
            terminate_group(proc, grace=0)
        unregister_child(proc)


def run_submission(*, task: str, run_dir: Path, data_dir: Path | str,
                   submission_command: str, deadline: float) -> tuple[int, dict]:
    """Run and validate one deadline-bound submission command."""
    command = _command(submission_command, task=task, run_dir=run_dir,
                       data_dir=data_dir)
    started = time.time()
    remaining = deadline - started
    if remaining <= 0:
        return 124, {
            "status": "deadline_expired",
            "started_at_unix": started,
            "command": command,
        }
    try:
        rc = _run(command, run_dir, remaining)
    except subprocess.TimeoutExpired:
        ended = time.time()
        return 124, {
            "status": "timeout",
            "started_at_unix": started,
            "ended_at_unix": ended,
            "elapsed_seconds": ended - started,
            "command": command,
        }
    except OSError as exc:
        ended = time.time()
        return 1, {
            "status": "failed",
            "returncode": 1,
            "started_at_unix": started,
            "ended_at_unix": ended,
            "elapsed_seconds": ended - started,
            "command": command,
            "error": str(exc),
        }

    ended = time.time()
    result = {
        "status": "failed",
        "returncode": rc,
        "started_at_unix": started,
        "ended_at_unix": ended,
        "elapsed_seconds": ended - started,
        "command": command,
    }
    if rc == 0:
        try:
            result["validation"] = validate_submission(
                run_dir / "submission.csv")
        except (OfficialPreflightError, OSError) as exc:
            result["error"] = str(exc)
        else:
            result["status"] = "success"
    effective_rc = 0 if result["status"] == "success" else (rc or 1)
    return effective_rc, result


def run_grader(*, task: str, run_dir: Path, data_dir: Path | str,
               grader_command: str, grader_timeout: float = 3600
               ) -> tuple[int, dict]:
    """Run the out-of-budget grader command against run_dir/submission.csv."""
    grader = _command(grader_command, task=task, run_dir=run_dir,
                      data_dir=data_dir)
    started = time.time()
    try:
        rc = _run(grader, run_dir, grader_timeout)
        return rc, {"status": "success" if rc == 0 else "failed",
                    "returncode": rc, "started_at_unix": started,
                    "ended_at_unix": time.time(),
                    "elapsed_seconds": time.time() - started,
                    "command": grader,
                    "counts_toward_submission_budget": False}
    except subprocess.TimeoutExpired:
        return 124, {"status": "timeout", "started_at_unix": started,
                     "ended_at_unix": time.time(), "command": grader,
                     "counts_toward_submission_budget": False}
    except OSError as exc:
        return 1, {"status": "failed", "returncode": 1,
                   "started_at_unix": started, "ended_at_unix": time.time(),
                   "command": grader, "error": str(exc),
                   "counts_toward_submission_budget": False}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--submission-command", required=True,
                   help="command producing {run_dir}/submission.csv; runs in budget")
    p.add_argument("--grader-command", required=True,
                   help="grade-sample command; receives {run_dir}/submission.csv")
    p.add_argument("--deadline", type=float, required=True,
                   help="absolute Unix deadline for submission generation")
    p.add_argument("--grader-timeout", type=float, default=3600)
    p.add_argument("--manifest", type=Path)
    a = p.parse_args(argv)
    run_dir, data_dir = a.run_dir.resolve(), a.data_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"task": a.task, "run_dir": str(run_dir),
                "deadline_unix": a.deadline, "submission": {}, "grader": {}}
    rc, manifest["submission"] = run_submission(
        task=a.task,
        run_dir=run_dir,
        data_dir=data_dir,
        submission_command=a.submission_command,
        deadline=a.deadline,
    )
    if rc == 0:
        rc, manifest["grader"] = run_grader(
            task=a.task, run_dir=run_dir, data_dir=data_dir,
            grader_command=a.grader_command, grader_timeout=a.grader_timeout)
    out = a.manifest or (run_dir / "finalization-manifest.json")
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
