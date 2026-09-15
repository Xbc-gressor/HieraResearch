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
except ImportError:  # direct script execution from the repository checkout
    from mlebench_official import OfficialPreflightError, validate_submission


def _command(value: str, *, task: str, run_dir: Path, data_dir: Path) -> list[str]:
    rendered = value.format(task=task, run_dir=str(run_dir), data_dir=str(data_dir))
    return shlex.split(rendered)


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
    sub = _command(a.submission_command, task=a.task, run_dir=run_dir, data_dir=data_dir)
    started = time.time()
    remaining = a.deadline - started
    if remaining <= 0:
        manifest["submission"] = {"status": "deadline_expired", "started_at_unix": started}
        rc = 124
    else:
        try:
            proc = subprocess.run(sub, cwd=run_dir, timeout=remaining, check=False)
            rc = proc.returncode
            status = "failed"
            if rc == 0:
                try:
                    validate_submission(run_dir / "submission.csv")
                except (OfficialPreflightError, OSError):
                    pass
                else:
                    status = "success"
            manifest["submission"] = {"status": status, "returncode": rc,
                                       "started_at_unix": started, "ended_at_unix": time.time(),
                                       "elapsed_seconds": time.time() - started,
                                       "command": sub}
            if status != "success":
                rc = rc or 1
        except subprocess.TimeoutExpired:
            rc = 124
            manifest["submission"] = {"status": "timeout", "started_at_unix": started,
                                       "ended_at_unix": time.time(), "command": sub}
        except OSError as exc:
            rc = 1
            manifest["submission"] = {"status": "failed", "returncode": rc,
                                       "started_at_unix": started, "ended_at_unix": time.time(),
                                       "command": sub, "error": str(exc)}
    if rc == 0:
        grader = _command(a.grader_command, task=a.task, run_dir=run_dir, data_dir=data_dir)
        gs = time.time()
        try:
            gp = subprocess.run(grader, cwd=run_dir, timeout=a.grader_timeout, check=False)
            manifest["grader"] = {"status": "success" if gp.returncode == 0 else "failed",
                                   "returncode": gp.returncode, "started_at_unix": gs,
                                   "ended_at_unix": time.time(),
                                   "elapsed_seconds": time.time() - gs, "command": grader,
                                   "counts_toward_submission_budget": False}
            rc = gp.returncode
        except subprocess.TimeoutExpired:
            rc = 124
            manifest["grader"] = {"status": "timeout", "started_at_unix": gs,
                                   "ended_at_unix": time.time(), "command": grader,
                                   "counts_toward_submission_budget": False}
        except OSError as exc:
            rc = 1
            manifest["grader"] = {"status": "failed", "returncode": rc,
                                   "started_at_unix": gs, "ended_at_unix": time.time(),
                                   "command": grader, "error": str(exc),
                                   "counts_toward_submission_budget": False}
    out = a.manifest or (run_dir / "finalization-manifest.json")
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
