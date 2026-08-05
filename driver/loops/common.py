"""Shared helpers for the two loops: tools/ subprocess wrappers."""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

from ..events import EventsLog
from ..roles import REPO_ROOT, ROLES, InvocationContext


def run_cmd(args, repo_root, check=True, capture=True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in args], cwd=repo_root,
                          capture_output=capture, text=True, check=check, **kw)


def load_task_toml(task: str, repo_root: Path) -> dict:
    with (repo_root / "tasks" / task / "task.toml").open("rb") as fh:
        return tomllib.load(fh)


def init_run(task, tag, repo_root, cmd, max_evaluations=None, timeout=None, extra=()):
    args = ["python", "tools/init_run.py", task, tag]
    if max_evaluations is not None:
        args += ["--max-evaluations", str(max_evaluations)]
    if timeout is not None:
        args += ["--timeout", str(timeout)]
    args += list(extra)
    cmd(args, repo_root)


def preflight_env(task, run_dir, repo_root, cmd) -> None:
    """The fixed environment gate; a non-zero exit blocks the run."""
    cmd(["uv", "--project", f"tasks/{task}", "run", "python",
         "tools/preflight_env.py", "--task", task, "--run-dir", run_dir],
        repo_root)


def set_phase(run_dir, repo_root, cmd, phase, stop_condition=None) -> None:
    args = ["python", "tools/ledger.py", "set-phase",
            "--ledger", run_dir / "ledger.json", "--phase", phase]
    if stop_condition:
        args += ["--stop-condition", stop_condition]
    cmd(args, repo_root)


def block(run_dir, repo_root, cmd, events: EventsLog, reason: str) -> None:
    events.emit("blocked", reason=reason)
    set_phase(run_dir, repo_root, cmd, "blocked", stop_condition=reason)


def crash_diagnose(runner, store, task, tag, run_dir, evidence: str) -> dict:
    """Level-2 escalation: read-only diagnosis of ONE failure.

    Returns the verdict receipt. The caller (a repair-capable session or
    the driver) applies the fix — crash-diagnosis never writes the ledger.
    """
    ctx = InvocationContext(
        task=task, tag=tag, run_dir=run_dir,
        invocation_id=store.next_invocation_id(),
        extra={"failure_evidence": evidence},
    )
    return runner.run(ROLES["crash-diagnosis"], ctx)


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "".join(path.read_text(encoding="utf-8", errors="replace")
                   .splitlines(keepends=True)[-lines:])
