"""Shared helpers for the two loops: tools/ subprocess wrappers."""

from __future__ import annotations

import json
import shlex
import subprocess
import tomllib
from pathlib import Path

from ..events import EventsLog
from ..roles import REPO_ROOT, ROLES, InvocationContext


class RunBlocked(Exception):
    """Raised to unwind the loop after block() persisted the blocked phase."""


def run_cmd(args, repo_root, check=True, capture=True, cwd=None, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in args], cwd=cwd or repo_root,
                          capture_output=capture, text=True, check=check, **kw)


def run_prepare(task, task_toml, repo_root, cmd) -> None:
    """Run [run].prepare_command in the task's declared working directory.

    cwd resolution: [run].working_dir when declared, else [env].project,
    else repo_root. A failed prepare raises RuntimeError; callers decide
    whether that blocks the run.
    """
    run_cfg = task_toml.get("run", {})
    prepare = run_cfg.get("prepare_command")
    if not prepare:
        return
    working_dir = run_cfg.get("working_dir")
    if working_dir:
        cwd = repo_root / working_dir
    else:
        project = task_toml.get("env", {}).get("project")
        cwd = repo_root / project if project else repo_root
    try:
        cmd(shlex.split(prepare), repo_root, cwd=cwd)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or str(exc)).strip()
        raise RuntimeError(f"prepare_command failed: {detail}") from exc


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
