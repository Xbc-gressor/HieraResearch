"""Shared helpers for the two loops: tools/ subprocess wrappers."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tomllib
from enum import StrEnum
from pathlib import Path

from ..events import EventsLog
from ..roles import REPO_ROOT, ROLES, InvocationContext


class RunBlocked(Exception):
    """Stop accepting work and unwind to the run boundary.

    Only ``_or_block``/``_refuse_if_blocked`` raise this. The first blocker
    persists the blocked phase; the joined boundary retries an unsuccessful
    persistence before returning status. Helpers report fatal conditions as
    ordinary exceptions, never as a bare RunBlocked with no stop recorded.
    """


class BlockClass(StrEnum):
    """The only failures allowed to block a run. Anything else enters a
    recovery frontier (``recovery.py``); EMPTY_FRONTIER is the block for a
    failure point that has no legal next action left."""

    COMPLIANCE = "compliance"
    LEDGER_INTEGRITY = "ledger_integrity"
    EVALUATION_SURFACE = "evaluation_surface"
    DEADLINE = "deadline"
    EMPTY_FRONTIER = "empty_frontier"


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


def task_project(task: str, task_toml: dict | None) -> str:
    """Resolve the shared or task-local evaluation environment."""
    return (task_toml or {}).get("env", {}).get("project", f"tasks/{task}")


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
    task_cfg = load_task_toml(task, repo_root)
    mlebench = task_cfg.get("mlebench") or {}
    if mlebench:
        public_env = mlebench.get("public_data_env")
        if not isinstance(public_env, str) or not public_env:
            # Ordinary exception: the run_experiment boundary converts it into
            # a persisted blocked phase (never a bare RunBlocked — that would
            # exit 0 with the ledger still claiming running).
            raise RuntimeError(
                f"tasks/{task}/task.toml declares invalid public_data_env")
        from tools.mlebench_stage import stage_public

        if os.environ.get("MLEBENCH_PRESTAGED") == "1":
            public_dir = Path(run_dir).resolve() / "run_input" / "public"
        else:
            public_dir = stage_public(
                task, Path(run_dir), competition_id=mlebench.get("competition_id")
            )
        os.environ[public_env] = str(public_dir)
    cmd(["uv", "--project", task_project(task, task_cfg), "run", "python",
         "tools/preflight_env.py", "--task", task, "--run-dir", run_dir],
        repo_root)


def set_phase(run_dir, repo_root, cmd, phase, stop_condition=None,
              terminal_leftover=False) -> None:
    args = ["python", "tools/ledger.py", "set-phase",
            "--ledger", run_dir / "ledger.json", "--phase", phase]
    if stop_condition:
        args += ["--stop-condition", stop_condition]
    if terminal_leftover:
        args += ["--terminal-leftover"]
    cmd(args, repo_root)


def block(run_dir, repo_root, cmd, events: EventsLog, reason: str, *,
          cls: BlockClass) -> None:
    events.emit("blocked", reason=reason, cls=str(cls))
    set_phase(run_dir, repo_root, cmd, "blocked", stop_condition=reason)


def crash_diagnose(runner, store, task, tag, run_dir, evidence: str) -> dict:
    """Level-2 escalation: read-only diagnosis of ONE failure.

    Returns the verdict receipt. The caller (a repair-capable session or
    the driver) applies the fix — crash-diagnosis never writes the ledger.
    """
    ctx = InvocationContext(
        task=task, tag=tag, run_dir=run_dir,
        invocation_id=store.issue_invocation_id(),
        extra={"failure_evidence": evidence},
    )
    return runner.run(ROLES["crash-diagnosis"], ctx)


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "".join(path.read_text(encoding="utf-8", errors="replace")
                   .splitlines(keepends=True)[-lines:])
