"""Deterministic hillclimb baseline: one working copy, edit → preflight →
reserve → run → record → keep/revert. Python owns everything the old
monolithic session did by diligence; the editor session only edits.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from ..events import EventsLog
from ..metadata import warn_on_mismatch, write_metadata
from ..receipts import ReceiptStore
from ..roles import REPO_ROOT, ROLES, InvocationContext
from ..session import InvocationFailed
from ..status import budget_status
from . import common

TSV_HEADER = "step\tscore\tstatus\tdescription\n"


# --- small file helpers -----------------------------------------------------


def _tsv_path(run_dir: Path) -> Path:
    return run_dir / "results.tsv"


def _tsv_rows(run_dir: Path) -> list[list[str]]:
    path = _tsv_path(run_dir)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.split("\t") for line in lines[1:]]


def _finite_scores(run_dir: Path) -> list[float]:
    scores = []
    for row in _tsv_rows(run_dir):
        try:
            value = float(row[1])
        except (IndexError, ValueError):
            continue
        if math.isfinite(value):
            scores.append(value)
    return scores


def _record(run_dir: Path, step: int, score: float, status: str, desc: str) -> None:
    rendered = "inf" if not math.isfinite(score) else f"{score:.6f}"
    with _tsv_path(run_dir).open("a", encoding="utf-8") as fh:
        fh.write(f"{step}\t{rendered}\t{status}\t{desc}\n")


def _keep(run_dir: Path, step: int) -> None:
    shutil.copy(run_dir / "train.py", run_dir / "best.py")
    history = run_dir / "history"
    history.mkdir(exist_ok=True)
    shutil.copy(run_dir / "train.py", history / f"{step:03d}.py")


def _revert(run_dir: Path) -> None:
    best = run_dir / "best.py"
    if best.exists():
        shutil.copy(best, run_dir / "train.py")


# --- evaluation plumbing ------------------------------------------------------


def _preflight(task, run_dir, repo_root, cmd):
    return cmd(["uv", "--project", f"tasks/{task}", "run", "python",
                repo_root / "tools" / "preflight_candidate.py",
                "--candidate-path", run_dir / "train.py"],
               repo_root, check=False)


def _reserve(run_dir, repo_root, cmd) -> bool:
    result = cmd(["python", "tools/evaluation_budget.py", "reserve",
                  "--ref-path", run_dir / "train.py",
                  "--phase", "hillclimb", "--method", "direct"],
                 repo_root, check=False)
    return result.returncode == 0


def _run_entrypoint(task, run_dir, per_runtime_limit, repo_root, cmd) -> Path:
    log_path = run_dir / "run.log"
    entrypoint = run_dir / "train.py"
    if per_runtime_limit:
        argv = ["uv", "--project", f"tasks/{task}", "run", "python",
                repo_root / "tools" / "timed_run.py", str(per_runtime_limit),
                "python", entrypoint]
    else:
        argv = ["uv", "--project", f"tasks/{task}", "run", "python", entrypoint]
    with log_path.open("w", encoding="utf-8") as fh:
        cmd(argv, repo_root, check=False, capture=False, stdout=fh)
    return log_path


def _parse_score(log_path: Path, metric: str) -> float:
    prefix = f"{metric}:"
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            try:
                return float(line[len(prefix):].strip())
            except ValueError:
                break
    return math.inf


# --- setup / reconciliation ---------------------------------------------------


def _setup(task, tag, run_dir, task_toml, repo_root, max_evaluations, timeout,
           cmd, events) -> None:
    events.emit("setup", task=task, tag=tag)
    common.init_run(task, tag, repo_root, cmd, max_evaluations, timeout)
    task_dir = repo_root / "tasks" / task
    shutil.copy(task_dir / "prepare.py", run_dir / "prepare.py")
    for editable in task_toml.get("constraints", {}).get("editable_files", []):
        source = task_dir / editable
        if source.exists():
            shutil.copy(source, run_dir / editable)
    cmd(["uv", "--project", f"tasks/{task}", "sync"], repo_root)
    if task_toml.get("run", {}).get("prepare_command"):
        cmd(task_toml["run"]["prepare_command"].split(), repo_root)
    common.preflight_env(task, run_dir, repo_root, cmd)
    _tsv_path(run_dir).write_text(TSV_HEADER, encoding="utf-8")
    cmd(["python", "tools/evaluation_budget.py", "status",
         "--run-dir", run_dir, "--initialize"], repo_root)


def _reconcile(run_dir, cmd, repo_root, events) -> None:
    """Helper syncs the attempt log; the DRIVER appends TSV recovery rows —
    two separate steps (spec crash-recovery table)."""
    cmd(["python", "tools/evaluation_budget.py", "status",
         "--run-dir", run_dir, "--initialize"], repo_root)
    done = budget_status(run_dir, repo_root, cmd).get("evaluations_done", 0)
    missing = done - len(_tsv_rows(run_dir))
    for i in range(missing):
        events.emit("reconcile_recovery_row", index=i)
        _record(run_dir, len(_tsv_rows(run_dir)), math.inf, "crash",
                "recovery: reserved attempt interrupted before outcome row")


# --- editor escalation ----------------------------------------------------------


def _editor_session(runner, store, task, tag, run_dir, extra=None,
                    resume_from: int | None = None) -> int:
    inv_id = store.next_invocation_id()
    resume = None
    if resume_from is not None:
        resume = store.load_session_id("hillclimb-editor", resume_from)
    ctx = InvocationContext(task=task, tag=tag, run_dir=run_dir,
                            invocation_id=inv_id, extra=extra or {},
                            resume_session_id=resume)
    runner.run(ROLES["hillclimb-editor"], ctx)
    return inv_id


# --- main loop --------------------------------------------------------------------


def run_hillclimb(task, tag, *, runner, model, repo_root=REPO_ROOT,
                  max_evaluations=None, timeout=None, cli_path=None,
                  cmd=common.run_cmd, crash_repairs=1) -> dict:
    run_dir = repo_root / "runs" / task / tag
    fresh = not _tsv_path(run_dir).exists()
    events = EventsLog(run_dir)
    task_toml = common.load_task_toml(task, repo_root)
    metric = task_toml["result"]["metric"]
    stop_condition = "none"

    if fresh:
        _setup(task, tag, run_dir, task_toml, repo_root,
               max_evaluations, timeout, cmd, events)
        write_metadata(run_dir, model, cli_path)
    else:
        for warning in warn_on_mismatch(run_dir, model, cli_path):
            events.emit("metadata_mismatch", warning=warning)
        _reconcile(run_dir, cmd, repo_root, events)
        common.preflight_env(task, run_dir, repo_root, cmd)

    cfg_text = (run_dir / "framework_cfg.json").read_text(encoding="utf-8")
    import json as _json
    per_runtime_limit = _json.loads(cfg_text).get("per_runtime_limit")

    store = ReceiptStore(run_dir)
    last_editor: int | None = None

    # Bootstrap the working copy if the task ships no entrypoint.
    if not (run_dir / "train.py").exists():
        last_editor = _editor_session(
            runner, store, task, tag, run_dir,
            extra={"bootstrap": "create the initial working copy per the "
                                 "task contract's tiny-driver fallback"})

    # Baseline: exactly one evaluation of the unmodified copy.
    if not _tsv_rows(run_dir):
        if _reserve(run_dir, repo_root, cmd):
            log = _run_entrypoint(task, run_dir, per_runtime_limit, repo_root, cmd)
            score = _parse_score(log, metric)
            _record(run_dir, 0, score,
                    "crash" if not math.isfinite(score) else "keep", "baseline")
            if math.isfinite(score):
                _keep(run_dir, 0)
            else:
                stop_condition = "baseline evaluation crashed"
                events.emit("blocked", reason=stop_condition)
                return _status(task, tag, run_dir, metric, stop_condition,
                               repo_root, cmd)

    while True:
        if budget_status(run_dir, repo_root, cmd).get("reached"):
            break
        _revert(run_dir)
        best_before = min(_finite_scores(run_dir), default=None)
        try:
            last_editor = _editor_session(runner, store, task, tag, run_dir,
                                          resume_from=last_editor)
        except InvocationFailed as exc:
            stop_condition = f"editor invocation failed: {exc.problems}"
            events.emit("blocked", reason=stop_condition)
            break

        if _preflight(task, run_dir, repo_root, cmd).returncode != 0:
            # Preflight failures consume no slot; one diagnosis cycle, else
            # abandon the idea (restore best) and move on.
            evidence = "candidate preflight failed after editor session"
            verdict = common.crash_diagnose(runner, store, task, tag, run_dir,
                                            evidence)["verdict"]
            if verdict == "abandon":
                _revert(run_dir)
                continue
            try:
                last_editor = _editor_session(
                    runner, store, task, tag, run_dir,
                    extra={"diagnosis_verdict": verdict},
                    resume_from=last_editor)
            except InvocationFailed as exc:
                stop_condition = f"editor repair failed: {exc.problems}"
                events.emit("blocked", reason=stop_condition)
                break
            if _preflight(task, run_dir, repo_root, cmd).returncode != 0:
                _revert(run_dir)
                continue

        if not _reserve(run_dir, repo_root, cmd):
            break  # normal budget completion (exit 4), never a crash
        log = _run_entrypoint(task, run_dir, per_runtime_limit, repo_root, cmd)
        score = _parse_score(log, metric)
        step = len(_tsv_rows(run_dir))
        if not math.isfinite(score):
            _record(run_dir, step, math.inf, "crash", "run produced no metric")
            repaired = False
            for _ in range(crash_repairs):
                verdict = common.crash_diagnose(
                    runner, store, task, tag, run_dir,
                    common.tail(log))["verdict"]
                if verdict == "abandon":
                    break
                try:
                    last_editor = _editor_session(
                        runner, store, task, tag, run_dir,
                        extra={"diagnosis_verdict": verdict,
                               "failure_evidence": common.tail(log)},
                        resume_from=last_editor)
                except InvocationFailed:
                    break
                if _preflight(task, run_dir, repo_root, cmd).returncode == 0:
                    repaired = True
                    break
            if not repaired:
                _revert(run_dir)
            continue  # a repair re-enters the loop; its retry reserves anew

        status = "keep" if (best_before is None or score < best_before) else "discard"
        _record(run_dir, step, score, status, "")
        if status == "keep":
            _keep(run_dir, step)

    return _status(task, tag, run_dir, metric, stop_condition, repo_root, cmd)


def _status(task, tag, run_dir: Path, metric: str, stop_condition: str,
            repo_root: Path, cmd) -> dict:
    rows = _tsv_rows(run_dir)
    finite = [(i, float(r[1])) for i, r in enumerate(rows)
              if r[1] != "inf" and math.isfinite(float(r[1]))]
    best_step, best_score = min(finite, key=lambda p: p[1]) if finite else (None, None)
    return {
        "task": task,
        "tag": tag,
        "run_dir": str(run_dir),
        "metric": metric,
        "best_score": best_score,
        "best_step": best_step,
        "last_status": rows[-1][2] if rows else "none",
        "last_score": rows[-1][1] if rows else "none",
        "steps_done": budget_status(run_dir, repo_root, cmd).get(
            "evaluations_done", len(rows)),
        "active_stop_condition": stop_condition,
    }
