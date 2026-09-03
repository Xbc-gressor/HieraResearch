"""Deterministic hillclimb baseline: one working copy, edit → preflight →
reserve → run → record → keep/revert. Python owns everything the old
monolithic session did by diligence; the editor session only edits.
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

from ..events import EventsLog
from ..metadata import warn_on_mismatch, write_metadata
from ..receipts import ReceiptStore
from ..resources import task_resource_lease
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


def _record_keep(run_dir: Path, step: int, score: float, desc: str) -> None:
    """Snapshot BEFORE the TSV row, best.py AFTER it. A kill before the row
    leaves an orphaned snapshot — harmless: the reserved attempt still gets
    its crash-recovery row from _reconcile. A kill after the row leaves the
    snapshot available for _restore_best to repair best.py."""
    history = run_dir / "history"
    history.mkdir(exist_ok=True)
    shutil.copy(run_dir / "train.py", history / f"{step:03d}.py")
    _record(run_dir, step, score, "keep", desc)
    shutil.copy(history / f"{step:03d}.py", run_dir / "best.py")


TELEMETRY_PREFIXES = ("training_seconds:", "total_seconds:", "peak_vram_mb:",
                      "mfu_percent:", "total_tokens_M:", "num_steps:",
                      "num_params_M:", "depth:")


def _telemetry(log_path: Path) -> str:
    """Informational summary lines from a run log, delivered to the editor
    alongside the score (the Karpathy agent greps the log itself; here the
    driver hands the same numbers over)."""
    if not log_path.exists():
        return ""
    lines = [line.strip() for line in
             log_path.read_text(encoding="utf-8", errors="replace").splitlines()
             if line.startswith(TELEMETRY_PREFIXES)]
    return "; ".join(lines[:10])


def _crash_tail(log_path: Path) -> str:
    return f"\nrun log tail:\n{common.tail(log_path, 40)}"


def _outcome_note(score: float, best_before: float | None, status: str,
                  telemetry: str = "") -> str:
    """The feedback the next editor invocation receives about the edit just
    evaluated (the Karpathy loop's score feedback, which the driver owns)."""
    before = (f"{best_before:.6f}" if best_before is not None
              else "none (no finite score yet)")
    if status == "keep":
        verdict = (f"KEEP — it strictly improved the then-best ({before}) "
                   "and is the new incumbent the working copy is synced to")
    else:
        verdict = (f"DISCARD — it did not improve the then-best ({before}); "
                   "the working copy was reverted to the incumbent")
    note = f"your last edit scored {score:.6f} (lower is better): {verdict}."
    if telemetry:
        note += f" Run telemetry — {telemetry}."
    return note


def _revert(run_dir: Path) -> None:
    best = run_dir / "best.py"
    if best.exists():
        shutil.copy(best, run_dir / "train.py")


# --- evaluation plumbing ------------------------------------------------------


def _preflight(task, run_dir, repo_root, cmd, task_toml=None):
    owner = {"task": task, "run_dir": str(run_dir), "kind": "candidate_preflight"}
    with task_resource_lease(task_toml or {}, owner=owner):
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


def _run_entrypoint(task, run_dir, per_runtime_limit, repo_root, cmd,
                    task_toml=None) -> tuple[Path, int]:
    log_path = run_dir / "run.log"
    entrypoint = run_dir / "train.py"
    if per_runtime_limit:
        argv = ["uv", "--project", f"tasks/{task}", "run", "python",
                repo_root / "tools" / "timed_run.py", str(per_runtime_limit),
                "python", entrypoint]
    else:
        argv = ["uv", "--project", f"tasks/{task}", "run", "python", entrypoint]
    lease_owner = {"task": task, "run_dir": str(run_dir), "kind": "hillclimb"}
    with task_resource_lease(task_toml or {}, owner=lease_owner):
        with log_path.open("w", encoding="utf-8") as fh:
            result = cmd(argv, repo_root, check=False, capture=False, stdout=fh)
    return log_path, result.returncode


def _evaluate_outcome(log_path: Path, returncode: int, metric: str,
                      required_patterns: list[str]) -> float:
    """Metric from a COMPLETE successful run only: a non-zero exit, a missing
    required line, or a missing/unparseable metric line all score +inf."""
    if returncode != 0:
        return math.inf
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if any(not re.search(pattern, text, re.MULTILINE)
           for pattern in required_patterns):
        return math.inf
    prefix = f"{metric}:"
    for line in text.splitlines():
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
    common.run_prepare(task, task_toml, repo_root, cmd)
    common.preflight_env(task, run_dir, repo_root, cmd)
    _tsv_path(run_dir).write_text(TSV_HEADER, encoding="utf-8")
    cmd(["python", "tools/evaluation_budget.py", "status",
         "--run-dir", run_dir, "--initialize"], repo_root)


def _reconcile(run_dir, cmd, repo_root, events) -> None:
    """Validate the attempt log, then append any missing TSV recovery rows."""
    cmd(["python", "tools/evaluation_budget.py", "status",
         "--run-dir", run_dir, "--initialize"], repo_root)
    done = budget_status(run_dir, repo_root, cmd).get("evaluations_done", 0)
    missing = done - len(_tsv_rows(run_dir))
    for i in range(missing):
        events.emit("reconcile_recovery_row", index=i)
        _record(run_dir, len(_tsv_rows(run_dir)), math.inf, "crash",
                "recovery: reserved attempt interrupted before outcome row")


def _resume_outcome_note(run_dir: Path) -> str | None:
    """Rebuild the pending editor feedback after a driver restart from the
    authoritative TSV: the last row's outcome, with the then-best taken from
    all earlier rows (mirrors the fresh-path wording)."""
    rows = _tsv_rows(run_dir)
    if not rows:
        return None
    prior_finite = []
    for row in rows[:-1]:
        try:
            value = float(row[1])
        except (IndexError, ValueError):
            continue
        if math.isfinite(value):
            prior_finite.append(value)
    last = rows[-1]
    try:
        score = float(last[1])
    except (IndexError, ValueError):
        return None
    log_path = run_dir / "run.log"
    if last[0] == "0":
        if math.isfinite(score):
            note = (f"the unmodified baseline scored {score:.6f} (lower is "
                    "better); it is the current incumbent")
            telemetry = _telemetry(log_path)
            if telemetry:
                note += f" Run telemetry — {telemetry}."
            return note
        return ("the unmodified baseline CRASHED (no valid metric, scored "
                "+inf)" + _crash_tail(log_path))
    if not math.isfinite(score):
        return ("your last edit CRASHED (no valid metric, scored +inf); the "
                "driver was interrupted before delivering the outcome"
                + _crash_tail(log_path))
    best_before = min(prior_finite, default=None)
    return _outcome_note(score, best_before,
                         "keep" if len(last) > 2 and last[2] == "keep"
                         else "discard", _telemetry(log_path))


def _restore_best(run_dir, events) -> None:
    """Crash-recovery row 2: a kill between the keep row and the best.py
    snapshot leaves best.py stale. The TSV is authoritative: restore best.py
    from the best step's history snapshot when one exists and differs."""
    finite = []
    for i, row in enumerate(_tsv_rows(run_dir)):
        try:
            score = float(row[1])
        except (IndexError, ValueError):
            continue
        if math.isfinite(score):
            finite.append((i, score))
    if not finite:
        return
    best_step = min(finite, key=lambda p: p[1])[0]
    snapshot = run_dir / "history" / f"{best_step:03d}.py"
    if not snapshot.exists():
        return  # no snapshot to restore from; leave best.py alone
    best = run_dir / "best.py"
    if best.exists() and best.read_bytes() == snapshot.read_bytes():
        return
    shutil.copy(snapshot, best)
    events.emit("reconcile_recovery_row", restore_best_step=best_step)


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
    required_patterns = task_toml["result"].get("required_patterns", [])
    stop_condition = "none"

    if fresh:
        try:
            _setup(task, tag, run_dir, task_toml, repo_root,
                   max_evaluations, timeout, cmd, events)
        except RuntimeError as exc:  # prepare_command failed: block, no traceback
            stop_condition = str(exc)
            events.emit("blocked", reason=stop_condition)
            return _status(task, tag, run_dir, metric, stop_condition,
                           repo_root, cmd)
        write_metadata(run_dir, model, cli_path)
    else:
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.exists():
            for warning in warn_on_mismatch(run_dir, model, cli_path):
                events.emit("metadata_mismatch", warning=warning)
        elif model is not None:  # killed before setup wrote the metadata
            write_metadata(run_dir, model, cli_path)
        else:
            events.emit("metadata_mismatch",
                        warning="run_metadata.json missing and no model "
                                "supplied; metadata not written")
        _reconcile(run_dir, cmd, repo_root, events)
        _restore_best(run_dir, events)
        common.preflight_env(task, run_dir, repo_root, cmd)

    cfg_text = (run_dir / "framework_cfg.json").read_text(encoding="utf-8")
    import json as _json
    per_runtime_limit = _json.loads(cfg_text).get("per_runtime_limit")

    store = ReceiptStore(run_dir)
    # Reconnect the editor session chain after a driver restart: session ids
    # are persisted at invocation init, so the latest one is resumable.
    last_editor: int | None = store.latest_session_invocation("hillclimb-editor")
    # Feedback handed to the next fresh idea; on a restart, rebuild the
    # outcome that was recorded but never delivered to the editor.
    outcome_note: str | None = None if fresh else _resume_outcome_note(run_dir)

    # Bootstrap the working copy if the task ships no entrypoint.
    if not (run_dir / "train.py").exists():
        try:
            last_editor = _editor_session(
                runner, store, task, tag, run_dir,
                extra={"bootstrap": "create the initial working copy per the "
                                     "task contract's tiny-driver fallback"})
        except InvocationFailed as exc:
            stop_condition = f"editor bootstrap failed: {exc.problems}"
            events.emit("blocked", reason=stop_condition)
            return _status(task, tag, run_dir, metric, stop_condition,
                           repo_root, cmd)

    # Baseline: exactly one evaluation of the unmodified copy.
    if not _tsv_rows(run_dir):
        if _reserve(run_dir, repo_root, cmd):
            log, rc = _run_entrypoint(task, run_dir, per_runtime_limit,
                                      repo_root, cmd, task_toml)
            score = _evaluate_outcome(log, rc, metric, required_patterns)
            if math.isfinite(score):
                _record_keep(run_dir, 0, score, "baseline")
                outcome_note = (f"the unmodified baseline scored {score:.6f} "
                                "(lower is better); it is the current "
                                "incumbent")
                telemetry = _telemetry(log)
                if telemetry:
                    outcome_note += f" Run telemetry — {telemetry}."
            else:
                _record(run_dir, 0, score, "crash", "baseline")
                outcome_note = ("the unmodified baseline CRASHED (no valid "
                                "metric, scored +inf)" + _crash_tail(log))
            # A crashed baseline CONTINUES (same semantics as the resume
            # path): the editor starts from the crashed train.py, _revert
            # no-ops without best.py, and the first finite score keeps.

    best_before = min(_finite_scores(run_dir), default=None)
    needs_editor = True
    while True:
        if budget_status(run_dir, repo_root, cmd).get("reached"):
            break
        if needs_editor:
            _revert(run_dir)
            best_before = min(_finite_scores(run_dir), default=None)
            extra = {"outcome": outcome_note} if outcome_note else None
            outcome_note = None
            try:
                last_editor = _editor_session(runner, store, task, tag, run_dir,
                                              extra=extra,
                                              resume_from=last_editor)
            except InvocationFailed as exc:
                stop_condition = f"editor invocation failed: {exc.problems}"
                events.emit("blocked", reason=stop_condition)
                break
        needs_editor = True  # a fresh idea next round starts from best.py

        proc = _preflight(task, run_dir, repo_root, cmd, task_toml)
        if proc.returncode != 0:
            # Preflight failures consume no slot; one diagnosis cycle, else
            # abandon the idea (restore best) and move on. The diagnosis is a
            # fresh session — hand it the preflight output it cannot recover.
            evidence = (
                "candidate preflight failed after editor session. "
                "preflight stderr tail:\n" + (proc.stderr or "")[-3000:]
                + "\npreflight stdout tail:\n" + (proc.stdout or "")[-1000:]
            )
            try:
                verdict = common.crash_diagnose(
                    runner, store, task, tag, run_dir, evidence)["verdict"]
            except InvocationFailed as exc:
                events.emit("crash_diagnosis_failed", stage="preflight",
                            problems=exc.problems)
                verdict = "abandon"
            if verdict == "abandon":
                _revert(run_dir)
                outcome_note = ("your last edit failed candidate preflight "
                                "and the idea was abandoned (no evaluation "
                                "budget was consumed); the working copy was "
                                "reverted to the incumbent")
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
            if _preflight(task, run_dir, repo_root, cmd, task_toml).returncode != 0:
                _revert(run_dir)
                outcome_note = ("your repaired edit still failed candidate "
                                "preflight and the idea was abandoned (no "
                                "evaluation budget was consumed); the working "
                                "copy was reverted to the incumbent")
                continue

        if not _reserve(run_dir, repo_root, cmd):
            break  # normal budget completion (exit 4), never a crash
        log, rc = _run_entrypoint(task, run_dir, per_runtime_limit,
                                  repo_root, cmd, task_toml)
        score = _evaluate_outcome(log, rc, metric, required_patterns)
        step = len(_tsv_rows(run_dir))
        if not math.isfinite(score):
            _record(run_dir, step, math.inf, "crash", "run produced no metric")
            repaired = False
            for _ in range(crash_repairs):
                # Give the fresh diagnosis session the log path (it can Read
                # the full file) plus the tail as a starting point.
                evidence = (f"run log: {log}\n\nlast 80 lines:\n"
                            + common.tail(log))
                try:
                    verdict = common.crash_diagnose(
                        runner, store, task, tag, run_dir, evidence)["verdict"]
                except InvocationFailed as exc:
                    events.emit("crash_diagnosis_failed", stage="eval",
                                problems=exc.problems)
                    break
                if verdict == "abandon":
                    break
                try:
                    last_editor = _editor_session(
                        runner, store, task, tag, run_dir,
                        extra={"diagnosis_verdict": verdict,
                               "failure_evidence": evidence},
                        resume_from=last_editor)
                except InvocationFailed:
                    break
                if _preflight(task, run_dir, repo_root, cmd, task_toml).returncode == 0:
                    repaired = True
                    break
            if not repaired:
                _revert(run_dir)
                outcome_note = ("your last edit CRASHED (no valid metric, "
                                "scored +inf) and was abandoned after "
                                "diagnosis; the working copy was reverted to "
                                "the incumbent" + _crash_tail(log))
            else:
                needs_editor = False
            continue  # a repair re-enters the loop; its retry reserves anew

        status = "keep" if (best_before is None or score < best_before) else "discard"
        if status == "keep":
            _record_keep(run_dir, step, score, "")
        else:
            _record(run_dir, step, score, "discard", "")
        outcome_note = _outcome_note(score, best_before, status,
                                     _telemetry(log))

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
