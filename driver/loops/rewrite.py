"""Rewrite-operator loop: schedule bouts for imported candidates.

Thin driver over the tools/ CLIs — every deterministic decision (snapshot,
keep/revert adjudication, reference score, journal) lives tool-side; the
loop only orchestrates: render context -> snapshot -> resume the
per-candidate editor session -> verify the receipt against the file ->
preflight -> evaluate -> finalize.

The editor's receipt is never trusted on its own: after each bout session
the loop compares train.py against the bout snapshot (rewrite_bout
changed). When receipt and file disagree the bout is journaled noop — a
dirty file is reverted first — and never reaches preflight/eval.

One bout never spends budget before the candidate is known runnable:
preflight and rewrite_eval's BASE_PARAMS literal check both run ahead of
any reservation, so their failures cost nothing (revert + reverted_crash).
An unverified edit never survives: on budget exhaustion mid-bout the edit
is rolled back and the run ends normally; after a hard kill mid-bout the
replayed bout restores train.py from its existing snapshot (tool-side).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..events import EventsLog
from ..metadata import warn_on_mismatch, write_metadata
from ..receipts import ReceiptStore
from ..resources import task_resource_lease
from ..roles import REPO_ROOT, ROLES, InvocationContext
from ..session import InvocationFailed
from ..status import budget_status
from . import common

SETUP_GUARD = "_rewrite_setup_done"
REWRITE_DIR = "_rewrite"
IMPORT_MANIFEST = "_import.json"


# --- run-artifact reads (tool-owned writes) ---------------------------------


def _load_bouts(candidate: Path) -> list[dict]:
    """Read-only mirror of tools/rewrite_bout.py load_bouts: the tool owns
    every write to bouts.jsonl; the loop only derives scheduling facts."""
    path = candidate / REWRITE_DIR / "bouts.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _consecutive_non_kept(bouts: list[dict]) -> int:
    """Trailing run of non-kept outcomes; mirrors
    tools/rewrite_bout.py consecutive_non_kept (the stall signal)."""
    count = 0
    for entry in reversed(bouts):
        if entry.get("outcome") == "kept":
            break
        count += 1
    return count


def _session_path(candidate: Path) -> Path:
    return candidate / REWRITE_DIR / "session.json"


def _load_session_inv(candidate: Path) -> int | None:
    """The per-candidate editor-session pointer (plain run-artifact json)."""
    path = _session_path(candidate)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("invocation_id")


def _save_session_inv(candidate: Path, invocation_id: int) -> None:
    path = _session_path(candidate)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"invocation_id": invocation_id}) + "\n",
                    encoding="utf-8")


def _discover_candidates(run_dir: Path) -> list[Path]:
    base = run_dir / "candidates"
    if not base.is_dir():
        return []
    return sorted(
        path
        for path in base.iterdir()
        if path.is_dir() and (path / IMPORT_MANIFEST).is_file()
    )


# --- tools/ CLI wrappers (all through the injected cmd) ----------------------


def _render_context(candidate: Path, context: str, repo_root: Path, cmd) -> None:
    args = ["python", "tools/rewrite_context.py", "--candidate", candidate]
    if context != "full":  # the ablation switch passes straight through
        args += ["--sections", context]
    cmd(args, repo_root)


def _snapshot(candidate: Path, bout: int, repo_root: Path, cmd) -> str:
    proc = cmd(["python", "tools/rewrite_bout.py", "snapshot",
                "--candidate", candidate, "--bout", str(bout)], repo_root)
    return proc.stdout.strip()


def _revert(candidate: Path, snapshot: str, repo_root: Path, cmd) -> None:
    cmd(["python", "tools/rewrite_bout.py", "revert",
         "--candidate", candidate, "--snapshot", snapshot], repo_root)


def _changed(candidate: Path, snapshot: str, repo_root: Path, cmd) -> bool:
    """True when train.py differs byte-wise from the bout snapshot (the
    changed CLI's exit 1); a missing file/snapshot also reads as changed."""
    proc = cmd(["python", "tools/rewrite_bout.py", "changed",
                "--candidate", candidate, "--snapshot", snapshot],
               repo_root, check=False)
    return proc.returncode != 0


def _current_best(candidate: Path, repo_root: Path, cmd) -> float | None:
    """None when the candidate has no finite score reference to adjudicate
    against (e.g. a crashed import with no kept bout yet)."""
    proc = cmd(["python", "tools/rewrite_bout.py", "current-best",
                "--candidate", candidate], repo_root, check=False)
    if proc.returncode != 0:
        return None
    return float(proc.stdout.strip())


def _journal(candidate: Path, bout: int, outcome: str, receipt: dict,
             repo_root: Path, cmd) -> None:
    cmd(["python", "tools/rewrite_bout.py", "journal",
         "--candidate", candidate, "--bout", str(bout),
         "--outcome", outcome,
         "--summary", receipt["summary"], "--basis", receipt["basis"]],
        repo_root)


def _finalize(candidate: Path, bout: int, payload: dict, noise_margin: float,
              receipt: dict, repo_root: Path, cmd) -> dict:
    score = payload.get("score")
    finite = (isinstance(score, (int, float)) and not isinstance(score, bool)
              and math.isfinite(score))
    args = ["python", "tools/rewrite_bout.py", "finalize",
            "--candidate", candidate, "--bout", str(bout),
            "--noise-margin", str(noise_margin),
            "--summary", receipt["summary"], "--basis", receipt["basis"]]
    attempt_id = payload.get("attempt_id")
    if attempt_id:  # unrecoverable ids are journaled as null
        args += ["--attempt-id", attempt_id]
    args += ["--score", str(score)] if finite else ["--nonfinite"]
    proc = cmd(args, repo_root)
    return json.loads(proc.stdout)


def _preflight(task: str, candidate: Path, repo_root: Path, cmd,
               task_toml) -> str | None:
    """The task's no-score gate; returns an error tail on failure."""
    owner = {"task": task, "run_dir": str(candidate),
             "kind": "candidate_preflight"}
    with task_resource_lease(task_toml or {}, owner=owner):
        proc = cmd(["uv", "--project", f"tasks/{task}", "run", "python",
                    repo_root / "tools" / "preflight_candidate.py",
                    "--candidate-path", candidate / "train.py"],
                   repo_root, check=False)
    if proc.returncode == 0:
        return None
    return ("candidate preflight failed. stderr tail:\n"
            + (proc.stderr or "")[-3000:]
            + "\npreflight stdout tail:\n" + (proc.stdout or "")[-1000:])


def _evaluate(task: str, candidate: Path, repo_root: Path, cmd,
              task_toml) -> tuple[int, dict]:
    """One rewrite_eval call: (returncode, payload). Exit 4 = budget out;
    stage "params" = BASE_PARAMS unreadable (no budget spent)."""
    owner = {"task": task, "run_dir": str(candidate), "kind": "rewrite_eval"}
    with task_resource_lease(task_toml or {}, owner=owner):
        proc = cmd(["python", "tools/rewrite_eval.py", "--candidate",
                    candidate], repo_root, check=False)
    try:
        payload = json.loads(proc.stdout or "")
    except ValueError:
        payload = {"attempt_id": None, "score": None, "stage": "eval",
                   "error": "unparseable rewrite_eval output: "
                            + (proc.stdout or "")[-200:]}
    return proc.returncode, payload


def _params_error(payload: dict) -> str | None:
    """The preflight-class failure rewrite_eval reports without spending
    budget: BASE_PARAMS unreadable (stage "params")."""
    if payload.get("stage") == "params":
        return "rewrite_eval could not read BASE_PARAMS: " + str(
            payload.get("error"))
    return None


# --- editor session -----------------------------------------------------------


def _editor_session(runner, store, task, tag, run_dir, extra,
                    resume_from: int | None) -> tuple[dict, int]:
    inv_id = store.next_invocation_id()
    resume = None
    if resume_from is not None:
        resume = store.load_session_id("rewrite-editor", resume_from)
    ctx = InvocationContext(task=task, tag=tag, run_dir=run_dir,
                            invocation_id=inv_id, extra=extra or {},
                            resume_session_id=resume)
    receipt = runner.run(ROLES["rewrite-editor"], ctx)
    return receipt, inv_id


# --- setup ---------------------------------------------------------------------


def _setup(task, tag, run_dir, task_toml, repo_root, max_evaluations, timeout,
           cmd, events) -> None:
    """Idempotent run bring-up (guarded by SETUP_GUARD): no run-level working
    copy — imported candidates carry their own train.py."""
    events.emit("setup", task=task, tag=tag)
    common.init_run(task, tag, repo_root, cmd, max_evaluations, timeout)
    cmd(["uv", "--project", f"tasks/{task}", "sync"], repo_root)
    common.run_prepare(task, task_toml, repo_root, cmd)
    common.preflight_env(task, run_dir, repo_root, cmd)
    cmd(["python", "tools/evaluation_budget.py", "status",
         "--run-dir", run_dir, "--initialize"], repo_root)


# --- one bout -------------------------------------------------------------------


def _editor_extra(candidate: Path, current_best: float, metric: str,
                  bouts: list[dict]) -> dict:
    extra = {
        "candidate_dir": str(candidate),
        "current_best": current_best,
        "metric": metric,
    }
    if bouts:
        last = bouts[-1]
        extra["last_outcome"] = last.get("outcome")
        score = last.get("score")
        extra["last_score"] = score if score is not None else "null"
        attempt_id = last.get("attempt_id")
        if attempt_id:
            extra["last_trace"] = str(
                candidate / "_traces" / f"{attempt_id}.log")
    return extra


def _run_bout(task, tag, run_dir, candidate, bouts, runner, store, metric,
              noise_margin, context, task_toml, repo_root, cmd,
              events) -> str:
    """Run one bout; returns "done" | "budget" | "no_reference"."""
    bout = len(bouts) + 1
    _render_context(candidate, context, repo_root, cmd)
    snapshot = _snapshot(candidate, bout, repo_root, cmd)
    best = _current_best(candidate, repo_root, cmd)
    if best is None:
        events.emit("candidate_no_reference", candidate=candidate.name)
        return "no_reference"

    extra = _editor_extra(candidate, best, metric, bouts)
    try:
        receipt, inv_id = _editor_session(
            runner, store, task, tag, run_dir, extra,
            resume_from=_load_session_inv(candidate))
    except InvocationFailed:
        # The failed session may have dirtied train.py; an unverified edit
        # must not survive — a blocked run leaves a clean tree even when it
        # is never restarted. (A hard kill mid-bout is covered separately:
        # the replayed bout's snapshot call restores from the existing
        # snapshot instead of re-snapshotting the dirty file.)
        _revert(candidate, snapshot, repo_root, cmd)
        raise
    _save_session_inv(candidate, inv_id)

    # Receipt/file agreement: the editor's `edited` claim is only trusted
    # when the bytes agree. A disagreement journaled noop never reaches
    # preflight/eval; a dirtied file is reverted first (an unverified edit
    # must not survive).
    file_changed = _changed(candidate, snapshot, repo_root, cmd)
    if not receipt["edited"]:
        if file_changed:
            _revert(candidate, snapshot, repo_root, cmd)
            receipt = {**receipt,
                       "summary": "editor reported no edit but train.py "
                                  "differed; reverted"}
        _journal(candidate, bout, "noop", receipt, repo_root, cmd)
        events.emit("bout", candidate=candidate.name, bout=bout,
                    outcome="noop")
        return "done"
    if not file_changed:
        _journal(candidate, bout, "noop", receipt, repo_root, cmd)
        events.emit("bout", candidate=candidate.name, bout=bout,
                    outcome="noop")
        return "done"

    # Preflight-class gate: preflight, then rewrite_eval's BASE_PARAMS check
    # (both free). One repair resume with the error tail; a second failure
    # reverts and journals reverted_crash without spending budget.
    error_tail = _preflight(task, candidate, repo_root, cmd, task_toml)
    payload = None
    if error_tail is None:
        returncode, payload = _evaluate(task, candidate, repo_root, cmd,
                                        task_toml)
        if returncode == 4:
            # Budget exhausted: the unverified edit must not survive.
            _revert(candidate, snapshot, repo_root, cmd)
            events.emit("budget_exhausted", candidate=candidate.name,
                        bout=bout)
            return "budget"
        error_tail = _params_error(payload)
        if error_tail is not None:
            payload = None
    if error_tail is not None:
        repair_extra = dict(extra)
        repair_extra["preflight_error"] = error_tail
        try:
            receipt, inv_id = _editor_session(
                runner, store, task, tag, run_dir, repair_extra,
                resume_from=inv_id)
        except InvocationFailed:
            # Same rollback rule as the initial session: a failed repair may
            # have left a partial edit behind.
            _revert(candidate, snapshot, repo_root, cmd)
            raise
        _save_session_inv(candidate, inv_id)
        if receipt["edited"]:
            error_tail = _preflight(task, candidate, repo_root, cmd,
                                    task_toml)
            if error_tail is None:
                returncode, payload = _evaluate(task, candidate, repo_root,
                                                cmd, task_toml)
                if returncode == 4:
                    _revert(candidate, snapshot, repo_root, cmd)
                    events.emit("budget_exhausted", candidate=candidate.name,
                                bout=bout)
                    return "budget"
                error_tail = _params_error(payload)
                if error_tail is not None:
                    payload = None
        if error_tail is not None:
            _revert(candidate, snapshot, repo_root, cmd)
            _journal(candidate, bout, "reverted_crash", receipt, repo_root,
                     cmd)
            events.emit("bout", candidate=candidate.name, bout=bout,
                        outcome="reverted_crash", stage="preflight")
            return "done"

    result = _finalize(candidate, bout, payload, noise_margin, receipt,
                       repo_root, cmd)
    events.emit("bout", candidate=candidate.name, bout=bout,
                outcome=result["outcome"], score=payload.get("score"))
    return "done"


# --- status ---------------------------------------------------------------------


def _status(task, tag, run_dir: Path, metric: str, candidates: list[Path],
            stop_condition: str, repo_root: Path, cmd) -> dict:
    summary = {}
    for candidate in candidates:
        bouts = _load_bouts(candidate)
        summary[candidate.name] = {
            "bouts": len(bouts),
            "consecutive_non_kept": _consecutive_non_kept(bouts),
            "best": _current_best(candidate, repo_root, cmd),
        }
    return {
        "task": task,
        "tag": tag,
        "run_dir": str(run_dir),
        "metric": metric,
        "candidates": summary,
        "steps_done": budget_status(run_dir, repo_root, cmd).get(
            "evaluations_done", 0),
        "active_stop_condition": stop_condition,
    }


# --- main loop --------------------------------------------------------------------


def run_rewrite(task, tag, *, runner, model, noise_margin=0.0, max_bouts=12,
                stall_after=5, context="full", max_evaluations=None,
                timeout=None, cli_path=None, repo_root=REPO_ROOT,
                cmd=common.run_cmd) -> dict:
    run_dir = repo_root / "runs" / task / tag
    events = EventsLog(run_dir)
    task_toml = common.load_task_toml(task, repo_root)
    metric = task_toml["result"]["metric"]
    stop_condition = "none"

    if not (run_dir / SETUP_GUARD).exists():
        try:
            _setup(task, tag, run_dir, task_toml, repo_root,
                   max_evaluations, timeout, cmd, events)
        except RuntimeError as exc:  # prepare_command failed: block, no traceback
            stop_condition = str(exc)
            events.emit("blocked", reason=stop_condition)
            return _status(task, tag, run_dir, metric, [], stop_condition,
                           repo_root, cmd)
        write_metadata(run_dir, model, cli_path)
        (run_dir / SETUP_GUARD).touch()
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

    store = ReceiptStore(run_dir)
    candidates = _discover_candidates(run_dir)
    # Candidates without a finite score reference cannot be adjudicated at
    # all (current_best fails); they sit out the rest of the run.
    dead: set[Path] = set()
    halt = False
    while not halt:
        if budget_status(run_dir, repo_root, cmd).get("reached"):
            break
        active = [
            (candidate, _load_bouts(candidate))
            for candidate in candidates
            if candidate not in dead
        ]
        active = [
            (candidate, bouts)
            for candidate, bouts in active
            if len(bouts) < max_bouts
            and _consecutive_non_kept(bouts) < stall_after
        ]
        if not active:
            break
        for candidate, bouts in active:
            try:
                result = _run_bout(task, tag, run_dir, candidate, bouts,
                                   runner, store, metric, noise_margin,
                                   context, task_toml, repo_root, cmd, events)
            except InvocationFailed as exc:
                stop_condition = f"editor invocation failed: {exc.problems}"
                events.emit("blocked", reason=stop_condition)
                halt = True
                break
            if result == "budget":
                halt = True
                break
            if result == "no_reference":
                dead.add(candidate)

    return _status(task, tag, run_dir, metric, candidates, stop_condition,
                   repo_root, cmd)
