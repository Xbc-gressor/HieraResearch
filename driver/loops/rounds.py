"""round_v1 optimization phase: rewrite bouts, then tune bouts.

The experiment loop alternates a generation phase (judged-slate generations
until the round's candidate threshold is met) with this optimization phase
over the whole candidate pool. Every decision lives tool-side
(``tools/scheduler/cli.py round ...``); this module sequences the bouts,
reuses the rewrite loop's bout runner, and commits a kept rewrite through
``tools/rewrite_rebase.py`` and ``tools/ledger.py record-rewrite``.

GPU work is strictly serial: each bout's evaluations run under the task
resource lease, one at a time.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import time

from ..session import InvocationFailed
from ..status import budget_status
from . import rewrite

POLICY_ID = "round_v1"


def _round(run_dir: Path, repo_root: Path, cmd, *args) -> dict:
    proc = cmd(["python", "tools/scheduler/cli.py", "round",
                "--ledger", run_dir / "ledger.json", *args], repo_root)
    return json.loads(proc.stdout)


def _record(run_dir, repo_root, cmd, decision_id, run_id, *, consumed,
            status, gain=None) -> None:
    args = ["python", "tools/scheduler/cli.py", "record",
            "--ledger", run_dir / "ledger.json",
            "--decision-id", decision_id, "--action", "REWRITE",
            "--run-id", run_id, "--consumed", str(consumed),
            "--status", status]
    if gain is not None:
        args += ["--gain", str(gain)]
    cmd(args, repo_root)


def _ledger_score(run_dir: Path, run_id: str) -> float | None:
    """The candidate's current best, tuned score first (like the scheduler)."""
    path = run_dir / "ledger.json"
    if not path.exists():
        return None
    for record in json.loads(path.read_text(encoding="utf-8")).get("records", []):
        if str(record.get("run_id")) != run_id:
            continue
        keys = ["best_warm_score", "final_best_score"]
        if record.get("tune"):
            keys.insert(0, "final_best_score")
        for key in keys:
            value = record.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and math.isfinite(value):
                return float(value)
    return None


def _run_best(run_dir: Path) -> float | None:
    """The best finite score over every candidate in the run (the editor's bar)."""
    path = run_dir / "ledger.json"
    if not path.exists():
        return None
    scores = []
    for record in json.loads(path.read_text(encoding="utf-8")).get("records", []):
        score = _ledger_score(run_dir, str(record.get("run_id")))
        if score is not None:
            scores.append(score)
    return min(scores) if scores else None


def status(run_dir, repo_root, cmd) -> dict:
    """The generate-or-optimize switch for this loop iteration."""
    return _round(run_dir, repo_root, cmd, "status")


def _eval_seconds(run_dir, repo_root, cmd, run_id) -> tuple[int, float | None]:
    view = budget_status(run_dir, repo_root, cmd)
    row = next((r for r in view.get("per_candidate", [])
                if r.get("run_id") == run_id), {})
    return int(view.get("evaluations_done") or 0), row.get("mean_seconds")


def _overhead(run_dir, repo_root, cmd, kind, run_id, started, evals_before):
    """Record the bout's non-evaluation seconds (session + tooling)."""
    evals_after, mean_seconds = _eval_seconds(run_dir, repo_root, cmd, run_id)
    wall = time.monotonic() - started
    eval_seconds = (evals_after - evals_before) * (mean_seconds or 0.0)
    _round(run_dir, repo_root, cmd, "overhead", "--kind", kind,
           "--seconds", str(max(0.0, wall - eval_seconds)))


def _rewrite_bout(runner, store, task, tag, run_dir, selection, task_toml,
                  noise_margin, repo_root, cmd, events) -> dict:
    """One rewrite bout on the selected candidate; commits a kept edit."""
    run_id = str(selection["run_id"])
    candidate = run_dir / "candidates" / run_id
    reference = selection.get("reference")
    if reference is None:
        reference = _ledger_score(run_dir, run_id)
    metric = task_toml["result"]["metric"]
    bouts = rewrite._load_bouts(candidate)
    started = time.monotonic()
    evals_before, _ = _eval_seconds(run_dir, repo_root, cmd, run_id)
    try:
        result = rewrite._run_bout(
            task, tag, run_dir, candidate, bouts, runner, store, metric,
            noise_margin, "full", task_toml, repo_root, cmd, events,
            reference=reference, confirm=True, run_best=_run_best(run_dir))
    except InvocationFailed as exc:
        # The failed session's edit is already rolled back; one candidate's
        # dead editor session does not stop the run.
        events.emit("rewrite_editor_failed", run_id=run_id,
                    problems=[str(p) for p in exc.problems])
        _record(run_dir, repo_root, cmd, selection["decision_id"], run_id,
                consumed=0, status="infra_failure")
        return {"status": "failed", "outcome": None}
    if result["outcome"] == "kept":
        _commit_kept(run_dir, run_id, candidate, len(bouts) + 1, result,
                     repo_root, cmd, events)
    _record(run_dir, repo_root, cmd, selection["decision_id"], run_id,
            consumed=result["attempts"],
            status="valid" if result["status"] == "done" else "infra_failure",
            gain=(None if result["outcome"] != "kept" or reference is None
                  or result["reference"] is None
                  else reference - result["reference"]))
    _overhead(run_dir, repo_root, cmd, "rewrite", run_id, started, evals_before)
    return result


def _commit_kept(run_dir, run_id, candidate, bout, result, repo_root, cmd,
                 events) -> None:
    """Rebase the tuning report onto the new code, then commit the score."""
    reference = result["reference"]
    report_args = []
    rebase = cmd(["python", "tools/rewrite_rebase.py",
                  "--candidate", candidate, "--bout", str(bout),
                  "--score", str(reference),
                  "--attempts", str(max(1, result["attempts"]))],
                 repo_root, check=False)
    if rebase.returncode == 0:
        report_args = ["--tune-report", candidate / "tune_report.json"]
    else:
        # The code improvement stands; the candidate merely stays untunable
        # until a later rebase succeeds (round_policy checks report currency).
        events.emit("rewrite_rebase_failed", run_id=run_id, bout=bout,
                    detail=(rebase.stderr or rebase.stdout or "")[-2000:])
    try:
        cmd(["python", "tools/ledger.py", "record-rewrite",
             "--ledger", run_dir / "ledger.json", "--run-id", run_id,
             "--score", str(reference), *report_args], repo_root)
    except subprocess.CalledProcessError as exc:
        events.emit("rewrite_ledger_failed", run_id=run_id, bout=bout,
                    detail=(exc.stderr or str(exc))[-2000:])
        return
    events.emit("rewrite_kept", run_id=run_id, bout=bout,
                score=result["score"], reference=reference)


def optimization_phase(runner, store, task, tag, run_dir, round_no, task_toml,
                       repo_root, cmd, events, *, tune, config) -> bool:
    """Run one optimization round; True when any bout ran."""
    begun = _round(run_dir, repo_root, cmd, "begin")
    events.emit("round_begin", round_no=round_no, cycle=begun.get("cycle"),
                phase_deadline=begun.get("phase_deadline"))
    progressed = False
    try:
        for _ in range(int(config["rewrite_bouts"])):
            if budget_status(run_dir, repo_root, cmd).get("reached"):
                break
            selection = _round(run_dir, repo_root, cmd, "select",
                               "--kind", "rewrite")
            events.emit("round_select", bout_kind="rewrite",
                        action=selection["action"],
                        run_id=selection.get("run_id"),
                        reason=selection.get("reason"))
            if selection["action"] != "REWRITE":
                break
            result = _rewrite_bout(runner, store, task, tag, run_dir, selection,
                                   task_toml, float(config["noise_margin"]),
                                   repo_root, cmd, events)
            progressed = progressed or result["status"] == "done"
            if result["status"] == "budget":
                break
        for _ in range(int(config["tune_bouts"])):
            if budget_status(run_dir, repo_root, cmd).get("reached"):
                break
            selection = _round(run_dir, repo_root, cmd, "select", "--kind", "tune")
            events.emit("round_select", bout_kind="tune",
                        action=selection["action"],
                        run_id=selection.get("run_id"),
                        reason=selection.get("reason"))
            if selection["action"] != "TUNE":
                break
            run_id = str(selection["run_id"])
            started = time.monotonic()
            evals_before, _ = _eval_seconds(run_dir, repo_root, cmd, run_id)
            receipt = tune(round_no)
            progressed = progressed or bool(receipt.get("tuned"))
            _overhead(run_dir, repo_root, cmd, "tune", run_id, started,
                      evals_before)
    finally:
        ended = _round(run_dir, repo_root, cmd, "end")
        events.emit("round_end", round_no=round_no, cycle=ended.get("cycle"),
                    cycle_start_count=ended.get("cycle_start_count"))
    return progressed
