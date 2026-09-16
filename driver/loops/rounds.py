"""round_v1 optimization phase: rewrite climbs, then tune bouts.

The experiment loop alternates a generation phase (judged-slate generations
until the round's candidate threshold is met) with this optimization phase
over the whole candidate pool. Every decision lives tool-side
(``tools/scheduler/cli.py round ...``); this module sequences the bouts,
reuses the rewrite loop's bout runner, and commits a kept rewrite through
``tools/rewrite_rebase.py`` and ``tools/ledger.py record-rewrite``.

One REWRITE decision is one *climb* on the selected candidate: repeated
edit -> evaluate -> keep/revert steps (each journaled as its own bout, the
editor session and the adjudication reference chained across steps) until
the candidate stalls, reaches its bout cap, or the phase quota / run
budget runs out. Depth on one candidate is where hillclimbing's kept
chains come from; rotating single edits across candidates never climbs.

GPU work is strictly serial: each bout's evaluations run under the task
resource lease, one at a time.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import time
import os

from ..session import InvocationFailed
from ..status import budget_status
from . import rewrite

POLICY_ID = "round_v1"

# The selection domain production evaluations land in: every objective path
# publishes its scalar through the legacy adapter as a (proxy, fast) record
# indexed on the ledger. The driver CLI sets EVALUATION_STAGE/FIDELITY to
# this pair unless the operator already did.
DEFAULT_EVALUATION_DOMAIN = ("proxy", "fast")


def _round(run_dir: Path, repo_root: Path, cmd, *args) -> dict:
    proc = cmd(["python", "tools/scheduler/cli.py", "round",
                "--ledger", run_dir / "ledger.json", *args], repo_root)
    return json.loads(proc.stdout)


def _record(run_dir, repo_root, cmd, decision_id, run_id, *, action, consumed,
            status, gain=None) -> None:
    args = ["python", "tools/scheduler/cli.py", "record",
            "--ledger", run_dir / "ledger.json",
            "--decision-id", decision_id, "--action", action,
            "--run-id", run_id, "--consumed", str(consumed),
            "--status", status]
    if gain is not None:
        args += ["--gain", str(gain)]
    cmd(args, repo_root)


def _load_ledger_view(run_dir: Path) -> dict | None:
    """One ledger load with evaluator records materialized; None when the
    view cannot be built (missing ledger or unreadable records)."""
    path = run_dir / "ledger.json"
    if not path.exists():
        return None
    ledger = json.loads(path.read_text(encoding="utf-8"))
    try:
        from tools.scheduler.state import materialize_evaluation_records
        ledger = materialize_evaluation_records(ledger)
    except (ImportError, OSError, ValueError, TypeError):
        return None
    return ledger


def _record_score(record: dict) -> float | None:
    """One record's current best, tuned score first (like the scheduler)."""
    stage = os.environ.get("EVALUATION_STAGE")
    fidelity = os.environ.get("EVALUATION_FIDELITY")
    if stage or fidelity:
        if not stage or not fidelity:
            return None
        try:
            from tools.evaluation_records import best_score
            value = best_score(record.get("evaluation_records", ()),
                               stage=stage, fidelity=fidelity)
        except (ImportError, ValueError):
            return None
        return float(value) if value is not None and math.isfinite(float(value)) else None
    keys = ["best_warm_score", "final_best_score"]
    if record.get("tune"):
        keys.insert(0, "final_best_score")
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and math.isfinite(value):
            return float(value)
    return None


def _ledger_score(run_dir: Path, run_id: str) -> float | None:
    """The candidate's current best, tuned score first (like the scheduler)."""
    ledger = _load_ledger_view(run_dir)
    if ledger is None:
        return None
    for record in ledger.get("records", []):
        if str(record.get("run_id")) != run_id:
            continue
        return _record_score(record)
    return None


def _run_best(run_dir: Path) -> float | None:
    """The best finite score over every candidate in the run (the editor's bar)."""
    ledger = _load_ledger_view(run_dir)
    if ledger is None:
        return None
    scores = []
    for record in ledger.get("records", []):
        score = _record_score(record)
        if score is not None:
            scores.append(score)
    return min(scores) if scores else None


def status(run_dir, repo_root, cmd) -> dict:
    """The generate-or-optimize switch for this loop iteration."""
    stage = os.environ.get("EVALUATION_STAGE")
    fidelity = os.environ.get("EVALUATION_FIDELITY")
    args = ["status"]
    if stage or fidelity:
        if not stage or not fidelity:
            raise RuntimeError("EVALUATION_STAGE and EVALUATION_FIDELITY must be set together")
        args += ["--stage", stage, "--fidelity", fidelity]
    return _round(run_dir, repo_root, cmd, *args)


def _eval_seconds(run_dir, repo_root, cmd, run_id) -> tuple[int, float | None]:
    """(admitted attempts of ``run_id``, its mean evaluation seconds).

    Per-candidate on purpose: a bout's consumption is the delta of ITS
    candidate's attempts, so evaluations another channel runs concurrently
    are never charged to this decision.
    """
    view = budget_status(run_dir, repo_root, cmd)
    row = next((r for r in view.get("per_candidate", [])
                if r.get("run_id") == run_id), {})
    return int(row.get("evals") or 0), row.get("mean_seconds")


def _overhead(run_dir, repo_root, cmd, kind, run_id, started, evals_before):
    """Record the bout's non-evaluation seconds (session + tooling)."""
    evals_after, mean_seconds = _eval_seconds(run_dir, repo_root, cmd, run_id)
    wall = time.monotonic() - started
    eval_seconds = (evals_after - evals_before) * (mean_seconds or 0.0)
    _round(run_dir, repo_root, cmd, "overhead", "--kind", kind,
           "--seconds", str(max(0.0, wall - eval_seconds)))


def _tune_reserve(run_dir: Path, repo_root: Path, cmd,
                  *, enabled: bool) -> float:
    """Return the currently selected tune bout's priced cost.

    Rewrite runs first, so keep one tune bout's admission ticket aside. The
    peek selection is read-only; the actual tune selection is repeated after
    rewrite, when the candidate pool may have changed.
    """
    if not enabled:
        return 0.0
    selection = _round(run_dir, repo_root, cmd, "select", "--kind", "tune", "--peek")
    if selection.get("action") != "TUNE":
        return 0.0
    mode = selection.get("evidence_mode") or {}
    ranked = mode.get("ranked") or []
    run_id = str(selection.get("run_id"))
    row = next((row for row in ranked if str(row.get("run_id")) == run_id), None)
    expected = row.get("expected_seconds") if row else None
    try:
        return max(0.0, float(expected)) if expected is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _rewrite_climb(runner, store, task, tag, run_dir, selection, task_toml,
                   config, repo_root, cmd, events, *, tune_reserve=0.0) -> dict:
    """One scheduler REWRITE decision: hillclimb the selected candidate.

    Repeated edit -> evaluate -> keep/revert steps, each journaled as its
    own bout and each kept step committed, until the candidate stalls
    (``rewrite_stall_after`` consecutive non-kept steps), reaches its
    ``rewrite_max_bouts`` cap, or the phase quota / run budget runs out.
    The adjudication reference chains across steps, so every step is
    measured against the candidate's current best. The decision is
    recorded once, with the climb's total attempts and reference gain.
    """
    run_id = str(selection["run_id"])
    candidate = run_dir / "candidates" / run_id
    metric = task_toml["result"]["metric"]
    noise_margin = float(config["noise_margin"])
    max_bouts = int(config["rewrite_max_bouts"])
    stall_after = int(config["rewrite_stall_after"])
    overhead = float((selection.get("evidence_mode") or {}).get(
        "overhead_seconds") or config["session_overhead_seconds"])
    initial_reference = selection.get("reference")
    if initial_reference is None:
        initial_reference = _ledger_score(run_dir, run_id)
    reference = initial_reference
    steps = 0
    kept = 0
    attempts = 0
    bouts = rewrite._load_bouts(candidate)
    # Seed the stall streak from the journal so a climb resumed after a
    # kill does not re-spend the non-kept steps it already paid for.
    streak = rewrite._consecutive_non_kept(bouts)
    status = "done"
    stop = None
    while stop is None:
        if len(bouts) >= max_bouts:
            stop = "bout_cap"
            break
        if streak >= stall_after:
            stop = "stalled"
            break
        view = budget_status(run_dir, repo_root, cmd)
        if view.get("reached"):
            status, stop = "budget", "run_budget"
            break
        quota = view.get("phase_quota_remaining_seconds")
        _, mean = _eval_seconds(run_dir, repo_root, cmd, run_id)
        expected = None if mean is None else 2.0 * float(mean) + overhead
        available = None if quota is None else max(0.0, quota - tune_reserve)
        if available is not None and (
                available <= 0 or (expected is not None and expected > available)):
            stop = "round_quota"
            break
        started = time.monotonic()
        evals_before, _ = _eval_seconds(run_dir, repo_root, cmd, run_id)
        try:
            result = rewrite._run_bout(
                task, tag, run_dir, candidate, bouts, runner, store, metric,
                noise_margin, "full", task_toml, repo_root, cmd, events,
                reference=reference, confirm=True,
                run_best=_run_best(run_dir))
        except InvocationFailed as exc:
            # The failed session's edit is already rolled back; one
            # candidate's dead editor session does not stop the run.
            events.emit("rewrite_editor_failed", run_id=run_id,
                        problems=[str(p) for p in exc.problems])
            status, stop = "failed", "editor_failed"
            break
        _overhead(run_dir, repo_root, cmd, "rewrite", run_id, started,
                  evals_before)
        steps += 1
        attempts += result["attempts"]
        if result["status"] == "budget":
            status, stop = "budget", "run_budget"
            break
        if result["status"] == "no_reference":
            stop = "no_reference"
            break
        if result["outcome"] == "kept":
            kept += 1
            streak = 0
            _commit_kept(run_dir, run_id, candidate, len(bouts) + 1, result,
                         repo_root, cmd, events)
        else:
            streak += 1
        if result["reference"] is not None:
            reference = result["reference"]
        bouts = rewrite._load_bouts(candidate)
    _record(run_dir, repo_root, cmd, selection["decision_id"], run_id,
            action="REWRITE",
            consumed=attempts,
            status="valid" if steps > 0 else "infra_failure",
            gain=(None if kept == 0 or initial_reference is None
                  or reference is None else initial_reference - reference))
    events.emit("rewrite_climb", run_id=run_id, steps=steps, kept=kept,
                attempts=attempts, stop=stop, reference=reference)
    return {"status": status, "steps": steps, "kept": kept,
            "attempts": attempts, "stop": stop, "reference": reference}


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


def tune_handoff(selection: dict) -> dict:
    """The compact exact-target contract the tuner session must obey.

    Budget is the contract bout cost (evaluation count); round_v1's mid-bout
    truncation semantics (reservation refusal + `close_exhausted_stage`)
    are unchanged by the handoff.
    """
    keys = ("decision_id", "run_id", "policy_version", "state_snapshot_id",
            "evidence_cursor", "bout_trials")
    return {key: selection.get(key) for key in keys}


def close_tune_outcome(run_dir, repo_root, cmd, events, selection, receipt,
                       *, evals_before, reference_before) -> None:
    """Bind the tune bout's result to the decision that selected it.

    Every terminal path of a TUNE selection ends here: a kept improvement, a
    valid no-op, or a receipt whose `tuned_run_id` names another candidate.
    The orchestrator's objective jobs are pinned to the selected candidate,
    so a mismatching id can only be a stale or fabricated handoff — the
    result is never re-attributed to the candidate the receipt names.
    """
    run_id = str(selection["run_id"])
    executed = str(receipt.get("tuned_run_id", "none"))
    if executed not in (run_id, "none"):
        events.emit("tune_target_mismatch",
                    decision_id=selection.get("decision_id"),
                    selected_run_id=run_id, executed_run_id=executed)
        _record(run_dir, repo_root, cmd, selection["decision_id"], run_id,
                action="TUNE", consumed=0, status="infra_failure")
        return
    evals_after, _ = _eval_seconds(run_dir, repo_root, cmd, run_id)
    consumed = max(0, evals_after - evals_before)
    reference_after = _ledger_score(run_dir, run_id)
    gain = (None if reference_before is None or reference_after is None
            else reference_before - reference_after)
    _record(run_dir, repo_root, cmd, selection["decision_id"], run_id,
            action="TUNE", consumed=consumed, status="valid", gain=gain)
    events.emit("tune_bout", decision_id=selection.get("decision_id"),
                selected_run_id=run_id, executed_run_id=executed,
                tuned=bool(receipt.get("tuned")),
                consumed_evaluations=consumed, realized_gain=gain)


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
            tune_reserve = _tune_reserve(
                run_dir, repo_root, cmd,
                enabled=int(config["tune_bouts"]) > 0,
            )
            result = _rewrite_climb(runner, store, task, tag, run_dir,
                                    selection, task_toml, config, repo_root,
                                    cmd, events, tune_reserve=tune_reserve)
            progressed = progressed or result["steps"] > 0
            if result["status"] == "budget":
                break
        for _ in range(int(config["tune_bouts"])):
            if budget_status(run_dir, repo_root, cmd).get("reached"):
                break
            selection = _round(run_dir, repo_root, cmd, "select",
                               "--kind", "tune")
            events.emit("round_select", bout_kind="tune",
                        action=selection["action"],
                        run_id=selection.get("run_id"),
                        decision_id=selection.get("decision_id"),
                        reason=selection.get("reason"))
            if selection["action"] != "TUNE":
                break
            run_id = str(selection["run_id"])
            reference_before = _ledger_score(run_dir, run_id)
            started = time.monotonic()
            evals_before, _ = _eval_seconds(run_dir, repo_root, cmd, run_id)
            receipt = tune(round_no, selection)
            progressed = progressed or bool(receipt.get("tuned"))
            _overhead(run_dir, repo_root, cmd, "tune", run_id, started,
                      evals_before)
            close_tune_outcome(run_dir, repo_root, cmd, events, selection,
                               receipt, evals_before=evals_before,
                               reference_before=reference_before)
    finally:
        args = ["end"]
        stage = os.environ.get("EVALUATION_STAGE")
        fidelity = os.environ.get("EVALUATION_FIDELITY")
        if stage or fidelity:
            args += ["--stage", stage, "--fidelity", fidelity]
        ended = _round(run_dir, repo_root, cmd, *args)
        events.emit("round_end", round_no=round_no, cycle=ended.get("cycle"),
                    cycle_start_count=ended.get("cycle_start_count"))
    return progressed
