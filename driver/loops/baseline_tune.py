"""Deterministic strong-baseline loop: the task-provided baseline plus ONE
HEBO MACE bout spanning the entire objective budget.

This is the experiment protocol's INITIAL bout stretched to
``max_evaluations``: setup and the provided-baseline step 0+1 are exactly the
experiment loop's (same background research, same candidate ``000``
admission, same warm screening). After that there is no ideation, no
scheduler, and no tuner-orchestrator session — the driver deterministically
owns candidate selection (``000``), method (``hebo``), and bout size (the
remaining budget), and launches the search as one driver-owned job, so the
whole tune runs in one continuous session.

The run freezes ``tuner.inner_policy = baseline-hebo-full-v1`` (single bout,
budget-sized) with ``tuner.scheduler_policy = legacy`` (never consulted);
``phase-c-action`` / ``finalize_tuning.py`` / the driver-job machinery stay
the single deterministic path for stage state, exactly as in the experiment
loop.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..events import EventsLog
from ..jobs import DriverJobError, execute_driver_job
from ..receipts import ReceiptStore
from ..roles import REPO_ROOT, InvocationContext
from ..status import budget_status, compact_status
from . import common
from .common import RunBlocked
from .experiment import (
    _complete_run,
    _ensure_provided_baseline,
    _init_run_extra,
    _or_block,
    _resume_setup,
    _setup,
)

BASELINE_RUN_ID = "000"
BASELINE_INNER_POLICY = "baseline-hebo-full-v1"


def _phase_c_action(run_dir: Path, candidate_path: Path, report_path: Path,
                    repo_root, cmd) -> dict:
    try:
        out = cmd(["python", "tools/tuners/tune_tools.py", "phase-c-action",
                   "--candidate-path", candidate_path,
                   "--tune-report-json", report_path], repo_root)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or str(exc)).strip()
        raise RunBlocked(f"phase-c-action failed: {detail}")
    return json.loads(out.stdout)


def _tune_full_budget(task, tag, run_dir, repo_root, cmd, events,
                      job_runner=execute_driver_job) -> None:
    """Run candidate 000's single HEBO bout until it finalizes.

    Recovery mirrors the tuner-orchestrator contract deterministically: an
    interrupted stage gets one immediate resume; if that also crashes, the
    run blocks with the stage left running so a later driver launch can
    retry. A budget-exhausted stage closes terminally and finalizes with its
    scored trials.
    """
    candidate_dir = run_dir / "candidates" / BASELINE_RUN_ID
    candidate_path = candidate_dir / "train.py"
    report_path = candidate_dir / "tune_report.json"
    invocation_id = 0
    while True:
        action = _phase_c_action(run_dir, candidate_path, report_path,
                                 repo_root, cmd)
        kind = action.get("action")
        if kind == "run":
            method = action.get("method")
            if method != "hebo":
                _or_block(run_dir, repo_root, cmd, events,
                          f"baseline bout must run hebo; got {method!r}")
            remaining = budget_status(run_dir, repo_root, cmd).get("remaining")
            cap = int(action.get("bout_trials") or 0)
            if isinstance(remaining, int):
                cap = min(cap, remaining)
            if cap <= 0:
                _or_block(run_dir, repo_root, cmd, events,
                          "no objective budget remaining for the baseline "
                          "bout")
            invocation_id += 1
            ctx = InvocationContext(
                task=task, tag=tag, run_dir=run_dir,
                invocation_id=invocation_id, run_id=BASELINE_RUN_ID,
                round_no=0,
            )
            try:
                result = job_runner(
                    "driver", ctx,
                    {"kind": "phase_c", "run_id": BASELINE_RUN_ID,
                     "method": "hebo", "trial_cap": cap},
                    repo_root=repo_root,
                )
            except (DriverJobError, OSError, subprocess.SubprocessError) as exc:
                _or_block(run_dir, repo_root, cmd, events,
                          f"baseline hebo job failed: {exc}")
            if int(result.get("returncode", 1)) != 0:
                recheck = _phase_c_action(run_dir, candidate_path, report_path,
                                          repo_root, cmd)
                if recheck.get("action") == "run":
                    # Still resumable: leave the stage running and let the
                    # next driver launch retry instead of hot-looping here.
                    _or_block(run_dir, repo_root, cmd, events,
                              "baseline hebo job crashed; the interrupted "
                              "stage stays open for a later resume")
        elif kind == "close_exhausted_stage":
            cmd(["python", "tools/tuners/tune_tools.py",
                 "close-exhausted-stage",
                 "--candidate-path", candidate_path,
                 "--tune-report-json", report_path], repo_root)
        elif kind == "finalize":
            cmd(["python", "tools/finalize_tuning.py",
                 "--candidate-path", candidate_path,
                 "--tune-report-json", report_path,
                 "--ledger", run_dir / "ledger.json",
                 "--run-id", BASELINE_RUN_ID], repo_root)
            return
        else:
            _or_block(run_dir, repo_root, cmd, events,
                      f"unexpected phase-c-action: {action}")


def run_baseline_tune(task, tag, *, runner, model, repo_root=REPO_ROOT,
                      max_evaluations=None, timeout=None, k_warm=None,
                      k_eval=None, cli_path=None, cmd=common.run_cmd,
                      job_runner=execute_driver_job) -> dict:
    """Set up or resume the run, then spend the whole budget on one bout."""
    run_dir = repo_root / "runs" / task / tag
    events = EventsLog(run_dir)
    task_toml = common.load_task_toml(task, repo_root)
    store = ReceiptStore(run_dir)

    if not task_toml.get("seed", {}).get("provided"):
        raise ValueError(
            f"baseline-tune requires [seed].provided in tasks/{task}/"
            "task.toml (a task-provided train.py to tune)"
        )

    try:
        if not (run_dir / "framework_cfg.json").exists():
            if max_evaluations is None:
                raise ValueError(
                    "baseline-tune requires --max-evaluations for a new run: "
                    "the single HEBO bout spans the whole budget"
                )
            _setup(runner, store, task, tag, run_dir, task_toml, repo_root,
                   cmd, events, max_evaluations, timeout, None, None,
                   None, "legacy", BASELINE_INNER_POLICY, k_warm, k_eval,
                   model, cli_path)
        else:
            extra = _init_run_extra(None, None, None, "legacy",
                                    BASELINE_INNER_POLICY, k_warm, k_eval)
            if max_evaluations is not None or timeout is not None or extra:
                common.init_run(task, tag, repo_root, cmd, max_evaluations,
                                timeout, extra=extra)
            _resume_setup(runner, store, task, tag, run_dir, repo_root, cmd,
                          events, model, cli_path)

        _ensure_provided_baseline(runner, store, task, tag, run_dir,
                                  task_toml, repo_root, cmd, events,
                                  job_runner)

        _tune_full_budget(task, tag, run_dir, repo_root, cmd, events,
                          job_runner)
        _complete_run(run_dir, repo_root, cmd, events)
    except RunBlocked:
        pass

    return compact_status(task, tag, run_dir, repo_root=repo_root, cmd=cmd)
