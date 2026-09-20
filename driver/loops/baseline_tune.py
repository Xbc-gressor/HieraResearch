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
from pathlib import Path

from ..events import EventsLog
from ..jobs import execute_driver_job
from ..receipts import ReceiptStore
from ..roles import REPO_ROOT
from ..status import budget_status, compact_status
from . import common
from . import phase_c
from .common import RunBlocked
from .experiment import (
    _brief,
    _complete_run,
    _ensure_provided_baseline,
    _init_run_extra,
    _or_block,
    _refresh,
    _resume_setup,
    _setup,
)

BASELINE_RUN_ID = "000"
BASELINE_INNER_POLICY = "baseline-hebo-full-v1"


def _tuning_finalized(run_dir: Path) -> bool:
    """Whether candidate 000's single bout already closed and applied.

    A finalized single-bout report makes ``phase-c-action`` raise the
    inner-policy single-bout contract error, so resume must skip the tuning
    loop entirely once the close has landed.
    """
    report_path = run_dir / "candidates" / BASELINE_RUN_ID / "tune_report.json"
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(report, dict) and report.get("applied_to_base_params") is True


def _tune_full_budget(task, tag, run_dir, repo_root, cmd, events,
                      job_runner=execute_driver_job) -> None:
    """Run candidate 000's single HEBO bout until it finalizes.

    A crashed stage stays open for a later driver launch to resume; the
    shared bout loop reports the failure and this baseline loop blocks the
    run (its only recovery is interactive). A budget-exhausted stage closes
    terminally and finalizes with its scored trials.
    """
    counter = iter(range(1, 10_000))

    def trial_cap(action: dict) -> int:
        method = action.get("method")
        if method != "hebo":
            raise phase_c.PhaseCBoutFailure(
                f"baseline bout must run hebo; got {method!r}")
        remaining = budget_status(run_dir, repo_root, cmd).get("remaining")
        cap = int(action.get("bout_trials") or 0)
        if isinstance(remaining, int):
            cap = min(cap, remaining)
        return cap

    try:
        phase_c.run_single_bout(
            task, tag, run_dir, BASELINE_RUN_ID, repo_root, cmd, job_runner,
            next_invocation_id=lambda: next(counter),
            trial_cap_fn=trial_cap, round_no=0)
    except phase_c.PhaseCBoutFailure as exc:
        _or_block(run_dir, repo_root, cmd, events, f"baseline bout: {exc}")


def run_baseline_tune(task, tag, *, runner, model, repo_root=REPO_ROOT,
                      max_evaluations=None, timeout=None, k_warm=None,
                      k_eval=None, proposer_arm=None, cli_path=None,
                      cmd=common.run_cmd,
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
                   model, cli_path, proposer_arm=proposer_arm)
        else:
            extra = _init_run_extra(None, None, None, "legacy",
                                    BASELINE_INNER_POLICY, k_warm, k_eval,
                                    proposer_arm=proposer_arm)
            if max_evaluations is not None or timeout is not None or extra:
                common.init_run(task, tag, repo_root, cmd, max_evaluations,
                                timeout, extra=extra)
            _resume_setup(runner, store, task, tag, run_dir, repo_root, cmd,
                          events, model, cli_path)

        _ensure_provided_baseline(runner, store, task, tag, run_dir,
                                  task_toml, repo_root, cmd, events,
                                  job_runner)

        if not _tuning_finalized(run_dir):
            _tune_full_budget(task, tag, run_dir, repo_root, cmd, events,
                              job_runner)
        brief = _brief(run_dir, repo_root, cmd)
        if brief.get("experience_refresh_required"):
            _refresh(runner, store, task, tag, run_dir, repo_root, cmd,
                     events)
        _complete_run(run_dir, repo_root, cmd, events)
    except RunBlocked:
        pass

    return compact_status(task, tag, run_dir, repo_root=repo_root, cmd=cmd)
