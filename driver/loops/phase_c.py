"""Driver-owned Phase-C bout sequencing, shared by baseline-tune and round_v1.

One call to :func:`run_single_bout` is the deterministic tuner contract for
one bout: repeat ``phase-c-action`` → run the driver job / close the
exhausted stage, until ``finalize`` lands. The loop never decides block
policy — a bout that cannot reach finalize raises :class:`PhaseCBoutFailure`
with the reason, and each caller settles it its own way (baseline blocks the
run interactively; round_v1 records the decision outcome and continues).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..jobs import DriverJobError, execute_driver_job
from ..roles import InvocationContext


class PhaseCBoutFailure(Exception):
    """The deterministic bout could not reach finalize; the caller settles."""


def phase_c_action(candidate_path: Path, report_path: Path, repo_root,
                   cmd) -> dict:
    out = cmd(["python", "tools/tuners/tune_tools.py", "phase-c-action",
               "--candidate-path", candidate_path,
               "--tune-report-json", report_path], repo_root)
    return json.loads(out.stdout)


def _action_or_fail(candidate_path, report_path, repo_root, cmd) -> dict:
    try:
        return phase_c_action(candidate_path, report_path, repo_root, cmd)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or str(exc)).strip()
        raise PhaseCBoutFailure(f"phase-c-action failed: {detail}") from exc
    except json.JSONDecodeError as exc:
        raise PhaseCBoutFailure(
            f"phase-c-action returned invalid JSON: {exc}") from exc


def run_single_bout(task, tag, run_dir: Path, run_id: str, repo_root, cmd,
                    job_runner=execute_driver_job, *, next_invocation_id,
                    trial_cap_fn, round_no=0, finalize_task=None,
                    resume_once=False) -> None:
    """Advance ``run_id``'s Phase-C report to its next finalize.

    ``trial_cap_fn(action)`` prices the job request from the deterministic
    action (baseline: min(bout_trials, remaining); round_v1: the complete
    bout). ``resume_once`` grants one in-loop retry of an interrupted stage,
    mirroring the retired orchestrator contract's single immediate resume;
    without it a crash leaves the stage open for a later driver launch.
    """
    candidate_dir = run_dir / "candidates" / str(run_id)
    candidate_path = candidate_dir / "train.py"
    report_path = candidate_dir / "tune_report.json"
    resumed = False
    while True:
        action = _action_or_fail(candidate_path, report_path, repo_root, cmd)
        kind = action.get("action")
        if kind == "run":
            cap = int(trial_cap_fn(action))
            if cap <= 0:
                raise PhaseCBoutFailure(
                    "no objective budget remaining for the bout")
            ctx = InvocationContext(
                task=task, tag=tag, run_dir=run_dir,
                invocation_id=next_invocation_id(), run_id=str(run_id),
                round_no=round_no,
            )
            try:
                result = job_runner(
                    "driver", ctx,
                    {"kind": "phase_c", "run_id": str(run_id),
                     "method": action.get("method"), "trial_cap": cap},
                    repo_root=repo_root,
                )
            except (DriverJobError, OSError, subprocess.SubprocessError) as exc:
                raise PhaseCBoutFailure(f"phase-c job failed: {exc}") from exc
            if int(result.get("returncode", 1)) != 0:
                recheck = _action_or_fail(candidate_path, report_path,
                                          repo_root, cmd)
                if recheck.get("action") == "run":
                    if resume_once and not resumed:
                        resumed = True
                        continue
                    raise PhaseCBoutFailure(
                        "phase-c job crashed; the interrupted stage stays "
                        "open for a later resume")
        elif kind == "close_exhausted_stage":
            try:
                cmd(["python", "tools/tuners/tune_tools.py",
                     "close-exhausted-stage",
                     "--candidate-path", candidate_path,
                     "--tune-report-json", report_path], repo_root)
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or str(exc)).strip()
                raise PhaseCBoutFailure(
                    f"close-exhausted-stage failed: {detail}") from exc
        elif kind == "finalize":
            args = ["python", "tools/finalize_tuning.py",
                    "--candidate-path", candidate_path,
                    "--tune-report-json", report_path,
                    "--ledger", run_dir / "ledger.json",
                    "--run-id", str(run_id)]
            if finalize_task:
                args += ["--task", finalize_task]
            try:
                cmd(args, repo_root)
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or str(exc)).strip()
                raise PhaseCBoutFailure(
                    f"finalize_tuning failed: {detail}") from exc
            return
        else:
            raise PhaseCBoutFailure(f"unexpected phase-c-action: {action}")
