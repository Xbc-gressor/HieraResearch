"""Validated, driver-owned execution of long objective jobs.

Agents may prepare a warm-screening or Phase-C job, but they never launch it.
The driver derives every path and argv field from the invocation context,
executes the child in the foreground without an outer timeout, and persists a
small job record before returning control to the same agent session.

Process hygiene: the child runs in its own process group and its pid goes
into the job record, so a driver that is killed mid-job leaves evidence that
the next launch can reconcile — a stale "running" record with a dead pid is
marked ``dead``, and a live one for the SAME candidate refuses the new launch
instead of letting a second objective write the same tune_report
concurrently. On SIGTERM / interrupt / interpreter exit the driver terminates
the child's whole process group and waits for it to exit (SIGTERM, grace,
SIGKILL); a SIGKILL of the driver itself still orphans, which is what
reconciliation is for. A job's ``proc.wait`` is bounded by the run's cutoff
(``deadline − final_reserve``): past it the job is terminated the same way
and the handoff returns ``accepted: False`` (``deadline_killed``).

Concurrency: role sessions may run on several driver threads, so jobs for
different candidates queue on the GPU channel (the device lease) instead of
failing; one candidate never has two jobs in flight. Lease queueing is bounded
by the run deadline, not the task's short ``lease_wait_timeout`` — waiting for
the GPU is not a candidate observation and is never shown to the session.
"""

from __future__ import annotations

import atexit
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .resources import ResourceUnavailable, task_resource_lease
from .roles import InvocationContext, REPO_ROOT, _ledger_records
from tools.evaluation_budget import median_eval_seconds, time_budget
from tools.run_cfg import load_run_cfg
from tools.process_group import terminate_group

# SIGTERM grace for a driver job's group before SIGKILL escalation.
JOB_TERMINATE_GRACE_SECONDS = 30.0


class DriverJobError(ValueError):
    pass


_live_children: list[subprocess.Popen] = []
_exit_hooks_armed = False
_candidate_locks: dict[str, threading.Lock] = {}
_candidate_locks_guard = threading.Lock()


def _candidate_lock(run_dir: Path, run_id: str) -> threading.Lock:
    """At most one objective job per candidate at a time (tune_report writer)."""
    key = f"{run_dir}:{run_id}"
    with _candidate_locks_guard:
        return _candidate_locks.setdefault(key, threading.Lock())


def _terminate_child_group(proc: subprocess.Popen) -> int | None:
    """Terminate the job's whole process group and wait for it to exit.

    start_new_session=True makes the child's pid its process-group id, so
    this reaches the whole uv → script → torchrun tree. Returns only after
    the group is gone (SIGTERM, grace, SIGKILL): the lease and the job
    record are handed over afterwards, never while the job may still run.
    """
    return terminate_group(proc, grace=JOB_TERMINATE_GRACE_SECONDS)


def _kill_live_children() -> None:
    for proc in list(_live_children):
        _terminate_child_group(proc)


def _arm_exit_hooks() -> None:
    """Idempotent; signal handlers can only be installed from the main thread,
    so loops arm this once at start-up and worker threads merely re-check."""
    global _exit_hooks_armed
    if _exit_hooks_armed:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    atexit.register(_kill_live_children)

    def _on_sigterm(signum, _frame) -> None:
        _kill_live_children()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _on_sigterm)
    _exit_hooks_armed = True


def _reconcile_running_jobs(jobs_dir: Path, run_id: str | None = None) -> None:
    """Reconcile leftover "running" records before launching a new job.

    A record whose pid is gone was orphaned by a killed driver: mark it
    ``dead``. A live pid owned by this driver is another candidate's job on
    the GPU channel and is left alone (the device lease queues behind it). A
    live pid this driver does not own is an orphaned objective (or another
    driver) that may still be writing: refuse when it works on ``run_id``, or
    on any candidate when the caller gave none.
    """
    own_pids = {proc.pid for proc in list(_live_children)}
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") != "running":
            continue
        pid = record.get("pid")
        alive = False
        if isinstance(pid, int) and pid > 0:
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
        if alive:
            if pid in own_pids:
                continue
            if run_id is None or str(record.get("run_id")) == str(run_id):
                raise DriverJobError(
                    f"a previous driver job is still running (pid {pid}, record "
                    f"{path.name}); confirm it has exited or kill its process "
                    "group before launching another objective job"
                )
            continue
        record.update(
            status="dead",
            note="pid gone at next driver launch; driver was killed mid-job",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        _atomic_json(path, record)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _positive_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DriverJobError(f"{label} must be a positive integer")
    return value


def _candidate(ctx: InvocationContext, requested_run_id: object) -> tuple[str, Path, Path]:
    run_id = str(requested_run_id)
    if not run_id or run_id == "None" or "/" in run_id or ".." in run_id:
        raise DriverJobError(f"invalid driver-job run_id: {requested_run_id!r}")
    if ctx.run_id is not None and run_id != str(ctx.run_id):
        raise DriverJobError(
            f"driver-job run_id {run_id!r} does not match invocation {ctx.run_id!r}"
        )
    candidate_dir = ctx.run_dir / "candidates" / run_id
    candidate_path = candidate_dir / "train.py"
    report_path = candidate_dir / "tune_report.json"
    if not candidate_path.is_file():
        raise DriverJobError(f"candidate entrypoint does not exist: {candidate_path}")
    return run_id, candidate_path, report_path


def _phase_c_action(repo_root: Path, candidate_path: Path, report_path: Path) -> dict:
    result = subprocess.run(
        [
            "python",
            "tools/tuners/tune_tools.py",
            "phase-c-action",
            "--candidate-path",
            str(candidate_path),
            "--tune-report-json",
            str(report_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise DriverJobError(
            "phase-c-action rejected the requested job: "
            + (result.stderr or result.stdout or "unknown error").strip()
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise DriverJobError(
            f"phase-c-action returned invalid JSON: {exc}"
        ) from exc


def _screening_cost_stop_args(run_dir: Path, run_id: str) -> list[str]:
    """Run-level scalars for warm screening's cost stop (tuner never reads the ledger).

    Omitted, so the stop never fires, without a phase_a duration median or an
    incumbent. The incumbent is the best screening score of never-rewritten
    records: ``record_rewrite`` overwrites ``best_warm_score`` with a rewrite
    reference, which is not a screening observation.
    """
    cfg = load_run_cfg(run_dir, "screening_cost_stop")
    if not cfg.get("enabled", True):
        return []
    median = median_eval_seconds(run_dir, "phase_a")
    scores = [
        float(record["best_warm_score"])
        for record in _ledger_records(run_dir)
        if record.get("run_id") != run_id
        and not record.get("rewrite_bouts")
        and isinstance(record.get("best_warm_score"), (int, float))
        and math.isfinite(record["best_warm_score"])
    ]
    if median is None or not scores:
        return []
    threshold = max(cfg.get("floor_seconds", 120), cfg.get("multiplier", 4) * median)
    return ["--cost-stop-threshold-seconds", str(threshold),
            "--cost-stop-incumbent", str(min(scores))]


def build_driver_job(
    role_name: str,
    ctx: InvocationContext,
    request: dict,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[str], Path, str]:
    """Validate a typed request and return ``(argv, log_path, run_id)``."""
    if not isinstance(request, dict):
        raise DriverJobError("driver_job must be an object")
    kind = request.get("kind")
    run_id, candidate_path, report_path = _candidate(ctx, request.get("run_id"))
    task_toml = repo_root / "tasks" / ctx.task / "task.toml"
    if not task_toml.is_file():
        raise DriverJobError(f"missing task.toml: {task_toml}")
    import tomllib

    task_cfg = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    project = task_cfg.get("env", {}).get("project")
    if not isinstance(project, str) or not project:
        raise DriverJobError("task env.project must be a non-empty string")

    if kind == "warmstart":
        if role_name != "driver":
            raise DriverJobError("warmstart jobs belong to the driver")
        k_eval = _positive_int(request.get("k_eval"), "driver_job.k_eval")
        expected_k_eval = ctx.extra.get("screening_k_eval")
        if expected_k_eval is not None:
            expected_k_eval = _positive_int(
                expected_k_eval, "invocation screening_k_eval"
            )
            if k_eval != expected_k_eval:
                raise DriverJobError(
                    "driver_job.k_eval must equal the driver-owned screening "
                    f"allocation {expected_k_eval}, got {k_eval}"
                )
        configs_path = candidate_path.parent / "_warm_configs.json"
        if not configs_path.is_file():
            raise DriverJobError(f"warm configs do not exist: {configs_path}")
        argv = [
            "uv",
            "--project",
            project,
            "run",
            "python",
            str(repo_root / "tools/tuners/warmstart_eval.py"),
            "--candidate-path",
            str(candidate_path),
            "--configs-json",
            str(configs_path),
            "--tune-report-json",
            str(report_path),
            "--k-eval",
            str(k_eval),
        ]
        target_k_eval = ctx.extra.get("screening_target_k_eval")
        if target_k_eval is not None:
            target_k_eval = _positive_int(
                target_k_eval, "invocation screening_target_k_eval"
            )
            if target_k_eval < k_eval:
                raise DriverJobError(
                    "screening_target_k_eval cannot be smaller than k_eval"
                )
            argv += ["--target-k-eval", str(target_k_eval)]
        # The generation-bound donor snapshot travels with the same mechanism:
        # the driver sets it from the manifest/per-candidate binding, and only
        # the transfer policy pair ever sets it (no_donor binds by omission).
        donor_snapshot = ctx.extra.get("donor_snapshot")
        if donor_snapshot is not None:
            argv += ["--donor-snapshot", str(donor_snapshot)]
        argv += _screening_cost_stop_args(ctx.run_dir, run_id)
        return argv, candidate_path.parent / "_warmstart.log", run_id

    if kind != "phase_c":
        raise DriverJobError(f"unsupported driver job kind: {kind!r}")
    # "driver" is the baseline-tune loop, which deterministically owns its
    # single full-budget bout and has no tuner-orchestrator session.
    if role_name not in ("tuner-orchestrator", "driver"):
        raise DriverJobError(
            "Phase-C jobs belong to tuner-orchestrator or the driver loop"
        )
    if not report_path.is_file():
        raise DriverJobError(f"tune report does not exist: {report_path}")
    method = request.get("method")
    action = _phase_c_action(repo_root, candidate_path, report_path)
    if action.get("action") != "run" or action.get("method") != method:
        raise DriverJobError(
            "requested Phase-C job does not match deterministic action: "
            f"requested={method!r}, action={action}"
        )
    trial_cap = _positive_int(request.get("trial_cap"), "driver_job.trial_cap")
    framework_cfg_path = ctx.run_dir / "framework_cfg.json"
    framework_cfg = (
        json.loads(framework_cfg_path.read_text(encoding="utf-8"))
        if framework_cfg_path.is_file()
        else {}
    )
    tuner_cfg = framework_cfg.get("tuner", {})
    if not isinstance(tuner_cfg, dict):
        raise DriverJobError("framework_cfg tuner section must be an object")
    # phase-c-action computes the exact policy-aware size from the candidate's
    # bout index (normally 8/10/10; the 24+20 policies are 24/10/10). The
    # legacy inner policy keeps tuner.bout_trials.
    bout_trials = _positive_int(
        action.get("bout_trials"), "phase-c-action.bout_trials"
    )
    if trial_cap > bout_trials:
        raise DriverJobError(
            f"driver_job.trial_cap {trial_cap} exceeds bout_trials {bout_trials}"
        )
    if tuner_cfg.get("scheduler_policy") in (
        "v3_2",
        "anchor_challenger_v1",
        "anchor_transfer_challenger_v1",
    ) and trial_cap != bout_trials:
        raise DriverJobError(
            "complete-bout scheduler requires one complete bout: "
            f"trial_cap {trial_cap} != bout_trials {bout_trials}"
        )
    script = repo_root / "tools" / "tuners" / f"{method}_search.py"
    if method in ("hebo", "local_tr", "selfrank", "mixup", "turbo"):
        # Repo-root env: HEBO needs the SDK session + official ranker;
        # selfrank/mixup need the SDK session; local_tr/turbo need the
        # inner-benchmark numerical stack. Evaluations stay in the task
        # project via timed_eval(python_cmd=...).
        argv = [
            "uv",
            "--project",
            str(repo_root),
            "run",
            "python",
            str(script),
            "--candidate-path",
            str(candidate_path),
            "--tune-report-json",
            str(report_path),
            "--n-evals",
            str(trial_cap),
        ]
        return argv, candidate_path.parent / f"_phase_c_{method}.log", run_id
    argv = [
        "uv",
        "--directory",
        project,
        "run",
        "python",
        str(script),
        "--candidate-path",
        str(candidate_path),
        "--tune-report-json",
        str(report_path),
    ]
    if method == "grid":
        argv += ["--resolution", "5", "--max-trials", str(min(100, trial_cap)),
                 "--patience", "6"]
    elif method == "bo":
        argv += ["--n-trials", str(trial_cap)]
        sampler = action.get("sampler")
        if sampler:
            argv += ["--sampler", str(sampler)]
    elif method == "spsa":
        argv += ["--n-evals", str(trial_cap)]
    elif method == "cmaes":
        argv += ["--popsize", "8", "--max-evals", str(min(64, trial_cap)),
                 "--patience", "20"]
    else:
        raise DriverJobError(f"unsupported Phase-C method: {method!r}")
    return argv, candidate_path.parent / f"_phase_c_{method}.log", run_id


def execute_driver_job(
    role_name: str,
    ctx: InvocationContext,
    request: dict,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict:
    argv, log_path, run_id = build_driver_job(
        role_name, ctx, request, repo_root=repo_root
    )
    import tomllib

    task_cfg = tomllib.loads(
        (repo_root / "tasks" / ctx.task / "task.toml").read_text(encoding="utf-8")
    )
    record_path = (
        ctx.run_dir / "driver_jobs" / f"{role_name}-{ctx.invocation_id:04d}.json"
    )
    record = {
        "schema_version": 1,
        "role": role_name,
        "invocation_id": ctx.invocation_id,
        "run_id": run_id,
        "request": request,
        "argv": argv,
        "log": str(log_path),
        "status": "queued",
        "queued_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    owner = {
        "task": ctx.task,
        "tag": ctx.tag,
        "run_id": run_id,
        "run_dir": str(ctx.run_dir),
        "kind": request["kind"],
    }
    _atomic_json(record_path, record)  # queued evidence survives a kill
    queued = time.monotonic()
    started = None
    try:
        with _candidate_lock(ctx.run_dir, run_id):
            _reconcile_running_jobs(ctx.run_dir / "driver_jobs", run_id)
            with task_resource_lease(
                    task_cfg, owner=owner,
                    wait_timeout=_lease_wait_timeout(ctx.run_dir)) as lease:
                env = {**os.environ, **(lease.get("env") or {})}
                # Queueing for the device may itself cross the cutoff; a job
                # that starts now would run into the export reserve.
                if time_budget(ctx.run_dir)["time_reached"]:
                    record.update(
                        status="refused_time_reached",
                        lease_wait_seconds=round(time.monotonic() - queued, 3),
                        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    )
                    _atomic_json(record_path, record)
                    return {
                        "kind": request["kind"],
                        "run_id": run_id,
                        "accepted": False,
                        "error": "time budget reached before the job could "
                                 "start; no further objective work is possible",
                        "job_record": str(record_path),
                    }
                started = time.monotonic()
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.Popen(
                        argv,
                        cwd=repo_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                        env=env,
                    )
                    record.update(
                        pid=proc.pid,
                        status="running",
                        devices=lease.get("devices"),
                        lease_wait_seconds=round(started - queued, 3),
                        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    )
                    _atomic_json(record_path, record)
                    _live_children.append(proc)
                    _arm_exit_hooks()
                    killed = False
                    try:
                        # Bounded by the run's cutoff, not by the job's own
                        # loop: past deadline − final_reserve the GPU must
                        # be free for export whatever the job is doing.
                        usable = time_budget(ctx.run_dir).get("usable_seconds")
                        try:
                            returncode = proc.wait(
                                timeout=None if usable is None
                                else max(0.0, float(usable)))
                        except subprocess.TimeoutExpired:
                            killed = True
                            returncode = _terminate_child_group(proc)
                    except BaseException:
                        _terminate_child_group(proc)
                        raise
                    finally:
                        # Only after the group has exited: the lease is
                        # released by the enclosing `with` once we leave.
                        _live_children.remove(proc)
    except ResourceUnavailable as exc:
        # Lease queueing is free: it is neither a candidate observation nor a
        # job result the session gets to see. The caller decides what an
        # unevaluable candidate at the deadline becomes.
        record.update(
            status="lease_unavailable",
            error=str(exc),
            lease_wait_seconds=round(time.monotonic() - queued, 3),
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        _atomic_json(record_path, record)
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        record.update(
            status="launch_failed",
            error=str(exc),
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        _atomic_json(record_path, record)
        raise DriverJobError(f"driver job could not run: {exc}") from exc
    tail = "".join(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-40:]
    )
    if killed:
        record.update(
            status="deadline_killed",
            returncode=None if returncode is None else int(returncode),
            eval_seconds=round(time.monotonic() - started, 3),
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        _atomic_json(record_path, record)
        return {
            "kind": request["kind"],
            "run_id": run_id,
            "accepted": False,
            "error": "time budget reached; the job was terminated and no "
                     "further objective work is possible",
            "log": str(log_path),
            "log_tail": tail[-6000:],
            "job_record": str(record_path),
        }
    record.update(
        status="completed",
        returncode=int(returncode),
        eval_seconds=round(time.monotonic() - started, 3),
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    _atomic_json(record_path, record)
    return {
        "kind": request["kind"],
        "run_id": run_id,
        "returncode": int(returncode),
        "log": str(log_path),
        "log_tail": tail[-6000:],
        "job_record": str(record_path),
    }


def _lease_wait_timeout(run_dir: Path) -> float | None:
    """Queue for the GPU until the run's usable time is gone; without a
    deadline, wait indefinitely (the process-group hygiene above keeps
    orphans from holding the lease forever)."""
    usable = time_budget(run_dir).get("usable_seconds")
    if usable is None:
        return None
    return max(0.0, float(usable))
