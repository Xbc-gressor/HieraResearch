"""Validated, driver-owned execution of long objective jobs.

Agents may prepare a warm-screening or Phase-C job, but they never launch it.
The driver derives every path and argv field from the invocation context,
executes the child in the foreground without an outer timeout, and persists a
small job record before returning control to the same agent session.

Process hygiene: the child runs in its own process group and its pid goes
into the job record, so a driver that is killed mid-job leaves evidence that
the next launch can reconcile — a stale "running" record with a dead pid is
marked ``dead``, and a live one refuses the new launch instead of letting a
second objective write the same tune_report concurrently. On SIGTERM /
interrupt / interpreter exit the driver terminates the child's whole process
group (best-effort; SIGKILL still orphans, which is what reconciliation is
for).
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .resources import task_resource_lease
from .roles import InvocationContext, REPO_ROOT


class DriverJobError(ValueError):
    pass


_live_children: list[subprocess.Popen] = []
_exit_hooks_armed = False


def _terminate_child_group(proc: subprocess.Popen) -> None:
    # start_new_session=True makes the child's pid its process-group id, so
    # this reaches the whole uv → script → torchrun tree.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass


def _kill_live_children() -> None:
    for proc in list(_live_children):
        _terminate_child_group(proc)


def _arm_exit_hooks() -> None:
    global _exit_hooks_armed
    if _exit_hooks_armed:
        return
    atexit.register(_kill_live_children)

    def _on_sigterm(signum, _frame) -> None:
        _kill_live_children()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _on_sigterm)
    _exit_hooks_armed = True


def _reconcile_running_jobs(jobs_dir: Path) -> None:
    """Reconcile leftover "running" records before launching a new job.

    A record whose pid is gone was orphaned by a killed driver: mark it
    ``dead``. A record whose pid is still alive means an orphaned objective
    (or another driver) may still be writing — refuse to launch a second one.
    """
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
            raise DriverJobError(
                f"a previous driver job is still running (pid {pid}, record "
                f"{path.name}); confirm it has exited or kill its process "
                "group before launching another objective job"
            )
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
        if role_name != "tunable-contract-extractor":
            raise DriverJobError("warmstart jobs belong to tunable-contract-extractor")
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
    _reconcile_running_jobs(ctx.run_dir / "driver_jobs")
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
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    owner = {"task": ctx.task, "tag": ctx.tag, "run_id": run_id, "kind": request["kind"]}
    try:
        with task_resource_lease(task_cfg, owner=owner):
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    argv,
                    cwd=repo_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                record["pid"] = proc.pid
                _atomic_json(record_path, record)
                _live_children.append(proc)
                _arm_exit_hooks()
                try:
                    returncode = proc.wait()
                except BaseException:
                    _terminate_child_group(proc)
                    raise
                finally:
                    _live_children.remove(proc)
    except (OSError, subprocess.SubprocessError) as exc:
        record.update(
            status="launch_failed",
            error=str(exc),
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        _atomic_json(record_path, record)
        raise DriverJobError(f"driver job could not run: {exc}") from exc
    record.update(
        status="completed",
        returncode=int(returncode),
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    _atomic_json(record_path, record)
    tail = "".join(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-40:]
    )
    return {
        "kind": request["kind"],
        "run_id": run_id,
        "returncode": int(returncode),
        "log": str(log_path),
        "log_tail": tail[-6000:],
        "job_record": str(record_path),
    }
