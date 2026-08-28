"""Wave-barrier matrix runner with job-level resume (no training resume).

Subcommands:

    plan          build matrix job requests + the frozen schedule manifest
    plan-repeats  calibration / adjudication repeats for one candidate
    run           execute a schedule wave by wave on explicit GPUs
    gate          full-fidelity equivalence verdict (epsilon_noise + pairs)

Scheduling rules (frozen at plan time so an interrupted run never reshuffles):
one candidate is pinned to one physical GPU (round-robin over candidates in
frozen pool/coverage order); its fidelity order is rotated by a stable hash of
the candidate id; each wave runs at most one job per GPU and the next wave
starts only after every job of the current wave is terminal and finalized.

Job-level resume: a job with a terminal result.json whose bindings match its
request is skipped; a job with only a request or provisional files reruns
from scratch. Crash/timeout are terminal observations — repeats need new job
ids via (purpose, repeat_index). No model checkpoint is read or written.

Timeouts are experiment-job-level (default 1800 s, the B-2 eval cap): the
whole process group is killed and the actual cost is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import manifest  # noqa: E402

REPO_ROOT = _HERE.parent.parent
DEFAULT_TIMEOUT_SECONDS = 1800.0
CHILD_PATH = _HERE / "autoresearch_eval_one.py"


class RunnerError(RuntimeError):
    pass


def _stable_int(*parts: str) -> int:
    material = "\x1f".join(parts)
    return int.from_bytes(
        hashlib.sha256(material.encode("utf-8")).digest()[:8], "big"
    )


def rotated_fidelities(candidate_id: str, fidelities: list[int]) -> list[int]:
    """Stable per-candidate rotation so 30 s is not always run first."""
    offset = _stable_int("fidelity-rotation", candidate_id) % len(fidelities)
    return list(fidelities[offset:]) + list(fidelities[:offset])


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def _write_request(jobs_dir: Path, request: dict) -> None:
    job_dir = jobs_dir / request["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest.freeze_immutable(job_dir / manifest.REQUEST_FILENAME, request)


def _schedule_doc(experiment_id: str, jobs: list[dict], waves: list[list[str]]) -> dict:
    return {
        "schema_version": manifest.SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "jobs": jobs,
        "waves": waves,
        "jobs_digest": manifest.digest(
            sorted(job["job_id"] for job in jobs)
        ),
    }


def plan_matrix(
    experiment: dict,
    pool_docs: list[dict],
    jobs_dir: Path,
) -> dict:
    """Build every (candidate, fidelity) matrix request plus the schedule."""
    manifest.validate_experiment(experiment)
    devices = experiment["hardware"]["devices"]
    fidelities = [int(f) for f in experiment["fidelities"]]
    task = experiment["task"]

    per_gpu: list[list[dict]] = [[] for _ in devices]
    jobs: list[dict] = []
    slot = 0
    for pool in pool_docs:
        manifest.validate_pool_manifest(pool)
        for candidate in sorted(
            pool["candidates"], key=lambda c: (c["coverage_rank"], c["candidate_id"])
        ):
            device = devices[slot % len(devices)]
            slot += 1
            for fidelity in rotated_fidelities(
                candidate["candidate_id"], fidelities
            ):
                request = manifest.build_request(
                    experiment_id=experiment["experiment_id"],
                    pool_id=pool["pool_id"],
                    candidate_id=candidate["candidate_id"],
                    candidate_path=candidate["candidate_path"],
                    candidate_execution_revision=candidate[
                        "candidate_execution_revision"
                    ],
                    params=candidate["params"],
                    requested_train_seconds=fidelity,
                    purpose="matrix",
                    repeat_index=0,
                    seed=int(experiment.get("seed", 42)),
                    gpu_id=device["gpu_id"],
                    gpu_uuid=device["gpu_uuid"],
                    task_artifact_digest=task["task_artifact_digest"],
                )
                _write_request(jobs_dir, request)
                row = {
                    "job_id": request["job_id"],
                    "pool_id": pool["pool_id"],
                    "candidate_id": candidate["candidate_id"],
                    "requested_train_seconds": fidelity,
                    "gpu_id": device["gpu_id"],
                    "gpu_uuid": device["gpu_uuid"],
                }
                jobs.append(row)
                per_gpu[(slot - 1) % len(devices)].append(row)

    waves: list[list[str]] = []
    depth = max((len(queue) for queue in per_gpu), default=0)
    for index in range(depth):
        wave = [
            queue[index]["job_id"] for queue in per_gpu if index < len(queue)
        ]
        waves.append(wave)
    return _schedule_doc(experiment["experiment_id"], jobs, waves)


def plan_repeats(
    experiment: dict,
    pool: dict,
    candidate_id: str,
    *,
    purpose: str,
    fidelity: int,
    repeats: int,
    repeat_start: int = 0,
    evaluation_path: str = "adapter",
    jobs_dir: Path,
) -> dict:
    """Calibration/adjudication repeats: same request, new (purpose, repeat)."""
    manifest.validate_experiment(experiment)
    manifest.validate_pool_manifest(pool)
    if purpose == "matrix":
        raise RunnerError("plan-repeats is for calibration/adjudication only")
    devices = experiment["hardware"]["devices"]
    candidate = next(
        (c for c in pool["candidates"] if c["candidate_id"] == candidate_id),
        None,
    )
    if candidate is None:
        raise RunnerError(
            f"candidate {candidate_id!r} is not in pool {pool['pool_id']!r}"
        )
    device = devices[
        _stable_int("gpu-binding", candidate_id) % len(devices)
    ]
    jobs = []
    for repeat_index in range(repeat_start, repeat_start + repeats):
        request = manifest.build_request(
            experiment_id=experiment["experiment_id"],
            pool_id=pool["pool_id"],
            candidate_id=candidate_id,
            candidate_path=candidate["candidate_path"],
            candidate_execution_revision=candidate[
                "candidate_execution_revision"
            ],
            params=candidate["params"],
            requested_train_seconds=fidelity,
            purpose=purpose,
            repeat_index=repeat_index,
            seed=int(experiment.get("seed", 42)),
            gpu_id=device["gpu_id"],
            gpu_uuid=device["gpu_uuid"],
            task_artifact_digest=experiment["task"]["task_artifact_digest"],
            evaluation_path=evaluation_path,
        )
        _write_request(jobs_dir, request)
        jobs.append(
            {
                "job_id": request["job_id"],
                "pool_id": pool["pool_id"],
                "candidate_id": candidate_id,
                "requested_train_seconds": fidelity,
                "gpu_id": device["gpu_id"],
                "gpu_uuid": device["gpu_uuid"],
            }
        )
    # Repeats of the same request on the same GPU are strictly sequential.
    waves = [[job["job_id"]] for job in jobs]
    return _schedule_doc(experiment["experiment_id"], jobs, waves)


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


def resume_action(job_dir: Path) -> str:
    """Decide 'skip' or 'run' for one job directory.

    A terminal result.json with matching bindings skips. A result.json that
    exists but does not bind to the request is a hard error (a semantic field
    changed without a new job id — the frozen contract forbids reuse AND
    silent rerun). Anything less than a terminal result reruns from scratch;
    stale provisional/log files are removed here.
    """
    request = manifest.load_json(job_dir / manifest.REQUEST_FILENAME)
    manifest.validate_request(request)
    result_path = job_dir / manifest.RESULT_FILENAME
    if result_path.exists():
        result = manifest.load_json(result_path)
        errors = manifest.validate_result_against_request(result, request)
        if errors:
            raise RunnerError(
                f"{result_path} does not bind to its request: "
                + "; ".join(errors)
            )
        return "skip"
    for name in (
        manifest.PROVISIONAL_FILENAME,
        manifest.STDOUT_FILENAME,
        manifest.STDERR_FILENAME,
    ):
        stale = job_dir / name
        if stale.exists():
            stale.unlink()
    return "run"


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def _query_busy_gpus(gpu_uuids: set[str]) -> list[str]:
    """Read-only check that no other compute process holds the target GPUs."""
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RunnerError(f"nvidia-smi failed: {proc.stderr.strip()}")
    busy = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and parts[0] in gpu_uuids:
            busy.append(line.strip())
    return busy


def _finalize_result(
    job_dir: Path,
    request: dict,
    *,
    parent_status: str | None,
    exit_code: int | None,
    elapsed_accelerator_seconds: float,
) -> dict:
    """Terminal result.json = child provisional + parent-side facts.

    parent_status overrides for timeout; a missing/unreadable provisional is a
    crash. Never fabricates a score.
    """
    provisional_path = job_dir / manifest.PROVISIONAL_FILENAME
    provisional = None
    if provisional_path.exists():
        try:
            provisional = manifest.load_json(provisional_path)
        except (OSError, ValueError):
            provisional = None
    if parent_status == "timeout" or provisional is None:
        status = parent_status or "crash"
        result = {
            "schema_version": manifest.SCHEMA_VERSION,
            "job_id": request["job_id"],
            "status": status,
            "score": None,
            "metric": "val_bpb",
            "requested_train_seconds": request["requested_train_seconds"],
            "completed_train_seconds": None,
            "num_steps": None,
            "candidate_execution_revision": request[
                "candidate_execution_revision"
            ],
            "params_digest": request["params_digest"],
            "task_artifact_digest": request["task_artifact_digest"],
            "evaluation_path": request["evaluation_path"],
            "purpose": request["purpose"],
            "repeat_index": request["repeat_index"],
            "seed": request["seed"],
            "summary": {},
            "error": (
                "job exceeded the experiment timeout"
                if status == "timeout"
                else f"child exited without a provisional result (code {exit_code})"
            ),
        }
    else:
        result = dict(provisional)
    result["exit_code"] = exit_code
    result["elapsed_accelerator_seconds"] = elapsed_accelerator_seconds
    result["gpu"] = {
        "gpu_id": request["gpu_id"],
        "gpu_uuid": request["gpu_uuid"],
    }
    for name, key in (
        (manifest.STDOUT_FILENAME, "stdout_digest"),
        (manifest.STDERR_FILENAME, "stderr_digest"),
    ):
        path = job_dir / name
        result[key] = manifest.sha256_file(path) if path.exists() else None
    manifest.atomic_write_json(job_dir / manifest.RESULT_FILENAME, result)
    return result


def run_one_job(
    job_dir: Path,
    *,
    python_cmd: list[str],
    timeout_seconds: float,
) -> dict:
    """One GPU worker slot: spawn the child pinned to its request's GPU UUID,
    wait, kill the whole process group on timeout, finalize result.json.
    Elapsed is counted from worker start to child exit."""
    request = manifest.load_json(job_dir / manifest.REQUEST_FILENAME)
    started = time.monotonic()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = request["gpu_uuid"]
    with open(job_dir / manifest.STDOUT_FILENAME, "wb") as out, open(
        job_dir / manifest.STDERR_FILENAME, "wb"
    ) as err:
        proc = subprocess.Popen(
            [*python_cmd, str(CHILD_PATH), str(job_dir)],
            stdout=out,
            stderr=err,
            env=env,
            cwd=str(REPO_ROOT),
            start_new_session=True,
        )
        parent_status = None
        try:
            exit_code = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            parent_status = "timeout"
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            exit_code = proc.wait()
    elapsed = time.monotonic() - started
    return _finalize_result(
        job_dir,
        request,
        parent_status=parent_status,
        exit_code=exit_code,
        elapsed_accelerator_seconds=elapsed,
    )


def run_schedule(
    schedule: dict,
    jobs_dir: Path,
    *,
    python_cmd: list[str],
    timeout_seconds: float,
    check_gpus: bool = True,
    log=print,
) -> None:
    by_id = {job["job_id"]: job for job in schedule["jobs"]}
    for wave_index, wave in enumerate(schedule["waves"]):
        uuids = [by_id[job_id]["gpu_uuid"] for job_id in wave]
        if len(set(uuids)) != len(uuids):
            raise RunnerError(
                f"wave {wave_index} schedules two jobs on one GPU"
            )
        pending = []
        for job_id in wave:
            job_dir = jobs_dir / job_id
            if resume_action(job_dir) == "skip":
                log(f"wave {wave_index}: {job_id} already terminal, skipping")
                continue
            pending.append(job_id)
        if not pending:
            continue
        if check_gpus:
            busy = _query_busy_gpus(
                {by_id[job_id]["gpu_uuid"] for job_id in pending}
            )
            if busy:
                raise RunnerError(
                    "target GPUs are not idle: " + "; ".join(busy)
                )
        log(
            f"wave {wave_index}: launching {len(pending)} job(s): "
            + ", ".join(pending)
        )
        failures: list[str] = []
        results: dict[str, dict] = {}

        def worker(job_id: str) -> None:
            try:
                results[job_id] = run_one_job(
                    jobs_dir / job_id,
                    python_cmd=python_cmd,
                    timeout_seconds=timeout_seconds,
                )
            except BaseException as exc:  # noqa: BLE001 — barrier must see it
                failures.append(f"{job_id}: {type(exc).__name__}: {exc}")

        threads = [
            threading.Thread(target=worker, args=(job_id,), daemon=True)
            for job_id in pending
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            raise RunnerError(
                f"wave {wave_index} worker failures: " + "; ".join(failures)
            )
        for job_id in pending:
            result = results.get(job_id)
            if not manifest.is_terminal_result(result):
                raise RunnerError(
                    f"wave {wave_index}: {job_id} finished without a terminal "
                    "result"
                )
            log(
                f"wave {wave_index}: {job_id} -> {result['status']}"
                + (
                    f" score={result['score']:.6f}"
                    if result.get("score") is not None
                    else ""
                )
            )


# ---------------------------------------------------------------------------
# full-fidelity equivalence gate
# ---------------------------------------------------------------------------


def epsilon_noise(scores: list[float]) -> float:
    """90th percentile (nearest-rank) of pairwise |delta| over anchor repeats."""
    if len(scores) < 2:
        raise RunnerError("epsilon_noise needs at least two calibration scores")
    deltas = sorted(
        abs(a - b)
        for i, a in enumerate(scores)
        for b in scores[i + 1:]
    )
    import math

    rank = math.ceil(0.9 * len(deltas))
    return deltas[rank - 1]


def _gate_score(jobs_dir: Path, job_id: str) -> dict:
    job_dir = jobs_dir / job_id
    request = manifest.load_json(job_dir / manifest.REQUEST_FILENAME)
    result = manifest.load_json(job_dir / manifest.RESULT_FILENAME)
    errors = manifest.validate_result_against_request(result, request)
    if errors:
        raise RunnerError(f"{job_id}: " + "; ".join(errors))
    return result


def gate_verdict(
    jobs_dir: Path,
    calibration_job_ids: list[str],
    pairs: list[dict],
) -> dict:
    """Compare official evaluate_config vs adapter(300) per pool anchor.

    Each pair: {"pool_id", "official_job_id", "adapter_job_ids": [primary,
    optional recheck]}. A first-comparison excess triggers one adapter(300)
    recheck; both exceeding epsilon_noise blocks that pool. Summary contract
    and task digests must match — an implementation difference is never
    reinterpreted as training noise by widening the threshold.
    """
    calibration = [_gate_score(jobs_dir, job_id) for job_id in calibration_job_ids]
    bad = [r["job_id"] for r in calibration if r["status"] != "ok"]
    if bad:
        raise RunnerError(f"calibration jobs not ok: {bad}")
    eps = epsilon_noise([float(r["score"]) for r in calibration])

    pools = []
    for pair in pairs:
        official = _gate_score(jobs_dir, pair["official_job_id"])
        adapters = [
            _gate_score(jobs_dir, job_id)
            for job_id in pair["adapter_job_ids"]
        ]
        entry: dict = {
            "pool_id": pair.get("pool_id"),
            "official_job_id": pair["official_job_id"],
            "adapter_job_ids": list(pair["adapter_job_ids"]),
            "epsilon_noise": eps,
        }
        problems = []
        if official["status"] != "ok":
            problems.append(f"official job status {official['status']}")
        for adapter in adapters:
            if adapter["status"] != "ok":
                problems.append(
                    f"adapter job {adapter['job_id']} status {adapter['status']}"
                )
            for field in (
                "task_artifact_digest",
                "candidate_execution_revision",
                "params_digest",
            ):
                if adapter[field] != official[field]:
                    problems.append(f"{field} differs across the pair")
        if problems:
            entry.update({"status": "blocked", "problems": problems})
            pools.append(entry)
            continue
        deltas = [
            abs(float(adapter["score"]) - float(official["score"]))
            for adapter in adapters
        ]
        entry["deltas"] = deltas
        within = [delta <= eps for delta in deltas]
        if within[0]:
            entry["status"] = "pass"
        elif len(deltas) == 1:
            entry["status"] = "recheck_required"
        elif within[1]:
            entry["status"] = "pass"
        else:
            entry["status"] = "blocked"
            entry["problems"] = [
                f"both comparisons exceed epsilon_noise {eps:g}: {deltas}"
            ]
        pools.append(entry)
    return {
        "schema_version": manifest.SCHEMA_VERSION,
        "epsilon_noise": eps,
        "calibration_job_ids": list(calibration_job_ids),
        "calibration_scores": [float(r["score"]) for r in calibration],
        "pools": pools,
        "status": (
            "pass"
            if all(entry["status"] == "pass" for entry in pools)
            else "blocked"
            if any(entry["status"] == "blocked" for entry in pools)
            else "recheck_required"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_python_cmd(task_dir: str) -> list[str]:
    return ["uv", "--project", str(REPO_ROOT / task_dir), "run", "python"]


def probe_hardware() -> dict:
    """Mechanical facts for the experiment manifest: GPU model/UUIDs from
    nvidia-smi and the task artifact binding. The experimenter freezes these
    into experiment.json; nothing here is guessed by hand."""
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RunnerError(f"nvidia-smi failed: {proc.stderr.strip()}")
    devices = []
    names = set()
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            devices.append({"gpu_id": int(parts[0]), "gpu_uuid": parts[1]})
            names.add(parts[2])
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    ).stdout.strip()
    return {
        "hardware": {"gpu_model": "/".join(sorted(names)), "devices": devices},
        "task_artifact_binding": manifest.task_artifact_binding(
            REPO_ROOT / "tasks" / "autoresearch-baseline", commit or None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="build matrix requests + schedule")
    plan.add_argument("--experiment", required=True)
    plan.add_argument("--pool", action="append", required=True)
    plan.add_argument("--jobs-dir", required=True)
    plan.add_argument("--schedule", required=True)

    repeats = sub.add_parser("plan-repeats", help="calibration/adjudication jobs")
    repeats.add_argument("--experiment", required=True)
    repeats.add_argument("--pool", required=True)
    repeats.add_argument("--candidate", required=True)
    repeats.add_argument("--purpose", required=True,
                         choices=["calibration", "adjudication"])
    repeats.add_argument("--fidelity", type=int, default=300)
    repeats.add_argument("--repeats", type=int, required=True)
    repeats.add_argument("--repeat-start", type=int, default=0)
    repeats.add_argument("--evaluation-path", default="adapter",
                         choices=list(manifest.EVALUATION_PATHS))
    repeats.add_argument("--jobs-dir", required=True)
    repeats.add_argument("--schedule", required=True)

    run = sub.add_parser("run", help="execute a schedule wave by wave")
    run.add_argument("--schedule", required=True)
    run.add_argument("--jobs-dir", required=True)
    run.add_argument("--task-dir", default="tasks/autoresearch-baseline")
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    run.add_argument("--no-gpu-check", action="store_true",
                     help="skip the nvidia-smi idle check (tests only)")

    sub.add_parser("probe", help="print GPU devices + task artifact binding")

    gate = sub.add_parser("gate", help="full-fidelity equivalence verdict")
    gate.add_argument("--jobs-dir", required=True)
    gate.add_argument("--calibration-jobs", required=True,
                      help="comma-separated calibration job ids")
    gate.add_argument("--pairs", required=True,
                      help="JSON file: [{pool_id, official_job_id, adapter_job_ids}]")
    gate.add_argument("--output", required=True)

    args = parser.parse_args(argv)

    if args.command == "plan":
        experiment = manifest.load_json(Path(args.experiment))
        pools = [manifest.load_json(Path(p)) for p in args.pool]
        schedule = plan_matrix(experiment, pools, Path(args.jobs_dir))
        manifest.freeze_immutable(Path(args.schedule), schedule)
        print(
            f"planned {len(schedule['jobs'])} jobs in "
            f"{len(schedule['waves'])} waves; jobs_digest {schedule['jobs_digest']}"
        )
        return 0
    if args.command == "plan-repeats":
        experiment = manifest.load_json(Path(args.experiment))
        pool = manifest.load_json(Path(args.pool))
        schedule = plan_repeats(
            experiment,
            pool,
            args.candidate,
            purpose=args.purpose,
            fidelity=args.fidelity,
            repeats=args.repeats,
            repeat_start=args.repeat_start,
            evaluation_path=args.evaluation_path,
            jobs_dir=Path(args.jobs_dir),
        )
        manifest.freeze_immutable(Path(args.schedule), schedule)
        print(f"planned {len(schedule['jobs'])} {args.purpose} job(s)")
        return 0
    if args.command == "run":
        schedule = manifest.load_json(Path(args.schedule))
        run_schedule(
            schedule,
            Path(args.jobs_dir),
            python_cmd=_default_python_cmd(args.task_dir),
            timeout_seconds=args.timeout,
            check_gpus=not args.no_gpu_check,
        )
        print("schedule complete: every job terminal")
        return 0
    if args.command == "probe":
        import json

        print(json.dumps(probe_hardware(), indent=1))
        return 0
    if args.command == "gate":
        verdict = gate_verdict(
            Path(args.jobs_dir),
            [s for s in args.calibration_jobs.split(",") if s],
            manifest.load_json(Path(args.pairs)),
        )
        manifest.atomic_write_json(Path(args.output), verdict)
        print(f"gate status: {verdict['status']}")
        return 0 if verdict["status"] == "pass" else 1
    raise RunnerError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
