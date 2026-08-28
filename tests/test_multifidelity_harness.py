"""Multifidelity harness: job-id digest sensitivity + job-level resume."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "multifidelity"))

import manifest  # noqa: E402
import run_matrix  # noqa: E402


def base_request(**overrides):
    kwargs = dict(
        experiment_id="exp-t",
        pool_id="pool-a",
        candidate_id="cand-1",
        candidate_path="/tmp/cand/train.py",
        candidate_execution_revision={"train_sha256": "aa", "prepare_sha256": "bb"},
        params={"x": 1.0},
        requested_train_seconds=60,
        purpose="matrix",
        repeat_index=0,
        seed=42,
        gpu_id=0,
        gpu_uuid="GPU-uuid-0",
        task_artifact_digest="sha256:task",
    )
    kwargs.update(overrides)
    return manifest.build_request(**kwargs)


SEMANTIC_PERTURBATIONS = {
    "candidate_id": "cand-2",
    "candidate_execution_revision": {"train_sha256": "cc", "prepare_sha256": "bb"},
    "params": {"x": 2.0},
    "requested_train_seconds": 120,
    "purpose": "calibration",
    "repeat_index": 1,
    "seed": 43,
    "gpu_uuid": "GPU-uuid-1",
    "task_artifact_digest": "sha256:other",
    "pool_id": "pool-b",
    "experiment_id": "exp-u",
    "evaluation_path": "official",
}


def test_job_id_binds_every_semantic_field():
    base = base_request()
    for field, value in SEMANTIC_PERTURBATIONS.items():
        changed = base_request(**{field: value})
        assert changed["job_id"] != base["job_id"], field


def test_job_id_ignores_gpu_index_but_not_uuid():
    # The UUID is the physical binding; the index is a display fact.
    assert base_request(gpu_id=3)["job_id"] == base_request()["job_id"]


def terminal_result(request, *, status="ok", score=1.0):
    return {
        "schema_version": manifest.SCHEMA_VERSION,
        "job_id": request["job_id"],
        "status": status,
        "score": score if status == "ok" else None,
        "metric": "val_bpb",
        "requested_train_seconds": request["requested_train_seconds"],
        "candidate_execution_revision": request["candidate_execution_revision"],
        "params_digest": request["params_digest"],
        "task_artifact_digest": request["task_artifact_digest"],
        "evaluation_path": request["evaluation_path"],
        "purpose": request["purpose"],
        "repeat_index": request["repeat_index"],
        "seed": request["seed"],
        "elapsed_accelerator_seconds": 12.0,
        "summary": {},
    }


def test_old_result_rejected_after_any_semantic_change():
    old = terminal_result(base_request())
    for field, value in SEMANTIC_PERTURBATIONS.items():
        changed = base_request(**{field: value})
        assert manifest.validate_result_against_request(old, changed), field


def test_non_terminal_and_non_finite_results_rejected():
    request = base_request()
    running = terminal_result(request)
    running["status"] = "running"
    assert manifest.validate_result_against_request(running, request)
    bad = terminal_result(request)
    bad["score"] = None
    assert manifest.validate_result_against_request(bad, request)


def _job_dir(tmp_path, request):
    job_dir = tmp_path / request["job_id"]
    job_dir.mkdir()
    manifest.atomic_write_json(job_dir / manifest.REQUEST_FILENAME, request)
    return job_dir


def test_resume_skips_terminal_job(tmp_path):
    request = base_request()
    job_dir = _job_dir(tmp_path, request)
    manifest.atomic_write_json(
        job_dir / manifest.RESULT_FILENAME, terminal_result(request)
    )
    assert run_matrix.resume_action(job_dir) == "skip"
    # crash/timeout are terminal observations too
    crashed = terminal_result(request, status="crash")
    (job_dir / manifest.RESULT_FILENAME).unlink()
    manifest.atomic_write_json(job_dir / manifest.RESULT_FILENAME, crashed)
    assert run_matrix.resume_action(job_dir) == "skip"


def test_resume_reruns_provisional_job_from_scratch(tmp_path):
    request = base_request()
    job_dir = _job_dir(tmp_path, request)
    manifest.atomic_write_json(
        job_dir / manifest.PROVISIONAL_FILENAME, {"status": "ok"}
    )
    (job_dir / manifest.STDOUT_FILENAME).write_text("partial")
    assert run_matrix.resume_action(job_dir) == "run"
    # stale provisional artifacts are removed before the rerun
    assert not (job_dir / manifest.PROVISIONAL_FILENAME).exists()
    assert not (job_dir / manifest.STDOUT_FILENAME).exists()


def test_resume_refuses_result_that_does_not_bind(tmp_path):
    request = base_request()
    job_dir = _job_dir(tmp_path, request)
    foreign = terminal_result(base_request(params={"x": 9.0}))
    manifest.atomic_write_json(job_dir / manifest.RESULT_FILENAME, foreign)
    with pytest.raises(run_matrix.RunnerError):
        run_matrix.resume_action(job_dir)


def _toy_experiment():
    return {
        "schema_version": 1,
        "experiment_id": "exp-t",
        "task": {
            "name": "autoresearch-baseline",
            "metric": "val_bpb",
            "direction": "min",
            "full_train_seconds": 300,
            "fidelity_semantics": "independent_compressed_schedule",
            "task_artifact_digest": "sha256:task",
        },
        "fidelities": [30, 60, 120, 300],
        "hardware": {
            "gpu_model": "toy",
            "devices": [
                {"gpu_id": 0, "gpu_uuid": "GPU-0"},
                {"gpu_id": 1, "gpu_uuid": "GPU-1"},
            ],
        },
        "seed": 42,
    }


def _toy_pool():
    candidates = []
    for index in range(6):
        params = {"x": float(index)}
        candidates.append(
            {
                "candidate_id": f"c{index}",
                "coverage_rank": index,
                "point_id": f"p{index}",
                "op": "improve",
                "parents": [],
                "judge_label": f"L{index}",
                "candidate_path": f"/frozen/c{index}/train.py",
                "candidate_execution_revision": {"train_sha256": f"s{index}"},
                "params": params,
                "params_digest": manifest.params_digest(params),
                "quality": {"tier": "competitive"},
                "freeze_checks": {"reads_env_train_budget_seconds": True,
                                  "summary_has_training_seconds": True},
            }
        )
    return {
        "schema_version": 1,
        "pool_id": "pool-a",
        "cohort": "natural",
        "anchor": {"candidate_id": "carrier", "qualification_score": 1.0},
        "candidates": candidates,
    }


def test_plan_matrix_schedule_invariants(tmp_path):
    experiment, pool = _toy_experiment(), _toy_pool()
    jobs_dir = tmp_path / "jobs"
    schedule = run_matrix.plan_matrix(experiment, [pool], jobs_dir)
    assert len(schedule["jobs"]) == 6 * 4
    by_id = {job["job_id"]: job for job in schedule["jobs"]}
    # every request landed on disk
    for job_id in by_id:
        assert (jobs_dir / job_id / manifest.REQUEST_FILENAME).exists()
    # one candidate -> one physical GPU across all four fidelities
    gpus_per_candidate = {}
    for job in schedule["jobs"]:
        gpus_per_candidate.setdefault(job["candidate_id"], set()).add(
            job["gpu_uuid"]
        )
    assert all(len(gpus) == 1 for gpus in gpus_per_candidate.values())
    # at most one job per GPU per wave
    for wave in schedule["waves"]:
        uuids = [by_id[job_id]["gpu_uuid"] for job_id in wave]
        assert len(set(uuids)) == len(uuids)
    assert sum(len(wave) for wave in schedule["waves"]) == 24
    # fidelity order is rotated per candidate, not fixed
    orders = {
        cid: [
            job["requested_train_seconds"]
            for job in schedule["jobs"]
            if job["candidate_id"] == cid
        ]
        for cid in gpus_per_candidate
    }
    assert len({tuple(order) for order in orders.values()}) > 1
    # replanning is deterministic: identical schedule, no reshuffle
    assert run_matrix.plan_matrix(experiment, [pool], jobs_dir) == schedule


def test_frozen_artifacts_are_immutable(tmp_path):
    path = tmp_path / "pool.json"
    doc = {"pool_id": "p", "candidates": []}
    assert manifest.freeze_immutable(path, doc) is True
    assert manifest.freeze_immutable(path, dict(doc)) is False  # identical: no-op
    with pytest.raises(manifest.ManifestError):
        manifest.freeze_immutable(path, {"pool_id": "p", "candidates": [1]})
