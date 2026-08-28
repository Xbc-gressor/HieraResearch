"""Multifidelity deterministic tools: quality freeze + Recall@2 analyzer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "multifidelity"))

import analyze_recall  # noqa: E402
import manifest  # noqa: E402
import qualify_pools  # noqa: E402

# ---------------------------------------------------------------------------
# qualification freeze
# ---------------------------------------------------------------------------

READS_BUDGET_TRAIN = (
    "PARAM_SCHEMA = {}\nSEARCH_SPACE = {}\nBASE_PARAMS = {}\n"
    "def make_model(env, params):\n"
    "    budget = env.train_budget_seconds\n"
    "    return budget\n"
)
IGNORES_BUDGET_TRAIN = (
    "PARAM_SCHEMA = {}\nSEARCH_SPACE = {}\nBASE_PARAMS = {}\n"
    "def make_model(env, params):\n"
    "    return 300\n"
)
SUMMARY_OK = "---\nval_bpb:          1.050000\ntraining_seconds: 300.1\n"
SUMMARY_MISSING = "---\nval_bpb:          1.050000\n"


def _bank_entry(tmp_path, index, score, *, reads_budget=True, summary=SUMMARY_OK):
    candidate_dir = tmp_path / f"cand-{index:02d}"
    candidate_dir.mkdir()
    (candidate_dir / "train.py").write_text(
        READS_BUDGET_TRAIN if reads_budget else IGNORES_BUDGET_TRAIN
    )
    (candidate_dir / "prepare.py").write_text("# frozen evaluator copy\n")
    stdout_path = candidate_dir / "qualification_stdout.log"
    stdout_path.write_text(summary)
    return {
        "candidate_id": f"cand-{index:02d}",
        "point_id": f"point-{index:02d}",
        "op": "improve",
        "parents": ["007"],
        "coverage_rank": index,
        "judge_label": f"L{index}",
        "candidate_path": str(candidate_dir / "train.py"),
        "candidate_execution_revision": {"train_sha256": f"t{index}"},
        "params": {"x": float(index)},
        "qualification": {
            "score": score,
            "result_digest": f"sha256:q{index}",
        },
        "qualification_stdout_path": str(stdout_path),
    }


def _bundle(tmp_path, cohort, scores, entry_kwargs_by_index=None):
    entry_kwargs_by_index = entry_kwargs_by_index or {}
    bank = [
        _bank_entry(tmp_path, index, score, **entry_kwargs_by_index.get(index, {}))
        for index, score in enumerate(scores)
    ]
    return {
        "schema_version": 1,
        "pool_id": "pool-t",
        "cohort": cohort,
        "source_prefix": {"run_dir": "hist", "cutoff": "012"},
        "production_pool_digest": "sha256:pool",
        "judge_bundle_digest": "sha256:judge",
        "anchors": [
            {"candidate_id": "carrier-a", "qualification_score": 1.0,
             "result_digest": "sha256:a"},
            {"candidate_id": "carrier-b", "qualification_score": None,
             "result_digest": "sha256:b"},
        ],
        "bank": bank,
    }


def test_competitive_freeze_selects_by_coverage_not_score(tmp_path):
    # competitive threshold: score <= 1.10 * anchor(1.0)
    # ranks 1 (crash), 3 (borderline), 5 (severe) break the coverage run;
    # rank 11 is the BEST score but must not displace earlier coverage ranks.
    scores = [1.05, None, 1.06, 1.20, 1.07, 1.50, 1.08, 1.09, 1.02, 1.30, 1.01, 0.90]
    doc = qualify_pools.freeze_pool(_bundle(tmp_path, "competitive", scores))
    assert doc["status"] == "frozen"
    selected = [c["candidate_id"] for c in doc["candidates"]]
    assert selected == [
        "cand-00", "cand-02", "cand-04", "cand-06", "cand-07", "cand-08"
    ]
    tiers = {c["candidate_id"]: c["quality"]["tier"] for c in doc["candidates"]}
    assert set(tiers.values()) == {"competitive"}
    # anchor is the strongest finite carrier
    assert doc["anchor"]["candidate_id"] == "carrier-a"


def test_insufficient_competitive_bank_is_rejected_wholesale(tmp_path):
    scores = [1.05, 1.06, 1.07, 1.08, 1.09, None, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    doc = qualify_pools.freeze_pool(_bundle(tmp_path, "competitive", scores))
    assert doc["status"] == "insufficient_competitive_bank"
    assert doc["competitive_count"] == 5
    assert doc["tiers"]["severe"] == 7


def test_natural_pool_keeps_severe_members_with_tier_annotations(tmp_path):
    scores = [1.05, None, 1.06, 1.20, 1.07, 1.50]
    doc = qualify_pools.freeze_pool(_bundle(tmp_path, "natural", scores))
    assert doc["status"] == "frozen"
    tiers = [c["quality"]["tier"] for c in doc["candidates"]]
    assert tiers == [
        "competitive", "severe", "competitive", "borderline", "competitive",
        "severe",
    ]


def test_freeze_checks_are_mechanical(tmp_path):
    scores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    doc = qualify_pools.freeze_pool(
        _bundle(
            tmp_path, "natural", scores,
            entry_kwargs_by_index={
                1: {"reads_budget": False},
                2: {"summary": SUMMARY_MISSING},
            },
        )
    )
    checks = {c["candidate_id"]: c["freeze_checks"] for c in doc["candidates"]}
    assert checks["cand-00"] == {
        "reads_env_train_budget_seconds": True,
        "summary_has_training_seconds": True,
    }
    assert checks["cand-01"]["reads_env_train_budget_seconds"] is False
    assert checks["cand-02"]["summary_has_training_seconds"] is False


# ---------------------------------------------------------------------------
# recall analyzer fixture
# ---------------------------------------------------------------------------

EXPERIMENT = {
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
        "devices": [{"gpu_id": 0, "gpu_uuid": "GPU-toy-0"}],
    },
    "seed": 42,
}

# pool A: consensus judge; a4/a5 severe; H300 a4 crashes.
POOL_A_SCORES = {
    #        30      60      120     300
    "a0": [0.95, 0.80, 0.72, 1.04],
    "a1": [None, 0.81, 0.70, 1.01],  # 30s crash -> +inf
    "a2": [0.90, 0.90, 0.71, 1.02],
    "a3": [0.91, 0.91, 0.90, 1.03],
    "a4": [1.50, 1.00, 1.50, None],  # H300 crash -> +inf oracle
    "a5": [1.60, 1.00, 1.60, 1.50],
}
# pool B: judge failed -> coverage fallback; b4 severe.
POOL_B_SCORES = {
    "b0": [0.70, 0.50, 0.45, 1.10],
    "b1": [0.80, 0.55, 0.90, 1.20],
    "b2": [0.90, 0.90, 0.91, 1.30],
    "b3": [0.50, 0.95, 0.40, 1.00],
    "b4": [0.55, 0.96, 0.92, 1.01],
    "b5": [0.60, 0.97, 0.50, 1.05],
}


def _pool_manifest(pool_id, prefix, severe, truncation=()):
    candidates = []
    for index in range(6):
        cid = f"{prefix}{index}"
        candidates.append(
            {
                "candidate_id": cid,
                "coverage_rank": index,
                "point_id": f"point-{cid}",
                "op": "improve",
                "parents": [],
                "judge_label": f"{prefix.upper()}{index}",
                "candidate_path": f"/frozen/{cid}/train.py",
                "candidate_execution_revision": {"train_sha256": cid},
                "params": {"x": float(index)},
                "params_digest": manifest.params_digest({"x": float(index)}),
                "quality": {
                    "tier": "severe" if cid in severe else "competitive",
                    "qualification_score": 1.05,
                    "rule": "fixture",
                },
                "freeze_checks": {
                    "reads_env_train_budget_seconds": cid not in truncation,
                    "summary_has_training_seconds": True,
                },
            }
        )
    return {
        "schema_version": 1,
        "status": "frozen",
        "pool_id": pool_id,
        "cohort": "natural",
        "anchor": {"candidate_id": "carrier", "qualification_score": 1.0,
                   "result_digest": "sha256:a"},
        "candidates": candidates,
    }


def _write_job(jobs_dir, pool, cid, fidelity, score, *, purpose="matrix",
               repeat_index=0, gpu_uuid="GPU-toy-0"):
    candidate = next(
        c for c in pool["candidates"] if c["candidate_id"] == cid
    )
    request = manifest.build_request(
        experiment_id="exp-t",
        pool_id=pool["pool_id"],
        candidate_id=cid,
        candidate_path=candidate["candidate_path"],
        candidate_execution_revision=candidate["candidate_execution_revision"],
        params=candidate["params"],
        requested_train_seconds=fidelity,
        purpose=purpose,
        repeat_index=repeat_index,
        seed=42,
        gpu_id=0,
        gpu_uuid=gpu_uuid,
        task_artifact_digest="sha256:task",
    )
    result = {
        "schema_version": 1,
        "job_id": request["job_id"],
        "status": "ok" if score is not None else "crash",
        "score": score,
        "metric": "val_bpb",
        "requested_train_seconds": fidelity,
        "candidate_execution_revision": request["candidate_execution_revision"],
        "params_digest": request["params_digest"],
        "task_artifact_digest": request["task_artifact_digest"],
        "evaluation_path": request["evaluation_path"],
        "purpose": purpose,
        "repeat_index": repeat_index,
        "seed": 42,
        "elapsed_accelerator_seconds": float(fidelity + 10),
        "summary": {},
    }
    job_dir = jobs_dir / request["job_id"]
    job_dir.mkdir(parents=True)
    manifest.atomic_write_json(job_dir / manifest.REQUEST_FILENAME, request)
    manifest.atomic_write_json(job_dir / manifest.RESULT_FILENAME, result)


def build_fixture(tmp_path):
    pool_a = _pool_manifest("pA", "a", severe={"a4", "a5"}, truncation={"a3"})
    pool_b = _pool_manifest("pB", "b", severe={"b4"})
    jobs_dir = tmp_path / "jobs"
    for pool, table in ((pool_a, POOL_A_SCORES), (pool_b, POOL_B_SCORES)):
        for cid, scores in table.items():
            for fidelity, score in zip((30, 60, 120, 300), scores):
                _write_job(jobs_dir, pool, cid, fidelity, score)
    # adjudication for a0: matrix 1.04 + repeats 1.00/0.99 -> median 1.00,
    # which restores a0 into the oracle top-2.
    _write_job(jobs_dir, pool_a, "a0", 300, 1.00, purpose="adjudication",
               repeat_index=1)
    _write_job(jobs_dir, pool_a, "a0", 300, 0.99, purpose="adjudication",
               repeat_index=2)

    judges_dir = tmp_path / "judges"
    a_dir = judges_dir / "pA"
    manifest.atomic_write_json(a_dir / "judge.json", {
        "aggregation": {"path": "consensus", "slate": ["A0", "A2"],
                        "reason": None},
        "judge_cost": {"session_ids": {}, "models": {}},
    })
    manifest.atomic_write_json(a_dir / "regular-0.json", {
        "stage": "regular-0", "status": "valid",
        "ranking": ["A0", "A2", "A1", "A3", "A4", "A5"],
    })
    manifest.atomic_write_json(a_dir / "regular-1.json", {
        "stage": "regular-1", "status": "valid",
        "ranking": ["A2", "A0", "A1", "A5", "A4", "A3"],
    })
    b_dir = judges_dir / "pB"
    manifest.atomic_write_json(b_dir / "judge.json", {
        "aggregation": {"path": "coverage_fallback", "slate": ["B0", "B1"],
                        "reason": "regular judge failed: regular-0"},
        "judge_cost": {"session_ids": {}, "models": {}},
    })
    return pool_a, pool_b, jobs_dir, judges_dir


def analyze_fixture(tmp_path, **kwargs):
    pool_a, pool_b, jobs_dir, judges_dir = build_fixture(tmp_path)
    return analyze_recall.analyze(
        EXPERIMENT, [pool_a, pool_b], jobs_dir, judges_dir,
        layer=kwargs.pop("layer", "N"), **kwargs,
    )


def test_recall_fixture_pool_details(tmp_path):
    report = analyze_fixture(tmp_path)
    by_id = {p["pool_id"]: p for p in report["pools"]}
    pa, pb = by_id["pA"], by_id["pB"]

    # adjudication median replaced the matrix H300 for a0
    assert pa["oracle_scores"]["a0"] == 1.00
    # crash maps to +inf
    assert pa["oracle_scores"]["a4"] == float("inf")
    assert pa["low_fidelity_scores"]["30"]["a1"] == float("inf")
    # oracle after adjudication: [a0, a1]
    assert pa["selections"]["all"]["oracle_top2"] == ["a0", "a1"]

    # lower-is-better F2 top-2
    assert pa["selections"]["all"]["F2(30)"] == ["a2", "a3"]
    assert pa["selections"]["all"]["F2(60)"] == ["a0", "a1"]
    assert pa["recall"]["all"]["F2(30)"] == 0.0
    assert pa["recall"]["all"]["F2(60)"] == 1.0
    assert pa["recall"]["all"]["F2(120)"] == 0.5

    # J2 = production slate; consensus tie broke by coverage
    assert pa["selections"]["all"]["J2"] == ["a0", "a2"]
    assert pa["recall"]["all"]["J2"] == 0.5
    # J4 = union {A0,A2} + fill by mean regular rank (a1), tie -> coverage (a3)
    assert pa["j4_shortlist"] == ["a0", "a2", "a1", "a3"]
    assert pa["recall"]["all"]["J4->F2(60)"] == 1.0
    assert pa["j4_retention"]["all"] == 1.0

    # judge failure: J2 falls back to coverage order, J4 = coverage top-4
    assert pb["selections"]["all"]["J2"] == ["b0", "b1"]
    assert pb["j4_shortlist"] == ["b0", "b1", "b2", "b3"]
    assert pb["j4_shortlist_derivation"]["path"] == "coverage_fallback"
    assert pb["selections"]["all"]["oracle_top2"] == ["b3", "b4"]
    assert pb["recall"]["all"]["J2"] == 0.0
    assert pb["recall"]["all"]["F2(30)"] == 1.0
    assert pb["j4_retention"]["all"] == 0.5

    # quality restriction: b4 is severe, so the competitive oracle differs
    assert pb["selections"]["competitive"]["oracle_top2"] == ["b3", "b5"]
    assert pb["recall"]["competitive"]["F2(30)"] == 1.0

    # truncation-semantics annotation from freeze_checks
    assert pa["truncation_semantics_candidates"] == ["a3"]

    # macro over pools, not candidates
    assert report["macro"]["all"]["F2(30)"] == 0.5
    assert report["macro"]["competitive"]["J2"] == 0.25


def test_n_layer_verdict_freezes_p_star_mechanically(tmp_path):
    report = analyze_fixture(tmp_path)
    verdict = report["verdict"]
    assert verdict["decision"] == "proceed_to_C"
    # recall ties at 0.5 resolve by lower actual accelerator seconds
    assert verdict["frozen_selector"]["selector"] == "F2(30)"
    assert verdict["j2_recall_competitive"] == 0.25


def test_c_layer_gate(tmp_path):
    n_report = analyze_fixture(tmp_path)
    passing = analyze_fixture(
        tmp_path / "c1", layer="C", frozen_selector="F2(30)",
        n_report=n_report, gate_status="pass",
    )
    assert passing["verdict"]["decision"] == "enter_production_e2e"
    assert passing["verdict"]["checks"]["c_competitive_margin"]["pass"]

    # hybrid P*: pool B misses b3, which IS in the shortlist -> the failure is
    # not explained by judge top-4 retention -> gate fails
    hybrid = analyze_fixture(
        tmp_path / "c2", layer="C", frozen_selector="J4->F2(60)",
        n_report=n_report, gate_status="pass",
    )
    assert hybrid["verdict"]["decision"] == "not_supported"
    assert hybrid["verdict"]["checks"][
        "hybrid_failures_explained_by_shortlist"
    ]["unexplained"]


def test_duplicate_matrix_result_is_rejected(tmp_path):
    pool_a, pool_b, jobs_dir, judges_dir = build_fixture(tmp_path)
    # same (candidate, fidelity) under a different GPU UUID -> new job id,
    # second matrix observation -> analyzer must refuse
    _write_job(jobs_dir, pool_a, "a0", 30, 0.99, gpu_uuid="GPU-toy-1")
    with pytest.raises(analyze_recall.AnalysisError):
        analyze_recall.analyze(
            EXPERIMENT, [pool_a, pool_b], jobs_dir, judges_dir, layer="N",
        )
