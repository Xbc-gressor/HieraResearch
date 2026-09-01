"""Tests for tools/import_candidate.py (rewrite-operator run bootstrap)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import import_candidate  # noqa: E402


TASK = "toy"
TRAIN_SRC = '''\
"""Toy candidate."""

BASE_PARAMS = {
    "x": 1.0,
    "depth": 3,
}
'''
TUNE_REPORT = {
    "inner_policy": "legacy",
    "phase_a": {"status": "ok", "best_warm_score": 1.1, "k_evaluated": 2},
    "phase_c": {
        "stages": [
            {"method": "grid", "status": "rejected", "trials": []},
            {"method": "bo", "status": "ok", "trials": [{"params": {"x": 2.0}, "score": 0.95}]},
        ]
    },
    "final_best_params": {"x": 1.0, "depth": 3},
    "final_best_score": 0.9,
}
EXPERIENCE = {"generation": 2, "lessons": ["mirror reflection beats clipping"]}


def _make_repo(root: Path) -> None:
    task_dir = root / "tasks" / TASK
    task_dir.mkdir(parents=True)
    (task_dir / "prepare.py").write_text("# fixed evaluation surface\n")


def _record(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "idea": f"idea {run_id}",
        "change": f"change {run_id}",
        "semantic_point": {"assignments": [{"dimension_id": "dim-a"}]},
        "final_best_score": 0.9,
        "best_warm_score": 1.1,
    }


def _make_source_run(root: Path, run_ids=("004",)) -> Path:
    source = root / "runs" / TASK / "src-run"
    for run_id in run_ids:
        candidate = source / "candidates" / run_id
        candidate.mkdir(parents=True)
        (candidate / "train.py").write_text(TRAIN_SRC)
        (candidate / "_search_space.json").write_text(json.dumps({"x": ["float", 0.1, 10.0]}))
        (candidate / "_warm_configs.json").write_text(json.dumps([{"params": {"x": 1.0}}]))
        (candidate / "tune_report.json").write_text(json.dumps(TUNE_REPORT))
        (candidate / "_candidate_brief.json").write_text(json.dumps({"run_id": run_id}))
        (candidate / "_failures").mkdir()
        (candidate / "_failures" / "f1.json").write_text(json.dumps({"error": "boom"}))
        (candidate / "_traces").mkdir()
        (candidate / "_traces" / "attempt-1.log").write_text("trace body\n")
    ledger = {
        "task": TASK,
        "records": [_record(run_id) for run_id in run_ids],
        "experience": EXPERIENCE,
    }
    (source / "ledger.json").write_text(json.dumps(ledger))
    (source / "background.md").write_text("# background\n")
    (source / "background_retrieval.json").write_text(json.dumps({"results": []}))
    return source


def _make_target(root: Path, task: str = TASK) -> Path:
    target = root / "runs" / task / "dst-run"
    target.mkdir(parents=True)
    return target


def test_import_copies_candidate_and_renames_traces(tmp_path) -> None:
    _make_repo(tmp_path)
    source = _make_source_run(tmp_path)
    target = _make_target(tmp_path)

    result = import_candidate.import_candidate(source, "004", target, repo_root=tmp_path)

    candidate = target / "candidates" / "004"
    assert result == {"candidate_dir": str(candidate)}
    src_candidate = source / "candidates" / "004"
    for name in (
        "train.py",
        "_search_space.json",
        "_warm_configs.json",
        "tune_report.json",
        "_candidate_brief.json",
    ):
        assert (candidate / name).read_text() == (src_candidate / name).read_text()
    assert (candidate / "_failures" / "f1.json").is_file()
    assert (candidate / "_traces_src" / "attempt-1.log").read_text() == "trace body\n"
    assert not (candidate / "_traces").exists()
    prepare = tmp_path / "tasks" / TASK / "prepare.py"
    assert (candidate / "prepare.py").read_text() == prepare.read_text()
    # run-level artifacts seed without the flag too; experience does not
    assert (target / "background.md").read_text() == "# background\n"
    assert not (target / "experience.seed.json").exists()


def test_import_rejects_task_mismatch(tmp_path) -> None:
    _make_repo(tmp_path)
    source = _make_source_run(tmp_path)
    target = _make_target(tmp_path, task="other-task")

    with pytest.raises(SystemExit):
        import_candidate.import_candidate(source, "004", target, repo_root=tmp_path)

    assert not (target / "candidates").exists()


def test_import_rejects_non_literal_base_params(tmp_path) -> None:
    _make_repo(tmp_path)
    source = _make_source_run(tmp_path)
    target = _make_target(tmp_path)
    train = source / "candidates" / "004" / "train.py"

    train.write_text("import random\nBASE_PARAMS = {'x': random.random()}\n")
    with pytest.raises(SystemExit):
        import_candidate.import_candidate(source, "004", target, repo_root=tmp_path)

    train.write_text("PARAMS = {'x': 1.0}\n")
    with pytest.raises(SystemExit):
        import_candidate.import_candidate(source, "004", target, repo_root=tmp_path)

    assert not (target / "candidates").exists()


def test_import_rejects_candidate_without_finite_baseline(tmp_path) -> None:
    _make_repo(tmp_path)
    source = _make_source_run(tmp_path)
    ledger_path = source / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["records"][0]["final_best_score"] = None
    ledger_path.write_text(json.dumps(ledger))
    target = _make_target(tmp_path)

    # no reference score to adjudicate against: dead on arrival, refuse early
    with pytest.raises(SystemExit):
        import_candidate.import_candidate(source, "004", target, repo_root=tmp_path)

    assert not (target / "candidates").exists()


def test_import_manifest_fields(tmp_path) -> None:
    _make_repo(tmp_path)
    source = _make_source_run(tmp_path)
    target = _make_target(tmp_path)

    import_candidate.import_candidate(source, "004", target, repo_root=tmp_path)

    manifest = json.loads((target / "candidates" / "004" / "_import.json").read_text())
    assert set(manifest) == {
        "source",
        "baseline_score",
        "warm_to_tuned_delta",
        "idea",
        "change",
        "semantic_point",
        "tune_summary",
    }
    assert manifest["source"] == str(source)
    assert manifest["baseline_score"] == 0.9
    assert manifest["warm_to_tuned_delta"] == pytest.approx(0.2)
    assert manifest["idea"] == "idea 004"
    assert manifest["change"] == "change 004"
    assert manifest["semantic_point"] == {"assignments": [{"dimension_id": "dim-a"}]}
    summary = manifest["tune_summary"]
    assert summary["final_best_score"] == 0.9
    assert summary["final_best_params"] == {"x": 1.0, "depth": 3}
    assert summary["phase_a"]["best_warm_score"] == 1.1
    assert summary["phase_c"]["stages"] == [
        {"method": "grid", "status": "rejected"},
        {"method": "bo", "status": "ok"},
    ]


def test_import_manifest_nulls_when_sources_missing(tmp_path) -> None:
    _make_repo(tmp_path)
    source = _make_source_run(tmp_path)
    candidate = source / "candidates" / "004"
    (candidate / "tune_report.json").unlink()
    ledger_path = source / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    del ledger["records"][0]["best_warm_score"]
    ledger_path.write_text(json.dumps(ledger))
    target = _make_target(tmp_path)

    import_candidate.import_candidate(source, "004", target, repo_root=tmp_path)

    manifest = json.loads((target / "candidates" / "004" / "_import.json").read_text())
    assert manifest["tune_summary"] is None
    assert manifest["warm_to_tuned_delta"] is None
    assert manifest["baseline_score"] == 0.9


def test_run_level_artifacts_seeded_only_once(tmp_path) -> None:
    _make_repo(tmp_path)
    source = _make_source_run(tmp_path, run_ids=("004", "005"))
    target = _make_target(tmp_path)

    import_candidate.import_candidate(
        source, "004", target, seed_experience=True, repo_root=tmp_path
    )
    assert (target / "background.md").read_text() == "# background\n"
    assert (target / "background_retrieval.json").read_text() == json.dumps({"results": []})
    assert json.loads((target / "experience.seed.json").read_text()) == EXPERIENCE

    (target / "background.md").write_text("sentinel background")
    (target / "background_retrieval.json").write_text("sentinel retrieval")
    (target / "experience.seed.json").write_text("sentinel experience")
    import_candidate.import_candidate(
        source, "005", target, seed_experience=True, repo_root=tmp_path
    )

    assert (target / "background.md").read_text() == "sentinel background"
    assert (target / "background_retrieval.json").read_text() == "sentinel retrieval"
    assert (target / "experience.seed.json").read_text() == "sentinel experience"
