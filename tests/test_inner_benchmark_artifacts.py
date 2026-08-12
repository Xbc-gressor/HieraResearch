from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))

import artifacts  # noqa: E402


def _manifest() -> dict:
    return {
        "checkpoint_id": "cp-001",
        "checkpoint_hash": "abc123",
        "candidate_execution_revision": {"revision": "rev123"},
        "arm": "spsa",
        "seed": 0,
        "model": {},
        "dependencies": {"numpy": "2.5.1"},
        "hardware": {"gpu": "none"},
        "active_dimensions": 2,
        "extra": {"s": 0.1, "a_0": 0.01},
    }


def test_manifest_round_trip_and_write_once(tmp_path) -> None:
    path = artifacts.write_manifest(tmp_path, _manifest())
    assert path == tmp_path / "manifest.json"
    assert json.loads(path.read_text())["extra"] == {"s": 0.1, "a_0": 0.01}
    with pytest.raises(FileExistsError):
        artifacts.write_manifest(tmp_path, _manifest())


def test_manifest_missing_key_rejected(tmp_path) -> None:
    broken = _manifest()
    del broken["seed"]
    with pytest.raises(ValueError, match="seed"):
        artifacts.write_manifest(tmp_path, broken)


def test_events_append_round_trip(tmp_path) -> None:
    first = artifacts.make_event(
        eval_index=0,
        transaction_id="tx-0",
        proposal={"x": 1},
        source="local_tr",
        rationale=None,
        preflight_status="ok",
        status="ok",
        score=1.5,
        incumbent_before={"config": {"x": 0}, "score": 2.0},
        incumbent_after={"config": {"x": 1}, "score": 1.5},
        arm_state={"radius": 0.3},
    )
    second = artifacts.make_event(eval_index=1, proposal={"x": 2}, status="crash")
    artifacts.append_event(tmp_path, first)
    artifacts.append_event(tmp_path, second)
    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    back = [json.loads(line) for line in lines]
    for key in artifacts.EVENT_KEYS:
        assert key in back[0] and key in back[1]
    assert back[0]["incumbent_after"] == {"config": {"x": 1}, "score": 1.5}
    assert back[0]["arm_state"] == {"radius": 0.3}
    assert back[1]["status"] == "crash" and back[1]["score"] is None


def test_make_event_envelope_and_extra(tmp_path) -> None:
    event = artifacts.make_event(pair_id=7)  # arm-specific extra key
    assert set(artifacts.EVENT_KEYS) <= set(event)
    assert event["arm_state"] == {} and event["pair_id"] == 7
    with pytest.raises(ValueError, match="envelope"):
        artifacts.append_event(tmp_path, {"score": 1.0})


def test_write_result(tmp_path) -> None:
    result = {"best_config": {"x": 1}, "best_score": 1.5, "metrics": {"n_ok": 9}}
    path = artifacts.write_result(tmp_path, result)
    assert path == tmp_path / "result.json"
    assert json.loads(path.read_text()) == result
