from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))

import objective  # noqa: E402

# Mirrors the production literal formats exactly (tuples included), same as
# tests/test_inner_benchmark_space.py.
TRAIN_SOURCE = '''
PARAM_SCHEMA = {
    "depth": "int",
    "lr": ("float", "log"),
    "dropout": "float",
    "mode": ("categorical", ["fast", "slow"]),
}
SEARCH_SPACE = {
    "depth": ("int", 1, 8),
    "lr": ("float", 0.0001, 0.1, "log"),
    "dropout": ("float", 0.0, 0.5),
    "mode": ("categorical", ["fast", "slow"]),
}
BASE_PARAMS = {
    "depth": 4,
    "lr": 0.001,
    "dropout": 0.1,
    "mode": "fast",
}
def make_model(params):
    return dict(params)
'''

# Deterministic cheap functions of params over the make_model stub.
# REJECT_DEPTH is the in-space magic value preflight_config rejects.
PREPARE_SOURCE = '''
import time

REJECT_DEPTH = 7

def _weighted_sum(model):
    return (
        model["depth"] * 10
        + model["lr"] * 100
        + model["dropout"]
        + (0 if model["mode"] == "fast" else 1)
    )

def evaluate_config(make_model, params):
    return _weighted_sum(make_model(params))

def alt_score(make_model, params):
    return _weighted_sum(make_model(params)) + 1000.0

def exploding_score(make_model, params):
    raise RuntimeError("toy score exploded")

def nan_score(make_model, params):
    return float("nan")

def inf_score(make_model, params):
    return float("inf")

def slow_score(make_model, params):
    time.sleep(30)
    return 0.0

def preflight_config(make_model, params):
    model = make_model(params)
    if model["depth"] == REJECT_DEPTH:
        raise ValueError("depth 7 is infeasible")
    return {"status": "ok"}

def buggy_preflight(make_model, params):
    return {"status": "ok", "seen": undefined_name}

def slow_preflight(make_model, params):
    time.sleep(30)
    return {"status": "ok"}
'''

BASE_PARAMS = {"depth": 4, "lr": 0.001, "dropout": 0.1, "mode": "fast"}
EXPECTED_SCORE = 4 * 10 + 0.001 * 100 + 0.1 + 0  # same ops as _weighted_sum


@pytest.fixture()
def candidate(tmp_path) -> Path:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "train.py").write_text(TRAIN_SOURCE)
    (candidate_dir / "prepare.py").write_text(PREPARE_SOURCE)
    return candidate_dir / "train.py"


def test_evaluate_ok_returns_expected_float(candidate, tmp_path) -> None:
    outcome = objective.evaluate(candidate, BASE_PARAMS, score_fn="evaluate_config")

    assert outcome.status == "ok"
    assert outcome.score == pytest.approx(EXPECTED_SCORE)
    assert outcome.detail is None
    assert outcome.elapsed_seconds >= 0
    # Accounting-free: no production attempt-log reservation anywhere.
    assert not list(tmp_path.rglob("evaluation_attempts.jsonl"))


def test_evaluate_explicit_score_fn_is_honored(candidate) -> None:
    outcome = objective.evaluate(candidate, BASE_PARAMS, score_fn="alt_score")

    assert outcome.status == "ok"
    assert outcome.score == pytest.approx(EXPECTED_SCORE + 1000.0)


def test_evaluate_score_fn_raises_is_crash(candidate) -> None:
    outcome = objective.evaluate(candidate, BASE_PARAMS, score_fn="exploding_score")

    assert outcome.status == "crash"
    assert outcome.score is None
    assert "toy score exploded" in outcome.detail


def test_evaluate_nonfinite_score_is_crash(candidate) -> None:
    for score_fn in ("nan_score", "inf_score"):
        outcome = objective.evaluate(candidate, BASE_PARAMS, score_fn=score_fn)

        assert outcome.status == "crash"
        assert outcome.score is None
        assert "non-finite" in outcome.detail


def test_evaluate_timeout_is_crash(candidate) -> None:
    outcome = objective.evaluate(
        candidate, BASE_PARAMS, score_fn="slow_score", per_runtime_limit=1.0
    )

    assert outcome.status == "crash"
    assert outcome.score is None
    assert "per_runtime_limit" in outcome.detail
    assert outcome.elapsed_seconds < 10  # the 30 s sleep was hard-killed


def test_preflight_ok(candidate) -> None:
    outcome = objective.preflight(
        candidate, BASE_PARAMS, preflight_fn="preflight_config"
    )

    assert outcome.status == "ok"
    assert outcome.detail is None


def test_preflight_magic_value_is_rejected(candidate) -> None:
    params = {**BASE_PARAMS, "depth": 7}
    outcome = objective.preflight(candidate, params, preflight_fn="preflight_config")

    assert outcome.status == "rejected"
    assert "infeasible" in outcome.detail


def test_preflight_fn_exception_is_rejected(candidate) -> None:
    outcome = objective.preflight(
        candidate, BASE_PARAMS, preflight_fn="buggy_preflight"
    )

    assert outcome.status == "rejected"
    assert "NameError" in outcome.detail


def test_preflight_timeout_is_rejected(candidate) -> None:
    outcome = objective.preflight(
        candidate, BASE_PARAMS, preflight_fn="slow_preflight", per_runtime_limit=1.0
    )

    assert outcome.status == "rejected"
    assert "preflight_runtime_limit" in outcome.detail


def test_python_cmd_for_project() -> None:
    # Test-fixture path: no project -> the cell's own interpreter.
    assert objective.python_cmd_for_project(None) == [sys.executable]
    # Real-checkpoint path: production's `uv --project tasks/<task>` split.
    cmd = objective.python_cmd_for_project("tasks/autoresearch-baseline")
    assert cmd[:2] == ["uv", "--project"]
    assert cmd[2] == str(ROOT / "tasks" / "autoresearch-baseline")
    assert cmd[3:] == ["run", "python"]


def test_evaluate_explicit_python_cmd(candidate) -> None:
    outcome = objective.evaluate(
        candidate,
        BASE_PARAMS,
        score_fn="evaluate_config",
        python_cmd=[sys.executable],
    )

    assert outcome.status == "ok"
    assert outcome.score == pytest.approx(EXPECTED_SCORE)
