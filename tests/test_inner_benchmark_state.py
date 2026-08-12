from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))

import state  # noqa: E402


def _cell() -> state.CellState:
    return state.CellState(
        incumbent_config={"x": 1}, incumbent_score=2.0, budget_remaining=5
    )


def test_strict_improvement_only() -> None:
    cell = _cell()
    cell.record_outcome({"x": 2}, status="ok", score=2.0)  # equal: not strict
    assert cell.best_so_far() == ({"x": 1}, 2.0)
    cell.record_outcome({"x": 3}, status="ok", score=2.5)  # worse
    assert cell.best_so_far() == ({"x": 1}, 2.0)
    cell.record_outcome({"x": 4}, status="ok", score=1.5)  # strict improvement
    assert cell.best_so_far() == ({"x": 4}, 1.5)
    cell.record_outcome({"x": 5}, status="ok", score=float("inf"))  # non-finite
    assert cell.best_so_far() == ({"x": 4}, 1.5)


def test_crash_and_rejected_rows() -> None:
    cell = _cell()
    trial = cell.record_outcome({"x": 9}, status="crash")
    assert trial.score is None  # no silent coercion to a number
    assert cell.budget_remaining == 4  # an executed crash consumes budget
    assert cell.best_so_far() == ({"x": 1}, 2.0)
    cell.record_outcome({"x": 10}, status="preflight_rejected")
    assert cell.budget_remaining == 4  # unexecuted proposal consumes none
    assert cell.best_so_far() == ({"x": 1}, 2.0)
    with pytest.raises(ValueError, match="score=None"):
        cell.record_outcome({"x": 11}, status="crash", score=1.0)
    with pytest.raises(ValueError, match="real score"):
        cell.record_outcome({"x": 12}, status="ok", score=None)
    with pytest.raises(ValueError, match="unknown trial status"):
        cell.record_outcome({"x": 13}, status="weird")


def test_finite_unique_history() -> None:
    cell = _cell()
    cell.record_outcome({"x": 1}, status="ok", score=2.0)
    cell.record_outcome({"x": 2}, status="crash")
    cell.record_outcome({"x": 3}, status="preflight_rejected")
    cell.record_outcome({"x": 4}, status="ok", score=float("inf"))
    # Same config identity again: dropped, first occurrence (score 2.0) kept.
    cell.record_outcome({"x": 1}, status="ok", score=1.5)
    cell.record_outcome({"x": 5}, status="ok", score=1.9)
    assert cell.finite_unique_history() == [({"x": 1}, 2.0), ({"x": 5}, 1.9)]


def test_arm_state_slot() -> None:
    cell = _cell()
    assert cell.arm_state == {}
    cell.arm_state["tr_radius"] = 0.3  # opaque dict the arm owns
    assert cell.arm_state["tr_radius"] == 0.3
