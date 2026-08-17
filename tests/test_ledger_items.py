from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ledger import _capture_task_baseline_item  # noqa: E402


def baseline_record(score: float, *, status: str = "keep") -> dict:
    return {
        "run_id": "000",
        "role": "task_provided_baseline",
        "status": status,
        "final_best_score": score,
    }


def test_task_baseline_item_first_finite_screening_value_wins() -> None:
    ledger = {"metric": "val_bpb", "items": {}}
    _capture_task_baseline_item(ledger, baseline_record(1.25))

    item = ledger["items"]["task_baseline"]
    assert item["value"] == pytest.approx(1.25)
    assert item["metric"] == "val_bpb"
    assert item["source"] == {
        "role": "task_provided_baseline",
        "run_id": "000",
        "stage": "screening",
    }

    _capture_task_baseline_item(ledger, baseline_record(1.25))
    with pytest.raises(ValueError, match="immutable"):
        _capture_task_baseline_item(ledger, baseline_record(1.10))


def test_task_baseline_item_ignores_crash_and_ordinary_candidate() -> None:
    ledger = {"metric": "val_bpb", "items": {}}
    _capture_task_baseline_item(ledger, baseline_record(float("inf"), status="crash"))
    _capture_task_baseline_item(
        ledger,
        {"run_id": "001", "status": "keep", "final_best_score": 0.9},
    )
    assert ledger["items"] == {}
