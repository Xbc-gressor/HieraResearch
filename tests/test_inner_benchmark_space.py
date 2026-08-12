from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))

import space  # noqa: E402

# Mirrors the production literal formats exactly (tuples included).
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
    return params
'''


def _write(tmp_path: Path, source: str = TRAIN_SOURCE) -> Path:
    train = tmp_path / "train.py"
    train.write_text(source)
    return train


@pytest.fixture()
def contract(tmp_path) -> space.CandidateContract:
    return space.read_contract(_write(tmp_path))


def test_read_contract_dimensions(contract) -> None:
    assert [d.name for d in contract.dimensions] == ["depth", "lr", "dropout", "mode"]
    assert [d.name for d in contract.numeric_dimensions] == ["depth", "lr", "dropout"]
    assert [d.name for d in contract.continuous_dimensions] == ["lr", "dropout"]
    assert [d.name for d in contract.categorical_dimensions] == ["mode"]
    depth, lr, _, mode = contract.dimensions
    assert (depth.kind, depth.log, depth.lo, depth.hi) == ("int", False, 1, 8)
    assert (lr.kind, lr.log, lr.lo, lr.hi) == ("float", True, 0.0001, 0.1)
    assert mode.kind == "categorical" and mode.options == ("fast", "slow")
    assert contract.base_params["depth"] == 4
    assert contract.search_space["lr"] == ("float", 0.0001, 0.1, "log")


def test_key_set_mismatch_rejected(tmp_path) -> None:
    bad = TRAIN_SOURCE.replace('    "dropout": 0.1,\n', "")
    with pytest.raises(ValueError, match="key_mismatch"):
        space.read_contract(_write(tmp_path, bad))


def test_invalid_space_entry_rejected(tmp_path) -> None:
    bad = TRAIN_SOURCE.replace('("int", 1, 8)', '("int", 8, 1)')
    with pytest.raises(ValueError, match="bad_tuple"):
        space.read_contract(_write(tmp_path, bad))


def test_missing_mapping_rejected(tmp_path) -> None:
    bad = TRAIN_SOURCE.replace("SEARCH_SPACE = {", "RENAMED = {", 1)
    with pytest.raises(ValueError, match="SEARCH_SPACE"):
        space.read_contract(_write(tmp_path, bad))


def test_duplicate_detection(contract) -> None:
    params = {"depth": 3, "lr": 0.01, "dropout": 0.2, "mode": "slow"}
    # Identity is canonical (sorted keys): key order cannot evade detection —
    # same values in a different dict order ARE duplicates.
    reordered = {"mode": "slow", "dropout": 0.2, "lr": 0.01, "depth": 3}
    assert contract.is_duplicate(params, [reordered])
    # Production cast semantics: 3.0 casts to 3 on an int dimension -> duplicate.
    cast_equivalent = {"depth": 3.0, "lr": 0.01, "dropout": 0.2, "mode": "slow"}
    assert contract.is_duplicate(params, [cast_equivalent])
    # Production int cast truncates toward zero: int(3.7) == 3 -> also duplicate.
    truncated = {"depth": 3.7, "lr": 0.01, "dropout": 0.2, "mode": "slow"}
    assert contract.is_duplicate(params, [truncated])
    # A genuinely different value is not a duplicate.
    different = {"depth": 5, "lr": 0.01, "dropout": 0.2, "mode": "slow"}
    assert not contract.is_duplicate(params, [different])
    assert not contract.is_duplicate(params, [])


DEGENERATE_SOURCE = '''
PARAM_SCHEMA = {
    "depth": "int",
    "fixed_lr": "float",
    "width": "int",
    "mode": ("categorical", ["only"]),
}
SEARCH_SPACE = {
    "depth": ("int", 1, 8),
    "fixed_lr": ("float", 0.01, 0.01),
    "width": ("int", 3, 3),
    "mode": ("categorical", ["only"]),
}
BASE_PARAMS = {
    "depth": 4,
    "fixed_lr": 0.01,
    "width": 3,
    "mode": "only",
}
def make_model(params):
    return params
'''


def test_degenerate_and_varying_dimensions(tmp_path) -> None:
    contract = space.read_contract(_write(tmp_path, DEGENERATE_SOURCE))

    degenerate = {d.name: d.is_degenerate for d in contract.dimensions}
    assert degenerate == {
        "depth": False,
        "fixed_lr": True,  # lo == hi float
        "width": True,  # lo == hi int
        "mode": True,  # single-option categorical
    }
    assert [d.name for d in contract.varying_dimensions] == ["depth"]
