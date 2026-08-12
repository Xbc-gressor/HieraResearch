from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))

import codec  # noqa: E402
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


@pytest.fixture()
def cdc(tmp_path) -> codec.Codec:
    train = tmp_path / "train.py"
    train.write_text(TRAIN_SOURCE)
    return codec.Codec(space.read_contract(train))


def test_round_trip_all_kinds(cdc) -> None:
    params = {"depth": 5, "lr": 0.0033, "dropout": 0.25, "mode": "slow"}
    z, cat_labels = cdc.encode(params)
    assert isinstance(z, np.ndarray) and z.shape == (cdc.z_dim,) == (3,)
    assert cat_labels == {"mode": "slow"}  # categorical dims stay out of z
    back = cdc.decode(z, cat_labels)
    assert list(back) == ["depth", "lr", "dropout", "mode"]  # canonical order
    assert back["depth"] == 5 and isinstance(back["depth"], int)
    assert back["lr"] == pytest.approx(0.0033)
    assert back["dropout"] == pytest.approx(0.25)
    assert back["mode"] == "slow"
    z2, _ = cdc.encode(back)
    assert z2 == pytest.approx(z)


def test_log_space_linearity(cdc) -> None:
    # lr: ("float", 1e-4, 1e-1, "log") -> three decades; 1e-2 sits at 2/3.
    z, _ = cdc.encode({"depth": 1, "lr": 0.01, "dropout": 0.0, "mode": "fast"})
    assert z[1] == pytest.approx(2.0 / 3.0)
    back = cdc.decode([0.0, 2.0 / 3.0, 0.0], {"mode": "fast"})
    assert back["lr"] == pytest.approx(0.01)


def test_int_nearest_round_and_clamp(cdc) -> None:
    # depth: ("int", 1, 8); z = 0.5 -> 4.5 -> deterministic nearest (half-up) 5.
    assert cdc.decode([0.5, 0.0, 0.0], {"mode": "fast"})["depth"] == 5
    assert cdc.decode([0.4, 0.0, 0.0], {"mode": "fast"})["depth"] == 4  # 3.8 -> 4
    # z projection clamps into [0,1] before inverse.
    params = cdc.decode([2.0, -1.0, 0.0], {"mode": "fast"})
    assert params["depth"] == 8
    assert params["lr"] == pytest.approx(0.0001)


def test_project(cdc) -> None:
    assert list(cdc.project([-0.5, 0.5, 1.5])) == [0.0, 0.5, 1.0]


def test_encode_categorical(cdc) -> None:
    assert cdc.encode_categorical("mode", "fast") == 0
    assert cdc.encode_categorical("mode", "slow") == 1
    with pytest.raises(ValueError, match="not in options"):
        cdc.encode_categorical("mode", "nope")
    with pytest.raises(ValueError, match="no categorical dimension"):
        cdc.encode_categorical("depth", "fast")


def test_decode_requires_categorical_labels(cdc) -> None:
    with pytest.raises(ValueError, match="missing categorical labels"):
        cdc.decode([0.0, 0.0, 0.0], {})
    with pytest.raises(ValueError, match="not in options"):
        cdc.decode([0.0, 0.0, 0.0], {"mode": "nope"})


def test_decode_z_length_checked(cdc) -> None:
    with pytest.raises(ValueError, match="length"):
        cdc.decode([0.0, 0.0], {"mode": "fast"})


DEGENERATE_SOURCE = '''
PARAM_SCHEMA = {
    "depth": "int",
    "fixed_lr": "float",
}
SEARCH_SPACE = {
    "depth": ("int", 1, 8),
    "fixed_lr": ("float", 0.01, 0.01),
}
BASE_PARAMS = {
    "depth": 4,
    "fixed_lr": 0.01,
}
def make_model(params):
    return params
'''


def test_degenerate_dimension_codec(tmp_path) -> None:
    train = tmp_path / "train.py"
    train.write_text(DEGENERATE_SOURCE)
    cdc = codec.Codec(space.read_contract(train))

    # The single legal value encodes to 0.0; out-of-range values map honestly
    # outside [0,1] (no silent projection on encode).
    z, _ = cdc.encode({"depth": 4, "fixed_lr": 0.01})
    assert z[1] == 0.0
    assert cdc.encode({"depth": 4, "fixed_lr": 0.005})[0][1] == -1.0
    assert cdc.encode({"depth": 4, "fixed_lr": 0.02})[0][1] == 2.0
    # Every z decodes to the one legal value — the dimension is immovable.
    for z_fixed in (0.0, 0.5, 1.0):
        assert cdc.decode([0.5, z_fixed], {})["fixed_lr"] == pytest.approx(0.01)
