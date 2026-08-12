from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codec as codec_mod  # noqa: E402
import runner  # noqa: E402
import space as space_mod  # noqa: E402
from arms import spsa  # noqa: E402
from ib_support import (  # noqa: E402
    BASE,
    cfg,
    evaluation_events,
    fake_eval_by_params,
    fake_eval_from,
    hrow,
    ok_preflight,
    read_events,
    write_checkpoint,
)

INT_ONLY_SOURCE = '''
PARAM_SCHEMA = {
    "width": "int",
}
SEARCH_SPACE = {
    "width": ("int", 1, 8),
}
BASE_PARAMS = {
    "width": 4,
}
def make_model(params):
    return params
'''

DEGENERATE_FLOAT_SOURCE = '''
PARAM_SCHEMA = {
    "fixed": "float",
}
SEARCH_SPACE = {
    "fixed": ("float", 0.5, 0.5),
}
BASE_PARAMS = {
    "fixed": 0.5,
}
def make_model(params):
    return params
'''

SINGLE_FLOAT_SOURCE = '''
PARAM_SCHEMA = {
    "x": "float",
}
SEARCH_SPACE = {
    "x": ("float", 0.0, 1.0),
}
BASE_PARAMS = {
    "x": 0.5,
}
def make_model(params):
    return params
'''


def _toy_contract_codec(ckpt):
    contract = space_mod.read_contract(ckpt / "candidate" / "train.py")
    return contract, codec_mod.Codec(contract)


def _moving_continuous_indices(contract):
    """Spec restatement: SPSA moves the non-degenerate float dimensions."""
    return [
        index
        for index, dim in enumerate(contract.numeric_dimensions)
        if dim.kind == "float" and not dim.is_degenerate
    ]


def test_unsupported_when_no_continuous_dimension_varies(tmp_path) -> None:
    for name, source in (
        ("int_only", INT_ONLY_SOURCE),
        ("degenerate", DEGENERATE_FLOAT_SOURCE),
    ):
        result = runner.run_cell(
            arm=spsa.ARM,
            checkpoint_dir=write_checkpoint(tmp_path, name=name, train_source=source),
            out_dir=tmp_path / f"out_{name}",
            seed=1,
            eval_fn=fake_eval_from([]),
            preflight_fn=ok_preflight,
        )
        assert result["status"] == "unsupported"
        assert result["evaluations"] == 0


def test_calibration_uses_iqr_of_frozen_history(tmp_path) -> None:
    # 8 history rows + the 100.0 incumbent -> 9 frozen finite unique scores;
    # IQR over the sorted [10..80, 100] is q75 - q25 = 70 - 30 = 40.
    history = [
        hrow(cfg(lr=0.002 + 0.001 * index), score)
        for index, score in enumerate(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        )
    ]
    ckpt = write_checkpoint(
        tmp_path, regime="continuation", stratum="cont_improved", history=history
    )
    out = tmp_path / "out"

    result = runner.run_cell(
        arm=spsa.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=1,
        budget=2,
        eval_fn=fake_eval_by_params(lambda params: 50.0),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "ok"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["extra"]["s"] == pytest.approx(40.0)
    assert manifest["extra"]["a_0"] == pytest.approx(2.0 * 0.10 * 0.05 / 40.0)
    assert manifest["active_dimensions"] == 2  # lr + dropout of the toy contract


def test_calibration_falls_back_to_range_then_default(tmp_path) -> None:
    # < 4 frozen scores -> best-worst range; a zero range -> the 0.05 default.
    cases = [
        ("range", [hrow(cfg(depth=1), 10.0), hrow(cfg(depth=2), 30.0)], 90.0),
        ("flat", [hrow(cfg(depth=1), 100.0)], 0.05),
    ]
    for name, history, expected_s in cases:
        out = tmp_path / f"out_{name}"
        result = runner.run_cell(
            arm=spsa.ARM,
            checkpoint_dir=write_checkpoint(tmp_path, name=name, history=history),
            out_dir=out,
            seed=1,
            budget=1,
            eval_fn=fake_eval_by_params(lambda params: 50.0),
            preflight_fn=ok_preflight,
        )
        assert result["status"] == "ok"
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["extra"]["s"] == pytest.approx(expected_s)
        assert manifest["extra"]["a_0"] == pytest.approx(
            2.0 * 0.10 * 0.05 / expected_s
        )


def test_pairs_straddle_center_and_keep_int_categorical_fixed(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)
    out = tmp_path / "out"

    result = runner.run_cell(
        arm=spsa.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=3,
        eval_fn=fake_eval_by_params(lambda params: 50.0),  # zero gradient
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "ok"
    events = evaluation_events(out)
    assert len(events) == 10
    contract, codec = _toy_contract_codec(ckpt)
    assert _moving_continuous_indices(contract) == [1, 2]  # lr, dropout
    theta0, _ = codec.encode(BASE)
    for k in range(5):
        plus, minus = events[2 * k], events[2 * k + 1]
        c_k = 0.10 / (k + 1) ** 0.101
        a_k = 0.2 / (k + 1) ** 0.602  # flat frozen scores -> s=0.05 -> a_0=0.2
        for event, side in ((plus, "plus"), (minus, "minus")):
            assert event["arm_state"]["spsa_k"] == k
            assert event["arm_state"]["spsa_side"] == side
            assert event["arm_state"]["c_k"] == pytest.approx(c_k)
            assert event["arm_state"]["a_k"] == pytest.approx(a_k)
            assert event["proposal"]["depth"] == 4  # BASE, never moved
            assert event["proposal"]["mode"] == "fast"
        z_plus, _ = codec.encode(plus["proposal"])
        z_minus, _ = codec.encode(minus["proposal"])
        # Constant score -> zero gradient -> the iterate never leaves theta0,
        # so each pair is a simultaneous +-c_k perturbation around it.
        assert (z_plus + z_minus) / 2 == pytest.approx(theta0, rel=1e-12, abs=1e-12)
        half = (z_plus - z_minus) / 2
        assert half[0] == pytest.approx(0.0, abs=1e-15)
        assert abs(half[1]) == pytest.approx(c_k, rel=1e-12)
        assert abs(half[2]) == pytest.approx(c_k, rel=1e-12)
    assert not [event for event in read_events(out) if event["kind"] == "preflight_rejected"]


def test_iterate_follows_spsa_gradient_update(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)
    contract, codec = _toy_contract_codec(ckpt)
    moving = _moving_continuous_indices(contract)

    def score_of(params):
        return params["dropout"]

    # Independent spec reference (PLAN §6.3): Rademacher draws from
    # default_rng(seed), projected +- pair, two-sided gradient update.
    rng = np.random.default_rng(5)
    theta, cats = codec.encode(BASE)
    a_0 = 2.0 * 0.10 * 0.05 / 0.05  # flat frozen scores -> s = 0.05
    expected = []
    for k in range(5):
        c_k = 0.10 / (k + 1) ** 0.101
        a_k = a_0 / (k + 1) ** 0.602
        delta = rng.integers(0, 2, size=len(moving)) * 2 - 1
        plus = theta.copy()
        plus[moving] += c_k * delta
        minus = theta.copy()
        minus[moving] -= c_k * delta
        params_plus = codec.decode(codec.project(plus), cats)
        params_minus = codec.decode(codec.project(minus), cats)
        expected.extend([params_plus, params_minus])
        g_hat = (score_of(params_plus) - score_of(params_minus)) / (2.0 * c_k * delta)
        theta[moving] -= a_k * g_hat
        theta = codec.project(theta)

    eval_fn = fake_eval_by_params(score_of)
    result = runner.run_cell(
        arm=spsa.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=5,
        eval_fn=eval_fn,
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "ok"
    assert result["internal_duplicate_count"] == 0  # reference assumes no resample
    assert result["beat_initial_incumbent"] is True  # scores <= 0.5 << 100
    assert len(eval_fn.calls) == len(expected)
    for got, want in zip(eval_fn.calls, expected):
        assert set(got) == set(want)
        for key, value in want.items():
            if isinstance(value, float):
                assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-12)
            else:
                assert got[key] == value


def test_crash_skips_gradient_update(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)
    out = tmp_path / "out"

    result = runner.run_cell(
        arm=spsa.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=4,
        eval_fn=fake_eval_from([("crash", None)] * 10),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "ok"  # budget exhaustion is the normal终点
    assert result["counts"]["crashes"] == 10
    assert result["beat_initial_incumbent"] is False
    events = evaluation_events(out)
    assert [event["arm_state"]["spsa_k"] for event in events] == [
        k for k in range(5) for _ in (0, 1)
    ]
    contract, codec = _toy_contract_codec(ckpt)
    theta0, _ = codec.encode(BASE)
    for k in range(5):
        z_plus, _ = codec.encode(events[2 * k]["proposal"])
        z_minus, _ = codec.encode(events[2 * k + 1]["proposal"])
        # No score pair -> update skipped -> every pair still centered at theta0.
        assert (z_plus + z_minus) / 2 == pytest.approx(theta0, rel=1e-12, abs=1e-12)


def test_pair_duplicate_exhaustion_is_arm_error(tmp_path) -> None:
    # theta0 = 0.5 (z), c_0 = 0.1: every Rademacher draw projects to the pair
    # {0.4, 0.6} on the single dimension, and both are already executed, so no
    # legal pair can ever form and the 32-draw bound must trip.
    ckpt = write_checkpoint(
        tmp_path,
        train_source=SINGLE_FLOAT_SOURCE,
        incumbent_params={"x": 0.5},
        incumbent_score=10.0,
        history=[hrow({"x": 0.4}, 20.0), hrow({"x": 0.6}, 30.0)],
    )

    result = runner.run_cell(
        arm=spsa.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=1,
        eval_fn=fake_eval_from([]),  # asserts if the objective is ever started
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "arm_error"
    assert "32" in result["reason"]
    assert result["internal_duplicate_count"] == 32
    assert result["internal_resample_count"] == 32
