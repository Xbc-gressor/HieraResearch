from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402
from arms import local_tr  # noqa: E402
from ib_support import (  # noqa: E402
    evaluation_events,
    fake_eval_by_params,
    fake_eval_from,
    hrow,
    ok_preflight,
    read_events,
    write_checkpoint,
)

TINY_INT_SOURCE = '''
PARAM_SCHEMA = {
    "width": "int",
}
SEARCH_SPACE = {
    "width": ("int", 1, 2),
}
BASE_PARAMS = {
    "width": 1,
}
def make_model(params):
    return params
'''

DEGENERATE_ONLY_SOURCE = '''
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


def test_radius_grows_on_improvements(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)
    out = tmp_path / "out"
    scores = [90.0 - 10.0 * index for index in range(10)]  # every eval improves

    result = runner.run_cell(
        arm=local_tr.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=7,
        eval_fn=fake_eval_from([("ok", score) for score in scores]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "ok"
    assert result["beat_initial_incumbent"] is True
    radii = [event["arm_state"]["tr_radius"] for event in evaluation_events(out)]
    assert radii == pytest.approx(
        [0.25, 0.375, 0.5625, 0.84375, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    assert result["counts"] == {
        "crashes": 0,
        "invalid": 0,
        "duplicates": 0,
        "task_preflight_rejected": 0,
    }
    assert result["internal_duplicate_count"] == 0
    assert result["internal_resample_count"] == 0


def test_radius_shrinks_on_crashes(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)
    out = tmp_path / "out"

    result = runner.run_cell(
        arm=local_tr.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=7,
        eval_fn=fake_eval_from([("crash", None)] * 10),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "ok"  # budget exhaustion is the normal终点
    assert result["counts"]["crashes"] == 10
    assert result["beat_initial_incumbent"] is False
    radii = [event["arm_state"]["tr_radius"] for event in evaluation_events(out)]
    assert radii == pytest.approx(
        [0.25, 0.125, 0.0625, 0.03125, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]
    )


def test_improves_incumbent_and_stays_in_space(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)
    out = tmp_path / "out"

    result = runner.run_cell(
        arm=local_tr.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=3,
        eval_fn=fake_eval_by_params(lambda params: params["depth"]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "ok"
    assert result["beat_initial_incumbent"] is True
    assert result["best_score"] <= 8.0  # depth range is [1, 8]
    # Every proposal came from codec.decode: none can bounce at preflight.
    assert not [
        event for event in read_events(out) if event["kind"] == "preflight_rejected"
    ]


def test_duplicate_exhaustion_is_arm_error(tmp_path) -> None:
    # Both configs of the tiny 2-config space are already executed: every
    # in-arm draw is a duplicate, so the 32-draw bound must trip.
    ckpt = write_checkpoint(
        tmp_path,
        train_source=TINY_INT_SOURCE,
        incumbent_params={"width": 1},
        incumbent_score=10.0,
        history=[hrow({"width": 2}, 20.0)],
    )

    result = runner.run_cell(
        arm=local_tr.ARM,
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


def test_unsupported_when_nothing_varies(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path, train_source=DEGENERATE_ONLY_SOURCE)

    result = runner.run_cell(
        arm=local_tr.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=1,
        eval_fn=fake_eval_from([]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "unsupported"
    assert result["evaluations"] == 0
