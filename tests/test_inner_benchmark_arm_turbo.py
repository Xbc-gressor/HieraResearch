from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402
import checkpoint as checkpoint_mod  # noqa: E402
import codec as codec_mod  # noqa: E402
import space as space_mod  # noqa: E402
import state as state_mod  # noqa: E402
from arms import load_arm, turbo  # noqa: E402
from ib_support import (  # noqa: E402
    cfg,
    evaluation_events,
    fake_eval_from,
    hrow,
    ok_preflight,
    read_events,
    write_checkpoint,
)


DEGENERATE_ONLY_SOURCE = '''
PARAM_SCHEMA = {"fixed": "float"}
SEARCH_SPACE = {"fixed": ("float", 0.5, 0.5)}
BASE_PARAMS = {"fixed": 0.5}
def make_model(params):
    return params
'''


def _test_extras(state=None):
    extras = {"turbo_fit_steps": 2, "turbo_n_candidates": 32}
    if state is not None:
        extras["turbo_state"] = state
    return extras


def _final_state(out_dir):
    cell_end = read_events(out_dir)[-1]
    assert cell_end["kind"] == "cell_end"
    return cell_end["arm_state"]["turbo_final_state"]


def test_registered_gp_ts_smoke_is_reproducible(tmp_path) -> None:
    history = [
        hrow(cfg(depth=2, lr=0.0005, dropout=0.05), 110.0),
        hrow(cfg(depth=6, lr=0.01, dropout=0.3), 105.0),
    ]
    ckpt = write_checkpoint(tmp_path, history=history)
    calls = []
    for name in ("out-a", "out-b"):
        fake = fake_eval_from([("ok", 99.0), ("ok", 98.0)])
        result = runner.run_cell(
            arm=load_arm("turbo"),
            checkpoint_dir=ckpt,
            out_dir=tmp_path / name,
            seed=17,
            budget=2,
            eval_fn=fake,
            preflight_fn=ok_preflight,
            extras=_test_extras(),
        )
        assert result["status"] == "ok"
        assert [event["source"] for event in evaluation_events(tmp_path / name)] == [
            "turbo_ts",
            "turbo_ts",
        ]
        calls.append(fake.calls)
    assert calls[0] == calls[1]
    assert _final_state(tmp_path / "out-a")["outcomes_seen"] == 2


def test_last_feedback_is_persisted_and_next_bout_inherits_shrink(tmp_path) -> None:
    # The toy contract has three varying numeric dimensions, so the sequential
    # failure tolerance is max(4, d)=4. The fourth (last) crash is never sent
    # through Feedback by the runner, but must still halve the persisted TR.
    first = write_checkpoint(tmp_path, name="first")
    first_out = tmp_path / "first-out"
    fake = fake_eval_from([("crash", None)] * 4)
    first_result = runner.run_cell(
        arm=turbo.ARM,
        checkpoint_dir=first,
        out_dir=first_out,
        seed=5,
        budget=4,
        eval_fn=fake,
        preflight_fn=ok_preflight,
        extras=_test_extras(),
    )
    assert first_result["status"] == "ok"
    state = _final_state(first_out)
    assert state["outcomes_seen"] == 4
    assert state["failure_counter"] == 0
    assert state["length"] == pytest.approx(0.4)

    # Reconstruct the next scheduler-visible bout with the first bout's crash
    # rows in factual history and the emitted TuRBO state in extras.
    second = write_checkpoint(
        tmp_path,
        name="second",
        history=[hrow(params, None, status="crash") for params in fake.calls],
    )
    second_out = tmp_path / "second-out"
    second_result = runner.run_cell(
        arm=turbo.ARM,
        checkpoint_dir=second,
        out_dir=second_out,
        seed=999,  # persisted rng_seed, not this new wrapper seed, owns continuity
        budget=1,
        eval_fn=fake_eval_from([("ok", 101.0)]),
        preflight_fn=ok_preflight,
        extras=_test_extras(state),
    )
    assert second_result["status"] == "ok"
    event = evaluation_events(second_out)[0]
    assert event["arm_state"]["tr_length"] == pytest.approx(0.4)
    assert event["arm_state"]["turbo_step"] == 4


def test_interrupted_improvement_replays_against_preproposal_best(tmp_path) -> None:
    improved = 99.0
    ckpt = write_checkpoint(
        tmp_path,
        history=[hrow(cfg(depth=2, lr=0.0005, dropout=0.05), improved)],
        incumbent_params=cfg(depth=2, lr=0.0005, dropout=0.05),
        incumbent_score=improved,
    )
    persisted = turbo.TurboState(
        dimension=3,
        rng_seed=7,
        length=0.8,
        success_counter=0,
        failure_counter=0,
        success_tolerance=3,
        failure_tolerance=4,
        best_score=100.0,
        proposal_index=0,
        outcomes_seen=0,
    ).to_json()
    extras = _test_extras(persisted)
    extras.update(
        {
            "turbo_recovery_outcomes": [
                {"status": "ok", "score": improved}
            ],
            "turbo_resume_proposal_consumed": True,
        }
    )

    result = runner.run_cell(
        arm=turbo.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "recovered",
        seed=999,
        budget=1,
        eval_fn=fake_eval_from([("ok", 101.0)]),
        preflight_fn=ok_preflight,
        extras=extras,
    )

    assert result["status"] == "ok"
    event = evaluation_events(tmp_path / "recovered")[0]
    assert event["arm_state"]["turbo_step"] == 1
    assert event["arm_state"]["success_counter"] == 1
    assert event["arm_state"]["failure_counter"] == 0


def test_three_significant_improvements_expand_before_fourth_proposal(
    tmp_path,
) -> None:
    ckpt = write_checkpoint(tmp_path)
    out_dir = tmp_path / "out"
    result = runner.run_cell(
        arm=turbo.ARM,
        checkpoint_dir=ckpt,
        out_dir=out_dir,
        seed=23,
        budget=4,
        eval_fn=fake_eval_from(
            [("ok", 99.0), ("ok", 98.0), ("ok", 97.0), ("ok", 96.0)]
        ),
        preflight_fn=ok_preflight,
        extras=_test_extras(),
    )
    assert result["status"] == "ok"
    lengths = [
        event["arm_state"]["tr_length"]
        for event in evaluation_events(out_dir)
    ]
    assert lengths == pytest.approx([0.8, 0.8, 0.8, 1.6])
    state = _final_state(out_dir)
    assert state["policy_id"] == "hotstart-turbo1-deep20-v1"
    assert state["success_tolerance"] == 3


def test_candidate_pool_filters_canonical_integer_absorption(
    tmp_path, monkeypatch
) -> None:
    ckpt_dir = write_checkpoint(tmp_path)
    checkpoint = checkpoint_mod.load_checkpoint(ckpt_dir)
    contract = space_mod.read_contract(checkpoint.candidate_path)
    codec = codec_mod.Codec(contract)
    incumbent = cfg()
    cell_state = state_mod.CellState(
        incumbent_config=incumbent,
        incumbent_score=100.0,
        budget_remaining=1,
        trials=[state_mod.Trial(incumbent, 100.0, "ok")],
        identity_fn=contract.params_identity,
    )
    ctx = SimpleNamespace(contract=contract, codec=codec, state=cell_state)
    center, categorical_labels = codec.encode(incumbent)

    # depth's local raw interval is [0.40, 0.55]. Values near its upper edge
    # round to depth=5, whose executed z=4/7 lies outside that interval. The
    # other rows collapse either onto history or onto one another after cast.
    draws = torch.tensor(
        [
            [0.05, center[1], center[2]],
            [0.05, center[1], center[2]],
            [0.05, 0.50, 0.40],
            [0.05, 0.50, 0.40],
            [0.95, 0.50, 0.40],
        ],
        dtype=torch.double,
    )

    class FixedSobol:
        def __init__(self, dimension, *, scramble, seed):
            assert dimension == 3
            assert scramble is True

        def draw(self, count):
            assert count == len(draws)
            return draws

    monkeypatch.setattr(turbo.torch.quasirandom, "SobolEngine", FixedSobol)
    pool, canonical, diagnostics = turbo._candidate_pool(
        ctx,
        moving=[0, 1, 2],
        categorical_labels=categorical_labels,
        center_full=center,
        lower=np.array([0.40, 0.0, 0.0]),
        upper=np.array([0.55, 1.0, 1.0]),
        n_candidates=len(draws),
        rng=np.random.default_rng(7),
    )

    assert len(pool) == 1
    assert canonical.shape == (1, 3)
    assert diagnostics == {
        "candidate_count_before": 5,
        "candidate_count_after": 1,
        "candidate_outside_tr": 1,
        "candidate_history_duplicates": 2,
        "candidate_pool_duplicates": 1,
    }


def test_unsupported_without_varying_numeric_dimension(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path, train_source=DEGENERATE_ONLY_SOURCE)
    result = runner.run_cell(
        arm=turbo.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=1,
        eval_fn=fake_eval_from([]),
        preflight_fn=ok_preflight,
        extras=_test_extras(),
    )
    assert result["status"] == "unsupported"
    assert result["evaluations"] == 0
