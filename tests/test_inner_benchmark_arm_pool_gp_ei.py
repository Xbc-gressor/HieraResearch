from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codec as codec_mod  # noqa: E402
import runner  # noqa: E402
import space  # noqa: E402
from arms import pool_gp_ei  # noqa: E402
from ib_support import (  # noqa: E402
    BASE,
    cfg,
    evaluation_events,
    fake_eval_from,
    hrow,
    ok_preflight,
    write_checkpoint,
)


class FakeSession:
    """Scripted BoutSession stand-in: receipts (or exceptions) in call order."""

    def __init__(self, script):
        self._script = list(script)
        self.asks = []

    def ask(self, extra=None):
        self.asks.append(extra)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def totals(self):
        return {
            "llm_calls": len(self.asks),
            "llm_input_tokens": 10 * len(self.asks),
            "llm_output_tokens": 0,
        }


def fake_session_factory(script):
    def factory(role_name, first_extras=None):
        return FakeSession(script)

    return factory


# Continuation-regime history shared by the ranker tests: 8 unique configs in
# one region (depth 4, lr ~0.001, dropout 0.05..0.19, mode fast) with scores
# spread 88..109; the incumbent (BASE, 85.0) is the 9th live observation.
HISTORY_PARAMS = [
    cfg(depth=4, lr=0.001 + i * 2e-5, dropout=0.05 + i * 0.02, mode="fast")
    for i in range(8)
]
HISTORY_SCORES = [88.0 + 3.0 * i for i in range(8)]
INCUMBENT_SCORE = 85.0

FAR = cfg(depth=8, lr=0.05, dropout=0.45, mode="slow")
NEAR = [
    cfg(depth=4, lr=0.0011, dropout=0.06, mode="fast"),
    cfg(depth=3, lr=0.0012, dropout=0.10, mode="fast"),
    cfg(depth=5, lr=0.0013, dropout=0.14, mode="fast"),
    cfg(depth=4, lr=0.0014, dropout=0.18, mode="fast"),
]


def continuation_checkpoint(tmp_path):
    return write_checkpoint(
        tmp_path,
        incumbent_params=dict(BASE),
        incumbent_score=INCUMBENT_SCORE,
        history=[hrow(params, score) for params, score in zip(HISTORY_PARAMS, HISTORY_SCORES)],
        regime="continuation",
        stratum="cont_improved",
    )


def receipt(configs, *, order=None):
    return {
        "configs": configs,
        "order": order if order is not None else list(range(len(configs))),
        "rationale": "scripted pool",
    }


def test_warmup_gate_executes_proposer_rank1(tmp_path) -> None:
    # Live finite-unique count is 4 (incumbent + 3 rows) < WARMUP=8: every step
    # must execute the proposer's rank-1 config and record the fallback.
    ckpt = write_checkpoint(
        tmp_path,
        history=[hrow(cfg(depth=depth), 90.0 + depth) for depth in (1, 2, 3)],
    )
    script = [
        receipt(
            [cfg(depth=4 + o, lr=0.002 * (1 + step) + 1e-5 * o, dropout=0.2, mode="slow")
             for o in range(5)]
        )
        for step in range(2)
    ]

    result = runner.run_cell(
        arm=pool_gp_ei.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=7,
        budget=2,
        eval_fn=fake_eval_from([("ok", 90.0), ("ok", 80.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(script)},
    )

    assert result["status"] == "ok"
    assert result["ranker_fallback_count"] == 2
    events = evaluation_events(tmp_path / "out")
    assert len(events) == 2
    for step, event in enumerate(events):
        assert event["source"] == "pool_gp_ei_warmup_fallback"
        assert event["arm_state"]["ranker_fallback"] is True
        assert event["arm_state"]["pool_eis"] is None  # ranker never ran
        assert event["arm_state"]["selected_index"] == 0
        assert event["proposal"] == script[step]["configs"][0]


def test_ard_lengthscales_not_isotropic(tmp_path) -> None:
    # Score depends only on dropout; depth/lr/mode vary freely. The fitted ARD
    # kernel must keep per-dimension lengthscales and rank dropout most relevant.
    contract = space.read_contract(write_checkpoint(tmp_path) / "candidate" / "train.py")
    codec = codec_mod.Codec(contract)
    ranker = pool_gp_ei.GpEiRanker(contract, codec, seed=7)
    rng = np.random.default_rng(0)
    history = []
    for index in range(14):
        dropout = float(rng.uniform(0.0, 0.5))
        history.append((
            cfg(
                depth=int(rng.integers(1, 9)),
                lr=float(10 ** rng.uniform(-4, -1)),
                dropout=dropout,
                mode="fast" if index % 2 else "slow",
            ),
            math.sin(10.0 * dropout) + 3.0,
        ))

    ranker.rank(history, [cfg(depth=2, lr=0.002, dropout=0.2, mode="slow")], 2.0)

    assert ranker.n_features == 5  # depth, lr, dropout + mode one-hot(2)
    lengthscales = np.asarray(ranker.gp_.kernel_.length_scale, dtype=float)
    assert lengthscales.shape == (5,)  # a vector: the kernel did not go isotropic
    assert not np.allclose(lengthscales, lengthscales[0])
    # dropout is feature 2; the irrelevant numeric dims are features 0 and 1.
    assert lengthscales[2] < lengthscales[0]
    assert lengthscales[2] < lengthscales[1]


def test_ei_selection_overrides_proposer_rank(tmp_path) -> None:
    # FAR is the proposer's LAST choice (order[4]); its unexplored region gives
    # it the max EI, so the ranker must override the LLM's ranking.
    ckpt = continuation_checkpoint(tmp_path)
    pool_configs = [NEAR[0], NEAR[1], FAR, NEAR[2], NEAR[3]]
    script = [receipt(pool_configs, order=[0, 1, 3, 4, 2])]

    result = runner.run_cell(
        arm=pool_gp_ei.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=7,
        budget=1,
        eval_fn=fake_eval_from([("ok", 80.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(script)},
    )

    assert result["status"] == "ok"
    assert result["ranker_fallback_count"] == 0
    assert result["best_config"] == FAR
    event = evaluation_events(tmp_path / "out")[0]
    assert event["source"] == "pool_gp_ei"
    assert event["arm_state"]["ranker_fallback"] is False
    # Filtered pool is in proposer rank order: FAR sits at index 4.
    assert event["arm_state"]["selected_index"] == 4
    eis = event["arm_state"]["pool_eis"]
    assert len(eis) == 5
    assert int(np.argmax(eis)) == 4
    assert event["proposal"] == FAR


def test_fit_updates_with_new_outcomes(tmp_path) -> None:
    # Step 1 picks FAR (max EI). FAR evaluates badly (112, worst yet). Step 2's
    # pool contains NEAR_FAR, a neighbor of FAR: the updated fit must drop it.
    ckpt = continuation_checkpoint(tmp_path)
    near_far = cfg(depth=8, lr=0.048, dropout=0.44, mode="slow")
    pool1 = [NEAR[0], NEAR[1], FAR, NEAR[2], NEAR[3]]
    pool2 = [
        near_far,
        cfg(depth=4, lr=0.0015, dropout=0.16, mode="fast"),
        cfg(depth=3, lr=0.0016, dropout=0.17, mode="fast"),
        cfg(depth=5, lr=0.0017, dropout=0.18, mode="fast"),
        cfg(depth=4, lr=0.0018, dropout=0.19, mode="fast"),
    ]
    script = [receipt(pool1), receipt(pool2)]

    result = runner.run_cell(
        arm=pool_gp_ei.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=7,
        budget=2,
        eval_fn=fake_eval_from([("ok", 112.0), ("ok", 90.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(script)},
    )

    assert result["status"] == "ok"
    events = evaluation_events(tmp_path / "out")
    assert events[0]["proposal"] == FAR
    # The stale model (pre-outcome history) would have picked NEAR_FAR again...
    contract = space.read_contract(ckpt / "candidate" / "train.py")
    codec = codec_mod.Codec(contract)
    stale_ranker = pool_gp_ei.GpEiRanker(contract, codec, seed=7)
    stale_history = [(dict(BASE), INCUMBENT_SCORE)] + list(
        zip(HISTORY_PARAMS, HISTORY_SCORES)
    )
    stale_eis = stale_ranker.rank(stale_history, pool2, INCUMBENT_SCORE)
    assert int(np.argmax(stale_eis)) == 0
    # ...but after FAR's bad outcome NEAR_FAR's EI collapses and it loses.
    assert events[1]["proposal"] != near_far
    updated_eis = events[1]["arm_state"]["pool_eis"]
    assert updated_eis[0] < 1e-10
    assert events[1]["arm_state"]["selected_index"] == int(np.argmax(updated_eis))


def test_zero_variance_history_still_ranks(tmp_path) -> None:
    # All history scores identical -> objective std is 0 and treated as 1: the
    # fit must not blow up; EI is computed (sigma-driven) and a config runs.
    constant_params = [
        cfg(depth=1 + (i % 7) + 1, lr=0.002 + i * 3e-4, dropout=0.12 + i * 0.02, mode="fast")
        for i in range(8)
    ]
    ckpt = write_checkpoint(
        tmp_path,
        incumbent_params=dict(BASE),
        incumbent_score=50.0,
        history=[hrow(params, 50.0) for params in constant_params],
        regime="continuation",
        stratum="cont_improved",
    )
    pool = [
        cfg(depth=2, lr=0.005, dropout=0.3, mode="slow"),
        cfg(depth=4, lr=0.0011, dropout=0.11, mode="fast"),
        cfg(depth=3, lr=0.002, dropout=0.2, mode="fast"),
        cfg(depth=5, lr=0.003, dropout=0.25, mode="slow"),
        cfg(depth=6, lr=0.004, dropout=0.35, mode="fast"),
    ]

    result = runner.run_cell(
        arm=pool_gp_ei.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=7,
        budget=1,
        eval_fn=fake_eval_from([("ok", 55.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([receipt(pool)])},
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 1
    assert result["ranker_fallback_count"] == 0
    event = evaluation_events(tmp_path / "out")[0]
    eis = event["arm_state"]["pool_eis"]
    assert len(eis) == 5
    assert all(math.isfinite(ei) and ei >= 0.0 for ei in eis)
    assert event["arm_state"]["selected_index"] == int(np.argmax(eis))


def test_gp_fit_failure_is_arm_error(tmp_path, monkeypatch) -> None:
    # A GP that cannot fit ends the cell as arm_error (PLAN §6.4: no kernel
    # swap, no degradation to another strategy).
    def broken_fit(self, X, y):
        raise np.linalg.LinAlgError("mocked singular matrix")

    monkeypatch.setattr(GaussianProcessRegressor, "fit", broken_fit)
    pool = [cfg(depth=1 + o, lr=0.005 + 1e-4 * o, dropout=0.3, mode="slow") for o in range(5)]

    result = runner.run_cell(
        arm=pool_gp_ei.ARM,
        checkpoint_dir=continuation_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=7,
        budget=1,
        eval_fn=fake_eval_from([]),  # asserts if the objective is ever started
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([receipt(pool)])},
    )

    assert result["status"] == "arm_error"
    assert "GP fit/predict failed" in result["reason"]
    assert result["evaluations"] == 0
    assert result["llm_calls"] == 1  # the pool was asked before the fit failed


def test_status_counts_and_per_step_arm_state(tmp_path) -> None:
    # A clean 3-step cell: aggregates land in result.json and every evaluation
    # event carries a consistent GP-EI arm_state snapshot.
    def step_receipt(step):
        return receipt([
            cfg(
                depth=1 + o,
                lr=0.01 + 0.001 * step + 1e-4 * o,
                dropout=0.3 + 0.01 * o,
                mode="slow" if o % 2 else "fast",
            )
            for o in range(5)
        ])

    result = runner.run_cell(
        arm=pool_gp_ei.ARM,
        checkpoint_dir=continuation_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=7,
        budget=3,
        eval_fn=fake_eval_from([("ok", 80.0), ("ok", 70.0), ("ok", 60.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([step_receipt(s) for s in range(3)])},
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 3
    assert result["best_score"] == 60.0
    assert result["beat_initial_incumbent"] is True
    assert result["llm_calls"] == 3
    assert result["llm_input_tokens"] == 30
    assert result["ranker_fallback_count"] == 0
    assert result["internal_duplicate_count"] == 0
    for event in evaluation_events(tmp_path / "out"):
        assert event["source"] == "pool_gp_ei"
        state = event["arm_state"]
        assert state["ranker_fallback"] is False
        assert state["pool_size"] == 5
        assert len(state["pool_eis"]) == state["pool_size"]
        assert state["selected_index"] == int(np.argmax(state["pool_eis"]))
