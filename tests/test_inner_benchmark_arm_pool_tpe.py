from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codec as codec_mod  # noqa: E402
import runner  # noqa: E402
import space as space_mod  # noqa: E402
from arms import pool_tpe  # noqa: E402
from ib_support import (  # noqa: E402
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


def fake_session_factory(script, created=None):
    def factory(role_name, first_extras=None):
        if created is not None:
            created.append((role_name, first_extras))
        return FakeSession(script)

    return factory


def pool_receipt(dropouts, *, order=None, depth=4, lr=0.001, mode="fast"):
    """A valid POOL=5 receipt over the toy contract, distinct by dropout."""
    return {
        "configs": [cfg(depth=depth, lr=lr, dropout=d, mode=mode) for d in dropouts],
        "order": order if order is not None else [0, 1, 2, 3, 4],
        "rationale": "scripted pool",
    }


# 8 unique executed rows; with the default incumbent (BASE, score 100) the live
# fit history is n=9, so the good set is the best 3 rows: the low-dropout
# "slow" cluster (scores 10/12/14). Everything else (incl. the incumbent) is bad.
CONT_HISTORY = [
    hrow(cfg(depth=1, lr=0.0003, dropout=0.02, mode="slow"), 10.0),
    hrow(cfg(depth=2, lr=0.0005, dropout=0.04, mode="slow"), 12.0),
    hrow(cfg(depth=3, lr=0.0007, dropout=0.06, mode="slow"), 14.0),
    hrow(cfg(depth=5, lr=0.005, dropout=0.33, mode="slow"), 55.0),
    hrow(cfg(depth=5, lr=0.01, dropout=0.30, mode="fast"), 50.0),
    hrow(cfg(depth=6, lr=0.02, dropout=0.36, mode="fast"), 60.0),
    hrow(cfg(depth=7, lr=0.03, dropout=0.42, mode="fast"), 70.0),
    hrow(cfg(depth=8, lr=0.05, dropout=0.48, mode="fast"), 80.0),
]


def write_continuation_checkpoint(base_dir) -> Path:
    return write_checkpoint(
        base_dir,
        regime="continuation",
        stratum="cont_improved",
        history=CONT_HISTORY,
    )


def test_warmup_fallback_executes_proposer_rank1(tmp_path) -> None:
    # first-regime checkpoint: live history is the incumbent row only (< 8),
    # so every step takes the fallback path (pool[0] in proposer rank order).
    ckpt = write_checkpoint(tmp_path)
    script = [
        pool_receipt([0.20, 0.22, 0.24, 0.26, 0.28], lr=0.002, order=[2, 0, 1, 3, 4]),
        pool_receipt([0.20, 0.22, 0.24, 0.26, 0.28], lr=0.003, order=[2, 0, 1, 3, 4]),
    ]

    result = runner.run_cell(
        arm=pool_tpe.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=5,
        budget=2,
        eval_fn=fake_eval_from([("ok", 90.0), ("ok", 80.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(script)},
    )

    assert result["status"] == "ok"
    assert result["ranker_fallback_count"] == 2
    events = evaluation_events(tmp_path / "out")
    states = [event["arm_state"] for event in events]
    assert [state["ranker_fallback"] for state in states] == [True, True]
    assert [state["tpe_scores"] for state in states] == [None, None]
    assert [state["tpe_chosen_index"] for state in states] == [0, 0]
    # pool[0] after ranking is configs[order[0]] = the dropout-0.24 member.
    proposals = [event["proposal"] for event in events]
    assert [p["dropout"] for p in proposals] == [0.24, 0.24]
    assert [p["lr"] for p in proposals] == [0.002, 0.003]


def test_tpe_ranker_overrides_proposer_order(tmp_path) -> None:
    # Proposer rank 1 is dropout 0.45 (bad region); the TPE scorer must pick
    # dropout 0.03, next to the good cluster at 0.02-0.06.
    ckpt = write_continuation_checkpoint(tmp_path)
    receipt = pool_receipt([0.03, 0.35, 0.40, 0.45, 0.48], order=[3, 0, 1, 2, 4])

    result = runner.run_cell(
        arm=pool_tpe.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=11,
        budget=1,
        eval_fn=fake_eval_from([("ok", 20.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([receipt])},
    )

    assert result["status"] == "ok"
    assert result["ranker_fallback_count"] == 0
    event = evaluation_events(tmp_path / "out")[0]
    assert event["proposal"]["dropout"] == 0.03
    state = event["arm_state"]
    assert state["ranker_fallback"] is False
    assert state["tpe_n_observations"] == 9
    # Filtered pool is in proposer rank order: [0.45, 0.03, 0.35, 0.40, 0.48].
    assert state["tpe_chosen_index"] == 1
    scores = state["tpe_scores"]
    assert len(scores) == 5
    assert scores[1] == max(scores)
    assert result["best_config"]["dropout"] == 0.03


def test_fit_history_grows_with_own_outcomes(tmp_path) -> None:
    # Each step refits on the LIVE history: after step 1's ok outcome the fit
    # set grows from 9 to 10 observations.
    ckpt = write_continuation_checkpoint(tmp_path)
    script = [
        pool_receipt([0.03, 0.35, 0.40, 0.45, 0.46]),
        pool_receipt([0.05, 0.37, 0.41, 0.43, 0.47]),
    ]

    result = runner.run_cell(
        arm=pool_tpe.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=3,
        budget=2,
        eval_fn=fake_eval_from([("ok", 5.0), ("ok", 4.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(script)},
    )

    assert result["status"] == "ok"
    states = [event["arm_state"] for event in evaluation_events(tmp_path / "out")]
    assert [state["tpe_n_observations"] for state in states] == [9, 10]
    assert [state["ranker_fallback"] for state in states] == [False, False]
    for state in states:
        assert state["tpe_scores"][state["tpe_chosen_index"]] == max(
            state["tpe_scores"]
        )


def test_tpe_scores_hand_computed(tmp_path) -> None:
    # Direct scorer check on 3 history points differing only in dropout
    # (z = dropout / 0.5); other dimensions are identical, so they contribute
    # equally to both pool members and cancel in the difference.
    #
    # good = {z 0.0 (score 1), z 0.2 (score 2)} (n=3 -> max(2, ceil(.75)) = 2),
    # bad = {z 1.0}. Silverman: bw_good = 1.06*0.1*2^-0.2 = 0.092278,
    # bw_bad = 0 (std of one point) -> floored to 0.05.
    # x1 = z 0.1: log l = -0.5*(0.1/bw)^2 - log(bw) - 0.5*log(2pi) = 0.876823;
    #             log g at distance 0.9 with bw 0.05 = -159.923207
    #   -> ratio 160.800030
    # x2 = z 0.9: log l = logsumexp(-46.097488, -27.305128) - log 2 = -27.998275;
    #             log g at distance 0.1 = 0.076793
    #   -> ratio -28.075068
    contract = space_mod.read_contract(
        write_checkpoint(tmp_path) / "candidate" / "train.py"
    )
    codec = codec_mod.Codec(contract)
    history = [
        (cfg(dropout=0.0), 1.0),
        (cfg(dropout=0.1), 2.0),
        (cfg(dropout=0.5), 3.0),
    ]
    pool = [cfg(dropout=0.05), cfg(dropout=0.45)]

    scores = pool_tpe.tpe_pool_scores(pool, history, contract, codec)

    assert all(isinstance(score, float) for score in scores)
    assert scores[0] > scores[1]
    assert scores[0] - scores[1] == pytest.approx(188.875, rel=1e-3)


def test_crash_outcome_stays_out_of_fit(tmp_path) -> None:
    # Step 1's executed config crashes: it consumes budget but never enters
    # the finite fit history, so step 2 still fits on the same 9 observations.
    ckpt = write_continuation_checkpoint(tmp_path)
    script = [
        pool_receipt([0.03, 0.35, 0.40, 0.45, 0.46]),
        pool_receipt([0.05, 0.37, 0.41, 0.43, 0.47]),
    ]

    result = runner.run_cell(
        arm=pool_tpe.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=3,
        budget=2,
        eval_fn=fake_eval_from([("crash", None), ("ok", 4.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(script)},
    )

    assert result["status"] == "ok"
    assert result["counts"]["crashes"] == 1
    assert result["ranker_fallback_count"] == 0
    states = [event["arm_state"] for event in evaluation_events(tmp_path / "out")]
    assert [state["tpe_n_observations"] for state in states] == [9, 9]
    assert states[1]["tpe_scores"][states[1]["tpe_chosen_index"]] == max(
        states[1]["tpe_scores"]
    )


def test_llm_and_fallback_totals_aggregated(tmp_path) -> None:
    # One invalid receipt burns a retry (2 LLM calls for 1 evaluation); the
    # fitted path reports zero ranker fallbacks.
    ckpt = write_continuation_checkpoint(tmp_path)
    bad = {"configs": [cfg()], "order": [0], "rationale": "too few"}
    good = pool_receipt([0.03, 0.35, 0.40, 0.45, 0.46])

    result = runner.run_cell(
        arm=pool_tpe.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=1,
        budget=1,
        eval_fn=fake_eval_from([("ok", 50.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([bad, good])},
    )

    assert result["status"] == "ok"
    assert result["llm_calls"] == 2
    assert result["llm_input_tokens"] == 20
    assert result["llm_output_tokens"] == 0
    assert result["ranker_fallback_count"] == 0
    assert result["internal_duplicate_count"] == 0
    assert evaluation_events(tmp_path / "out")[0]["proposal"]["dropout"] == 0.03
