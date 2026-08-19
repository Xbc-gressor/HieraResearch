from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402
from arms import bernsalt  # noqa: E402
from ib_support import (  # noqa: E402
    cfg,
    evaluation_events,
    fake_eval_from,
    hrow,
    ok_preflight,
    write_checkpoint,
)


class FakeSession:
    def __init__(self, receipts):
        self.receipts = list(receipts)
        self.asks = []

    def ask(self, extra=None):
        self.asks.append(extra)
        return self.receipts.pop(0)

    def totals(self):
        return {
            "llm_calls": len(self.asks),
            "llm_input_tokens": 10 * len(self.asks),
            "llm_output_tokens": 0,
        }


def pool_receipt(lr_base):
    return {
        "configs": [
            cfg(depth=index + 1, lr=lr_base + 0.0001 * index,
                dropout=0.3, mode="slow")
            for index in range(5)
        ],
        "order": [0, 1, 2, 3, 4],
        "rationale": "scripted pool",
    }


def continuation_checkpoint(tmp_path):
    return write_checkpoint(
        tmp_path,
        regime="continuation",
        stratum="cont_improved",
        history=[
            hrow(
                cfg(depth=index, lr=0.0001 * (index + 1),
                    dropout=0.05 * index, mode="slow"),
                90.0 + index,
            )
            for index in range(1, 9)
        ],
    )


def test_llm_direct_when_probability_zero(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bernsalt, "SHIFT", 1000.0)
    receipts = [pool_receipt(0.005 + 0.002 * index) for index in range(4)]
    session = FakeSession(receipts)

    def rank_fn(**kwargs):
        raise AssertionError("rank_fn must not be called at p=0")

    result = runner.run_cell(
        arm=bernsalt.ARM,
        checkpoint_dir=continuation_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        budget=4,
        eval_fn=fake_eval_from([("ok", 80.0 - index) for index in range(4)]),
        preflight_fn=ok_preflight,
        extras={
            "session_factory": lambda role_name, first_extras=None: session,
            "hebo_rank_fn": rank_fn,
        },
    )

    assert result["status"] == "ok"
    assert result["llm_calls"] == 4
    events = evaluation_events(tmp_path / "out")
    assert [event["source"] for event in events] == ["bernsalt_llm"] * 4
    assert [event["proposal"] for event in events] == [
        receipts[index]["configs"][0] for index in range(4)
    ]
    assert [event["arm_state"]["t"] for event in events] == [9, 10, 11, 12]
    assert [event["arm_state"]["p_hebo"] for event in events] == [0.0] * 4
    assert [event["arm_state"]["selection"] for event in events] == [
        "llm_direct"
    ] * 4
    assert all(
        "[llm_direct]" in str(extra) for extra in session.asks
    )


def test_hebo_rerank_when_probability_one(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bernsalt, "SHIFT", 0.0)
    monkeypatch.setattr(bernsalt, "SCALE", 0.001)
    receipts = [pool_receipt(0.005 + 0.002 * index) for index in range(2)]
    session = FakeSession(receipts)
    rank_calls = []
    rank_values = iter(
        [
            [[0.0], [1.0], [4.0], [2.0], [3.0]],
            [[0.0], [1.0], [2.0], [4.0], [3.0]],
        ]
    )

    def rank_fn(*, search_space, history, pool, seed):
        rank_calls.append({"history": history, "pool": pool})
        return next(rank_values)

    result = runner.run_cell(
        arm=bernsalt.ARM,
        checkpoint_dir=continuation_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        budget=2,
        eval_fn=fake_eval_from([("ok", 80.0 - index) for index in range(2)]),
        preflight_fn=ok_preflight,
        extras={
            "session_factory": lambda role_name, first_extras=None: session,
            "hebo_rank_fn": rank_fn,
        },
    )

    assert result["status"] == "ok"
    assert result["ranker_fallback_count"] == 0
    events = evaluation_events(tmp_path / "out")
    assert [event["source"] for event in events] == ["bernsalt_hebo"] * 2
    assert [event["proposal"] for event in events] == [
        receipts[0]["configs"][2],
        receipts[1]["configs"][3],
    ]
    assert [event["arm_state"]["selection"] for event in events] == [
        "hebo_rerank"
    ] * 2
    assert [event["arm_state"]["p_hebo"] for event in events] == [1.0] * 2
    assert len(rank_calls) == 2
    assert "[hebo_rerank]" in str(session.asks[0])
