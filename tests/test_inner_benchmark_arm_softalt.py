from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402
from arms import softalt  # noqa: E402
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


def test_alternates_hebo_rerank_with_direct_llm_choice(tmp_path) -> None:
    receipts = [pool_receipt(0.005 + 0.002 * index) for index in range(4)]
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
        arm=softalt.ARM,
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
    assert result["ranker_fallback_count"] == 0
    events = evaluation_events(tmp_path / "out")
    assert [event["source"] for event in events] == [
        "softalt_hebo",
        "softalt_llm",
        "softalt_hebo",
        "softalt_llm",
    ]
    assert [event["proposal"] for event in events] == [
        receipts[0]["configs"][2],
        receipts[1]["configs"][0],
        receipts[2]["configs"][3],
        receipts[3]["configs"][0],
    ]
    assert [event["arm_state"]["selection"] for event in events] == [
        "hebo_rerank",
        "llm_direct",
        "hebo_rerank",
        "llm_direct",
    ]
    assert len(rank_calls) == 2
    assert len(rank_calls[1]["history"]) == len(rank_calls[0]["history"]) + 2
    assert session.asks[0] == {}
    assert session.asks[1]
