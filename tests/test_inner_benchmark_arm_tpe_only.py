from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402
from arms import tpe_only  # noqa: E402
from ib_support import (  # noqa: E402
    evaluation_events,
    fake_eval_from,
    ok_preflight,
    write_checkpoint,
)


def exploding_session_factory(role_name, first_extras=None):
    raise AssertionError("tpe_only must never create an LLM session")


def test_full_budget_without_llm(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)
    out = tmp_path / "out"

    result = runner.run_cell(
        arm=tpe_only.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=5,
        budget=6,
        eval_fn=fake_eval_from([("ok", 90.0 - index) for index in range(6)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": exploding_session_factory},
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 6
    events = evaluation_events(out)
    assert [event["source"] for event in events] == ["tpe"] * 6
    assert result["llm_calls"] == 0
    assert result["llm_input_tokens"] == 0


def test_same_seed_replays_proposal_sequence(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)

    def run(out_dir, seed):
        return runner.run_cell(
            arm=tpe_only.ARM,
            checkpoint_dir=ckpt,
            out_dir=out_dir,
            seed=seed,
            budget=5,
            eval_fn=fake_eval_from([("ok", 90.0 - index) for index in range(5)]),
            preflight_fn=ok_preflight,
        )

    out_a, out_b, out_c = (tmp_path / name for name in ("a", "b", "c"))
    assert run(out_a, seed=7)["status"] == "ok"
    assert run(out_b, seed=7)["status"] == "ok"
    assert run(out_c, seed=8)["status"] == "ok"
    proposals = [
        [event["proposal"] for event in evaluation_events(out)]
        for out in (out_a, out_b, out_c)
    ]
    encode = lambda rows: json.dumps(rows, sort_keys=True, default=str)
    assert encode(proposals[0]) == encode(proposals[1])  # reproducible per seed
    assert encode(proposals[0]) != encode(proposals[2])  # seed actually matters
