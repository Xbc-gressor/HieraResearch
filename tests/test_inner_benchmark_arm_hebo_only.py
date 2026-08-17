from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402
from arms import hebo_common, hebo_only  # noqa: E402
from ib_support import (  # noqa: E402
    cfg,
    evaluation_events,
    fake_eval_from,
    hrow,
    ok_preflight,
    write_checkpoint,
)

# Toy contract has 4 dimensions -> official rand_sample = 1 + 4 = 5.
RAND_SAMPLE = 5


def fake_suggest_fn(calls):
    """Offline hebo_suggest_fn seam: official gate by history length, unique
    in-space suggestions, one Sobol point consumed per quasi step."""

    def suggest(*, search_space, history, seed, scramble_seed, quasi_index,
                initial_suggest_extra):
        calls.append(
            {
                "history": history,
                "seed": seed,
                "scramble_seed": scramble_seed,
                "quasi_index": quasi_index,
                "initial_suggest_extra": initial_suggest_extra,
            }
        )
        n = len(calls)
        params = cfg(
            depth=(n % 8) + 1,
            lr=0.002 + 0.0001 * n,
            dropout=0.3 + 0.01 * n,
            mode="slow",
        )
        if len(history) < RAND_SAMPLE:
            return {"suggestion": params, "mode": "quasi", "quasi_consumed": 1}
        return {
            "suggestion": params,
            "mode": "surrogate",
            "quasi_consumed": 0,
            "front_size": 47,
        }

    return suggest


def run(ckpt, out, *, seed=1, budget=10, calls):
    return runner.run_cell(
        arm=hebo_only.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=seed,
        budget=budget,
        eval_fn=fake_eval_from([("ok", 90.0 - i) for i in range(budget)]),
        preflight_fn=ok_preflight,
        extras={"hebo_suggest_fn": fake_suggest_fn(calls)},
    )


def short_checkpoint(base_dir, name="ckpt"):
    """1 history row + the incumbent -> live finite-unique count starts at 2."""
    return write_checkpoint(
        base_dir,
        name=name,
        history=[hrow(cfg(depth=1, lr=0.0002, dropout=0.05, mode="slow"), 95.0)],
    )


def test_ten_step_cell_sources_and_quasi_consumption(tmp_path) -> None:
    ckpt = short_checkpoint(tmp_path)
    calls = []
    result = run(ckpt, tmp_path / "out", budget=10, calls=calls)

    assert result["status"] == "ok"
    assert result["evaluations"] == 10
    assert result["llm_calls"] == 0
    events = evaluation_events(tmp_path / "out")
    # Live history starts at 2 and grows by 1 per ok outcome: steps 0-2 are
    # quasi (2,3,4 < 5), steps 3+ surrogate.
    sources = [event["source"] for event in events]
    assert sources == ["hebo_quasi"] * 3 + ["hebo_suggest"] * 7
    quasi_positions = [event["arm_state"]["quasi_index"] for event in events]
    assert quasi_positions == [1, 2, 3, 3, 3, 3, 3, 3, 3, 3]
    modes = [event["arm_state"]["hebo_mode"] for event in events]
    assert modes == ["quasi"] * 3 + ["surrogate"] * 7
    # front_size is recorded on surrogate steps only.
    assert all("front_size" not in event["arm_state"] for event in events[:3])
    assert all(event["arm_state"]["front_size"] == 47 for event in events[3:])
    # Every executed proposal is exactly what the seam suggested.
    assert [event["proposal"] for event in events] == [
        call_suggestion(calls, i) for i in range(10)
    ]


def call_suggestion(calls, index):
    n = index + 1
    return cfg(depth=(n % 8) + 1, lr=0.002 + 0.0001 * n, dropout=0.3 + 0.01 * n, mode="slow")


def test_step_seed_and_scramble_are_pure_functions(tmp_path) -> None:
    # F1: the per-step seed is step_seed(ctx.seed, step_index), the scramble
    # seed the fixed trajectory derivation — both checked against the shared
    # helpers the other HEBO arms use.
    seed = 7
    calls = []
    run(short_checkpoint(tmp_path), tmp_path / "out", seed=seed, budget=4, calls=calls)

    assert len(calls) == 4
    for step_index, call in enumerate(calls):
        assert call["seed"] == hebo_common.step_seed(seed, step_index)
        assert call["scramble_seed"] == hebo_common.trajectory_scramble_seed(seed)
        assert call["quasi_index"] == step_index  # all quasi here
        assert call["initial_suggest_extra"] == []
    # Live history grows into the seam in execution order: incumbent first
    # (the freezer did not list it), then the history row, then own outcomes.
    assert calls[0]["history"] == [
        {"params": cfg(), "score": 100.0},
        {"params": cfg(depth=1, lr=0.0002, dropout=0.05, mode="slow"), "score": 95.0},
    ]
    assert calls[1]["history"][-1] == {"params": call_suggestion(calls, 0), "score": 90.0}


def test_suggest_failure_is_arm_error(tmp_path) -> None:
    def broken(**kwargs):
        raise RuntimeError("boom")

    result = runner.run_cell(
        arm=hebo_only.ARM,
        checkpoint_dir=short_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        budget=2,
        eval_fn=fake_eval_from([("ok", 90.0)] * 2),
        preflight_fn=ok_preflight,
        extras={"hebo_suggest_fn": broken},
    )
    assert result["status"] == "arm_error"
    assert "boom" in result["reason"]
    assert result["evaluations"] == 0
