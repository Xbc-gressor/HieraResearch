from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402
from arms import pool3_hebo_mace, pool7_hebo_mace, pool_hebo_mace  # noqa: E402
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
    return lambda role_name, first_extras=None: FakeSession(script)


def pool_receipt(lr_base, *, depths=(1, 2, 3, 4, 5), order=None):
    """A valid POOL=5 receipt over the toy contract; lr_base separates steps."""
    return {
        "configs": [
            cfg(depth=depth, lr=lr_base + 0.0001 * index, dropout=0.3, mode="slow")
            for index, depth in enumerate(depths)
        ],
        "order": order if order is not None else [0, 1, 2, 3, 4],
        "rationale": "scripted pool",
    }


def test_pool_size_variants_request_and_execute_configured_pool(tmp_path) -> None:
    variants = ((pool3_hebo_mace.ARM, 3), (pool7_hebo_mace.ARM, 7))
    for arm, pool_size in variants:
        ckpt = continuation_checkpoint(tmp_path, name=f"ckpt-{pool_size}")
        receipt = pool_receipt(0.005, depths=tuple(range(1, pool_size + 1)))
        receipt["order"] = list(range(pool_size))
        values = [[float(index)] * 3 for index in range(pool_size)]
        out = tmp_path / f"out-{pool_size}"

        result = runner.run_cell(
            arm=arm,
            checkpoint_dir=ckpt,
            out_dir=out,
            seed=1,
            budget=1,
            eval_fn=fake_eval_from([("ok", 50.0)]),
            preflight_fn=ok_preflight,
            extras={
                "session_factory": fake_session_factory([receipt]),
                "hebo_rank_fn": scripted_rank_fn([values]),
            },
        )

        assert result["status"] == "ok"
        (event,) = evaluation_events(out)
        assert event["source"] == arm.name
        assert event["arm_state"]["pool_size"] == pool_size
        assert event["arm_state"]["chosen_index"] == pool_size - 1


def continuation_checkpoint(base_dir, name="ckpt"):
    """8 finite unique history rows (+ the incumbent) => live WARMUP is met."""
    history = [
        hrow(
            cfg(depth=index, lr=0.0001 * (index + 1), dropout=0.05 * index, mode="slow"),
            90.0 + index,
        )
        for index in range(1, 9)
    ]
    return write_checkpoint(
        base_dir,
        name=name,
        regime="continuation",
        stratum="cont_improved",
        history=history,
    )


def scripted_rank_fn(script, calls=None):
    """fake hebo_rank_fn: one acquisition-values list per call, in call order."""

    def rank_fn(*, search_space, history, pool, seed):
        if calls is not None:
            calls.append(
                {"search_space": search_space, "history": history, "pool": pool, "seed": seed}
            )
        return script.pop(0)

    return rank_fn


def test_default_ranker_uses_root_uv_interpreter() -> None:
    completed = mock.Mock(returncode=0, stdout='{"values": [[1.0]]}', stderr="")
    with mock.patch.object(pool_hebo_mace.subprocess, "run", return_value=completed) as run:
        values = pool_hebo_mace._subprocess_rank_fn(
            search_space={"x": ["float", 0.0, 1.0]},
            history=[{"params": {"x": 0.5}, "score": 1.0}],
            pool=[{"x": 0.25}],
            seed=7,
        )

    assert values == [[1.0]]
    assert run.call_args.args[0] == [
        sys.executable,
        str(pool_hebo_mace.HEBO_PROJECT_DIR / "rank.py"),
    ]


def run(ckpt, out, *, seed=1, budget=1, receipts, values_script, eval_outcomes=None, calls=None):
    extras = {"session_factory": fake_session_factory(receipts)}
    if values_script is not None:
        extras["hebo_rank_fn"] = scripted_rank_fn(values_script, calls)
    else:
        # Any call is a contract violation for tests that never reach the ranker.
        extras["hebo_rank_fn"] = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("rank_fn must not be called")
        )
    return runner.run_cell(
        arm=pool_hebo_mace.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=seed,
        budget=budget,
        eval_fn=fake_eval_from(eval_outcomes or [("ok", 50.0)] * budget),
        preflight_fn=ok_preflight,
        extras=extras,
    )


def test_warmup_fallback_executes_proposer_rank1(tmp_path) -> None:
    # Live finite-unique history is 3 (< WARMUP=8): every step executes pool[0]
    # with ranker_fallback=true and the ranker is never invoked.
    ckpt = write_checkpoint(
        tmp_path,
        history=[
            hrow(cfg(depth=1, lr=0.0002, dropout=0.05, mode="slow"), 95.0),
            hrow(cfg(depth=2, lr=0.0003, dropout=0.1, mode="slow"), 96.0),
        ],
    )
    receipts = [pool_receipt(0.005), pool_receipt(0.006)]

    result = run(
        ckpt,
        tmp_path / "out",
        budget=2,
        receipts=receipts,
        values_script=None,
        eval_outcomes=[("ok", 90.0), ("ok", 80.0)],
    )

    assert result["status"] == "ok"
    events = evaluation_events(tmp_path / "out")
    assert len(events) == 2
    for step, event in enumerate(events):
        assert event["proposal"] == receipts[step]["configs"][0]
        assert event["arm_state"]["ranker_fallback"] is True
        assert event["arm_state"]["chosen_index"] == 0
    assert result["ranker_fallback_count"] == 2
    assert result["llm_calls"] == 2


def test_unique_dominant_front_member_is_executed(tmp_path) -> None:
    ckpt = continuation_checkpoint(tmp_path)
    receipt = pool_receipt(0.005)
    # Member 2 dominates every other member on all three objectives.
    values = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],
        [2.0, 2.0, 1.0],
        [1.0, 0.0, 0.0],
    ]

    result = run(ckpt, tmp_path / "out", receipts=[receipt], values_script=[values])

    assert result["status"] == "ok"
    assert result["best_config"] == receipt["configs"][2]
    (event,) = evaluation_events(tmp_path / "out")
    assert event["proposal"] == receipt["configs"][2]
    assert event["arm_state"]["ranker_fallback"] is False
    assert event["arm_state"]["pareto_front"] == [2]
    assert event["arm_state"]["chosen_index"] == 2
    assert event["arm_state"]["acquisition_values"] == values
    assert result["ranker_fallback_count"] == 0


def test_tied_front_uses_seeded_uniform_choice(tmp_path) -> None:
    # Front is exactly {0, 1}: 2/3/4 are dominated by member 0 or 1, and the
    # two front members do not dominate each other.
    values = [[0.8, 0.6], [0.6, 0.8], [0.5, 0.5], [0.7, 0.2], [0.2, 0.7]]

    chosen_by_seed = {}
    for seed in range(8):
        ckpt = continuation_checkpoint(tmp_path, name=f"ck{seed}")
        receipt = pool_receipt(0.005)
        result = run(
            ckpt,
            tmp_path / f"out{seed}",
            seed=seed,
            receipts=[receipt],
            values_script=[list(values)],
        )
        assert result["status"] == "ok"
        (event,) = evaluation_events(tmp_path / f"out{seed}")
        assert event["arm_state"]["pareto_front"] == [0, 1]
        chosen_by_seed[seed] = event["arm_state"]["chosen_index"]
        assert event["proposal"] == receipt["configs"][chosen_by_seed[seed]]
    # Uniform draw over the tied front: both members win across seeds (the
    # draws are deterministic per cell seed — np_rng.integers gives both 0
    # and 1 over seeds 0..7), so this cannot pass via "always front[0]".
    assert set(chosen_by_seed.values()) == {0, 1}

    # Same seed => identical selection (run the seed-5 cell again verbatim).
    receipt = pool_receipt(0.005)
    run(
        continuation_checkpoint(tmp_path, name="ck5b"),
        tmp_path / "out5b",
        seed=5,
        receipts=[receipt],
        values_script=[list(values)],
    )
    (event,) = evaluation_events(tmp_path / "out5b")
    assert event["arm_state"]["chosen_index"] == chosen_by_seed[5]


def test_ranker_failure_is_arm_error(tmp_path) -> None:
    ckpt = continuation_checkpoint(tmp_path)

    def broken_rank_fn(**kwargs):
        raise RuntimeError("boom")

    result = runner.run_cell(
        arm=pool_hebo_mace.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=1,
        budget=1,
        eval_fn=fake_eval_from([("ok", 50.0)]),
        preflight_fn=ok_preflight,
        extras={
            "session_factory": fake_session_factory([pool_receipt(0.005)]),
            "hebo_rank_fn": broken_rank_fn,
        },
    )

    assert result["status"] == "arm_error"
    assert "boom" in result["reason"]
    assert result["evaluations"] == 0
    assert result["llm_calls"] == 1  # the proposer was asked before ranking


def test_rank_fn_receives_filtered_pool_and_live_fittable_history(tmp_path) -> None:
    ckpt = continuation_checkpoint(tmp_path)
    # Rank-1 member duplicates the incumbent: filtered before ranking, so the
    # ranker sees a 4-member pool.
    receipt = {
        "configs": [dict(BASE)] + pool_receipt(0.005)["configs"][:4],
        "order": [0, 1, 2, 3, 4],
        "rationale": "dup at rank1",
    }
    calls = []
    values = [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.1]]

    result = run(
        ckpt,
        tmp_path / "out",
        seed=7,
        receipts=[receipt],
        values_script=[values],
        calls=calls,
    )

    assert result["status"] == "ok"
    assert result["internal_duplicate_count"] == 1
    assert len(calls) == 1
    call = calls[0]
    assert call["seed"] == 7
    assert call["search_space"]["lr"] == ("float", 0.0001, 0.1, "log")
    assert len(call["pool"]) == 4
    assert dict(BASE) not in call["pool"]
    # Live fittable history: the checkpoint incumbent first, then the 8
    # unique history rows; all finite.
    assert len(call["history"]) == 9
    assert call["history"][0] == {"params": BASE, "score": 100.0}
    assert all(isinstance(row["score"], float) for row in call["history"])
    # Dominant member 1 of the filtered pool was executed.
    (event,) = evaluation_events(tmp_path / "out")
    assert event["proposal"] == receipt["configs"][2]


def test_two_step_cell_aggregates_counts(tmp_path) -> None:
    ckpt = continuation_checkpoint(tmp_path)
    receipts = [pool_receipt(0.005), pool_receipt(0.007)]
    values = [
        [[3.0, 3.0, 3.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 0.0, 2.0]],
        [[0.0, 1.0, 0.0], [3.0, 3.0, 3.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [2.0, 0.0, 1.0]],
    ]

    result = run(
        ckpt,
        tmp_path / "out",
        budget=2,
        receipts=receipts,
        values_script=values,
        eval_outcomes=[("ok", 90.0), ("ok", 80.0)],
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 2
    assert result["llm_calls"] == 2
    assert result["llm_input_tokens"] == 20
    assert result["ranker_fallback_count"] == 0
    assert result["internal_duplicate_count"] == 0
    events = evaluation_events(tmp_path / "out")
    assert [event["proposal"] for event in events] == [
        receipts[0]["configs"][0],
        receipts[1]["configs"][1],
    ]
    assert all(event["arm_state"]["ranker_fallback"] is False for event in events)
