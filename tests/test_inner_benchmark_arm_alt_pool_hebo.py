from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import objective  # noqa: E402
import runner  # noqa: E402
from arms import alt_pool_hebo  # noqa: E402
from ib_support import (  # noqa: E402
    BASE,
    cfg,
    evaluation_events,
    fake_eval_from,
    hrow,
    ok_preflight,
    write_checkpoint,
)

# Toy contract has 4 dimensions -> official rand_sample = 1 + 4 = 5, so with
# a 1-row checkpoint (live count starts at 2) BO steps 0/2 are quasi and BO
# steps >= 4 are surrogate.


class FakeSession:
    def __init__(self, script):
        self._script = list(script)
        self.asks = []

    def ask(self, extra=None):
        self.asks.append(extra)
        return self._script.pop(0)

    def totals(self):
        return {
            "llm_calls": len(self.asks),
            "llm_input_tokens": 10 * len(self.asks),
            "llm_output_tokens": 0,
        }


class RecordingFactory:
    def __init__(self, script):
        self._script = script
        self.created = []

    def __call__(self, role_name, first_extras=None):
        session = FakeSession(self._script)
        self.created.append(
            {"role": role_name, "first_extras": first_extras, "session": session}
        )
        return session


def pool_receipt(lr_base, *, depths=(1, 2, 3, 4, 5), order=None, configs=None):
    return {
        "configs": configs
        or [
            cfg(depth=depth, lr=lr_base + 0.0001 * index, dropout=0.3, mode="slow")
            for index, depth in enumerate(depths)
        ],
        "order": order if order is not None else [0, 1, 2, 3, 4],
        "rationale": "scripted pool",
    }


def fake_suggest_fn(calls):
    """Offline seam: official gate by history length; unique in-space BO points."""

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
        if len(history) < 5:
            return {"suggestion": params, "mode": "quasi", "quasi_consumed": 1}
        return {
            "suggestion": params,
            "mode": "surrogate",
            "quasi_consumed": 0,
            "front_size": 47,
        }

    return suggest


def short_checkpoint(base_dir, name="ckpt"):
    return write_checkpoint(
        base_dir,
        name=name,
        history=[hrow(cfg(depth=1, lr=0.0002, dropout=0.05, mode="slow"), 95.0)],
    )


def run(ckpt, out, *, seed=1, budget, factory, calls, preflight_fn=ok_preflight):
    return runner.run_cell(
        arm=alt_pool_hebo.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=seed,
        budget=budget,
        eval_fn=fake_eval_from([("ok", 90.0 - 0.5 * i) for i in range(budget)]),
        preflight_fn=preflight_fn,
        extras={"session_factory": factory, "hebo_suggest_fn": fake_suggest_fn(calls)},
    )


def expected_bo_source(step: int) -> str:
    """BO source by the actual mode at that step: live history starts at 2
    and every step executes ok, so BO0 sees 2, BO2 sees 4, BO4 sees 6 —
    quasi only below the official rand_sample = 5."""
    return "alt_hebo_quasi" if step < 4 else "alt_hebo"


def test_24_step_alternation_and_pairs(tmp_path) -> None:
    budget = 24
    factory = RecordingFactory(
        [pool_receipt(0.005 + 0.001 * k) for k in range(12)]
    )
    calls = []
    result = run(short_checkpoint(tmp_path), tmp_path / "out",
                 budget=budget, factory=factory, calls=calls)

    assert result["status"] == "ok"
    assert result["evaluations"] == budget
    # E2 primary metric surface: at_24 exists and equals the final value.
    assert result["relative_improvement_at"][24] == result["final_relative_improvement"]
    assert result["llm_calls"] == 12
    assert result["llm_input_tokens"] == 120
    assert result["internal_duplicate_count"] == 0
    assert result["ranker_fallback_count"] == 0
    events = evaluation_events(tmp_path / "out")
    assert len(events) == budget
    for step, event in enumerate(events):
        if step % 2 == 0:
            assert event["source"] == expected_bo_source(step), (step, event["source"])
            assert event["arm_state"]["hebo_mode"] == ("quasi" if step < 4 else "surrogate")
        else:
            assert event["source"] == "alt_llm"
            assert event["arm_state"]["literal_rank1"] is True
    # Exactly 12 complete BO -> LLM pairs.
    pairs = sum(
        1
        for k in range(12)
        if events[2 * k]["source"].startswith("alt_hebo")
        and events[2 * k + 1]["source"] == "alt_llm"
    )
    assert pairs == 12
    # BO steps carry the suggest step seeds at their (even) step indexes.
    assert len(calls) == 12
    from arms import hebo_common

    for k, call in enumerate(calls):
        assert call["seed"] == hebo_common.step_seed(1, 2 * k)
        assert call["initial_suggest_extra"] == []
    # quasi_index stops advancing once the surrogate phase starts (BO4 on).
    quasi_positions = [call["quasi_index"] for call in calls]
    assert quasi_positions == [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


def test_first_ask_after_bo0_and_tagged_messages(tmp_path) -> None:
    budget = 6
    factory = RecordingFactory([pool_receipt(0.005 + 0.001 * k) for k in range(3)])
    calls = []
    result = run(short_checkpoint(tmp_path), tmp_path / "out",
                 budget=budget, factory=factory, calls=calls)

    assert result["status"] == "ok"
    # One session, created after BO0 executed: its first message renders the
    # LIVE history including the BO0 row.
    assert len(factory.created) == 1
    first_extras = factory.created[0]["first_extras"]
    assert first_extras["protocol"] == alt_pool_hebo.ALT_PROTOCOL
    assert "[alt_hebo_quasi]" in first_extras["history"]
    assert "Current incumbent" in first_extras["incumbent"]
    session = factory.created[0]["session"]
    assert len(session.asks) == 3
    # Every BO outcome rides the next ask with its mode's tag.
    import llm

    ask0_outcome = session.asks[0][llm.OUTCOME_KEY]
    assert "[hebo_quasi]" in ask0_outcome  # BO0
    assert "Sobol warmup point" in ask0_outcome
    ask1_outcome = session.asks[1][llm.OUTCOME_KEY]
    assert "[hebo_quasi]" in ask1_outcome  # BO2
    ask2_outcome = session.asks[2][llm.OUTCOME_KEY]
    assert "[hebo_probe]" in ask2_outcome  # BO4 (surrogate phase)
    assert "executed by the BO surrogate" in ask2_outcome


def test_rank1_duplicate_falls_back_to_filtered_pool(tmp_path) -> None:
    # F2: rank-1 duplicates the incumbent -> filtered by ask_pool; the arm
    # executes the filtered pool[0], counts the duplicate, pushes a
    # correction, and the parity keeps alternating.
    dup_configs = [dict(BASE)] + pool_receipt(0.005)["configs"][:4]
    receipts = [
        pool_receipt(0.005, configs=dup_configs),
        pool_receipt(0.009),
    ]
    factory = RecordingFactory(receipts)
    calls = []
    result = run(short_checkpoint(tmp_path), tmp_path / "out",
                 budget=4, factory=factory, calls=calls)

    assert result["status"] == "ok"
    assert result["internal_duplicate_count"] == 1
    events = evaluation_events(tmp_path / "out")
    assert [event["source"] for event in events] == [
        "alt_hebo_quasi",
        "alt_llm",
        "alt_hebo_quasi",
        "alt_llm",
    ]
    llm_event = events[1]
    assert llm_event["arm_state"]["literal_rank1"] is False
    assert llm_event["proposal"] == dup_configs[1]  # filtered pool[0]
    assert llm_event["arm_state"]["pool_duplicate_mask"][0] is True
    # The correction note reached the next ask.
    import llm

    session = factory.created[0]["session"]
    next_outcome = session.asks[1][llm.OUTCOME_KEY]
    assert "correction:" in next_outcome
    assert "rank-1" in next_outcome


def test_llm_preflight_rejection_retries_same_phase(tmp_path) -> None:
    # An LLM proposal rejected by task preflight consumes no budget, so the
    # arm retries LLM and preserves alternation over objective evaluations.
    rejecting = pool_receipt(0.005)["configs"][0]

    def preflight(
        candidate_path,
        params,
        *,
        preflight_fn,
        per_runtime_limit,
        python_cmd=None,
    ):
        if params == rejecting:
            return objective.PreflightOutcome(
                status="failed", detail="scripted task reject"
            )
        return objective.PreflightOutcome(status="ok", detail=None)

    factory = RecordingFactory([pool_receipt(0.005), pool_receipt(0.009)])
    calls = []
    result = run(
        short_checkpoint(tmp_path),
        tmp_path / "out",
        budget=3,
        factory=factory,
        calls=calls,
        preflight_fn=preflight,
    )

    assert result["status"] == "ok"
    assert result["counts"]["task_preflight_rejected"] == 1
    # The rejection is not an evaluation: the three evaluated sources remain
    # BO0, LLM1, BO2.
    events = evaluation_events(tmp_path / "out")
    assert [event["source"] for event in events] == [
        "alt_hebo_quasi",  # step 0
        "alt_llm",  # step 1 retry succeeded
        "alt_hebo_quasi",  # step 2
    ]
    # The rejection was reported into the session and rode the next ask.
    import llm

    session = factory.created[0]["session"]
    assert "REJECTED by the task preflight" in session.asks[1][llm.OUTCOME_KEY]


def test_bo_preflight_rejection_is_tagged_and_retries_bo(tmp_path) -> None:
    # The first BO point is rejected. Its Sobol position is consumed, BO is
    # retried before the LLM phase, and the delayed first ask receives both
    # the tagged rejection and the successful BO outcome in order.
    rejecting = cfg(depth=2, lr=0.0021, dropout=0.31, mode="slow")

    def preflight(
        candidate_path,
        params,
        *,
        preflight_fn,
        per_runtime_limit,
        python_cmd=None,
    ):
        if params == rejecting:
            return objective.PreflightOutcome(
                status="failed", detail="scripted BO reject"
            )
        return objective.PreflightOutcome(status="ok", detail=None)

    factory = RecordingFactory([pool_receipt(0.005)])
    calls = []
    result = run(
        short_checkpoint(tmp_path),
        tmp_path / "out",
        budget=2,
        factory=factory,
        calls=calls,
        preflight_fn=preflight,
    )

    assert result["status"] == "ok"
    assert result["counts"]["task_preflight_rejected"] == 1
    events = evaluation_events(tmp_path / "out")
    assert [event["source"] for event in events] == ["alt_hebo_quasi", "alt_llm"]
    assert len(calls) == 2
    assert [call["quasi_index"] for call in calls] == [0, 1]
    assert calls[0]["seed"] != calls[1]["seed"]

    import llm

    first_ask = factory.created[0]["session"].asks[0][llm.OUTCOME_KEY]
    assert first_ask.count("[hebo_quasi]") == 2
    assert "REJECTED by the task preflight (scripted BO reject)" in first_ask
    assert first_ask.index("REJECTED") < first_ask.index("score =")
