from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402
from arms import hebo_common, hebo_only, mixup_pool_hebo  # noqa: E402
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
    """session_factory that records every creation (role + first_extras)."""

    def __init__(self, script):
        self._script = script
        self.created = []

    def __call__(self, role_name, first_extras=None):
        session = FakeSession(self._script)
        self.created.append(
            {"role": role_name, "first_extras": first_extras, "session": session}
        )
        return session


def pool_receipt(lr_base, *, depths=(1, 2, 3, 4, 5), order=None):
    return {
        "configs": [
            cfg(depth=depth, lr=lr_base + 0.0001 * index, dropout=0.3, mode="slow")
            for index, depth in enumerate(depths)
        ],
        "order": order if order is not None else [0, 1, 2, 3, 4],
        "rationale": "scripted pool",
    }


def fake_suggest_fn(calls):
    """Same offline seam shape as the hebo_only tests: official gate by
    history length, unique in-space suggestions."""

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


def short_checkpoint(base_dir, name="ckpt"):
    """1 history row + the incumbent -> live finite-unique count starts at 2."""
    return write_checkpoint(
        base_dir,
        name=name,
        history=[hrow(cfg(depth=1, lr=0.0002, dropout=0.05, mode="slow"), 95.0)],
    )


def run(ckpt, out, *, seed=1, budget, factory, calls):
    return runner.run_cell(
        arm=mixup_pool_hebo.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=seed,
        budget=budget,
        eval_fn=fake_eval_from([("ok", 90.0 - i) for i in range(budget)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": factory, "hebo_suggest_fn": fake_suggest_fn(calls)},
    )


def test_quasi_prefix_creates_no_llm_session(tmp_path) -> None:
    # Budget 3 keeps live history below rand_sample=5 the whole cell.
    factory = RecordingFactory([pool_receipt(0.005)])
    calls = []
    result = run(short_checkpoint(tmp_path), tmp_path / "out", budget=3,
                 factory=factory, calls=calls)

    assert result["status"] == "ok"
    assert result["evaluations"] == 3
    assert factory.created == []  # no session, no pool request during warmup
    assert result["llm_calls"] == 0
    sources = [event["source"] for event in evaluation_events(tmp_path / "out")]
    assert sources == ["hebo_quasi"] * 3
    # Official quasi is not a ranker fallback.
    assert result["ranker_fallback_count"] == 0


def test_surrogate_phase_session_pool_and_arm_state(tmp_path) -> None:
    factory = RecordingFactory([pool_receipt(0.005), pool_receipt(0.007)])
    calls = []
    result = run(short_checkpoint(tmp_path), tmp_path / "out", budget=5,
                 factory=factory, calls=calls)

    assert result["status"] == "ok"
    events = evaluation_events(tmp_path / "out")
    assert [event["source"] for event in events] == (
        ["hebo_quasi"] * 3 + ["mixup_hebo"] * 2
    )
    # The session is created exactly once, at the first surrogate step.
    assert len(factory.created) == 1
    first_extras = factory.created[0]["first_extras"]
    assert first_extras["protocol"] == mixup_pool_hebo.MIXUP_PROTOCOL
    # The first message renders the LIVE history (this cell's quasi rows
    # included) and the LIVE incumbent/budget at creation time.
    assert "[hebo_quasi]" in first_extras["history"]
    assert "Current incumbent" in first_extras["incumbent"]
    assert "score: 88.0" in first_extras["incumbent"]  # after 3 improving evals
    assert "bout: 2." in first_extras["budget"]  # 5 - 3 consumed
    # The filtered pool is the initial_suggest_extra of the surrogate calls.
    receipt0 = pool_receipt(0.005)
    assert calls[3]["initial_suggest_extra"] == receipt0["configs"]
    assert calls[4]["initial_suggest_extra"] == pool_receipt(0.007)["configs"]
    assert calls[3]["quasi_index"] == 3  # the quasi prefix was consumed
    # Surrogate arm_state: pool persistence + front size + seed geometry.
    arm_state = events[3]["arm_state"]
    assert arm_state["pool_configs"] == receipt0["configs"]
    assert arm_state["front_size"] == 47
    assert arm_state["hit_seed"] is False
    assert isinstance(arm_state["nearest_seed_z_dist"], float)
    assert result["llm_calls"] == 2
    assert result["ranker_fallback_count"] == 0


def test_paired_quasi_prefix_and_step_seeds_match_hebo_only(tmp_path) -> None:
    # F1: same (checkpoint, seed) — the quasi prefix is bit-identical and
    # every step hands the seam the same pure-function step seed.
    seed = 11
    hebo_calls, mixup_calls = [], []
    runner.run_cell(
        arm=hebo_only.ARM,
        checkpoint_dir=short_checkpoint(tmp_path, "a"),
        out_dir=tmp_path / "out_hebo",
        seed=seed,
        budget=5,
        eval_fn=fake_eval_from([("ok", 90.0 - i) for i in range(5)]),
        preflight_fn=ok_preflight,
        extras={"hebo_suggest_fn": fake_suggest_fn(hebo_calls)},
    )
    run(short_checkpoint(tmp_path, "b"), tmp_path / "out_mixup", seed=seed,
        budget=5, factory=RecordingFactory(
            [pool_receipt(0.005), pool_receipt(0.007)]), calls=mixup_calls)

    assert len(hebo_calls) == len(mixup_calls) == 5
    for step_index, (hebo_call, mixup_call) in enumerate(zip(hebo_calls, mixup_calls)):
        assert hebo_call["seed"] == mixup_call["seed"] == hebo_common.step_seed(seed, step_index)
        assert hebo_call["scramble_seed"] == mixup_call["scramble_seed"]
        assert hebo_call["quasi_index"] == mixup_call["quasi_index"]
        assert hebo_call["history"] == mixup_call["history"]
    # The quasi prefix proposals are identical; afterwards mixup carries the
    # pool as initial_suggest_extra — the only structural difference.
    for call in range(3):
        assert hebo_calls[call]["initial_suggest_extra"] == mixup_calls[call]["initial_suggest_extra"] == []
    hebo_events = evaluation_events(tmp_path / "out_hebo")
    mixup_events = evaluation_events(tmp_path / "out_mixup")
    assert [e["proposal"] for e in hebo_events[:3]] == [e["proposal"] for e in mixup_events[:3]]
    assert mixup_calls[3]["initial_suggest_extra"] != []
    assert hebo_calls[3]["initial_suggest_extra"] == []
