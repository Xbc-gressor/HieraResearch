from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402
import llm  # noqa: E402
from arms import hebo_common, hebo_only, hands, pool  # noqa: E402
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


def fake_suggest_fn(calls, surrogate_script):
    """Offline seam: official gate by history length; surrogate steps follow
    the scripted provenance ('pool:<index>' executes that literal pool
    member, 'front' executes a fixed evolved-side config)."""
    front_params = cfg(depth=8, lr=0.05, dropout=0.45, mode="fast")

    def suggest(*, search_space, history, seed, scramble_seed, quasi_index,
                initial_suggest_extra, pool=None):
        calls.append(
            {
                "history": history,
                "seed": seed,
                "scramble_seed": scramble_seed,
                "quasi_index": quasi_index,
                "initial_suggest_extra": initial_suggest_extra,
                "pool": list(pool) if pool else [],
            }
        )
        n = len(calls)
        if len(history) < RAND_SAMPLE:
            return {
                "suggestion": cfg(
                    depth=(n % 8) + 1,
                    lr=0.002 + 0.0001 * n,
                    dropout=0.3 + 0.01 * n,
                    mode="slow",
                ),
                "mode": "quasi",
                "quasi_consumed": 1,
            }
        step = surrogate_script.pop(0)
        if step.startswith("pool:"):
            index = int(step.split(":")[1])
            return {
                "suggestion": pool[index],
                "mode": "surrogate",
                "quasi_consumed": 0,
                "front_size": 47,
                "chosen_from": "pool",
                "chosen_pool_index": index,
                "union_front_size": 3,
                "pool_survivor_indices": [index],
            }
        return {
            "suggestion": dict(front_params),
            "mode": "surrogate",
            "quasi_consumed": 0,
            "front_size": 47,
            "chosen_from": "front",
            "chosen_pool_index": None,
            "union_front_size": 2,
            "pool_survivor_indices": [],
        }

    return suggest


def short_checkpoint(base_dir, name="ckpt"):
    """1 history row + the incumbent -> live finite-unique count starts at 2."""
    return write_checkpoint(
        base_dir,
        name=name,
        history=[hrow(cfg(depth=1, lr=0.0002, dropout=0.05, mode="slow"), 95.0)],
    )


def run(ckpt, out, *, seed=1, budget, factory, calls, script):
    return runner.run_cell(
        arm=hands.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=seed,
        budget=budget,
        eval_fn=fake_eval_from([("ok", 90.0 - i) for i in range(budget)]),
        preflight_fn=ok_preflight,
        extras={
            "session_factory": factory,
            "hebo_suggest_fn": fake_suggest_fn(calls, script),
        },
    )


def test_quasi_prefix_creates_no_llm_session(tmp_path) -> None:
    # Budget 3 keeps live history below rand_sample=5 the whole cell.
    factory = RecordingFactory([pool_receipt(0.005)])
    calls = []
    result = run(short_checkpoint(tmp_path), tmp_path / "out", budget=3,
                 factory=factory, calls=calls, script=[])

    assert result["status"] == "ok"
    assert result["evaluations"] == 3
    assert factory.created == []  # no session, no pool request during warmup
    assert result["llm_calls"] == 0
    sources = [event["source"] for event in evaluation_events(tmp_path / "out")]
    assert sources == ["hands_quasi"] * 3
    # Official quasi is not a ranker fallback.
    assert result["ranker_fallback_count"] == 0


def test_surrogate_phase_sources_arm_state_and_probe_tagging(tmp_path) -> None:
    # 3 quasi steps, then 3 surrogate steps: pool win / front win / pool win.
    factory = RecordingFactory(
        [pool_receipt(0.005), pool_receipt(0.007), pool_receipt(0.009)]
    )
    calls = []
    result = run(
        short_checkpoint(tmp_path), tmp_path / "out", budget=6,
        factory=factory, calls=calls,
        script=["pool:1", "front", "pool:0"],
    )

    assert result["status"] == "ok"
    events = evaluation_events(tmp_path / "out")
    assert [event["source"] for event in events] == (
        ["hands_quasi"] * 3 + ["hands_pool", "hands_front", "hands_pool"]
    )
    # The session is created exactly once, at the first surrogate step, with
    # the DEFAULT pool protocol — no surrogate/seeds disclosure (DESIGN §4).
    assert len(factory.created) == 1
    first_extras = factory.created[0]["first_extras"]
    assert first_extras["protocol"] == pool.POOL_PROTOCOL
    assert "[hands_quasi]" in first_extras["history"]
    assert "score: 88.0" in first_extras["incumbent"]  # after 3 improving evals
    assert "bout: 3." in first_extras["budget"]  # 6 - 3 consumed
    # The union calls carry the filtered pool and NO evolution seeds.
    receipt0 = pool_receipt(0.005)
    assert calls[3]["pool"] == receipt0["configs"]
    assert calls[3]["initial_suggest_extra"] == []
    assert calls[3]["quasi_index"] == 3  # the quasi prefix was consumed
    assert calls[4]["pool"] == pool_receipt(0.007)["configs"]
    assert calls[5]["pool"] == pool_receipt(0.009)["configs"]

    # Pool win: literal member executed, provenance in arm_state.
    state3 = events[3]["arm_state"]
    assert events[3]["proposal"] == receipt0["configs"][1]
    assert state3["pool_configs"] == receipt0["configs"]
    assert state3["front_size"] == 47
    assert state3["union_front_size"] == 3
    assert state3["pool_survivors"] == 1 and state3["pool_survivor_indices"] == [1]
    assert state3["chosen_from"] == "pool" and state3["chosen_pool_index"] == 1
    assert "nearest_pool_z_dist" not in state3
    # Front win: evolved-side config, geometry diagnostic instead.
    state4 = events[4]["arm_state"]
    assert state4["chosen_from"] == "front"
    assert state4["pool_survivors"] == 0 and state4["pool_survivor_indices"] == []
    assert isinstance(state4["nearest_pool_z_dist"], float)
    assert "chosen_pool_index" not in state4

    # Feedback channel: the pool win's outcome rides UNTAGGED, the front
    # win's outcome rides the [hebo_probe] tag on the next ask.
    session = factory.created[0]["session"]
    assert session.asks[0] is None or llm.OUTCOME_KEY not in session.asks[0]
    assert "[hebo_probe]" not in session.asks[1].get(llm.OUTCOME_KEY, "")
    assert "[hebo_probe]" in session.asks[2][llm.OUTCOME_KEY]
    assert result["llm_calls"] == 3
    assert result["ranker_fallback_count"] == 0


def test_paired_contracts_and_only_structural_difference_vs_hebo_only(tmp_path) -> None:
    # Same (checkpoint, seed): identical per-step seeds / scramble /
    # quasi-index bookkeeping / quasi prefix; on surrogate steps the ONLY
    # difference is that hands hands the seam the filtered pool (union mode)
    # while hebo_only never sends one, and neither arm seeds the evolution.
    seed = 11

    def paired_suggest_fn(calls):
        def suggest(**kwargs):
            calls.append(kwargs)
            n = len(calls)
            params = cfg(
                depth=(n % 8) + 1,
                lr=0.002 + 0.0001 * n,
                dropout=0.3 + 0.01 * n,
                mode="slow",
            )
            if len(kwargs["history"]) < RAND_SAMPLE:
                return {"suggestion": params, "mode": "quasi", "quasi_consumed": 1}
            pool = kwargs.get("pool") or []
            if pool:
                return {
                    "suggestion": pool[0],
                    "mode": "surrogate",
                    "quasi_consumed": 0,
                    "front_size": 47,
                    "chosen_from": "pool",
                    "chosen_pool_index": 0,
                    "union_front_size": 2,
                    "pool_survivor_indices": [0],
                }
            return {
                "suggestion": params,
                "mode": "surrogate",
                "quasi_consumed": 0,
                "front_size": 47,
            }

        return suggest

    hebo_calls, hands_calls = [], []
    runner.run_cell(
        arm=hebo_only.ARM,
        checkpoint_dir=short_checkpoint(tmp_path, "a"),
        out_dir=tmp_path / "out_hebo",
        seed=seed,
        budget=5,
        eval_fn=fake_eval_from([("ok", 90.0 - i) for i in range(5)]),
        preflight_fn=ok_preflight,
        extras={"hebo_suggest_fn": paired_suggest_fn(hebo_calls)},
    )
    run(
        short_checkpoint(tmp_path, "b"), tmp_path / "out_hands", seed=seed,
        budget=5, factory=RecordingFactory([pool_receipt(0.005), pool_receipt(0.007)]),
        calls=hands_calls, script=["pool:0", "pool:0"],
    )

    assert len(hebo_calls) == len(hands_calls) == 5
    for step_index, (hebo_call, hands_call) in enumerate(zip(hebo_calls, hands_calls)):
        assert hebo_call["seed"] == hands_call["seed"] == hebo_common.step_seed(seed, step_index)
        assert hebo_call["scramble_seed"] == hands_call["scramble_seed"]
        assert hebo_call["quasi_index"] == hands_call["quasi_index"]
    hebo_events = evaluation_events(tmp_path / "out_hebo")
    hands_events = evaluation_events(tmp_path / "out_hands")
    assert [e["proposal"] for e in hebo_events[:3]] == [e["proposal"] for e in hands_events[:3]]
    # Surrogate steps: hands carries the pool (union mode), hebo_only never
    # sends one; neither arm seeds the evolutionary population.
    for call in hebo_calls[3:]:
        assert "pool" not in call
        assert call["initial_suggest_extra"] == []
    assert hands_calls[3]["pool"] == pool_receipt(0.005)["configs"]
    assert hands_calls[3]["initial_suggest_extra"] == []
