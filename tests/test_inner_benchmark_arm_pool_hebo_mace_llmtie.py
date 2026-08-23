from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm  # noqa: E402  (inserts the repo root into sys.path itself)
from driver import roles  # noqa: E402


def test_tiebreak_judge_role_registered() -> None:
    role = llm.BENCH_ROLES["bench-tiebreak-judge"]
    assert role.prompt_file == "bench-tiebreak-judge.md"
    assert role.tools == ()
    assert role.receipt_schema == {"choice": "int", "rationale": "?str"}
    assert (roles.PROMPT_DIR / role.prompt_file).exists()


import runner  # noqa: E402
from arms import pool_hebo_mace_llmtie  # noqa: E402
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


def role_scripted_factory(scripts_by_role):
    """session_factory dispatching per-role scripts; records created sessions."""
    created = []

    def factory(role_name, first_extras=None):
        session = FakeSession(scripts_by_role[role_name])
        session.role_name = role_name
        session.first_extras = first_extras
        created.append(session)
        return session

    factory.created = created
    return factory


def pool_receipt(lr_base, *, depths=(1, 2, 3, 4, 5), order=None):
    return {
        "configs": [
            cfg(depth=depth, lr=lr_base + 0.0001 * index, dropout=0.3, mode="slow")
            for index, depth in enumerate(depths)
        ],
        "order": order if order is not None else [0, 1, 2, 3, 4],
        "rationale": "scripted pool",
    }


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


def scripted_rank_fn(script):
    def rank_fn(*, search_space, history, pool, seed):
        return script.pop(0)

    return rank_fn


def run(ckpt, out, *, seed=1, budget=1, scripts_by_role, values_script, eval_fn=None):
    factory = role_scripted_factory(scripts_by_role)
    result = runner.run_cell(
        arm=pool_hebo_mace_llmtie.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=seed,
        budget=budget,
        eval_fn=eval_fn or fake_eval_from([("ok", 50.0)] * budget),
        preflight_fn=ok_preflight,
        extras={
            "session_factory": factory,
            "hebo_rank_fn": scripted_rank_fn(values_script),
        },
    )
    return result, factory


def judge_session(factory):
    return next(s for s in factory.created if s.role_name == "bench-tiebreak-judge")


# First Pareto front is exactly {0, 1} (same fixture as the baseline arm's
# tied-front test): 2/3/4 are dominated by member 0 or 1; 0 and 1 do not
# dominate each other.
TIE_VALUES = [[0.8, 0.6], [0.6, 0.8], [0.5, 0.5], [0.7, 0.2], [0.2, 0.7]]


def test_tie_uses_judge_choice_with_shuffled_display(tmp_path) -> None:
    displayed_orders = set()
    for seed in range(8):
        ckpt = continuation_checkpoint(tmp_path, name=f"ck{seed}")
        result, factory = run(
            ckpt,
            tmp_path / f"out{seed}",
            seed=seed,
            scripts_by_role={
                "bench-pool-proposer": [pool_receipt(0.005)],
                "bench-tiebreak-judge": [{"choice": 1, "rationale": "scripted"}],
            },
            values_script=[list(TIE_VALUES)],
        )
        assert result["status"] == "ok"
        (event,) = evaluation_events(tmp_path / f"out{seed}")
        arm_state = event["arm_state"]
        assert arm_state["ranker_fallback"] is False
        assert arm_state["pareto_front"] == [0, 1]
        tiebreak = arm_state["tiebreak"]
        assert tiebreak["front_size"] == 2
        assert tiebreak["judge_choice"] == 1
        displayed = tiebreak["displayed_front_pool_indices"]
        assert sorted(displayed) == [0, 1]
        # choice is a DISPLAY index: chosen pool member = displayed[choice].
        assert arm_state["chosen_index"] == displayed[1]
        assert event["proposal"] == pool_receipt(0.005)["configs"][displayed[1]]
        displayed_orders.add(tuple(displayed))
        assert result["llm_calls"] == 2  # 1 proposer + 1 judge
    # Display order is actually shuffled (the identity = proposer rank order
    # must not be the only possibility): both orders occur over seeds 0..7.
    assert displayed_orders == {(0, 1), (1, 0)}


def test_judge_sees_production_cast_params(tmp_path) -> None:
    ckpt = continuation_checkpoint(tmp_path)
    receipt = {
        "configs": [
            cfg(depth=1.2, lr=0.0051, dropout=0.3, mode="slow"),
            cfg(depth=2.7, lr=0.0052, dropout=0.3, mode="slow"),
            cfg(depth=3.7, lr=0.0053, dropout=0.3, mode="slow"),
            cfg(depth=4.4, lr=0.0054, dropout=0.3, mode="slow"),
            cfg(depth=5.9, lr=0.0055, dropout=0.3, mode="slow"),
        ],
        "order": [0, 1, 2, 3, 4],
        "rationale": "fractional depths",
    }
    # Front is exactly {1, 2} (the 2.7 / 3.7 members).
    values = [[0.4, 0.4], [0.9, 0.5], [0.5, 0.9], [0.8, 0.3], [0.3, 0.8]]
    eval_fn = fake_eval_from([("ok", 50.0)])
    result, factory = run(
        ckpt,
        tmp_path / "out",
        scripts_by_role={
            "bench-pool-proposer": [receipt],
            "bench-tiebreak-judge": [{"choice": 0}],
        },
        values_script=[values],
        eval_fn=eval_fn,
    )
    assert result["status"] == "ok"
    front_text = judge_session(factory).asks[0]["front"]
    # The judge sees production-cast values (int cast truncates toward
    # zero) — never the raw fractional receipt values.
    assert '"depth":2' in front_text
    assert '"depth":3' in front_text
    assert "2.7" not in front_text
    assert "3.7" not in front_text
    # ...matching what the objective actually ran (the runner casts too).
    executed = eval_fn.calls[0]
    assert executed["depth"] in (2, 3)
    assert isinstance(executed["depth"], int)


def test_out_of_range_choice_is_corrected_then_accepted(tmp_path) -> None:
    result, factory = run(
        continuation_checkpoint(tmp_path),
        tmp_path / "out",
        scripts_by_role={
            "bench-pool-proposer": [pool_receipt(0.005)],
            "bench-tiebreak-judge": [{"choice": 7}, {"choice": 1, "rationale": "fixed"}],
        },
        values_script=[list(TIE_VALUES)],
    )
    assert result["status"] == "ok"
    judge = judge_session(factory)
    assert len(judge.asks) == 2
    assert "correction" in judge.asks[1]
    (event,) = evaluation_events(tmp_path / "out")
    assert event["arm_state"]["tiebreak"]["judge_choice"] == 1
    assert result["llm_calls"] == 3  # 1 proposer + 2 judge attempts


def test_judge_failure_is_arm_error(tmp_path) -> None:
    result, _ = run(
        continuation_checkpoint(tmp_path),
        tmp_path / "out",
        scripts_by_role={
            "bench-pool-proposer": [pool_receipt(0.005)],
            "bench-tiebreak-judge": [RuntimeError("judge down")] * 3,
        },
        values_script=[list(TIE_VALUES)],
    )
    assert result["status"] == "arm_error"
    assert "judge down" in result["reason"]
    assert result["evaluations"] == 0


def test_unique_front_member_skips_judge(tmp_path) -> None:
    # Member 2 dominates every other member on all objectives (baseline
    # fixture): no tie, so no judge session may be created at all — the
    # missing "bench-tiebreak-judge" script key would KeyError otherwise.
    values = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],
        [2.0, 2.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
    result, factory = run(
        continuation_checkpoint(tmp_path),
        tmp_path / "out",
        scripts_by_role={"bench-pool-proposer": [pool_receipt(0.005)]},
        values_script=[values],
    )
    assert result["status"] == "ok"
    assert [s.role_name for s in factory.created] == ["bench-pool-proposer"]
    (event,) = evaluation_events(tmp_path / "out")
    assert event["arm_state"]["chosen_index"] == 2
    assert "tiebreak" not in event["arm_state"]
    assert result["llm_calls"] == 1
