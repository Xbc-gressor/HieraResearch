from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import objective  # noqa: E402
import runner  # noqa: E402
from arms import current  # noqa: E402
from ib_support import (  # noqa: E402
    BASE,
    cfg,
    evaluation_events,
    fake_eval_by_params,
    fake_eval_from,
    hrow,
    ok_preflight,
    read_events,
    write_checkpoint,
)

GRID_2D_SOURCE = '''
PARAM_SCHEMA = {
    "width": "int",
    "mode": ("categorical", ["a", "b"]),
}
SEARCH_SPACE = {
    "width": ("int", 1, 3),
    "mode": ("categorical", ["a", "b"]),
}
BASE_PARAMS = {
    "width": 2,
    "mode": "a",
}
def make_model(params):
    return params
'''

INCUMBENT = cfg(depth=8, lr=0.05, dropout=0.4, mode="slow")


def continuation_history():
    """8 finite unique executed rows, none equal to the incumbent."""
    return [
        hrow(
            cfg(
                depth=index,
                lr=0.001 * index,
                dropout=0.05 * index,
                mode="fast" if index % 2 else "slow",
            ),
            60.0 + index,
            origin="bout_0",
        )
        for index in range(1, 9)
    ]


def write_continuation(base_dir, **kwargs):
    return write_checkpoint(
        base_dir,
        regime="continuation",
        stratum="cont_improved",
        incumbent_params=INCUMBENT,
        incumbent_score=50.0,
        history=continuation_history(),
        **kwargs,
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


def reject_preflight(candidate_path, params, *, preflight_fn, per_runtime_limit, python_cmd=None):
    return objective.PreflightOutcome(status="rejected", detail="always infeasible")


def test_first_bout_evaluates_deferred_without_llm(tmp_path) -> None:
    deferred = [
        {"params": cfg(depth=2, lr=0.002, dropout=0.2, mode="slow")},
        {"params": cfg(depth=7, lr=0.007, dropout=0.05, mode="fast")},
    ]
    ckpt = write_checkpoint(tmp_path, deferred=deferred)
    out = tmp_path / "out"
    created = []

    result = runner.run_cell(
        arm=current.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=5,
        budget=2,
        eval_fn=fake_eval_from([("ok", 90.0), ("ok", 80.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([], created)},
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 2
    events = evaluation_events(out)
    # Deferred configs occupy the bout's first trial slots and count toward B.
    assert [event["proposal"] for event in events] == [d["params"] for d in deferred]
    assert [event["source"] for event in events] == ["deferred", "deferred"]
    assert events[0]["arm_state"]["current_method"] == "bo"  # 4-dim: select_method
    # First bouts never rewarm: no session, zero LLM usage.
    assert created == []
    assert result["llm_calls"] == 0
    assert result["llm_input_tokens"] == 0
    assert not [event for event in read_events(out) if event["kind"] == "preflight_rejected"]


def test_continuation_rewarm_proposals_lead_and_count(tmp_path) -> None:
    ckpt = write_continuation(tmp_path)
    out = tmp_path / "out"
    created = []
    c1 = cfg(depth=3, lr=0.003, dropout=0.15, mode="slow")
    c2 = cfg(depth=5, lr=0.004, dropout=0.25, mode="fast")
    receipt = {"configs": [c1, c2], "rationale": "near the incumbent region"}

    result = runner.run_cell(
        arm=current.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=7,
        budget=3,
        eval_fn=fake_eval_from([("ok", 90.0), ("ok", 85.0), ("ok", 95.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([receipt], created)},
    )

    assert result["status"] == "ok"
    events = evaluation_events(out)
    # Accepted rewarm proposals occupy the bout's first slots (production
    # enqueue order) and count toward B; the third slot is a TPE draw.
    assert [event["source"] for event in events] == ["rewarm", "rewarm", "tpe"]
    assert [events[0]["proposal"], events[1]["proposal"]] == [c1, c2]
    assert events[0]["rationale"] == "near the incumbent region"
    assert result["evaluations"] == 3
    # One rewarm invocation, accounted via the aggregate keys.
    assert result["llm_calls"] == 1
    assert result["llm_input_tokens"] == 10
    role_name, first_extras = created[0]
    assert role_name == "bench-rewarm-proposer"
    assert {
        "search_space",
        "candidate",
        "incumbent",
        "history",
        "protocol",
        "budget",
    } <= set(first_extras)


def test_invalid_rewarm_proposals_dropped_without_budget(tmp_path) -> None:
    ckpt = write_continuation(tmp_path)
    out = tmp_path / "out"
    valid = cfg(depth=6, lr=0.006, dropout=0.3, mode="fast")
    receipt = {
        "configs": [
            cfg(depth=99),  # out of space
            dict(INCUMBENT),  # duplicate of the incumbent
            valid,
        ],
        "rationale": "mixed quality",
    }

    result = runner.run_cell(
        arm=current.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=7,
        budget=2,
        eval_fn=fake_eval_from([("ok", 90.0), ("ok", 88.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([receipt])},
    )

    assert result["status"] == "ok"
    events = evaluation_events(out)
    # Only the valid proposal leads; invalid ones never reached preflight.
    assert events[0]["proposal"] == valid
    assert [event["source"] for event in events] == ["rewarm", "tpe"]
    assert not [event for event in read_events(out) if event["kind"] == "preflight_rejected"]
    assert result["evaluations"] == 2
    assert result["llm_calls"] == 1
    # The in-arm dropped duplicate is counted under the aggregate key.
    assert result["internal_duplicate_count"] == 1


def test_same_seed_replays_tpe_sequence_with_warm_history(tmp_path) -> None:
    # Continuation history (>= WARMUP finite unique) is injected as priors, so
    # TPE drives every draw; an empty rewarm receipt keeps the bout pure TPE.
    ckpt = write_continuation(tmp_path)

    def run(out_dir, seed):
        return runner.run_cell(
            arm=current.ARM,
            checkpoint_dir=ckpt,
            out_dir=out_dir,
            seed=seed,
            budget=5,
            eval_fn=fake_eval_by_params(
                lambda params: params["depth"] * 10.0 + params["dropout"],
                crash_if=lambda params: params["depth"] == 8,
            ),
            preflight_fn=ok_preflight,
            extras={
                "session_factory": fake_session_factory(
                    [{"configs": [], "rationale": "nothing to add"}]
                )
            },
        )

    out_a, out_b, out_c = (tmp_path / name for name in ("a", "b", "c"))
    result_a = run(out_a, seed=7)
    result_b = run(out_b, seed=7)
    result_c = run(out_c, seed=8)

    assert result_a["status"] == result_b["status"] == "ok"
    proposals_a = [event["proposal"] for event in evaluation_events(out_a)]
    proposals_b = [event["proposal"] for event in evaluation_events(out_b)]
    proposals_c = [event["proposal"] for event in evaluation_events(out_c)]
    encode = lambda rows: json.dumps(rows, sort_keys=True, default=str)
    assert encode(proposals_a) == encode(proposals_b)  # reproducible per seed
    assert encode(proposals_a) != encode(proposals_c)  # seed actually matters
    assert result_a["llm_calls"] == 1  # the (empty) rewarm invocation
    assert not [event for event in read_events(out_a) if event["kind"] == "preflight_rejected"]


def test_task_preflight_rejections_feed_back_until_tripwire(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)

    result = runner.run_cell(
        arm=current.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=1,
        budget=4,
        eval_fn=fake_eval_from([]),  # asserts if the objective ever starts
        preflight_fn=reject_preflight,
    )

    # Every rejection is fed back as an infeasible constraint and the arm
    # keeps proposing fresh configs until the RUNNER's tripwire ends the cell.
    assert result["status"] == "arm_error"
    assert "5 consecutive preflight rejections" in result["reason"]
    assert result["evaluations"] == 0
    assert result["counts"]["task_preflight_rejected"] == 5


def test_two_dimensional_space_uses_production_grid(tmp_path) -> None:
    ckpt = write_checkpoint(
        tmp_path,
        train_source=GRID_2D_SOURCE,
        incumbent_params={"width": 2, "mode": "a"},
        incumbent_score=10.0,
    )
    out = tmp_path / "out"

    result = runner.run_cell(
        arm=current.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=3,
        budget=3,
        eval_fn=fake_eval_from([("ok", 90.0), ("ok", 80.0), ("ok", 70.0)]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "ok"
    events = evaluation_events(out)
    # Production grid: 6 combos shuffled by random.Random(3) =
    # [(1,a),(2,a),(2,b),(3,b),(3,a),(1,b)]; (2,a) is the incumbent and is
    # deduped against attempted history before the sweep.
    assert [event["proposal"] for event in events] == [
        {"width": 1, "mode": "a"},
        {"width": 2, "mode": "b"},
        {"width": 3, "mode": "b"},
    ]
    assert [event["source"] for event in events] == ["grid"] * 3
    assert events[0]["arm_state"]["current_method"] == "grid"
    assert result["llm_calls"] == 0


GRID_CONT_SOURCE = '''
PARAM_SCHEMA = {
    "width": "int",
    "mode": ("categorical", ["a", "b"]),
}
SEARCH_SPACE = {
    "width": ("int", 1, 10),
    "mode": ("categorical", ["a", "b"]),
}
BASE_PARAMS = {
    "width": 5,
    "mode": "a",
}
def make_model(params):
    return params
'''


def test_grid_continuation_evaluates_rewarm_proposals(tmp_path) -> None:
    """Regression: rewarm validation must not pollute the grid consumer's
    ``attempted`` set — accepted rewarm proposals lead the sweep (production
    two-phase semantics: validation-phase `seen` is a per-call local), not
    get silently dropped by consumer-side dedupe."""
    history = [
        hrow({"width": index, "mode": "a"}, 60.0 + index, origin="bout_0")
        for index in range(1, 9)
    ]
    ckpt = write_checkpoint(
        tmp_path,
        train_source=GRID_CONT_SOURCE,
        regime="continuation",
        stratum="cont_improved",
        incumbent_params={"width": 9, "mode": "b"},
        incumbent_score=10.0,
        history=history,
    )
    c1 = {"width": 10, "mode": "b"}
    c2 = {"width": 9, "mode": "a"}
    receipt = {"configs": [c1, c2], "rationale": "near the incumbent region"}
    out = tmp_path / "out"

    result = runner.run_cell(
        arm=current.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=3,
        budget=3,
        eval_fn=fake_eval_from([("ok", 9.0), ("ok", 8.0), ("ok", 7.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([receipt])},
    )

    assert result["status"] == "ok"
    events = evaluation_events(out)
    # Both accepted rewarm proposals are evaluated FIRST and count toward B;
    # the third slot comes from the shuffled grid sweep.
    assert [event["source"] for event in events] == ["rewarm", "rewarm", "grid"]
    assert [events[0]["proposal"], events[1]["proposal"]] == [c1, c2]
    assert events[0]["arm_state"]["current_method"] == "grid"
    assert result["llm_calls"] == 1


def test_outside_space_deferred_skipped_without_budget(tmp_path) -> None:
    # Production order: split_configs_by_space first, so a deferred config
    # outside the (possibly clamped) box is skipped silently — never
    # attempted, no budget — and accounted via deferred_skipped_outside_space.
    valid = cfg(depth=2, lr=0.002, dropout=0.2, mode="slow")
    ckpt = write_checkpoint(
        tmp_path,
        deferred=[
            {"params": {"depth": 2}},  # missing keys -> outside the box
            {"params": valid},
        ],
    )
    out = tmp_path / "out"

    result = runner.run_cell(
        arm=current.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=5,
        budget=2,
        eval_fn=fake_eval_from([("ok", 90.0), ("ok", 80.0)]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "ok"
    events = evaluation_events(out)
    assert events[0]["proposal"] == valid
    assert [event["source"] for event in events] == ["deferred", "tpe"]
    assert events[0]["arm_state"]["deferred_skipped_outside_space"] == 1
    assert result["evaluations"] == 2
    assert not [event for event in read_events(out) if event["kind"] == "preflight_rejected"]
