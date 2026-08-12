from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codec as codec_mod  # noqa: E402
import llm  # noqa: E402
import objective  # noqa: E402
import runner  # noqa: E402
import space as space_mod  # noqa: E402
from arms import active_set  # noqa: E402
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


def fake_session_factory(script, created=None):
    def factory(role_name, first_extras=None):
        if created is not None:
            created.append((role_name, first_extras))
        return FakeSession(script)

    return factory


INT_AND_DEGENERATE_SOURCE = '''
PARAM_SCHEMA = {
    "width": "int",
    "fixed": "float",
}
SEARCH_SPACE = {
    "width": ("int", 1, 8),
    "fixed": ("float", 0.5, 0.5),
}
BASE_PARAMS = {
    "width": 3,
    "fixed": 0.5,
}
def make_model(params):
    return params
'''

TINY_INT_SOURCE = '''
PARAM_SCHEMA = {
    "width": "int",
}
SEARCH_SPACE = {
    "width": ("int", 1, 3),
}
BASE_PARAMS = {
    "width": 2,
}
def make_model(params):
    return params
'''


def test_poll_pairs_share_anchor_and_move_incumbent(tmp_path) -> None:
    # Toy contract anchors: depth z=3/7, lr z=1/3, dropout z=0.2. Poll 1 picks
    # (dropout, 0.2): x+ z 0.4 -> 0.2 (score 80, new incumbent), x- z 0.0 ->
    # 0.0 (score 90). Poll 2 anchors at dropout 0.2 and picks (dropout, 0.1):
    # x+ z 0.5 -> 0.25 (score 70), x- z 0.3 -> 0.15 (score 75).
    ckpt = write_checkpoint(tmp_path)
    out = tmp_path / "out"
    script = [
        {"parameter": "dropout", "step": 0.2, "rationale": "p1"},
        {"parameter": "dropout", "step": 0.1, "rationale": "p2"},
    ]

    result = runner.run_cell(
        arm=active_set.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=5,
        budget=4,
        eval_fn=fake_eval_from(
            [("ok", 80.0), ("ok", 90.0), ("ok", 70.0), ("ok", 75.0)]
        ),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(script)},
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 4
    assert result["llm_calls"] == 2
    assert result["llm_input_tokens"] == 20
    assert result["counts"] == {
        "crashes": 0,
        "invalid": 0,
        "duplicates": 0,
        "task_preflight_rejected": 0,
    }
    events = evaluation_events(out)
    proposals = [event["proposal"] for event in events]
    # The scripted (parameter, step) is executed faithfully: both sides decode
    # from the same anchor and move only dropout.
    assert [p["dropout"] for p in proposals] == pytest.approx([0.2, 0.0, 0.25, 0.15])
    assert [p["lr"] for p in proposals] == pytest.approx([0.001] * 4)
    assert all(p["depth"] == 4 and p["mode"] == "fast" for p in proposals)
    # z-space verification: decoded proposals sit at anchor_z +/- step exactly.
    contract = space_mod.read_contract(ckpt / "candidate" / "train.py")
    codec = codec_mod.Codec(contract)
    assert [codec.encode(p)[0][2] for p in proposals] == pytest.approx(
        [0.4, 0.0, 0.5, 0.3]
    )
    states = [event["arm_state"] for event in events]
    assert [s["side"] for s in states] == ["x+", "x-", "x+", "x-"]
    assert [s["step"] for s in states] == [0.2, 0.2, 0.1, 0.1]
    assert [s["poll_index"] for s in states] == [1, 1, 2, 2]
    assert all(s["parameter"] == "dropout" for s in states)
    assert states[0]["anchor"] == states[1]["anchor"] == contract.params_identity(BASE)
    assert (
        states[2]["anchor"]
        == states[3]["anchor"]
        == contract.params_identity(cfg(dropout=0.2))
    )
    # New incumbent = best of {anchor, x+, x-}: 80 takes poll 1, 70 takes poll 2.
    assert [e["incumbent_after"]["score"] for e in events] == [80.0, 80.0, 70.0, 70.0]
    assert result["best_score"] == 70.0
    assert result["best_config"]["dropout"] == pytest.approx(0.25)
    assert result["beat_initial_incumbent"] is True


def test_session_wiring_feasible_set_text_and_outcome_feedback(tmp_path) -> None:
    session = FakeSession(
        [
            {"parameter": "dropout", "step": 0.2},
            {"parameter": "dropout", "step": 0.1},
        ]
    )
    created = []

    def factory(role_name, first_extras=None):
        created.append((role_name, first_extras))
        return session

    result = runner.run_cell(
        arm=active_set.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=5,
        budget=4,
        eval_fn=fake_eval_from(
            [("ok", 80.0), ("ok", 90.0), ("ok", 70.0), ("ok", 75.0)]
        ),
        preflight_fn=ok_preflight,
        extras={"session_factory": factory},
    )

    assert result["status"] == "ok"
    # One bout-scoped bench-active-set session with the §四 first-message blocks.
    assert len(created) == 1
    role_name, first_extras = created[0]
    assert role_name == "bench-active-set"
    assert {
        "search_space",
        "candidate",
        "incumbent",
        "history",
        "protocol",
        "budget",
    } <= set(first_extras)
    assert len(session.asks) == 2
    # Every ask carries the filtered feasible set (categoricals never listed).
    feasible1 = session.asks[0][llm.FEASIBLE_SET_KEY]
    assert (
        "- depth (int in [1, 8]; anchor value 4): feasible steps 0.1, 0.2, 0.4"
        in feasible1
    )
    assert (
        "- dropout (float in [0.0, 0.5]; anchor value 0.1): feasible steps "
        "0.05, 0.1, 0.2" in feasible1
    )
    assert "mode" not in feasible1
    # Poll 2, anchor dropout 0.2: (dropout, 0.2) is pruned because x- rebuilds
    # the original incumbent (dropout 0.1), (dropout, 0.4) because x- rebuilds
    # poll 1's x- (dropout 0.0) — duplicate sides kill the whole combo.
    feasible2 = session.asks[1][llm.FEASIBLE_SET_KEY]
    dropout_line = [
        line for line in feasible2.splitlines() if line.startswith("- dropout")
    ][0]
    assert dropout_line.split("feasible steps ")[1] == "0.05, 0.1"
    # Poll-1's two outcomes ride on the poll-2 ask, one block per side,
    # only after both sides completed.
    outcome = session.asks[1][llm.OUTCOME_KEY]
    assert "--- x+ ---" in outcome and "--- x- ---" in outcome
    assert "score = 80.0" in outcome and "score = 90.0" in outcome
    assert "incumbent after this poll: score 80.0" in outcome


def test_feasibility_rules_exclude_int_collisions_and_degenerate_dims(tmp_path) -> None:
    # width anchor 3 (z=2/7): step 0.05 rounds back to 3 on both sides (4 +/- no
    # wait — nearest(3 +/- 0.35) == 3), so 0.05 is infeasible; the degenerate
    # float encodes to z=0 so z-s < 0 excludes it with no special-casing.
    ckpt = write_checkpoint(
        tmp_path,
        train_source=INT_AND_DEGENERATE_SOURCE,
        incumbent_params={"width": 3, "fixed": 0.5},
        incumbent_score=100.0,
    )
    session = FakeSession([{"parameter": "width", "step": 0.1}])

    result = runner.run_cell(
        arm=active_set.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=1,
        budget=2,
        eval_fn=fake_eval_from([("ok", 60.0), ("ok", 70.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": lambda role, first_extras=None: session},
    )

    assert result["status"] == "ok"
    feasible_text = session.asks[0][llm.FEASIBLE_SET_KEY]
    width_line = [
        line for line in feasible_text.splitlines() if line.startswith("- width")
    ][0]
    assert width_line.split("feasible steps ")[1] == "0.1, 0.2, 0.4"
    assert "fixed" not in feasible_text
    proposals = [event["proposal"] for event in evaluation_events(tmp_path / "out")]
    # decode(2/7 +/- 0.1) on width = nearest(3 +/- 0.7) = 4 / 2.
    assert [p["width"] for p in proposals] == [4, 2]
    assert all(p["fixed"] == 0.5 for p in proposals)
    assert result["best_score"] == 60.0


def test_out_of_set_receipt_retries_with_correction(tmp_path) -> None:
    # (dropout, 0.4) at anchor z=0.2 has z-s < 0 — not in the feasible set;
    # the arm re-asks with a correction, then executes the valid receipt.
    session = FakeSession(
        [
            {"parameter": "dropout", "step": 0.4},
            {"parameter": "dropout", "step": 0.2},
        ]
    )

    result = runner.run_cell(
        arm=active_set.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        budget=2,
        eval_fn=fake_eval_from([("ok", 80.0), ("ok", 90.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": lambda role, first_extras=None: session},
    )

    assert result["status"] == "ok"
    assert result["llm_calls"] == 2  # the retry burned a second call
    retry = session.asks[1][llm.OUTCOME_KEY]
    assert "correction" in retry
    assert "0.4" in retry
    proposals = [event["proposal"] for event in evaluation_events(tmp_path / "out")]
    assert [p["dropout"] for p in proposals] == pytest.approx([0.2, 0.0])


def test_three_consecutive_failed_attempts_is_arm_error(tmp_path) -> None:
    # Unknown parameter, a failed invocation, then an out-of-set step: three
    # consecutive failed attempts with no successful selection in between.
    session = FakeSession(
        [
            {"parameter": "nope", "step": 0.1},
            RuntimeError("schema retries exhausted"),
            {"parameter": "dropout", "step": 0.4},
        ]
    )

    result = runner.run_cell(
        arm=active_set.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        eval_fn=fake_eval_from([]),  # asserts if the objective is ever started
        preflight_fn=ok_preflight,
        extras={"session_factory": lambda role, first_extras=None: session},
    )

    assert result["status"] == "arm_error"
    assert "3 consecutive" in result["reason"]
    assert result["llm_calls"] == 3
    assert result["evaluations"] == 0


def test_empty_feasible_set_is_unsupported(tmp_path) -> None:
    # width [1,3] anchored at 2 (z=0.5): steps < 0.4 round back to the anchor
    # value 2; step 0.4 reaches 1 and 3, but both are already executed
    # history — every combo dies on a duplicate side, so the set is empty at
    # the first poll start.
    ckpt = write_checkpoint(
        tmp_path,
        train_source=TINY_INT_SOURCE,
        incumbent_params={"width": 2},
        incumbent_score=10.0,
        history=[hrow({"width": 1}, 20.0), hrow({"width": 3}, 30.0)],
    )

    result = runner.run_cell(
        arm=active_set.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=1,
        eval_fn=fake_eval_from([]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "unsupported"
    assert "feasible" in result["reason"]
    assert result["evaluations"] == 0
    assert result["llm_calls"] == 0
    assert result["internal_duplicate_count"] == 1  # the (width, 0.4) combo


def test_task_preflight_rejection_leaves_a_half_pair(tmp_path) -> None:
    # x- (dropout 0.0) is rejected by the TASK preflight: no score, no budget.
    # The poll still completes, the rejection is reported back to the LLM, and
    # the next poll anchors at x+ (the improved incumbent, dropout 0.2).

    def reject_zero_dropout(
        candidate_path, params, *, preflight_fn, per_runtime_limit, python_cmd=None
    ):
        if params["dropout"] == 0.0:
            return objective.PreflightOutcome(
                status="rejected", detail="scripted task rejection"
            )
        return objective.PreflightOutcome(status="ok", detail=None)

    session = FakeSession(
        [
            {"parameter": "dropout", "step": 0.2},
            {"parameter": "dropout", "step": 0.1},
        ]
    )

    result = runner.run_cell(
        arm=active_set.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        budget=2,
        eval_fn=fake_eval_from([("ok", 80.0), ("ok", 70.0)]),
        preflight_fn=reject_zero_dropout,
        extras={"session_factory": lambda role, first_extras=None: session},
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 2  # the rejected side consumed no budget
    assert result["counts"]["task_preflight_rejected"] == 1
    assert "REJECTED by the task preflight" in session.asks[1][llm.OUTCOME_KEY]
    proposals = [event["proposal"] for event in evaluation_events(tmp_path / "out")]
    # Poll 2 anchored at the improved incumbent (dropout 0.2): x+ z 0.5 -> 0.25.
    assert [p["dropout"] for p in proposals] == pytest.approx([0.2, 0.25])
    assert result["best_score"] == 70.0
