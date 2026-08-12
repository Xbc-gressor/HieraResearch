from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import apply_base_params  # noqa: E402
import runner  # noqa: E402
import tune_tools  # noqa: E402
from arms import llm_hillclimb  # noqa: E402
from ib_support import (  # noqa: E402
    cfg,
    evaluation_events,
    fake_eval_from,
    ok_preflight,
    write_checkpoint,
)

DEGENERATE_ONLY_SOURCE = '''
PARAM_SCHEMA = {
    "fixed": "float",
}
SEARCH_SPACE = {
    "fixed": ("float", 0.5, 0.5),
}
BASE_PARAMS = {
    "fixed": 0.5,
}
def make_model(params):
    return params
'''

REFUSAL = {"edited": False, "summary": "nothing worth changing"}


class FakeEditorSession:
    """Scripted BoutSession stand-in. Each ask pops the next scripted action:
    a receipt dict, an exception (raised), or a callable(working_copy) ->
    receipt performing the editor's file edit as a side effect."""

    def __init__(self, actions):
        self._actions = list(actions)
        self.asks = []

    def ask(self, extra=None):
        self.asks.append(dict(extra or {}))
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        if callable(action):
            return action(Path(self.asks[-1]["working_copy"]))
        return action

    def totals(self):
        return {
            "llm_calls": len(self.asks),
            "llm_input_tokens": 10 * len(self.asks),
            "llm_output_tokens": 0,
        }


def edit_to(params, seen=None):
    """Fake editor turn: record the synced BASE_PARAMS it was handed, then
    write the new config with the production apply tool (its Edit's effect)."""

    def action(working_copy):
        if seen is not None:
            seen.append(
                tune_tools._read_literal_mapping(working_copy, "BASE_PARAMS")
            )
        apply_base_params.apply(working_copy, params)
        return {"edited": True, "summary": f"one change towards {params}"}

    return action


def break_file(working_copy):
    working_copy.write_text("BASE_PARAMS = {\n")  # syntactically broken
    return {"edited": True, "summary": "broke the file"}


def fake_session_factory(session, created=None):
    def factory(role_name, first_extras=None):
        if created is not None:
            created.append((role_name, first_extras))
        return session

    return factory


def test_happy_path_syncs_working_copy_to_incumbent(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)
    out = tmp_path / "out"
    seen = []
    created = []
    # Every round improves, so the incumbent advances to the just-written
    # config and the NEXT round's working copy must be synced to it.
    writes = [cfg(lr=value) for value in (0.002, 0.003, 0.004, 0.005)]
    session = FakeEditorSession([edit_to(params, seen) for params in writes])

    result = runner.run_cell(
        arm=llm_hillclimb.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=5,
        budget=4,
        eval_fn=fake_eval_from([("ok", score) for score in (80.0, 70.0, 60.0, 50.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(session, created)},
    )

    assert result["status"] == "ok"
    assert result["beat_initial_incumbent"] is True
    assert result["best_score"] == 50.0
    # The editor saw the working copy synced to the then-current incumbent:
    # the checkpoint incumbent first, then each kept config in turn.
    assert seen == [cfg(), *writes[:-1]]
    proposals = [event["proposal"] for event in evaluation_events(out)]
    assert [p["lr"] for p in proposals] == pytest.approx([0.002, 0.003, 0.004, 0.005])
    assert all(event["source"] == "hillclimb_edit" for event in evaluation_events(out))
    # Outcome回传: no outcome on the first ask; later asks carry KEEP + score.
    assert "outcome" not in session.asks[0]
    assert "(100.0 -> 80.0)" in session.asks[1]["outcome"]
    assert "KEEP" in session.asks[1]["outcome"]
    assert "70.0" in session.asks[2]["outcome"]
    # Structured bout history rides from the second ask on (the first
    # message's history block — the checkpoint history — is not overridden),
    # one line per executed bout trial.
    assert "history" not in session.asks[0]
    assert "#1" in session.asks[1]["history"]
    assert "#2" in session.asks[2]["history"]
    assert "hillclimb_edit" in session.asks[1]["history"]
    # One working copy path for the whole bout, removed at cell end.
    assert len({ask["working_copy"] for ask in session.asks}) == 1
    assert not Path(session.asks[0]["working_copy"]).exists()
    # LLM accounting: one bout session, four calls, aggregated into result.
    assert result["llm_calls"] == 4
    assert result["llm_input_tokens"] == 40
    assert len(created) == 1
    role_name, first_extras = created[0]
    assert role_name == "bench-hillclimb-editor"
    assert {
        "search_space",
        "candidate",
        "incumbent",
        "history",
        "protocol",
        "budget",
    } <= set(first_extras)


def test_discard_outcome_resyncs_to_incumbent(tmp_path) -> None:
    seen = []
    session = FakeEditorSession(
        [
            edit_to(cfg(lr=0.002), seen),  # 150.0 > incumbent 100.0 -> discard
            edit_to(cfg(lr=0.003), seen),  # 90.0 -> keep
        ]
    )

    result = runner.run_cell(
        arm=llm_hillclimb.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=2,
        budget=2,
        eval_fn=fake_eval_from([("ok", 150.0), ("ok", 90.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(session)},
    )

    assert result["status"] == "ok"
    assert result["best_score"] == 90.0
    # After the discard the working copy was re-synced to the UNCHANGED
    # incumbent, not left at the editor's rejected edit.
    assert seen == [cfg(), cfg()]
    assert "DISCARD" in session.asks[1]["outcome"]
    assert "150.0" in session.asks[1]["outcome"]


def test_crash_discards_and_reports_failure(tmp_path) -> None:
    seen = []
    session = FakeEditorSession(
        [
            edit_to(cfg(lr=0.002), seen),  # crashes: +inf, dropped
            edit_to(cfg(lr=0.003), seen),  # 90.0 -> keep
        ]
    )

    result = runner.run_cell(
        arm=llm_hillclimb.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=2,
        budget=2,
        eval_fn=fake_eval_from([("crash", None), ("ok", 90.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(session)},
    )

    assert result["status"] == "ok"
    assert result["counts"]["crashes"] == 1
    assert result["best_score"] == 90.0
    # The crash never touched the incumbent, so the re-sync restores the
    # checkpoint incumbent config, and the editor is told the failure.
    assert seen == [cfg(), cfg()]
    assert "CRASH" in session.asks[1]["outcome"]
    assert "+inf" in session.asks[1]["outcome"]


def test_consecutive_invalid_edits_is_arm_error(tmp_path) -> None:
    session = FakeEditorSession([REFUSAL, break_file, dict(REFUSAL)])

    result = runner.run_cell(
        arm=llm_hillclimb.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        eval_fn=fake_eval_from([]),  # asserts if the objective is ever started
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(session)},
    )

    assert result["status"] == "arm_error"
    assert "3 consecutive" in result["reason"]
    assert result["evaluations"] == 0
    assert result["llm_calls"] == 3
    # Each invalid edit was reported back as the next ask's outcome.
    assert "edited=false" in session.asks[1]["outcome"]
    assert "unreadable" in session.asks[2]["outcome"]
    # Cleanup is total even on the arm_error path.
    assert not Path(session.asks[0]["working_copy"]).exists()


def test_edit_identical_to_incumbent_counts_internal_duplicate(tmp_path) -> None:
    session = FakeEditorSession(
        [
            edit_to(cfg()),  # exactly the incumbent: invalid edit, dropped in-arm
            edit_to(cfg(lr=0.002)),  # a real change -> evaluated
        ]
    )

    result = runner.run_cell(
        arm=llm_hillclimb.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        budget=1,
        eval_fn=fake_eval_from([("ok", 90.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(session)},
    )

    assert result["status"] == "ok"
    assert result["best_score"] == 90.0
    assert result["internal_duplicate_count"] == 1
    assert result["llm_calls"] == 2
    assert "identical" in session.asks[1]["outcome"]
    # The duplicate never reached the runner's preflight.
    assert result["counts"] == {
        "crashes": 0,
        "invalid": 0,
        "duplicates": 0,
        "task_preflight_rejected": 0,
    }


def test_unsupported_when_nothing_varies(tmp_path) -> None:
    result = runner.run_cell(
        arm=llm_hillclimb.ARM,
        checkpoint_dir=write_checkpoint(tmp_path, train_source=DEGENERATE_ONLY_SOURCE),
        out_dir=tmp_path / "out",
        seed=1,
        eval_fn=fake_eval_from([]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(FakeEditorSession([]))},
    )

    assert result["status"] == "unsupported"
    assert result["evaluations"] == 0


def test_first_ask_failure_keeps_checkpoint_history_channel(tmp_path) -> None:
    """Regression: a failed FIRST invocation must not consume the first-ask
    marker — otherwise the retry shadows the first_extras checkpoint history
    with an empty bout history, for the whole bout."""
    session = FakeEditorSession(
        [RuntimeError("transient"), edit_to(cfg(lr=0.002))]
    )

    result = runner.run_cell(
        arm=llm_hillclimb.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        budget=1,
        eval_fn=fake_eval_from([("ok", 90.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(session)},
    )

    assert result["status"] == "ok"
    assert result["llm_calls"] == 2
    # The retry is still the first SUCCESSFUL ask: no bout "history" extra
    # shadows the first_extras checkpoint history (BoutSession re-merges
    # first_extras after the failed invocation).
    assert "history" not in session.asks[1]
    # The invocation failure rode as a correction.
    assert "correction" in session.asks[1]["outcome"]


def test_multi_parameter_edit_is_an_invalid_edit(tmp_path) -> None:
    """§6.0's only narrowing is enforced: an edit changing TWO parameter
    values never reaches the objective — it rides back as a correction."""
    session = FakeEditorSession(
        [
            edit_to(cfg(depth=7, lr=0.009)),  # two changes: invalid edit
            edit_to(cfg(lr=0.002)),  # one change: evaluated
        ]
    )

    result = runner.run_cell(
        arm=llm_hillclimb.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        budget=1,
        eval_fn=fake_eval_from([("ok", 90.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(session)},
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 1
    assert result["llm_calls"] == 2
    # The correction names the changed parameters and the one-change rule.
    outcome = session.asks[1]["outcome"]
    assert "2 parameters" in outcome and "depth" in outcome and "lr" in outcome
    # Only the single-parameter edit was evaluated.
    proposals = [event["proposal"] for event in evaluation_events(tmp_path / "out")]
    assert proposals == [cfg(lr=0.002)]
