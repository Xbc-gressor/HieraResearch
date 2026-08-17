from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))

import checkpoint as checkpoint_mod  # noqa: E402
import objective  # noqa: E402
import runner  # noqa: E402
from arm_api import ArmError, Proposal, Unsupported  # noqa: E402

# Same production-literal toy contract as tests/test_inner_benchmark_objective.py.
TRAIN_SOURCE = '''
PARAM_SCHEMA = {
    "depth": "int",
    "lr": ("float", "log"),
    "dropout": "float",
    "mode": ("categorical", ["fast", "slow"]),
}
SEARCH_SPACE = {
    "depth": ("int", 1, 8),
    "lr": ("float", 0.0001, 0.1, "log"),
    "dropout": ("float", 0.0, 0.5),
    "mode": ("categorical", ["fast", "slow"]),
}
BASE_PARAMS = {
    "depth": 4,
    "lr": 0.001,
    "dropout": 0.1,
    "mode": "fast",
}
def make_model(params):
    return dict(params)
'''

# The objective is faked in these tests; prepare.py only has to exist and parse
# so the manifest's candidate_execution_revision can hash the evaluation surface.
PREPARE_SOURCE = '''
def evaluate_config(make_model, params):
    return float(make_model(params)["depth"])
'''


def cfg(depth=4, lr=0.001, dropout=0.1, mode="fast"):
    return {"depth": depth, "lr": lr, "dropout": dropout, "mode": mode}


BASE = cfg()


def hrow(params, score, status="ok", origin="phase_a", role=None):
    return {
        "params": params,
        "score": score,
        "status": status,
        "origin": origin,
        "role": role,
    }


@pytest.fixture()
def make_checkpoint(tmp_path):
    """Write a toy checkpoint dir (checkpoint.json + candidate/train.py)."""
    counter = itertools.count()

    def _make(
        *,
        incumbent_params=None,
        incumbent_score=100.0,
        history=(),
        regime="first",
        stratum="first",
        task=None,
        deferred=(),
        inherited=False,
        extra=None,
        payload_override=None,
    ) -> Path:
        index = next(counter)
        directory = tmp_path / f"ckpt{index}"
        candidate_dir = directory / "candidate"
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "train.py").write_text(TRAIN_SOURCE)
        # The evaluation surface: _candidate_execution_revision (via the
        # manifest) resolves it next to train.py.
        (candidate_dir / "prepare.py").write_text(PREPARE_SOURCE)
        payload = {
            "schema_version": 2,
            "checkpoint_id": f"ck-{index}",
            "regime": regime,
            "stratum": stratum,
            "source": {"run": "toy"},
            "candidate_relpath": "candidate",
            "task": task
            or {
                "score_fn": "evaluate_config",
                "preflight_fn": "preflight_config",
                "per_runtime_limit": 900,
            },
            "incumbent": {
                "params": incumbent_params if incumbent_params is not None else BASE,
                "score": incumbent_score,
            },
            "incumbent_is_inherited_control": inherited,
            "history": list(history),
            "deferred_configs": list(deferred),
            # run_cell refuses a checkpoint without a valid completed
            # re-measurement; runner tests are about the protocol, so the
            # fixture is remeasured by default.
            "extra": {"remeasure": {"valid": True, "complete": True}}
            if extra is None
            else extra,
        }
        if payload_override is not None:
            payload_override(payload)
        (directory / "checkpoint.json").write_text(json.dumps(payload))
        return directory

    return _make


class ToyArm:
    """name + run(ctx) generator protocol; optional duck-typed hooks."""

    def __init__(self, body, *, name="toy", active_dimensions=None, calibration_report=None):
        self.name = name
        self._body = body
        if active_dimensions is not None:
            self.active_dimensions = active_dimensions
        if calibration_report is not None:
            self.calibration_report = calibration_report

    def run(self, ctx):
        return self._body(ctx)


def scripted_arm(params_list, *, name="scripted", feedbacks=None, on_ctx=None, **hooks):
    """Yields one Proposal per params entry; optionally records feedbacks."""

    def body(ctx):
        if on_ctx is not None:
            on_ctx(ctx)
        for params in params_list:
            feedback = yield Proposal(params=params, source="scripted")
            if feedbacks is not None:
                feedbacks.append(feedback)

    return ToyArm(body, name=name, **hooks)


def fake_eval_from(outcomes):
    """Scripted outcomes ((status, score) tuples), consumed in call order."""
    remaining = list(outcomes)
    calls = []

    def fake(candidate_path, params, *, score_fn, per_runtime_limit, python_cmd=None):
        assert remaining, "eval_fn called more times than scripted"
        calls.append(dict(params))
        status, score = remaining.pop(0)
        return objective.EvalOutcome(
            status=status,
            score=score,
            detail=None if status == "ok" else "scripted crash",
            elapsed_seconds=0.0,
        )

    fake.calls = calls
    return fake


def fake_eval_by_params(score_of, crash_if=None):
    calls = []

    def fake(candidate_path, params, *, score_fn, per_runtime_limit, python_cmd=None):
        calls.append(dict(params))
        if crash_if is not None and crash_if(params):
            return objective.EvalOutcome(
                status="crash", score=None, detail="scripted crash", elapsed_seconds=0.0
            )
        return objective.EvalOutcome(
            status="ok", score=float(score_of(params)), detail=None, elapsed_seconds=0.0
        )

    fake.calls = calls
    return fake


def ok_preflight(candidate_path, params, *, preflight_fn, per_runtime_limit, python_cmd=None):
    return objective.PreflightOutcome(status="ok", detail=None)


def task_reject_preflight(pred, detail="task preflight rejected"):
    def fake(candidate_path, params, *, preflight_fn, per_runtime_limit, python_cmd=None):
        if pred(params):
            return objective.PreflightOutcome(status="rejected", detail=detail)
        return objective.PreflightOutcome(status="ok", detail=None)

    return fake


def read_events(out_dir):
    return [
        json.loads(line)
        for line in (Path(out_dir) / "events.jsonl").read_text().splitlines()
    ]


def evaluation_events(out_dir):
    return [event for event in read_events(out_dir) if event["kind"] == "evaluation"]


# --- 1. happy path ---------------------------------------------------------


def test_happy_path_ten_evaluations(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()
    scores = [90.0, 95.0, 80.0, 85.0, 70.0, 75.0, 60.0, 65.0, 50.0, 55.0]
    proposals = [cfg(dropout=0.05 * i + 0.01) for i in range(10)]

    def body(ctx):
        for i in range(10):
            ctx.emit({"emitted_step": i, "llm_input_tokens": 5})
            yield Proposal(
                params=proposals[i],
                source="toy",
                rationale=f"step {i}",
                arm_state={"llm_calls": 1, "probe": i},
            )

    arm = ToyArm(
        body,
        name="toy-arm",
        active_dimensions=lambda contract: 3,
        calibration_report=lambda ctx: {"s": 1.5, "a_0": 0.01},
    )
    out_dir = tmp_path / "cell"
    result = runner.run_cell(
        arm=arm,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        seed=123,  # default budget B=10
        eval_fn=fake_eval_from([("ok", score) for score in scores]),
        preflight_fn=ok_preflight,
        manifest_extra={"note": "happy"},
    )

    assert result["status"] == "ok"
    assert result["reason"] is None
    assert result["traceback"] is None
    assert result["budget"] == 10
    assert result["evaluations"] == 10
    # incumbent: strict improvements at evals 1,3,5,7,9 (90/80/70/60/50).
    assert result["best_score"] == pytest.approx(50.0)
    assert result["best_config"] == proposals[8]
    # hand-computed: best-so-far 90,90,80,80,70,70,60,60,50,50 -> rel 10..50.
    assert result["relative_improvement_at"] == {
        2: pytest.approx(10.0),
        4: pytest.approx(20.0),
        6: pytest.approx(30.0),
        8: pytest.approx(40.0),
        10: pytest.approx(50.0),
        24: None,  # beyond this cell's budget horizon
    }
    assert result["auc"] == pytest.approx(30.0)  # mean of 10,10,...,50,50
    assert result["final_relative_improvement"] == pytest.approx(50.0)
    assert result["beat_initial_incumbent"] is True
    assert result["first_improvement_eval"] == 1
    assert result["counts"] == {
        "crashes": 0,
        "invalid": 0,
        "duplicates": 0,
        "task_preflight_rejected": 0,
    }
    assert result["llm_calls"] == 10
    assert result["llm_input_tokens"] == 50
    assert result["llm_output_tokens"] == 0
    assert result["ranker_fallback_count"] == 0

    events = read_events(out_dir)
    assert len(events) == 12  # cell_start + 10 evaluations + cell_end
    assert [event["transaction_id"] for event in events] == list(range(12))
    start, end = events[0], events[-1]
    assert start["kind"] == "cell_start"
    assert start["eval_index"] is None
    assert start["arm"] == "toy-arm"
    assert start["seed"] == 123
    assert start["budget"] == 10
    assert end["kind"] == "cell_end"
    assert end["status"] == "ok"
    assert end["evaluations"] == 10

    expected_best = [90.0, 90.0, 80.0, 80.0, 70.0, 70.0, 60.0, 60.0, 50.0, 50.0]
    for i, event in enumerate(events[1:-1], start=1):
        assert event["kind"] == "evaluation"
        assert event["eval_index"] == i
        assert event["proposal"] == proposals[i - 1]
        assert event["source"] == "toy"
        assert event["rationale"] == f"step {i - 1}"
        assert event["preflight_status"] == "ok"
        assert event["status"] == "ok"
        assert event["score"] == pytest.approx(scores[i - 1])
        assert event["arm_state"] == {
            "llm_calls": 1,
            "probe": i - 1,
            "emitted_step": i - 1,
            "llm_input_tokens": 5,
        }
        before_score = 100.0 if i == 1 else expected_best[i - 2]
        assert event["incumbent_before"]["score"] == pytest.approx(before_score)
        assert event["incumbent_after"]["score"] == pytest.approx(expected_best[i - 1])
    assert events[1]["incumbent_before"] == {"config": BASE, "score": 100.0}
    assert events[9]["incumbent_after"] == {"config": proposals[8], "score": 50.0}
    assert events[10]["incumbent_after"] == events[9]["incumbent_after"]

    # result.json matches the returned dict (modulo JSON key stringification).
    on_disk = json.loads((out_dir / "result.json").read_text())
    assert on_disk == json.loads(json.dumps(result))

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["checkpoint_id"] == "ck-0"
    assert manifest["checkpoint_hash"] == hashlib.sha256(
        (checkpoint_dir / "checkpoint.json").read_bytes()
    ).hexdigest()
    assert manifest["arm"] == "toy-arm"
    assert manifest["seed"] == 123
    assert manifest["model"] == {}
    assert manifest["active_dimensions"] == 3
    assert manifest["extra"] == {"note": "happy", "s": 1.5, "a_0": 0.01}
    assert manifest["dependencies"]["python"]
    assert manifest["dependencies"]["numpy"]
    assert manifest["hardware"]["platform"]


# --- 2. crash outcomes -----------------------------------------------------


def test_crash_consumes_budget_never_improves(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()
    feedbacks = []
    arm = scripted_arm(
        [cfg(dropout=0.01), cfg(dropout=0.02), cfg(dropout=0.03)],
        feedbacks=feedbacks,
    )
    out_dir = tmp_path / "cell"
    result = runner.run_cell(
        arm=arm,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        seed=1,
        budget=3,
        eval_fn=fake_eval_from([("ok", 90.0), ("crash", None), ("ok", 95.0)]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 3  # the crash consumed one unit
    assert result["counts"]["crashes"] == 1
    assert result["best_score"] == pytest.approx(90.0)  # 95.0 never improves
    assert result["best_config"] == cfg(dropout=0.01)

    crash_event = evaluation_events(out_dir)[1]
    assert crash_event["eval_index"] == 2
    assert crash_event["status"] == "crash"
    assert crash_event["score"] is None
    assert crash_event["detail"] == "scripted crash"
    assert crash_event["incumbent_after"]["score"] == pytest.approx(90.0)

    assert feedbacks[1].kind == "outcome"
    assert feedbacks[1].status == "crash"
    assert feedbacks[1].score is None
    assert feedbacks[1].detail == "scripted crash"


# --- 3. deterministic preflight rejects ------------------------------------


def test_deterministic_rejects_consume_no_budget(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()
    feedbacks = []
    arm = scripted_arm(
        [
            {**cfg(), "depth": 99},  # out_of_space
            {**cfg(), "depth": "abc"},  # schema_invalid (int cast fails)
            cfg(),  # duplicate of the incumbent
            cfg(dropout=0.42),  # executed
        ],
        feedbacks=feedbacks,
    )
    out_dir = tmp_path / "cell"
    result = runner.run_cell(
        arm=arm,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        seed=1,
        budget=1,
        eval_fn=fake_eval_from([("ok", 90.0)]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 1
    assert result["counts"] == {
        "crashes": 0,
        "invalid": 2,
        "duplicates": 1,
        "task_preflight_rejected": 0,
    }
    # The final evaluation fills the budget, so the runner closes the
    # generator WITHOUT sending that last feedback (documented protocol): the
    # arm observes only the 3 rejects; the outcome lives in events/result.
    assert [fb.kind for fb in feedbacks] == ["preflight_rejected"] * 3
    assert feedbacks[0].stage == "deterministic"
    assert feedbacks[0].reason.startswith("out_of_space")
    assert feedbacks[1].reason.startswith("schema_invalid")
    assert feedbacks[2].reason.startswith("duplicate")
    assert evaluation_events(out_dir)[0]["score"] == pytest.approx(90.0)

    rejects = [
        event for event in read_events(out_dir) if event["kind"] == "preflight_rejected"
    ]
    assert len(rejects) == 3
    assert all(event["eval_index"] is None for event in rejects)
    assert all(event["preflight_status"] == "rejected" for event in rejects)
    assert all(event["status"] == "preflight_rejected" for event in rejects)
    assert all(event["preflight_stage"] == "deterministic" for event in rejects)
    assert rejects[0]["preflight_detail"].startswith("out_of_space")
    assert rejects[1]["preflight_detail"].startswith("schema_invalid")
    assert rejects[2]["preflight_detail"].startswith("duplicate")


def test_five_consecutive_rejects_terminate_as_arm_error(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()
    eval_fn = fake_eval_from([])
    arm = scripted_arm([{**cfg(), "depth": 99} for _ in range(7)])
    out_dir = tmp_path / "cell"
    result = runner.run_cell(
        arm=arm,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        seed=1,
        budget=3,
        eval_fn=eval_fn,
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "arm_error"
    assert "5 consecutive preflight rejections" in result["reason"]
    assert result["evaluations"] == 0
    assert eval_fn.calls == []
    events = read_events(out_dir)
    assert [event["kind"] for event in events] == ["cell_start"] + [
        "preflight_rejected"
    ] * 5 + ["cell_end"]
    assert events[-1]["status"] == "arm_error"


def test_reject_counter_resets_on_objective_start(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()
    bad = {**cfg(), "depth": 99}
    arm = scripted_arm(
        [bad, cfg(dropout=0.11), bad, bad, bad, bad, cfg(dropout=0.22)]
    )
    result = runner.run_cell(
        arm=arm,
        checkpoint_dir=checkpoint_dir,
        out_dir=tmp_path / "cell",
        seed=1,
        budget=2,
        eval_fn=fake_eval_from([("ok", 90.0), ("ok", 80.0)]),
        preflight_fn=ok_preflight,
    )

    # 1 reject, objective start (reset), then only 4 consecutive rejects.
    assert result["status"] == "ok"
    assert result["evaluations"] == 2
    assert result["counts"]["invalid"] == 5


def test_mixed_deterministic_and_task_rejects_count_together(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()
    arm = scripted_arm(
        [
            {**cfg(), "depth": 99},  # deterministic
            cfg(depth=7),  # task
            {**cfg(), "depth": 99},  # deterministic
            cfg(depth=7),  # task
            {**cfg(), "depth": 99},  # deterministic -> 5th consecutive
        ]
    )
    result = runner.run_cell(
        arm=arm,
        checkpoint_dir=checkpoint_dir,
        out_dir=tmp_path / "cell",
        seed=1,
        budget=2,
        eval_fn=fake_eval_from([]),
        preflight_fn=task_reject_preflight(lambda params: params["depth"] == 7),
    )

    assert result["status"] == "arm_error"
    assert "5 consecutive preflight rejections" in result["reason"]
    assert result["evaluations"] == 0
    assert result["counts"]["invalid"] == 3
    assert result["counts"]["task_preflight_rejected"] == 2


# --- 4. task preflight -----------------------------------------------------


def test_task_preflight_reject_consumes_no_budget(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()
    feedbacks = []
    arm = scripted_arm(
        [cfg(depth=7), cfg(dropout=0.11), cfg(dropout=0.22)], feedbacks=feedbacks
    )
    out_dir = tmp_path / "cell"
    result = runner.run_cell(
        arm=arm,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        seed=1,
        budget=2,
        eval_fn=fake_eval_from([("ok", 90.0), ("ok", 80.0)]),
        preflight_fn=task_reject_preflight(lambda params: params["depth"] == 7),
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 2
    assert result["counts"]["task_preflight_rejected"] == 1
    assert feedbacks[0].kind == "preflight_rejected"
    assert feedbacks[0].stage == "task"
    assert feedbacks[0].reason == "task preflight rejected"
    reject = [
        event
        for event in read_events(out_dir)
        if event["kind"] == "preflight_rejected"
    ][0]
    assert reject["preflight_stage"] == "task"
    assert reject["preflight_detail"] == "task preflight rejected"


# --- 5. unsupported --------------------------------------------------------


def test_unsupported_before_first_proposal(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()

    def body(ctx):
        raise Unsupported("zero continuous dimensions")
        yield  # makes body a generator function

    out_dir = tmp_path / "cell"
    result = runner.run_cell(
        arm=ToyArm(body),
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        seed=1,
        budget=4,
        eval_fn=fake_eval_from([]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "unsupported"
    assert result["reason"] == "zero continuous dimensions"
    assert result["evaluations"] == 0
    assert result["best_score"] == pytest.approx(100.0)
    assert result["relative_improvement_at"] == {
        2: None,
        4: None,
        6: None,
        8: None,
        10: None,
        24: None,
    }
    assert result["auc"] is None
    assert result["beat_initial_incumbent"] is False
    assert result["first_improvement_eval"] is None
    assert result["final_relative_improvement"] == pytest.approx(0.0)
    events = read_events(out_dir)
    assert [event["kind"] for event in events] == ["cell_start", "cell_end"]
    assert events[-1]["status"] == "unsupported"
    assert (out_dir / "result.json").exists()


def test_unsupported_mid_cell_keeps_partial_history(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()

    def body(ctx):
        yield Proposal(params=cfg(dropout=0.11), source="toy")
        yield Proposal(params=cfg(dropout=0.22), source="toy")
        raise Unsupported("empty feasible set at poll start")

    out_dir = tmp_path / "cell"
    result = runner.run_cell(
        arm=ToyArm(body),
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        seed=1,
        budget=5,
        eval_fn=fake_eval_from([("ok", 95.0), ("ok", 90.0)]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "unsupported"
    assert result["reason"] == "empty feasible set at poll start"
    assert result["evaluations"] == 2
    assert result["best_score"] == pytest.approx(90.0)
    assert result["relative_improvement_at"][2] == pytest.approx(10.0)
    # The curve is constant past termination, so k=4 reads the same value
    # (metrics pads to the horizon rather than reporting None).
    assert result["relative_improvement_at"][4] == pytest.approx(10.0)
    # ... but AUC normalizes by budget=5, so the 3 unused evaluations cost:
    # (0 + 10 + 10 + 10 + 10) / 5 rather than a mean over 2 evaluations.
    assert result["auc"] == pytest.approx((5.0 + 10.0 * 4) / 5)
    assert result["beat_initial_incumbent"] is True
    assert result["first_improvement_eval"] == 1
    assert [event["kind"] for event in read_events(out_dir)] == [
        "cell_start",
        "evaluation",
        "evaluation",
        "cell_end",
    ]


# --- 6. early return -------------------------------------------------------


def test_arm_returning_early_is_arm_error(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()

    def body(ctx):
        yield Proposal(params=cfg(dropout=0.11), source="toy")
        yield Proposal(params=cfg(dropout=0.22), source="toy")
        return  # budget remains

    result = runner.run_cell(
        arm=ToyArm(body),
        checkpoint_dir=checkpoint_dir,
        out_dir=tmp_path / "cell",
        seed=1,
        budget=5,
        eval_fn=fake_eval_from([("ok", 90.0), ("ok", 80.0)]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "arm_error"
    assert "arm exhausted early" in result["reason"]
    assert result["evaluations"] == 2


def test_arm_error_paths(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()

    def raise_arm_error(ctx):
        yield Proposal(params=cfg(dropout=0.11), source="toy")
        raise ArmError("duplicate-resample exhaustion")

    def raise_unhandled(ctx):
        yield Proposal(params=cfg(dropout=0.11), source="toy")
        raise RuntimeError("boom")

    def yield_non_proposal(ctx):
        yield {"not": "a proposal"}

    cases = [
        ("arm_error", raise_arm_error, "duplicate-resample exhaustion", False),
        ("unhandled", raise_unhandled, "unhandled arm exception", True),
        ("non_proposal", yield_non_proposal, "non-Proposal", False),
    ]
    for name, body, reason_part, has_traceback in cases:
        result = runner.run_cell(
            arm=ToyArm(body),
            checkpoint_dir=checkpoint_dir,
            out_dir=tmp_path / f"cell_{name}",
            seed=1,
            budget=3,
            eval_fn=fake_eval_from([("ok", 90.0)]),
            preflight_fn=ok_preflight,
        )
        assert result["status"] == "arm_error"
        assert reason_part in result["reason"]
        if has_traceback:
            assert "RuntimeError: boom" in result["traceback"]
        else:
            assert result["traceback"] is None
        assert read_events(tmp_path / f"cell_{name}")[-1]["status"] == "arm_error"


# --- 7. seed determinism ---------------------------------------------------


def test_seed_determinism_of_ctx_rngs(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()

    def body(ctx):
        for _ in range(6):
            depth = ctx.rng.randint(1, 8)
            dropout = ctx.np_rng.random() * 0.5
            yield Proposal(params=cfg(depth=depth, dropout=dropout), source="stoch")

    def run(seed, out_name):
        return runner.run_cell(
            arm=ToyArm(body),
            checkpoint_dir=checkpoint_dir,
            out_dir=tmp_path / out_name,
            seed=seed,
            budget=6,
            eval_fn=fake_eval_by_params(
                lambda params: params["depth"] * 10.0 + params["dropout"]
            ),
            preflight_fn=ok_preflight,
        )

    result_a = run(42, "cell_a")
    result_b = run(42, "cell_b")
    run(7, "cell_c")
    trace_a = [
        (event["proposal"], event["score"])
        for event in evaluation_events(tmp_path / "cell_a")
    ]
    trace_b = [
        (event["proposal"], event["score"])
        for event in evaluation_events(tmp_path / "cell_b")
    ]
    trace_c = [
        (event["proposal"], event["score"])
        for event in evaluation_events(tmp_path / "cell_c")
    ]
    assert result_a["status"] == result_b["status"] == "ok"
    assert len(trace_a) == 6
    assert trace_a == trace_b
    assert trace_a != trace_c


# --- 8. checkpoint history visible to arms ---------------------------------


def test_finite_unique_history_visible_via_ctx(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint(
        incumbent_params=cfg(depth=4),
        incumbent_score=45.0,
        history=[
            hrow(cfg(depth=1), 50.0),
            hrow(cfg(depth=1), 55.0),  # duplicate identity -> dropped
            hrow(cfg(depth=2), None, status="crash"),  # excluded
            hrow(cfg(depth=3), 60.0),
        ],
        deferred=[{"params": cfg(depth=8)}],
    )
    captured = {}

    def body(ctx):
        captured["finite_unique"] = ctx.checkpoint.finite_unique_history(ctx.contract)
        captured["trials_at_start"] = len(ctx.state.trials)
        captured["budget_at_start"] = ctx.state.budget_remaining
        captured["deferred"] = ctx.checkpoint.deferred_configs
        yield Proposal(params=cfg(depth=5), source="toy")

    result = runner.run_cell(
        arm=ToyArm(body),
        checkpoint_dir=checkpoint_dir,
        out_dir=tmp_path / "cell",
        seed=1,
        budget=1,
        eval_fn=fake_eval_from([("ok", 40.0)]),
        preflight_fn=ok_preflight,
    )

    # incumbent first, then history in order, deduped by identity, no crashes.
    assert captured["finite_unique"] == [
        (cfg(depth=4), 45.0),
        (cfg(depth=1), 50.0),
        (cfg(depth=3), 60.0),
    ]
    assert captured["trials_at_start"] == 5  # 4 history rows + the incumbent
    assert captured["budget_at_start"] == 1  # history consumed no budget
    assert captured["deferred"] == (cfg(depth=8),)
    assert result["evaluations"] == 1
    assert result["best_score"] == pytest.approx(40.0)


# --- 9. exact budget accounting --------------------------------------------


def test_budget_exact_with_interleaved_rejects_and_crashes(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()
    eval_fn = fake_eval_by_params(
        lambda params: params["depth"] * 10.0,
        crash_if=lambda params: params["depth"] == 2,
    )
    arm = scripted_arm(
        [
            {**cfg(), "depth": 99},  # deterministic reject
            cfg(depth=1),  # ok -> 10
            cfg(depth=2),  # crash
            cfg(depth=7),  # task reject
            {**cfg(), "depth": 42},  # deterministic reject
            cfg(depth=3),  # ok -> 30
            cfg(depth=5),  # ok -> 50; budget exhausted here
            cfg(depth=6),  # never processed
        ]
    )
    out_dir = tmp_path / "cell"
    result = runner.run_cell(
        arm=arm,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        seed=1,
        budget=4,
        eval_fn=eval_fn,
        preflight_fn=task_reject_preflight(lambda params: params["depth"] == 7),
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 4
    assert result["counts"] == {
        "crashes": 1,
        "invalid": 2,
        "duplicates": 0,
        "task_preflight_rejected": 1,
    }
    assert result["best_score"] == pytest.approx(10.0)
    assert [call["depth"] for call in eval_fn.calls] == [1, 2, 3, 5]
    assert [event["eval_index"] for event in evaluation_events(out_dir)] == [1, 2, 3, 4]


# --- checkpoint loader validation ------------------------------------------


@pytest.mark.parametrize(
    "mutate,exc",
    [
        (lambda p: p.update(schema_version=1), ValueError),
        (lambda p: p.update(regime="continuation"), ValueError),  # stratum stays "first"
        (lambda p: p.update(stratum="deep"), ValueError),
        (lambda p: p.update(incumbent={"params": cfg(), "score": float("inf")}), ValueError),
        (lambda p: p.update(incumbent={"params": [1, 2], "score": 1.0}), ValueError),
        (lambda p: p.update(history=[{"params": "nope", "score": 1.0, "status": "ok"}]), ValueError),
        (lambda p: p.update(history=[hrow(cfg(), None, status="ok")]), ValueError),
        (lambda p: p.update(history=[hrow(cfg(), 3.0, status="crash")]), ValueError),
        (lambda p: p.update(candidate_relpath="missing"), FileNotFoundError),
        (
            lambda p: p.update(
                task={"score_fn": "", "preflight_fn": "x", "per_runtime_limit": 1}
            ),
            ValueError,
        ),
    ],
)
def test_load_checkpoint_validation(make_checkpoint, mutate, exc):
    directory = make_checkpoint(payload_override=mutate)
    with pytest.raises(exc):
        checkpoint_mod.load_checkpoint(directory)


# --- 10. pre-drive arm failures contained as arm_error ----------------------


def test_active_dimensions_hook_failure_contained(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()

    def boom(contract):
        raise RuntimeError("active_dimensions boom")

    def body(ctx):  # never reached
        yield Proposal(params=cfg(dropout=0.11), source="toy")

    arm = ToyArm(body, active_dimensions=boom)
    out_dir = tmp_path / "cell"
    result = runner.run_cell(
        arm=arm,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        seed=1,
        budget=3,
        eval_fn=fake_eval_from([]),  # asserts if ever called
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "arm_error"
    assert "pre-drive arm hook" in result["reason"]
    assert "RuntimeError: active_dimensions boom" in result["traceback"]
    assert result["evaluations"] == 0
    # result.json exists, parses, and matches the returned dict.
    on_disk = json.loads((out_dir / "result.json").read_text())
    assert on_disk == json.loads(json.dumps(result))
    events = read_events(out_dir)
    assert [event["kind"] for event in events] == ["cell_start", "cell_end"]
    assert events[-1]["status"] == "arm_error"
    # Manifest still written; the failed field fell back to null / {}.
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["active_dimensions"] is None
    assert manifest["extra"] == {}


def test_calibration_report_hook_failure_contained(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()

    def boom(ctx):
        raise RuntimeError("calibration boom")

    def body(ctx):  # never reached
        yield Proposal(params=cfg(dropout=0.11), source="toy")

    arm = ToyArm(
        body,
        active_dimensions=lambda contract: 2,
        calibration_report=boom,
    )
    out_dir = tmp_path / "cell"
    result = runner.run_cell(
        arm=arm,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        seed=1,
        budget=3,
        eval_fn=fake_eval_from([]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "arm_error"
    assert "pre-drive arm hook" in result["reason"]
    assert "RuntimeError: calibration boom" in result["traceback"]
    on_disk = json.loads((out_dir / "result.json").read_text())
    assert on_disk["status"] == "arm_error"
    assert "RuntimeError: calibration boom" in on_disk["traceback"]
    assert [event["kind"] for event in read_events(out_dir)] == [
        "cell_start",
        "cell_end",
    ]
    manifest = json.loads((out_dir / "manifest.json").read_text())
    # The hook that ran before the failure kept its value; calibration fell
    # back to {}.
    assert manifest["active_dimensions"] == 2
    assert manifest["extra"] == {}


def test_run_call_raising_before_first_yield_contained(make_checkpoint, tmp_path):
    checkpoint_dir = make_checkpoint()

    def not_a_generator(ctx):  # raises at the arm.run(ctx) call itself
        raise RuntimeError("run boom")

    out_dir = tmp_path / "cell"
    result = runner.run_cell(
        arm=ToyArm(not_a_generator),
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        seed=1,
        budget=3,
        eval_fn=fake_eval_from([]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "arm_error"
    assert result["reason"] == "unhandled arm exception"
    assert "RuntimeError: run boom" in result["traceback"]
    assert result["evaluations"] == 0
    on_disk = json.loads((out_dir / "result.json").read_text())
    assert on_disk == json.loads(json.dumps(result))
    events = read_events(out_dir)
    assert [event["kind"] for event in events] == ["cell_start", "cell_end"]
    assert events[-1]["status"] == "arm_error"


def test_eval_subprocess_runs_under_task_project(make_checkpoint, tmp_path):
    """Environment partition: with a real task.project the evaluation and
    preflight subprocesses run under ``uv --project <repo>/<project> run
    python`` (production's split), never the cell's own interpreter. A
    regression here silently evaluates in the root env — this is the only
    assertion that pins the wiring end to end."""
    seen = {"eval": [], "preflight": []}

    def fake_eval(candidate_path, params, *, score_fn, per_runtime_limit, python_cmd=None):
        seen["eval"].append(python_cmd)
        return objective.EvalOutcome(
            status="ok", score=1.0, detail=None, elapsed_seconds=0.0
        )

    def fake_preflight(candidate_path, params, *, preflight_fn, per_runtime_limit, python_cmd=None):
        seen["preflight"].append(python_cmd)
        return objective.PreflightOutcome(status="ok", detail=None)

    checkpoint_dir = make_checkpoint(
        task={
            "score_fn": "evaluate_config",
            "preflight_fn": "preflight_config",
            "per_runtime_limit": 900,
            "project": "tasks/toy-task",
        }
    )
    result = runner.run_cell(
        arm=scripted_arm([cfg(dropout=0.01)]),
        checkpoint_dir=checkpoint_dir,
        out_dir=tmp_path / "cell",
        seed=1,
        budget=1,
        eval_fn=fake_eval,
        preflight_fn=fake_preflight,
    )

    assert result["status"] == "ok"
    assert len(seen["eval"]) == 1
    assert len(seen["preflight"]) == 1
    for cmd in seen["eval"] + seen["preflight"]:
        assert cmd is not None
        assert cmd[:2] == ["uv", "--project"]
        assert cmd[2].endswith("tasks/toy-task")
        assert cmd[3:] == ["run", "python"]


def test_tie_score_is_not_an_improvement(make_checkpoint, tmp_path):
    """§5.2: improvement is STRICTLY below the running incumbent — a score
    equal to it moves nothing (mutation anchor for metrics' comparison)."""
    result = runner.run_cell(
        arm=scripted_arm([cfg(dropout=0.01), cfg(dropout=0.02)]),
        checkpoint_dir=make_checkpoint(incumbent_score=100.0),
        out_dir=tmp_path / "cell",
        seed=1,
        budget=2,
        eval_fn=fake_eval_from([("ok", 100.0), ("ok", 99.999)]),
        preflight_fn=ok_preflight,
    )

    assert result["status"] == "ok"
    # The 100.0 tie was not an improvement; 99.999 was, at evaluation 2.
    assert result["first_improvement_eval"] == 2
    assert result["final_best_score"] == 99.999
    assert result["beat_initial_incumbent"] is True
