"""Shared fixtures/helpers for inner-benchmark ARM tests (test_inner_benchmark_arm_*).

Plain functions (not pytest fixtures) so every arm test file wires its own
scenarios: write_checkpoint mirrors the runner test's make_checkpoint fixture,
the fake eval/preflight callables mirror its sanctioned test seam (plus the
python_cmd kwarg). Same production-literal toy contract as
tests/test_inner_benchmark_runner.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))

import objective  # noqa: E402

# Same production-literal toy contract as tests/test_inner_benchmark_runner.py.
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


def preflight_config(make_model, params):
    return None
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


def write_checkpoint(
    base_dir,
    *,
    name="ckpt",
    train_source=TRAIN_SOURCE,
    incumbent_params=None,
    incumbent_score=100.0,
    history=(),
    regime="first",
    stratum="first",
    task=None,
    deferred=(),
    inherited=False,
    extra=None,
) -> Path:
    """Write a toy checkpoint dir (checkpoint.json + candidate/{train,prepare}.py)."""
    directory = Path(base_dir) / name
    candidate_dir = directory / "candidate"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "train.py").write_text(train_source)
    (candidate_dir / "prepare.py").write_text(PREPARE_SOURCE)
    payload = {
        "schema_version": 2,
        "checkpoint_id": name,
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
        # run_cell refuses a checkpoint without a valid completed re-measurement;
        # arm tests are about the arm protocol, so checkpoints are remeasured by
        # default.
        "extra": {"remeasure": {"valid": True, "complete": True}}
        if extra is None
        else extra,
    }
    (directory / "checkpoint.json").write_text(json.dumps(payload))
    return directory


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


def read_events(out_dir):
    return [
        json.loads(line)
        for line in (Path(out_dir) / "events.jsonl").read_text().splitlines()
    ]


def evaluation_events(out_dir):
    return [event for event in read_events(out_dir) if event["kind"] == "evaluation"]
