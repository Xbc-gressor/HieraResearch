"""Multifidelity adapter child: env.train_budget_seconds is the only override,
and crash / non-finite / budget-noncompliance classify correctly.

The toy candidate mirrors the production tuner contract (PARAM_SCHEMA /
SEARCH_SPACE / BASE_PARAMS / make_model + a sibling prepare.py exposing
PretrainEnv and evaluate_config) so the child's real load_candidate_modules
boundary is exercised, in a fresh subprocess exactly as run_matrix launches it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "multifidelity"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import manifest  # noqa: E402
import run_matrix  # noqa: E402
from tune_tools import _candidate_execution_revision  # noqa: E402

CHILD = ROOT / "tools" / "multifidelity" / "autoresearch_eval_one.py"

TRAIN_SOURCE = '''
import time

PARAM_SCHEMA = {"x": "float"}
SEARCH_SPACE = {"x": ("float", 0.0, 10.0)}
BASE_PARAMS = {"x": 1.0}


class Trainer:
    def __init__(self, env, params):
        self.env = env
        self.params = params

    def run(self):
        budget = self.env.train_budget_seconds
        x = self.params["x"]
        if x == 2.0:
            raise RuntimeError("toy crash")
        if x == 3.0:
            return float("nan")
        if x == 5.0:
            time.sleep(30)
        # x == 4.0 ignores the requested budget (truncation-style candidate)
        completed = 300.0 if x == 4.0 else float(budget)
        print("---")
        print(f"val_bpb:          {0.5:.6f}")
        print(f"training_seconds: {completed:.1f}")
        print(f"total_seconds:    {completed + 1.0:.1f}")
        print(f"num_steps:        12")
        return 0.5


def make_model(env, params):
    return Trainer(env, params)
'''

PREPARE_SOURCE = '''
import math


class PretrainEnv:
    def __init__(self):
        self.train_budget_seconds = 300
        self.seed = 42


def evaluate_config(make_model, params):
    env = PretrainEnv()
    val = float(make_model(env, params).run())
    if not math.isfinite(val):
        raise ValueError(f"non-finite {val!r}")
    return val
'''


def make_candidate(tmp_path: Path) -> Path:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "train.py").write_text(TRAIN_SOURCE)
    (candidate_dir / "prepare.py").write_text(PREPARE_SOURCE)
    return candidate_dir / "train.py"


def run_child(tmp_path: Path, train_path: Path, *, x: float, seconds: int = 30):
    request = manifest.build_request(
        experiment_id="exp-toy",
        pool_id="pool-toy",
        candidate_id="cand-toy",
        candidate_path=str(train_path),
        candidate_execution_revision=_candidate_execution_revision(train_path),
        params={"x": x},
        requested_train_seconds=seconds,
        purpose="matrix",
        repeat_index=0,
        seed=42,
        gpu_id=0,
        gpu_uuid="GPU-toy",
        task_artifact_digest="sha256:toy",
    )
    job_dir = tmp_path / "jobs" / request["job_id"]
    job_dir.mkdir(parents=True)
    manifest.atomic_write_json(job_dir / manifest.REQUEST_FILENAME, request)
    with open(job_dir / manifest.STDOUT_FILENAME, "wb") as out, open(
        job_dir / manifest.STDERR_FILENAME, "wb"
    ) as err:
        proc = subprocess.run(
            [sys.executable, str(CHILD), str(job_dir)],
            stdout=out,
            stderr=err,
        )
    assert proc.returncode == 0
    return manifest.load_json(job_dir / manifest.PROVISIONAL_FILENAME)


def test_adapter_overrides_only_the_train_budget(tmp_path):
    result = run_child(tmp_path, make_candidate(tmp_path), x=1.0, seconds=30)
    assert result["status"] == "ok"
    assert result["score"] == 0.5
    # the toy trainer echoes env.train_budget_seconds — proof the 300s default
    # was asserted and replaced by the requested value, nothing else
    assert result["completed_train_seconds"] == 30.0
    assert result["num_steps"] == 12


def test_adapter_classifies_crash(tmp_path):
    result = run_child(tmp_path, make_candidate(tmp_path), x=2.0)
    assert result["status"] == "crash"
    assert result["score"] is None
    assert "toy crash" in result["error"]


def test_adapter_classifies_non_finite_as_crash(tmp_path):
    result = run_child(tmp_path, make_candidate(tmp_path), x=3.0)
    assert result["status"] == "crash"
    assert result["score"] is None


def test_adapter_classifies_budget_noncompliance(tmp_path):
    # requested 30s, candidate reports 300s: outside max(2, 0.15*30)
    result = run_child(tmp_path, make_candidate(tmp_path), x=4.0)
    assert result["status"] == "budget_contract_failure"
    assert result["score"] is None


def test_parent_timeout_is_terminal_and_kills_the_job(tmp_path):
    train_path = make_candidate(tmp_path)
    request = manifest.build_request(
        experiment_id="exp-toy",
        pool_id="pool-toy",
        candidate_id="cand-toy",
        candidate_path=str(train_path),
        candidate_execution_revision=_candidate_execution_revision(train_path),
        params={"x": 5.0},
        requested_train_seconds=30,
        purpose="matrix",
        repeat_index=0,
        seed=42,
        gpu_id=0,
        gpu_uuid="GPU-toy",
        task_artifact_digest="sha256:toy",
    )
    job_dir = tmp_path / "jobs" / request["job_id"]
    job_dir.mkdir(parents=True)
    manifest.atomic_write_json(job_dir / manifest.REQUEST_FILENAME, request)
    result = run_matrix.run_one_job(
        job_dir, python_cmd=[sys.executable], timeout_seconds=3.0
    )
    assert result["status"] == "timeout"
    assert result["score"] is None
    assert result["elapsed_accelerator_seconds"] >= 3.0
    assert result["stdout_digest"]
    # terminal: resume skips it
    assert run_matrix.resume_action(job_dir) == "skip"
