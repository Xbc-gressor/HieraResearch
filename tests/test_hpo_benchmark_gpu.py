from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.hpo_benchmark.core import ConfigInfeasibleError
from tools.hpo_benchmark.gpu import CandidateObjective, _source_checkpoint
from tools.hpo_benchmark.providers import ClaudeCLIProposalProvider


def test_continuation_checkpoint_stops_after_ten_admitted_trials(tmp_path: Path):
    trials = [
        {"params": {"x": 99}, "score": None, "status": "preflight_rejected"},
        *({"params": {"x": index}, "score": float(index)} for index in range(11)),
    ]
    report = {
        "phase_a": {
            "search_space": {"x": ["int", 0, 20]},
            "warm_start_configs": [
                {
                    "params": {"x": 1},
                    "score": 1.0,
                    "role": "inherited_control",
                },
                {"params": {"x": 2}, "score": 2.0},
            ],
            "deferred_configs": [{"params": {"x": 3}}],
        },
        "search_space_clamp": {
            "clamped_search_space": {"x": ["int", 0, 20]}
        },
        "phase_c": {"stages": [{"trials": trials}]},
    }
    (tmp_path / "tune_report.json").write_text(json.dumps(report))

    space, rows, deferred = _source_checkpoint(tmp_path, "continuation")

    assert space == {"x": ["int", 0, 20]}
    assert len(rows) == 12
    assert rows[0]["eligible_incumbent"] is False
    assert [row["params"]["x"] for row in rows[-10:]] == list(range(10))
    assert deferred == [{"x": 3}]


def test_claude_cli_provider_reads_structured_output(tmp_path: Path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "structured_output": {"changes": {"x": 1}},
                    "total_cost_usd": 0.01,
                    "session_id": "fresh-session",
                }
            ),
            stderr="",
        )

    provider = ClaudeCLIProposalProvider(
        model="sonnet", cwd=tmp_path, command_runner=fake_run
    )
    response = provider.complete(
        "prompt", output_schema={"type": "object", "properties": {}}
    )

    assert response.output == {"changes": {"x": 1}}
    assert response.metadata["total_cost_usd"] == 0.01
    assert response.metadata["provider_attempt_count"] == 1
    assert "--no-session-persistence" in calls[0][0]
    assert calls[0][1]["cwd"] == tmp_path


def test_claude_cli_provider_retries_two_nonzero_exits(tmp_path: Path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) < 3:
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr="transient failure"
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"structured_output": {"changes": {"x": 1}}}),
            stderr="",
        )

    provider = ClaudeCLIProposalProvider(
        model="sonnet", cwd=tmp_path, command_runner=fake_run
    )
    response = provider.complete(
        "prompt", output_schema={"type": "object", "properties": {}}
    )

    assert len(calls) == 3
    assert response.metadata["provider_attempt_count"] == 3
    assert response.metadata["provider_retry_count"] == 2
    assert response.metadata["provider_total_attempts"] == 3


def test_candidate_objective_only_classifies_config_specific_failures(
    tmp_path: Path, monkeypatch
):
    objective = CandidateObjective(tmp_path / "train.py", evaluation_timeout=1.0)

    def unknown_preflight(params, candidate_path):
        raise RuntimeError("preflight subprocess protocol failure")

    monkeypatch.setattr(
        "tools.hpo_benchmark.gpu.timed_preflight", unknown_preflight
    )
    with pytest.raises(RuntimeError, match="protocol failure"):
        objective.preflight({"x": 1})

    def timed_out(command, *, limit, label):
        raise TimeoutError("evaluation timed out")

    monkeypatch.setattr(
        "tools.hpo_benchmark.gpu._communicate_with_limit", timed_out
    )
    with pytest.raises(ConfigInfeasibleError, match="timed out"):
        objective.evaluate({"x": 1})
