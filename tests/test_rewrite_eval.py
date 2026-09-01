"""Tests for tools/rewrite_eval.py (rewrite-operator measurement CLI)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import rewrite_eval  # noqa: E402

TRAIN_SRC = '''\
BASE_PARAMS = {
    "x": 1.0,
    "depth": 3,
}
'''


def _run_cli(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", argv)
    code = rewrite_eval.main()
    return code, json.loads(capsys.readouterr().out)


def test_read_base_params(tmp_path) -> None:
    train = tmp_path / "train.py"
    train.write_text(TRAIN_SRC)
    assert rewrite_eval.read_base_params(train) == {"x": 1.0, "depth": 3}


def test_read_base_params_rejects_missing_and_non_literal(tmp_path) -> None:
    train = tmp_path / "train.py"
    train.write_text("PARAMS = {'x': 1}\n")
    with pytest.raises(ValueError, match="BASE_PARAMS"):
        rewrite_eval.read_base_params(train)
    train.write_text("BASE_PARAMS = {'x': len('ab')}\n")
    with pytest.raises(ValueError, match="literal"):
        rewrite_eval.read_base_params(train)


def test_cli_success(tmp_path, monkeypatch, capsys) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "train.py").write_text(TRAIN_SRC)
    calls = {}

    def fake_timed_eval(evaluate, make_model, params, candidate_path, **kwargs):
        calls.update(params=params, candidate_path=candidate_path, kwargs=kwargs)
        return 0.42

    monkeypatch.setattr(rewrite_eval, "timed_eval", fake_timed_eval)
    code, payload = _run_cli(
        monkeypatch, capsys, ["rewrite_eval.py", "--candidate", str(candidate)]
    )
    assert code == 0
    assert payload == {"attempt_id": None, "score": 0.42, "error": None,
                       "stage": "eval"}
    assert calls["params"] == {"x": 1.0, "depth": 3}
    assert calls["candidate_path"] == candidate / "train.py"
    assert calls["kwargs"]["phase"] == "rewrite"
    assert calls["kwargs"]["method"] == "rewrite"


def test_cli_eval_error(tmp_path, monkeypatch, capsys) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "train.py").write_text(TRAIN_SRC)

    def fake_timed_eval(*args, **kwargs):
        raise RuntimeError("evaluation subprocess died")

    monkeypatch.setattr(rewrite_eval, "timed_eval", fake_timed_eval)
    code, payload = _run_cli(
        monkeypatch, capsys, ["rewrite_eval.py", "--candidate", str(candidate)]
    )
    assert code == 0
    assert payload["stage"] == "eval"
    assert payload["attempt_id"] is None
    assert payload["score"] is None
    assert "evaluation subprocess died" in payload["error"]


def test_cli_budget_exhausted_exits_4(tmp_path, monkeypatch, capsys) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "train.py").write_text(TRAIN_SRC)

    def fake_timed_eval(*args, **kwargs):
        raise rewrite_eval.EvaluationBudgetExhausted(used=9, budget=9, run_dir=tmp_path)

    monkeypatch.setattr(rewrite_eval, "timed_eval", fake_timed_eval)
    code, payload = _run_cli(
        monkeypatch, capsys, ["rewrite_eval.py", "--candidate", str(candidate)]
    )
    assert code == 4
    assert payload["stage"] == "eval"
    assert payload["attempt_id"] is None
    assert payload["score"] is None


def test_cli_param_failure_spends_no_budget(tmp_path, monkeypatch, capsys) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "train.py").write_text("BASE_PARAMS = {'x': len('ab')}\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("timed_eval must not run when params are unreadable")

    monkeypatch.setattr(rewrite_eval, "timed_eval", forbidden)
    code, payload = _run_cli(
        monkeypatch, capsys, ["rewrite_eval.py", "--candidate", str(candidate)]
    )
    assert code == 0
    assert payload["stage"] == "params"
    assert payload["attempt_id"] is None
    assert payload["score"] is None
    assert "BASE_PARAMS" in payload["error"]


def test_cli_recovers_attempt_id(tmp_path, monkeypatch, capsys) -> None:
    run_dir = tmp_path / "runs" / "unit-rewrite" / "tag"
    candidate = run_dir / "candidates" / "007"
    candidate.mkdir(parents=True)
    (run_dir / "framework_cfg.json").write_text("{}")
    (candidate / "train.py").write_text(TRAIN_SRC)

    def fake_timed_eval(evaluate, make_model, params, candidate_path, **kwargs):
        # timed_eval's reservation appends the receipt row before evaluating
        row = {
            "schema_version": 1,
            "kind": "score_attempt",
            "attempt_id": "eval-000001",
            "run_id": "007",
            "phase": "rewrite",
            "method": "rewrite",
            "params": params,
        }
        with (run_dir / "evaluation_attempts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return 0.3

    monkeypatch.setattr(rewrite_eval, "timed_eval", fake_timed_eval)
    code, payload = _run_cli(
        monkeypatch, capsys, ["rewrite_eval.py", "--candidate", str(candidate)]
    )
    assert code == 0
    assert payload == {"attempt_id": "eval-000001", "score": 0.3,
                       "error": None, "stage": "eval"}
