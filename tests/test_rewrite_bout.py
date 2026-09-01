"""Tests for tools/rewrite_bout.py (rewrite-operator bout adjudication)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import rewrite_bout  # noqa: E402


def _make_candidate(tmp_path: Path, baseline: float = 1.0) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "_import.json").write_text(json.dumps({"baseline_score": baseline}))
    return candidate


def _finalize(monkeypatch, capsys, candidate: Path, bout: int, score: str | None,
              margin: str = "0.1", attempt_id: str | None = "default"):
    argv = [
        "rewrite_bout.py",
        "finalize",
        "--candidate", str(candidate),
        "--bout", str(bout),
        "--noise-margin", margin,
        "--summary", f"bout {bout} summary",
        "--basis", f"bout {bout} basis",
    ]
    if attempt_id is not None:
        argv += ["--attempt-id",
                 f"eval-{bout:06d}" if attempt_id == "default" else attempt_id]
    argv += ["--nonfinite"] if score is None else ["--score", score]
    monkeypatch.setattr(sys, "argv", argv)
    code = rewrite_bout.main()
    return code, json.loads(capsys.readouterr().out)


def test_classify_boundaries() -> None:
    classify = rewrite_bout.classify
    assert classify(1.0, None, 0.1) == "reverted_crash"
    assert classify(1.0, float("nan"), 0.1) == "reverted_crash"
    assert classify(1.0, float("inf"), 0.1) == "reverted_crash"
    assert classify(1.0, 0.85, 0.1) == "kept"
    assert classify(1.0, 0.9, 0.1) == "reverted_marginal"  # exactly the margin
    assert classify(1.0, 0.95, 0.1) == "reverted_marginal"
    assert classify(1.0, 1.0, 0.1) == "reverted_worse"  # no improvement at all
    assert classify(1.0, 1.2, 0.1) == "reverted_worse"
    # margin=0 degenerates to strict improvement
    assert classify(1.0, 0.999, 0.0) == "kept"
    assert classify(1.0, 1.0, 0.0) == "reverted_worse"


def test_snapshot_and_revert_restore_bytes(tmp_path) -> None:
    candidate = _make_candidate(tmp_path)
    original = "BASE_PARAMS = {'x': 1}\n# 中文 comment\n".encode("utf-8")
    (candidate / "train.py").write_bytes(original)

    snap = rewrite_bout.snapshot(candidate, 3)

    assert snap == candidate / "_rewrite" / "bout-003.pre.py"
    assert snap.read_bytes() == original
    assert rewrite_bout.load_bouts(candidate) == []

    (candidate / "train.py").write_bytes(b"broken edit\n")
    rewrite_bout.revert(candidate, snap)
    assert (candidate / "train.py").read_bytes() == original


def test_snapshot_replay_restores_interrupted_edit(tmp_path) -> None:
    """A bout replayed after a mid-bout kill must not adopt the dirty file."""
    candidate = _make_candidate(tmp_path)
    original = b"BASE_PARAMS = {'x': 1}\n"
    (candidate / "train.py").write_bytes(original)
    snap = rewrite_bout.snapshot(candidate, 1)

    # driver killed after the editor's write, before finalize/revert
    (candidate / "train.py").write_bytes(b"unverified edit\n")

    assert rewrite_bout.snapshot(candidate, 1) == snap
    assert (candidate / "train.py").read_bytes() == original
    assert snap.read_bytes() == original


def test_current_best(tmp_path) -> None:
    candidate = _make_candidate(tmp_path, baseline=1.0)
    assert rewrite_bout.current_best(candidate) == 1.0
    rewrite_bout.append_bout(candidate, {"bout": 1, "outcome": "reverted_worse", "score": 0.5})
    assert rewrite_bout.current_best(candidate) == 1.0
    rewrite_bout.append_bout(candidate, {"bout": 2, "outcome": "kept", "score": 0.8})
    assert rewrite_bout.current_best(candidate) == 0.8
    rewrite_bout.append_bout(candidate, {"bout": 3, "outcome": "kept", "score": 0.9})
    assert rewrite_bout.current_best(candidate) == 0.8


def test_current_best_requires_a_finite_reference(tmp_path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with pytest.raises(ValueError):
        rewrite_bout.current_best(candidate)
    # a crashed import (baseline +Infinity) is not a reference point either
    (candidate / "_import.json").write_text(json.dumps({"baseline_score": float("inf")}))
    with pytest.raises(ValueError):
        rewrite_bout.current_best(candidate)
    rewrite_bout.append_bout(candidate, {"bout": 1, "outcome": "kept", "score": 0.8})
    assert rewrite_bout.current_best(candidate) == 0.8


def test_consecutive_non_kept() -> None:
    count = rewrite_bout.consecutive_non_kept
    assert count([]) == 0
    assert count([{"outcome": "kept"}]) == 0
    assert count([{"outcome": "reverted_marginal"}]) == 1
    assert count([
        {"outcome": "kept"},
        {"outcome": "reverted_worse"},
        {"outcome": "reverted_crash"},
    ]) == 2


def test_finalize_reverts_non_kept_byte_exactly(tmp_path, monkeypatch, capsys) -> None:
    candidate = _make_candidate(tmp_path, baseline=1.0)
    original = b"BASE_PARAMS = {'x': 1}\n"
    (candidate / "train.py").write_bytes(original)
    rewrite_bout.snapshot(candidate, 1)
    (candidate / "train.py").write_bytes(b"BASE_PARAMS = {'x': 1}\n# worse edit\n")

    code, payload = _finalize(monkeypatch, capsys, candidate, 1, "1.2")

    assert code == 0
    assert payload == {"outcome": "reverted_worse", "best": 1.0}
    assert (candidate / "train.py").read_bytes() == original
    bouts = rewrite_bout.load_bouts(candidate)
    assert bouts == [{
        "bout": 1,
        "attempt_id": "eval-000001",
        "score": 1.2,
        "outcome": "reverted_worse",
        "summary": "bout 1 summary",
        "basis": "bout 1 basis",
        "snapshot": "bout-001.pre.py",
    }]


def test_finalize_kept_then_crash(tmp_path, monkeypatch, capsys) -> None:
    candidate = _make_candidate(tmp_path, baseline=1.0)
    (candidate / "train.py").write_bytes(b"v0\n")

    # bout 1: improvement beyond the margin -> kept, edit survives, best drops
    rewrite_bout.snapshot(candidate, 1)
    (candidate / "train.py").write_bytes(b"v1\n")
    code, payload = _finalize(monkeypatch, capsys, candidate, 1, "0.7")
    assert code == 0
    assert payload == {"outcome": "kept", "best": 0.7}
    assert (candidate / "train.py").read_bytes() == b"v1\n"
    assert rewrite_bout.current_best(candidate) == 0.7

    # bout 2: crash -> byte-exact rollback to the kept v1, null score journaled
    # (--attempt-id omitted: an unrecoverable attempt id is journaled as null)
    rewrite_bout.snapshot(candidate, 2)
    (candidate / "train.py").write_bytes(b"v2-broken\n")
    code, payload = _finalize(monkeypatch, capsys, candidate, 2, None,
                              attempt_id=None)
    assert code == 0
    assert payload == {"outcome": "reverted_crash", "best": 0.7}
    assert (candidate / "train.py").read_bytes() == b"v1\n"
    bouts = rewrite_bout.load_bouts(candidate)
    assert [entry["outcome"] for entry in bouts] == ["kept", "reverted_crash"]
    assert bouts[1]["score"] is None
    assert bouts[1]["attempt_id"] is None
    assert rewrite_bout.consecutive_non_kept(bouts) == 1


def test_finalize_refuses_revert_without_snapshot(tmp_path, monkeypatch, capsys) -> None:
    candidate = _make_candidate(tmp_path, baseline=1.0)
    (candidate / "train.py").write_bytes(b"edited without a snapshot\n")

    with pytest.raises(SystemExit):
        _finalize(monkeypatch, capsys, candidate, 1, "1.2")

    # nothing journaled, edit left in place for manual inspection
    assert not (candidate / "_rewrite" / "bouts.jsonl").exists()


def test_finalize_rejects_negative_noise_margin(tmp_path, monkeypatch, capsys) -> None:
    candidate = _make_candidate(tmp_path, baseline=1.0)
    (candidate / "train.py").write_bytes(b"v1\n")
    rewrite_bout.snapshot(candidate, 1)

    with pytest.raises(SystemExit):
        _finalize(monkeypatch, capsys, candidate, 1, "0.5", margin="-0.1")

    # rejected before doing anything: no journal entry, no revert
    assert not (candidate / "_rewrite" / "bouts.jsonl").exists()
    assert (candidate / "train.py").read_bytes() == b"v1\n"


# --- loop-facing CLI subcommands -----------------------------------------------


def _run_cli(monkeypatch, capsys, argv: list[str]):
    monkeypatch.setattr(sys, "argv", ["rewrite_bout.py"] + argv)
    code = rewrite_bout.main()
    return code, capsys.readouterr().out


def test_snapshot_and_revert_cli(tmp_path, monkeypatch, capsys) -> None:
    candidate = _make_candidate(tmp_path)
    original = b"BASE_PARAMS = {'x': 1}\n"
    (candidate / "train.py").write_bytes(original)

    code, out = _run_cli(monkeypatch, capsys, [
        "snapshot", "--candidate", str(candidate), "--bout", "2"])
    assert code == 0
    snap = Path(out.strip())
    assert snap == candidate / "_rewrite" / "bout-002.pre.py"
    assert snap.read_bytes() == original

    (candidate / "train.py").write_bytes(b"edited\n")
    code, _ = _run_cli(monkeypatch, capsys, [
        "revert", "--candidate", str(candidate), "--snapshot", str(snap)])
    assert code == 0
    assert (candidate / "train.py").read_bytes() == original


def test_current_best_cli(tmp_path, monkeypatch, capsys) -> None:
    candidate = _make_candidate(tmp_path, baseline=1.0)
    rewrite_bout.append_bout(candidate, {"bout": 1, "outcome": "kept", "score": 0.8})
    code, out = _run_cli(monkeypatch, capsys, [
        "current-best", "--candidate", str(candidate)])
    assert code == 0
    assert float(out.strip()) == 0.8


def test_changed_cli(tmp_path, monkeypatch, capsys) -> None:
    candidate = _make_candidate(tmp_path)
    (candidate / "train.py").write_bytes(b"v1\n")
    snap = rewrite_bout.snapshot(candidate, 1)

    code, _ = _run_cli(monkeypatch, capsys, [
        "changed", "--candidate", str(candidate), "--snapshot", str(snap)])
    assert code == 0

    (candidate / "train.py").write_bytes(b"v2\n")
    code, _ = _run_cli(monkeypatch, capsys, [
        "changed", "--candidate", str(candidate), "--snapshot", str(snap)])
    assert code == 1

    # a missing snapshot is exit 1 with a message, never a crash trace
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, capsys, [
            "changed", "--candidate", str(candidate),
            "--snapshot", str(candidate / "_rewrite" / "missing.py")])


def test_changed_cli_missing_train_py(tmp_path, monkeypatch, capsys) -> None:
    candidate = _make_candidate(tmp_path)
    (candidate / "train.py").write_bytes(b"v1\n")
    snap = rewrite_bout.snapshot(candidate, 1)
    (candidate / "train.py").unlink()
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, capsys, [
            "changed", "--candidate", str(candidate), "--snapshot", str(snap)])


def test_current_best_cli_fails_without_reference(tmp_path, monkeypatch, capsys) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "_import.json").write_text(json.dumps({"baseline_score": float("inf")}))
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, capsys, ["current-best", "--candidate", str(candidate)])


def test_journal_cli_appends_non_scored_entry(tmp_path, monkeypatch, capsys) -> None:
    candidate = _make_candidate(tmp_path)
    code, out = _run_cli(monkeypatch, capsys, [
        "journal", "--candidate", str(candidate), "--bout", "1",
        "--outcome", "noop", "--summary", "no change", "--basis", "none"])
    assert code == 0
    entry = json.loads(out)
    assert entry == {
        "bout": 1,
        "attempt_id": None,
        "score": None,
        "outcome": "noop",
        "summary": "no change",
        "basis": "none",
        "snapshot": "bout-001.pre.py",
    }
    assert rewrite_bout.load_bouts(candidate) == [entry]
    assert rewrite_bout.consecutive_non_kept([entry]) == 1
