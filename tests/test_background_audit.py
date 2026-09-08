"""Offline coverage for driver/loops/background_audit.py (no LLM; stubbed judge)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from driver.loops import background_audit  # noqa: E402
from search_backends import canonical_key  # noqa: E402
from tests.fixtures import (  # noqa: E402
    TOY_SOURCE,
    background_text,
    fixture_registry,
    retrieval_hit_manifest,
)


class _Blocked(Exception):
    pass


class _Events:
    def __init__(self) -> None:
        self.calls = []

    def emit(self, name, **fields) -> None:
        self.calls.append((name, fields))


def _or_block(reason) -> None:
    raise _Blocked(reason)


def _run_dir(parent: Path, *, visit_content: str | None = None) -> Path:
    run_dir = parent / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "background.md").write_text(background_text(fixture_registry()))
    manifest = retrieval_hit_manifest()
    if visit_content is not None:
        (run_dir / "retrieval").mkdir()
        (run_dir / "retrieval" / "000-toy.txt").write_text(visit_content)
        manifest["visits"].append(
            {
                "url": TOY_SOURCE["url"],
                "canonical_key": canonical_key(TOY_SOURCE["url"]),
                "view": "full_text",
                "status": "success",
                "content_file": "retrieval/000-toy.txt",
                "content_chars": len(visit_content),
            }
        )
    (run_dir / "background_retrieval.json").write_text(json.dumps(manifest))
    return run_dir


def _receipt() -> dict:
    count = len(background_audit.collect_claim_mappings(fixture_registry()))
    return {
        "verdicts": [
            {"mapping": f"M{i}", "verdict": "faithful", "rationale": f"M{i} matches"}
            for i in range(1, count + 1)
        ]
    }


def test_verdict_errors_legitimacy() -> None:
    labels = ["M1", "M2"]
    good = [
        {"mapping": "M1", "verdict": "faithful", "rationale": "matches"},
        {"mapping": "M2", "verdict": "unverifiable", "rationale": "no receipt"},
    ]
    assert background_audit.verdict_errors({"verdicts": good}, labels) == []
    assert background_audit.verdict_errors({"verdicts": "M1"}, labels)
    for verdicts, needle in (
        ([{"mapping": "M9", "verdict": "faithful", "rationale": "x"}], "not a presented"),
        ([good[0], good[0]], "duplicates"),
        ([{"mapping": "M1", "verdict": "ok", "rationale": "x"}], "must be one of"),
        ([{"mapping": "M1", "verdict": "faithful", "rationale": " "}], "non-empty"),
        ([good[0]], "missing mappings"),
    ):
        errors = background_audit.verdict_errors({"verdicts": verdicts}, labels)
        assert any(needle in error for error in errors), (verdicts, errors)


def test_audit_completed_fails_closed(tmp_path) -> None:
    artifact = tmp_path / background_audit.ARTIFACT_NAME
    assert not background_audit.audit_completed(tmp_path)  # missing artifact
    artifact.write_text("not json")
    assert not background_audit.audit_completed(tmp_path)  # unreadable
    for outcome, expected in (
        ("passed", True),
        ("no_mappings", True),
        ("unfaithful", False),
        ("judge_failed", False),
    ):
        artifact.write_text(json.dumps({"rounds": [{"outcome": outcome}]}))
        assert background_audit.audit_completed(tmp_path) is expected


def test_sample_mappings_bounds_and_injectable_rng() -> None:
    mappings = [{"item_id": f"item-{index}"} for index in range(10)]
    sample = background_audit.sample_mappings(mappings, rng=random.Random(1), limit=5)
    assert len(sample) == 5
    assert len({entry["item_id"] for entry in sample}) == 5
    assert sample == background_audit.sample_mappings(
        mappings, rng=random.Random(1), limit=5
    )
    assert len(background_audit.sample_mappings(mappings, rng=random.Random(1), limit=99)) == 10


def test_build_judge_payload_excerpts_and_caps(tmp_path) -> None:
    registry = fixture_registry()
    sampled = background_audit.sample_mappings(
        background_audit.collect_claim_mappings(registry), rng=random.Random(2)
    )
    run_dir = _run_dir(
        tmp_path / "visited", visit_content="V" * (background_audit.EXCERPT_CHARS + 500)
    )
    manifest = json.loads((run_dir / "background_retrieval.json").read_text())

    payload, entries = background_audit.build_judge_payload(
        registry, manifest, run_dir, sampled
    )
    assert all(entry["verification"] == "full_text" for entry in entries)
    assert all(
        entry["excerpt_kind"] == "retained full_text visit content" for entry in entries
    )
    assert all(entry["excerpt_chars"] == background_audit.EXCERPT_CHARS for entry in entries)
    assert payload.count(", truncated") == len(entries)

    # the excerpt budget is shared: a sample beyond it renders empty
    _, tail = background_audit.build_judge_payload(
        registry, manifest, run_dir, sampled + [sampled[0]]
    )
    assert tail[-1]["excerpt_chars"] == 0

    # without a substantive visit the snippet receipt is the excerpt
    snippet_run = _run_dir(tmp_path / "snippet")
    snippet_manifest = json.loads((snippet_run / "background_retrieval.json").read_text())
    snippet_payload, snippet_entries = background_audit.build_judge_payload(
        registry, snippet_manifest, snippet_run, sampled[:1]
    )
    assert snippet_entries[0]["verification"] == "snippet_only"
    assert snippet_entries[0]["excerpt_text"] == "A search-hit snippet receipt."
    assert "A search-hit snippet receipt." in snippet_payload


def test_gate_faithful_passes_and_writes_artifact(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    events = _Events()

    def invoke(runner, store, role, task, tag, gate_run_dir, inline_payload):
        assert role == background_audit.JUDGE_ROLE and "## M1" in inline_payload
        return _receipt(), 7

    background_audit.run_faithfulness_gate(
        None, None, "toy", "tag", run_dir, events,
        invoke=invoke, or_block=_or_block, rng=random.Random(0),
    )

    artifact = json.loads((run_dir / background_audit.ARTIFACT_NAME).read_text())
    assert artifact["rounds"][-1]["outcome"] == "passed"
    assert artifact["rounds"][-1]["invocation_id"] == 7
    assert background_audit.audit_completed(run_dir)
    assert events.calls[-1][1]["outcome"] == "passed"


def test_gate_unfaithful_on_preseeded_blocks(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    events = _Events()

    def invoke(runner, store, role, task, tag, gate_run_dir, inline_payload):
        receipt = _receipt()
        receipt["verdicts"][0].update(verdict="unfaithful", rationale="not in source")
        return receipt, 3

    with pytest.raises(_Blocked):
        background_audit.run_faithfulness_gate(
            None, None, "toy", "tag", run_dir, events,
            invoke=invoke, or_block=_or_block, repair=None, rng=random.Random(0),
        )

    artifact = json.loads((run_dir / background_audit.ARTIFACT_NAME).read_text())
    assert artifact["rounds"][-1]["outcome"] == "unfaithful"
    assert artifact["rounds"][-1]["unfaithful"] == ["M1"]
    assert not background_audit.audit_completed(run_dir)  # fail closed
