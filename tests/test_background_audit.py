"""Offline coverage for driver/loops/background_audit.py (no LLM; stubbed judge)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from driver.loops import background_audit  # noqa: E402
from driver.session import InvocationFailed  # noqa: E402
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


def _run_dir(
    parent: Path, *, visit_content: str | None = None, registry: dict | None = None
) -> Path:
    run_dir = parent / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "background.md").write_text(
        background_text(registry if registry is not None else fixture_registry())
    )
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


def _verdicts_for(payload: str, script: dict[str, str]) -> dict:
    """Receipt verdicts covering every presented label, per item-id → verdict."""
    verdicts = []
    label = None
    for line in payload.splitlines():
        match = re.fullmatch(r"## (M\d+)", line)
        if match:
            label = match.group(1)
        elif line.startswith("- item: ") and label is not None:
            ids = re.findall(r"`([^`]+)`", line)
            verdict = next((script[item] for item in ids if item in script), "faithful")
            verdicts.append(
                {"mapping": label, "verdict": verdict, "rationale": f"{label} judged"}
            )
            label = None
    return {"verdicts": verdicts}


def _invoke_stub(scripts: list[dict[str, str]], payloads: list[str] | None = None):
    """Stubbed judge invocation; call N judges per ``scripts[N - 1]``."""
    state = {"calls": 0}

    def invoke(runner, store, role, task, tag, run_dir, inline_payload):
        assert role == background_audit.JUDGE_ROLE
        if payloads is not None:
            payloads.append(inline_payload)
        script = scripts[min(state["calls"], len(scripts) - 1)]
        state["calls"] += 1
        return _verdicts_for(inline_payload, script), 7

    return invoke


def _rewrite_claim(run_dir: Path, item_id: str, claim: str) -> None:
    registry = fixture_registry()
    for dimension in registry["dimensions"]:
        for hypothesis in dimension["hypotheses"]:
            if hypothesis["id"] == item_id:
                hypothesis["claim"] = claim
    (run_dir / "background.md").write_text(background_text(registry))


def _finding(item_id: str, rationale: str) -> dict:
    """One unfaithful finding in the shape the gate hands to _format_findings."""
    return {
        "entry": {
            "item_kind": "hypothesis",
            "item_ids": [item_id],
            "audit_text": f"claim: text for {item_id}",
            "link_role": "supports",
            "source_id": "src-01",
            "source_url": TOY_SOURCE["url"],
        },
        "rationale": rationale,
    }


def _payload_items(text: str) -> list[dict]:
    """The JSON finding entries of a repair payload (fails if one is cut)."""
    body = text.split("\n\nTruncated:", 1)[0]
    return json.loads(body[body.index("["):])


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
    version = background_audit.ARTIFACT_VERSION
    for doc, expected in (
        ({"rounds": [{"outcome": "passed"}]}, False),  # no version
        ({"version": version - 1, "rounds": [{"outcome": "passed"}]}, False),
        ({"version": version, "rounds": [{"outcome": "passed"}]}, True),
        ({"version": version, "rounds": [{"outcome": "no_mappings"}]}, True),
        (
            {"version": version, "rounds": [{"outcome": "unfaithful_irreparable_warning"}]},
            True,
        ),
        ({"version": version, "rounds": [{"outcome": "unfaithful"}]}, False),
        ({"version": version, "rounds": [{"outcome": "judge_failed"}]}, False),
    ):
        artifact.write_text(json.dumps(doc))
        assert background_audit.audit_completed(tmp_path) is expected


def test_collect_claim_mappings_dedupes_by_identity() -> None:
    item = {
        "id": "hyp-1",
        "title": "one",
        "claim": "Claim text.",
        "scope": {"metrics": ["m"]},
        "credibility_rationale": "rationale",
        "reopen_when": "when",
        "evidence": [
            {"source_id": "src-01", "role": "supports"},
            {"source_id": "src-01", "role": "supports"},  # duplicate link
        ],
    }
    twin = {
        **item,
        "id": "hyp-2",
        "evidence": [{"source_id": "src-01", "role": "supports"}],
    }
    other_role = {
        **item,
        "id": "hyp-3",
        "evidence": [{"source_id": "src-01", "role": "context"}],
    }
    other_text = {
        **item,
        "id": "hyp-4",
        "claim": "A different claim.",
        "evidence": [{"source_id": "src-01", "role": "supports"}],
    }
    registry = {
        "dimensions": [{"hypotheses": [item, twin, other_role, other_text]}],
        "guidance": [],
        "sources": [dict(TOY_SOURCE)],
    }

    mappings = background_audit.collect_claim_mappings(registry)
    assert len(mappings) == 3
    supports = next(m for m in mappings if m["link_role"] == "supports"
                    and "Claim text." in m["audit_text"])
    assert supports["item_ids"] == ["hyp-1", "hyp-2"]
    assert supports["source_url"] == TOY_SOURCE["url"]
    assert "claim: Claim text." in supports["audit_text"]
    assert "scope: metrics: m" in supports["audit_text"]
    assert "credibility_rationale: rationale" in supports["audit_text"]
    assert "reopen_when: when" in supports["audit_text"]
    assert {m["link_role"] for m in mappings} == {"supports", "context"}

    # the fixture registry carries five distinct identities
    assert len(background_audit.collect_claim_mappings(fixture_registry())) == 5


def _bare_entry(source_url: str, text: str) -> dict:
    return {
        "item_kind": "hypothesis",
        "item_ids": ["hyp-x"],
        "title": None,
        "audit_text": text,
        "link_role": "supports",
        "source_id": "src",
        "source_title": None,
        "source_url": source_url,
        "verification": "snippet_only",
        "excerpt": {
            "kind": "search-result snippet",
            "chars": len(text),
            "sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            "truncated": False,
            "offset": 0,
            "content_file": None,
            "view": None,
            "section": None,
            "text": text,
        },
        "number_presence": {"item": "none", "this_source": "none"},
        "missing_tokens": {"item": [], "this_source": []},
        "coverage": {
            "tier": "snippet_only",
            "routing": "partial",
            "retained_chars": None,
            "store_cap_hit": False,
            "content_file": None,
            "view": None,
            "section": None,
        },
        "_excerpt_text": text,
    }


def test_pack_batches_budget_cap_and_adjacency() -> None:
    entries = [
        _bare_entry("https://a.test/", f"text-{index}")
        for index in range(3)
    ] + [
        _bare_entry("https://b.test/", f"text-{index}")
        for index in range(3)
    ]
    batches = background_audit.pack_batches(entries, max_mappings=2)
    assert [[entry["source_url"] for entry in batch] for batch in batches] == [
        ["https://a.test/", "https://a.test/"],
        ["https://a.test/", "https://b.test/"],
        ["https://b.test/", "https://b.test/"],
    ]

    # budget overflow starts a new batch; an oversized entry packs alone
    big = _bare_entry("https://c.test/", "x" * 100)
    block = len(background_audit._entry_block(big, "M1"))
    packed = background_audit.pack_batches([big, big, big], budget=block + 10)
    assert [len(batch) for batch in packed] == [1, 1, 1]


def test_build_audit_entries_window_presence_coverage(tmp_path) -> None:
    content = "The study reports a 98.5% gain. " + "V" * background_audit.EXCERPT_CHARS
    registry = fixture_registry()
    registry["guidance"][0]["credibility_rationale"] = "Reported gain is 98.5% here."
    run_dir = _run_dir(tmp_path, visit_content=content, registry=registry)
    manifest = json.loads((run_dir / "background_retrieval.json").read_text())

    entries = background_audit.build_audit_entries(
        registry, manifest, run_dir, background_audit.collect_claim_mappings(registry)
    )
    assert len(entries) == 5
    for entry in entries:
        assert entry["verification"] == "full_text"
        window = entry["excerpt"]
        assert window["kind"] == "retained full_text visit content"
        assert window["chars"] == background_audit.EXCERPT_CHARS
        assert window["truncated"] is True
        assert window["view"] == "full_text"
        assert window["content_file"] == "retrieval/000-toy.txt"
        assert window["sha256"] == (
            "sha256:" + hashlib.sha256(entry["_excerpt_text"].encode()).hexdigest()
        )
        assert entry["coverage"]["routing"] == "sufficient"  # full_text, no cap hit
    guidance = next(e for e in entries if e["item_kind"] == "guidance")
    assert guidance["number_presence"] == {"item": "present", "this_source": "present"}
    hypotheses = [e for e in entries if e["item_kind"] == "hypothesis"]
    assert all(e["number_presence"]["item"] == "none" for e in hypotheses)


def test_build_audit_entries_snippet_fallback_and_abstract(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    manifest = json.loads((run_dir / "background_retrieval.json").read_text())
    registry = fixture_registry()

    # without a substantive visit the snippet receipt is the excerpt
    entries = background_audit.build_audit_entries(
        registry, manifest, run_dir, background_audit.collect_claim_mappings(registry)
    )
    assert entries[0]["verification"] == "snippet_only"
    assert entries[0]["excerpt"]["kind"] == "search-result snippet"
    assert entries[0]["excerpt"]["text"] == "A search-hit snippet receipt."
    assert entries[0]["coverage"]["routing"] == "partial"

    # an abstract-only visit counts as audit content (preview tier)
    key = canonical_key(TOY_SOURCE["url"])
    (run_dir / "retrieval").mkdir()
    (run_dir / "retrieval" / "001-abs.txt").write_text("Abstract of the toy study.")
    manifest["visits"].append(
        {
            "url": TOY_SOURCE["url"],
            "canonical_key": key,
            "view": "abstract",
            "status": "success",
            "content_file": "retrieval/001-abs.txt",
            "content_chars": 26,
        }
    )
    entries = background_audit.build_audit_entries(
        registry, manifest, run_dir, background_audit.collect_claim_mappings(registry)
    )
    assert entries[0]["verification"] == "preview"
    assert entries[0]["excerpt"]["kind"] == "retained abstract visit content"
    assert entries[0]["excerpt"]["view"] == "abstract"


def test_window_located_by_audit_text_skips_nav_header(tmp_path) -> None:
    nav = "Home | Articles | Courses | Newsletter | Sign in\nMenu Search Topics Tags"
    body = (
        "We study quality-filtered examples as one attributed semantic choice: "
        "a directly inspected primary source supports a local test."
    )
    content = nav + "\n\n## Results\n\n" + body
    run_dir = _run_dir(tmp_path, visit_content=content)
    manifest = json.loads((run_dir / "background_retrieval.json").read_text())
    registry = fixture_registry()

    entries = background_audit.build_audit_entries(
        registry, manifest, run_dir,
        background_audit.collect_claim_mappings(registry),
    )
    assert len(entries) == 5
    for entry in entries:
        window = entry["excerpt"]
        assert window["offset"] == len(content) - len(body)
        assert entry["_excerpt_text"] == body  # the body block, not the nav head
        assert "Newsletter" not in entry["_excerpt_text"]
        assert window["section"] == "Results"  # heading above the window (F9.2)
        assert window["truncated"] is True  # the nav head before the window is omitted


def test_window_no_hit_fallback_skips_boilerplate(tmp_path) -> None:
    wall = (
        "We use cookies to improve your experience. Accept all cookies. "
        "Enable JavaScript to continue browsing."
    )
    body = (
        "The quarterly gardening almanac lists tomato planting dates for the "
        "northern hemisphere and companion planting charts for raised beds."
    )
    content = wall + "\n\n" + body
    run_dir = _run_dir(tmp_path, visit_content=content)
    manifest = json.loads((run_dir / "background_retrieval.json").read_text())
    registry = fixture_registry()

    entries = background_audit.build_audit_entries(
        registry, manifest, run_dir,
        background_audit.collect_claim_mappings(registry),
    )
    assert len(entries) == 5
    for entry in entries:
        window = entry["excerpt"]
        # nothing matches the audit text: the longest non-boilerplate block
        # wins, never the bare boilerplate head of the visit
        assert window["offset"] == len(wall) + 2
        assert entry["_excerpt_text"] == body
        assert "cookies" not in entry["_excerpt_text"]


def test_window_anchors_at_hit_inside_oversized_block(tmp_path) -> None:
    # direct-fallback retrievals collapse all whitespace: one giant block
    head = "navigation banner " * 400
    match = (
        "a directly inspected primary source supports a local test of "
        "quality-filtered examples"
    )
    content = head + match + " " + "filler prose " * 400  # no blank lines
    assert "\n" not in content and len(content) > background_audit.EXCERPT_CHARS
    run_dir = _run_dir(tmp_path, visit_content=content)
    manifest = json.loads((run_dir / "background_retrieval.json").read_text())
    registry = fixture_registry()

    entries = background_audit.build_audit_entries(
        registry, manifest, run_dir,
        background_audit.collect_claim_mappings(registry),
    )
    assert len(entries) == 5
    excerpt_chars = background_audit.EXCERPT_CHARS
    for entry in entries:
        window = entry["excerpt"]
        # the window is anchored at the match deep in the block, not the head
        assert 0 < window["offset"] <= content.index(match)
        assert match in entry["_excerpt_text"]
        assert entry["_excerpt_text"] == (
            content[window["offset"] : window["offset"] + excerpt_chars]
        )


def test_window_fallback_anchors_mid_oversized_block(tmp_path) -> None:
    content = "gardening almanac " * 600  # one giant block, zero audit-text hits
    assert len(content) > background_audit.EXCERPT_CHARS
    run_dir = _run_dir(tmp_path, visit_content=content)
    manifest = json.loads((run_dir / "background_retrieval.json").read_text())
    registry = fixture_registry()

    entries = background_audit.build_audit_entries(
        registry, manifest, run_dir,
        background_audit.collect_claim_mappings(registry),
    )
    assert len(entries) == 5
    excerpt_chars = background_audit.EXCERPT_CHARS
    expected = (len(content) - excerpt_chars) // 2
    for entry in entries:
        window = entry["excerpt"]
        assert window["offset"] == expected  # mid-block, not the block head
        assert entry["_excerpt_text"] == content[expected : expected + excerpt_chars]


def test_build_judge_payload_renders_identity_and_facts(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    manifest = json.loads((run_dir / "background_retrieval.json").read_text())
    registry = fixture_registry()
    entries = background_audit.build_audit_entries(
        registry, manifest, run_dir, background_audit.collect_claim_mappings(registry)
    )

    payload = background_audit.build_judge_payload(entries[:2])
    assert payload.startswith("Mappings to audit: 2")
    assert "located by matching the mapping's audit text" in payload
    assert [entry["label"] for entry in entries[:2]] == ["M1", "M2"]
    assert "- audit text:" in payload and "claim: " in payload
    assert "- number presence: item=none this_source=none" in payload
    assert "- coverage: routing=partial tier=snippet_only" in payload
    assert "search-result snippet" in payload
    assert "A search-hit snippet receipt." in payload
    assert f"<{TOY_SOURCE['url']}>" in payload


def test_format_findings_under_cap_keeps_every_entry() -> None:
    findings = [_finding(f"hyp-{index}", f"reason {index}") for index in range(3)]
    text = background_audit._format_findings(findings)
    assert "Truncated:" not in text
    assert len(text) <= background_audit.FINDINGS_CHARS
    items = _payload_items(text)
    assert [item["item"] for item in items] == [
        "hypothesis hyp-0",
        "hypothesis hyp-1",
        "hypothesis hyp-2",
    ]
    assert items[0]["judge_rationale"] == "reason 0"


def test_format_findings_truncates_at_entry_boundary_and_declares() -> None:
    findings = [_finding(f"hyp-{index}", "r" * 200) for index in range(5)]
    budget = len(background_audit._format_findings(findings[:2]))
    text = background_audit._format_findings(findings, budget=budget)
    items = _payload_items(text)  # parses as JSON: no entry is cut in half
    assert [item["item"] for item in items] == ["hypothesis hyp-0", "hypothesis hyp-1"]
    assert items[-1]["judge_rationale"] == "r" * 200  # the last entry is whole
    assert "3 more finding(s) not shown" in text
    assert background_audit.ARTIFACT_NAME in text


def test_gate_faithful_passes_and_writes_artifact(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    events = _Events()
    payloads: list[str] = []

    background_audit.run_faithfulness_gate(
        None, None, "toy", "tag", run_dir, events,
        invoke=_invoke_stub([{}], payloads), or_block=_or_block,
    )

    artifact = json.loads((run_dir / background_audit.ARTIFACT_NAME).read_text())
    assert artifact["version"] == background_audit.ARTIFACT_VERSION
    round_doc = artifact["rounds"][-1]
    assert round_doc["outcome"] == "passed"
    assert round_doc["batches"][0]["invocation_id"] == 7
    assert round_doc["batches"][0]["source_keys"] == [canonical_key(TOY_SOURCE["url"])]
    entries = round_doc["batches"][0]["entries"]
    assert {entry["label"] for entry in entries} == {"M1", "M2", "M3", "M4", "M5"}
    assert all("_excerpt_text" not in entry for entry in entries)
    assert all(entry["excerpt"]["sha256"].startswith("sha256:") for entry in entries)
    assert background_audit.audit_completed(run_dir)
    event = events.calls[-1][1]
    assert event["outcome"] == "passed"
    assert event["batches"] == 1 and event["mappings"] == 5 and event["faithful"] == 5
    assert "## M1" in payloads[0]


def test_gate_no_mappings_is_terminal(tmp_path) -> None:
    registry = {
        "schema_version": 3,
        "kind": "semantic_search_space",
        "space_id": "empty-space",
        "dimensions": [],
        "relations": [],
        "guidance": [],
        "sources": [],
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "background.md").write_text(
        "# bg\n\n## Search space registry\n```json\n"
        + json.dumps(registry)
        + "\n```\n"
    )
    (run_dir / "background_retrieval.json").write_text(
        json.dumps({"schema_version": 4, "rounds": [], "visits": []})
    )
    events = _Events()

    background_audit.run_faithfulness_gate(
        None, None, "toy", "tag", run_dir, events,
        invoke=_invoke_stub([{}]), or_block=_or_block,
    )

    artifact = json.loads((run_dir / background_audit.ARTIFACT_NAME).read_text())
    assert artifact["version"] == background_audit.ARTIFACT_VERSION
    assert artifact["rounds"][-1]["outcome"] == "no_mappings"
    assert background_audit.audit_completed(run_dir)
    assert events.calls[-1][1]["outcome"] == "no_mappings"


def test_gate_preseeded_unfaithful_records_warning(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    events = _Events()

    background_audit.run_faithfulness_gate(
        None, None, "toy", "tag", run_dir, events,
        invoke=_invoke_stub([{"hyp-data-filtered": "unfaithful"}]),
        or_block=_or_block, repair=None,
    )

    artifact = json.loads((run_dir / background_audit.ARTIFACT_NAME).read_text())
    round_doc = artifact["rounds"][-1]
    assert round_doc["outcome"] == "unfaithful_irreparable_warning"
    assert round_doc["unfaithful"][0]["item_ids"] == ["hyp-data-filtered"]
    assert background_audit.audit_completed(run_dir)  # terminal-ok, no re-audit loop
    event = events.calls[-1][1]
    assert event["outcome"] == "unfaithful_irreparable_warning"
    assert event["unfaithful"] == 1
    assert event["findings"][0]["item_ids"] == ["hyp-data-filtered"]


def test_gate_attempt2_rechecks_only_repeat_and_changed(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    events = _Events()
    payloads: list[str] = []

    def repair(findings_text: str) -> None:
        assert "hyp-data-filtered" in findings_text
        _rewrite_claim(run_dir, "hyp-valid-cv", "Rewritten claim after repair.")

    background_audit.run_faithfulness_gate(
        None, None, "toy", "tag", run_dir, events,
        invoke=_invoke_stub(
            [
                {"hyp-data-filtered": "unfaithful", "hyp-model-multibranch": "unverifiable"},
                {},
            ],
            payloads,
        ),
        or_block=_or_block,
        repair=repair,
    )

    assert len(payloads) == 2
    recheck = payloads[1]
    assert "hyp-data-filtered" in recheck  # repeat unfaithful, still present
    assert "hyp-valid-cv" in recheck  # audit text changed by the repair
    assert "hyp-model-multibranch" not in recheck  # unverifiable is not re-judged
    assert "hyp-ensemble-stacking" not in recheck  # unchanged faithful stays out
    artifact = json.loads((run_dir / background_audit.ARTIFACT_NAME).read_text())
    assert [r["outcome"] for r in artifact["rounds"]] == ["unfaithful", "passed"]
    round_one = artifact["rounds"][0]
    assert round_one["unfaithful"][0]["item_ids"] == ["hyp-data-filtered"]
    assert round_one["unverifiable"][0]["item_ids"] == ["hyp-model-multibranch"]
    assert background_audit.audit_completed(run_dir)


def test_gate_truncated_repair_payload_leaves_complete_artifact(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    events = _Events()
    captured: list[str] = []
    state = {"calls": 0}

    def invoke(runner, store, role, task, tag, gate_run_dir, inline_payload):
        state["calls"] += 1
        receipt = _verdicts_for(inline_payload, {})
        if state["calls"] == 1:
            for verdict in receipt["verdicts"]:  # five ~6KB findings > the cap
                verdict["verdict"] = "unfaithful"
                verdict["rationale"] = f"{verdict['mapping']} " + "r" * 6000
        return receipt, 7

    def repair(findings_text: str) -> None:
        captured.append(findings_text)

    background_audit.run_faithfulness_gate(
        None, None, "toy", "tag", run_dir, events,
        invoke=invoke, or_block=_or_block, repair=repair,
    )

    payload = captured[0]
    shown = _payload_items(payload)
    assert 0 < len(shown) < 5  # the ~20KB cap cannot hold all five
    assert all(item["judge_rationale"].endswith("r" * 100) for item in shown)
    assert f"{5 - len(shown)} more finding(s) not shown" in payload
    assert background_audit.ARTIFACT_NAME in payload
    artifact = json.loads((run_dir / background_audit.ARTIFACT_NAME).read_text())
    assert [r["outcome"] for r in artifact["rounds"]] == ["unfaithful", "passed"]
    assert len(artifact["rounds"][0]["unfaithful"]) == 5  # always complete


def test_gate_attempt2_recheck_empty_passes(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    events = _Events()
    payloads: list[str] = []

    def repair(findings_text: str) -> None:
        registry = fixture_registry()
        for dimension in registry["dimensions"]:
            dimension["hypotheses"] = [
                h for h in dimension["hypotheses"] if h["id"] != "hyp-data-filtered"
            ]
        (run_dir / "background.md").write_text(background_text(registry))

    background_audit.run_faithfulness_gate(
        None, None, "toy", "tag", run_dir, events,
        invoke=_invoke_stub([{"hyp-data-filtered": "unfaithful"}], payloads),
        or_block=_or_block,
        repair=repair,
    )

    assert len(payloads) == 1  # no judge call on the empty recheck
    artifact = json.loads((run_dir / background_audit.ARTIFACT_NAME).read_text())
    assert [r["outcome"] for r in artifact["rounds"]] == ["unfaithful", "passed"]
    assert artifact["rounds"][1]["recheck_empty"] is True
    assert artifact["rounds"][1]["batches"] == []
    assert background_audit.audit_completed(run_dir)
    assert events.calls[-1][1]["recheck_empty"] is True


def test_gate_attempt2_unfaithful_blocks_with_identity_message(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    events = _Events()

    def repair(findings_text: str) -> None:
        _rewrite_claim(run_dir, "hyp-valid-cv", "Rewritten claim after repair.")

    with pytest.raises(_Blocked) as blocked:
        background_audit.run_faithfulness_gate(
            None, None, "toy", "tag", run_dir, events,
            invoke=_invoke_stub(
                [
                    {"hyp-data-filtered": "unfaithful"},
                    {"hyp-data-filtered": "unfaithful", "hyp-valid-cv": "unfaithful"},
                ]
            ),
            or_block=_or_block,
            repair=repair,
        )

    message = str(blocked.value)
    assert "[repeat]" in message and "hyp-data-filtered" in message
    assert "[new]" in message and "hyp-valid-cv" in message
    assert TOY_SOURCE["url"] in message and "role: supports" in message
    artifact = json.loads((run_dir / background_audit.ARTIFACT_NAME).read_text())
    assert artifact["rounds"][-1]["outcome"] == "unfaithful"
    assert not background_audit.audit_completed(run_dir)  # fail closed


def test_gate_judge_retries_twice_then_passes(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    events = _Events()
    state = {"calls": 0}

    def invoke(runner, store, role, task, tag, gate_run_dir, inline_payload):
        state["calls"] += 1
        if state["calls"] < 3:
            raise InvocationFailed(role, [f"transient failure {state['calls']}"])
        return _verdicts_for(inline_payload, {}), 7

    background_audit.run_faithfulness_gate(
        None, None, "toy", "tag", run_dir, events,
        invoke=invoke, or_block=_or_block,
    )

    assert state["calls"] == 3  # initial call plus two fresh-session retries
    artifact = json.loads((run_dir / background_audit.ARTIFACT_NAME).read_text())
    assert artifact["rounds"][-1]["outcome"] == "passed"
    assert background_audit.audit_completed(run_dir)


def test_gate_judge_failed_block_names_batch_sources(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    events = _Events()
    state = {"calls": 0}

    def invoke(runner, store, role, task, tag, gate_run_dir, inline_payload):
        state["calls"] += 1
        raise InvocationFailed(role, ["judge produced no receipt"])

    with pytest.raises(_Blocked) as blocked:
        background_audit.run_faithfulness_gate(
            None, None, "toy", "tag", run_dir, events,
            invoke=invoke, or_block=_or_block,
            repair=lambda findings: None,
        )

    assert state["calls"] == 3  # retries exhausted per batch before blocking
    message = str(blocked.value)
    assert "batch 1/1" in message
    assert canonical_key(TOY_SOURCE["url"]) in message
    artifact = json.loads((run_dir / background_audit.ARTIFACT_NAME).read_text())
    assert artifact["rounds"][-1]["outcome"] == "judge_failed"
    assert artifact["rounds"][-1]["batches"][-1]["error"] == ["judge produced no receipt"]
    assert not background_audit.audit_completed(run_dir)  # fail closed


def test_gate_judge_unavailable_preseeded_records_and_proceeds(tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    events = _Events()
    state = {"calls": 0}

    def invoke(runner, store, role, task, tag, gate_run_dir, inline_payload):
        state["calls"] += 1
        raise InvocationFailed(role, ["judge produced no receipt"])

    # A pre-seeded frozen background (repair=None): an unavailable judge is
    # recorded as audit_unavailable and the run proceeds; a later resume
    # retries the audit (not terminal-ok).
    background_audit.run_faithfulness_gate(
        None, None, "toy", "tag", run_dir, events,
        invoke=invoke, or_block=_or_block,
    )

    assert state["calls"] == 3
    artifact = json.loads((run_dir / background_audit.ARTIFACT_NAME).read_text())
    assert artifact["rounds"][-1]["outcome"] == "audit_unavailable"
    assert artifact["rounds"][-1]["batches"][-1]["error"] == [
        "judge produced no receipt"]
    assert not background_audit.audit_completed(run_dir)
    assert events.calls[-1][0] == "background_faithfulness_audit"
    assert events.calls[-1][1]["outcome"] == "audit_unavailable"
