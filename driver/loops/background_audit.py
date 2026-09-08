"""Synchronous faithfulness audit for the background phase (design §4).

Deterministic contract validators own receipts ("what was actually
retrieved"); this module is the non-deterministic backstop that runs after
they pass and before the semantic space freezes.  Every source→claim mapping
identity in the evidence registry is judged against the tool-recorded receipt
content by the tool-free ``background-faithfulness-judge`` role: mappings are
deduped by identity, packed into budget-bounded batches with one source's
mappings adjacent, and each batch gets its own judge invocation plus two
fresh-session retries.  Unfaithful findings on the generated path trigger
exactly one researcher repair round followed by a narrowed re-audit (repeat
unfaithful identities still present, plus identities whose audit text or
excerpt window changed); a pre-seeded frozen background cannot be rewritten,
so its unfaithful findings are recorded as a terminal warning instead of
blocking.  Every other failure mode blocks the run — the audit is the run's
only faithfulness gate, so it fails closed.

The pure pieces (mapping collection, entry construction, batching, payload
rendering, verdict validation) are separated from the gate so they can be
smoke-tested offline with stubbed role invocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..roles import REPO_ROOT
from ..session import InvocationFailed

# tools/ modules use script-style sibling imports; every in-process consumer
# (tests, tools/validate_*.py) puts the directory on sys.path first.
_TOOLS_DIR = str(REPO_ROOT / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import background_contract  # noqa: E402
from search_backends import (  # noqa: E402
    _SECTION_STOPWORDS,
    _VERIFICATION_RANK,
    _VIEW_VERIFICATION,
    _WORD_RE,
    WEB_BOILERPLATE_MARKERS,
    canonical_key,
    merged_results,
)

JUDGE_ROLE = "background-faithfulness-judge"
ARTIFACT_NAME = "background_faithfulness.json"
ARTIFACT_VERSION = 2

EXCERPT_CHARS = 4000  # per-mapping receipt excerpt cap
WINDOW_NUMBER_WEIGHT = 10  # window score of one claimed number found in a block
WINDOW_LEAD_IN_CHARS = 200  # context kept before the first hit in a giant block
BATCH_PAYLOAD_CHARS = 40960  # per-batch judge payload budget (~40KB)
BATCH_MAX_MAPPINGS = 12  # per-batch mapping cap
FINDINGS_CHARS = 20480  # ~20KB cap on the repair payload handed to the researcher

VERDICTS = ("faithful", "unfaithful", "unverifiable")
TERMINAL_OK_OUTCOMES = ("passed", "no_mappings", "unfaithful_irreparable_warning")
# Audit content collection ranks a visit's view by the verification tier it
# raises; the abstract view counts as preview-tier content.
_VIEW_RANK = {
    view: _VERIFICATION_RANK[tier] for view, tier in _VIEW_VERIFICATION.items()
}


# =============================================================================
# Pure pieces: mapping collection, entry construction, batching, payloads
# =============================================================================


def collect_claim_mappings(registry: dict) -> list[dict]:
    """Every source→claim evidence link, deduped by audit identity.

    The identity is ``(item_kind, audit text, source_url, link_role)``: the
    shared audit text (claim/scope/credibility_rationale/reopen_when, present
    fields only, field-name annotated) is what the judge audits, so links
    presenting the same text from the same source under the same role are one
    mapping carrying every originating item id — findings can then name each
    item to fix.
    """
    sources = {
        source.get("id"): source
        for source in registry.get("sources", [])
        if isinstance(source, dict)
    }
    mappings: list[dict] = []
    seen: dict[tuple, dict] = {}

    def collect(item_kind: str, item: dict) -> None:
        text = background_contract.audit_text(item)
        for link in item.get("evidence") or []:
            if not isinstance(link, dict):
                continue
            source = sources.get(link.get("source_id")) or {}
            url = source.get("url") or ""
            identity = (item_kind, text, url, link.get("role"))
            mapping = seen.get(identity)
            if mapping is not None:
                if item.get("id") not in mapping["item_ids"]:
                    mapping["item_ids"].append(item.get("id"))
                continue
            mapping = {
                "item_kind": item_kind,
                "item_ids": [item.get("id")],
                "title": item.get("title"),
                "audit_text": text,
                "link_role": link.get("role"),
                "source_id": link.get("source_id"),
                "source_url": url or None,
                "_item": item,
                "_link": link,
            }
            seen[identity] = mapping
            mappings.append(mapping)

    for dimension in registry.get("dimensions", []):
        if not isinstance(dimension, dict):
            continue
        for hypothesis in dimension.get("hypotheses", []):
            if isinstance(hypothesis, dict):
                collect("hypothesis", hypothesis)
    for item in registry.get("guidance", []):
        if isinstance(item, dict):
            collect("guidance", item)
    return mappings


def _identity_key(entry: dict) -> str:
    """Stable string key of one mapping identity, for attempt-2 diffing."""
    return json.dumps(
        [
            entry["item_kind"],
            entry["audit_text"],
            entry.get("source_url"),
            entry["link_role"],
        ]
    )


def _identity_record(entry: dict) -> dict:
    """The artifact's persisted form of one mapping identity."""
    return {
        "item_kind": entry["item_kind"],
        "item_ids": entry["item_ids"],
        "source_url": entry.get("source_url"),
        "link_role": entry["link_role"],
    }


def _key_visits(
    manifest: dict, manifest_dir: Path, key: str, cache: dict[str, list]
) -> list[tuple[int, dict, str]]:
    """(view rank, visit, retained content) per successful visit of one key.

    Read once per source per audit invocation and shared across that source's
    mappings; tier-ordered so window ties prefer the higher-tier visit.
    """
    if key in cache:
        return cache[key]
    visits: list[tuple[int, dict, str]] = []
    for visit in manifest.get("visits", []):
        if not isinstance(visit, dict) or visit.get("status") != "success":
            continue
        if visit.get("canonical_key") != key:
            continue
        rank = _VIEW_RANK.get(str(visit.get("view")))
        if rank is None:
            continue
        content_file = visit.get("content_file")
        if not isinstance(content_file, str) or not content_file:
            continue
        try:
            content = (manifest_dir / content_file).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        visits.append((rank, visit, content))
    visits.sort(key=lambda item: -item[0])
    cache[key] = visits
    return visits


_BLOCK_SPLIT_RE = re.compile(r"\n\s*\n")
_TOKEN_RE = re.compile(r"\S+")
_HEADING_RE = re.compile(r"^ {0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)
# Audit-text field annotations are labels, not claim content.
_ANNOTATION_WORDS = {
    word
    for field in background_contract.AUDIT_TEXT_FIELDS
    for word in field.split("_")
}


def _blocks(content: str) -> list[tuple[int, str]]:
    """(offset, text) per blank-line-delimited block, in document order."""
    blocks: list[tuple[int, str]] = []
    start = 0
    for match in _BLOCK_SPLIT_RE.finditer(content):
        block = content[start : match.start()]
        if block.strip():
            blocks.append((start, block))
        start = match.end()
    tail = content[start:]
    if tail.strip():
        blocks.append((start, tail))
    return blocks


def _is_boilerplate(block: str) -> bool:
    lowered = block.casefold()
    return any(marker in lowered for marker in WEB_BOILERPLATE_MARKERS)


def _content_words(text: str) -> set[str]:
    words = {
        word
        for word in _WORD_RE.findall(text.casefold())
        if len(word) > 2 and word not in _SECTION_STOPWORDS
    }
    return words - _ANNOTATION_WORDS


def _block_score(words: set[str], numbers: list[str], block: str) -> float:
    """Content-word hits plus claimed-number matches, the precheck's semantics."""
    score = float(len(words & _content_words(block)))
    if numbers:
        block_numbers = background_contract.extract_result_numbers(block)
        score += WINDOW_NUMBER_WEIGHT * sum(
            1
            for token in numbers
            if any(
                background_contract.result_number_matches(token, source_token)
                for source_token in block_numbers
            )
        )
    return score


def _first_hit(words: set[str], numbers: list[str], block: str) -> int | None:
    """Offset of the first scoring hit inside the block, or None."""
    first = next(
        (
            match.start()
            for match in _WORD_RE.finditer(block.casefold())
            if match.group(0) in words
        ),
        None,
    )
    if numbers:
        number_hit = next(
            (
                match.start()
                for match in _TOKEN_RE.finditer(block)
                if any(
                    background_contract.result_number_matches(token, source_token)
                    for token in numbers
                    for source_token in background_contract.extract_result_numbers(
                        match.group(0)
                    )
                )
            ),
            None,
        )
        if number_hit is not None and (first is None or number_hit < first):
            first = number_hit
    return first


def _anchor(block: str, hit: int | None) -> int:
    """In-block window anchor for a block exceeding the excerpt cap.

    A scored block anchors at its first hit with a small lead-in; an unhit
    (fallback) block anchors mid-block.  Anchoring at the start of a giant
    no-blank-line block shows the page head — direct-fallback content
    collapses all whitespace into one block, so block-start anchoring is the
    bare-head failure this locator exists to kill.
    """
    if len(block) <= EXCERPT_CHARS:
        return 0
    if hit is not None:
        return max(0, hit - WINDOW_LEAD_IN_CHARS)
    return (len(block) - EXCERPT_CHARS) // 2


def _locate_window(
    audit_text: str, visits: list[tuple[int, dict, str]]
) -> tuple[dict, str, int] | None:
    """(visit, content, offset) of the excerpt window for one mapping.

    Every non-boilerplate block of every visit scores against the audit
    text; ties keep the higher-tier visit's earlier block.  With no hit at
    all, the longest non-boilerplate block wins — never the bare head of the
    top visit, which is how a judge used to see only navigation headers.
    Blocks larger than the excerpt cap anchor at the first hit (or
    mid-block), not at the block start.
    """
    words = _content_words(audit_text)
    numbers = background_contract.extract_result_numbers(audit_text)
    best: tuple[float, dict, str, int, str] | None = None
    longest: tuple[int, dict, str, int, str] | None = None
    longest_any: tuple[int, dict, str, int, str] | None = None
    for _rank, visit, content in visits:
        for offset, block in _blocks(content):
            if longest_any is None or len(block) > longest_any[0]:
                longest_any = (len(block), visit, content, offset, block)
            if _is_boilerplate(block):
                continue
            if longest is None or len(block) > longest[0]:
                longest = (len(block), visit, content, offset, block)
            score = _block_score(words, numbers, block)
            if score > 0 and (best is None or score > best[0]):
                best = (score, visit, content, offset, block)
    chosen = best or longest or longest_any
    if chosen is None:
        return None
    _size, visit, content, offset, block = chosen
    hit = _first_hit(words, numbers, block) if best is not None else None
    return visit, content, offset + _anchor(block, hit)


def _nearest_heading(content: str, offset: int) -> str | None:
    """The markdown heading the offset sits under, when the content has any."""
    own = _HEADING_RE.match(content, offset)  # the window starts with a heading
    if own is not None:
        return own.group(1)
    title = None
    for match in _HEADING_RE.finditer(content, 0, offset):
        title = match.group(1)
    return title


def build_audit_entries(
    registry: dict, manifest: dict, manifest_dir: Path, mappings: list[dict]
) -> list[dict]:
    """One audit entry per mapping: excerpt window, number presence, coverage.

    Sources are joined to manifest receipts exactly the way
    ``background_contract validate`` joins them (registry URL → canonical key
    → receipt), so the judge audits what the deterministic gate accepted.
    Each entry's excerpt window is located by matching the audit text
    against the source's retained visit content (content-word hits plus
    claimed-number matches, falling back to the longest non-boilerplate
    block), and carries a content hash plus its provenance (visit content
    file, view, section, offset) so a later attempt can diff windows without
    re-rendering payloads; the number presence and coverage facts come from
    the deterministic precheck in ``background_contract``.
    """
    tiers = background_contract.source_verification(registry, manifest)
    sources = {
        source.get("id"): source
        for source in registry.get("sources", [])
        if isinstance(source, dict)
    }
    results_by_key = {r["canonical_key"]: r for r in merged_results(manifest)}

    entries: list[dict] = []
    content_cache: dict[str, list] = {}
    for mapping in mappings:
        source = sources.get(mapping["source_id"]) or {}
        url = source.get("url") or ""
        key = canonical_key(url) if url else ""
        tier = tiers.get(mapping["source_id"], "none")

        visit = None
        raw = None
        offset = 0
        section = None
        if tier in ("preview", "section", "full_text") and key:
            located = _locate_window(
                mapping["audit_text"],
                _key_visits(manifest, manifest_dir, key, content_cache),
            )
            if located is not None:
                visit, raw, offset = located
                section = visit.get("section") or _nearest_heading(raw, offset)
        if raw is not None:
            excerpt_kind = f"retained {visit['view']} visit content"
        else:
            raw = (results_by_key.get(key) or {}).get("snippet") or ""
            excerpt_kind = "search-result snippet"
        excerpt = raw[offset : offset + EXCERPT_CHARS]

        window = {
            "kind": excerpt_kind,
            "chars": len(excerpt),
            "sha256": "sha256:"
            + hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
            "truncated": offset > 0 or offset + len(excerpt) < len(raw),
            "offset": offset,
            "content_file": visit.get("content_file") if visit else None,
            "view": visit.get("view") if visit else None,
            "section": section,
        }
        if visit is None:
            window["text"] = excerpt  # snippets are short; keep them
        presence = background_contract.mapping_number_presence(
            mapping["_item"], mapping["_link"], registry, manifest, manifest_dir
        )
        entries.append(
            {
                "item_kind": mapping["item_kind"],
                "item_ids": mapping["item_ids"],
                "title": mapping["title"],
                "audit_text": mapping["audit_text"],
                "link_role": mapping["link_role"],
                "source_id": mapping["source_id"],
                "source_title": source.get("title"),
                "source_url": url or None,
                "verification": tier,
                "excerpt": window,
                "number_presence": presence["number_presence"],
                "missing_tokens": presence["missing_tokens"],
                "coverage": presence["coverage"],
                "_excerpt_text": excerpt,
            }
        )
    return entries


def _entry_block(entry: dict, label: str) -> str:
    """One mapping's block in the judge payload."""
    window = entry["excerpt"]
    coverage = entry["coverage"]
    presence = entry["number_presence"]
    header = f"- receipt excerpt ({window['kind']}"
    if window["view"]:
        header += f", view {window['view']}"
        if window["section"]:
            header += f", section {window['section']!r}"
        header += f", offset {window['offset']}"
    if window["truncated"]:
        header += ", truncated"
    if not window["chars"]:
        header += ", empty"
    header += "):"
    ids = ", ".join(f"`{item_id}`" for item_id in entry["item_ids"])
    return "\n".join(
        [
            f"## {label}",
            f"- item: {entry['item_kind']} {ids}"
            + (f" — {entry['title']}" if entry.get("title") else ""),
            "- audit text:",
            '"""',
            entry["audit_text"],
            '"""',
            f"- link role: {entry['link_role']}",
            f"- source: {entry['source_id']} — {entry['source_title']}"
            f" <{entry['source_url'] or ''}>",
            f"- verification: {entry['verification']}",
            f"- number presence: item={presence['item']}"
            f" this_source={presence['this_source']}",
            f"- coverage: routing={coverage['routing']} tier={coverage['tier']}"
            f" retained_chars={coverage['retained_chars']}"
            f" store_cap_hit={coverage['store_cap_hit']}",
            header,
            '"""',
            entry["_excerpt_text"],
            '"""',
            "",
        ]
    )


def build_judge_payload(entries: list[dict]) -> str:
    """Render one batch's judge payload, assigning per-batch labels M1..Mn."""
    lines = [
        f"Mappings to audit: {len(entries)}",
        "",
        "Each receipt excerpt is a window of the source's retained content "
        "located by matching the mapping's audit text (or the whole "
        "search-result snippet when no visit content is recorded); the "
        "window's view, section, and offset say where in the source it sits.",
        "",
    ]
    for index, entry in enumerate(entries, 1):
        entry["label"] = f"M{index}"
        lines.append(_entry_block(entry, entry["label"]))
    return "\n".join(lines)


def pack_batches(
    entries: list[dict],
    *,
    budget: int = BATCH_PAYLOAD_CHARS,
    max_mappings: int = BATCH_MAX_MAPPINGS,
) -> list[list[dict]]:
    """Greedy budget packing with one source's mappings adjacent.

    The payload budget bounds the blast radius of a lost batch verdict;
    mappings are grouped by source in first-appearance order so the judge
    sees one source's mappings together, then packed until the next entry
    would overflow the budget or the mapping cap.  A single oversized entry
    gets its own batch.
    """
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        groups.setdefault(
            entry.get("source_url") or str(entry.get("source_id")), []
        ).append(entry)
    batches: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for entry in (item for group in groups.values() for item in group):
        block = len(_entry_block(entry, "M1"))
        if current and (size + block > budget or len(current) >= max_mappings):
            batches.append(current)
            current, size = [], 0
        current.append(entry)
        size += block
    if current:
        batches.append(current)
    return batches


def verdict_errors(receipt: dict, labels: list[str]) -> list[str]:
    """Deterministic legitimacy check on a judge receipt's per-mapping verdicts."""
    verdicts = receipt.get("verdicts")
    if not isinstance(verdicts, list):
        return ["receipt.verdicts must be a list"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(verdicts):
        where = f"verdicts[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} must be an object")
            continue
        label = entry.get("mapping")
        if label not in labels:
            errors.append(f"{where}.mapping {label!r} is not a presented mapping")
        elif label in seen:
            errors.append(f"{where} duplicates {label}")
        else:
            seen.add(label)
        if entry.get("verdict") not in VERDICTS:
            errors.append(f"{where}.verdict must be one of {list(VERDICTS)}")
        rationale = entry.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"{where}.rationale must be a non-empty string")
    missing = sorted(set(labels) - seen)
    if missing:
        errors.append(f"verdicts missing mappings: {missing}")
    return errors


# =============================================================================
# Gate orchestration
# =============================================================================


def _load_audit_inputs(run_dir: Path) -> tuple[dict, dict]:
    """The same registry/manifest loading path the deterministic validate uses."""
    registry = background_contract.load_registry(run_dir / "background.md")
    manifest = json.loads(
        (run_dir / "background_retrieval.json").read_text(encoding="utf-8")
    )
    return registry, manifest


def _write_artifact(run_dir: Path, rounds: list[dict]) -> None:
    path = run_dir / ARTIFACT_NAME
    doc = {
        "version": ARTIFACT_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "rounds": rounds,
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def audit_completed(run_dir: Path) -> bool:
    """True when a prior audit reached a terminal-ok outcome on the current artifacts.

    Only an artifact at the current ``ARTIFACT_VERSION`` counts: anything
    older, unreadable, or ended on ``unfaithful``/``judge_failed`` re-runs the
    audit (fail closed).  ``unfaithful_irreparable_warning`` is terminal-ok:
    the pre-seeded background it flags is frozen, so re-auditing would just
    re-record the same warning.
    """
    path = run_dir / ARTIFACT_NAME
    if not path.exists():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("version") != ARTIFACT_VERSION:
            return False
        outcome = doc["rounds"][-1]["outcome"]
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        return False
    return outcome in TERMINAL_OK_OUTCOMES


def _judge(invoke, runner, store, task, tag, run_dir, payload, labels) -> tuple[dict, int]:
    """One judge rollout plus two fresh-session retries; raise InvocationFailed."""
    problems: list[str] = ["judge produced no legitimate receipt"]
    for _ in range(3):
        try:
            receipt, inv_id = invoke(
                runner, store, JUDGE_ROLE, task, tag, run_dir,
                inline_payload=payload,
            )
        except InvocationFailed as exc:
            problems = [str(p) for p in exc.problems]
            continue
        problems = verdict_errors(receipt, labels)
        if not problems:
            return receipt, inv_id
    raise InvocationFailed(JUDGE_ROLE, problems)


def _persisted_entry(entry: dict) -> dict:
    """An entry stripped of in-process-only keys, for the artifact."""
    return {key: value for key, value in entry.items() if not key.startswith("_")}


def _batch_record(
    batch: list[dict], receipt: dict | None, invocation_id, error=None
) -> dict:
    record = {
        "labels": [entry["label"] for entry in batch],
        "source_keys": sorted(
            {
                canonical_key(entry["source_url"])
                for entry in batch
                if entry.get("source_url")
            }
        ),
        "entries": [_persisted_entry(entry) for entry in batch],
        "receipt": receipt,
        "invocation_id": invocation_id,
    }
    if error is not None:
        record["error"] = error
    return record


def _format_findings(unfaithful: list[dict], *, budget: int = FINDINGS_CHARS) -> str:
    """Bounded repair payload: whole finding entries, truncation declared.

    The truncation unit is one finding entry — an entry is never cut in
    half.  Entries are added until the next one would overflow the budget
    (a single oversized entry still ships alone); whatever does not fit is
    declared with its count, since the artifact always carries the complete
    findings.
    """
    header = (
        "A faithfulness audit found these registry claims misrepresent their "
        "cited sources. For each one, either correct the claim and its "
        "evidence links so they match the recorded source content, or remove "
        "the item; then re-run the background validators.\n"
    )
    items = [
        {
            "item": f"{finding['entry']['item_kind']} "
                    + ", ".join(str(i) for i in finding["entry"]["item_ids"]),
            "audit_text": finding["entry"]["audit_text"],
            "link_role": finding["entry"]["link_role"],
            "source": f"{finding['entry']['source_id']} "
                      f"{finding['entry'].get('source_url')}",
            "judge_rationale": finding["rationale"],
        }
        for finding in unfaithful
    ]
    kept: list[dict] = []
    for item in items:
        candidate = len(header) + len(json.dumps(kept + [item], indent=2))
        if kept and candidate > budget:
            break
        kept.append(item)
    text = header + json.dumps(kept, indent=2)
    omitted = len(items) - len(kept)
    if omitted:
        text += (
            f"\n\nTruncated: {omitted} more finding(s) not shown; the "
            f"complete findings list is in {ARTIFACT_NAME}."
        )
    return text


def _clip(text: str, limit: int = 200) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _block_message(unfaithful: list[dict], prior_unfaithful: set[str]) -> str:
    """Terminal block rendered per identity, repeats vs new findings apart."""
    lines = ["background faithfulness audit still unfaithful after repair:"]
    for finding in unfaithful:
        entry = finding["entry"]
        status = "repeat" if _identity_key(entry) in prior_unfaithful else "new"
        ids = ", ".join(str(i) for i in entry["item_ids"])
        lines.append(
            f"- [{status}] {entry['item_kind']} {ids} — "
            f"{_clip(entry['audit_text'])} — {entry['source_id']}"
            f" <{entry.get('source_url')}> (role: {entry['link_role']}): "
            f"{finding['rationale']}"
        )
    return "\n".join(lines)


def run_faithfulness_gate(
    runner,
    store,
    task: str,
    tag: str,
    run_dir: Path,
    events,
    *,
    invoke,
    or_block,
    repair=None,
) -> None:
    """Audit a validated background before freeze; block on any failure.

    ``invoke``/``or_block`` are the driver loop's own helpers (injected to
    keep this module import-light).  ``repair`` is the one researcher repair
    round for the generated path; the re-audit after it re-judges only repeat
    unfaithful identities still present plus identities whose audit text or
    excerpt window changed (an empty recheck set passes as ``recheck_empty``).
    ``repair=None`` marks the pre-seeded path: the frozen background cannot be
    rewritten, so unfaithful findings are recorded into the artifact and
    events as ``unfaithful_irreparable_warning`` — a terminal-ok outcome —
    instead of blocking.  ``or_block`` and a failing ``repair`` never return
    (they raise RunBlocked).
    """
    rounds: list[dict] = []
    prior_windows: dict[str, str] = {}
    prior_unfaithful: set[str] = set()
    for attempt in (1, 2):
        try:
            registry, manifest = _load_audit_inputs(run_dir)
        except (OSError, ValueError) as exc:
            or_block(f"background faithfulness audit cannot load artifacts: {exc}")
        mappings = collect_claim_mappings(registry)
        if not mappings:
            rounds.append({"attempt": attempt, "outcome": "no_mappings", "batches": []})
            _write_artifact(run_dir, rounds)
            events.emit(
                "background_faithfulness_audit",
                outcome="no_mappings",
                attempt=attempt,
            )
            return
        entries = build_audit_entries(registry, manifest, run_dir, mappings)
        if attempt == 2:
            entries = [
                entry
                for entry in entries
                if _identity_key(entry) in prior_unfaithful
                or prior_windows.get(_identity_key(entry))
                != entry["excerpt"]["sha256"]
            ]
            if not entries:
                rounds.append(
                    {
                        "attempt": attempt,
                        "outcome": "passed",
                        "recheck_empty": True,
                        "batches": [],
                    }
                )
                _write_artifact(run_dir, rounds)
                events.emit(
                    "background_faithfulness_audit",
                    outcome="passed",
                    attempt=attempt,
                    recheck_empty=True,
                )
                return
        batches = pack_batches(entries)
        judged: list[dict] = []
        unfaithful: list[dict] = []
        unverifiable: list[dict] = []
        faithful_count = 0
        failed: tuple[list[dict], InvocationFailed] | None = None
        for batch in batches:
            payload = build_judge_payload(batch)
            labels = [entry["label"] for entry in batch]
            try:
                receipt, inv_id = _judge(
                    invoke, runner, store, task, tag, run_dir, payload, labels
                )
            except InvocationFailed as exc:
                failed = (batch, exc)
                break
            judged.append(_batch_record(batch, receipt, inv_id))
            by_label = {entry["label"]: entry for entry in batch}
            for verdict in receipt["verdicts"]:
                entry = by_label[verdict["mapping"]]
                if verdict["verdict"] == "unfaithful":
                    unfaithful.append(
                        {"entry": entry, "rationale": verdict.get("rationale")}
                    )
                elif verdict["verdict"] == "unverifiable":
                    unverifiable.append(entry)
                else:
                    faithful_count += 1

        if failed is not None:
            batch, exc = failed
            error = [str(p) for p in exc.problems]
            judged.append(_batch_record(batch, None, None, error=error))
            rounds.append(
                {
                    "attempt": attempt,
                    "audited_at": datetime.now(timezone.utc).isoformat(),
                    "outcome": "judge_failed",
                    "batches": judged,
                }
            )
            _write_artifact(run_dir, rounds)
            sources = judged[-1]["source_keys"]
            events.emit(
                "background_faithfulness_audit",
                outcome="judge_failed",
                attempt=attempt,
                batch=judged[-1]["labels"],
                sources=sources,
            )
            or_block(
                f"background faithfulness judge failed on batch"
                f" {len(judged)}/{len(batches)}"
                f" (sources: {', '.join(sources) or 'unknown'}): {exc.problems}"
            )

        outcome = "passed"
        if unfaithful:
            outcome = (
                "unfaithful"
                if repair is not None
                else "unfaithful_irreparable_warning"
            )
        rounds.append(
            {
                "attempt": attempt,
                "audited_at": datetime.now(timezone.utc).isoformat(),
                "outcome": outcome,
                "batches": judged,
                "unfaithful": [
                    {**_identity_record(f["entry"]), "rationale": f["rationale"]}
                    for f in unfaithful
                ],
                "unverifiable": [_identity_record(entry) for entry in unverifiable],
            }
        )
        _write_artifact(run_dir, rounds)
        event = {
            "outcome": outcome,
            "attempt": attempt,
            "batches": len(judged),
            "mappings": sum(len(b["labels"]) for b in judged),
            "faithful": faithful_count,
            "unfaithful": len(unfaithful),
            "unverifiable": len(unverifiable),
        }
        if outcome == "unfaithful_irreparable_warning":
            event["findings"] = rounds[-1]["unfaithful"]
        events.emit("background_faithfulness_audit", **event)
        if not unfaithful:
            return
        if repair is None:
            return  # pre-seeded frozen background: the warning above is terminal
        if attempt == 1:
            prior_windows = {
                _identity_key(entry): entry["excerpt"]["sha256"]
                for batch in batches
                for entry in batch
            }
            prior_unfaithful = {_identity_key(f["entry"]) for f in unfaithful}
            repair(_format_findings(unfaithful))
            continue  # re-validate already happened inside repair; narrowed recheck
        or_block(_block_message(unfaithful, prior_unfaithful))
