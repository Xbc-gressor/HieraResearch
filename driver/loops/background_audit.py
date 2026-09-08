"""Synchronous faithfulness audit for the background phase (design §4).

Deterministic contract validators own receipts ("what was actually
retrieved"); this module is the non-deterministic backstop that runs after
they pass and before the semantic space freezes.  An unpredictable random
sample of source→claim mappings is judged against the tool-recorded receipt
content by the tool-free ``background-faithfulness-judge`` role.  Unfaithful
findings on the generated path trigger exactly one researcher repair round
followed by a fresh audit; every other failure mode blocks the run — the
audit is the run's only faithfulness gate, so it fails closed.

The pure pieces (mapping collection, sampling, payload construction, verdict
validation) are separated from the gate so they can be smoke-tested offline
with an injected rng and stubbed role invocation.
"""

from __future__ import annotations

import json
import os
import random
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
from search_backends import canonical_key, merged_results  # noqa: E402

JUDGE_ROLE = "background-faithfulness-judge"
ARTIFACT_NAME = "background_faithfulness.json"

SAMPLE_SIZE = 5
EXCERPT_CHARS = 4000  # per-mapping receipt excerpt cap
EXCERPT_BUDGET = 20000  # total excerpt chars per audit payload
FINDINGS_CHARS = 6000  # cap on the repair payload handed to the researcher

VERDICTS = ("faithful", "unfaithful", "unverifiable")
_VIEW_RANK = {"preview": 1, "section": 2, "full_text": 3, "page": 3}


# =============================================================================
# Pure pieces: mapping collection, sampling, payload construction
# =============================================================================


def collect_claim_mappings(registry: dict) -> list[dict]:
    """Every source→claim evidence link carried by hypotheses and guidance."""
    mappings: list[dict] = []

    def collect(item_kind: str, item: dict) -> None:
        for link in item.get("evidence") or []:
            if not isinstance(link, dict):
                continue
            mappings.append(
                {
                    "item_kind": item_kind,
                    "item_id": item.get("id"),
                    "title": item.get("title"),
                    "claim": item.get("claim"),
                    "link_role": link.get("role"),
                    "source_id": link.get("source_id"),
                }
            )

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


def sample_mappings(
    mappings: list[dict], *, rng: random.Random | None = None, limit: int = SAMPLE_SIZE
) -> list[dict]:
    """Unpredictable bounded sample; ``rng`` is injectable for offline smoke.

    SystemRandom is the default on purpose: a sample the researcher cannot
    predict cannot be gamed (anti-Goodhart).
    """
    rng = rng or random.SystemRandom()
    return rng.sample(mappings, min(limit, len(mappings)))


def _visit_excerpt(manifest: dict, manifest_dir: Path, key: str) -> tuple[dict | None, str | None]:
    """Best successful substantive visit for one canonical key + its content."""
    best: tuple[int, dict] | None = None
    for visit in manifest.get("visits", []):
        if not isinstance(visit, dict) or visit.get("status") != "success":
            continue
        if visit.get("canonical_key") != key:
            continue
        rank = _VIEW_RANK.get(str(visit.get("view")))
        if rank is None:
            continue
        if best is None or rank > best[0]:
            best = (rank, visit)
    if best is None:
        return None, None
    visit = best[1]
    content_file = visit.get("content_file")
    if not isinstance(content_file, str) or not content_file:
        return None, None
    try:
        return visit, (manifest_dir / content_file).read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None, None


def build_judge_payload(
    registry: dict, manifest: dict, manifest_dir: Path, sampled: list[dict]
) -> tuple[str, list[dict]]:
    """Render the judge's inline payload and the artifact's sample records.

    Sources are joined to manifest receipts exactly the way
    ``background_contract validate`` joins them (registry URL → canonical key
    → receipt), so the judge audits what the deterministic gate accepted.
    """
    tiers = background_contract.source_verification(registry, manifest)
    sources = {
        source.get("id"): source
        for source in registry.get("sources", [])
        if isinstance(source, dict)
    }
    results_by_key = {r["canonical_key"]: r for r in merged_results(manifest)}

    lines = [f"Mappings to audit: {len(sampled)}", ""]
    entries: list[dict] = []
    budget = EXCERPT_BUDGET
    for index, mapping in enumerate(sampled, 1):
        label = f"M{index}"
        source = sources.get(mapping["source_id"]) or {}
        url = source.get("url") or ""
        key = canonical_key(url) if url else ""
        tier = tiers.get(mapping["source_id"], "none")
        cap = min(EXCERPT_CHARS, budget)

        excerpt = ""
        excerpt_kind = "none"
        visit = None
        raw = None
        if tier in ("preview", "section", "full_text") and key:
            visit, raw = _visit_excerpt(manifest, manifest_dir, key)
        if raw is not None:
            excerpt_kind = f"retained {visit['view']} visit content"
        else:
            raw = (results_by_key.get(key) or {}).get("snippet") or ""
            excerpt_kind = "search-result snippet"
        truncated = len(raw) > cap
        excerpt = raw[:cap]
        budget -= len(excerpt)

        entry = {
            **mapping,
            "label": label,
            "source_title": source.get("title"),
            "source_url": url or None,
            "verification": tier,
            "excerpt_kind": excerpt_kind,
            "excerpt_chars": len(excerpt),
        }
        if visit is not None:
            entry["content_file"] = visit.get("content_file")
            entry["view"] = visit.get("view")
        else:
            entry["excerpt_text"] = excerpt  # snippets are short; keep them
        entries.append(entry)

        lines += [
            f"## {label}",
            f"- item: {mapping['item_kind']} `{mapping['item_id']}`"
            + (f" — {mapping['title']}" if mapping.get("title") else ""),
            f"- claim: {mapping['claim']}",
            f"- link role: {mapping['link_role']}",
            f"- source: {mapping['source_id']} — {source.get('title')} <{url}>",
            f"- verification: {tier}",
            f"- receipt excerpt ({excerpt_kind}"
            + (", truncated" if truncated else "")
            + (", excerpt budget exhausted" if not excerpt else "")
            + "):",
            '"""',
            excerpt,
            '"""',
            "",
        ]
    return "\n".join(lines), entries


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
    doc = {"written_at": datetime.now(timezone.utc).isoformat(), "rounds": rounds}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def audit_completed(run_dir: Path) -> bool:
    """True when a prior audit reached a terminal-ok outcome on the current artifacts.

    A last round of ``unfaithful``/``judge_failed`` means the repair or judge
    retry that follows never completed (kill mid-audit); a repair may have
    rewritten the background since, so the audit must re-run.  A missing or
    unreadable artifact re-runs it too (fail closed).
    """
    path = run_dir / ARTIFACT_NAME
    if not path.exists():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        outcome = doc["rounds"][-1]["outcome"]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False
    return outcome in ("passed", "no_mappings")


def _judge(invoke, runner, store, task, tag, run_dir, payload, labels) -> tuple[dict, int]:
    """One judge rollout plus one fresh-session retry; raise InvocationFailed."""
    problems: list[str] = ["judge produced no legitimate receipt"]
    for _ in range(2):
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


def _format_findings(entries: list[dict], unfaithful: list[dict]) -> str:
    """Bounded repair payload: each unfaithful mapping plus the judge's reason."""
    by_label = {entry["label"]: entry for entry in entries}
    items = [
        {
            "item": f"{by_label[v['mapping']]['item_kind']} "
                    f"{by_label[v['mapping']]['item_id']}",
            "claim": by_label[v["mapping"]]["claim"],
            "link_role": by_label[v["mapping"]]["link_role"],
            "source": f"{by_label[v['mapping']]['source_id']} "
                      f"{by_label[v['mapping']].get('source_url')}",
            "judge_rationale": v.get("rationale"),
        }
        for v in unfaithful
    ]
    text = (
        "A faithfulness audit found these registry claims misrepresent their "
        "cited sources. For each one, either correct the claim and its "
        "evidence links so they match the recorded source content, or remove "
        "the item; then re-run the background validators.\n"
        + json.dumps(items, indent=2)
    )
    return text[:FINDINGS_CHARS]


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
    rng: random.Random | None = None,
) -> None:
    """Audit a validated background before freeze; block on any failure.

    ``invoke``/``or_block`` are the driver loop's own helpers (injected to
    keep this module import-light).  ``repair`` is the one researcher repair
    round for the generated path — None marks the preseeded path, where an
    unfaithful finding blocks immediately.  ``or_block`` and a failing
    ``repair`` never return (they raise RunBlocked).
    """
    rounds: list[dict] = []
    for attempt in (1, 2):
        try:
            registry, manifest = _load_audit_inputs(run_dir)
        except (OSError, ValueError) as exc:
            or_block(f"background faithfulness audit cannot load artifacts: {exc}")
        mappings = collect_claim_mappings(registry)
        sampled = sample_mappings(mappings, rng=rng)
        if not sampled:
            rounds.append(
                {"attempt": attempt, "outcome": "no_mappings", "sample": []}
            )
            _write_artifact(run_dir, rounds)
            events.emit("background_faithfulness_audit", outcome="no_mappings")
            return
        payload, entries = build_judge_payload(registry, manifest, run_dir, sampled)
        labels = [entry["label"] for entry in entries]
        try:
            receipt, inv_id = _judge(
                invoke, runner, store, task, tag, run_dir, payload, labels
            )
        except InvocationFailed as exc:
            rounds.append(
                {
                    "attempt": attempt,
                    "outcome": "judge_failed",
                    "sample": entries,
                    "error": [str(p) for p in exc.problems],
                }
            )
            _write_artifact(run_dir, rounds)
            events.emit(
                "background_faithfulness_audit",
                outcome="judge_failed",
                attempt=attempt,
            )
            or_block(f"background faithfulness judge failed: {exc.problems}")

        verdicts = receipt["verdicts"]
        unfaithful = [v for v in verdicts if v.get("verdict") == "unfaithful"]
        rounds.append(
            {
                "attempt": attempt,
                "audited_at": datetime.now(timezone.utc).isoformat(),
                "invocation_id": inv_id,
                "sample": entries,
                "judge_receipt": receipt,
                "outcome": "unfaithful" if unfaithful else "passed",
                "unfaithful": [v["mapping"] for v in unfaithful],
                "unverifiable": [
                    v["mapping"]
                    for v in verdicts
                    if v.get("verdict") == "unverifiable"
                ],
            }
        )
        _write_artifact(run_dir, rounds)
        events.emit(
            "background_faithfulness_audit",
            outcome="unfaithful" if unfaithful else "passed",
            attempt=attempt,
            invocation_id=inv_id,
            sample=len(entries),
            unfaithful=len(unfaithful),
            unverifiable=len(rounds[-1]["unverifiable"]),
        )
        if not unfaithful:
            return
        if repair is None:
            or_block(
                "pre-seeded background failed the faithfulness audit: "
                + json.dumps(rounds[-1]["unfaithful"])
            )
        if attempt == 1:
            repair(_format_findings(entries, unfaithful))
            continue  # re-validate already happened inside repair; fresh sample
        or_block(
            "background faithfulness audit still unfaithful after repair: "
            + json.dumps(rounds[-1]["unfaithful"])
        )
