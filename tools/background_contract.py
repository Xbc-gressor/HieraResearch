#!/usr/bin/env python3
"""Validate and join HieraResearch background directions with run evidence.

``background.md`` remains a readable Markdown brief, but its ``Direction
registry`` JSON fence is the machine-readable contract shared by the
orchestrator, background researcher, experience extractor, and idea generator.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from search_backends import canonical_key, validate_manifest


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, SCHEMA_VERSION}
TF_RE = re.compile(r"^tf-(\d{2,})$")
SOURCE_RE = re.compile(r"^src-(\d{2,})$")
GUIDANCE_RE = re.compile(r"^g-(\d{2,})$")
SCOPE_TAG_RE = re.compile(r"^(?:\*|[a-z0-9][a-z0-9._-]*)$")

LITERATURE_CREDIBILITY = {
    "unverified",
    "preliminary",
    "corroborated",
    "replicated",
    "contested",
}
SOURCE_TYPES = {
    "paper",
    "official_code",
    "official_docs",
    "benchmark",
    "dataset",
    "first_party_report",
    "web_lead",
}
BINDING_SOURCE_TYPES = {"paper", "benchmark", "first_party_report"}
PUBLICATION_STATUS = {
    "preprint_only",
    "peer_reviewed",
    "published_status_unknown",
    "withdrawn_or_retracted",
    "not_applicable",
}
VALIDATION_STATUS = {
    "claim_only",
    "artifact_available",
    "independently_reproduced",
    "not_assessed",
}
EVIDENCE_ROLES = {"supports", "contradicts", "context"}
DIRECTION_KINDS = {"evidence_prior", "scope_probe"}
SCOPE_AXES = (
    "model_families",
    "data_regimes",
    "metrics",
    "interventions",
    "evaluation_protocols",
)
GUIDANCE_SECTIONS = {"pitfall", "deprioritize"}
GUIDANCE_EFFECTS = {"caution", "deprioritize", "exclude"}
RUN_STATUS = {
    "untested",
    "inconclusive",
    "supported_here",
    "contradicted_here",
    "mixed",
}
CONFIDENCE = {"low", "med", "high"}
CLAIM_COVERAGE = {"none", "partial", "direct"}


class ContractError(ValueError):
    """A malformed or incompatible background contract."""


def load_registry(path: Path) -> dict[str, Any]:
    """Extract the canonical JSON registry from a Markdown background brief."""
    text = path.read_text(errors="replace")
    marker = re.search(r"^## Direction registry\s*$", text, flags=re.MULTILINE)
    if marker is None:
        raise ContractError(f"{path}: missing '## Direction registry'")
    fence = re.search(
        r"```json\s*(\{.*?\})\s*```",
        text[marker.end() :],
        flags=re.DOTALL,
    )
    if fence is None:
        raise ContractError(f"{path}: direction registry must be a fenced JSON object")
    try:
        registry = json.loads(fence.group(1))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path}: invalid direction registry JSON: {exc}") from exc
    if not isinstance(registry, dict):
        raise ContractError(f"{path}: direction registry must be an object")
    return registry


def validate_background_markdown(
    path: Path, registry: dict[str, Any]
) -> list[str]:
    """Keep human negative guidance aligned with the machine contract."""
    if registry.get("schema_version") != SCHEMA_VERSION:
        return []
    text = path.read_text(errors="replace")
    guidance = {
        item.get("id"): item
        for item in registry.get("guidance", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    errors: list[str] = []
    referenced: set[str] = set()
    section_names = {"Pitfalls": "pitfall", "Deprioritize": "deprioritize"}

    for heading, expected_section in section_names.items():
        match = re.search(rf"^## {re.escape(heading)}\s*$", text, flags=re.MULTILINE)
        if match is None:
            if heading == "Pitfalls" or any(
                item.get("section") == expected_section for item in guidance.values()
            ):
                errors.append(f"background.md is missing required '## {heading}' section")
            continue
        next_heading = re.search(r"^##\s+", text[match.end() :], flags=re.MULTILINE)
        end = match.end() + next_heading.start() if next_heading else len(text)
        body = text[match.end() : end]
        has_marked_bullet = False
        for line_number, raw_line in enumerate(body.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if raw_line[:1].isspace() and has_marked_bullet:
                continue
            marker = re.match(
                r"^-\s+`(?P<id>g-\d{2,}|task-constraint|operational)`(?:\s|$)",
                line,
            )
            if marker is None:
                has_marked_bullet = False
                errors.append(
                    f"{heading} line {line_number} must start with a registered `g-NN`, "
                    "`task-constraint`, or `operational` marker"
                )
                continue
            has_marked_bullet = True
            marker_id = marker.group("id")
            if not GUIDANCE_RE.fullmatch(marker_id):
                continue
            item = guidance.get(marker_id)
            if item is None:
                errors.append(f"{heading} references unknown guidance id {marker_id}")
                continue
            referenced.add(marker_id)
            if item.get("section") != expected_section:
                errors.append(
                    f"{heading} references {marker_id}, but registry section is "
                    f"{item.get('section')!r}"
                )

    missing = sorted(set(guidance) - referenced)
    if missing:
        errors.append(f"structured guidance is not referenced in its Markdown section: {missing}")
    return errors


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_scope(scope: Any, where: str) -> list[str]:
    """Validate a conservative, exact-tag scope shared by claims and directions."""
    errors: list[str] = []
    if not isinstance(scope, dict):
        return [f"{where} must be an object"]
    unknown_axes = sorted(set(scope) - set(SCOPE_AXES))
    if unknown_axes:
        errors.append(f"{where} has unknown axes {unknown_axes}")
    for axis in SCOPE_AXES:
        values = scope.get(axis)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str) or SCOPE_TAG_RE.fullmatch(value) is None
                for value in values
            )
        ):
            errors.append(
                f"{where}.{axis} must be a non-empty list of lowercase scope tags"
            )
        elif len(values) != len(set(values)):
            errors.append(f"{where}.{axis} must not contain duplicate tags")
        elif "*" in values and len(values) != 1:
            errors.append(f"{where}.{axis} wildcard must be the only tag")
    return errors


def scope_relation(claim_scope: Any, target_scope: Any) -> str:
    """Return direct, partial, mismatch, or unknown for claim -> target scope.

    A claim is direct only when every target tag is covered on every axis.
    Any disjoint axis makes it a mismatch. Overlap without containment is only
    partial and cannot drive deterministic deprioritization or exclusion.
    """
    if _validate_scope(claim_scope, "claim_scope") or _validate_scope(
        target_scope, "target_scope"
    ):
        return "unknown"
    direct = True
    for axis in SCOPE_AXES:
        claim_values = set(claim_scope[axis])
        target_values = set(target_scope[axis])
        if "*" in claim_values:
            continue
        if claim_values.isdisjoint(target_values):
            return "mismatch"
        if not target_values.issubset(claim_values):
            direct = False
    return "direct" if direct else "partial"


def validate_registry(
    registry: dict[str, Any],
    *,
    ledger: dict[str, Any] | None = None,
    retrieval_manifest: dict[str, Any] | None = None,
) -> list[str]:
    """Return contract errors; an empty list means the registry is valid."""
    errors: list[str] = []
    schema_version = registry.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(
            f"schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    scoped_contract = schema_version == SCHEMA_VERSION

    directions = registry.get("directions")
    sources = registry.get("sources")
    if not isinstance(directions, list) or not directions:
        errors.append("directions must be a non-empty list")
        directions = []
    if not isinstance(sources, list) or not sources:
        errors.append("sources must be a non-empty list")
        sources = []

    source_ids: set[str] = set()
    source_urls: dict[str, str] = {}
    source_keys: dict[str, str] = {}
    source_by_id: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources, start=1):
        where = f"sources[{index - 1}]"
        if not isinstance(source, dict):
            errors.append(f"{where} must be an object")
            continue
        source_id = source.get("id")
        if not isinstance(source_id, str) or SOURCE_RE.fullmatch(source_id) is None:
            errors.append(f"{where}.id must match src-NN")
        elif source_id in source_ids:
            errors.append(f"duplicate source id {source_id}")
        else:
            source_ids.add(source_id)
            source_by_id[source_id] = source
        if source.get("type") not in SOURCE_TYPES:
            errors.append(f"{where}.type must be one of {sorted(SOURCE_TYPES)}")
        if source.get("publication_status") not in PUBLICATION_STATUS:
            errors.append(
                f"{where}.publication_status must be one of {sorted(PUBLICATION_STATUS)}"
            )
        if source.get("validation_status") not in VALIDATION_STATUS:
            errors.append(
                f"{where}.validation_status must be one of {sorted(VALIDATION_STATUS)}"
            )
        for field in ("title", "url"):
            if not _nonempty_string(source.get(field)):
                errors.append(f"{where}.{field} must be a non-empty string")
        if scoped_contract:
            errors.extend(_validate_scope(source.get("studied_scope"), f"{where}.studied_scope"))
        url = source.get("url")
        if _nonempty_string(url) and not url.startswith(("https://", "http://")):
            errors.append(f"{where}.url must be an HTTP(S) URL")
        if isinstance(source_id, str) and _nonempty_string(url):
            source_urls[source_id] = url
            if scoped_contract:
                key = canonical_key(url)
                previous = source_keys.get(key)
                if previous is not None:
                    errors.append(
                        f"sources {previous} and {source_id} duplicate canonical work {key}"
                    )
                else:
                    source_keys[key] = source_id

    if retrieval_manifest is not None:
        errors.extend(validate_manifest(retrieval_manifest))
        if retrieval_manifest.get("retrieval_condition") == "mixed":
            errors.append(
                "background evidence cannot mix frozen and live retrieval in one condition"
            )
        visited_grounding = {
            visit.get("canonical_key")
            for visit in retrieval_manifest.get("visits", [])
            if isinstance(visit, dict)
            and visit.get("status") == "success"
            and visit.get("lane") == "grounding"
        }
        for source_id, url in source_urls.items():
            if canonical_key(url) not in visited_grounding:
                errors.append(
                    f"source {source_id} was not successfully visited in the grounding lane"
                )

    direction_ids: list[str] = []
    direction_by_id: dict[str, dict[str, Any]] = {}
    required_text = (
        "title",
        "claim",
        "credibility_rationale",
        "testable_expectation",
    )
    for index, direction in enumerate(directions, start=1):
        where = f"directions[{index - 1}]"
        if not isinstance(direction, dict):
            errors.append(f"{where} must be an object")
            continue
        direction_id = direction.get("id")
        expected_id = f"tf-{index:02d}"
        if direction_id != expected_id:
            errors.append(f"{where}.id must be {expected_id} (priority order, no gaps)")
        if isinstance(direction_id, str):
            if direction_id in direction_by_id:
                errors.append(f"duplicate direction id {direction_id}")
            else:
                direction_ids.append(direction_id)
                direction_by_id[direction_id] = direction
        for field in required_text:
            if not _nonempty_string(direction.get(field)):
                errors.append(f"{where}.{field} must be a non-empty string")
        if scoped_contract:
            kind = direction.get("kind")
            if kind not in DIRECTION_KINDS:
                errors.append(f"{where}.kind must be one of {sorted(DIRECTION_KINDS)}")
            if kind == "evidence_prior" and direction.get("probe_for"):
                errors.append(f"{where}.probe_for is only valid for a scope_probe")
            if not _nonempty_string(direction.get("claim_scope")):
                errors.append(f"{where}.claim_scope must be a non-empty string")
            errors.extend(_validate_scope(direction.get("scope"), f"{where}.scope"))
            values = direction.get("required_comparisons")
            if (
                not isinstance(values, list)
                or not values
                or any(not _nonempty_string(value) for value in values)
            ):
                errors.append(
                    f"{where}.required_comparisons must be a non-empty string list"
                )
            if not _nonempty_string(direction.get("reopen_when")):
                errors.append(f"{where}.reopen_when must be a non-empty string")
        credibility = direction.get("literature_credibility")
        if credibility not in LITERATURE_CREDIBILITY:
            errors.append(
                f"{where}.literature_credibility must be one of "
                f"{sorted(LITERATURE_CREDIBILITY)}"
            )
        evidence_links = direction.get("evidence")
        evidence_roles: list[str] = []
        if not isinstance(evidence_links, list) or not evidence_links:
            errors.append(f"{where}.evidence must be a non-empty list")
        else:
            for evidence_index, link in enumerate(evidence_links):
                link_where = f"{where}.evidence[{evidence_index}]"
                if not isinstance(link, dict):
                    errors.append(f"{link_where} must be an object")
                    continue
                source_id = link.get("source_id")
                role = link.get("role")
                if source_id not in source_ids:
                    errors.append(f"{link_where} references unknown source id {source_id!r}")
                if role not in EVIDENCE_ROLES:
                    errors.append(f"{link_where}.role must be one of {sorted(EVIDENCE_ROLES)}")
                else:
                    evidence_roles.append(role)
        if credibility == "contested" and "contradicts" not in evidence_roles:
            errors.append(f"{where} is contested but cites no contradicting evidence")
        if credibility == "replicated":
            replicated_support = any(
                link.get("role") == "supports"
                and source_by_id.get(link.get("source_id"), {}).get("validation_status")
                == "independently_reproduced"
                for link in (evidence_links if isinstance(evidence_links, list) else [])
                if isinstance(link, dict)
            )
            if not replicated_support:
                errors.append(
                    f"{where} is replicated but has no independently reproduced supporting source"
                )

    if scoped_contract:
        guidance = registry.get("guidance")
        if not isinstance(guidance, list):
            errors.append("guidance must be a list")
            guidance = []
        guidance_ids: list[str] = []
        guidance_by_id: dict[str, dict[str, Any]] = {}
        binding_guidance_ids: set[str] = set()
        for index, item in enumerate(guidance, start=1):
            where = f"guidance[{index - 1}]"
            if not isinstance(item, dict):
                errors.append(f"{where} must be an object")
                continue
            guidance_id = item.get("id")
            expected_id = f"g-{index:02d}"
            if guidance_id != expected_id:
                errors.append(f"{where}.id must be {expected_id} (order, no gaps)")
            if isinstance(guidance_id, str):
                if guidance_id in guidance_by_id:
                    errors.append(f"duplicate guidance id {guidance_id}")
                else:
                    guidance_ids.append(guidance_id)
                    guidance_by_id[guidance_id] = item
            if item.get("section") not in GUIDANCE_SECTIONS:
                errors.append(
                    f"{where}.section must be one of {sorted(GUIDANCE_SECTIONS)}"
                )
            effect = item.get("effect")
            if effect not in GUIDANCE_EFFECTS:
                errors.append(f"{where}.effect must be one of {sorted(GUIDANCE_EFFECTS)}")
            for field in ("claim", "credibility_rationale", "reopen_when"):
                if not _nonempty_string(item.get(field)):
                    errors.append(f"{where}.{field} must be a non-empty string")
            credibility = item.get("literature_credibility")
            if credibility not in LITERATURE_CREDIBILITY:
                errors.append(
                    f"{where}.literature_credibility must be one of "
                    f"{sorted(LITERATURE_CREDIBILITY)}"
                )
            errors.extend(_validate_scope(item.get("scope"), f"{where}.scope"))

            evidence_links = item.get("evidence")
            direct_support_ids: set[str] = set()
            evidence_roles: list[str] = []
            if not isinstance(evidence_links, list) or not evidence_links:
                errors.append(f"{where}.evidence must be a non-empty list")
                evidence_links = []
            for evidence_index, link in enumerate(evidence_links):
                link_where = f"{where}.evidence[{evidence_index}]"
                if not isinstance(link, dict):
                    errors.append(f"{link_where} must be an object")
                    continue
                source_id = link.get("source_id")
                role = link.get("role")
                if source_id not in source_ids:
                    errors.append(f"{link_where} references unknown source id {source_id!r}")
                if role not in EVIDENCE_ROLES:
                    errors.append(
                        f"{link_where}.role must be one of {sorted(EVIDENCE_ROLES)}"
                    )
                    continue
                evidence_roles.append(role)
                source = source_by_id.get(source_id, {})
                relation = scope_relation(source.get("studied_scope"), item.get("scope"))
                if (
                    role == "supports"
                    and relation == "direct"
                    and source.get("type") in BINDING_SOURCE_TYPES
                    and source.get("publication_status") != "withdrawn_or_retracted"
                ):
                    direct_support_ids.add(str(source_id))
            if credibility == "contested" and "contradicts" not in evidence_roles:
                errors.append(f"{where} is contested but cites no contradicting evidence")
            if effect in {"deprioritize", "exclude"}:
                if credibility not in {"preliminary", "corroborated", "replicated"}:
                    errors.append(
                        f"{where}.{effect} requires preliminary, corroborated, or "
                        f"replicated evidence; {credibility!r} may only caution"
                    )
                if not direct_support_ids:
                    errors.append(
                        f"{where}.{effect} requires a non-withdrawn primary empirical source "
                        "whose studied scope directly contains the guidance scope"
                    )
                elif isinstance(guidance_id, str):
                    binding_guidance_ids.add(guidance_id)
            if effect == "exclude":
                if credibility not in {"corroborated", "replicated"}:
                    errors.append(
                        f"{where}.exclude requires corroborated or replicated evidence"
                    )
                if len(direct_support_ids) < 2:
                    errors.append(
                        f"{where}.exclude requires two directly scoped supporting sources"
                    )
                reproduced_support = any(
                    source_by_id.get(source_id, {}).get("validation_status")
                    == "independently_reproduced"
                    for source_id in direct_support_ids
                )
                if not reproduced_support:
                    errors.append(
                        f"{where}.exclude requires directly scoped independent reproduction"
                    )

        probed_guidance_ids: set[str] = set()
        for direction in directions:
            if not isinstance(direction, dict) or direction.get("kind") != "scope_probe":
                continue
            where = f"direction {direction.get('id')}"
            probe_for = direction.get("probe_for")
            if (
                not isinstance(probe_for, list)
                or not probe_for
                or any(not isinstance(item, str) for item in probe_for)
            ):
                errors.append(f"{where}.probe_for must be a non-empty guidance-id list")
                continue
            unknown_targets = sorted(set(probe_for) - set(guidance_ids))
            if unknown_targets:
                errors.append(f"{where}.probe_for references unknown ids {unknown_targets}")
            for guidance_id in set(probe_for) & set(guidance_ids):
                relation = scope_relation(
                    guidance_by_id[guidance_id].get("scope"), direction.get("scope")
                )
                if relation == "direct":
                    errors.append(
                        f"{where} is not a boundary probe for {guidance_id}; scopes match directly"
                    )
                else:
                    probed_guidance_ids.add(guidance_id)
        unprobed_guidance = sorted(binding_guidance_ids - probed_guidance_ids)
        if unprobed_guidance:
            errors.append(
                "binding external guidance requires an out-of-scope scope_probe direction; "
                f"unprobed guidance: {unprobed_guidance}"
            )

    if ledger is not None:
        for record in ledger.get("records", []):
            run_id = str(record.get("run_id"))
            for source in record.get("source_run_ids") or []:
                source = str(source)
                if source.startswith("tf-") and source not in direction_by_id:
                    errors.append(f"ledger run {run_id} references unknown direction {source}")

    return errors


def derive_direction_selection(
    registry: dict[str, Any], ledger: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Derive eligibility from typed scope; prose pitfalls have no blocking power."""
    experience_entries = (
        (ledger or {}).get("experience", {}).get("direction_evidence", [])
        if isinstance((ledger or {}).get("experience"), dict)
        else []
    )
    experience_by_id = {
        entry.get("direction_id"): entry
        for entry in experience_entries
        if isinstance(entry, dict) and isinstance(entry.get("direction_id"), str)
    }
    guidance = [
        item for item in registry.get("guidance", []) if isinstance(item, dict)
    ]
    result: dict[str, dict[str, Any]] = {}
    effect_rank = {"caution": 0, "deprioritize": 1, "exclude": 2}
    for direction in registry.get("directions", []):
        if not isinstance(direction, dict) or not isinstance(direction.get("id"), str):
            continue
        matched = []
        binding = []
        strongest = "caution"
        for item in guidance:
            relation = scope_relation(item.get("scope"), direction.get("scope"))
            if relation != "direct":
                continue
            effect = item.get("effect")
            if effect not in GUIDANCE_EFFECTS:
                continue
            receipt = {"id": item.get("id"), "effect": effect}
            matched.append(receipt)
            if effect in {"deprioritize", "exclude"}:
                binding.append(receipt)
            if effect_rank[effect] > effect_rank[strongest]:
                strongest = effect

        selection_status = {
            "caution": "active",
            "deprioritize": "deprioritized",
            "exclude": "excluded",
        }[strongest]
        local = experience_by_id.get(direction["id"], {})
        reopened = (
            local.get("claim_coverage") == "direct"
            and local.get("run_status") in {"supported_here", "mixed"}
        )
        if reopened:
            selection_status = "active"
        result[direction["id"]] = {
            "selection_status": selection_status,
            "matched_guidance": matched,
            "binding_guidance": binding,
            "reopened_by_run_status": local.get("run_status") if reopened else None,
        }
    return result


def _safe_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def derive_lineage(registry: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    """Map every candidate to its originating tf-* hypotheses.

    A single-origin descendant is useful lineage evidence. Multi-origin
    descendants are reported separately because crossover success cannot be
    credited causally to every contributing direction.
    """
    directions = registry.get("directions", [])
    direction_ids = [d["id"] for d in directions if isinstance(d, dict) and "id" in d]
    records = {
        str(record.get("run_id")): record
        for record in ledger.get("records", [])
        if record.get("run_id") is not None
    }
    memo: dict[str, set[str]] = {}
    warnings: list[str] = []

    def origins(run_id: str, visiting: set[str] | None = None) -> set[str]:
        if run_id in memo:
            return memo[run_id]
        visiting = set() if visiting is None else visiting
        if run_id in visiting:
            warnings.append(f"cycle while resolving lineage at run {run_id}")
            return set()
        record = records.get(run_id)
        if record is None:
            warnings.append(f"missing parent record {run_id}")
            return set()
        visiting.add(run_id)
        found: set[str] = set()
        for source in record.get("source_run_ids") or []:
            source = str(source)
            if TF_RE.fullmatch(source):
                found.add(source)
            elif source in records:
                found.update(origins(source, visiting))
            elif source.isdigit():
                warnings.append(f"run {run_id} references missing parent {source}")
        visiting.remove(run_id)
        memo[run_id] = found
        return found

    evidence = {
        direction_id: {
            "literature_credibility": next(
                d.get("literature_credibility")
                for d in directions
                if isinstance(d, dict) and d.get("id") == direction_id
            ),
            "direct_runs": [],
            "single_origin_descendants": [],
            "combination_runs": [],
        }
        for direction_id in direction_ids
    }
    run_origins: dict[str, list[str]] = {}

    for run_id, record in records.items():
        origin_set = origins(run_id)
        run_origins[run_id] = sorted(origin_set)
        item = {
            "run_id": run_id,
            "op": record.get("op"),
            "status": record.get("status"),
            "score": _safe_score(record.get("final_best_score")),
        }
        direct_tags = {
            str(source)
            for source in record.get("source_run_ids") or []
            if TF_RE.fullmatch(str(source))
        }
        for direction_id in origin_set:
            if direction_id not in evidence:
                continue
            if record.get("op") == "fresh" and direction_id in direct_tags:
                evidence[direction_id]["direct_runs"].append(item)
            elif len(origin_set) == 1:
                evidence[direction_id]["single_origin_descendants"].append(item)
            else:
                evidence[direction_id]["combination_runs"].append(item)

    return {
        "schema_version": registry.get("schema_version", SCHEMA_VERSION),
        "directions": evidence,
        "run_origins": run_origins,
        "warnings": sorted(set(warnings)),
    }


def validate_experience(
    experience: dict[str, Any], registry: dict[str, Any], ledger: dict[str, Any]
) -> list[str]:
    """Validate the structured tf-* run-status view against actual lineage."""
    errors: list[str] = []
    entries = experience.get("direction_evidence")
    if not isinstance(entries, list):
        return ["experience.direction_evidence must be a list"]

    lineage = derive_lineage(registry, ledger)
    errors.extend(f"lineage: {warning}" for warning in lineage["warnings"])
    expected = lineage["directions"]
    records = {
        str(record.get("run_id")): record
        for record in ledger.get("records", [])
        if record.get("run_id") is not None
    }
    scoped_contract = registry.get("schema_version") == SCHEMA_VERSION
    if scoped_contract and "lessons" in experience:
        lessons = experience.get("lessons")
        if not isinstance(lessons, list):
            errors.append("experience.lessons must be a list")
            lessons = []
        for index, lesson in enumerate(lessons):
            where = f"lessons[{index}]"
            if not isinstance(lesson, dict):
                errors.append(f"{where} must be an object")
                continue
            if lesson.get("kind") != "deadend":
                continue
            errors.extend(_validate_scope(lesson.get("scope"), f"{where}.scope"))
            if not _nonempty_string(lesson.get("reopen_when")):
                errors.append(
                    f"{where}.reopen_when must be a non-empty string for a deadend"
                )
            evidence_runs = lesson.get("evidence")
            if not isinstance(evidence_runs, list) or any(
                not isinstance(run_id, str) for run_id in evidence_runs
            ):
                errors.append(f"{where}.evidence must be a list of run-id strings")
                continue
            evidence_runs = list(dict.fromkeys(evidence_runs))
            unknown_runs = sorted(set(evidence_runs) - set(records))
            if unknown_runs:
                errors.append(f"{where}.evidence contains unknown runs {unknown_runs}")
            scored_runs = [
                run_id
                for run_id in evidence_runs
                if records.get(run_id, {}).get("status") in {"keep", "discard"}
                and _safe_score(records.get(run_id, {}).get("final_best_score")) is not None
            ]
            if len(scored_runs) < 2:
                errors.append(
                    f"{where} deadend requires at least two scored non-crash runs; "
                    f"got {scored_runs}"
                )
    ledger_edges = {
        f"{source}->{run_id}"
        for run_id, record in records.items()
        for source in (record.get("source_run_ids") or [])
        if str(source) in records
    }
    seen: set[str] = set()

    relation_fields = {
        "direct_runs": "direct_runs",
        "descendant_runs": "single_origin_descendants",
        "combination_runs": "combination_runs",
    }
    for index, entry in enumerate(entries):
        where = f"direction_evidence[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} must be an object")
            continue
        direction_id = entry.get("direction_id")
        if direction_id not in expected:
            errors.append(f"{where}.direction_id is unknown: {direction_id!r}")
            continue
        if direction_id in seen:
            errors.append(f"duplicate direction evidence for {direction_id}")
            continue
        seen.add(direction_id)

        expected_direction = expected[direction_id]
        if entry.get("literature_credibility") != expected_direction["literature_credibility"]:
            errors.append(
                f"{where}.literature_credibility does not match background.md for {direction_id}"
            )
        if entry.get("run_status") not in RUN_STATUS:
            errors.append(f"{where}.run_status must be one of {sorted(RUN_STATUS)}")
        if entry.get("confidence") not in CONFIDENCE:
            errors.append(f"{where}.confidence must be one of {sorted(CONFIDENCE)}")
        if not _nonempty_string(entry.get("rationale")):
            errors.append(f"{where}.rationale must be a non-empty string")

        if scoped_contract:
            coverage = entry.get("claim_coverage")
            comparison_runs = entry.get("comparison_runs")
            missing_comparisons = entry.get("missing_comparisons")
            if coverage not in CLAIM_COVERAGE:
                errors.append(
                    f"{where}.claim_coverage must be one of {sorted(CLAIM_COVERAGE)}"
                )
            if not isinstance(comparison_runs, list) or any(
                not isinstance(run_id, str) for run_id in comparison_runs
            ):
                errors.append(f"{where}.comparison_runs must be a list of run-id strings")
                comparison_runs = []
            else:
                comparison_runs = list(dict.fromkeys(comparison_runs))
                unknown_runs = sorted(set(comparison_runs) - set(records))
                if unknown_runs:
                    errors.append(
                        f"{where}.comparison_runs contains unknown runs {unknown_runs}"
                    )
            if not isinstance(missing_comparisons, list) or any(
                not _nonempty_string(item) for item in missing_comparisons
            ):
                errors.append(f"{where}.missing_comparisons must be a string list")
                missing_comparisons = []

            run_status = entry.get("run_status")
            decisive = run_status in {"supported_here", "contradicted_here", "mixed"}
            if run_status == "untested" and coverage != "none":
                errors.append(f"{where}.claim_coverage must be none while untested")
            if coverage == "none" and comparison_runs:
                errors.append(f"{where}.comparison_runs must be empty when coverage is none")
            if coverage == "partial" and not missing_comparisons:
                errors.append(
                    f"{where}.missing_comparisons must name the uncovered comparator or scope"
                )
            if coverage == "direct" and missing_comparisons:
                errors.append(
                    f"{where}.missing_comparisons must be empty when coverage is direct"
                )
            if decisive and coverage != "direct":
                errors.append(
                    f"{where}.run_status {run_status} requires direct claim coverage"
                )
            if decisive and len(comparison_runs) < 2:
                errors.append(
                    f"{where}.run_status {run_status} requires at least two comparison runs"
                )
            if coverage == "direct":
                invalid_runs = [
                    run_id
                    for run_id in comparison_runs
                    if records.get(run_id, {}).get("status") not in {"keep", "discard"}
                    or _safe_score(records.get(run_id, {}).get("final_best_score")) is None
                ]
                if invalid_runs:
                    errors.append(
                        f"{where}.direct comparison runs must be scored non-crashes; "
                        f"invalid {invalid_runs}"
                    )

        for output_field, lineage_field in relation_fields.items():
            actual_ids = entry.get(output_field)
            if not isinstance(actual_ids, list) or any(
                not isinstance(run_id, str) for run_id in actual_ids
            ):
                errors.append(f"{where}.{output_field} must be a list of run-id strings")
                continue
            expected_ids = [item["run_id"] for item in expected_direction[lineage_field]]
            if sorted(actual_ids) != sorted(expected_ids):
                errors.append(
                    f"{where}.{output_field} must match DAG lineage; "
                    f"expected {expected_ids}, got {actual_ids}"
                )

        evidence_edges = entry.get("evidence_edges")
        if not isinstance(evidence_edges, list) or any(
            not isinstance(edge, str) for edge in evidence_edges
        ):
            errors.append(f"{where}.evidence_edges must be a list of parent->child strings")
        else:
            unknown_edges = sorted(set(evidence_edges) - ledger_edges)
            if unknown_edges:
                errors.append(f"{where}.evidence_edges contains unknown edges {unknown_edges}")

        related_ids = {
            item["run_id"]
            for field in relation_fields.values()
            for item in expected_direction[field]
        }
        terminal = {
            run_id
            for run_id in related_ids
            if records.get(run_id, {}).get("status") in {"keep", "discard", "crash"}
        }
        noncrash = {
            run_id
            for run_id in terminal
            if records.get(run_id, {}).get("status") != "crash"
        }
        run_status = entry.get("run_status")
        if not terminal and run_status != "untested":
            errors.append(f"{where}.run_status must be untested before any completed evidence")
        if terminal and run_status == "untested":
            errors.append(f"{where}.run_status cannot be untested after completed evidence")
        if not noncrash and run_status in {"supported_here", "contradicted_here", "mixed"}:
            errors.append(f"{where}.run_status {run_status} requires non-crash evidence")

    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"direction_evidence is missing {missing}")
    return errors


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return data


def cmd_validate(args: argparse.Namespace) -> int:
    registry = load_registry(args.background)
    ledger = _load_json(args.ledger) if args.ledger else None
    retrieval_manifest = _load_json(args.retrieval_manifest) if args.retrieval_manifest else None
    errors = validate_registry(
        registry,
        ledger=ledger,
        retrieval_manifest=retrieval_manifest,
    )
    errors.extend(validate_background_markdown(args.background, registry))
    result = {
        "ok": not errors,
        "schema_version": registry.get("schema_version"),
        "scope_contract": (
            "explicit"
            if registry.get("schema_version") == SCHEMA_VERSION
            else "legacy_unspecified"
        ),
        "directions": len(registry.get("directions", [])),
        "guidance": len(registry.get("guidance", [])),
        "sources": len(registry.get("sources", [])),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


def cmd_lineage(args: argparse.Namespace) -> int:
    registry = load_registry(args.background)
    ledger = _load_json(args.ledger)
    errors = validate_registry(registry, ledger=ledger)
    errors.extend(validate_background_markdown(args.background, registry))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps(derive_lineage(registry, ledger), indent=2))
    return 0


def _consumed_direction_ids(ledger: dict[str, Any] | None) -> set[str]:
    return {
        source
        for record in (ledger or {}).get("records", [])
        for source in (record.get("source_run_ids") or [])
        if isinstance(source, str) and source.startswith("tf-")
    }


def cmd_preflight(args: argparse.Namespace) -> int:
    """Return the small background-maintenance action owned by the orchestrator."""
    registry = load_registry(args.background)
    ledger = _load_json(args.ledger) if args.ledger else None
    errors = validate_registry(registry, ledger=ledger)
    errors.extend(validate_background_markdown(args.background, registry))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, separators=(",", ":")))
        return 1
    consumed = _consumed_direction_ids(ledger)
    direction_ids = {
        direction.get("id")
        for direction in registry.get("directions", [])
        if isinstance(direction, dict) and isinstance(direction.get("id"), str)
    }
    refresh_legacy = (
        registry.get("schema_version") == 1
        and bool(direction_ids)
        and direction_ids.issubset(consumed)
    )
    print(json.dumps({
        "ok": True,
        "schema_version": registry.get("schema_version"),
        "action": "refresh_background" if refresh_legacy else "none",
        "reason": "legacy_directions_exhausted" if refresh_legacy else None,
    }, separators=(",", ":")))
    return 0


def cmd_directions(args: argparse.Namespace) -> int:
    """Print the compact hypothesis view used for fresh-candidate ideation."""
    registry = load_registry(args.background)
    ledger = _load_json(args.ledger) if args.ledger else None
    errors = validate_registry(registry, ledger=ledger)
    errors.extend(validate_background_markdown(args.background, registry))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, separators=(",", ":")))
        return 1
    consumed = _consumed_direction_ids(ledger)
    scoped_contract = registry.get("schema_version") == SCHEMA_VERSION
    selection = derive_direction_selection(registry, ledger) if scoped_contract else {}
    directions = []
    excluded = []
    eligible_count = 0
    for priority, direction in enumerate(registry.get("directions", [])):
        direction_id = direction.get("id")
        decision = selection.get(direction_id, {
            "selection_status": "active",
            "matched_guidance": [],
            "binding_guidance": [],
            "reopened_by_run_status": None,
        })
        if decision["selection_status"] == "excluded":
            excluded.append({
                "id": direction_id,
                "title": direction.get("title"),
                "binding_guidance": decision["binding_guidance"],
            })
            continue
        eligible_count += 1
        if args.unconsumed and direction_id in consumed:
            continue
        item = {
            "id": direction_id,
            "title": direction.get("title"),
            "claim": direction.get("claim"),
            "literature_credibility": direction.get("literature_credibility"),
            "testable_expectation": direction.get("testable_expectation"),
            "selection_status": decision["selection_status"],
            "matched_guidance": decision["matched_guidance"],
            "binding_guidance": decision["binding_guidance"],
            "reopened_by_run_status": decision["reopened_by_run_status"],
            "_priority": priority,
        }
        if scoped_contract:
            item.update({
                "kind": direction.get("kind"),
                "probe_for": direction.get("probe_for", []),
                "claim_scope": direction.get("claim_scope"),
                "scope": direction.get("scope"),
                "required_comparisons": direction.get("required_comparisons"),
                "reopen_when": direction.get("reopen_when"),
            })
        directions.append(item)
    directions.sort(
        key=lambda item: (
            0 if item["selection_status"] == "active" else 1,
            item["_priority"],
        )
    )
    for item in directions:
        item.pop("_priority", None)
    print(json.dumps({
        "ok": True,
        "schema_version": registry.get("schema_version"),
        "scope_contract": (
            "explicit"
            if registry.get("schema_version") == SCHEMA_VERSION
            else "legacy_unspecified"
        ),
        "consumed": sorted(consumed),
        "all_consumed": bool(eligible_count) and not directions,
        "excluded": excluded,
        "directions": directions,
    }, separators=(",", ":")))
    return 0


def cmd_validate_experience(args: argparse.Namespace) -> int:
    registry = load_registry(args.background)
    ledger = _load_json(args.ledger)
    if args.experience:
        experience = _load_json(args.experience)
    else:
        experience = ledger.get("experience")
        if not isinstance(experience, dict):
            print(
                json.dumps(
                    {"ok": False, "errors": ["ledger has no object-valued experience block"]},
                    indent=2,
                )
            )
            return 1
    errors = validate_registry(registry, ledger=ledger)
    errors.extend(validate_background_markdown(args.background, registry))
    if not errors:
        errors.extend(validate_experience(experience, registry, ledger))
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a background registry")
    validate.add_argument("--background", type=Path, required=True)
    validate.add_argument("--ledger", type=Path)
    validate.add_argument("--retrieval-manifest", type=Path)
    validate.set_defaults(func=cmd_validate)

    lineage = sub.add_parser("lineage", help="join tf-* directions to candidate lineage")
    lineage.add_argument("--background", type=Path, required=True)
    lineage.add_argument("--ledger", type=Path, required=True)
    lineage.set_defaults(func=cmd_lineage)

    directions = sub.add_parser(
        "directions", help="print compact tf-* hypotheses without source metadata"
    )
    directions.add_argument("--background", type=Path, required=True)
    directions.add_argument("--ledger", type=Path)
    directions.add_argument("--unconsumed", action="store_true")
    directions.set_defaults(func=cmd_directions)

    preflight = sub.add_parser(
        "preflight", help="print the orchestrator-owned background maintenance action"
    )
    preflight.add_argument("--background", type=Path, required=True)
    preflight.add_argument("--ledger", type=Path)
    preflight.set_defaults(func=cmd_preflight)

    validate_experience_parser = sub.add_parser(
        "validate-experience", help="validate tf-* run statuses against DAG lineage"
    )
    validate_experience_parser.add_argument("--background", type=Path, required=True)
    validate_experience_parser.add_argument("--ledger", type=Path, required=True)
    validate_experience_parser.add_argument(
        "--experience",
        type=Path,
        help="experience JSON file; omit to validate ledger's top-level experience block",
    )
    validate_experience_parser.set_defaults(func=cmd_validate_experience)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
