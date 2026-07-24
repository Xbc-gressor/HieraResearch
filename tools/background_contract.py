#!/usr/bin/env python3
"""Validate and render the P1 hierarchical ``background.md`` contract.

The Markdown document is the human view.  Its fenced ``Search space
registry`` JSON object is the machine contract shared by both runtimes.  This
module validates literature receipts and typed guidance around the structural
contract owned by :mod:`semantic_space`.

Pre-P1 flat ``tf-*`` registries are intentionally rejected.  Runs are local
and disposable, so there is no implicit migration or mixed-mode behavior.
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
from semantic_space import (
    DEFAULT_DIMENSION_STRATEGY,
    SemanticSpaceError,
    catalog_receipt,
    complete_point,
    coverage_from_records,
    derive_semantic_lineage,
    hypothesis_map,
    load_catalog,
    point_id,
    resolve_dimension_catalog,
    resolve_dimension_strategy,
    selected_assignments,
    space_receipt,
    space_revision,
    validate_point,
    validate_space_core,
)


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
SCOPE_FACETS = (
    "model_families",
    "data_regimes",
    "metrics",
    "interventions",
    "evaluation_protocols",
)
GUIDANCE_SECTIONS = {"pitfall", "deprioritize"}
GUIDANCE_EFFECTS = {"caution", "deprioritize", "exclude"}
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
POLICY_CONFIG_KEYS = {"coverage_weight", "cost_weight", "uncertainty_weight"}


class ContractError(ValueError):
    """A malformed or incompatible background contract."""


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def load_registry(path: Path) -> dict[str, Any]:
    """Extract the canonical semantic-search-space registry from Markdown."""
    text = path.read_text(errors="replace")
    marker = re.search(r"^## Search space registry\s*$", text, flags=re.MULTILINE)
    if marker is None:
        if re.search(r"^## Direction registry\s*$", text, flags=re.MULTILINE):
            raise ContractError(
                f"{path}: legacy flat 'Direction registry' is unsupported; "
                "regenerate a schema_version 3 hierarchical search space"
            )
        raise ContractError(f"{path}: missing '## Search space registry'")
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text[marker.end() :], flags=re.DOTALL)
    if fence is None:
        raise ContractError(f"{path}: search space registry must be a fenced JSON object")
    try:
        registry = json.loads(fence.group(1))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path}: invalid search space registry JSON: {exc}") from exc
    if not isinstance(registry, dict):
        raise ContractError(f"{path}: search space registry must be an object")
    return registry


def _validate_scope(scope: Any, where: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(scope, dict):
        return [f"{where} must be an object"]
    unknown_facets = sorted(set(scope) - set(SCOPE_FACETS))
    if unknown_facets:
        errors.append(f"{where} has unknown scope facets {unknown_facets}")
    for facet in SCOPE_FACETS:
        values = scope.get(facet)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str) or SCOPE_TAG_RE.fullmatch(value) is None
                for value in values
            )
        ):
            errors.append(f"{where}.{facet} must be a non-empty list of lowercase scope tags")
        elif len(values) != len(set(values)):
            errors.append(f"{where}.{facet} must not contain duplicate tags")
        elif "*" in values and len(values) != 1:
            errors.append(f"{where}.{facet} wildcard must be the only tag")
    return errors


def scope_relation(claim_scope: Any, target_scope: Any) -> str:
    """Return direct, partial, mismatch, or unknown for claim -> target scope."""
    if _validate_scope(claim_scope, "claim_scope") or _validate_scope(
        target_scope, "target_scope"
    ):
        return "unknown"
    direct = True
    for facet in SCOPE_FACETS:
        claim_values = set(claim_scope[facet])
        target_values = set(target_scope[facet])
        if "*" in claim_values:
            continue
        if claim_values.isdisjoint(target_values):
            return "mismatch"
        if not target_values.issubset(claim_values):
            direct = False
    return "direct" if direct else "partial"


def _evidence_link_errors(
    links: Any,
    where: str,
    source_ids: set[str],
    *,
    allow_empty: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    if not isinstance(links, list) or (not allow_empty and not links):
        return [f"{where} must be {'a' if allow_empty else 'a non-empty'} list"], []
    valid: list[dict[str, Any]] = []
    for index, link in enumerate(links):
        item_where = f"{where}[{index}]"
        if not isinstance(link, dict):
            errors.append(f"{item_where} must be an object")
            continue
        if link.get("source_id") not in source_ids:
            errors.append(f"{item_where} references unknown source id {link.get('source_id')!r}")
        if link.get("role") not in EVIDENCE_ROLES:
            errors.append(f"{item_where}.role must be one of {sorted(EVIDENCE_ROLES)}")
        unknown = sorted(set(link) - {"source_id", "role"})
        if unknown:
            errors.append(f"{item_where} has unknown fields {unknown}")
        valid.append(link)
    return errors, valid


def _validate_sources(
    registry: dict[str, Any], retrieval_manifest: dict[str, Any] | None
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    errors: list[str] = []
    sources = registry.get("sources")
    if not isinstance(sources, list):
        return ["sources must be a list"], {}
    source_by_id: dict[str, dict[str, Any]] = {}
    source_keys: dict[str, str] = {}
    source_urls: dict[str, str] = {}
    for index, source in enumerate(sources):
        where = f"sources[{index}]"
        if not isinstance(source, dict):
            errors.append(f"{where} must be an object")
            continue
        source_id = source.get("id")
        expected_id = f"src-{index + 1:02d}"
        if source_id != expected_id:
            errors.append(f"{where}.id must be {expected_id} (order, no gaps)")
        if isinstance(source_id, str):
            if source_id in source_by_id:
                errors.append(f"duplicate source id {source_id}")
            else:
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
            if not _nonempty(source.get(field)):
                errors.append(f"{where}.{field} must be a non-empty string")
        errors.extend(_validate_scope(source.get("studied_scope"), f"{where}.studied_scope"))
        url = source.get("url")
        if _nonempty(url):
            if not url.startswith(("https://", "http://")):
                errors.append(f"{where}.url must be an HTTP(S) URL")
            else:
                source_urls[str(source_id)] = url
                key = canonical_key(url)
                previous = source_keys.get(key)
                if previous is not None:
                    errors.append(
                        f"sources {previous} and {source_id} duplicate canonical work {key}"
                    )
                else:
                    source_keys[key] = str(source_id)
        unknown = sorted(
            set(source)
            - {"id", "type", "title", "url", "publication_status", "validation_status", "studied_scope"}
        )
        if unknown:
            errors.append(f"{where} has unknown fields {unknown}")

    if retrieval_manifest is not None:
        errors.extend(validate_manifest(retrieval_manifest))
        if retrieval_manifest.get("retrieval_condition") == "mixed":
            errors.append("background evidence cannot mix frozen and live retrieval in one condition")
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
    return errors, source_by_id


def _validate_query_dimension_coverage(
    registry: dict[str, Any], retrieval_manifest: dict[str, Any] | None
) -> list[str]:
    if retrieval_manifest is None:
        return []
    dimensions = registry.get("dimensions")
    if not isinstance(dimensions, list):
        return []
    dimension_by_id = {
        dimension.get("id"): dimension
        for dimension in dimensions
        if isinstance(dimension, dict) and isinstance(dimension.get("id"), str)
    }
    known_dimension_ids = set(dimension_by_id)
    grounding_coverage: set[str] = set()
    errors: list[str] = []

    queries = retrieval_manifest.get("queries")
    if isinstance(queries, list):
        for index, query in enumerate(queries):
            if not isinstance(query, dict):
                continue
            targets = query.get("target_dimension_ids")
            if not isinstance(targets, list):
                continue
            valid_targets = {item for item in targets if isinstance(item, str)}
            unknown = sorted(valid_targets - known_dimension_ids)
            if unknown:
                errors.append(
                    f"retrieval query {query.get('id', index)!r} targets unknown registry "
                    f"dimensions {unknown}"
                )
            if query.get("lane") == "grounding":
                grounding_coverage.update(valid_targets & known_dimension_ids)

    exempted: set[str] = set()
    exemptions = retrieval_manifest.get("coverage_exemptions")
    if isinstance(exemptions, list):
        for index, exemption in enumerate(exemptions):
            if not isinstance(exemption, dict):
                continue
            dimension_id = exemption.get("dimension_id")
            if not isinstance(dimension_id, str):
                continue
            if dimension_id not in known_dimension_ids:
                errors.append(
                    f"retrieval coverage exemption {index} names unknown registry dimension "
                    f"{dimension_id!r}"
                )
            else:
                exempted.add(dimension_id)

    uncovered = sorted(
        dimension_id
        for dimension_id, dimension in dimension_by_id.items()
        if dimension.get("mode") == "searchable"
        and dimension_id not in grounding_coverage
        and dimension_id not in exempted
    )
    if uncovered:
        errors.append(
            "searchable registry dimensions lack a grounding query or coverage exemption: "
            f"{uncovered}"
        )
    return errors


def _credibility_errors(
    item: dict[str, Any],
    links: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
    where: str,
) -> list[str]:
    errors: list[str] = []
    credibility = item.get("literature_credibility")
    if credibility not in LITERATURE_CREDIBILITY:
        errors.append(
            f"{where}.literature_credibility must be one of {sorted(LITERATURE_CREDIBILITY)}"
        )
    roles = [link.get("role") for link in links]
    if credibility == "contested" and "contradicts" not in roles:
        errors.append(f"{where} is contested but cites no contradicting evidence")
    if credibility == "replicated":
        replicated_support = any(
            link.get("role") == "supports"
            and source_by_id.get(link.get("source_id"), {}).get("validation_status")
            == "independently_reproduced"
            for link in links
        )
        if not replicated_support:
            errors.append(
                f"{where} is replicated but has no independently reproduced supporting source"
            )
    return errors


def _validate_hypotheses(
    registry: dict[str, Any], source_by_id: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    source_ids = set(source_by_id)
    dimensions = registry.get("dimensions")
    if not isinstance(dimensions, list):
        return errors
    for dimension_index, dimension in enumerate(dimensions):
        if not isinstance(dimension, dict):
            continue
        for hypothesis_index, hypothesis in enumerate(dimension.get("hypotheses", [])):
            if not isinstance(hypothesis, dict):
                continue
            where = f"dimensions[{dimension_index}].hypotheses[{hypothesis_index}]"
            errors.extend(_validate_scope(hypothesis.get("scope"), f"{where}.scope"))
            allow_empty = hypothesis.get("kind") == "baseline"
            link_errors, links = _evidence_link_errors(
                hypothesis.get("evidence"),
                f"{where}.evidence",
                source_ids,
                allow_empty=allow_empty,
            )
            errors.extend(link_errors)
            errors.extend(_credibility_errors(hypothesis, links, source_by_id, where))
    return errors


def _validate_relations_evidence(
    registry: dict[str, Any], source_by_id: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    source_ids = set(source_by_id)
    relations = registry.get("relations")
    if not isinstance(relations, list):
        return errors
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            continue
        link_errors, _ = _evidence_link_errors(
            relation.get("evidence"),
            f"relations[{index}].evidence",
            source_ids,
            allow_empty=True,
        )
        errors.extend(link_errors)
    return errors


def _validate_provenance_refs(
    registry: dict[str, Any], source_by_id: dict[str, dict[str, Any]]
) -> list[str]:
    """Make literature provenance receipts resolve to inspected sources."""
    errors: list[str] = []
    source_ids = set(source_by_id)

    def check(receipts: Any, where: str) -> None:
        if not isinstance(receipts, list):
            return
        for index, receipt in enumerate(receipts):
            if (
                isinstance(receipt, dict)
                and receipt.get("kind") == "literature"
                and receipt.get("ref") not in source_ids
            ):
                errors.append(
                    f"{where}[{index}].ref must name an inspected source id; "
                    f"got {receipt.get('ref')!r}"
                )

    dimensions = registry.get("dimensions")
    if isinstance(dimensions, list):
        for dimension_index, dimension in enumerate(dimensions):
            if not isinstance(dimension, dict):
                continue
            check(dimension.get("evidence"), f"dimensions[{dimension_index}].evidence")
            for hypothesis_index, hypothesis in enumerate(dimension.get("hypotheses", [])):
                if isinstance(hypothesis, dict):
                    check(
                        hypothesis.get("provenance"),
                        f"dimensions[{dimension_index}].hypotheses[{hypothesis_index}].provenance",
                    )
    relations = registry.get("relations")
    if isinstance(relations, list):
        for relation_index, relation in enumerate(relations):
            if isinstance(relation, dict):
                check(relation.get("provenance"), f"relations[{relation_index}].provenance")
    return errors


def _validate_guidance(
    registry: dict[str, Any], source_by_id: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    guidance = registry.get("guidance")
    if not isinstance(guidance, list):
        return ["guidance must be a list"]
    guidance_ids = {
        item.get("id") for item in guidance if isinstance(item, dict)
    }
    for dimension_index, dimension in enumerate(registry.get("dimensions", [])):
        if not isinstance(dimension, dict):
            continue
        for hypothesis_index, hypothesis in enumerate(dimension.get("hypotheses", [])):
            if not isinstance(hypothesis, dict) or hypothesis.get("kind") != "scope_probe":
                continue
            where = f"dimensions[{dimension_index}].hypotheses[{hypothesis_index}]"
            for guidance_id in hypothesis.get("probe_for") or []:
                if guidance_id not in guidance_ids:
                    errors.append(
                        f"{where}.probe_for references unknown guidance id {guidance_id!r}"
                    )
    source_ids = set(source_by_id)
    hypotheses = hypothesis_map(registry)
    for index, item in enumerate(guidance):
        where = f"guidance[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be an object")
            continue
        guidance_id = item.get("id")
        expected_id = f"g-{index + 1:02d}"
        if guidance_id != expected_id:
            errors.append(f"{where}.id must be {expected_id} (order, no gaps)")
        if item.get("section") not in GUIDANCE_SECTIONS:
            errors.append(f"{where}.section must be one of {sorted(GUIDANCE_SECTIONS)}")
        effect = item.get("effect")
        if effect not in GUIDANCE_EFFECTS:
            errors.append(f"{where}.effect must be one of {sorted(GUIDANCE_EFFECTS)}")
        for field in ("claim", "credibility_rationale", "reopen_when"):
            if not _nonempty(item.get(field)):
                errors.append(f"{where}.{field} must be a non-empty string")
        errors.extend(_validate_scope(item.get("scope"), f"{where}.scope"))
        link_errors, links = _evidence_link_errors(
            item.get("evidence"), f"{where}.evidence", source_ids, allow_empty=False
        )
        errors.extend(link_errors)
        errors.extend(_credibility_errors(item, links, source_by_id, where))
        credibility = item.get("literature_credibility")
        direct_support_ids = {
            link.get("source_id")
            for link in links
            if link.get("role") == "supports"
            and source_by_id.get(link.get("source_id"), {}).get("type") in BINDING_SOURCE_TYPES
            and source_by_id.get(link.get("source_id"), {}).get("publication_status")
            != "withdrawn_or_retracted"
            and scope_relation(
                source_by_id.get(link.get("source_id"), {}).get("studied_scope"),
                item.get("scope"),
            )
            == "direct"
        }
        if effect in {"deprioritize", "exclude"}:
            if credibility in {"unverified", "contested"}:
                errors.append(
                    f"{where} {credibility} negative guidance may only caution"
                )
            if not direct_support_ids:
                errors.append(
                    f"{where} binding guidance requires a directly scoped primary empirical source"
                )
            source_contains_guidance = any(
                scope_relation(source.get("studied_scope"), item.get("scope")) == "direct"
                for source in source_by_id.values()
                if source.get("type") in BINDING_SOURCE_TYPES
                and source.get("publication_status") != "withdrawn_or_retracted"
            )
            if not source_contains_guidance:
                errors.append(
                    f"{where} no eligible source directly contains the guidance scope"
                )
            probes = [
                hypothesis
                for _, hypothesis in hypotheses.values()
                if hypothesis.get("kind") == "scope_probe"
                and guidance_id in (hypothesis.get("probe_for") or [])
                and scope_relation(item.get("scope"), hypothesis.get("scope")) != "direct"
            ]
            if not probes:
                errors.append(
                    f"{where} binding guidance requires an out-of-scope scope_probe hypothesis"
                )
        if effect == "exclude":
            if credibility not in {"corroborated", "replicated"}:
                errors.append(f"{where} exclusion requires corroborated or replicated evidence")
            if len(direct_support_ids) < 2:
                errors.append(f"{where} exclusion requires two directly scoped supporting sources")
            reproduced = any(
                source_by_id.get(source_id, {}).get("validation_status")
                == "independently_reproduced"
                for source_id in direct_support_ids
            )
            if not reproduced:
                errors.append(f"{where} exclusion requires directly scoped independent reproduction")
        unknown = sorted(
            set(item)
            - {
                "id",
                "section",
                "effect",
                "claim",
                "scope",
                "literature_credibility",
                "credibility_rationale",
                "reopen_when",
                "evidence",
            }
        )
        if unknown:
            errors.append(f"{where} has unknown fields {unknown}")
    return errors


def derive_hypothesis_selection(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Derive external-guidance eligibility without mutating the frozen space."""
    guidance = [item for item in registry.get("guidance", []) if isinstance(item, dict)]
    result: dict[str, dict[str, Any]] = {}
    for hypothesis_id, (_, hypothesis) in hypothesis_map(registry).items():
        matched: list[dict[str, str]] = []
        binding: list[dict[str, str]] = []
        for item in guidance:
            if scope_relation(item.get("scope"), hypothesis.get("scope")) != "direct":
                continue
            receipt = {"id": item.get("id"), "effect": item.get("effect")}
            matched.append(receipt)
            if item.get("effect") in {"deprioritize", "exclude"}:
                binding.append(receipt)
        status = "active"
        if any(item["effect"] == "exclude" for item in binding):
            status = "excluded"
        elif any(item["effect"] == "deprioritize" for item in binding):
            status = "deprioritized"
        result[hypothesis_id] = {
            "selection_status": status,
            "matched_guidance": matched,
            "binding_guidance": binding,
        }
    return result


def validate_candidate_point(point: Any, registry: dict[str, Any]) -> list[str]:
    """Validate one structural point plus P1 guidance-derived eligibility."""
    errors = validate_point(point, registry)
    if errors or not isinstance(point, dict):
        return errors
    selection = derive_hypothesis_selection(registry)
    excluded = sorted(
        hypothesis_id
        for hypothesis_id in selected_assignments(point).values()
        if selection.get(hypothesis_id, {}).get("selection_status") == "excluded"
    )
    if excluded:
        errors.append(f"semantic_point selects guidance-excluded hypotheses {excluded}")
    return errors


def _validate_policy_receipt(record: dict[str, Any], where: str) -> list[str]:
    receipt = record.get("policy_receipt")
    point = record.get("semantic_point")
    if not isinstance(receipt, dict):
        return [f"{where}.policy_receipt must be an object distinct from observations"]
    point_object = point if isinstance(point, dict) else {}
    errors: list[str] = []
    if receipt.get("schema_version") != 1:
        errors.append(f"{where}.policy_receipt.schema_version must be 1")
    if receipt.get("space_revision") != point_object.get("space_revision"):
        errors.append(f"{where}.policy_receipt.space_revision must match semantic_point")
    proposal_revision = receipt.get("proposal_set_revision")
    if not isinstance(proposal_revision, str) or DIGEST_RE.fullmatch(proposal_revision) is None:
        errors.append(f"{where}.policy_receipt.proposal_set_revision must be a sha256 digest")
    if receipt.get("selected_point_id") != point_object.get("point_id"):
        errors.append(f"{where}.policy_receipt.selected_point_id must match semantic_point")
    action = receipt.get("action")
    raw_parents = record.get("source_run_ids")
    expected_parents = [str(item) for item in raw_parents] if isinstance(raw_parents, list) else []
    if not isinstance(action, dict) or set(action) != {"op", "parents"}:
        errors.append(f"{where}.policy_receipt.action must contain only op and parents")
    elif action.get("op") != record.get("op"):
        errors.append(f"{where}.policy_receipt.action.op must match record.op")
    elif action.get("parents") != expected_parents:
        errors.append(f"{where}.policy_receipt.action.parents must match numeric ancestry")
    policy = receipt.get("policy")
    policy_name = policy.get("name") if isinstance(policy, dict) else None
    if not isinstance(policy, dict) or set(policy) != {"name", "config"}:
        errors.append(f"{where}.policy_receipt.policy must contain only name and config")
    elif policy_name not in {
        "coverage",
        "gain",
        "gain_uncertainty",
        "gain_uncertainty_nocost",
    }:
        errors.append(
            f"{where}.policy_receipt.policy.name must be coverage, gain, "
            "gain_uncertainty, or gain_uncertainty_nocost"
        )
    config = policy.get("config") if isinstance(policy, dict) else None
    config_valid = True
    if not isinstance(config, dict) or set(config) != POLICY_CONFIG_KEYS:
        errors.append(
            f"{where}.policy_receipt.policy.config must keep exactly "
            f"{sorted(POLICY_CONFIG_KEYS)}"
        )
        config = {}
        config_valid = False
    else:
        for key, value in config.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                config_valid = False
                errors.append(
                    f"{where}.policy_receipt.policy.config.{key} must be a finite "
                    "non-negative number"
                )
    components = receipt.get("components")
    coverage_valid = False
    model_components_valid = False
    required_components = {"coverage", "predicted_gain", "uncertainty", "cost"}
    if not isinstance(components, dict) or set(components) != required_components:
        errors.append(
            f"{where}.policy_receipt.components must keep {sorted(required_components)} separate"
        )
        components = {}
    else:
        coverage = components["coverage"]
        coverage_valid = (
            isinstance(coverage, (int, float))
            and not isinstance(coverage, bool)
            and math.isfinite(float(coverage))
            and 0.0 <= float(coverage) <= 1.0
        )
        if not coverage_valid:
            errors.append(f"{where}.policy_receipt coverage must be a finite number in [0, 1]")
        model_components = [
            components["predicted_gain"], components["uncertainty"], components["cost"]
        ]
        if policy_name == "coverage":
            model_components_valid = all(value is None for value in model_components)
            if any(value is not None for value in model_components):
                errors.append(
                    f"{where}.policy_receipt coverage policy must not invent model components"
                )
        elif policy_name == "gain_uncertainty_nocost":
            model_components_valid = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0
                for value in (components["predicted_gain"], components["uncertainty"])
            ) and components["cost"] is None
            if not model_components_valid:
                errors.append(
                    f"{where}.policy_receipt gain_uncertainty_nocost requires separate "
                    "finite predicted_gain and uncertainty values in [0, 1] and no cost"
                )
        elif policy_name in {"gain", "gain_uncertainty"}:
            model_components_valid = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0
                for value in model_components
            )
            if not model_components_valid:
                errors.append(
                    f"{where}.policy_receipt gain policies require separate finite "
                    "predicted_gain, uncertainty, and cost values in [0, 1]"
                )
    evidence = receipt.get("evidence")
    if (
        not isinstance(evidence, list)
        or len(evidence) > 5
        or any(not _nonempty(item) or len(item) > 240 for item in evidence)
    ):
        errors.append(
            f"{where}.policy_receipt.evidence must contain at most five short strings"
        )
        evidence = []
    if policy_name == "coverage" and evidence:
        errors.append(f"{where}.policy_receipt coverage policy evidence must be empty")
    if policy_name in {"gain", "gain_uncertainty", "gain_uncertainty_nocost"} and not evidence:
        errors.append(f"{where}.policy_receipt gain policies require selection evidence")
    acquisition_score = receipt.get("acquisition_score")
    if (
        not isinstance(acquisition_score, (int, float))
        or isinstance(acquisition_score, bool)
        or not math.isfinite(float(acquisition_score))
    ):
        errors.append(f"{where}.policy_receipt.acquisition_score must be finite numeric")
    elif (
        components
        and config_valid
        and coverage_valid
        and model_components_valid
        and policy_name in {"coverage", "gain", "gain_uncertainty", "gain_uncertainty_nocost"}
    ):
        coverage = float(components["coverage"])
        if policy_name == "coverage":
            expected_score = coverage
        elif policy_name == "gain_uncertainty_nocost":
            expected_score = (
                float(components["predicted_gain"])
                + float(config["uncertainty_weight"]) * float(components["uncertainty"])
                + float(config["coverage_weight"]) * coverage
            )
        elif all(components[key] is not None for key in ("predicted_gain", "uncertainty", "cost")):
            expected_score = (
                float(components["predicted_gain"])
                + float(config["coverage_weight"]) * coverage
                - float(config["cost_weight"]) * float(components["cost"])
            )
            if policy_name == "gain_uncertainty":
                expected_score += float(config["uncertainty_weight"]) * float(
                    components["uncertainty"]
                )
        else:
            expected_score = None
        if expected_score is not None and not math.isclose(
            float(acquisition_score), round(expected_score, 10), rel_tol=0.0, abs_tol=1e-9
        ):
            errors.append(
                f"{where}.policy_receipt.acquisition_score does not match its separate components"
            )
    ranked = receipt.get("ranked_point_ids")
    if (
        not isinstance(ranked, list)
        or not ranked
        or len(ranked) > 128
        or any(not isinstance(item, str) for item in ranked)
        or len(ranked) != len(set(ranked))
        or ranked[0] != receipt.get("selected_point_id")
    ):
        errors.append(
            f"{where}.policy_receipt.ranked_point_ids must be unique and start with the selected point"
        )
    allowed = {
        "schema_version",
        "space_revision",
        "proposal_set_revision",
        "policy",
        "action",
        "selected_point_id",
        "components",
        "acquisition_score",
        "evidence",
        "ranked_point_ids",
    }
    unknown = sorted(set(receipt) - allowed)
    if unknown:
        errors.append(f"{where}.policy_receipt has unknown fields {unknown}")
    return errors


def validate_ledger(registry: dict[str, Any], ledger: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    records = ledger.get("records")
    if not isinstance(records, list):
        return ["ledger.records must be a list"]
    expected_receipt = space_receipt(registry)
    if (records or ledger.get("search_space") is not None) and ledger.get(
        "search_space"
    ) != expected_receipt:
        errors.append(
            "ledger.search_space must preserve the exact background and catalog revision receipt"
        )
    known: set[str] = set()
    for index, record in enumerate(records):
        where = f"ledger.records[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{where} must be an object")
            continue
        raw_run_id = record.get("run_id")
        run_id = str(raw_run_id)
        valid_run_id = isinstance(raw_run_id, str) and raw_run_id.isdigit()
        if not valid_run_id:
            errors.append(f"{where}.run_id must be a numeric string")
        elif run_id in known:
            errors.append(f"{where}.run_id duplicates an earlier record")
        if record.get("kind") != "optimization":
            errors.append(f"{where}.kind must be optimization")
        for field in ("idea", "change", "candidate_name", "description"):
            if not _nonempty(record.get(field)):
                errors.append(f"{where}.{field} must be a non-empty candidate field")
        op = record.get("op")
        parents = record.get("source_run_ids")
        if not isinstance(parents, list) or any(
            not isinstance(parent, str) or not parent.isdigit() for parent in parents
        ):
            errors.append(
                f"{where}.source_run_ids must contain only numeric parent ids; "
                "hypothesis attribution belongs in semantic_point"
            )
            parents = []
        if any(parent not in known for parent in parents):
            errors.append(f"{where}.source_run_ids must reference earlier records")
        expected_parent_count = {"fresh": 0, "improve": 1, "crossover": 2}.get(op)
        if expected_parent_count is None:
            errors.append(f"{where}.op must be fresh, improve, or crossover")
        elif len(parents) != expected_parent_count or len(parents) != len(set(parents)):
            errors.append(
                f"{where} {op} requires {expected_parent_count} distinct numeric parents"
            )
        point_errors = validate_candidate_point(record.get("semantic_point"), registry)
        errors.extend(f"{where}: {error}" for error in point_errors)
        errors.extend(_validate_policy_receipt(record, where))
        if valid_run_id:
            known.add(run_id)
    experience = ledger.get("experience")
    if experience not in (None, {}):
        errors.extend(validate_experience(experience, registry, ledger))
    return errors


def validate_registry(
    registry: dict[str, Any],
    *,
    ledger: dict[str, Any] | None = None,
    retrieval_manifest: dict[str, Any] | None = None,
    catalog: dict[str, Any] | None = None,
    dimension_strategy: str = DEFAULT_DIMENSION_STRATEGY,
) -> list[str]:
    errors = validate_space_core(
        registry, catalog=catalog, dimension_strategy=dimension_strategy
    )
    source_errors, source_by_id = _validate_sources(registry, retrieval_manifest)
    errors.extend(source_errors)
    errors.extend(_validate_query_dimension_coverage(registry, retrieval_manifest))
    errors.extend(_validate_hypotheses(registry, source_by_id))
    errors.extend(_validate_relations_evidence(registry, source_by_id))
    errors.extend(_validate_provenance_refs(registry, source_by_id))
    errors.extend(_validate_guidance(registry, source_by_id))
    if not errors:
        selection = derive_hypothesis_selection(registry)
        for dimension in registry.get("dimensions", []):
            if not isinstance(dimension, dict):
                continue
            baseline_id = dimension.get("baseline_hypothesis_id")
            status = selection.get(baseline_id, {}).get("selection_status")
            if status != "active":
                errors.append(
                    f"dimension {dimension.get('id')} baseline {baseline_id} is {status}; "
                    "the frozen space would have no unconditional valid baseline"
                )
        if complete_point(registry) is None:
            errors.append(
                "the frozen registry has no valid explicit-baseline point under its relations"
            )
    if ledger is not None:
        errors.extend(validate_ledger(registry, ledger))
    return errors


def validate_background_markdown(path: Path, registry: dict[str, Any]) -> list[str]:
    """Keep the human hierarchy and structured guidance aligned with JSON."""
    text = path.read_text(errors="replace")
    marker = re.search(r"^## Search space registry\s*$", text, flags=re.MULTILINE)
    human = text[: marker.start()] if marker else text
    errors: list[str] = []
    for heading in ("Dimension coverage", "Dimensions", "Relations", "Pitfalls"):
        if re.search(rf"^## {re.escape(heading)}\s*$", human, flags=re.MULTILINE) is None:
            errors.append(f"background.md is missing required '## {heading}' section")

    def section_body(heading: str) -> str:
        match = re.search(rf"^## {re.escape(heading)}\s*$", human, flags=re.MULTILINE)
        if match is None:
            return ""
        next_heading = re.search(r"^##\s+", human[match.end() :], flags=re.MULTILINE)
        end = match.end() + next_heading.start() if next_heading else len(human)
        return human[match.end() : end]

    coverage_body = section_body("Dimension coverage")
    dimensions_body = section_body("Dimensions")
    relations_body = section_body("Relations")
    dimensions = registry.get("dimensions")
    if not isinstance(dimensions, list):
        dimensions = []
    known_dimension_ids = {
        str(dimension.get("id")) for dimension in dimensions if isinstance(dimension, dict)
    }
    known_hypothesis_ids = set(hypothesis_map(registry))
    human_dimension_ids = set(re.findall(r"`(dim-[a-z0-9-]+)`", coverage_body + dimensions_body))
    human_hypothesis_ids = set(re.findall(r"`(hyp-[a-z0-9-]+)`", coverage_body + dimensions_body))
    unknown_human_dimensions = sorted(human_dimension_ids - known_dimension_ids)
    unknown_human_hypotheses = sorted(human_hypothesis_ids - known_hypothesis_ids)
    if unknown_human_dimensions:
        errors.append(f"human hierarchy references unknown dimensions {unknown_human_dimensions}")
    if unknown_human_hypotheses:
        errors.append(f"human hierarchy references unknown hypotheses {unknown_human_hypotheses}")
    for dimension in dimensions:
        if not isinstance(dimension, dict):
            continue
        dimension_id = dimension.get("id")
        if f"`{dimension_id}`" not in coverage_body:
            errors.append(f"Dimension coverage does not reference dimension {dimension_id}")
        if f"`{dimension.get('baseline_hypothesis_id')}`" not in coverage_body:
            errors.append(
                f"Dimension coverage does not show the baseline for {dimension_id}"
            )
        dimension_heading = re.search(
            rf"^###\s+`{re.escape(str(dimension_id))}`\s*$",
            dimensions_body,
            flags=re.MULTILINE,
        )
        if dimension_heading is None:
            errors.append(f"Dimensions section is missing hierarchy heading {dimension_id}")
            dimension_section = ""
        else:
            next_dimension = re.search(
                r"^###\s+", dimensions_body[dimension_heading.end() :], flags=re.MULTILINE
            )
            end = (
                dimension_heading.end() + next_dimension.start()
                if next_dimension
                else len(dimensions_body)
            )
            dimension_section = dimensions_body[dimension_heading.end() : end]
        for hypothesis in dimension.get("hypotheses", []):
            if (
                isinstance(hypothesis, dict)
                and f"`{hypothesis.get('id')}`" not in dimension_section
            ):
                errors.append(
                    f"Dimensions hierarchy {dimension_id} does not reference hypothesis "
                    f"{hypothesis.get('id')}"
                )
    relations = registry.get("relations")
    if not isinstance(relations, list):
        relations = []
    for relation in relations:
        if isinstance(relation, dict) and f"`{relation.get('id')}`" not in relations_body:
            errors.append(f"Relations section does not reference relation {relation.get('id')}")
    known_relation_ids = {
        str(relation.get("id")) for relation in relations if isinstance(relation, dict)
    }
    human_relation_ids = set(re.findall(r"`(rel-[a-z0-9-]+)`", relations_body))
    unknown_human_relations = sorted(human_relation_ids - known_relation_ids)
    if unknown_human_relations:
        errors.append(f"Relations section references unknown relations {unknown_human_relations}")

    guidance = {
        item.get("id"): item
        for item in registry.get("guidance", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    referenced: set[str] = set()
    section_names = {"Pitfalls": "pitfall", "Deprioritize": "deprioritize"}
    for heading, expected_section in section_names.items():
        match = re.search(rf"^## {re.escape(heading)}\s*$", human, flags=re.MULTILINE)
        if match is None:
            if any(item.get("section") == expected_section for item in guidance.values()):
                errors.append(f"background.md is missing required '## {heading}' section")
            continue
        next_heading = re.search(r"^##\s+", human[match.end() :], flags=re.MULTILINE)
        end = match.end() + next_heading.start() if next_heading else len(human)
        body = human[match.end() : end]
        has_marked_bullet = False
        for line_number, raw_line in enumerate(body.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if raw_line[:1].isspace() and has_marked_bullet:
                continue
            bullet = re.match(
                r"^-\s+`(?P<id>g-\d{2,}|task-constraint|operational)`(?:\s|$)", line
            )
            if bullet is None:
                has_marked_bullet = False
                errors.append(
                    f"{heading} line {line_number} must start with a registered `g-NN`, "
                    "`task-constraint`, or `operational` marker"
                )
                continue
            has_marked_bullet = True
            marker_id = bullet.group("id")
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


def render_space(
    registry: dict[str, Any], ledger: dict[str, Any] | None, *, max_hypotheses: int
) -> dict[str, Any]:
    coverage = coverage_from_records(registry, (ledger or {}).get("records", []))
    counts = {
        item["hypothesis_id"]: item["count"]
        for dimension in coverage["dimensions"]
        for item in dimension["hypotheses"]
    }
    selection = derive_hypothesis_selection(registry)
    dimensions = []
    for dimension in registry.get("dimensions", []):
        if not isinstance(dimension, dict):
            continue
        hypotheses = []
        for hypothesis in dimension.get("hypotheses", []):
            if not isinstance(hypothesis, dict):
                continue
            hypothesis_id = hypothesis.get("id")
            hypotheses.append(
                {
                    "id": hypothesis_id,
                    "title": hypothesis.get("title"),
                    "kind": hypothesis.get("kind"),
                    "claim": hypothesis.get("claim"),
                    "literature_credibility": hypothesis.get("literature_credibility"),
                    "selection": selection.get(hypothesis_id),
                    "coverage_count": counts.get(hypothesis_id, 0),
                }
            )
        hypotheses.sort(
            key=lambda item: (
                {"active": 0, "deprioritized": 1, "excluded": 2}.get(
                    (item.get("selection") or {}).get("selection_status"), 3
                ),
                item["coverage_count"],
                item["id"],
            )
        )
        dimensions.append(
            {
                "id": dimension.get("id"),
                "mode": dimension.get("mode"),
                "baseline_hypothesis_id": dimension.get("baseline_hypothesis_id"),
                "selection_reason": dimension.get("selection_reason"),
                "hypotheses": hypotheses[:max_hypotheses],
                "omitted_hypotheses": max(0, len(hypotheses) - max_hypotheses),
            }
        )
    return {
        "ok": True,
        "space": space_receipt(registry),
        "coverage": {
            "n_valid_records": coverage["n_valid_records"],
            "n_unique_points": coverage["n_unique_points"],
        },
        "dimensions": dimensions,
        "relations": [
            {
                key: relation.get(key)
                for key in ("id", "type", "when", "then", "target_dimension_id", "members")
                if key in relation
            }
            for relation in registry.get("relations", [])
            if isinstance(relation, dict)
        ],
    }


def validate_experience(experience: Any, registry: dict[str, Any], ledger: dict[str, Any]) -> list[str]:
    """Validate the bounded P1 belief view without implementing P2 semantics."""
    if not isinstance(experience, dict):
        return ["experience must be an object"]
    errors: list[str] = []
    deferred = {"direction_evidence", "dimension_evidence", "hypothesis_evidence"}
    present = sorted(deferred & set(experience))
    if present:
        errors.append(
            f"experience fields {present} are not a P1 contract; two-level belief extraction is P2"
        )
    allowed_top = {
        "schema_version",
        "updated_at_run",
        "generation",
        "summary",
        "promising_regions",
        "lessons",
        "bottlenecks",
        "dag_revision",
    }
    unknown_top = sorted(set(experience) - allowed_top - deferred)
    if unknown_top:
        errors.append(f"experience has unknown fields {unknown_top}")
    if experience.get("schema_version") != 2:
        errors.append("experience.schema_version must be 2")
    generation = experience.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        errors.append("experience.generation must be a non-negative integer")
    if not _nonempty(experience.get("summary")) or len(experience.get("summary", "")) > 2000:
        errors.append(
            "experience.summary must be a non-empty interpretation of at most 2000 characters"
        )

    terminal_records = [
        record
        for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("status") in {"keep", "discard", "crash"}
    ]
    terminal_ids = {str(record.get("run_id")) for record in terminal_records}
    updated_at_run = experience.get("updated_at_run")
    if not isinstance(updated_at_run, str) or updated_at_run not in terminal_ids:
        errors.append("experience.updated_at_run must reference a terminal ledger run")

    ledger_dag_revision = ledger.get("dag_revision", 0)
    valid_ledger_dag_revision = (
        isinstance(ledger_dag_revision, int)
        and not isinstance(ledger_dag_revision, bool)
        and ledger_dag_revision >= 0
    )
    dag_revision = experience.get("dag_revision")
    if dag_revision is not None and (
        not isinstance(dag_revision, int)
        or isinstance(dag_revision, bool)
        or dag_revision < 0
        or not valid_ledger_dag_revision
        or dag_revision > ledger_dag_revision
    ):
        errors.append(
            "experience.dag_revision must be a valid helper-owned cursor at or before the ledger revision"
        )

    def validate_items(
        field: str,
        *,
        limit: int,
        required: set[str],
        optional: set[str] | None = None,
    ) -> None:
        items = experience.get(field)
        if not isinstance(items, list):
            errors.append(f"experience.{field} must be a list")
            return
        if len(items) > limit:
            errors.append(f"experience.{field} may contain at most {limit} items")
        optional_fields = optional or set()
        for index, item in enumerate(items):
            where = f"experience.{field}[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{where} must be an object")
                continue
            missing = sorted(required - set(item))
            unknown = sorted(set(item) - required - optional_fields)
            if missing:
                errors.append(f"{where} is missing fields {missing}")
            if unknown:
                errors.append(f"{where} has unknown fields {unknown}")
            if not _nonempty(item.get("claim")) or len(item.get("claim", "")) > 600:
                errors.append(f"{where}.claim must be non-empty and at most 600 characters")
            evidence = item.get("evidence")
            if (
                not isinstance(evidence, list)
                or not evidence
                or len(evidence) > 5
                or any(not isinstance(run_id, str) or run_id not in terminal_ids for run_id in evidence)
                or len(evidence) != len(set(evidence))
            ):
                errors.append(
                    f"{where}.evidence must contain 1–5 unique terminal ledger run ids"
                )
            if item.get("confidence") not in {"low", "med", "high"}:
                errors.append(f"{where}.confidence must be low, med, or high")

    validate_items(
        "promising_regions",
        limit=8,
        required={"claim", "evidence", "confidence", "uncertainty"},
    )
    promising_regions = experience.get("promising_regions")
    for index, item in enumerate(promising_regions if isinstance(promising_regions, list) else []):
        if isinstance(item, dict) and (
            not _nonempty(item.get("uncertainty"))
            or len(item.get("uncertainty", "")) > 600
        ):
            errors.append(
                f"experience.promising_regions[{index}].uncertainty must be non-empty "
                "and at most 600 characters"
            )

    validate_items(
        "lessons",
        limit=12,
        required={"kind", "claim", "evidence", "confidence"},
        optional={"reopen_when"},
    )
    lessons = experience.get("lessons")
    for index, item in enumerate(lessons if isinstance(lessons, list) else []):
        if not isinstance(item, dict):
            continue
        where = f"experience.lessons[{index}]"
        if item.get("kind") not in {"lever", "deadend", "feasibility"}:
            errors.append(f"{where}.kind must be lever, deadend, or feasibility")
        if item.get("kind") == "deadend" and not _nonempty(item.get("reopen_when")):
            errors.append(f"{where}.reopen_when is required for a deadend")
        if "reopen_when" in item and (
            not _nonempty(item.get("reopen_when"))
            or len(item.get("reopen_when", "")) > 600
        ):
            errors.append(
                f"{where}.reopen_when must be non-empty and at most 600 characters when present"
            )

    validate_items(
        "bottlenecks",
        limit=6,
        required={"claim", "evidence", "confidence"},
    )
    return errors


def validate_experience_replacement(experience: Any, ledger: dict[str, Any]) -> list[str]:
    """Additional freshness checks for a snapshot about to replace the old one."""
    if not isinstance(experience, dict):
        return []
    errors: list[str] = []
    if "dag_revision" in experience:
        errors.append("replacement experience must omit helper-owned dag_revision")
    terminal_ids = [
        str(record.get("run_id"))
        for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("status") in {"keep", "discard", "crash"}
    ]
    if terminal_ids and experience.get("updated_at_run") != terminal_ids[-1]:
        errors.append("replacement experience.updated_at_run must be the latest terminal ledger run")
    prior = ledger.get("experience")
    prior_generation = prior.get("generation") if isinstance(prior, dict) and prior else None
    expected_generation = (
        prior_generation + 1
        if isinstance(prior_generation, int) and not isinstance(prior_generation, bool)
        else 0
    )
    if experience.get("generation") != expected_generation:
        errors.append(
            f"replacement experience.generation must be {expected_generation}"
        )
    return errors


def _validated_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    registry = load_registry(args.background)
    dimension_strategy = resolve_dimension_strategy(args.background)
    catalog = resolve_dimension_catalog(
        args.background, explicit_path=getattr(args, "catalog", None)
    )
    ledger = _load_json(args.ledger) if getattr(args, "ledger", None) else None
    manifest = (
        _load_json(args.retrieval_manifest)
        if getattr(args, "retrieval_manifest", None)
        else None
    )
    errors = validate_registry(
        registry,
        ledger=ledger,
        retrieval_manifest=manifest,
        catalog=catalog,
        dimension_strategy=dimension_strategy,
    )
    errors.extend(validate_background_markdown(args.background, registry))
    return registry, ledger, errors


def cmd_catalog(args: argparse.Namespace) -> int:
    path = getattr(args, "path", None)
    catalog = load_catalog(path) if path else load_catalog()
    value = {"catalog": catalog, "receipt": catalog_receipt(catalog)}
    print(json.dumps(value, indent=None if args.compact else 2, separators=(",", ":") if args.compact else None))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    registry, _, errors = _validated_inputs(args)
    result = {
        "ok": not errors,
        "schema_version": registry.get("schema_version"),
        "space": space_receipt(registry),
        "dimensions": len(registry.get("dimensions", [])),
        "hypotheses": len(hypothesis_map(registry)),
        "relations": len(registry.get("relations", [])),
        "guidance": len(registry.get("guidance", [])),
        "sources": len(registry.get("sources", [])),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


def cmd_render(args: argparse.Namespace) -> int:
    if not 1 <= args.max_hypotheses <= 32:
        print(
            json.dumps(
                {"ok": False, "errors": ["--max-hypotheses must be in [1, 32]"]},
                separators=(",", ":"),
            )
        )
        return 1
    registry, ledger, errors = _validated_inputs(args)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, separators=(",", ":")))
        return 1
    print(
        json.dumps(
            render_space(registry, ledger, max_hypotheses=args.max_hypotheses),
            separators=(",", ":"),
        )
    )
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    registry, _, errors = _validated_inputs(args)
    result = {
        "ok": not errors,
        "schema_version": registry.get("schema_version"),
        "action": "none" if not errors else "reject",
        "space": space_receipt(registry),
        "errors": errors,
    }
    print(json.dumps(result, separators=(",", ":")))
    return 0 if not errors else 1


def cmd_validate_point(args: argparse.Namespace) -> int:
    registry, _, errors = _validated_inputs(args)
    point = _load_json(args.point)
    if not errors:
        errors.extend(validate_candidate_point(point, registry))
    print(
        json.dumps(
            {
                "ok": not errors,
                "point_id": point_id(point) if isinstance(point, dict) else None,
                "space_revision": space_revision(registry),
                "errors": errors,
            },
            indent=2,
        )
    )
    return 0 if not errors else 1


def cmd_lineage(args: argparse.Namespace) -> int:
    if args.compact and not 1 <= args.limit <= 64:
        print(
            json.dumps(
                {"ok": False, "errors": ["compact --limit must be in [1, 64]"]},
                separators=(",", ":"),
            )
        )
        return 1
    registry, ledger, errors = _validated_inputs(args)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, separators=(",", ":")))
        return 1
    value = derive_semantic_lineage(registry, ledger or {}, limit=args.limit if args.compact else None)
    print(json.dumps(value, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2))
    return 0


def cmd_validate_experience(args: argparse.Namespace) -> int:
    registry, ledger, errors = _validated_inputs(args)
    if args.experience:
        experience = _load_json(args.experience)
    else:
        experience = (ledger or {}).get("experience")
    if not errors:
        errors.extend(validate_experience(experience, registry, ledger or {}))
        if args.experience:
            errors.extend(validate_experience_replacement(experience, ledger or {}))
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="print a dimension catalog and its digest")
    catalog.add_argument("--path", type=Path, help="catalog JSON; defaults to the built-in catalog")
    catalog.add_argument("--compact", action="store_true")
    catalog.set_defaults(func=cmd_catalog)

    validate = sub.add_parser("validate", help="validate a hierarchical background")
    validate.add_argument("--background", type=Path, required=True)
    validate.add_argument("--catalog", type=Path, help="explicit dimension catalog override")
    validate.add_argument("--ledger", type=Path)
    validate.add_argument("--retrieval-manifest", type=Path)
    validate.set_defaults(func=cmd_validate)

    render = sub.add_parser("render", help="bounded dimension/hypothesis/coverage view")
    render.add_argument("--background", type=Path, required=True)
    render.add_argument("--catalog", type=Path, help="explicit dimension catalog override")
    render.add_argument("--ledger", type=Path)
    render.add_argument("--max-hypotheses", type=int, default=6)
    render.set_defaults(func=cmd_render)

    preflight = sub.add_parser("preflight", help="reject incompatible or mixed-mode run state")
    preflight.add_argument("--background", type=Path, required=True)
    preflight.add_argument("--catalog", type=Path, help="explicit dimension catalog override")
    preflight.add_argument("--ledger", type=Path)
    preflight.set_defaults(func=cmd_preflight)

    point = sub.add_parser("validate-point", help="validate a complete candidate semantic point")
    point.add_argument("--background", type=Path, required=True)
    point.add_argument("--catalog", type=Path, help="explicit dimension catalog override")
    point.add_argument("--point", type=Path, required=True)
    point.set_defaults(func=cmd_validate_point)

    lineage = sub.add_parser("lineage", help="render ancestry and mechanical point diffs separately")
    lineage.add_argument("--background", type=Path, required=True)
    lineage.add_argument("--catalog", type=Path, help="explicit dimension catalog override")
    lineage.add_argument("--ledger", type=Path, required=True)
    lineage.add_argument("--compact", action="store_true")
    lineage.add_argument("--limit", type=int, default=8)
    lineage.set_defaults(func=cmd_lineage)

    experience = sub.add_parser(
        "validate-experience", help="validate P1 experience without inventing P2 belief fields"
    )
    experience.add_argument("--background", type=Path, required=True)
    experience.add_argument("--catalog", type=Path, help="explicit dimension catalog override")
    experience.add_argument("--ledger", type=Path, required=True)
    experience.add_argument("--experience", type=Path)
    experience.set_defaults(func=cmd_validate_experience)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (ContractError, SemanticSpaceError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
