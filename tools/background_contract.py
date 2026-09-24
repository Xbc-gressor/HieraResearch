#!/usr/bin/env python3
"""Validate and render the P2 hierarchical ``background.md`` contract.

The Markdown document is the human view.  Its fenced ``Search space
registry`` JSON object is the machine contract shared by both runtimes.  This
module validates literature receipts and typed guidance around the structural
contract owned by :mod:`semantic_space`, checks the schema-3 experience
snapshot and the append-only ``search_space_state`` overlay against persisted
receipts, and renders the bounded per-target evidence view consumed by the
experience extractor.

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

from competition_policy import (
    COMPETITION_POLICY_VERSION,
    resolve_profile,
    scan_artifact_text,
)
from search_backends import (
    _VERIFICATION_RANK,
    _VIEW_VERIFICATION,
    canonical_key,
    merged_results,
    resolve_visit_content,
    unexplored_leads,
    validate_manifest,
    verification_statuses,
)
from search_space_state import (
    compose_effective_selection,
    empty_search_space_state,
    replay_search_space_state,
    validate_point_eligibility,
    validate_search_space_state,
)
from semantic_attempts import (
    validate_attempt_detail,
    validate_attempt_observations,
)
from semantic_evidence import (
    acquisition_target_relations,
    COVERAGE_KEYS,
    EVALUATION_STATES,
    LIFECYCLE_TERMINAL_STATUSES,
    MAX_DIMENSION_TARGETS,
    MAX_EDGES_PER_TARGET,
    MAX_HYPOTHESIS_TARGETS,
    MAX_RUNS_PER_TARGET,
    SemanticEvidenceError,
    _contradiction_depth_bar,
    comparator_coverage,
    edge_index,
    hypothesis_carriers,
    mechanical_gain_direction,
    normalize_coverage,
    render_target_evidence,
    target_evaluation_state,
    validate_conditioning_against_ledger,
    validate_conditioned_adjustment,
    validate_lineage_snapshots,
    validate_parameter_transfer_binding,
    validate_semantic_edges,
)
from semantic_space import (
    DEFAULT_DIMENSION_STRATEGY,
    DIMENSION_FIELDS,
    HYPOTHESIS_FIELDS,
    SemanticSpaceError,
    catalog_receipt,
    complete_point,
    coverage_from_records,
    derive_semantic_lineage,
    dimension_map,
    hypothesis_map,
    load_catalog,
    point_id,
    resolve_dimension_catalog,
    resolve_dimension_strategy,
    selected_assignments,
    space_receipt,
    space_revision,
    validate_point,
    validate_record_point,
    registry_for_point,
    validate_space_core,
    _validate_provenance,
)


SOURCE_RE = re.compile(r"^src-(\d{2,})$")
GUIDANCE_RE = re.compile(r"^g-(\d{2,})$")
SCOPE_TAG_RE = re.compile(r"^(?:\*|[a-z0-9][a-z0-9._-]*)$")
# Every lifecycle-terminal status, plus the one non-terminal state a record can
# legitimately sit in. Derived so a new terminal status cannot be accepted here
# while the terminal-state helpers still reject it.
RECORD_STATUSES = {"pending"} | LIFECYCLE_TERMINAL_STATUSES

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
GUIDANCE_EFFECTS = {"caution", "deprioritize"}
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
BASELINE_INVENTORY_SCHEMA_VERSION = 1
BASELINE_INVENTORY_KIND = "baseline_mechanism_inventory"
BASELINE_CITATION_RE = re.compile(r"^[^\s:]+:[0-9]+$")
POLICY_CONFIG_KEYS_V6 = {
    "coverage_weight",
    "cost_weight",
    "uncertainty_weight",
    "deprioritized_budget_interval",
    "llm_intelligence_score",
}
POLICY_CONFIG_KEYS_V7 = POLICY_CONFIG_KEYS_V6 | {
    "carrier_pos_weight",
    "carrier_pos_cap",
    "carrier_neg_weight",
    "carrier_neg_cap",
    "attempt_noise_threshold",
    "attempt_screen_weight",
    "attempt_crash_weight",
    "attempt_screen_prior_strength",
    "attempt_crash_prior_strength",
    "attempt_cap",
}
# Coverage-family arms. They share the "no model components" rule and differ
# only in which deterministic channel is added to coverage, which is what makes
# the carrier / attempt increments separately attributable.
COVERAGE_POLICY_NAMES = {
    "coverage",
    "coverage_experience",
    "coverage_attempt",
    "coverage_carrier_attempt",
}
CARRIER_POLICY_NAMES = {"coverage_experience", "coverage_carrier_attempt"}
ATTEMPT_POLICY_NAMES = {"coverage_attempt", "coverage_carrier_attempt"}
POLICY_NAMES = COVERAGE_POLICY_NAMES | {
    "gain",
    "gain_uncertainty",
    "gain_uncertainty_nocost",
    # The judged-slate arm admits via schema 8 receipts only; it carries no
    # acquisition components because the listwise facts live in the manifest.
    "judged_slate",
}
POLICY_CONFIG_KEYS_V8 = {"pool_size", "slate_size", "regular_rollouts"}
# Aggregation paths that can appear in a manifest with a non-empty slate
# (mirrors the decision tree in tools/slate.py, which imports this module).
SLATE_AGGREGATION_PATHS = {
    "consensus",
    "boundary",
    "coverage_fallback",
    "judge_skipped_cardinality",
    "judge_skipped_pool_le_B",
}
SLATE_POOL_SIZE_RANGE = (3, 12)  # mirrors slate.POOL_SIZE_MIN/MAX
SLATE_CONFIG_FROZEN = {"slate_size": 2, "regular_rollouts": 2}


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
    """Resolve the current published registry, or the initial Markdown registry."""
    from space_revisions import load_revision_state
    state = load_revision_state(path)
    if state is not None:
        return state["versions"][-1]["registry"]
    return load_initial_registry(path)


def load_initial_registry(path: Path) -> dict[str, Any]:
    """Read the unchanged research-time registry for source/Markdown auditing."""
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


def _competition_policy_profile(manifest_path: Path | None) -> dict[str, Any] | None:
    """The resolved identity profile when the manifest sits in competition
    scope; non-MLE and unresolvable ad-hoc paths stay ungated."""
    if manifest_path is None:
        return None
    profile = resolve_profile(manifest_path).profile
    if not profile or not str(profile.get("competition_id") or "").strip():
        return None
    return profile


def _competition_policy_manifest_errors(
    retrieval_manifest: dict[str, Any], profile: dict[str, Any]
) -> list[str]:
    """Fail closed on manifests that predate or contradict the resolved
    competition identity."""
    errors: list[str] = []
    competition_id = str(profile.get("competition_id") or "").strip()
    stamped = retrieval_manifest.get("competition_id")
    if not isinstance(stamped, str) or not stamped.strip():
        errors.append(
            "competition policy: retrieval manifest competition_id is missing; "
            f"the run resolves to competition {competition_id!r} and an "
            "undetermined manifest fails closed"
        )
    elif stamped.strip() != competition_id:
        errors.append(
            f"competition policy: retrieval manifest competition_id {stamped!r} "
            f"does not match the resolved competition {competition_id!r}"
        )
    version = retrieval_manifest.get("competition_policy_version")
    if not isinstance(version, int):
        errors.append(
            "competition policy: retrieval manifest competition_policy_version "
            f"is missing; a retrieval over competition {competition_id!r} must "
            f"stamp version >= {COMPETITION_POLICY_VERSION}"
        )
    elif version < COMPETITION_POLICY_VERSION:
        errors.append(
            "competition policy: retrieval manifest competition_policy_version "
            f"{version} predates {COMPETITION_POLICY_VERSION}; re-run retrieval "
            "under the current policy"
        )
    return errors


def _validate_sources(
    registry: dict[str, Any],
    retrieval_manifest: dict[str, Any] | None,
    manifest_dir: Path | None = None,
    manifest_path: Path | None = None,
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
        errors.extend(validate_manifest(retrieval_manifest, manifest_dir))
        policy_profile = _competition_policy_profile(manifest_path)
        if policy_profile is not None:
            errors.extend(
                _competition_policy_manifest_errors(retrieval_manifest, policy_profile)
            )
        # Receipt hard gate: every cited URL needs a tool-recorded receipt —
        # a search hit (snippet receipt) or a successful visit.  The tier of
        # that receipt is reported by source_verification, never gated here.
        # Search-hit rows are inherently allowed (blocked rows are diverted to
        # blocked_results); a visit counts only when policy-allowed whenever a
        # competition profile resolves.
        receipt_keys = {
            result["canonical_key"] for result in merged_results(retrieval_manifest)
        }
        policy_excluded_keys: set[str] = set()
        for visit in retrieval_manifest.get("visits", []):
            if not isinstance(visit, dict) or not _nonempty(visit.get("canonical_key")):
                continue
            if visit.get("status") == "success" and (
                policy_profile is None or visit.get("policy_status") == "allowed"
            ):
                receipt_keys.add(visit["canonical_key"])
            elif policy_profile is not None and visit.get("status") in {"success", "blocked"}:
                policy_excluded_keys.add(visit["canonical_key"])
        for source_id, url in source_urls.items():
            key = canonical_key(url)
            if key in receipt_keys:
                continue
            if key in policy_excluded_keys:
                errors.append(
                    f"competition policy: source {source_id} is backed only by "
                    f"policy-blocked or undetermined visit receipts for {url}; "
                    "the retrieval manifest must record an allowed visit or a "
                    "search hit"
                )
            else:
                errors.append(
                    f"source {source_id} has no retrieval receipt for {url}; "
                    "the retrieval manifest must record a search hit or a "
                    "successful visit"
                )
    return errors, source_by_id


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
            synthesis = hypothesis.get("kind") == "synthesis_probe"
            allow_empty = hypothesis.get("kind") == "baseline" or synthesis
            link_errors, links = _evidence_link_errors(
                hypothesis.get("evidence"),
                f"{where}.evidence",
                source_ids,
                allow_empty=allow_empty,
            )
            errors.extend(link_errors)
            errors.extend(_credibility_errors(hypothesis, links, source_by_id, where))
            if synthesis:
                if not any(p.get("kind") == "agent_synthesis"
                           for p in hypothesis.get("provenance", []) if isinstance(p, dict)):
                    errors.append(f"{where} synthesis_probe requires agent_synthesis provenance")
                if not links and hypothesis.get("literature_credibility") != "unverified":
                    errors.append(f"{where} unsourced synthesis must have unverified literature credibility")
    return errors


def validate_reserves(registry: dict[str, Any]) -> list[str]:
    """Lightweight retained directions; these are not registered hypotheses."""
    reserves = registry.get("reserves", [])
    if not isinstance(reserves, list) or len(reserves) > 12:
        return ["reserves must be a list of at most 12 directions"]
    errors: list[str] = []
    seen: set[str] = set()
    sources = {s.get("id") for s in registry.get("sources", []) if isinstance(s, dict)}
    for index, lead in enumerate(reserves):
        where = f"reserves[{index}]"
        fields = {"id", "mechanism", "provenance", "deferred_reason", "open_questions"}
        if not isinstance(lead, dict) or set(lead) != fields:
            errors.append(f"{where} must contain {sorted(fields)}")
            continue
        for field in ("id", "mechanism", "deferred_reason"):
            if not _nonempty(lead.get(field)):
                errors.append(f"{where}.{field} must be non-empty")
        lead_id = lead.get("id")
        if isinstance(lead_id, str):
            if lead_id in seen:
                errors.append(f"{where} duplicate reserve id {lead_id}")
            seen.add(lead_id)
        questions = lead.get("open_questions")
        if not isinstance(questions, list) or not questions or any(not _nonempty(q) for q in questions):
            errors.append(f"{where}.open_questions must be a non-empty string list")
        provenance = lead.get("provenance")
        errors.extend(_validate_provenance(provenance, f"{where}.provenance"))
        for receipt in provenance if isinstance(provenance, list) else []:
            if isinstance(receipt, dict) and receipt.get("kind") == "literature" and receipt.get("ref") not in sources:
                errors.append(f"{where} references unknown source {receipt.get('ref')!r}")
    return errors


def _baseline_interventions(registry: dict[str, Any]) -> dict[str, tuple[str, set[str]]]:
    """Map each dimension id to its baseline hypothesis id and declared mechanisms."""
    baselines: dict[str, tuple[str, set[str]]] = {}
    for dimension in registry.get("dimensions", []):
        if not isinstance(dimension, dict):
            continue
        dimension_id = dimension.get("id")
        baseline_id = dimension.get("baseline_hypothesis_id")
        for hypothesis in dimension.get("hypotheses", []):
            if not isinstance(hypothesis, dict) or hypothesis.get("id") != baseline_id:
                continue
            scope = hypothesis.get("scope")
            if not isinstance(scope, dict):
                continue
            values = scope.get("interventions")
            if isinstance(values, list) and all(isinstance(v, str) for v in values):
                baselines[str(dimension_id)] = (str(baseline_id), set(values))
    return baselines


def _validate_baseline_mechanism_disjointness(
    registry: dict[str, Any],
    inventory: dict[str, Any] | None = None,
) -> list[str]:
    """Keep alternative hypotheses mechanically distinct from every baseline.

    A non-baseline hypothesis that names a mechanism a baseline already applies
    is not a contrast: candidates attributed to it re-implement the control, so
    their observations measure implementation noise while the ledger records a
    clean single-dimension edge.  ``interventions`` is the declared mechanism
    set, so overlap with any baseline is a frozen-space defect.

    When a baseline mechanism inventory is supplied, each dimension's baseline
    must also *declare* everything the provided entrypoint already does; that
    closes the gap where an omitted baseline mechanism hides the collision.
    """
    errors: list[str] = []
    baselines = _baseline_interventions(registry)
    if inventory is not None:
        errors.extend(_validate_baseline_inventory(registry, inventory, baselines))
    baseline_ids = {baseline_id for baseline_id, _ in baselines.values()}
    for dimension in registry.get("dimensions", []):
        if not isinstance(dimension, dict):
            continue
        for hypothesis in dimension.get("hypotheses", []):
            if not isinstance(hypothesis, dict):
                continue
            hypothesis_id = hypothesis.get("id")
            if hypothesis.get("kind") == "baseline" or hypothesis_id in baseline_ids:
                continue
            scope = hypothesis.get("scope")
            if not isinstance(scope, dict):
                continue
            values = scope.get("interventions")
            if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                continue
            mechanisms = set(values)
            for owner_dimension, (baseline_id, baseline_mechanisms) in sorted(
                baselines.items()
            ):
                overlap = mechanisms & baseline_mechanisms
                if not overlap:
                    continue
                errors.append(
                    f"hypothesis {hypothesis_id} in {dimension.get('id')} shares "
                    f"interventions {sorted(overlap)} with the {owner_dimension} "
                    f"baseline {baseline_id}; a mechanism the baseline already "
                    "applies is not a contrast"
                )
    return errors


def _validate_baseline_inventory(
    registry: dict[str, Any],
    inventory: Any,
    baselines: dict[str, tuple[str, set[str]]],
) -> list[str]:
    """Require every provided-entrypoint mechanism to be declared by its baseline."""
    errors: list[str] = []
    if not isinstance(inventory, dict):
        return ["baseline mechanism inventory must be a JSON object"]
    if inventory.get("schema_version") != BASELINE_INVENTORY_SCHEMA_VERSION:
        errors.append(
            "baseline mechanism inventory schema_version must be "
            f"{BASELINE_INVENTORY_SCHEMA_VERSION}"
        )
    if inventory.get("kind") != BASELINE_INVENTORY_KIND:
        errors.append(
            f"baseline mechanism inventory kind must be {BASELINE_INVENTORY_KIND!r}"
        )
    entrypoint = inventory.get("entrypoint")
    if not isinstance(entrypoint, dict) or not _nonempty(entrypoint.get("path")):
        errors.append("baseline mechanism inventory requires entrypoint.path")
    dimensions = inventory.get("dimensions")
    if not isinstance(dimensions, dict) or not dimensions:
        return errors + ["baseline mechanism inventory requires a non-empty dimensions map"]
    known = {
        str(dimension.get("id"))
        for dimension in registry.get("dimensions", [])
        if isinstance(dimension, dict)
    }
    unknown = sorted(set(dimensions) - known)
    if unknown:
        errors.append(f"baseline mechanism inventory references unknown dimensions {unknown}")
    missing = sorted(known - set(dimensions))
    if missing:
        errors.append(
            f"baseline mechanism inventory does not cover dimensions {missing}; "
            "every resolved dimension needs its baseline mechanisms"
        )
    for dimension_id in sorted(set(dimensions) & known):
        entry = dimensions[dimension_id]
        where = f"baseline mechanism inventory {dimension_id}"
        if not isinstance(entry, dict):
            errors.append(f"{where} must be an object")
            continue
        values = entry.get("interventions")
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str) or SCOPE_TAG_RE.fullmatch(value) is None
                for value in values
            )
        ):
            errors.append(f"{where}.interventions must be a non-empty list of scope tags")
            continue
        if len(values) != len(set(values)):
            errors.append(f"{where}.interventions must not contain duplicate tags")
            continue
        citations = entry.get("citations")
        if (
            not isinstance(citations, list)
            or not citations
            or any(
                not isinstance(citation, str)
                or BASELINE_CITATION_RE.fullmatch(citation) is None
                for citation in citations
            )
        ):
            errors.append(
                f"{where}.citations must be a non-empty list of '<file>:<line>' receipts"
            )
        declared = baselines.get(dimension_id)
        if declared is None:
            errors.append(f"{where} has no resolvable baseline hypothesis to check")
            continue
        baseline_id, baseline_mechanisms = declared
        undeclared = sorted(set(values) - baseline_mechanisms)
        if undeclared:
            errors.append(
                f"{where} baseline {baseline_id} does not declare interventions "
                f"{undeclared} that the provided entrypoint already applies; add "
                "them to its scope.interventions so alternatives stay distinct"
            )
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
        if effect == "deprioritize":
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
            if item.get("effect") == "deprioritize":
                binding.append(receipt)
        result[hypothesis_id] = {
            "selection_status": "deprioritized" if binding else "active",
            "matched_guidance": matched,
            "binding_guidance": binding,
        }
    return result


def _validate_policy_receipt_v8(record: dict[str, Any], where: str) -> list[str]:
    """The judged-slate receipt shape (schema 8).

    The receipt binds the record to one generation manifest slot; it carries
    no ``selected_point_id``/``ranked_point_ids``/``components`` — those
    listwise facts exist only in the manifest the receipt points to.
    """
    receipt = record["policy_receipt"]
    point = record.get("semantic_point")
    point_object = point if isinstance(point, dict) else {}
    errors: list[str] = []
    allowed = {
        "schema_version",
        "space",
        "search_space_state_revision",
        "policy",
        "generation_id",
        "judge",
        "carrier_proposal_set_revision",
        "budget",
        "experience",
    }
    unknown = sorted(set(receipt) - allowed)
    if unknown:
        errors.append(f"{where}.policy_receipt has unknown fields {unknown}")
    state_revision = receipt.get("search_space_state_revision")
    if (
        not isinstance(state_revision, int)
        or isinstance(state_revision, bool)
        or state_revision < 0
    ):
        errors.append(
            f"{where}.policy_receipt.search_space_state_revision must be a "
            "non-negative integer"
        )
    space = receipt.get("space")
    if not isinstance(space, dict):
        errors.append(f"{where}.policy_receipt.space must be the frozen space receipt")
    elif space.get("space_revision") != point_object.get("space_revision"):
        errors.append(
            f"{where}.policy_receipt.space.space_revision must match semantic_point"
        )
    policy = receipt.get("policy")
    if not isinstance(policy, dict) or set(policy) != {"name", "config"}:
        errors.append(f"{where}.policy_receipt.policy must contain only name and config")
        policy = {}
    elif policy.get("name") != "judged_slate":
        errors.append(
            f"{where}.policy_receipt.policy.name must be judged_slate at schema 8"
        )
    config = policy.get("config") if isinstance(policy, dict) else None
    if not isinstance(config, dict) or set(config) != POLICY_CONFIG_KEYS_V8:
        errors.append(
            f"{where}.policy_receipt.policy.config must keep exactly "
            f"{sorted(POLICY_CONFIG_KEYS_V8)}"
        )
    else:
        pool_size = config.get("pool_size")
        low, high = SLATE_POOL_SIZE_RANGE
        if (
            not isinstance(pool_size, int)
            or isinstance(pool_size, bool)
            or not low <= pool_size <= high
        ):
            errors.append(
                f"{where}.policy_receipt.policy.config.pool_size must be "
                f"an integer in [{low}, {high}]"
            )
        for key, expected in SLATE_CONFIG_FROZEN.items():
            if config.get(key) != expected:
                errors.append(
                    f"{where}.policy_receipt.policy.config.{key} is fixed at {expected}"
                )
    generation_id = receipt.get("generation_id")
    if not isinstance(generation_id, str) or DIGEST_RE.fullmatch(generation_id) is None:
        errors.append(f"{where}.policy_receipt.generation_id must be a sha256 digest")
    judge = receipt.get("judge")
    judge_fields = {
        "manifest_path",
        "slate_index",
        "candidate_id",
        "aggregation",
    }
    if not isinstance(judge, dict) or set(judge) != judge_fields:
        errors.append(
            f"{where}.policy_receipt.judge must contain exactly "
            f"{sorted(judge_fields)}"
        )
    else:
        if not _nonempty(judge.get("manifest_path")):
            errors.append(
                f"{where}.policy_receipt.judge.manifest_path must be a non-empty "
                "run-relative path"
            )
        candidate_id = judge.get("candidate_id")
        if not isinstance(candidate_id, str) or DIGEST_RE.fullmatch(candidate_id) is None:
            errors.append(
                f"{where}.policy_receipt.judge.candidate_id must be a sha256 digest"
            )
        slate_index = judge.get("slate_index")
        if (
            not isinstance(slate_index, int)
            or isinstance(slate_index, bool)
            or slate_index < 0
        ):
            errors.append(
                f"{where}.policy_receipt.judge.slate_index must be a "
                "non-negative integer"
            )
        if judge.get("aggregation") not in SLATE_AGGREGATION_PATHS:
            errors.append(
                f"{where}.policy_receipt.judge.aggregation must be one of "
                f"{sorted(SLATE_AGGREGATION_PATHS)}"
            )
    carrier_revision = receipt.get("carrier_proposal_set_revision")
    if not isinstance(carrier_revision, str) or DIGEST_RE.fullmatch(carrier_revision) is None:
        errors.append(
            f"{where}.policy_receipt.carrier_proposal_set_revision must be a "
            "sha256 digest"
        )
    budget = receipt.get("budget")
    if not isinstance(budget, dict) or set(budget) != {"selection_index", "admission_cap"}:
        errors.append(
            f"{where}.policy_receipt.budget must contain exactly "
            "['admission_cap', 'selection_index']"
        )
    else:
        selection_index = budget.get("selection_index")
        if (
            not isinstance(selection_index, int)
            or isinstance(selection_index, bool)
            or selection_index < 1
        ):
            errors.append(
                f"{where}.policy_receipt.budget.selection_index must be a "
                "positive integer"
            )
        admission_cap = budget.get("admission_cap")
        if admission_cap is not None and (
            not isinstance(admission_cap, int)
            or isinstance(admission_cap, bool)
            or admission_cap < 1
        ):
            errors.append(
                f"{where}.policy_receipt.budget.admission_cap must be null or a "
                "positive integer"
            )
    experience = receipt.get("experience")
    experience_fields = {"generation", "updated_at_run", "revision"}
    if not isinstance(experience, dict) or set(experience) != experience_fields:
        errors.append(
            f"{where}.policy_receipt.experience must contain exactly "
            f"{sorted(experience_fields)}"
        )
    else:
        generation = experience.get("generation")
        updated_at_run = experience.get("updated_at_run")
        revision = experience.get("revision")
        snapshot_absent = (
            generation is None and updated_at_run is None and revision is None
        )
        snapshot_present = (
            isinstance(generation, int)
            and not isinstance(generation, bool)
            and generation >= 0
            and ((generation == 0 and updated_at_run is None)
                 or (isinstance(updated_at_run, str) and updated_at_run.isdigit()))
            and isinstance(revision, str)
            and DIGEST_RE.fullmatch(revision) is not None
        )
        if not (snapshot_absent or snapshot_present):
            errors.append(
                f"{where}.policy_receipt.experience snapshot fields must be "
                "all null or a valid generation/run/revision receipt"
            )
    return errors


def _validate_policy_receipt(record: dict[str, Any], where: str) -> list[str]:
    receipt = record.get("policy_receipt")
    point = record.get("semantic_point")
    if not isinstance(receipt, dict):
        return [f"{where}.policy_receipt must be an object distinct from observations"]
    if receipt.get("schema_version") == 8:
        return _validate_policy_receipt_v8(record, where)
    point_object = point if isinstance(point, dict) else {}
    errors: list[str] = []
    receipt_schema = receipt.get("schema_version")
    if receipt_schema not in {6, 7}:
        errors.append(
            f"{where}.policy_receipt.schema_version must be 6, 7, or 8"
        )
    state_revision = receipt.get("search_space_state_revision")
    if (
        not isinstance(state_revision, int)
        or isinstance(state_revision, bool)
        or state_revision < 0
    ):
        errors.append(
            f"{where}.policy_receipt.search_space_state_revision must be a "
            "non-negative integer"
        )
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
    elif policy_name not in POLICY_NAMES:
        errors.append(
            f"{where}.policy_receipt.policy.name must be one of "
            f"{sorted(POLICY_NAMES)}"
        )
    elif policy_name in ATTEMPT_POLICY_NAMES and receipt_schema != 7:
        errors.append(
            f"{where}.policy_receipt.policy.name {policy_name} requires "
            "receipt schema 7"
        )
    config = policy.get("config") if isinstance(policy, dict) else None
    config_valid = True
    expected_config_keys = (
        POLICY_CONFIG_KEYS_V7 if receipt_schema == 7 else POLICY_CONFIG_KEYS_V6
    )
    if not isinstance(config, dict) or set(config) != expected_config_keys:
        errors.append(
            f"{where}.policy_receipt.policy.config must keep exactly "
            f"{sorted(expected_config_keys)}"
        )
        config = {}
        config_valid = False
    else:
        for key, value in config.items():
            if key == "deprioritized_budget_interval":
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 2 <= value <= 1000
                ):
                    config_valid = False
                    errors.append(
                        f"{where}.policy_receipt.policy.config.{key} must be "
                        "an integer in [2, 1000]"
                    )
                continue
            if key in {"carrier_pos_cap", "carrier_neg_cap"}:
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 1 <= value <= 100
                ):
                    config_valid = False
                    errors.append(
                        f"{where}.policy_receipt.policy.config.{key} must be "
                        "an integer in [1, 100]"
                    )
                continue
            if key == "llm_intelligence_score":
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 100.0
                ):
                    config_valid = False
                    errors.append(
                        f"{where}.policy_receipt.policy.config.{key} must be "
                        "a finite number in [0, 100]"
                    )
                continue
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
    required_components = {
        "coverage",
        "predicted_gain",
        "uncertainty",
        "cost",
        "prior_gain",
        "experience_gain_adjustment",
        "prior_uncertainty",
        "experience_uncertainty_adjustment",
        "llm_judgment_weight",
    }
    if receipt_schema == 7:
        required_components |= {
            "experience_prior",
            "carriers",
            "attempt_prior",
            "attempts",
        }
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
        conditioned_components = [
            components.get("prior_gain"),
            components.get("experience_gain_adjustment"),
            components.get("prior_uncertainty"),
            components.get("experience_uncertainty_adjustment"),
        ]
        llm_judgment_weight = components.get("llm_judgment_weight")
        if policy_name in COVERAGE_POLICY_NAMES:
            model_components_valid = (
                all(value is None for value in model_components)
                and all(value is None for value in conditioned_components)
                and llm_judgment_weight is None
            )
            if not model_components_valid:
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
        if policy_name not in COVERAGE_POLICY_NAMES:
            prior_values = (
                components.get("prior_gain"),
                components.get("prior_uncertainty"),
            )
            adjustment_values = (
                components.get("experience_gain_adjustment"),
                components.get("experience_uncertainty_adjustment"),
            )
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0
                for value in prior_values
            ):
                model_components_valid = False
                errors.append(
                    f"{where}.policy_receipt schema {receipt_schema} prior gain and uncertainty "
                    "must be finite values in [0, 1]"
                )
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and -1.0 <= float(value) <= 1.0
                for value in adjustment_values
            ):
                model_components_valid = False
                errors.append(
                    f"{where}.policy_receipt schema {receipt_schema} experience adjustments "
                    "must be finite values in [-1, 1]"
                )
            arithmetic = (
                (
                    components.get("prior_gain"),
                    components.get("experience_gain_adjustment"),
                    components.get("predicted_gain"),
                    "predicted_gain",
                ),
                (
                    components.get("prior_uncertainty"),
                    components.get("experience_uncertainty_adjustment"),
                    components.get("uncertainty"),
                    "uncertainty",
                ),
            )
            for prior, adjustment, final, label in arithmetic:
                if all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in (prior, adjustment, final)
                ) and not math.isclose(
                    float(prior) + float(adjustment),
                    float(final),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    model_components_valid = False
                    errors.append(
                        f"{where}.policy_receipt {label} must equal its prior "
                        "plus experience adjustment"
                    )
        if policy_name not in COVERAGE_POLICY_NAMES:
            expected_llm_weight = (
                float(config["llm_intelligence_score"]) / 100.0
                if config_valid
                else None
            )
            if (
                not isinstance(llm_judgment_weight, (int, float))
                or isinstance(llm_judgment_weight, bool)
                or not math.isfinite(float(llm_judgment_weight))
                or not 0.0 <= float(llm_judgment_weight) <= 1.0
            ):
                model_components_valid = False
                errors.append(
                    f"{where}.policy_receipt.components.llm_judgment_weight "
                    "must be a finite number in [0, 1]"
                )
            elif expected_llm_weight is not None and not math.isclose(
                float(llm_judgment_weight),
                expected_llm_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                model_components_valid = False
                errors.append(
                    f"{where}.policy_receipt.components.llm_judgment_weight "
                    "must equal policy.config.llm_intelligence_score / 100"
                )
        if receipt_schema == 7:
            carrier_prior = components.get("experience_prior")
            carrier_detail = components.get("carriers")
            if policy_name in CARRIER_POLICY_NAMES:
                if not (
                    isinstance(carrier_prior, (int, float))
                    and not isinstance(carrier_prior, bool)
                    and math.isfinite(float(carrier_prior))
                ):
                    errors.append(
                        f"{where}.policy_receipt {policy_name} requires a "
                        "finite components.experience_prior"
                    )
                if not isinstance(carrier_detail, dict) or not all(
                    isinstance(detail, dict)
                    and isinstance(detail.get("negative"), int)
                    and not isinstance(detail.get("negative"), bool)
                    and detail["negative"] >= 0
                    and isinstance(detail.get("positive"), int)
                    and not isinstance(detail.get("positive"), bool)
                    and detail["positive"] >= 0
                    for detail in carrier_detail.values()
                ):
                    errors.append(
                        f"{where}.policy_receipt {policy_name} components."
                        "carriers must map hypothesis ids to non-negative "
                        "integer counts"
                    )
            elif carrier_prior is not None or carrier_detail is not None:
                errors.append(
                    f"{where}.policy_receipt {policy_name} must not invent "
                    "carrier components"
                )
            attempt_prior = components.get("attempt_prior")
            attempt_detail = components.get("attempts")
            if policy_name in ATTEMPT_POLICY_NAMES:
                # The attempt channel is a downside only: an observed success
                # dilutes an existing penalty but never buys a bonus.
                if not (
                    isinstance(attempt_prior, (int, float))
                    and not isinstance(attempt_prior, bool)
                    and math.isfinite(float(attempt_prior))
                    and float(attempt_prior) <= 0.0
                ):
                    errors.append(
                        f"{where}.policy_receipt {policy_name} requires a finite "
                        "non-positive components.attempt_prior"
                    )
                errors.extend(
                    f"{where}.policy_receipt.components.{error}"
                    for error in validate_attempt_detail(attempt_detail)
                )
            elif attempt_prior is not None or attempt_detail is not None:
                errors.append(
                    f"{where}.policy_receipt {policy_name} must not invent "
                    "attempt components"
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
    if policy_name in COVERAGE_POLICY_NAMES and evidence:
        errors.append(f"{where}.policy_receipt coverage policy evidence must be empty")
    if policy_name in {"gain", "gain_uncertainty", "gain_uncertainty_nocost"} and not evidence:
        errors.append(f"{where}.policy_receipt gain policies require selection evidence")

    experience_receipt = receipt.get("experience")
    if receipt_schema in {6, 7}:
        experience_fields = {
            "generation",
            "updated_at_run",
            "revision",
            "evidence_run_ids",
            "evidence_edge_ids",
            "rationale",
            "conditioning",
        }
        if (
            not isinstance(experience_receipt, dict)
            or set(experience_receipt) != experience_fields
        ):
            errors.append(
                f"{where}.policy_receipt.experience must contain exactly "
                f"{sorted(experience_fields)}"
            )
            experience_receipt = {}
        else:
            generation = experience_receipt.get("generation")
            updated_at_run = experience_receipt.get("updated_at_run")
            revision = experience_receipt.get("revision")
            snapshot_absent = (
                generation is None and updated_at_run is None and revision is None
            )
            snapshot_present = (
                isinstance(generation, int)
                and not isinstance(generation, bool)
                and generation >= 0
                and ((generation == 0 and updated_at_run is None)
                     or (isinstance(updated_at_run, str) and updated_at_run.isdigit()))
                and isinstance(revision, str)
                and DIGEST_RE.fullmatch(revision) is not None
            )
            if not (snapshot_absent or snapshot_present):
                errors.append(
                    f"{where}.policy_receipt.experience snapshot fields must be "
                    "all null or a valid generation/run/revision receipt"
                )
            run_ids = experience_receipt.get("evidence_run_ids")
            if (
                not isinstance(run_ids, list)
                or len(run_ids) > 5
                or any(not isinstance(run_id, str) or not run_id.isdigit() for run_id in run_ids)
                or len(run_ids) != len(set(run_ids))
            ):
                errors.append(
                    f"{where}.policy_receipt.experience.evidence_run_ids must "
                    "contain 0–5 unique numeric run ids"
                )
                run_ids = []
            edge_ids = experience_receipt.get("evidence_edge_ids")
            if (
                not isinstance(edge_ids, list)
                or len(edge_ids) > 5
                or any(not _nonempty(edge_id) for edge_id in edge_ids)
                or len(edge_ids) != len(set(edge_ids))
            ):
                errors.append(
                    f"{where}.policy_receipt.experience.evidence_edge_ids must "
                    "contain 0–5 unique non-empty edge ids"
                )
                edge_ids = []
            rationale = experience_receipt.get("rationale")
            if not _nonempty(rationale) or len(rationale) > 240:
                errors.append(
                    f"{where}.policy_receipt.experience.rationale must be "
                    "non-empty and at most 240 characters"
                )
            if policy_name in COVERAGE_POLICY_NAMES:
                if run_ids or edge_ids or experience_receipt.get("conditioning") != []:
                    errors.append(
                        f"{where}.policy_receipt coverage policy must not cite "
                        "experience as a model-score input"
                    )
            else:
                conditioning = experience_receipt.get("conditioning")
                errors.extend(
                    f"{where}.policy_receipt.experience: {error}"
                    for error in validate_conditioned_adjustment(
                        conditioning,
                        point_object,
                        evidence_run_ids=run_ids,
                        evidence_edge_ids=edge_ids,
                        gain_adjustment=components.get(
                            "experience_gain_adjustment"
                        ),
                        uncertainty_adjustment=components.get(
                            "experience_uncertainty_adjustment"
                        ),
                    )
                )
                if not snapshot_present and (
                    conditioning or run_ids or edge_ids
                ):
                    errors.append(
                        f"{where}.policy_receipt cannot condition on experience "
                        "without an experience snapshot"
                    )

    budget = receipt.get("budget")
    if receipt_schema == 6:
        budget_fields = {
            "selection_index",
            "deprioritized_interval",
            "scheduled_lane",
            "selected_lane",
            "fallback",
            "base_rank",
        }
        if not isinstance(budget, dict) or set(budget) != budget_fields:
            errors.append(
                f"{where}.policy_receipt.budget must contain exactly "
                f"{sorted(budget_fields)}"
            )
            budget = {}
        else:
            selection_index = budget.get("selection_index")
            interval = budget.get("deprioritized_interval")
            if (
                not isinstance(selection_index, int)
                or isinstance(selection_index, bool)
                or selection_index < 1
            ):
                errors.append(
                    f"{where}.policy_receipt.budget.selection_index must be "
                    "a positive integer"
                )
            if (
                not isinstance(interval, int)
                or isinstance(interval, bool)
                or not 2 <= interval <= 1000
            ):
                errors.append(
                    f"{where}.policy_receipt.budget.deprioritized_interval "
                    "must be an integer in [2, 1000]"
                )
            elif config and interval != config.get("deprioritized_budget_interval"):
                errors.append(
                    f"{where}.policy_receipt.budget.deprioritized_interval "
                    "must match policy.config"
                )
            scheduled = budget.get("scheduled_lane")
            selected = budget.get("selected_lane")
            fallback = budget.get("fallback")
            if scheduled not in {"active", "deprioritized"}:
                errors.append(
                    f"{where}.policy_receipt.budget.scheduled_lane must be "
                    "active or deprioritized"
                )
            if selected not in {"active", "deprioritized"}:
                errors.append(
                    f"{where}.policy_receipt.budget.selected_lane must be "
                    "active or deprioritized"
                )
            if (
                isinstance(selection_index, int)
                and not isinstance(selection_index, bool)
                and isinstance(interval, int)
                and not isinstance(interval, bool)
                and interval >= 2
            ):
                expected_lane = (
                    "deprioritized"
                    if selection_index % interval == 0
                    else "active"
                )
                if scheduled != expected_lane:
                    errors.append(
                        f"{where}.policy_receipt.budget.scheduled_lane does "
                        "not match its deterministic slot"
                    )
            expected_fallbacks = {
                ("active", "active"): "none",
                ("deprioritized", "deprioritized"): "none",
                ("active", "deprioritized"): "no_active_proposals",
                ("deprioritized", "active"): "no_deprioritized_proposals",
            }
            expected_fallback = expected_fallbacks.get((scheduled, selected))
            if fallback != expected_fallback:
                errors.append(
                    f"{where}.policy_receipt.budget.fallback must explain the "
                    "selected lane"
                )
            base_rank = budget.get("base_rank")
            if (
                not isinstance(base_rank, int)
                or isinstance(base_rank, bool)
                or base_rank < 1
            ):
                errors.append(
                    f"{where}.policy_receipt.budget.base_rank must be a "
                    "positive integer"
                )
    elif receipt_schema == 7:
        budget_fields = {
            "selection_index",
            "deprioritized_interval",
            "scheduled_lane",
            "selected_lane",
            "fallback",
            "base_rank",
        }
        if not isinstance(budget, dict) or set(budget) != budget_fields:
            errors.append(
                f"{where}.policy_receipt.budget must contain exactly "
                f"{sorted(budget_fields)}"
            )
        else:
            if (
                not isinstance(budget.get("selection_index"), int)
                or isinstance(budget.get("selection_index"), bool)
                or budget["selection_index"] < 1
            ):
                errors.append(
                    f"{where}.policy_receipt.budget.selection_index must be "
                    "a positive integer"
                )
            if (
                budget.get("deprioritized_interval") is not None
                or budget.get("scheduled_lane") is not None
                or budget.get("selected_lane") is not None
            ):
                errors.append(
                    f"{where}.policy_receipt.budget lane fields must be null: "
                    "lane scheduling was removed in favor of the carrier prior"
                )
            if budget.get("fallback") != "lanes_removed":
                errors.append(
                    f"{where}.policy_receipt.budget.fallback must be lanes_removed"
                )
            if budget.get("base_rank") != 1:
                errors.append(
                    f"{where}.policy_receipt.budget.base_rank must be 1 without lanes"
                )
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
        and policy_name in POLICY_NAMES
    ):
        coverage = float(components["coverage"])
        if policy_name in COVERAGE_POLICY_NAMES:
            channels = []
            if policy_name in CARRIER_POLICY_NAMES:
                channels.append(components.get("experience_prior"))
            if policy_name in ATTEMPT_POLICY_NAMES:
                channels.append(components.get("attempt_prior"))
            if all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in channels
            ):
                expected_score = coverage + sum(float(value) for value in channels)
            else:
                # A malformed channel already produced its own error above.
                expected_score = None
        elif policy_name == "gain_uncertainty_nocost":
            model_score = (
                float(components["predicted_gain"])
                + float(config["uncertainty_weight"]) * float(components["uncertainty"])
            )
            model_score *= float(components["llm_judgment_weight"])
            expected_score = model_score + float(config["coverage_weight"]) * coverage
        elif all(components[key] is not None for key in ("predicted_gain", "uncertainty", "cost")):
            model_score = (
                float(components["predicted_gain"])
                - float(config["cost_weight"]) * float(components["cost"])
            )
            if policy_name == "gain_uncertainty":
                model_score += float(config["uncertainty_weight"]) * float(
                    components["uncertainty"]
                )
            model_score *= float(components["llm_judgment_weight"])
            expected_score = model_score + float(config["coverage_weight"]) * coverage
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
    elif (
        isinstance(budget, dict)
        and isinstance(budget.get("base_rank"), int)
        and budget["base_rank"] > len(ranked)
    ):
        errors.append(
            f"{where}.policy_receipt.budget.base_rank cannot exceed ranked_point_ids"
        )
    allowed = {
        "schema_version",
        "space_revision",
        "search_space_state_revision",
        "proposal_set_revision",
        "policy",
        "action",
        "selected_point_id",
        "components",
        "acquisition_score",
        "evidence",
        "ranked_point_ids",
        "budget",
        "experience",
    }
    unknown = sorted(set(receipt) - allowed)
    if unknown:
        errors.append(f"{where}.policy_receipt has unknown fields {unknown}")
    return errors


def validate_ledger(registry: dict[str, Any], ledger: dict[str, Any], *,
                    registry_history: dict | None = None) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_lineage_snapshots(ledger))
    errors.extend(validate_attempt_observations(ledger))
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
    state = ledger.get("search_space_state")
    state_revision = state.get("revision") if isinstance(state, dict) else None
    state_revision_valid = (
        isinstance(state_revision, int)
        and not isinstance(state_revision, bool)
        and state_revision >= 0
    )
    guidance = derive_hypothesis_selection(registry)
    known: set[str] = set()
    frozen_llm_intelligence_score: float | None = None
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
        status = record.get("status")
        if status not in RECORD_STATUSES:
            errors.append(
                f"{where}.status must be one of {sorted(RECORD_STATUSES)}"
            )
        unevaluated_receipt = record.get("unevaluated_receipt")
        if status == "unevaluated":
            receipt_fields = {
                "schema_version",
                "kind",
                "budget",
                "evaluations_done",
                "candidate_objective_attempts",
                "attempt_log",
            }
            actual_receipt_fields = (
                set(unevaluated_receipt)
                if isinstance(unevaluated_receipt, dict)
                else set()
            )
            if (
                not isinstance(unevaluated_receipt, dict)
                or actual_receipt_fields != receipt_fields
            ):
                errors.append(
                    f"{where}.unevaluated_receipt has an invalid shape"
                )
            else:
                receipt_kind = unevaluated_receipt.get("kind")
                receipt_budget = unevaluated_receipt.get("budget")
                receipt_done = unevaluated_receipt.get("evaluations_done")
                common_ok = (
                    unevaluated_receipt.get("schema_version") == 1
                    and unevaluated_receipt.get("candidate_objective_attempts") == 0
                    and isinstance(receipt_done, int)
                    and not isinstance(receipt_done, bool)
                    and receipt_done >= 0
                    and unevaluated_receipt.get("attempt_log")
                    == "evaluation_attempts.jsonl"
                )
                evals_ok = (
                    receipt_kind == "budget_exhausted_before_candidate_attempt"
                    and isinstance(receipt_budget, int)
                    and not isinstance(receipt_budget, bool)
                    and receipt_budget > 0
                    and receipt_done >= receipt_budget
                )
                time_ok = (
                    receipt_kind == "time_budget_reached_before_candidate_attempt"
                    and receipt_budget is None
                )
                if not common_ok or not (evals_ok or time_ok):
                    errors.append(
                        f"{where}.unevaluated_receipt is not a valid "
                        "stop-condition zero-attempt receipt"
                    )
            if record.get("final_best_score") is not None or int(
                record.get("trials_attempted") or 0
            ) != 0:
                errors.append(
                    f"{where} unevaluated records must have no score and zero "
                    "objective attempts"
                )
        elif unevaluated_receipt is not None:
            errors.append(
                f"{where}.unevaluated_receipt is only legal for status unevaluated"
            )
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
        point_errors = validate_record_point(record.get("semantic_point"), registry, registry_history)
        errors.extend(f"{where}: {error}" for error in point_errors)
        record_registry = registry if point_errors else registry_for_point(
            record["semantic_point"], registry, registry_history)
        errors.extend(_validate_policy_receipt(record, where))
        # Replay the overlay at the record's historical selection revision, so
        # later pruning never invalidates an earlier admitted record.
        receipt = record.get("policy_receipt")
        if (
            isinstance(receipt, dict)
            and receipt.get("schema_version") == 8
            and receipt.get("space") != space_receipt(record_registry)
        ):
            errors.append(
                f"{where}.policy_receipt.space must equal the frozen space receipt"
            )
        if isinstance(receipt, dict) and receipt.get("schema_version") in {6, 7}:
            policy = receipt.get("policy")
            config = policy.get("config") if isinstance(policy, dict) else None
            score = (
                config.get("llm_intelligence_score")
                if isinstance(config, dict)
                else None
            )
            if (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
                and 0.0 <= float(score) <= 100.0
            ):
                if frozen_llm_intelligence_score is None:
                    frozen_llm_intelligence_score = float(score)
                elif not math.isclose(
                    float(score),
                    frozen_llm_intelligence_score,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    errors.append(
                        f"{where}.policy_receipt.policy.config."
                        "llm_intelligence_score must stay fixed at "
                        f"{frozen_llm_intelligence_score:g} after the first "
                        "current-schema admission"
                    )
        record_revision = (
            receipt.get("search_space_state_revision")
            if isinstance(receipt, dict)
            else None
        )
        if (
            isinstance(record_revision, int)
            and not isinstance(record_revision, bool)
            and state_revision_valid
        ):
            if not 0 <= record_revision <= state_revision:
                errors.append(
                    f"{where}.policy_receipt.search_space_state_revision must be in "
                    f"[0, {state_revision}]"
                )
            else:
                runtime = replay_search_space_state(record_registry, state, revision=record_revision)
                effective = compose_effective_selection(record_registry, derive_hypothesis_selection(record_registry), runtime)
                eligibility = validate_point_eligibility(
                    record.get("semantic_point"), record_registry, effective
                )
                errors.extend(f"{where}: {error}" for error in eligibility)
                if isinstance(receipt, dict) and receipt.get("schema_version") in {6, 7, 8}:
                    budget = receipt.get("budget")
                    point = record.get("semantic_point")
                    if isinstance(point, dict) and receipt.get("schema_version") == 6:
                        selected = selected_assignments(point)
                        expected_lane = (
                            "deprioritized"
                            if any(
                                effective.get(hypothesis_id, {}).get("effective_status")
                                == "deprioritized"
                                for hypothesis_id in selected.values()
                            )
                            else "active"
                        )
                        if (
                            isinstance(budget, dict)
                            and budget.get("selected_lane") != expected_lane
                        ):
                            errors.append(
                                f"{where}.policy_receipt.budget.selected_lane must "
                                "match the point's effective status at its selection revision"
                            )
                    if (
                        isinstance(budget, dict)
                        and budget.get("selection_index") != index + 1
                    ):
                        errors.append(
                            f"{where}.policy_receipt.budget.selection_index must "
                            f"equal the one-based admission index {index + 1}"
                        )
                    experience_receipt = receipt.get("experience")
                    if isinstance(experience_receipt, dict):
                        updated_at_run = experience_receipt.get("updated_at_run")
                        if updated_at_run is not None and updated_at_run not in known:
                            errors.append(
                                f"{where}.policy_receipt.experience.updated_at_run "
                                "must reference an earlier record"
                            )
                        evidence_run_ids = experience_receipt.get(
                            "evidence_run_ids"
                        )
                        if isinstance(evidence_run_ids, list):
                            unknown_evidence = sorted(set(evidence_run_ids) - known)
                            if unknown_evidence:
                                errors.append(
                                    f"{where}.policy_receipt.experience evidence "
                                    f"must reference earlier records {unknown_evidence}"
                                )
                        evidence_edge_ids = experience_receipt.get(
                            "evidence_edge_ids"
                        )
                        if isinstance(evidence_edge_ids, list):
                            earlier_edge_ids = {
                                edge.get("edge_id")
                                for prior_record in records[:index]
                                if isinstance(prior_record, dict)
                                for edge in prior_record.get("semantic_edges", [])
                                if isinstance(edge, dict)
                                and isinstance(edge.get("edge_id"), str)
                            }
                            unknown_edges = sorted(
                                set(evidence_edge_ids) - earlier_edge_ids
                            )
                            if unknown_edges:
                                errors.append(
                                    f"{where}.policy_receipt.experience evidence "
                                    f"must reference earlier edges {unknown_edges}"
                                )
        errors.extend(validate_semantic_edges(records[:index], record, registry, registry_history=registry_history))
        errors.extend(
            f"{where}: {error}"
            for error in validate_parameter_transfer_binding(ledger, record)
        )
        if isinstance(receipt, dict) and receipt.get("schema_version") in {6, 7}:
            experience_receipt = receipt.get("experience")
            conditioning = (
                experience_receipt.get("conditioning")
                if isinstance(experience_receipt, dict)
                else None
            )
            prior_by_id = {
                str(prior.get("run_id")): prior
                for prior in records[:index]
                if isinstance(prior, dict)
            }
            parent_points = [
                prior_by_id[parent].get("semantic_point")
                for parent in parents
                if parent in prior_by_id
                and isinstance(prior_by_id[parent].get("semantic_point"), dict)
            ]
            expected_relations = acquisition_target_relations(
                record.get("semantic_point"),
                None if op == "fresh" else parent_points,
            )
            for item in conditioning if isinstance(conditioning, list) else []:
                if (
                    not isinstance(item, dict)
                    or expected_relations.get(item.get("target_id"))
                    != item.get("proposal_relation")
                ):
                    errors.append(
                        f"{where}.policy_receipt.experience conditioning "
                        "does not match the admitted parent-to-child move"
                    )
            errors.extend(
                f"{where}.policy_receipt.experience: {error}"
                for error in validate_conditioning_against_ledger(
                    conditioning,
                    {"records": records[:index]},
                )
            )
        if valid_run_id:
            known.add(run_id)
    errors.extend(validate_search_space_state(registry, ledger))
    experience = ledger.get("experience")
    if experience not in (None, {}):
        errors.extend(validate_experience(experience, registry, ledger))
    return errors


def validate_registry(
    registry: dict[str, Any],
    *,
    ledger: dict[str, Any] | None = None,
    retrieval_manifest: dict[str, Any] | None = None,
    manifest_dir: Path | None = None,
    manifest_path: Path | None = None,
    catalog: dict[str, Any] | None = None,
    dimension_strategy: str = DEFAULT_DIMENSION_STRATEGY,
    baseline_mechanisms: dict[str, Any] | None = None,
    number_gate: bool = False,
    registry_history: dict | None = None,
) -> list[str]:
    errors = validate_space_core(
        registry, catalog=catalog, dimension_strategy=dimension_strategy
    )
    source_errors, source_by_id = _validate_sources(
        registry, retrieval_manifest, manifest_dir=manifest_dir, manifest_path=manifest_path
    )
    errors.extend(source_errors)
    errors.extend(_validate_hypotheses(registry, source_by_id))
    errors.extend(validate_reserves(registry))
    errors.extend(
        _validate_baseline_mechanism_disjointness(registry, baseline_mechanisms)
    )
    errors.extend(_validate_relations_evidence(registry, source_by_id))
    errors.extend(_validate_provenance_refs(registry, source_by_id))
    errors.extend(_validate_guidance(registry, source_by_id))
    if number_gate and retrieval_manifest is not None and manifest_dir is not None:
        # Generation-path runs only: the researcher must ground every result
        # number in a cited source's retained content.  Pre-seeded frozen
        # backgrounds never get this gate (the audit's judge-side rules carry
        # their number-absence cases as warnings instead).
        errors.extend(
            _number_presence_gate_errors(registry, retrieval_manifest, manifest_dir)
        )
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
        errors.extend(validate_ledger(registry, ledger, registry_history=registry_history))
    return errors


def source_verification(
    registry: dict[str, Any], retrieval_manifest: dict[str, Any] | None
) -> dict[str, str]:
    """Join registry sources to manifest-derived verification tiers.

    Each source id maps to the tier of its tool-recorded receipt
    (``snippet_only``/``preview``/``section``/``full_text``), joined by
    ``canonical_key(url)``.  A source with no receipt is absent — with a
    manifest supplied, ``validate_registry`` rejects it.
    """
    if retrieval_manifest is None:
        return {}
    tiers = verification_statuses(retrieval_manifest)
    result: dict[str, str] = {}
    for source in registry.get("sources", []):
        if not isinstance(source, dict):
            continue
        source_id = source.get("id")
        url = source.get("url")
        if not isinstance(source_id, str) or not _nonempty(url):
            continue
        tier = tiers.get(canonical_key(url))
        if tier is not None:
            result[source_id] = tier
    return result


# Audit text and the result-number precheck.  One shared text per registry
# item feeds mapping identity, the judge payload, and the validate-time
# number gate; the deterministic matcher below answers whether that text's
# result-type numbers survive in the tool-recorded retained content.

AUDIT_TEXT_FIELDS = ("claim", "scope", "credibility_rationale", "reopen_when")

_ARXIV_ID_RE = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\\*%?")
_RANGE_SEPARATOR_RE = re.compile(r"[-–~]")
_TOKEN_EDGE_PUNCT = "\"'`“”‘’.,;:!?()[]{}<>*_#$-–~"
# HTML tags are delimiters: deepxiv stores table cells as ``<td>0.839</td>``.
# Pin the tag shape; a bare ``<...>`` also spans comparison and LaTeX intervals
# (``<0.05; RUS: -0.004, p>``, ``$\\mu<1.4$ ... $>``) and swallows their numbers.
_NUMBER_SPLIT_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^>]*)?/?>|\s+")


def audit_text(item: dict[str, Any]) -> str:
    """The shared audit text of one claim-bearing registry item.

    Present fields are taken in a fixed order, each annotated with its field
    name; the structured ``scope`` facet dict is serialized deterministically
    with facets sorted by facet name.  Every consumer — mapping identity, the
    number precheck, the contract gate, the judge payload — reads this text.
    """
    lines: list[str] = []
    for field in AUDIT_TEXT_FIELDS:
        value = item.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if isinstance(value, dict):
            facets = []
            for facet, tags in sorted(value.items()):
                tag_list = tags if isinstance(tags, list) else [tags]
                facets.append(f"{facet}: {', '.join(str(tag) for tag in tag_list)}")
            rendered = "; ".join(facets)
        else:
            rendered = str(value)
        lines.append(f"{field}: {rendered}")
    return "\n".join(lines)


def _number_tokens(raw: str) -> list[str]:
    """Result-type number tokens inside one whitespace-delimited raw token.

    A result-type number carries a decimal point or a ``%``; LaTeX-escaped
    percent counts too — deepxiv stores LaTeX-ish text JSON-wrapped, so the
    raw content carries a backslash run before the ``%`` (``$(97\\%)$``).  Citation-shaped tokens (arXiv ids) never count.  A range token
    splits into its endpoints with the unit inherited (``94-95%`` → ``94%``,
    ``95%``); a verbatim range in a source hits the same endpoints, so no
    separate range match is needed.
    """
    token = raw.strip(_TOKEN_EDGE_PUNCT)
    if not token or not any(char.isdigit() for char in token):
        return []
    if _ARXIV_ID_RE.fullmatch(token):
        return []
    parts = _RANGE_SEPARATOR_RE.split(token)
    if len(parts) == 2 and all(_NUMBER_RE.fullmatch(part) for part in parts):
        endpoints = list(parts)
        for index, other in ((0, 1), (1, 0)):
            if endpoints[other].endswith("%") and not (
                endpoints[index].endswith("%") or "." in endpoints[index]
            ):
                endpoints[index] += "%"
        return [
            endpoint
            for endpoint in endpoints
            if ("." in endpoint or "%" in endpoint)
            and _ARXIV_ID_RE.fullmatch(endpoint) is None
        ]
    if _NUMBER_RE.fullmatch(token) and ("." in token or "%" in token):
        return [token]
    return []


def extract_result_numbers(text: str) -> list[str]:
    """Result-type number tokens in ``text``, deduped in first-appearance order.

    HTML tags are delimiters, so a number glued inside markup (``<td>0.839</td>``)
    still counts as retained content.
    """
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _NUMBER_SPLIT_RE.split(text):
        for token in _number_tokens(raw):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


def _normalized_digits(token: str) -> tuple[str, int] | None:
    """Canonical ``(digits, exponent)`` with value ``int(digits) * 10**exponent``.

    The decimal point, a trailing ``%`` (÷100, any backslash run before it
    ignored — LaTeX/JSON escaping), leading zeros, and trailing zeros all
    fold into the pair, so ``0.3843`` and ``38.43%`` normalize equal and
    ``0.380`` equals ``0.38``.
    """
    match = re.fullmatch(r"(\d+)(?:\.(\d+))?\\*(%)?", token)
    if match is None:
        return None
    integer, fraction, percent = match.groups()
    exponent = -len(fraction or "") - (2 if percent else 0)
    digits = (integer + (fraction or "")).lstrip("0")
    trailing = len(digits) - len(digits.rstrip("0"))
    if trailing:
        digits = digits[:-trailing]
        exponent += trailing
    if not digits:
        return ("0", 0)
    return digits, exponent


def result_number_matches(claim_token: str, source_token: str) -> bool:
    """Whether one claim number is supported by one source number.

    Normalized equality passes (``0.3843`` ↔ ``38.43%``).  A claim carrying
    *fewer* digits than the source passes as a rounding truncation
    (``0.38`` vs source ``0.3843``); a claim carrying *more* is fabricated
    precision and fails (``0.3843`` vs source ``0.38``).
    """
    claim = _normalized_digits(claim_token)
    source = _normalized_digits(source_token)
    if claim is None or source is None:
        return False
    claim_digits, claim_exponent = claim
    source_digits, source_exponent = source
    if claim_digits == source_digits:
        return claim_exponent == source_exponent
    return source_digits.startswith(claim_digits) and claim_exponent == (
        source_exponent + len(source_digits) - len(claim_digits)
    )


def _presence(claim_tokens: list[str], source_tokens: list[str]) -> tuple[str, list[str]]:
    """present/absent/none plus the claim tokens no source token supports."""
    if not claim_tokens:
        return "none", []
    missing = [
        token
        for token in claim_tokens
        if not any(result_number_matches(token, source) for source in source_tokens)
    ]
    return ("absent" if missing else "present"), missing


def _best_substantive_visit(manifest: dict[str, Any], key: str) -> dict[str, Any] | None:
    """The highest-tier successful substantive visit for one canonical key."""
    best: dict[str, Any] | None = None
    best_rank = -1
    for visit in manifest.get("visits", []):
        if not isinstance(visit, dict) or visit.get("status") != "success":
            continue
        if visit.get("canonical_key") != key:
            continue
        tier = _VIEW_VERIFICATION.get(str(visit.get("view")))
        if tier is None:
            continue
        rank = _VERIFICATION_RANK[tier]
        if rank > best_rank:
            best, best_rank = visit, rank
    return best


def _cited_source_contents(
    item: dict[str, Any],
    sources_by_id: dict[str, dict[str, Any]],
    tiers: dict[str, str],
    manifest: dict[str, Any],
    manifest_dir: Path,
) -> dict[str, str]:
    """source_id → retained content for cited sources at tier preview or better.

    A source's retained content is every successful substantive visit receipt,
    not just the best one: several visits can sit at the same tier (e.g.
    section views of one paper) with the number carried by any of them, so
    their contents are concatenated in manifest order for the matcher.
    """
    contents: dict[str, str] = {}
    for link in item.get("evidence") or []:
        if not isinstance(link, dict):
            continue
        source_id = link.get("source_id")
        if not isinstance(source_id, str) or source_id in contents:
            continue
        source = sources_by_id.get(source_id)
        url = source.get("url") if isinstance(source, dict) else None
        if not _nonempty(url):
            continue
        key = canonical_key(url)
        if _VERIFICATION_RANK.get(tiers.get(key, ""), -1) < _VERIFICATION_RANK["preview"]:
            continue
        retained: list[str] = []
        for visit in manifest.get("visits", []):
            if not isinstance(visit, dict) or visit.get("status") != "success":
                continue
            if visit.get("canonical_key") != key:
                continue
            if _VIEW_VERIFICATION.get(str(visit.get("view"))) is None:
                continue
            content = resolve_visit_content(visit, manifest_dir)
            if content:
                retained.append(content)
        if retained:
            contents[source_id] = "\n".join(retained)
    return contents


def item_number_presence(
    item: dict[str, Any],
    registry: dict[str, Any],
    manifest: dict[str, Any],
    manifest_dir: Path,
) -> dict[str, Any]:
    """Item-level result-number presence across all cited tier≥preview sources.

    The validate-time number gate consumes this: ``missing`` names the audit
    text tokens that no qualifying cited source's retained content contains.
    """
    tokens = extract_result_numbers(audit_text(item))
    sources_by_id = {
        source.get("id"): source
        for source in registry.get("sources", [])
        if isinstance(source, dict)
    }
    contents = _cited_source_contents(
        item, sources_by_id, verification_statuses(manifest), manifest, manifest_dir
    )
    source_tokens = [
        token for content in contents.values() for token in extract_result_numbers(content)
    ]
    presence, missing = _presence(tokens, source_tokens)
    return {"presence": presence, "tokens": tokens, "missing": missing}


def _number_presence_gate_errors(
    registry: dict[str, Any],
    manifest: dict[str, Any],
    manifest_dir: Path,
) -> list[str]:
    """The item-level result-number gate, one error per missing token.

    Every result-type number in a claim-bearing item's audit text must appear
    in the retained content of at least one of the item's tier≥preview cited
    sources.  Retained content is tool receipts, so a visit that never
    surfaced the number does not satisfy the gate; the error names the token,
    the item, and the cited candidates, and points at the three ways out.
    """
    errors: list[str] = []
    items: list[tuple[str, dict[str, Any]]] = []
    for dimension in registry.get("dimensions", []):
        if not isinstance(dimension, dict):
            continue
        for hypothesis in dimension.get("hypotheses", []):
            if isinstance(hypothesis, dict):
                items.append(("hypothesis", hypothesis))
    for item in registry.get("guidance", []):
        if isinstance(item, dict):
            items.append(("guidance", item))
    for item_kind, item in items:
        # Source-free synthesis states a proposed mechanism or runtime
        # observation, not a number attributed to literature. Its provenance
        # and unverified credibility remain subject to _validate_hypotheses.
        # A synthesis with any citation still owes the full evidence contract.
        if item_kind == "hypothesis" and item.get("kind") == "synthesis_probe" and not item.get("evidence"):
            continue
        result = item_number_presence(item, registry, manifest, manifest_dir)
        candidates = list(
            dict.fromkeys(
                str(link.get("source_id"))
                for link in item.get("evidence") or []
                if isinstance(link, dict) and link.get("source_id")
            )
        )
        for token in result["missing"]:
            errors.append(
                f"{item_kind} {item.get('id')} number {token} is in no cited "
                "source's retained content at preview tier or better "
                f"(candidates: {', '.join(candidates) or 'none'}); visit a "
                "cited source containing the number, cite a different source "
                "that carries it, or downgrade the claim to a qualitative "
                "statement"
            )
    return errors


def mapping_number_presence(
    item: dict[str, Any],
    link: dict[str, Any],
    registry: dict[str, Any],
    manifest: dict[str, Any],
    manifest_dir: Path,
) -> dict[str, Any]:
    """Per-mapping number presence, coverage facts, and the coverage routing.

    ``number_presence`` carries the item-level verdict (the number must ride
    on *some* cited tier≥preview source of the item) next to this source's
    own; source-level absence alone does not make a mapping unfaithful, since
    the number may be carried by a sibling link.  ``coverage.routing`` is the
    pinned routing fact: only a ``full_text`` receipt without a store cap hit
    is ``sufficient`` (an absence-based negative may then be judged
    unfaithful); section/preview or truncated coverage is ``partial``, where
    an absence-based negative is at most unverifiable.
    """
    tokens = extract_result_numbers(audit_text(item))
    sources_by_id = {
        source.get("id"): source
        for source in registry.get("sources", [])
        if isinstance(source, dict)
    }
    tiers = verification_statuses(manifest)
    contents = _cited_source_contents(item, sources_by_id, tiers, manifest, manifest_dir)
    item_presence, item_missing = _presence(
        tokens,
        [token for content in contents.values() for token in extract_result_numbers(content)],
    )
    this_content = contents.get(str(link.get("source_id")))
    this_presence, this_missing = _presence(
        tokens, extract_result_numbers(this_content) if this_content else []
    )

    source = sources_by_id.get(link.get("source_id"))
    url = source.get("url") if isinstance(source, dict) else None
    key = canonical_key(url) if _nonempty(url) else ""
    tier = tiers.get(key, "none")
    visit = _best_substantive_visit(manifest, key) if key else None
    best_content = (
        resolve_visit_content(visit, manifest_dir) if isinstance(visit, dict) else None
    )
    store_cap_hit = bool(visit.get("store_cap_hit")) if isinstance(visit, dict) else False
    return {
        "number_presence": {"item": item_presence, "this_source": this_presence},
        "missing_tokens": {"item": item_missing, "this_source": this_missing},
        "coverage": {
            "tier": tier,
            "routing": (
                "sufficient" if tier == "full_text" and not store_cap_hit else "partial"
            ),
            # Coverage facts describe the best visit （决策 6 attribution), not
            # the concatenated retained content the matcher reads.
            "retained_chars": len(best_content) if best_content is not None else None,
            "store_cap_hit": store_cap_hit,
            "content_file": visit.get("content_file") if isinstance(visit, dict) else None,
            "view": visit.get("view") if isinstance(visit, dict) else None,
            "section": visit.get("section") if isinstance(visit, dict) else None,
        },
    }


def validate_background_markdown(
    path: Path,
    registry: dict[str, Any],
    policy_profile: dict[str, Any] | None = None,
) -> list[str]:
    """Keep the human hierarchy and structured guidance aligned with JSON."""
    text = path.read_text(errors="replace")
    marker = re.search(r"^## Search space registry\s*$", text, flags=re.MULTILINE)
    human = text[: marker.start()] if marker else text
    errors: list[str] = []
    if policy_profile is not None:
        # Product-side competition-policy scan over the full artifact (the
        # text is the artifact's own content, so echoing it here is safe).
        for hit in scan_artifact_text(text, policy_profile):
            errors.append(
                "competition policy: background.md matches "
                f"{', '.join(hit['categories'])} near {hit['context']!r}"
            )
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
    registry: dict[str, Any],
    ledger: dict[str, Any] | None,
    *,
    max_hypotheses: int,
    retrieval_manifest: dict[str, Any] | None = None,
    registry_history: dict | None = None,
) -> dict[str, Any]:
    coverage = coverage_from_records(registry, (ledger or {}).get("records", []), registry_history=registry_history)
    counts = {
        item["hypothesis_id"]: item["count"]
        for dimension in coverage["dimensions"]
        for item in dimension["hypotheses"]
    }
    state = (ledger or {}).get("search_space_state")
    if not isinstance(state, dict):
        state = empty_search_space_state()
    state_revision = state.get("revision")
    if (
        not isinstance(state_revision, int)
        or isinstance(state_revision, bool)
        or state_revision < 0
    ):
        state_revision = 0
    runtime = replay_search_space_state(registry, state)
    effective = compose_effective_selection(
        registry, derive_hypothesis_selection(registry), runtime
    )
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
                    "selection": effective.get(hypothesis_id),
                    "coverage_count": counts.get(hypothesis_id, 0),
                }
            )
        hypotheses.sort(
            key=lambda item: (
                {"active": 0, "deprioritized": 1, "pruned": 2}.get(
                    (item.get("selection") or {}).get("effective_status"), 4
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
                "runtime_status": runtime["dimensions"].get(dimension.get("id"), "active"),
                "selection_reason": dimension.get("selection_reason"),
                "hypotheses": hypotheses[:max_hypotheses],
                "omitted_hypotheses": max(0, len(hypotheses) - max_hypotheses),
            }
        )
    view = {
        "ok": True,
        "space": space_receipt(registry),
        "search_space_state_revision": state_revision,
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
    if retrieval_manifest is not None:
        leads = []
        for entry in unexplored_leads(retrieval_manifest)[:10]:
            title = str(entry.get("title") or "")
            tldr = entry.get("tldr")
            leads.append(
                {
                    "title": title[:120],
                    "url": entry.get("url"),
                    "best_rank": entry.get("best_rank"),
                    "tldr": str(tldr)[:200] if tldr else None,
                }
            )
        view["unexplored_leads"] = leads
    return view


TARGET_EVIDENCE_REQUIRED = {
    "target_id",
    "evaluation_state",
    "assessment",
    "recommended_status",
    "claim",
    "evidence_run_ids",
    "evidence_edge_ids",
    "comparator_coverage",
    "confidence",
    "uncertainty",
}
TARGET_EVIDENCE_OPTIONAL = {"reopen_when"}
TARGET_ASSESSMENTS = {"unknown", "promising", "mixed", "unpromising"}
TARGET_RECOMMENDATIONS = {"active", "deprioritized", "pruned"}
EXPERIENCE_CONFIDENCE = {"low", "med", "high"}
# Schema 4 split direct comparators by tuning depth in every target's
# `comparator_coverage`: `direct_tuned_edges` and `direct_lightly_tuned_edges`
# sit ahead of the screening-depth `direct_noncrash_edges`. Older shapes stay
# readable: the three-key and four-key coverages normalize forward with the
# newer depth buckets at 0.
EXPERIENCE_SCHEMA_VERSION = 4
READABLE_EXPERIENCE_SCHEMA_VERSIONS = {3, 4}


def _validate_target_evidence(
    items: Any,
    *,
    field: str,
    target_kind: str,
    registry: dict[str, Any],
    ledger: dict[str, Any],
    limit: int,
    enforce_judgments: bool = True,
) -> list[str]:
    """Validate one bounded two-level belief collection against cited receipts.

    Comparator counts and the evaluation state are recomputed from the cited
    run/edge ids by :mod:`semantic_evidence`; the authored values must match
    exactly.  Recommendation gates apply identically to both levels.
    """
    if not isinstance(items, list):
        return [f"experience.{field} must be a list"]
    errors: list[str] = []
    if len(items) > limit:
        errors.append(f"experience.{field} may contain at most {limit} items")
    known_targets = (
        dimension_map(registry) if target_kind == "dimension" else hypothesis_map(registry)
    )
    records = {
        str(record.get("run_id")): record
        for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("run_id") is not None
    }
    terminal_ids = {
        run_id
        for run_id, record in records.items()
        if record.get("status") in {"keep", "discard", "crash"}
    }
    index = edge_index(ledger)
    seen: set[str] = set()
    for position, item in enumerate(items):
        where = f"experience.{field}[{position}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be an object")
            continue
        missing = sorted(TARGET_EVIDENCE_REQUIRED - set(item))
        unknown_fields = sorted(set(item) - TARGET_EVIDENCE_REQUIRED - TARGET_EVIDENCE_OPTIONAL)
        if missing:
            errors.append(f"{where} is missing fields {missing}")
        if unknown_fields:
            errors.append(f"{where} has unknown fields {unknown_fields}")
        target_id = item.get("target_id")
        if not _nonempty(target_id):
            errors.append(f"{where}.target_id must be a non-empty {target_kind} id")
            continue
        target = f"{where} target '{target_id}'"
        if target_id in seen:
            errors.append(f"{target} duplicates an earlier {field} entry")
        seen.add(target_id)
        if target_id not in known_targets:
            errors.append(f"{target} is not a known {target_kind} id")
            continue
        if not _nonempty(item.get("claim")) or len(item.get("claim", "")) > 600:
            errors.append(f"{target}.claim must be non-empty and at most 600 characters")
        if not _nonempty(item.get("uncertainty")) or len(item.get("uncertainty", "")) > 600:
            errors.append(f"{target}.uncertainty must be non-empty and at most 600 characters")
        if "reopen_when" in item and (
            not _nonempty(item.get("reopen_when"))
            or len(item.get("reopen_when", "")) > 600
        ):
            errors.append(
                f"{target}.reopen_when must be non-empty and at most 600 characters when present"
            )

        raw_edge_ids = item.get("evidence_edge_ids")
        edge_ids = (
            [str(edge_id) for edge_id in raw_edge_ids if isinstance(edge_id, str)]
            if isinstance(raw_edge_ids, list)
            else []
        )
        if (
            not isinstance(raw_edge_ids, list)
            or len(raw_edge_ids) > MAX_EDGES_PER_TARGET
            or len(edge_ids) != len(raw_edge_ids)
            or len(set(edge_ids)) != len(edge_ids)
            or any(edge_id not in index for edge_id in edge_ids)
        ):
            errors.append(
                f"{target}.evidence_edge_ids must contain "
                f"0–{MAX_EDGES_PER_TARGET} unique persisted semantic edge ids"
            )
        touching_endpoints: set[str] = set()
        for edge_id in edge_ids:
            receipt = index.get(edge_id)
            if receipt is None:
                continue
            touches = any(
                comparator_coverage(
                    ledger, [edge_id], target_kind=target_kind, target_id=target_id
                ).values()
            )
            if not touches:
                errors.append(
                    f"{target}.evidence_edge_ids cites edge '{edge_id}' "
                    "that does not touch the target"
                )
            else:
                touching_endpoints.add(str(receipt.get("parent_run_id")))
                touching_endpoints.add(str(receipt.get("child_run_id")))

        raw_run_ids = item.get("evidence_run_ids")
        run_ids = (
            [str(run_id) for run_id in raw_run_ids if isinstance(run_id, str)]
            if isinstance(raw_run_ids, list)
            else []
        )
        if (
            not isinstance(raw_run_ids, list)
            or len(raw_run_ids) > MAX_RUNS_PER_TARGET
            or len(run_ids) != len(raw_run_ids)
            or len(set(run_ids)) != len(run_ids)
            or any(run_id not in terminal_ids for run_id in run_ids)
        ):
            errors.append(
                f"{target}.evidence_run_ids must contain "
                f"0–{MAX_RUNS_PER_TARGET} unique terminal ledger run ids"
            )
        for run_id in run_ids:
            record = records.get(run_id)
            if run_id in terminal_ids and run_id not in touching_endpoints:
                point = record.get("semantic_point") if record is not None else None
                selected = selected_assignments(point) if isinstance(point, dict) else {}
                bears_target = (
                    target_id in selected.values()
                    if target_kind == "hypothesis"
                    else target_id in selected
                )
                if not bears_target:
                    errors.append(
                        f"{target}.evidence_run_ids cites run '{run_id}' "
                        "whose point does not bear the target"
                    )

        expected_coverage = comparator_coverage(
            ledger, edge_ids, target_kind=target_kind, target_id=target_id
        )
        coverage = normalize_coverage(item.get("comparator_coverage"))
        if coverage is None:
            errors.append(
                f"{target}.comparator_coverage must hold non-negative integer "
                f"counts for {', '.join(COVERAGE_KEYS)}"
            )
        elif coverage != expected_coverage:
            errors.append(
                f"{target}.comparator_coverage must equal the recomputed "
                "coverage of its cited target-touching edges"
            )

        expected_state = target_evaluation_state(
            ledger,
            target_kind=target_kind,
            target_id=target_id,
            evidence_run_ids=run_ids,
            evidence_edge_ids=edge_ids,
        )
        state = item.get("evaluation_state")
        if state not in EVALUATION_STATES:
            errors.append(
                f"{target}.evaluation_state must be unevaluated, failed, "
                "observed, or comparator_covered"
            )
        elif state != expected_state:
            errors.append(
                f"{target}.evaluation_state does not match the mechanical "
                "state of its cited evidence"
            )

        assessment = item.get("assessment")
        if assessment not in TARGET_ASSESSMENTS:
            errors.append(
                f"{target}.assessment must be unknown, promising, mixed, or unpromising"
            )
        recommended = item.get("recommended_status")
        if recommended not in TARGET_RECOMMENDATIONS:
            errors.append(
                f"{target}.recommended_status must be active, deprioritized, or pruned"
            )
        confidence = item.get("confidence")
        if confidence not in EXPERIENCE_CONFIDENCE:
            errors.append(f"{target}.confidence must be low, med, or high")

        if not enforce_judgments:
            continue
        depth_bar = _contradiction_depth_bar(expected_coverage)
        mechanical_direction = mechanical_gain_direction(
            ledger,
            target_kind=target_kind,
            target_id=target_id,
            evidence_edge_ids=edge_ids,
        )
        if state in {"unevaluated", "failed"} and (
            recommended != "active" or assessment != "unknown" or confidence != "low"
        ):
            errors.append(
                f"{target} with evaluation_state '{state}' must keep assessment "
                "unknown, confidence low, and recommended_status active"
            )
        carriers = (
            hypothesis_carriers(ledger, target_id=target_id, edge_ids=edge_ids)
            if target_kind == "hypothesis"
            else {"negative": 0, "positive": 0}
        )
        carrier_demote = carriers["negative"] >= 2 and carriers["positive"] == 0
        carrier_prune = carriers["negative"] >= 3 and carriers["positive"] == 0
        strict_demote = state == "comparator_covered" and depth_bar
        if recommended == "deprioritized" and not (
            assessment == "unpromising"
            and confidence in {"med", "high"}
            and (strict_demote or carrier_demote)
            and _nonempty(item.get("reopen_when"))
        ):
            errors.append(
                f"{target}.recommended_status deprioritized requires assessment "
                "unpromising, confidence med or high, a non-empty reopen_when, "
                "and either comparator_covered evaluation_state with at least "
                "two direct tuned edges, or at least three direct edges at "
                "tuned_lightly or deeper, or at least two independent negative "
                "carrier contexts with no positive context"
            )
        if recommended == "pruned" and not (
            assessment == "unpromising"
            and confidence == "high"
            and (strict_demote or carrier_prune)
            and _nonempty(item.get("reopen_when"))
        ):
            errors.append(
                f"{target}.recommended_status pruned requires assessment "
                "unpromising, confidence high, a non-empty reopen_when, and "
                "either comparator_covered evaluation_state with at least two "
                "direct tuned edges, or at least three direct edges at "
                "tuned_lightly or deeper, or at least three independent "
                "negative carrier contexts with no positive context"
            )
        if assessment == "promising" and not strict_demote:
            errors.append(
                f"{target} assessment promising requires comparator_covered "
                "evaluation_state with at least two direct tuned edges, or at "
                "least three direct edges at tuned_lightly or deeper"
            )
        if assessment == "unpromising" and not (strict_demote or carrier_demote):
            errors.append(
                f"{target} assessment unpromising requires comparator_covered "
                "evaluation_state with at least two direct tuned edges, or at "
                "least three direct edges at tuned_lightly or deeper, or at "
                "least two independent negative carrier contexts with no "
                "positive context"
            )
        if (
            target_kind == "hypothesis"
            and assessment in {"promising", "unpromising"}
            and not (assessment == "unpromising" and carrier_demote)
        ):
            required_direction = (
                "positive" if assessment == "promising" else "negative"
            )
            if mechanical_direction != required_direction:
                errors.append(
                    f"{target} assessment {assessment} conflicts with the "
                    "direction of its repeated matched semantic-control pairs"
                )
    return errors


def validate_experience(experience: Any, registry: dict[str, Any], ledger: dict[str, Any],
                        *, enforce_judgments: bool = True) -> list[str]:
    """Validate the bounded schema-3 belief snapshot over the durable records.

    Generic collections stay bounded.  The two-level ``dimension_evidence``
    and ``hypothesis_evidence`` collections are replaceable belief: their
    comparator counts and evaluation states are recomputed from cited run and
    edge receipts and conservative recommendation gates are enforced exactly.
    """
    if not isinstance(experience, dict):
        return ["experience must be an object"]
    if experience.get("schema_version") == 5:
        from experience_updates import validation_snapshot, COLLECTIONS
        ids = []
        known_targets = set(dimension_map(registry)) | set(hypothesis_map(registry))
        for field in COLLECTIONS:
            if not isinstance(experience.get(field), list):
                return [f"experience.{field} must be a list"]
            for item in experience[field]:
                if not isinstance(item, dict):
                    return ["experience entries must be objects"]
                ident = item.get("id")
                if not isinstance(ident, str) or not re.fullmatch(r"experience-[1-9][0-9]*", ident) or ident in ids:
                    return ["incremental experience needs unique stable entry ids"]
                ids.append(ident)
                for key, ceiling in (("basis_dag_revision", ledger.get("dag_revision", 0)),
                                     ("judgment_generation", experience.get("generation", 0))):
                    value = item.get(key)
                    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= ceiling:
                        return [f"invalid experience entry {key}"]
                scope = item.get("target_ids")
                if not isinstance(scope, list) or any(not isinstance(t, str) or t not in known_targets for t in scope):
                    return ["experience entry scope must reference registered targets"]
                if field.endswith("_evidence") and not isinstance(item.get("target_id"), str):
                    return ["experience target entry needs target_id"]
        return validate_experience(validation_snapshot({**ledger, "experience": experience}),
                                   registry, ledger, enforce_judgments=False)
    errors: list[str] = []
    allowed_top = {
        "schema_version",
        "updated_at_run",
        "generation",
        "summary",
        "promising_regions",
        "lessons",
        "bottlenecks",
        "dimension_evidence",
        "hypothesis_evidence",
        "dag_revision",
    }
    unknown_top = sorted(set(experience) - allowed_top)
    if unknown_top:
        errors.append(f"experience has unknown fields {unknown_top}")
    if experience.get("schema_version") not in READABLE_EXPERIENCE_SCHEMA_VERSIONS:
        errors.append(
            "experience.schema_version must be "
            f"{sorted(READABLE_EXPERIENCE_SCHEMA_VERSIONS)}"
        )
    generation = experience.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        errors.append("experience.generation must be a non-negative integer")
    if (
        not isinstance(experience.get("summary"), str)
        or len(experience.get("summary", "")) > 2000
    ):
        errors.append(
            "experience.summary must be a display-only string of at most 2000 characters"
        )

    terminal_records = [
        record
        for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("status") in {"keep", "discard", "crash"}
    ]
    terminal_ids = {str(record.get("run_id")) for record in terminal_records}
    updated_at_run = experience.get("updated_at_run")
    empty_cursor = not enforce_judgments and generation == 0 and updated_at_run is None
    if not empty_cursor and (not isinstance(updated_at_run, str) or updated_at_run not in terminal_ids):
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

    errors.extend(
        _validate_target_evidence(
            experience.get("dimension_evidence"),
            field="dimension_evidence",
            target_kind="dimension",
            registry=registry,
            ledger=ledger,
            limit=MAX_DIMENSION_TARGETS,
            enforce_judgments=enforce_judgments,
        )
    )
    errors.extend(
        _validate_target_evidence(
            experience.get("hypothesis_evidence"),
            field="hypothesis_evidence",
            target_kind="hypothesis",
            registry=registry,
            ledger=ledger,
            limit=MAX_HYPOTHESIS_TARGETS,
            enforce_judgments=enforce_judgments,
        )
    )
    return errors


def _validated_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    from space_revisions import load_registry_history
    registry = load_registry(args.background)
    dimension_strategy = resolve_dimension_strategy(args.background)
    catalog = resolve_dimension_catalog(
        args.background, explicit_path=getattr(args, "catalog", None)
    )
    ledger = _load_json(args.ledger) if getattr(args, "ledger", None) else None
    manifest_path = getattr(args, "retrieval_manifest", None)
    manifest = _load_json(manifest_path) if manifest_path else None
    manifest_dir = manifest_path.parent if manifest_path else None
    baseline_mechanisms = (
        _load_json(args.baseline_mechanisms)
        if getattr(args, "baseline_mechanisms", None)
        else None
    )
    errors = validate_registry(
        registry,
        ledger=ledger,
        retrieval_manifest=manifest,
        manifest_dir=manifest_dir,
        manifest_path=manifest_path,
        catalog=catalog,
        dimension_strategy=dimension_strategy,
        baseline_mechanisms=baseline_mechanisms,
        number_gate=getattr(args, "number_gate", False),
        registry_history=load_registry_history(args.background),
    )
    # The product-side scan also applies when only --background is passed in a
    # run dir: resolve the profile from the sibling retrieval manifest path.
    policy_profile = _competition_policy_profile(
        manifest_path
        if manifest_path is not None
        else args.background.parent / "background_retrieval.json"
    )
    errors.extend(
        validate_background_markdown(args.background, load_initial_registry(args.background), policy_profile=policy_profile)
    )
    return registry, ledger, errors


def normalize_registry(registry: dict[str, Any], catalog: dict[str, Any]) -> list[str]:
    """Fill the catalog-derived fields and strip unknown fields in place.

    Returns one warning per stripped field. Validation then only reports what
    the author has to decide.
    """
    registry["catalog"] = catalog_receipt(catalog)
    by_id = {item["id"]: item for item in catalog.get("dimensions", [])}
    warnings: list[str] = []

    def strip(item: dict[str, Any], allowed: set[str], where: str) -> None:
        for key in sorted(set(item) - allowed):
            del item[key]
            warnings.append(f"{where}: stripped unknown field {key!r}")

    for dimension in registry.get("dimensions") or []:
        if not isinstance(dimension, dict):
            continue
        entry = by_id.get(dimension.get("id"))
        if entry is not None:
            dimension["definition"] = entry["definition"]
            dimension["boundary"] = entry["boundary"]
            dimension["catalog_provenance"] = catalog["provenance"]
        strip(dimension, DIMENSION_FIELDS, f"dimension {dimension.get('id')}")
        for hypothesis in dimension.get("hypotheses") or []:
            if isinstance(hypothesis, dict):
                strip(hypothesis, HYPOTHESIS_FIELDS,
                      f"hypothesis {hypothesis.get('id')}")
    return warnings


def _with_registry(text: str, registry: dict[str, Any]) -> str:
    marker = re.search(r"^## Search space registry\s*$", text, flags=re.MULTILINE)
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text[marker.end():], flags=re.DOTALL)
    start, end = marker.end() + fence.start(1), marker.end() + fence.end(1)
    return text[:start] + json.dumps(registry, indent=2, ensure_ascii=False) + text[end:]


def cmd_normalize(args: argparse.Namespace) -> int:
    from space_revisions import load_revision_state
    if load_revision_state(args.background) is not None:
        print(json.dumps({"changed": False, "reason": "the space is already published"}))
        return 0
    text = args.background.read_text()
    registry = load_initial_registry(args.background)
    catalog = resolve_dimension_catalog(args.background, explicit_path=args.catalog)
    before = json.dumps(registry, sort_keys=True)
    warnings = normalize_registry(registry, catalog)
    changed = json.dumps(registry, sort_keys=True) != before
    if changed:
        args.background.write_text(_with_registry(text, registry))
    print(json.dumps({"changed": changed, "warnings": warnings}, indent=2))
    return 0


_UNIT_PATH_RE = re.compile(
    r"(?<![\w.])(?:dimensions\[(\d+)\](?:\.hypotheses\[(\d+)\])?"
    r"|relations\[(\d+)\]|(guidance)\[(\d+)\]|sources\[\d+\])"
)
_UNIT_ID_RE = re.compile(r"\b(?:rel|hyp|dim)-[a-z0-9-]*[a-z0-9]")


def _error_subject(message: str, registry: dict[str, Any]) -> tuple[str, Any] | None:
    """The one registry unit an error names, or None when it names none that
    can be quarantined. A structural path wins over ids; among ids the most
    specific kind wins (relation, then hypothesis, then dimension)."""
    dimensions = registry.get("dimensions") or []
    match = _UNIT_PATH_RE.search(message)
    if match:
        dim, hyp, rel, guidance, guidance_index = match.groups()
        try:
            if rel is not None:
                return "relation", registry["relations"][int(rel)]["id"]
            if guidance is not None:
                return "guidance", int(guidance_index)
            if dim is not None:
                dimension = dimensions[int(dim)]
                if hyp is not None:
                    return "hypothesis", dimension["hypotheses"][int(hyp)]["id"]
                return "dimension", dimension["id"]
        except (IndexError, KeyError, TypeError):
            return None
        return None
    known = {
        "rel": {r.get("id") for r in registry.get("relations") or [] if isinstance(r, dict)},
        "hyp": set(hypothesis_map(registry)),
        "dim": {d.get("id") for d in dimensions if isinstance(d, dict)},
    }
    found = [token for token in _UNIT_ID_RE.findall(message) if token in known[token[:3]]]
    for prefix, kind in (("rel", "relation"), ("hyp", "hypothesis"), ("dim", "dimension")):
        for token in found:
            if token.startswith(prefix):
                return kind, token
    return None


def _quarantine_unit(registry: dict[str, Any], kind: str, target: Any) -> str | None:
    """Drop (or, for binding guidance, downgrade) one unit in place; returns the
    quarantined id, or None when the unit cannot be quarantined."""
    if kind == "guidance":
        item = (registry.get("guidance") or [])[target]
        if not isinstance(item, dict) or item.get("effect") != "deprioritize":
            return None
        item["effect"] = "caution"
        return str(item.get("id"))
    key = "relations" if kind == "relation" else "dimensions"
    items = registry.get(key) or []
    if kind in ("relation", "dimension"):
        kept = [i for i in items if not (isinstance(i, dict) and i.get("id") == target)]
        registry[key] = kept
        return target if len(kept) != len(items) else None
    for dimension in items:  # a hypothesis; its baseline takes the dimension along
        if not isinstance(dimension, dict):
            continue
        if dimension.get("baseline_hypothesis_id") == target:
            return _quarantine_unit(registry, "dimension", dimension.get("id"))
        hypotheses = dimension.get("hypotheses") or []
        kept = [h for h in hypotheses if not (isinstance(h, dict) and h.get("id") == target)]
        if len(kept) != len(hypotheses):
            dimension["hypotheses"] = kept
            return target
    return None


def _unmark(text: str, unit_id: str) -> str:
    """Keep the human view readable while it stops referencing a dropped unit."""
    marker = re.search(r"^## Search space registry\s*$", text, flags=re.MULTILINE)
    human, rest = (text[:marker.start()], text[marker.start():]) if marker else (text, "")
    human = re.sub(rf"^(###\s+)`{re.escape(unit_id)}`\s*$",
                   rf"\1{unit_id} (quarantined)", human, flags=re.MULTILINE)
    return human.replace(f"`{unit_id}`", unit_id) + rest


def cmd_quarantine(args: argparse.Namespace) -> int:
    """Degraded fallback after repair: drop the units that validation errors (or
    ``--drop``) name until the space validates, then publish it only if it does."""
    from space_revisions import load_revision_state
    if load_revision_state(args.background) is not None:
        print(json.dumps({"ok": False, "errors": ["the space is already published"]}))
        return 1
    text = args.background.read_text()
    registry = load_initial_registry(args.background)
    scratch = args.background.with_name(".quarantine-" + args.background.name)
    quarantined: list[str] = []

    def apply(kind: str, target: Any) -> bool:
        nonlocal text
        unit_id = _quarantine_unit(registry, kind, target)
        if unit_id is None:
            return False
        quarantined.append(unit_id)
        if kind != "guidance":
            text = _unmark(text, unit_id)
        return True

    for unit_id in args.drop or []:
        kind = {"rel": "relation", "hyp": "hypothesis", "dim": "dimension"}.get(unit_id[:3])
        if kind is not None:
            apply(kind, unit_id)
    try:
        while True:
            text = _with_registry(text, registry)
            scratch.write_text(text)
            _, _, errors = _validated_inputs(
                argparse.Namespace(**{**vars(args), "background": scratch}))
            if not errors:
                break
            before = text
            # resolve every subject before dropping anything: paths are indices
            subjects = [(_error_subject(message, registry), message) for message in errors]
            known = set(hypothesis_map(registry)) | {
                item.get("id") for key in ("dimensions", "relations")
                for item in registry.get(key) or [] if isinstance(item, dict)}
            for subject, message in subjects:
                if subject is not None:
                    apply(*subject)
                    continue
                # a human-view reference to a unit the registry no longer has
                for token in set(_UNIT_ID_RE.findall(message)) - known:
                    text = _unmark(text, token)
            if _with_registry(text, registry) == before:
                break
    finally:
        scratch.unlink(missing_ok=True)
    ok = not errors and bool(registry.get("dimensions"))
    if ok and quarantined:
        args.background.write_text(text)
    print(json.dumps({"ok": ok, "quarantined": quarantined, "errors": errors}, indent=2))
    return 0 if ok else 1


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
    from space_revisions import load_registry_history
    if not 1 <= args.max_hypotheses <= 32:
        print(
            json.dumps(
                {"ok": False, "errors": ["--max-hypotheses must be in [1, 32]"]},
                separators=(",", ":"),
            )
        )
        return 1
    # Full background/source/Markdown validation is a setup responsibility.
    # This high-frequency command is a bounded projection over the frozen
    # registry and the current helper-owned ledger state.
    registry = load_registry(args.background)
    ledger = _load_json(args.ledger) if args.ledger else None
    manifest = (
        _load_json(args.retrieval_manifest) if args.retrieval_manifest else None
    )
    print(
        json.dumps(
            render_space(
                registry,
                ledger,
                max_hypotheses=args.max_hypotheses,
                retrieval_manifest=manifest,
                registry_history=load_registry_history(args.background),
            ),
            separators=(",", ":"),
        )
    )
    return 0


def cmd_validate_point(args: argparse.Namespace) -> int:
    registry, _, errors = _validated_inputs(args)
    point = _load_json(args.point)
    if not errors:
        errors.extend(validate_point(point, registry))
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
    from space_revisions import load_registry_history
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
    value = derive_semantic_lineage(registry, ledger or {}, limit=args.limit if args.compact else None,
                                    registry_history=load_registry_history(args.background))
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
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


def cmd_target_evidence(args: argparse.Namespace) -> int:
    registry, ledger, errors = _validated_inputs(args)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, separators=(",", ":")))
        return 1
    try:
        view = render_target_evidence(
            registry,
            ledger or {},
            max_dimensions=args.max_dimensions,
            max_hypotheses=args.max_hypotheses,
            max_edges_per_target=args.max_edges_per_target,
            target_ids=args.target_id,
        )
    except SemanticEvidenceError as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, separators=(",", ":")))
        return 1
    print(json.dumps(view, separators=(",", ":")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="print a dimension catalog and its digest")
    catalog.add_argument("--path", type=Path, help="catalog JSON; defaults to the built-in catalog")
    catalog.add_argument("--compact", action="store_true")
    catalog.set_defaults(func=cmd_catalog)

    normalize = sub.add_parser(
        "normalize",
        help="fill catalog-derived fields and strip unknown fields in background.md",
    )
    normalize.add_argument("--background", type=Path, required=True)
    normalize.add_argument("--catalog", type=Path, help="explicit dimension catalog override")
    normalize.set_defaults(func=cmd_normalize)

    validate = sub.add_parser("validate", help="validate a hierarchical background")
    quarantine = sub.add_parser(
        "quarantine",
        help="drop the units validation errors name until the space validates",
    )
    for command in (validate, quarantine):
        command.add_argument("--background", type=Path, required=True)
        command.add_argument("--catalog", type=Path, help="explicit dimension catalog override")
        command.add_argument("--ledger", type=Path)
        command.add_argument("--retrieval-manifest", type=Path)
        command.add_argument(
            "--baseline-mechanisms",
            type=Path,
            help="baseline mechanism inventory for a task with a provided entrypoint; "
            "required whenever [seed].provided resolves",
        )
        command.add_argument(
            "--number-gate",
            action="store_true",
            help="generation-path item-level number gate: every result-type number "
            "in an item's audit text must appear in a tier≥preview cited source's "
            "retained content",
        )
    validate.set_defaults(func=cmd_validate)
    quarantine.add_argument("--drop", action="append",
                            help="a hypothesis/dimension/relation id to drop first")
    quarantine.set_defaults(func=cmd_quarantine)

    render = sub.add_parser("render", help="bounded dimension/hypothesis/coverage view")
    render.add_argument("--background", type=Path, required=True)
    render.add_argument("--ledger", type=Path)
    render.add_argument(
        "--retrieval-manifest",
        type=Path,
        help="retrieval manifest; adds a bounded unexplored_leads section",
    )
    render.add_argument("--max-hypotheses", type=int, default=6)
    render.set_defaults(func=cmd_render)

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
        "validate-experience",
        help="validate a schema-3 experience snapshot against cited receipts",
    )
    experience.add_argument("--background", type=Path, required=True)
    experience.add_argument("--catalog", type=Path, help="explicit dimension catalog override")
    experience.add_argument("--ledger", type=Path, required=True)
    experience.add_argument("--experience", type=Path)
    experience.set_defaults(func=cmd_validate_experience)

    evidence = sub.add_parser(
        "target-evidence",
        help="bounded per-target comparator evidence from persisted semantic edges",
    )
    evidence.add_argument("--background", type=Path, required=True)
    evidence.add_argument("--catalog", type=Path, help="explicit dimension catalog override")
    evidence.add_argument("--ledger", type=Path, required=True)
    evidence.add_argument(
        "--target-id",
        action="append",
        default=None,
        help="exact known dimension/hypothesis id; repeatable; bypasses target-count caps",
    )
    evidence.add_argument("--max-dimensions", type=int, default=16)
    evidence.add_argument("--max-hypotheses", type=int, default=32)
    evidence.add_argument("--max-edges-per-target", type=int, default=5)
    evidence.set_defaults(func=cmd_target_evidence)
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
