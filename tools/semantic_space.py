#!/usr/bin/env python3
"""Deterministic contract helpers for the P2 semantic search space.

This module owns only search-space structure, candidate point validity, stable
revision receipts, and mechanically derived coverage/diffs.  Literature
evidence validation stays in :mod:`background_contract`; runtime eligibility
decisions stay in :mod:`search_space_state`; graph action selection stays in
:mod:`got_select`; acquisition policy stays in :mod:`semantic_search`.

Formally (see ``docs/search-space.md``): a point is an equivalence class of
concrete implementations under the attribution map, and the frozen registry is
a subspace of the full catalog product restricted by validity relations.  The
frozen registry never authors runtime pruning state: every element keeps
``status: active`` here, while evidence-preserving pruning lives only in the
append-only ``ledger.search_space_state`` overlay.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "contracts" / "semantic-dimensions-v1.json"
DEFAULT_DIMENSION_STRATEGY = "catalog_subset"
DIMENSION_STRATEGIES = {DEFAULT_DIMENSION_STRATEGY, "llm_induced"}
INDUCED_CATALOG_FILENAME = "dimension_catalog.json"
SEARCH_SPACE_SCHEMA_VERSION = 3
POINT_SCHEMA_VERSION = 1

DIMENSION_RE = re.compile(r"^dim-[a-z0-9][a-z0-9-]*$")
HYPOTHESIS_RE = re.compile(r"^hyp-[a-z0-9][a-z0-9-]*$")
RELATION_RE = re.compile(r"^rel-[a-z0-9][a-z0-9-]*$")

DIMENSION_MODES = {"searchable", "baseline_only"}
ELEMENT_STATUSES = {"active"}
HYPOTHESIS_KINDS = {"baseline", "evidence_prior", "scope_probe"}
RELATION_TYPES = {"activates", "requires", "excludes"}
PROVENANCE_KINDS = {"catalog", "task_contract", "literature", "agent_synthesis"}


class SemanticSpaceError(ValueError):
    """Malformed or incompatible semantic-search-space data."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    try:
        catalog = json.loads(path.read_text())
    except OSError:
        # Filesystem availability is operational, not evidence that authored
        # catalog content violated the semantic-space contract.
        raise
    except json.JSONDecodeError as exc:
        raise SemanticSpaceError(f"cannot load semantic dimension catalog {path}: {exc}") from exc
    if not isinstance(catalog, dict):
        raise SemanticSpaceError(f"semantic dimension catalog {path} must be an object")
    errors = validate_catalog(catalog)
    if errors:
        raise SemanticSpaceError("invalid semantic dimension catalog: " + "; ".join(errors))
    return catalog


def resolve_dimension_catalog(
    background_path: Path,
    *,
    explicit_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve the catalog for one run, with an explicit CLI path taking priority."""
    if explicit_path is not None:
        return load_catalog(Path(explicit_path))

    strategy = resolve_dimension_strategy(background_path)
    if strategy == DEFAULT_DIMENSION_STRATEGY:
        return load_catalog()

    run_dir = Path(background_path).parent
    config_path = run_dir / "framework_cfg.json"
    catalog_path = run_dir / INDUCED_CATALOG_FILENAME
    if not catalog_path.is_file():
        raise SemanticSpaceError(
            f"{config_path}: llm_induced requires {catalog_path}"
        )
    return load_catalog(catalog_path)


def resolve_dimension_strategy(background_path: Path) -> str:
    """Resolve one run's configured dimension strategy without loading its catalog."""
    run_dir = Path(background_path).parent
    config_path = run_dir / "framework_cfg.json"
    if not config_path.is_file():
        return DEFAULT_DIMENSION_STRATEGY
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticSpaceError(
            f"cannot read space initialization config {config_path}: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise SemanticSpaceError(f"{config_path}: framework config must be an object")
    section = config.get("space_initialization", {})
    if not isinstance(section, dict):
        raise SemanticSpaceError(f"{config_path}: space_initialization must be an object")
    unknown = sorted(set(section) - {"dimension_strategy"})
    if unknown:
        raise SemanticSpaceError(
            f"{config_path}: unknown space_initialization keys {unknown}"
        )
    strategy = section.get("dimension_strategy", DEFAULT_DIMENSION_STRATEGY)
    if strategy not in DIMENSION_STRATEGIES:
        raise SemanticSpaceError(
            f"{config_path}: space_initialization.dimension_strategy must be one of "
            f"{sorted(DIMENSION_STRATEGIES)}"
        )
    return strategy


def validate_catalog(catalog: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    unknown_top = sorted(
        set(catalog) - {"schema_version", "catalog_id", "provenance", "dimensions"}
    )
    if unknown_top:
        errors.append(f"catalog has unknown fields {unknown_top}")
    if catalog.get("schema_version") != 1:
        errors.append("catalog.schema_version must be 1")
    if not _nonempty(catalog.get("catalog_id")):
        errors.append("catalog.catalog_id must be a non-empty stable id")
    if not _nonempty(catalog.get("provenance")):
        errors.append("catalog.provenance must be a non-empty string")
    dimensions = catalog.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        return errors + ["catalog.dimensions must be a non-empty list"]
    seen: set[str] = set()
    for index, dimension in enumerate(dimensions):
        where = f"catalog.dimensions[{index}]"
        if not isinstance(dimension, dict):
            errors.append(f"{where} must be an object")
            continue
        dimension_id = dimension.get("id")
        if not isinstance(dimension_id, str) or not DIMENSION_RE.fullmatch(dimension_id):
            errors.append(f"{where}.id must match dim-<slug>")
        elif dimension_id in seen:
            errors.append(f"duplicate catalog dimension {dimension_id}")
        else:
            seen.add(dimension_id)
        for field in ("definition", "boundary"):
            if not _nonempty(dimension.get(field)):
                errors.append(f"{where}.{field} must be a non-empty string")
        unknown = sorted(set(dimension) - {"id", "definition", "boundary"})
        if unknown:
            errors.append(f"{where} has unknown fields {unknown}")
    return errors


def catalog_revision(catalog: dict[str, Any] | None = None) -> str:
    return digest(load_catalog() if catalog is None else catalog)


def catalog_receipt(catalog: dict[str, Any] | None = None) -> dict[str, str]:
    value = load_catalog() if catalog is None else catalog
    return {"id": value["catalog_id"], "revision": catalog_revision(value)}


def space_revision(registry: dict[str, Any]) -> str:
    """Content address the complete frozen run search-space contract."""
    return digest(registry)


def space_receipt(registry: dict[str, Any]) -> dict[str, Any]:
    return {
        "space_id": registry.get("space_id"),
        "space_revision": space_revision(registry),
        "catalog": dict(registry.get("catalog") or {}),
        "dimension_ids": [
            dimension.get("id")
            for dimension in registry.get("dimensions", [])
            if isinstance(dimension, dict)
        ],
    }


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_provenance(value: Any, where: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, list) or not value:
        return [f"{where} must be a non-empty list of provenance receipts"]
    for index, receipt in enumerate(value):
        item_where = f"{where}[{index}]"
        if not isinstance(receipt, dict):
            errors.append(f"{item_where} must be an object")
            continue
        if receipt.get("kind") not in PROVENANCE_KINDS:
            errors.append(
                f"{item_where}.kind must be one of {sorted(PROVENANCE_KINDS)}"
            )
        if not _nonempty(receipt.get("ref")):
            errors.append(f"{item_where}.ref must be a non-empty string")
        unknown = sorted(set(receipt) - {"kind", "ref"})
        if unknown:
            errors.append(f"{item_where} has unknown fields {unknown}")
    return errors


def _choice_ref_errors(
    value: Any,
    where: str,
    hypotheses_by_dimension: dict[str, set[str]],
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{where} must be an object"]
    errors: list[str] = []
    dimension_id = value.get("dimension_id")
    hypothesis_ids = value.get("hypothesis_ids")
    if dimension_id not in hypotheses_by_dimension:
        errors.append(f"{where}.dimension_id is not a selected dimension: {dimension_id!r}")
    if (
        not isinstance(hypothesis_ids, list)
        or not hypothesis_ids
        or any(not isinstance(item, str) for item in hypothesis_ids)
    ):
        errors.append(f"{where}.hypothesis_ids must be a non-empty string list")
    else:
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            errors.append(f"{where}.hypothesis_ids must not contain duplicates")
        unknown = sorted(set(hypothesis_ids) - hypotheses_by_dimension.get(dimension_id, set()))
        if unknown:
            errors.append(f"{where}.hypothesis_ids contains unknown choices {unknown}")
    unknown_fields = sorted(set(value) - {"dimension_id", "hypothesis_ids"})
    if unknown_fields:
        errors.append(f"{where} has unknown fields {unknown_fields}")
    return errors


def _relation_dimension_edges(relations: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for relation in relations:
        if relation.get("type") != "activates":
            continue
        when = relation.get("when")
        if isinstance(when, dict):
            source = when.get("dimension_id")
            target = relation.get("target_dimension_id")
            if isinstance(source, str) and isinstance(target, str):
                edges.append((source, target))
    return edges


def _cycle(edges: list[tuple[str, str]]) -> list[str] | None:
    children: dict[str, list[str]] = {}
    for source, target in edges:
        children.setdefault(source, []).append(target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, path: list[str]) -> list[str] | None:
        if node in visiting:
            start = path.index(node)
            return path[start:] + [node]
        if node in visited:
            return None
        visiting.add(node)
        path.append(node)
        for child in children.get(node, []):
            found = visit(child, path)
            if found:
                return found
        path.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in list(children):
        found = visit(node, [])
        if found:
            return found
    return None


def validate_space_core(
    registry: dict[str, Any],
    catalog: dict[str, Any] | None = None,
    *,
    dimension_strategy: str = DEFAULT_DIMENSION_STRATEGY,
) -> list[str]:
    """Validate hierarchy and relations, excluding source/guidance semantics."""
    errors: list[str] = []
    if dimension_strategy not in DIMENSION_STRATEGIES:
        errors.append(
            f"dimension_strategy must be one of {sorted(DIMENSION_STRATEGIES)}"
        )
    catalog = load_catalog() if catalog is None else catalog
    catalog_errors = validate_catalog(catalog)
    if catalog_errors:
        return [f"catalog: {error}" for error in catalog_errors]
    expected_catalog = catalog_receipt(catalog)
    if registry.get("schema_version") != SEARCH_SPACE_SCHEMA_VERSION:
        if "directions" in registry or registry.get("schema_version") in {1, 2}:
            errors.append(
                "legacy flat background registry is not supported; create a schema_version 3 "
                "semantic search space with dimensions and hypotheses"
            )
        else:
            errors.append(f"schema_version must be {SEARCH_SPACE_SCHEMA_VERSION}")
    if registry.get("kind") != "semantic_search_space":
        errors.append("kind must be 'semantic_search_space'")
    if not _nonempty(registry.get("space_id")):
        errors.append("space_id must be a non-empty stable run-local id")
    if registry.get("catalog") != expected_catalog:
        errors.append(
            "catalog must exactly match the supplied dimension catalog receipt "
            f"{expected_catalog}"
        )

    dimensions = registry.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        return errors + ["dimensions must be a non-empty list"]
    if dimension_strategy == "llm_induced":
        selected_ids = [
            dimension.get("id") if isinstance(dimension, dict) else None
            for dimension in dimensions
        ]
        catalog_ids = [dimension["id"] for dimension in catalog["dimensions"]]
        if selected_ids != catalog_ids:
            errors.append(
                "llm_induced registry dimensions must exactly match the run-local "
                "catalog ids and order"
            )
    catalog_by_id = {item["id"]: item for item in catalog["dimensions"]}
    dimension_ids: set[str] = set()
    hypothesis_ids: set[str] = set()
    hypotheses_by_dimension: dict[str, set[str]] = {}
    baseline_by_dimension: dict[str, str] = {}
    dimension_mode: dict[str, str] = {}

    allowed_dimension_fields = {
        "id",
        "definition",
        "boundary",
        "catalog_provenance",
        "selection_reason",
        "evidence",
        "mode",
        "status",
        "baseline_hypothesis_id",
        "hypotheses",
    }
    allowed_hypothesis_fields = {
        "id",
        "title",
        "claim",
        "kind",
        "status",
        "provenance",
        "claim_scope",
        "scope",
        "required_comparisons",
        "reopen_when",
        "literature_credibility",
        "credibility_rationale",
        "testable_expectation",
        "evidence",
        "probe_for",
    }
    for index, dimension in enumerate(dimensions):
        where = f"dimensions[{index}]"
        if not isinstance(dimension, dict):
            errors.append(f"{where} must be an object")
            continue
        dimension_id = dimension.get("id")
        catalog_entry = catalog_by_id.get(dimension_id)
        if catalog_entry is None:
            errors.append(
                f"{where}.id is not in catalog {catalog.get('catalog_id')!r}: {dimension_id!r}"
            )
        elif dimension_id in dimension_ids:
            errors.append(f"duplicate selected dimension {dimension_id}")
        else:
            dimension_ids.add(dimension_id)
        if catalog_entry is not None:
            for field in ("definition", "boundary"):
                if dimension.get(field) != catalog_entry[field]:
                    errors.append(f"{where}.{field} must preserve the catalog value exactly")
            if dimension.get("catalog_provenance") != catalog["provenance"]:
                errors.append(f"{where}.catalog_provenance must preserve catalog provenance")
        if not _nonempty(dimension.get("selection_reason")):
            errors.append(f"{where}.selection_reason must be a non-empty string")
        dimension_evidence = dimension.get("evidence")
        errors.extend(_validate_provenance(dimension_evidence, f"{where}.evidence"))
        if isinstance(dimension_evidence, list) and not any(
            isinstance(receipt, dict) and receipt.get("kind") != "catalog"
            for receipt in dimension_evidence
        ):
            errors.append(
                f"{where}.evidence must justify run selection beyond catalog membership"
            )
        mode = dimension.get("mode")
        if mode not in DIMENSION_MODES:
            errors.append(f"{where}.mode must be one of {sorted(DIMENSION_MODES)}")
        elif isinstance(dimension_id, str):
            dimension_mode[dimension_id] = mode
        if dimension.get("status") not in ELEMENT_STATUSES:
            errors.append(
                f"{where}.status must be active; the frozen registry never "
                "authors runtime pruning state"
            )
        hypotheses = dimension.get("hypotheses")
        if not isinstance(hypotheses, list) or not hypotheses:
            errors.append(f"{where}.hypotheses must be a non-empty list")
            hypotheses = []
        local_ids: set[str] = set()
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            hyp_where = f"{where}.hypotheses[{hypothesis_index}]"
            if not isinstance(hypothesis, dict):
                errors.append(f"{hyp_where} must be an object")
                continue
            hypothesis_id = hypothesis.get("id")
            if not isinstance(hypothesis_id, str) or not HYPOTHESIS_RE.fullmatch(hypothesis_id):
                errors.append(f"{hyp_where}.id must match hyp-<slug>")
            elif hypothesis_id in hypothesis_ids:
                errors.append(f"duplicate hypothesis id {hypothesis_id}")
            else:
                hypothesis_ids.add(hypothesis_id)
                local_ids.add(hypothesis_id)
            for field in (
                "title",
                "claim",
                "claim_scope",
                "reopen_when",
                "credibility_rationale",
                "testable_expectation",
            ):
                if not _nonempty(hypothesis.get(field)):
                    errors.append(f"{hyp_where}.{field} must be a non-empty string")
            if hypothesis.get("kind") not in HYPOTHESIS_KINDS:
                errors.append(
                    f"{hyp_where}.kind must be one of {sorted(HYPOTHESIS_KINDS)}"
                )
            if hypothesis.get("status") not in ELEMENT_STATUSES:
                errors.append(
                    f"{hyp_where}.status must be active; runtime pruning state "
                    "lives only in ledger.search_space_state"
                )
            hypothesis_provenance = hypothesis.get("provenance")
            errors.extend(
                _validate_provenance(hypothesis_provenance, f"{hyp_where}.provenance")
            )
            if hypothesis.get("kind") == "baseline" and isinstance(
                hypothesis_provenance, list
            ) and not any(
                isinstance(receipt, dict) and receipt.get("kind") == "task_contract"
                for receipt in hypothesis_provenance
            ):
                errors.append(
                    f"{hyp_where}.provenance must trace the task-specific baseline"
                )
            comparisons = hypothesis.get("required_comparisons")
            if (
                not isinstance(comparisons, list)
                or not comparisons
                or any(not _nonempty(item) for item in comparisons)
            ):
                errors.append(f"{hyp_where}.required_comparisons must be a non-empty string list")
            evidence = hypothesis.get("evidence")
            if not isinstance(evidence, list):
                errors.append(f"{hyp_where}.evidence must be a list")
            if hypothesis.get("kind") == "scope_probe":
                probe_for = hypothesis.get("probe_for")
                if not isinstance(probe_for, list) or not probe_for or any(
                    not isinstance(item, str) for item in probe_for
                ):
                    errors.append(f"{hyp_where}.probe_for must be a non-empty guidance-id list")
            elif hypothesis.get("probe_for"):
                errors.append(f"{hyp_where}.probe_for is only valid for a scope_probe")
            unknown_hypothesis_fields = sorted(set(hypothesis) - allowed_hypothesis_fields)
            if unknown_hypothesis_fields:
                errors.append(f"{hyp_where} has unknown fields {unknown_hypothesis_fields}")
        if isinstance(dimension_id, str):
            hypotheses_by_dimension[dimension_id] = local_ids
        baseline_id = dimension.get("baseline_hypothesis_id")
        baseline_by_dimension[str(dimension_id)] = str(baseline_id)
        if baseline_id not in local_ids:
            errors.append(f"{where}.baseline_hypothesis_id must name a local hypothesis")
        else:
            baseline = next(
                (item for item in hypotheses if isinstance(item, dict) and item.get("id") == baseline_id),
                {},
            )
            if baseline.get("kind") != "baseline":
                errors.append(f"{where} baseline hypothesis must have kind='baseline'")
        if mode == "baseline_only" and len(hypotheses) != 1:
            errors.append(f"{where} baseline_only dimension must contain exactly its baseline")
        unknown_dimension_fields = sorted(set(dimension) - allowed_dimension_fields)
        if unknown_dimension_fields:
            errors.append(f"{where} has unknown fields {unknown_dimension_fields}")

    relations = registry.get("relations")
    if not isinstance(relations, list):
        errors.append("relations must be a list")
        relations = []
    relation_ids: set[str] = set()
    activation_targets: set[str] = set()
    for index, relation in enumerate(relations):
        where = f"relations[{index}]"
        if not isinstance(relation, dict):
            errors.append(f"{where} must be an object")
            continue
        relation_id = relation.get("id")
        if not isinstance(relation_id, str) or not RELATION_RE.fullmatch(relation_id):
            errors.append(f"{where}.id must match rel-<slug>")
        elif relation_id in relation_ids:
            errors.append(f"duplicate relation id {relation_id}")
        else:
            relation_ids.add(relation_id)
        relation_type = relation.get("type")
        if relation_type not in RELATION_TYPES:
            errors.append(f"{where}.type must be one of {sorted(RELATION_TYPES)}")
        if relation.get("status") not in ELEMENT_STATUSES:
            errors.append(f"{where}.status must be active")
        errors.extend(_validate_provenance(relation.get("provenance"), f"{where}.provenance"))
        if not isinstance(relation.get("evidence"), list):
            errors.append(f"{where}.evidence must be a list")
        if relation_type == "activates":
            errors.extend(
                _choice_ref_errors(relation.get("when"), f"{where}.when", hypotheses_by_dimension)
            )
            target = relation.get("target_dimension_id")
            if target not in dimension_ids:
                errors.append(f"{where}.target_dimension_id is not selected: {target!r}")
            else:
                activation_targets.add(target)
            when = relation.get("when") if isinstance(relation.get("when"), dict) else {}
            if target == when.get("dimension_id"):
                errors.append(f"{where} cannot activate its own dimension")
            allowed = {
                "id", "type", "status", "provenance", "evidence", "when", "target_dimension_id"
            }
        elif relation_type == "requires":
            errors.extend(
                _choice_ref_errors(relation.get("when"), f"{where}.when", hypotheses_by_dimension)
            )
            errors.extend(
                _choice_ref_errors(relation.get("then"), f"{where}.then", hypotheses_by_dimension)
            )
            allowed = {"id", "type", "status", "provenance", "evidence", "when", "then"}
        else:
            members = relation.get("members")
            if not isinstance(members, list) or len(members) < 2:
                errors.append(f"{where}.members must contain at least two choice scopes")
                members = []
            for member_index, member in enumerate(members):
                errors.extend(
                    _choice_ref_errors(
                        member,
                        f"{where}.members[{member_index}]",
                        hypotheses_by_dimension,
                    )
                )
            allowed = {"id", "type", "status", "provenance", "evidence", "members"}
        unknown_relation_fields = sorted(set(relation) - allowed)
        if unknown_relation_fields:
            errors.append(f"{where} has unknown fields {unknown_relation_fields}")

    cycle = _cycle(_relation_dimension_edges([r for r in relations if isinstance(r, dict)]))
    if cycle:
        errors.append(f"activation relations must be acyclic; cycle: {' -> '.join(cycle)}")

    # A baseline-only dimension can be conditional, but the condition may not
    # depend on itself (already checked). Searchable and baseline-only both use
    # explicit inactive point entries when their activation is false.
    unknown_top = sorted(
        set(registry)
        - {"schema_version", "kind", "space_id", "catalog", "dimensions", "relations", "guidance", "sources"}
    )
    if unknown_top:
        errors.append(f"search-space registry has unknown fields {unknown_top}")
    return errors


def dimension_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        dimension["id"]: dimension
        for dimension in registry.get("dimensions", [])
        if isinstance(dimension, dict) and isinstance(dimension.get("id"), str)
    }


def hypothesis_map(registry: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    result: dict[str, tuple[str, dict[str, Any]]] = {}
    for dimension in registry.get("dimensions", []):
        if not isinstance(dimension, dict):
            continue
        dimension_id = dimension.get("id")
        for hypothesis in dimension.get("hypotheses", []):
            if isinstance(hypothesis, dict) and isinstance(hypothesis.get("id"), str):
                result[hypothesis["id"]] = (dimension_id, hypothesis)
    return result


def relation_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        relation["id"]: relation
        for relation in registry.get("relations", [])
        if isinstance(relation, dict) and isinstance(relation.get("id"), str)
    }


def _choice_matches(choice: dict[str, Any], selected: dict[str, str]) -> bool:
    return selected.get(choice.get("dimension_id")) in set(choice.get("hypothesis_ids") or [])


def incoming_activation_relations(
    registry: dict[str, Any], dimension_id: str
) -> list[dict[str, Any]]:
    return [
        relation
        for relation in registry.get("relations", [])
        if isinstance(relation, dict)
        and relation.get("type") == "activates"
        and relation.get("target_dimension_id") == dimension_id
    ]


def _point_payload(point: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": point.get("schema_version"),
        "space_id": point.get("space_id"),
        "space_revision": point.get("space_revision"),
        "assignments": point.get("assignments"),
    }


def point_id(point: dict[str, Any]) -> str:
    return "point-" + digest(_point_payload(point)).split(":", 1)[1][:20]


def selected_assignments(point: dict[str, Any]) -> dict[str, str]:
    return {
        assignment.get("dimension_id"): assignment.get("hypothesis_id")
        for assignment in point.get("assignments", [])
        if isinstance(assignment, dict) and assignment.get("state") == "selected"
    }


def validate_point(point: Any, registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(point, dict):
        return ["semantic_point must be an object"]
    if point.get("schema_version") != POINT_SCHEMA_VERSION:
        errors.append(f"semantic_point.schema_version must be {POINT_SCHEMA_VERSION}")
    if point.get("space_id") != registry.get("space_id"):
        errors.append("semantic_point.space_id does not match background search space")
    expected_revision = space_revision(registry)
    if point.get("space_revision") != expected_revision:
        errors.append(
            "semantic_point.space_revision does not match the exact current background revision"
        )
    assignments = point.get("assignments")
    dimensions = list(dimension_map(registry).values())
    if not isinstance(assignments, list):
        return errors + ["semantic_point.assignments must be a list"]
    expected_order = [dimension["id"] for dimension in dimensions]
    actual_order = [
        assignment.get("dimension_id") if isinstance(assignment, dict) else None
        for assignment in assignments
    ]
    if actual_order != expected_order:
        errors.append(
            "semantic_point.assignments must contain every selected dimension exactly once "
            "in registry order"
        )
    selected = selected_assignments(point)
    relations = relation_map(registry)
    for index, (dimension, assignment) in enumerate(zip(dimensions, assignments)):
        where = f"semantic_point.assignments[{index}]"
        if not isinstance(assignment, dict):
            errors.append(f"{where} must be an object")
            continue
        dimension_id = dimension["id"]
        incoming = incoming_activation_relations(registry, dimension_id)
        active = not incoming or any(_choice_matches(item["when"], selected) for item in incoming)
        state = assignment.get("state")
        if active and state != "selected":
            errors.append(f"{where} dimension is active and must select a hypothesis")
        if not active and state != "inactive":
            errors.append(f"{where} dimension is inactive and must say so explicitly")
        if state == "selected":
            hypothesis_id = assignment.get("hypothesis_id")
            local_hypotheses = {
                hypothesis.get("id"): hypothesis
                for hypothesis in dimension.get("hypotheses", [])
                if isinstance(hypothesis, dict)
            }
            if hypothesis_id not in local_hypotheses:
                errors.append(f"{where}.hypothesis_id is not a local hypothesis: {hypothesis_id!r}")
            elif local_hypotheses[hypothesis_id].get("status") != "active":
                errors.append(f"{where}.hypothesis_id is not active")
            if dimension.get("mode") == "baseline_only" and hypothesis_id != dimension.get(
                "baseline_hypothesis_id"
            ):
                errors.append(f"{where} baseline_only dimension must select its baseline")
            unknown = sorted(set(assignment) - {"dimension_id", "state", "hypothesis_id"})
            if unknown:
                errors.append(f"{where} selected assignment has unknown fields {unknown}")
        elif state == "inactive":
            activation_ids = assignment.get("activation_relation_ids")
            expected_ids = [item["id"] for item in incoming]
            if activation_ids != expected_ids:
                errors.append(
                    f"{where}.activation_relation_ids must list the unsatisfied incoming "
                    f"relations in registry order: {expected_ids}"
                )
            unknown = sorted(set(assignment) - {"dimension_id", "state", "activation_relation_ids"})
            if unknown:
                errors.append(f"{where} inactive assignment has unknown fields {unknown}")
        else:
            errors.append(f"{where}.state must be 'selected' or 'inactive'")

    for relation in relations.values():
        if relation.get("type") == "requires" and _choice_matches(relation["when"], selected):
            if not _choice_matches(relation["then"], selected):
                errors.append(
                    f"semantic_point violates requires relation {relation['id']}: "
                    f"{relation['then']}"
                )
        elif relation.get("type") == "excludes":
            if all(_choice_matches(member, selected) for member in relation.get("members", [])):
                errors.append(f"semantic_point violates exclusion relation {relation['id']}")
    expected_point_id = point_id(point)
    if point.get("point_id") != expected_point_id:
        errors.append(f"semantic_point.point_id must be {expected_point_id}")
    unknown_point_fields = sorted(
        set(point) - {"schema_version", "space_id", "space_revision", "point_id", "assignments"}
    )
    if unknown_point_fields:
        errors.append(f"semantic_point has unknown fields {unknown_point_fields}")
    return errors


def complete_point(
    registry: dict[str, Any], overrides: dict[str, str] | None = None
) -> dict[str, Any] | None:
    """Complete sparse desired choices with explicit baselines and inactivity.

    Returns ``None`` when the desired choices cannot satisfy the frozen
    relations.  This is a deterministic completion helper, not a search policy.
    """
    overrides = dict(overrides or {})
    dimensions = dimension_map(registry)
    if set(overrides) - set(dimensions):
        return None
    for dimension_id, hypothesis_id in overrides.items():
        local_ids = {
            hypothesis.get("id")
            for hypothesis in dimensions[dimension_id].get("hypotheses", [])
            if isinstance(hypothesis, dict)
        }
        if hypothesis_id not in local_ids:
            return None
    selected: dict[str, str] = {}
    conditional = [
        dimension_id
        for dimension_id in dimensions
        if incoming_activation_relations(registry, dimension_id)
    ]
    for dimension_id, dimension in dimensions.items():
        if dimension_id not in conditional:
            selected[dimension_id] = overrides.get(
                dimension_id, dimension["baseline_hypothesis_id"]
            )

    relations = [item for item in registry.get("relations", []) if isinstance(item, dict)]
    for _ in range(max(2, len(dimensions) * 2)):
        before = dict(selected)
        for dimension_id in conditional:
            incoming = incoming_activation_relations(registry, dimension_id)
            if any(_choice_matches(item["when"], selected) for item in incoming):
                selected[dimension_id] = overrides.get(
                    dimension_id, dimensions[dimension_id]["baseline_hypothesis_id"]
                )
            else:
                selected.pop(dimension_id, None)
        for relation in relations:
            if relation.get("type") != "requires" or not _choice_matches(
                relation.get("when", {}), selected
            ):
                continue
            target = relation.get("then", {})
            dimension_id = target.get("dimension_id")
            allowed = target.get("hypothesis_ids") or []
            desired = overrides.get(dimension_id)
            if desired in allowed:
                selected[dimension_id] = desired
            elif selected.get(dimension_id) not in allowed and allowed:
                selected[dimension_id] = allowed[0]
        if selected == before:
            break

    assignments: list[dict[str, Any]] = []
    for dimension_id, dimension in dimensions.items():
        incoming = incoming_activation_relations(registry, dimension_id)
        if dimension_id in selected:
            assignments.append(
                {
                    "dimension_id": dimension_id,
                    "state": "selected",
                    "hypothesis_id": selected[dimension_id],
                }
            )
        else:
            assignments.append(
                {
                    "dimension_id": dimension_id,
                    "state": "inactive",
                    "activation_relation_ids": [item["id"] for item in incoming],
                }
            )
    point = {
        "schema_version": POINT_SCHEMA_VERSION,
        "space_id": registry.get("space_id"),
        "space_revision": space_revision(registry),
        "assignments": assignments,
    }
    point["point_id"] = point_id(point)
    if validate_point(point, registry):
        return None
    return point


def point_diff(parent: dict[str, Any], child: dict[str, Any]) -> list[dict[str, Any]]:
    parent_by_dim = {
        item["dimension_id"]: item
        for item in parent.get("assignments", [])
        if isinstance(item, dict) and isinstance(item.get("dimension_id"), str)
    }
    changes: list[dict[str, Any]] = []
    for current in child.get("assignments", []):
        if not isinstance(current, dict) or not isinstance(current.get("dimension_id"), str):
            continue
        previous = parent_by_dim.get(current["dimension_id"], {})
        before = previous.get("hypothesis_id") if previous.get("state") == "selected" else None
        after = current.get("hypothesis_id") if current.get("state") == "selected" else None
        if before == after:
            continue
        operation = (
            "dimension_activated" if before is None
            else "dimension_deactivated" if after is None
            else "hypothesis_changed"
        )
        changes.append({
            "dimension_id": current["dimension_id"],
            "operation": operation,
            "from_hypothesis_id": before,
            "to_hypothesis_id": after,
        })
    return changes


def coverage_from_records(registry: dict[str, Any], records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    hypothesis_counts = {hypothesis_id: 0 for hypothesis_id in hypothesis_map(registry)}
    point_counts: dict[str, int] = {}
    invalid_records: list[dict[str, Any]] = []
    for record in records:
        run_id = str(record.get("run_id"))
        point = record.get("semantic_point")
        point_errors = validate_point(point, registry)
        if point_errors:
            invalid_records.append({"run_id": run_id, "errors": point_errors})
            continue
        point_counts[point["point_id"]] = point_counts.get(point["point_id"], 0) + 1
        for hypothesis_id in selected_assignments(point).values():
            hypothesis_counts[hypothesis_id] = hypothesis_counts.get(hypothesis_id, 0) + 1
    dimensions = []
    for dimension in registry.get("dimensions", []):
        if not isinstance(dimension, dict):
            continue
        dimensions.append(
            {
                "dimension_id": dimension.get("id"),
                "mode": dimension.get("mode"),
                "hypotheses": [
                    {
                        "hypothesis_id": hypothesis.get("id"),
                        "count": hypothesis_counts.get(hypothesis.get("id"), 0),
                    }
                    for hypothesis in dimension.get("hypotheses", [])
                    if isinstance(hypothesis, dict)
                ],
            }
        )
    return {
        "space_revision": space_revision(registry),
        "n_valid_records": sum(point_counts.values()),
        "n_unique_points": len(point_counts),
        "point_counts": point_counts,
        "dimensions": dimensions,
        "invalid_records": invalid_records,
    }


def derive_semantic_lineage(
    registry: dict[str, Any], ledger: dict[str, Any], *, limit: int | None = None
) -> dict[str, Any]:
    records = {
        str(record.get("run_id")): record
        for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("run_id") is not None
    }
    warnings: list[str] = []
    runs: list[dict[str, Any]] = []
    hypothesis_runs: dict[str, list[str]] = {
        hypothesis_id: [] for hypothesis_id in hypothesis_map(registry)
    }
    for run_id, record in records.items():
        point = record.get("semantic_point")
        point_errors = validate_point(point, registry)
        if point_errors:
            warnings.append(f"run {run_id} has invalid semantic_point: {'; '.join(point_errors)}")
            continue
        parents = [str(item) for item in (record.get("source_run_ids") or [])]
        missing = [parent for parent in parents if parent not in records]
        if missing:
            warnings.append(f"run {run_id} references missing parents {missing}")
        semantic_edges = record.get("semantic_edges")
        if not isinstance(semantic_edges, list):
            semantic_edges = []
        selected = selected_assignments(point)
        for hypothesis_id in selected.values():
            hypothesis_runs.setdefault(hypothesis_id, []).append(run_id)
        runs.append(
            {
                "run_id": run_id,
                "op": record.get("op"),
                "parent_run_ids": parents,
                "point_id": point.get("point_id"),
                "assignments": selected,
                "semantic_edges": semantic_edges,
                "status": record.get("status"),
                "score": record.get("final_best_score"),
            }
        )
    if limit is not None and limit >= 0:
        runs = [] if limit == 0 else runs[-limit:]
        warnings = [] if limit == 0 else warnings[-limit:]
        hypothesis_runs = {
            hypothesis_id: ([] if limit == 0 else run_ids[-limit:])
            for hypothesis_id, run_ids in hypothesis_runs.items()
        }
    coverage = coverage_from_records(registry, records.values())
    if limit is not None and limit >= 0:
        recent_point_ids = list(
            dict.fromkeys(
                run["point_id"] for run in runs if isinstance(run.get("point_id"), str)
            )
        )
        all_point_counts = coverage["point_counts"]
        coverage["point_counts"] = {
            point_id_value: all_point_counts[point_id_value]
            for point_id_value in recent_point_ids
            if point_id_value in all_point_counts
        }
        coverage["omitted_point_counts"] = max(
            0, len(all_point_counts) - len(coverage["point_counts"])
        )
        coverage["invalid_records"] = (
            [] if limit == 0 else coverage["invalid_records"][-limit:]
        )
    return {
        "space": space_receipt(registry),
        "runs": runs,
        "hypothesis_runs": hypothesis_runs,
        "coverage": coverage,
        "warnings": warnings,
        "attribution_notice": (
            "Point membership is attribution only. Parent diffs are mechanical and do not "
            "claim that a semantic change caused the observed score."
        ),
    }
