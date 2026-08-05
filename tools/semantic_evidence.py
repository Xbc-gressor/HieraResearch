#!/usr/bin/env python3
"""Mechanical semantic DAG edge receipts and bounded per-target evidence views.

Every non-fresh candidate record persists one receipt per numeric parent in
``semantic_edges``.  Receipts are derived here from the exact persisted parent
and child ``semantic_point`` objects; models never author them.  This module
owns receipt building and validation, the persisted edge index, observation
lookup, comparator coverage, the mechanical target evaluation states, and the
bounded per-target evidence renderer consumed by the experience extractor.

Scores stay lower-is-better.  A crash is distinct from an unevaluated target
and cannot by itself contradict or prune a semantic element.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from semantic_space import (
    dimension_map,
    hypothesis_map,
    point_diff,
    selected_assignments,
    space_revision,
    validate_point,
)


EDGE_SCHEMA_VERSION = 1
CHANGE_CLASSES = {"same_point", "single_dimension", "multi_dimension"}
OPERATIONS = {"hypothesis_changed", "dimension_activated", "dimension_deactivated"}
EDGE_ID_RE = re.compile(r"^sedge-([0-9]+)-([0-9]+)$")

EVALUATION_STATES = {"unevaluated", "failed", "observed", "comparator_covered"}
TERMINAL_STATUSES = {"keep", "discard", "crash"}
NONCRASH_TERMINAL_STATUSES = {"keep", "discard"}
# Every status a record can hold once its lifecycle is over. `unevaluated` is
# terminal but is NOT evidence: the candidate was admitted after the objective
# budget was exhausted and never ran, so it belongs here and not in
# TERMINAL_STATUSES, which gates what may be cited as an observation.
LIFECYCLE_TERMINAL_STATUSES = {"keep", "discard", "crash", "unevaluated"}
# Terminal states that yield no usable parameter observation, and therefore no
# parent-incumbent binding to preserve.
NO_OBSERVATION_TERMINAL_STATUSES = {"crash", "unevaluated"}

MAX_DIMENSION_TARGETS = 16
MAX_HYPOTHESIS_TARGETS = 32
MIN_EDGES_PER_TARGET = 2
MAX_EDGES_PER_TARGET = 5
MAX_RUNS_PER_TARGET = 5
MAX_COMPARATOR_GAIN_ADJUSTMENT = 0.15
MAX_EVIDENCE_UNCERTAINTY_ADJUSTMENT = 0.10

# Contradiction-grade depth bar: >=MIN_EDGES_PER_TARGET direct tuned edges, or
# >=DEPTH_BAR_LIGHT_MIN direct edges at tuned_lightly or deeper.
DEPTH_BAR_LIGHT_MIN = 3


def _contradiction_depth_bar(coverage: dict) -> bool:
    """Whether a coverage receipt clears the contradiction-grade depth bar."""
    tuned = coverage.get("direct_tuned_edges", 0)
    light = coverage.get("direct_lightly_tuned_edges", 0)
    return tuned >= MIN_EDGES_PER_TARGET or (tuned + light) >= DEPTH_BAR_LIGHT_MIN


COVERAGE_KEYS = (
    "direct_tuned_edges",
    "direct_lightly_tuned_edges",
    "direct_noncrash_edges",
    "confounded_noncrash_edges",
    "crash_edges",
)
# Coverage shape before `direct_tuned_edges` split screening-depth direct
# comparators out of `direct_noncrash_edges`. Legacy artifacts and decision
# receipts carry exactly these three keys.
LEGACY_COVERAGE_KEYS = (
    "direct_noncrash_edges",
    "confounded_noncrash_edges",
    "crash_edges",
)
# Coverage shape before `direct_lightly_tuned_edges` split lightly-tuned
# direct comparators into their own bucket. Schema-2 artifacts and decision
# receipts carry exactly these four keys.
SCHEMA_2_COVERAGE_KEYS = (
    "direct_tuned_edges",
    "direct_noncrash_edges",
    "confounded_noncrash_edges",
    "crash_edges",
)


def normalize_coverage(raw: Any) -> dict[str, int] | None:
    """Read any known coverage shape, or return None if it is none of them.

    Legacy maps are read forward with the newer direct-depth buckets at 0: the
    three-key shape predates the tuned split and the four-key shape predates
    the lightly-tuned split, so their direct edges are not known to be tuned
    (or lightly tuned) and the conservative reading is that none were.
    Callers must treat the result as backward-readable evidence, never as a
    recomputed claim.
    """
    if not isinstance(raw, dict):
        return None

    def _counts(keys: tuple[str, ...]) -> dict[str, int] | None:
        out: dict[str, int] = {}
        for key in keys:
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return None
            out[key] = value
        return out

    if set(raw) == set(COVERAGE_KEYS):
        return _counts(COVERAGE_KEYS)
    if set(raw) == set(SCHEMA_2_COVERAGE_KEYS):
        counts = _counts(SCHEMA_2_COVERAGE_KEYS)
        if counts is None:
            return None
        return {
            "direct_tuned_edges": counts["direct_tuned_edges"],
            "direct_lightly_tuned_edges": 0,
            "direct_noncrash_edges": counts["direct_noncrash_edges"],
            "confounded_noncrash_edges": counts["confounded_noncrash_edges"],
            "crash_edges": counts["crash_edges"],
        }
    if set(raw) == set(LEGACY_COVERAGE_KEYS):
        counts = _counts(LEGACY_COVERAGE_KEYS)
        if counts is None:
            return None
        return {"direct_tuned_edges": 0, "direct_lightly_tuned_edges": 0, **counts}
    return None
DIRECT_COMPARATOR_CAPABILITY_KEY = "direct_comparator_capability"
DIRECT_COMPARATOR_CAPABILITY = {
    "schema_version": 1,
    "status": "unavailable",
    "reason": "no_production_same_child_code_control_treatment_evaluator",
}
DIRECT_COMPARATOR_FIXTURE_CAPABILITY = {
    "schema_version": 1,
    "status": "fixture_only",
    "reason": "synthetic_downstream_contract_validation",
}
PROPOSAL_RELATIONS = {
    "selected_fresh",
    "introduced",
    "removed",
    "changed_dimension",
    "ambiguous",
}


class SemanticEvidenceError(ValueError):
    """A malformed semantic edge receipt or evidence-view request."""


def direct_comparator_capability(ledger: dict[str, Any]) -> dict[str, Any]:
    """Return the explicit production capability gate for direct comparators.

    Ledgers written by ``ledger.py`` always carry the schema-1 unavailable
    receipt. A missing field remains backward-readable for historical
    artifacts and isolated downstream unit fixtures, but it is never emitted
    as a claim that production can create paired evidence.
    """
    raw = ledger.get(DIRECT_COMPARATOR_CAPABILITY_KEY)
    if raw is None:
        return {
            "schema_version": 0,
            "status": "legacy_unspecified",
            "reason": "ledger_predates_explicit_direct_comparator_gate",
        }
    if raw in (
        DIRECT_COMPARATOR_CAPABILITY,
        DIRECT_COMPARATOR_FIXTURE_CAPABILITY,
    ):
        return dict(raw)
    return {
        "schema_version": 1,
        "status": "unavailable",
        "reason": "invalid_direct_comparator_capability_receipt",
    }


def _direct_comparators_enabled(ledger: dict[str, Any]) -> bool:
    """Fail closed whenever a ledger carries the production capability gate."""
    raw = ledger.get(DIRECT_COMPARATOR_CAPABILITY_KEY)
    return (
        raw is None
        or raw == DIRECT_COMPARATOR_FIXTURE_CAPABILITY
    )


def experience_cited_ids(experience: Any) -> tuple[set[str], set[str]]:
    """Return the run and semantic-edge receipts carried by bounded experience.

    This is the shared trust boundary for gain-context rendering, prediction
    validation, and candidate admission.  Keep collection mechanics here so
    producer and consumers cannot silently diverge as experience schema 3
    evolves.
    """
    if not isinstance(experience, dict):
        return set(), set()
    run_ids: set[str] = set()
    edge_ids: set[str] = set()
    for field in ("promising_regions", "lessons", "bottlenecks"):
        for item in experience.get(field, []):
            if isinstance(item, dict) and isinstance(item.get("evidence"), list):
                run_ids.update(
                    run_id
                    for run_id in item["evidence"]
                    if isinstance(run_id, str)
                )
    for field in ("dimension_evidence", "hypothesis_evidence"):
        for item in experience.get(field, []):
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("evidence_run_ids"), list):
                run_ids.update(
                    run_id
                    for run_id in item["evidence_run_ids"]
                    if isinstance(run_id, str)
                )
            if isinstance(item.get("evidence_edge_ids"), list):
                edge_ids.update(
                    edge_id
                    for edge_id in item["evidence_edge_ids"]
                    if isinstance(edge_id, str)
                )
    return run_ids, edge_ids


def acquisition_target_relations(
    point: Any, parent_points: list[dict[str, Any]] | None = None
) -> dict[str, str]:
    """Return proposal-relevant targets and their mechanical move relation.

    ``None`` means a fresh point.  A concrete parent list means a relative
    proposal, including an empty-delta same-point proposal.  Crossover
    relations that disagree are retained as ``ambiguous`` and therefore can
    inform uncertainty but never signed gain.
    """
    if not isinstance(point, dict):
        return {}
    if parent_points is not None:
        relation_sets: dict[str, set[str]] = {}
        for parent in parent_points:
            if not isinstance(parent, dict):
                continue
            try:
                changes = point_diff(parent, point)
            except Exception:
                continue
            for change in changes:
                if not isinstance(change, dict):
                    continue
                dimension_id = change.get("dimension_id")
                from_hypothesis_id = change.get("from_hypothesis_id")
                to_hypothesis_id = change.get("to_hypothesis_id")
                if isinstance(dimension_id, str):
                    relation_sets.setdefault(dimension_id, set()).add(
                        "changed_dimension"
                    )
                if isinstance(from_hypothesis_id, str):
                    relation_sets.setdefault(from_hypothesis_id, set()).add(
                        "removed"
                    )
                if isinstance(to_hypothesis_id, str):
                    relation_sets.setdefault(to_hypothesis_id, set()).add(
                        "introduced"
                    )
        return {
            target_id: (
                next(iter(relations))
                if len(relations) == 1
                else "ambiguous"
            )
            for target_id, relations in relation_sets.items()
        }
    try:
        # Fresh proposals have no relative move.  Exact selected hypotheses are
        # relevant; broad dimensions are intentionally excluded.
        return {
            target_id: "selected_fresh"
            for target_id in selected_assignments(point).values()
        }
    except Exception:
        return {}


def acquisition_target_ids(
    point: Any, parent_points: list[dict[str, Any]] | None = None
) -> set[str]:
    """Backward-compatible ID view over :func:`acquisition_target_relations`."""
    return set(acquisition_target_relations(point, parent_points))


def acquisition_conditioning(
    experience: Any,
    point: Any,
    *,
    target_ids: set[str] | None = None,
    target_relations: dict[str, str] | None = None,
    gain_directions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Render the only experience entries allowed to affect acquisition.

    Free-text summary/generic collections are deliberately excluded.  A
    proposal may see only structured entries mechanically related to its move.
    Signed gain is available only from repeated same-child-code semantic
    control/treatment pairs with a consistent direction; all weaker evidence is
    uncertainty-only.  LLM-authored assessments and prose never determine the
    direction.
    """
    if not isinstance(experience, dict) or not isinstance(point, dict):
        return []
    try:
        selected = selected_assignments(point)
    except Exception:
        return []
    if target_relations is None:
        allowed_ids = (
            set(target_ids)
            if target_ids is not None
            else set(selected) | set(selected.values())
        )
        relations = {
            target_id: (
                "selected_fresh"
                if target_id in set(selected.values())
                else "changed_dimension"
            )
            for target_id in allowed_ids
        }
    else:
        relations = {
            str(target_id): relation
            for target_id, relation in target_relations.items()
            if isinstance(target_id, str) and relation in PROPOSAL_RELATIONS
        }
        if target_ids is not None:
            relations = {
                target_id: relation
                for target_id, relation in relations.items()
                if target_id in target_ids
            }
    directions = gain_directions or {}
    rendered: list[dict[str, Any]] = []
    for field, target_kind in (
        ("dimension_evidence", "dimension"),
        ("hypothesis_evidence", "hypothesis"),
    ):
        for item in experience.get(field, []) or []:
            if not isinstance(item, dict):
                continue
            target_id = item.get("target_id")
            relation = relations.get(str(target_id))
            if relation is None:
                continue
            run_ids = [
                value
                for value in item.get("evidence_run_ids", [])
                if isinstance(value, str)
            ]
            edge_ids = [
                value
                for value in item.get("evidence_edge_ids", [])
                if isinstance(value, str)
            ]
            if not run_ids and not edge_ids:
                continue
            raw_coverage = item.get("comparator_coverage")
            coverage = normalize_coverage(raw_coverage) or {
                key: 0 for key in COVERAGE_KEYS
            }
            direction = directions.get(str(target_id), "none")
            if target_kind != "hypothesis" or relation in {
                "changed_dimension",
                "ambiguous",
            }:
                direction = "none"
            elif relation == "removed":
                direction = {
                    "positive": "negative",
                    "negative": "positive",
                }.get(direction, "none")
            comparator_gain = (
                item.get("evaluation_state") == "comparator_covered"
                and _contradiction_depth_bar(coverage)
                and direction in {"positive", "negative"}
            )
            rendered.append(
                {
                    "target_kind": target_kind,
                    "target_id": target_id,
                    "proposal_relation": relation,
                    "evaluation_state": item.get("evaluation_state"),
                    "comparator_coverage": coverage,
                    "evidence_run_ids": run_ids[:MAX_RUNS_PER_TARGET],
                    "evidence_edge_ids": edge_ids[:MAX_EDGES_PER_TARGET],
                    "acquisition_role": (
                        "comparator_gain" if comparator_gain else "uncertainty_only"
                    ),
                    "gain_direction": (
                        direction if comparator_gain else "none"
                    ),
                }
            )
    return sorted(
        rendered,
        key=lambda item: (
            0 if item["target_kind"] == "dimension" else 1,
            str(item["target_id"]),
        ),
    )


def conditioning_cited_ids(
    conditioning: Any,
) -> tuple[set[str], set[str]]:
    """Return evidence ids from normalized acquisition-conditioning entries."""
    run_ids: set[str] = set()
    edge_ids: set[str] = set()
    for item in conditioning if isinstance(conditioning, list) else []:
        if not isinstance(item, dict):
            continue
        run_ids.update(
            value
            for value in item.get("evidence_run_ids", [])
            if isinstance(value, str)
        )
        edge_ids.update(
            value
            for value in item.get("evidence_edge_ids", [])
            if isinstance(value, str)
        )
    return run_ids, edge_ids


def validate_conditioned_adjustment(
    conditioning: Any,
    point: Any,
    *,
    evidence_run_ids: Any,
    evidence_edge_ids: Any,
    gain_adjustment: Any,
    uncertainty_adjustment: Any,
) -> list[str]:
    """Validate one persisted/model-authored experience adjustment.

    The conditioning entries themselves are helper-rendered receipts.  This
    function checks their shape/relevance and then enforces that weak or
    confounded evidence cannot alter signed gain, while exact zero remains a
    legal abstention even when relevant observations exist.
    """
    errors: list[str] = []
    entries = conditioning if isinstance(conditioning, list) else []
    if not isinstance(conditioning, list):
        errors.append("conditioning must be a list")
    try:
        selected = selected_assignments(point) if isinstance(point, dict) else {}
    except Exception:
        selected = {}
    relevant = {
        "dimension": set(selected),
        "hypothesis": set(selected.values()),
    }
    seen_targets: set[tuple[str, str]] = set()
    valid_entries: list[dict[str, Any]] = []
    expected_fields = {
        "target_kind",
        "target_id",
        "proposal_relation",
        "evaluation_state",
        "comparator_coverage",
        "evidence_run_ids",
        "evidence_edge_ids",
        "acquisition_role",
        "gain_direction",
    }
    for index, item in enumerate(entries):
        where = f"conditioning[{index}]"
        if not isinstance(item, dict) or set(item) != expected_fields:
            errors.append(f"{where} has an invalid shape")
            continue
        target_kind = item.get("target_kind")
        target_id = item.get("target_id")
        relation = item.get("proposal_relation")
        if target_kind not in relevant or not isinstance(target_id, str):
            errors.append(f"{where} has an invalid semantic target")
        if relation not in PROPOSAL_RELATIONS:
            errors.append(f"{where}.proposal_relation is invalid")
        elif target_kind == "dimension" and relation not in {
            "changed_dimension",
            "ambiguous",
        }:
            errors.append(
                f"{where}.proposal_relation is invalid for a dimension target"
            )
        elif target_kind == "hypothesis" and relation == "changed_dimension":
            errors.append(
                f"{where}.proposal_relation is invalid for a hypothesis target"
            )
        elif (
            target_kind == "hypothesis"
            and relation in {"selected_fresh", "introduced"}
            and target_id not in relevant["hypothesis"]
        ):
            errors.append(
                f"{where} introduced/selected target is not selected by the point"
            )
        elif (
            target_kind == "hypothesis"
            and relation == "removed"
            and target_id in relevant["hypothesis"]
        ):
            errors.append(f"{where} removed target is still selected by the point")
        identity = (str(target_kind), str(target_id))
        if identity in seen_targets:
            errors.append(f"{where} duplicates target {target_id}")
        seen_targets.add(identity)
        coverage = normalize_coverage(item.get("comparator_coverage"))
        if coverage is None:
            errors.append(f"{where}.comparator_coverage is invalid")
            coverage = {key: 0 for key in COVERAGE_KEYS}
        run_ids = item.get("evidence_run_ids")
        edge_ids = item.get("evidence_edge_ids")
        if (
            not isinstance(run_ids, list)
            or len(run_ids) > MAX_RUNS_PER_TARGET
            or len(run_ids) != len(set(run_ids))
            or any(not isinstance(value, str) for value in run_ids)
        ):
            errors.append(f"{where}.evidence_run_ids is invalid")
        if (
            not isinstance(edge_ids, list)
            or len(edge_ids) > MAX_EDGES_PER_TARGET
            or len(edge_ids) != len(set(edge_ids))
            or any(not isinstance(value, str) for value in edge_ids)
        ):
            errors.append(f"{where}.evidence_edge_ids is invalid")
        expected_role = (
            "comparator_gain"
            if (
                item.get("evaluation_state") == "comparator_covered"
                and _contradiction_depth_bar(coverage)
                and item.get("gain_direction") in {"positive", "negative"}
            )
            else "uncertainty_only"
        )
        if item.get("acquisition_role") != expected_role:
            errors.append(f"{where}.acquisition_role is not mechanically derived")
        if (
            item.get("gain_direction") not in {"positive", "negative", "none"}
            or (
                expected_role == "uncertainty_only"
                and item.get("gain_direction") != "none"
            )
        ):
            errors.append(f"{where}.gain_direction is invalid for its role")
        valid_entries.append(item)

    cited_runs = (
        evidence_run_ids
        if isinstance(evidence_run_ids, list)
        and all(isinstance(value, str) for value in evidence_run_ids)
        else []
    )
    cited_edges = (
        evidence_edge_ids
        if isinstance(evidence_edge_ids, list)
        and all(isinstance(value, str) for value in evidence_edge_ids)
        else []
    )
    if (
        not isinstance(evidence_run_ids, list)
        or len(cited_runs) > MAX_RUNS_PER_TARGET
        or len(cited_runs) != len(set(cited_runs))
    ):
        errors.append("experience evidence_run_ids is invalid")
    if (
        not isinstance(evidence_edge_ids, list)
        or len(cited_edges) > MAX_EDGES_PER_TARGET
        or len(cited_edges) != len(set(cited_edges))
    ):
        errors.append("experience evidence_edge_ids is invalid")
    allowed_runs, allowed_edges = conditioning_cited_ids(valid_entries)
    if set(cited_runs) != allowed_runs:
        errors.append(
            "experience run citations must equal the exact union derived from "
            "the cited target conditioning"
        )
    if set(cited_edges) != allowed_edges:
        errors.append(
            "experience edge citations must equal the exact union derived from "
            "the cited target conditioning"
        )

    numeric_adjustments: dict[str, float | None] = {}
    for label, value in (
        ("gain", gain_adjustment),
        ("uncertainty", uncertainty_adjustment),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            errors.append(f"experience {label} adjustment must be finite numeric")
            numeric_adjustments[label] = None
        else:
            numeric_adjustments[label] = float(value)
    gain = numeric_adjustments["gain"]
    uncertainty = numeric_adjustments["uncertainty"]
    cited = set(cited_runs) | set(cited_edges)

    def entry_is_cited(item: dict[str, Any]) -> bool:
        return bool(
            cited
            & (
                set(item.get("evidence_run_ids", []))
                | set(item.get("evidence_edge_ids", []))
            )
        )

    if gain is not None and not math.isclose(gain, 0.0, abs_tol=1e-12):
        if abs(gain) > MAX_COMPARATOR_GAIN_ADJUSTMENT:
            errors.append(
                "experience gain adjustment exceeds the comparator-evidence cap "
                f"{MAX_COMPARATOR_GAIN_ADJUSTMENT:.2f}"
            )
        supporting = [
            item
            for item in valid_entries
            if item.get("acquisition_role") == "comparator_gain"
            and entry_is_cited(item)
        ]
        required_direction = "positive" if gain > 0 else "negative"
        if not supporting:
            errors.append(
                "nonzero experience gain requires cited comparator-covered target evidence"
            )
        elif any(
            item.get("gain_direction") != required_direction for item in supporting
        ):
            errors.append(
                "experience gain sign conflicts with cited comparator-covered assessments"
            )

    if uncertainty is not None and not math.isclose(
        uncertainty, 0.0, abs_tol=1e-12
    ):
        if abs(uncertainty) > MAX_EVIDENCE_UNCERTAINTY_ADJUSTMENT:
            errors.append(
                "experience uncertainty adjustment exceeds the evidence cap "
                f"{MAX_EVIDENCE_UNCERTAINTY_ADJUSTMENT:.2f}"
            )
        supporting = [item for item in valid_entries if entry_is_cited(item)]
        if not supporting:
            errors.append(
                "nonzero experience uncertainty requires cited relevant target evidence"
            )
        elif uncertainty < 0 and any(
            item.get("acquisition_role") != "comparator_gain"
            for item in supporting
        ):
            errors.append(
                "weak or confounded evidence may only preserve or increase uncertainty"
            )

    if (
        (gain is not None and not math.isclose(gain, 0.0, abs_tol=1e-12))
        or (
            uncertainty is not None
            and not math.isclose(uncertainty, 0.0, abs_tol=1e-12)
        )
    ) and not cited:
        errors.append("nonzero experience adjustments require evidence citations")
    if not valid_entries and (cited_runs or cited_edges):
        errors.append("experience citations require structured conditioning")
    return errors


def _change_class(changes: list[dict[str, Any]]) -> str:
    if not changes:
        return "same_point"
    return "single_dimension" if len(changes) == 1 else "multi_dimension"


def build_semantic_edges(
    prior_records: list[dict[str, Any]], child_record: dict[str, Any]
) -> list[dict[str, Any]]:
    by_id = {str(record["run_id"]): record for record in prior_records}
    child_id = str(child_record["run_id"])
    child_point = child_record["semantic_point"]
    receipts = []
    for parent_id in child_record.get("source_run_ids", []):
        parent_id = str(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            raise SemanticEvidenceError(
                f"record {child_id} parent {parent_id} is not an earlier record"
            )
        parent_point = parent["semantic_point"]
        changes = point_diff(parent_point, child_point)
        receipts.append({
            "schema_version": EDGE_SCHEMA_VERSION,
            "edge_id": f"sedge-{parent_id}-{child_id}",
            "space_revision": child_point["space_revision"],
            "parent_run_id": str(parent_id),
            "child_run_id": child_id,
            "change_class": _change_class(changes),
            "changes": changes,
        })
    return receipts


def validate_semantic_edges(
    prior_records: list[dict[str, Any]], child_record: dict[str, Any], registry: dict[str, Any]
) -> list[str]:
    """Rebuild the expected receipts and require exact persisted equality."""
    errors: list[str] = []
    child_id = str(child_record.get("run_id"))
    for error in validate_point(child_record.get("semantic_point"), registry):
        errors.append(f"record {child_id}: {error}")
    by_id = {
        str(record.get("run_id")): record
        for record in prior_records
        if isinstance(record, dict) and record.get("run_id") is not None
    }
    parents = child_record.get("source_run_ids")
    if not isinstance(parents, list):
        errors.append(f"record {child_id}.source_run_ids must be a list")
        parents = []
    for parent_id in parents:
        parent = by_id.get(str(parent_id))
        if parent is None:
            errors.append(f"record {child_id} parent {parent_id} is not an earlier record")
            continue
        for error in validate_point(parent.get("semantic_point"), registry):
            errors.append(f"record {parent_id}: {error}")
    if errors:
        return errors
    actual = child_record.get("semantic_edges")
    if not isinstance(actual, list):
        return [f"record {child_id}.semantic_edges must be a list of mechanical receipts"]
    if actual != build_semantic_edges(prior_records, child_record):
        errors.append(
            f"record {child_id}.semantic_edges must equal the receipts derived "
            "mechanically from the persisted parent and child semantic points"
        )
    return errors


def _records_by_id(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("run_id")): record
        for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("run_id") is not None
    }


def edge_index(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index every persisted semantic edge receipt by id, first write winning."""
    index: dict[str, dict[str, Any]] = {}
    for record in ledger.get("records", []):
        if not isinstance(record, dict):
            continue
        receipts = record.get("semantic_edges")
        if not isinstance(receipts, list):
            continue
        for receipt in receipts:
            if isinstance(receipt, dict) and isinstance(receipt.get("edge_id"), str):
                index.setdefault(receipt["edge_id"], receipt)
    return index


def _terminal_score(record: dict[str, Any]) -> float | None:
    if record.get("status") not in NONCRASH_TERMINAL_STATUSES:
        return None
    score = record.get("final_best_score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
    ):
        return None
    return float(score)


def _json_sha256(value: Any) -> str | None:
    """Return the tuner-compatible canonical JSON digest, or ``None``."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError):
        return None
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _finite_score(value: Any) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _like_for_like_delta(
    parent: dict[str, Any], child: dict[str, Any]
) -> float | None:
    """child − parent at matched evaluation depth, else ``None``.

    Warm scores pair with warm scores; terminal finals pair with finals.  A
    warm score is never compared against a tuning-lowered final.
    """
    parent_warm = _finite_score(parent.get("best_warm_score"))
    child_warm = _finite_score(child.get("best_warm_score"))
    if parent_warm is not None and child_warm is not None:
        return child_warm - parent_warm
    parent_final = _terminal_score(parent)
    child_final = _terminal_score(child)
    if parent_final is not None and child_final is not None:
        return child_final - parent_final
    return None


def hypothesis_carriers(ledger: dict[str, Any], *, target_id: str) -> dict[str, Any]:
    """Count independent contexts where adding ``target_id`` hurt or helped.

    A carrier edge's child point adds the hypothesis relative to the edge's
    parent.  Contexts group by parent run id: a context is negative when
    every carrier delta in it is strictly worse (positive — scores are
    lower-is-better) and positive when every delta is strictly better.
    Mixed, zero-delta, crash, and depth-unpaired edges never count.
    """
    records = _records_by_id(ledger)
    contexts: dict[str, list[float]] = {}
    for record in ledger.get("records", []):
        if not isinstance(record, dict):
            continue
        if record.get("status") not in NONCRASH_TERMINAL_STATUSES:
            continue
        receipts = record.get("semantic_edges")
        if not isinstance(receipts, list):
            continue
        for receipt in receipts:
            if not isinstance(receipt, dict):
                continue
            changes = receipt.get("changes")
            if not isinstance(changes, list):
                continue
            adds = any(
                isinstance(change, dict)
                and change.get("to_hypothesis_id") == target_id
                and change.get("from_hypothesis_id") != target_id
                for change in changes
            )
            if not adds:
                continue
            parent = records.get(str(receipt.get("parent_run_id")))
            if (
                parent is None
                or parent.get("status") not in NONCRASH_TERMINAL_STATUSES
            ):
                continue
            delta = _like_for_like_delta(parent, record)
            if delta is None or abs(delta) <= 1e-12:
                continue
            contexts.setdefault(str(receipt.get("parent_run_id")), []).append(delta)
    negative = sorted(
        parent_id
        for parent_id, deltas in contexts.items()
        if all(delta > 0 for delta in deltas)
    )
    positive = sorted(
        parent_id
        for parent_id, deltas in contexts.items()
        if all(delta < 0 for delta in deltas)
    )
    return {
        "negative": len(negative),
        "positive": len(positive),
        "negative_contexts": negative,
        "positive_contexts": positive,
    }


def validate_parameter_transfer_evidence(record: dict[str, Any]) -> list[str]:
    """Validate the durable inherited-control evidence on one ledger record.

    New (policy-receipt schema 6) non-fresh terminal records must carry the
    tuner-produced transfer receipt.  Its inherited row proves parameter
    continuity, not semantic isolation; only an additional paired semantic
    control can qualify. Historical records remain readable but never acquire
    direct-comparator status retroactively.
    """
    errors: list[str] = []
    run_id = str(record.get("run_id"))
    transfer = record.get("parameter_transfer")
    policy = record.get("policy_receipt")
    is_new_contract = (
        isinstance(policy, dict) and policy.get("schema_version") in {5, 6}
    )
    nonfresh = record.get("op") in {"improve", "crossover"}
    terminal_noncrash = record.get("status") in NONCRASH_TERMINAL_STATUSES

    if record.get("op") == "fresh":
        if transfer is not None:
            errors.append(
                f"record {run_id}.parameter_transfer must be null for a fresh root"
            )
        return errors
    if transfer is None:
        if is_new_contract and nonfresh and terminal_noncrash:
            errors.append(
                f"record {run_id}.parameter_transfer is required for a new "
                "non-fresh scored candidate"
            )
        return errors
    if not nonfresh:
        return [f"record {run_id}.parameter_transfer requires non-fresh ancestry"]
    if not isinstance(transfer, dict) or set(transfer) != {
        "receipt",
        "inherited_control",
        "warm_start_observations",
    }:
        return [
            f"record {run_id}.parameter_transfer must contain exactly receipt, "
            "inherited_control, and warm_start_observations"
        ]

    receipt = transfer.get("receipt")
    expected_receipt_fields = {
        "schema_version",
        "kind",
        "candidate",
        "primary_parent",
        "projection",
        "receipt_sha256",
    }
    receipt_version = (
        receipt.get("schema_version") if isinstance(receipt, dict) else None
    )
    if receipt_version == 2:
        expected_receipt_fields.add("semantic_control")
    if not isinstance(receipt, dict) or set(receipt) != expected_receipt_fields:
        return errors + [
            f"record {run_id}.parameter_transfer.receipt has an invalid shape"
        ]
    if receipt.get("schema_version") not in {1, 2} or receipt.get("kind") != (
        "primary_parent_parameter_transfer"
    ):
        errors.append(
            f"record {run_id}.parameter_transfer.receipt has an unsupported contract"
        )
    unhashed = dict(receipt)
    receipt_hash = unhashed.pop("receipt_sha256", None)
    if receipt_hash != _json_sha256(unhashed):
        errors.append(
            f"record {run_id}.parameter_transfer.receipt_sha256 is invalid"
        )

    candidate = receipt.get("candidate")
    candidate_fields = {
        "run_id",
        "path",
        "brief_path",
        "brief_sha256",
        "structure_sha256",
        "param_schema",
        "param_schema_sha256",
        "defaults",
        "defaults_sha256",
    }
    if not isinstance(candidate, dict) or set(candidate) != candidate_fields:
        errors.append(
            f"record {run_id}.parameter_transfer.receipt.candidate has an invalid shape"
        )
        candidate = {}
    if candidate.get("run_id") != run_id:
        errors.append(
            f"record {run_id}.parameter_transfer candidate run_id does not match"
        )
    child_schema = candidate.get("param_schema")
    child_defaults = candidate.get("defaults")
    if not isinstance(child_schema, dict) or not isinstance(child_defaults, dict):
        errors.append(
            f"record {run_id}.parameter_transfer candidate schema/defaults must be objects"
        )
        child_schema, child_defaults = {}, {}
    if candidate.get("param_schema_sha256") != _json_sha256(child_schema):
        errors.append(
            f"record {run_id}.parameter_transfer candidate schema hash is invalid"
        )
    if candidate.get("defaults_sha256") != _json_sha256(child_defaults):
        errors.append(
            f"record {run_id}.parameter_transfer candidate defaults hash is invalid"
        )
    if set(child_defaults) != set(child_schema):
        errors.append(
            f"record {run_id}.parameter_transfer candidate defaults do not cover its schema"
        )

    primary = receipt.get("primary_parent")
    primary_fields = {
        "run_id",
        "path",
        "entrypoint_sha256",
        "tune_report_path",
        "tune_report_sha256",
        "ledger_path",
        "ledger_record_sha256",
        "incumbent_source",
        "incumbent_score",
        "incumbent_params",
        "incumbent_params_sha256",
        "param_schema",
        "param_schema_sha256",
    }
    if not isinstance(primary, dict) or set(primary) != primary_fields:
        errors.append(
            f"record {run_id}.parameter_transfer.receipt.primary_parent has an invalid shape"
        )
        primary = {}
    parents = record.get("source_run_ids")
    parents = parents if isinstance(parents, list) else []
    if not parents or primary.get("run_id") != parents[0]:
        errors.append(
            f"record {run_id}.parameter_transfer must use the first numeric "
            "ancestor as its primary parent"
        )
    if primary.get("incumbent_source") not in {
        "applied_phase_a",
        "finalized_phase_c",
    }:
        errors.append(
            f"record {run_id}.parameter_transfer primary incumbent source is invalid"
        )
    parent_score = _finite_score(primary.get("incumbent_score"))
    if parent_score is None:
        errors.append(
            f"record {run_id}.parameter_transfer parent incumbent score must be finite"
        )
    parent_params = primary.get("incumbent_params")
    parent_schema = primary.get("param_schema")
    if not isinstance(parent_params, dict) or not isinstance(parent_schema, dict):
        errors.append(
            f"record {run_id}.parameter_transfer parent schema/params must be objects"
        )
        parent_params, parent_schema = {}, {}
    if primary.get("incumbent_params_sha256") != _json_sha256(parent_params):
        errors.append(
            f"record {run_id}.parameter_transfer parent params hash is invalid"
        )
    if primary.get("param_schema_sha256") != _json_sha256(parent_schema):
        errors.append(
            f"record {run_id}.parameter_transfer parent schema hash is invalid"
        )
    if set(parent_params) != set(parent_schema):
        errors.append(
            f"record {run_id}.parameter_transfer parent params do not cover its schema"
        )

    projection = receipt.get("projection")
    projection_fields = {"params", "params_sha256", "copied", "reset", "new", "dropped"}
    if not isinstance(projection, dict) or set(projection) != projection_fields:
        errors.append(
            f"record {run_id}.parameter_transfer.receipt.projection has an invalid shape"
        )
        projection = {}
    projected = projection.get("params")
    if not isinstance(projected, dict):
        errors.append(
            f"record {run_id}.parameter_transfer projected params must be an object"
        )
        projected = {}
    if projection.get("params_sha256") != _json_sha256(projected):
        errors.append(
            f"record {run_id}.parameter_transfer projected params hash is invalid"
        )
    if set(projected) != set(child_schema):
        errors.append(
            f"record {run_id}.parameter_transfer projection does not cover the child schema"
        )

    categories = {
        "copied": {"key", "value"},
        "reset": {"key", "parent_value", "child_value", "reason"},
        "new": {"key", "value", "reason"},
        "dropped": {"key", "value", "reason"},
    }
    keyed: dict[str, dict[str, dict[str, Any]]] = {}
    for category, fields in categories.items():
        items = projection.get(category)
        if (
            not isinstance(items, list)
            or any(not isinstance(item, dict) or set(item) != fields for item in items)
        ):
            errors.append(
                f"record {run_id}.parameter_transfer projection.{category} is invalid"
            )
            items = []
        mapping: dict[str, dict[str, Any]] = {}
        for item in items:
            key = item.get("key")
            if not isinstance(key, str) or not key or key in mapping:
                errors.append(
                    f"record {run_id}.parameter_transfer projection.{category} "
                    "must use unique non-empty keys"
                )
                continue
            mapping[key] = item
        keyed[category] = mapping
    child_classified = (
        set(keyed["copied"]) | set(keyed["reset"]) | set(keyed["new"])
    )
    parent_classified = (
        set(keyed["copied"]) | set(keyed["reset"]) | set(keyed["dropped"])
    )
    if child_classified != set(child_schema) or sum(
        len(keyed[name]) for name in ("copied", "reset", "new")
    ) != len(child_classified):
        errors.append(
            f"record {run_id}.parameter_transfer child keys are not classified exactly once"
        )
    if parent_classified != set(parent_schema) or sum(
        len(keyed[name]) for name in ("copied", "reset", "dropped")
    ) != len(parent_classified):
        errors.append(
            f"record {run_id}.parameter_transfer parent keys are not classified exactly once"
        )
    for key, item in keyed["copied"].items():
        if item.get("value") != parent_params.get(key) or item.get("value") != projected.get(key):
            errors.append(
                f"record {run_id}.parameter_transfer copied key {key!r} is not exact"
            )
    for key, item in keyed["reset"].items():
        if (
            item.get("parent_value") != parent_params.get(key)
            or item.get("child_value") != projected.get(key)
            or item.get("child_value") != child_defaults.get(key)
            or not isinstance(item.get("reason"), str)
            or not item.get("reason")
        ):
            errors.append(
                f"record {run_id}.parameter_transfer reset key {key!r} is inconsistent"
            )
    for key, item in keyed["new"].items():
        if (
            item.get("value") != projected.get(key)
            or item.get("value") != child_defaults.get(key)
            or item.get("reason") != "child_only"
        ):
            errors.append(
                f"record {run_id}.parameter_transfer new key {key!r} is inconsistent"
            )
    for key, item in keyed["dropped"].items():
        if (
            item.get("value") != parent_params.get(key)
            or item.get("reason") != "parent_only"
        ):
            errors.append(
                f"record {run_id}.parameter_transfer dropped key {key!r} is inconsistent"
            )

    semantic_control = receipt.get("semantic_control")
    semantic_control_status = "legacy_unverified"
    if receipt_version == 2:
        if not isinstance(semantic_control, dict):
            errors.append(
                f"record {run_id}.parameter_transfer semantic_control is invalid"
            )
            semantic_control = {}
        semantic_control_status = semantic_control.get("status")
        if semantic_control_status == "unverified":
            if set(semantic_control) != {"status", "reason"} or not isinstance(
                semantic_control.get("reason"), str
            ) or not semantic_control.get("reason"):
                errors.append(
                    f"record {run_id}.parameter_transfer unverified semantic_control "
                    "must carry a reason"
                )
        elif semantic_control_status == "paired":
            paired_fields = {
                "status",
                "method",
                "parameter",
                "target_dimension_id",
                "target_hypothesis_id",
                "control_params_sha256",
                "treatment_params_sha256",
            }
            if (
                set(semantic_control) != paired_fields
                or semantic_control.get("method")
                != "same_child_code_single_parameter"
                or not isinstance(semantic_control.get("parameter"), str)
                or not semantic_control.get("parameter")
                or not isinstance(
                    semantic_control.get("target_dimension_id"), str
                )
                or not isinstance(
                    semantic_control.get("target_hypothesis_id"), str
                )
            ):
                errors.append(
                    f"record {run_id}.parameter_transfer paired semantic_control "
                    "has an invalid contract"
                )
        else:
            errors.append(
                f"record {run_id}.parameter_transfer semantic_control.status "
                "must be unverified or paired"
            )

    control = transfer.get("inherited_control")
    control_fields = {
        "warm_config_index",
        "selected",
        "primary_parent_run_id",
        "parent_incumbent_score",
        "params_sha256",
        "receipt_sha256",
    }
    if not isinstance(control, dict) or set(control) != control_fields:
        errors.append(
            f"record {run_id}.parameter_transfer.inherited_control has an invalid shape"
        )
        control = {}
    if control.get("warm_config_index") != 0 or control.get("selected") is not True:
        errors.append(
            f"record {run_id}.parameter_transfer inherited control must be mandatory index 0"
        )
    if (
        control.get("primary_parent_run_id") != primary.get("run_id")
        or control.get("parent_incumbent_score") != primary.get("incumbent_score")
        or control.get("params_sha256") != projection.get("params_sha256")
        or control.get("receipt_sha256") != receipt_hash
    ):
        errors.append(
            f"record {run_id}.parameter_transfer inherited-control pointer is inconsistent"
        )

    observations = transfer.get("warm_start_observations")
    max_observations = 2 if semantic_control_status == "paired" else 1
    if not isinstance(observations, list) or len(observations) > max_observations:
        errors.append(
            f"record {run_id}.parameter_transfer has an invalid number of "
            "control observations"
        )
        observations = []
    for observation_index, observation in enumerate(observations):
        required = {
            "params",
            "score",
            "proposed_index",
            "role",
            "parameter_transfer_receipt_sha256",
            "params_sha256",
        }
        if not isinstance(observation, dict) or not required.issubset(observation):
            errors.append(
                f"record {run_id}.parameter_transfer control observation is invalid"
            )
            continue
        expected_role = (
            "inherited_control"
            if observation_index == 0
            else "semantic_treatment"
        )
        expected_index = observation_index
        expected_params_hash = (
            projection.get("params_sha256")
            if observation_index == 0
            else (
                semantic_control.get("treatment_params_sha256")
                if isinstance(semantic_control, dict)
                else None
            )
        )
        if (
            observation.get("proposed_index") != expected_index
            or observation.get("role") != expected_role
            or observation.get("parameter_transfer_receipt_sha256") != receipt_hash
            or observation.get("params_sha256") != expected_params_hash
            or (
                observation_index == 0
                and observation.get("params") != projected
            )
            or _finite_score(observation.get("score")) is None
        ):
            errors.append(
                f"record {run_id}.parameter_transfer control observation "
                "does not match the projected config"
            )
    if semantic_control_status == "paired" and len(observations) == 2:
        control_params = observations[0].get("params")
        treatment_params = observations[1].get("params")
        switch = semantic_control.get("parameter")
        if (
            not isinstance(control_params, dict)
            or not isinstance(treatment_params, dict)
            or set(control_params) != set(treatment_params)
            or set(control_params) != set(child_schema)
            or treatment_params == control_params
            or [
                key
                for key in control_params
                if control_params.get(key) != treatment_params.get(key)
            ]
            != [switch]
            or semantic_control.get("control_params_sha256")
            != _json_sha256(control_params)
            or semantic_control.get("treatment_params_sha256")
            != _json_sha256(treatment_params)
        ):
            errors.append(
                f"record {run_id}.parameter_transfer paired semantic control "
                "must differ in exactly its declared parameter"
            )
    if (
        is_new_contract
        and record.get("status") in NONCRASH_TERMINAL_STATUSES
        and len(observations) != max_observations
    ):
        errors.append(
            f"record {run_id}.parameter_transfer requires every mandatory "
            "control observation before a non-crash terminal result"
        )
    return errors


def validate_parameter_transfer_binding(
    ledger: dict[str, Any], record: dict[str, Any]
) -> list[str]:
    """Bind a self-consistent transfer receipt to its durable parent snapshot.

    ``validate_parameter_transfer_evidence`` proves only the receipt's internal
    hashes and pointers.  That is insufficient at the ledger boundary: an
    internally rehashed receipt could otherwise invent a different parent
    score.  The binding may target the parent's current record or an append-only
    lineage snapshot captured before that parent was tuned again.
    """
    errors = validate_parameter_transfer_evidence(record)
    if errors or record.get("parameter_transfer") is None:
        return errors
    transfer = record["parameter_transfer"]
    receipt = transfer.get("receipt") if isinstance(transfer, dict) else None
    primary = receipt.get("primary_parent") if isinstance(receipt, dict) else None
    if not isinstance(primary, dict):
        return errors

    parent_run_id = primary.get("run_id")
    records = _records_by_id(ledger)
    parent = records.get(str(parent_run_id))
    run_id = str(record.get("run_id"))
    if parent is None or parent is record:
        errors.append(
            f"record {run_id}.parameter_transfer primary parent is not an "
            "earlier durable ledger record"
        )
        return errors

    expected_record_hash = primary.get("ledger_record_sha256")
    current_record_hash = _json_sha256(parent)
    snapshot = None
    parent_score = None
    if expected_record_hash == current_record_hash:
        snapshot = parent.get("applied_incumbent")
        parent_score = _terminal_score(parent)
    else:
        matching_revisions = [
            item
            for item in ledger.get("lineage_snapshots", [])
            if isinstance(item, dict)
            and item.get("parent_run_id") == str(parent_run_id)
            and item.get("ledger_record_sha256") == expected_record_hash
        ]
        if len(matching_revisions) != 1:
            errors.append(
                f"record {run_id}.parameter_transfer parent ledger-record hash "
                "matches neither the current parent nor one durable lineage snapshot"
            )
            return errors
        revision = matching_revisions[0]
        revision_fields = {
            "schema_version",
            "kind",
            "parent_run_id",
            "ledger_record_sha256",
            "final_best_score",
            "applied_incumbent",
            "captured_by_run_id",
            "receipt_sha256",
        }
        unhashed_revision = dict(revision)
        revision_hash = unhashed_revision.pop("receipt_sha256", None)
        if (
            set(revision) != revision_fields
            or revision.get("schema_version") != 1
            or revision.get("kind") != "parameter_transfer_parent_snapshot"
            or revision_hash != _json_sha256(unhashed_revision)
        ):
            errors.append(
                f"record {run_id}.parameter_transfer parent lineage snapshot is invalid"
            )
            return errors
        snapshot = revision.get("applied_incumbent")
        parent_score = _finite_score(revision.get("final_best_score"))

    snapshot_fields = {
        "schema_version",
        "source",
        "score",
        "params",
        "params_sha256",
        "param_schema",
        "param_schema_sha256",
        "entrypoint_sha256",
        "tune_report_sha256",
    }
    if not isinstance(snapshot, dict) or set(snapshot) != snapshot_fields:
        errors.append(
            f"record {run_id}.parameter_transfer parent has no exact applied "
            "incumbent snapshot"
        )
        return errors
    snapshot_params = snapshot.get("params")
    snapshot_schema = snapshot.get("param_schema")
    if (
        snapshot.get("schema_version") != 1
        or snapshot.get("source") not in {
            "applied_phase_a",
            "finalized_phase_c",
        }
        or not isinstance(snapshot_params, dict)
        or not isinstance(snapshot_schema, dict)
        or snapshot.get("params_sha256") != _json_sha256(snapshot_params)
        or snapshot.get("param_schema_sha256") != _json_sha256(snapshot_schema)
    ):
        errors.append(
            f"record {run_id}.parameter_transfer parent applied-incumbent "
            "snapshot is invalid"
        )
        return errors

    if (
        parent_score is None
        or _finite_score(primary.get("incumbent_score")) != parent_score
        or _finite_score(snapshot.get("score")) != parent_score
    ):
        errors.append(
            f"record {run_id}.parameter_transfer parent incumbent score does "
            "not match the durable parent record"
        )
    bound_pairs = (
        ("incumbent_source", "source"),
        ("incumbent_params", "params"),
        ("incumbent_params_sha256", "params_sha256"),
        ("param_schema", "param_schema"),
        ("param_schema_sha256", "param_schema_sha256"),
        ("entrypoint_sha256", "entrypoint_sha256"),
        ("tune_report_sha256", "tune_report_sha256"),
    )
    for primary_field, snapshot_field in bound_pairs:
        if primary.get(primary_field) != snapshot.get(snapshot_field):
            errors.append(
                f"record {run_id}.parameter_transfer parent {primary_field} "
                "does not match its durable applied-incumbent snapshot"
            )
    return errors


def unbound_primary_descendants(
    ledger: dict[str, Any], parent_run_id: str
) -> list[str]:
    """Primary children of ``parent_run_id`` whose transfer binding is not settled.

    A child is settled when its receipt names this parent and
    :func:`validate_parameter_transfer_binding` accepts it. Two states are not
    settled and must block a parent mutation: a *pending* child whose transfer
    is still being built (mutating the parent would race it) and a *scored*
    child whose binding is broken (mutating the parent would hide it). A
    terminal child that produced no usable observation — ``crash`` or
    ``unevaluated`` with no transfer at all — never had a binding to preserve
    and is skipped.

    Single source of truth for both consumers: ``ledger._preserve_descendant_bindings``
    (which additionally snapshots each settled binding before mutating) and
    ``tune_tools.select_candidate`` (which uses it as a tuning-eligibility gate).
    Returned ids are sorted so callers can report them deterministically.
    """
    target = str(parent_run_id)
    unbound: list[str] = []
    for child in ledger.get("records", []):
        if not isinstance(child, dict):
            continue
        parents = child.get("source_run_ids")
        if not isinstance(parents, list) or not parents or str(parents[0]) != target:
            continue
        transfer = child.get("parameter_transfer")
        receipt = transfer.get("receipt") if isinstance(transfer, dict) else None
        primary = receipt.get("primary_parent") if isinstance(receipt, dict) else None
        if not isinstance(primary, dict) or str(primary.get("run_id")) != target:
            if (
                transfer is None
                and child.get("status") in NO_OBSERVATION_TERMINAL_STATUSES
            ):
                continue
            unbound.append(str(child.get("run_id")))
            continue
        if validate_parameter_transfer_binding(ledger, child):
            unbound.append(str(child.get("run_id")))
    return sorted(unbound)


def validate_lineage_snapshots(ledger: dict[str, Any]) -> list[str]:
    """Validate the append-only parent revisions used by transfer bindings."""
    snapshots = ledger.get("lineage_snapshots", [])
    if snapshots is None:
        return []
    if not isinstance(snapshots, list):
        return ["ledger.lineage_snapshots must be a list"]
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    fields = {
        "schema_version",
        "kind",
        "parent_run_id",
        "ledger_record_sha256",
        "final_best_score",
        "applied_incumbent",
        "captured_by_run_id",
        "receipt_sha256",
    }
    for index, item in enumerate(snapshots):
        where = f"ledger.lineage_snapshots[{index}]"
        if not isinstance(item, dict) or set(item) != fields:
            errors.append(f"{where} has an invalid shape")
            continue
        unhashed = dict(item)
        receipt_hash = unhashed.pop("receipt_sha256", None)
        key = (
            str(item.get("parent_run_id")),
            str(item.get("ledger_record_sha256")),
        )
        if key in seen:
            errors.append(f"{where} duplicates parent revision {key}")
        seen.add(key)
        if (
            item.get("schema_version") != 1
            or item.get("kind") != "parameter_transfer_parent_snapshot"
            or not str(item.get("parent_run_id", "")).isdigit()
            or not str(item.get("captured_by_run_id", "")).isdigit()
            or not isinstance(item.get("ledger_record_sha256"), str)
            or _finite_score(item.get("final_best_score")) is None
            or not isinstance(item.get("applied_incumbent"), dict)
            or receipt_hash != _json_sha256(unhashed)
        ):
            errors.append(f"{where} is not a valid parent revision receipt")
    return errors


def matched_inherited_control(
    ledger: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any] | None:
    """Return a strict same-child-code semantic pair, if fully qualified.

    The historical function name remains for backward-readable callers.  A
    parent-parameter inheritance row by itself deliberately returns ``None``.
    Production ledgers currently carry an explicit unavailable-capability gate,
    so the paired branch below is retained only as a tested future contract.
    """
    if not _direct_comparators_enabled(ledger):
        return None
    records = _records_by_id(ledger)
    parent = records.get(str(receipt.get("parent_run_id")))
    child = records.get(str(receipt.get("child_run_id")))
    if (
        parent is None
        or child is None
        or receipt.get("change_class") != "single_dimension"
        or parent.get("status") not in NONCRASH_TERMINAL_STATUSES
        or child.get("status") not in NONCRASH_TERMINAL_STATUSES
        or not isinstance(child.get("policy_receipt"), dict)
        or child["policy_receipt"].get("schema_version") != 6
        or validate_parameter_transfer_binding(ledger, child)
    ):
        return None
    transfer = child.get("parameter_transfer")
    if not isinstance(transfer, dict):
        return None
    transfer_receipt = transfer["receipt"]
    primary = transfer_receipt["primary_parent"]
    projection = transfer_receipt["projection"]
    semantic_control = transfer_receipt.get("semantic_control")
    changes = receipt.get("changes")
    if (
        transfer_receipt.get("schema_version") != 2
        or not isinstance(semantic_control, dict)
        or semantic_control.get("status") != "paired"
        or not isinstance(changes, list)
        or len(changes) != 1
        or semantic_control.get("target_dimension_id")
        != changes[0].get("dimension_id")
        or semantic_control.get("target_hypothesis_id")
        != changes[0].get("to_hypothesis_id")
        or primary.get("run_id") != str(receipt.get("parent_run_id"))
        or projection.get("reset") != []
    ):
        return None
    observations = transfer.get("warm_start_observations")
    if not isinstance(observations, list) or len(observations) != 2:
        return None
    parent_score = _finite_score(primary.get("incumbent_score"))
    baseline_score = _finite_score(observations[0].get("score"))
    treatment_score = _finite_score(observations[1].get("score"))
    child_final_score = _terminal_score(child)
    if (
        parent_score is None
        or baseline_score is None
        or treatment_score is None
        or child_final_score is None
    ):
        return None
    return {
        "score_basis": "paired_semantic_control",
        "parent_incumbent_score": parent_score,
        "child_baseline_score": baseline_score,
        "child_control_score": treatment_score,
        "parent_reproduction_delta": baseline_score - parent_score,
        "semantic_delta": treatment_score - baseline_score,
        "child_final_score": child_final_score,
        "tuning_delta": child_final_score - treatment_score,
        "total_delta": child_final_score - parent_score,
        "parameter_transfer_receipt_sha256": transfer_receipt["receipt_sha256"],
    }


def edge_observation(ledger: dict[str, Any], edge_id: str) -> dict[str, Any]:
    """Report one edge, separating semantic-control and tuning deltas."""
    receipt = edge_index(ledger).get(str(edge_id))
    if receipt is None:
        return {}
    records = _records_by_id(ledger)
    parent = records.get(str(receipt.get("parent_run_id")), {})
    child = records.get(str(receipt.get("child_run_id")), {})
    parent_score = _terminal_score(parent)
    child_score = _terminal_score(child)
    matched = matched_inherited_control(ledger, receipt)
    delta = None
    if parent_score is not None and child_score is not None:
        delta = round(child_score - parent_score, 4)
    observation = {
        "edge_id": receipt["edge_id"],
        "change_class": receipt.get("change_class"),
        "parent_run_id": receipt.get("parent_run_id"),
        "child_run_id": receipt.get("child_run_id"),
        "parent_status": parent.get("status"),
        "child_status": child.get("status"),
        "parent_score": parent_score,
        "child_score": child_score,
        "delta": delta,
        "score_basis": "independently_tuned_final",
    }
    if matched is not None:
        observation.update(
            {
                **matched,
                "parent_score": matched["parent_incumbent_score"],
                "child_score": matched["child_control_score"],
                "delta": round(matched["semantic_delta"], 4),
                "semantic_delta": round(matched["semantic_delta"], 4),
                "tuning_delta": round(matched["tuning_delta"], 4),
                "total_delta": round(matched["total_delta"], 4),
            }
        )
    return observation


def _edge_touches(receipt: dict[str, Any], *, target_kind: str, target_id: str) -> bool:
    changes = receipt.get("changes")
    if not isinstance(changes, list):
        return False
    for change in changes:
        if not isinstance(change, dict):
            continue
        if target_kind == "dimension" and change.get("dimension_id") == target_id:
            return True
        if target_kind == "hypothesis" and target_id in (
            change.get("from_hypothesis_id"),
            change.get("to_hypothesis_id"),
        ):
            return True
    return False


def _coverage_category(
    ledger: dict[str, Any], receipt: dict[str, Any]
) -> str | None:
    records = _records_by_id(ledger)
    statuses = {
        records.get(str(receipt.get("parent_run_id")), {}).get("status"),
        records.get(str(receipt.get("child_run_id")), {}).get("status"),
    }
    if not statuses.issubset(TERMINAL_STATUSES):
        return None
    if "crash" in statuses:
        return "crash_edges"
    if matched_inherited_control(ledger, receipt) is not None:
        # A direct comparator is contradiction-grade when the child was
        # deep-tuned, intermediate at `tuned_lightly`; legacy records without
        # evaluation_depth read as screening and fail closed into the weaker
        # category.
        child = records.get(str(receipt.get("child_run_id")), {})
        depth = child.get("evaluation_depth")
        if depth == "tuned":
            return "direct_tuned_edges"
        if depth == "tuned_lightly":
            return "direct_lightly_tuned_edges"
        return "direct_noncrash_edges"
    return "confounded_noncrash_edges"


def comparator_coverage(
    ledger: dict[str, Any], edge_ids: list[str], *, target_kind: str, target_id: str
) -> dict[str, int]:
    """Classify only cited receipts that touch the requested target."""
    coverage = {key: 0 for key in COVERAGE_KEYS}
    index = edge_index(ledger)
    for edge_id in dict.fromkeys(str(value) for value in edge_ids):
        receipt = index.get(str(edge_id))
        if receipt is None or not _edge_touches(
            receipt, target_kind=target_kind, target_id=target_id
        ):
            continue
        category = _coverage_category(ledger, receipt)
        if category is not None:
            coverage[category] += 1
    return coverage


def mechanical_gain_direction(
    ledger: dict[str, Any],
    *,
    target_kind: str,
    target_id: str,
    evidence_edge_ids: list[str],
) -> str:
    """Orient repeated paired semantic-control deltas for one hypothesis.

    A negative target-present score effect is mechanically favorable because
    scores are lower-is-better.  Direct controls must clear the contradiction
    depth bar and agree in sign: at least two tuned, or at least three at
    tuned_lightly or deeper.  Mixed, zero, legacy final-vs-final,
    reset-bearing, and crash edges all abstain.
    """
    if target_kind != "hypothesis":
        return "none"
    records = _records_by_id(ledger)
    index = edge_index(ledger)
    oriented_effects: list[tuple[float, bool]] = []
    for edge_id in dict.fromkeys(str(value) for value in evidence_edge_ids):
        receipt = index.get(edge_id)
        if receipt is None:
            continue
        # Tuned-child controls orient a direction at full weight;
        # tuned_lightly controls orient only in numbers (the depth bar).
        # Screening-depth controls measure one parameter point and abstain.
        child = records.get(str(receipt.get("child_run_id")), {})
        depth = child.get("evaluation_depth")
        if depth not in {"tuned", "tuned_lightly"}:
            continue
        matched = matched_inherited_control(ledger, receipt)
        if matched is None:
            continue
        changes = receipt.get("changes")
        if not isinstance(changes, list) or len(changes) != 1:
            continue
        change = changes[0]
        delta = float(matched["semantic_delta"])
        is_tuned = depth == "tuned"
        if change.get("to_hypothesis_id") == target_id:
            oriented_effects.append((delta, is_tuned))
        elif change.get("from_hypothesis_id") == target_id:
            oriented_effects.append((-delta, is_tuned))
    strong = [effect for effect, is_tuned in oriented_effects if is_tuned]
    effects = [effect for effect, _ in oriented_effects]
    if not (
        len(strong) >= MIN_EDGES_PER_TARGET
        or len(effects) >= DEPTH_BAR_LIGHT_MIN
    ):
        return "none"
    if all(effect < -1e-12 for effect in effects):
        return "positive"
    if all(effect > 1e-12 for effect in effects):
        return "negative"
    return "none"


def mechanical_gain_directions(
    ledger: dict[str, Any], experience: Any
) -> dict[str, str]:
    """Derive hypothesis directions from each snapshot entry's cited controls."""
    if not isinstance(experience, dict):
        return {}
    directions: dict[str, str] = {}
    for item in experience.get("hypothesis_evidence", []) or []:
        if not isinstance(item, dict) or not isinstance(item.get("target_id"), str):
            continue
        target_id = item["target_id"]
        directions[target_id] = mechanical_gain_direction(
            ledger,
            target_kind="hypothesis",
            target_id=target_id,
            evidence_edge_ids=(
                item.get("evidence_edge_ids")
                if isinstance(item.get("evidence_edge_ids"), list)
                else []
            ),
        )
    return directions


def validate_conditioning_against_ledger(
    conditioning: Any, ledger: dict[str, Any]
) -> list[str]:
    """Recompute every persisted conditioning fact from cited ledger evidence."""
    if not isinstance(conditioning, list):
        return ["conditioning must be a list"]
    errors: list[str] = []
    records = _records_by_id(ledger)
    index = edge_index(ledger)
    for position, item in enumerate(conditioning):
        where = f"conditioning[{position}]"
        if not isinstance(item, dict):
            continue
        target_kind = item.get("target_kind")
        target_id = item.get("target_id")
        relation = item.get("proposal_relation")
        if target_kind not in {"dimension", "hypothesis"} or not isinstance(
            target_id, str
        ):
            continue
        run_ids = item.get("evidence_run_ids")
        run_ids = run_ids if isinstance(run_ids, list) else []
        edge_ids = item.get("evidence_edge_ids")
        edge_ids = edge_ids if isinstance(edge_ids, list) else []
        touching_endpoints: set[str] = set()
        for edge_id in edge_ids:
            receipt = index.get(str(edge_id))
            if receipt is None or not _edge_touches(
                receipt, target_kind=target_kind, target_id=target_id
            ):
                errors.append(
                    f"{where}.evidence_edge_ids contains a missing or "
                    "non-target-touching edge"
                )
                continue
            touching_endpoints.update(
                {
                    str(receipt.get("parent_run_id")),
                    str(receipt.get("child_run_id")),
                }
            )
        for run_id in run_ids:
            record = records.get(str(run_id))
            if (
                record is None
                or record.get("status") not in TERMINAL_STATUSES
                or (
                    str(run_id) not in touching_endpoints
                    and not _run_bears_target(
                        record, target_kind=target_kind, target_id=target_id
                    )
                )
            ):
                errors.append(
                    f"{where}.evidence_run_ids contains a missing, non-terminal, "
                    "or target-irrelevant run"
                )
        expected_coverage = comparator_coverage(
            ledger,
            edge_ids,
            target_kind=target_kind,
            target_id=target_id,
        )
        if item.get("comparator_coverage") != expected_coverage:
            errors.append(
                f"{where}.comparator_coverage is not the mechanical cited-edge coverage"
            )
        expected_state = target_evaluation_state(
            ledger,
            target_kind=target_kind,
            target_id=target_id,
            evidence_run_ids=run_ids,
            evidence_edge_ids=edge_ids,
        )
        if item.get("evaluation_state") != expected_state:
            errors.append(
                f"{where}.evaluation_state is not mechanically derived"
            )
        raw_direction = mechanical_gain_direction(
            ledger,
            target_kind=target_kind,
            target_id=target_id,
            evidence_edge_ids=edge_ids,
        )
        expected_direction = raw_direction
        if target_kind != "hypothesis" or relation in {
            "changed_dimension",
            "ambiguous",
        }:
            expected_direction = "none"
        elif relation == "removed":
            expected_direction = {
                "positive": "negative",
                "negative": "positive",
            }.get(raw_direction, "none")
        expected_role = (
            "comparator_gain"
            if (
                expected_state == "comparator_covered"
                and _contradiction_depth_bar(expected_coverage)
                and expected_direction in {"positive", "negative"}
            )
            else "uncertainty_only"
        )
        if (
            item.get("gain_direction") != expected_direction
            or item.get("acquisition_role") != expected_role
        ):
            errors.append(
                f"{where} gain direction/role is not derived from repeated "
                "matched semantic-control pairs"
            )
    return errors


def _run_bears_target(record: dict[str, Any], *, target_kind: str, target_id: str) -> bool:
    point = record.get("semantic_point")
    if not isinstance(point, dict):
        return False
    selected = selected_assignments(point)
    if target_kind == "hypothesis":
        return target_id in selected.values()
    return target_id in selected


def target_evaluation_state(
    ledger: dict[str, Any],
    *,
    target_kind: str,
    target_id: str,
    evidence_run_ids: list[str],
    evidence_edge_ids: list[str],
) -> str:
    """Apply the four mechanical belief-coverage rules to cited evidence only.

    Both the bounded renderer and the experience validator call this same
    function so their state labels cannot drift.
    """
    records = _records_by_id(ledger)
    index = edge_index(ledger)
    cited_terminal = [
        records[str(run_id)]
        for run_id in evidence_run_ids or []
        if str(run_id) in records
        and records[str(run_id)].get("status") in TERMINAL_STATUSES
    ]
    cited_edges = [
        index[str(edge_id)]
        for edge_id in evidence_edge_ids or []
        if str(edge_id) in index
        and _coverage_category(ledger, index[str(edge_id)]) is not None
    ]
    if not cited_terminal and not cited_edges:
        return "unevaluated"
    coverage = comparator_coverage(
        ledger,
        [str(edge_id) for edge_id in evidence_edge_ids or []],
        target_kind=target_kind,
        target_id=target_id,
    )
    if (
        coverage["direct_tuned_edges"] + coverage["direct_lightly_tuned_edges"]
        >= MIN_EDGES_PER_TARGET
    ):
        return "comparator_covered"
    noncrash_observation = (
        coverage["direct_tuned_edges"]
        + coverage["direct_lightly_tuned_edges"]
        + coverage["direct_noncrash_edges"]
        + coverage["confounded_noncrash_edges"]
        > 0
    ) or any(
        record.get("status") in NONCRASH_TERMINAL_STATUSES
        and _run_bears_target(record, target_kind=target_kind, target_id=target_id)
        for record in cited_terminal
    )
    if noncrash_observation:
        return "observed"
    return "failed"


def _edge_sort_key(receipt: dict[str, Any]) -> tuple[int, int]:
    match = EDGE_ID_RE.fullmatch(str(receipt.get("edge_id", "")))
    if match is None:
        return (-1, -1)
    return (int(match.group(2)), int(match.group(1)))


def _record_dag_revision(record: dict[str, Any]) -> int:
    value = record.get("dag_revision", 0)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _run_sort_key(record: dict[str, Any]) -> int:
    run_id = str(record.get("run_id"))
    return int(run_id) if run_id.isdigit() else -1


def _require_cap(name: str, value: Any, low: int, high: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise SemanticEvidenceError(f"{name} must be an integer in [{low}, {high}]")


def _target_block(
    ledger: dict[str, Any],
    touching: list[dict[str, Any]],
    *,
    target_kind: str,
    target_id: str,
    max_edges_per_target: int,
) -> dict[str, Any]:
    pools: dict[str, list[dict[str, Any]]] = {key: [] for key in COVERAGE_KEYS}
    for receipt in touching:
        category = _coverage_category(ledger, receipt)
        if category is not None:
            pools[category].append(receipt)
    for pool in pools.values():
        pool.sort(key=_edge_sort_key, reverse=True)
    # Direct comparators first, strongest depth first; each pool is already
    # recency-sorted. Screening-depth direct edges remain displayable direct
    # evidence — they just cannot drive contradiction gates.
    direct = (
        pools["direct_tuned_edges"]
        + pools["direct_lightly_tuned_edges"]
        + pools["direct_noncrash_edges"]
    )
    confounded = pools["confounded_noncrash_edges"]
    crash = pools["crash_edges"]

    selected: list[dict[str, Any]] = []

    def take(pool: list[dict[str, Any]], count: int) -> None:
        remaining = max_edges_per_target - len(selected)
        if remaining > 0:
            selected.extend(pool[:min(count, remaining)])

    take(direct, 2)
    take(confounded, 1)
    take(crash, 1)
    for pool in (direct, confounded, crash):
        for receipt in pool:
            if len(selected) >= max_edges_per_target:
                break
            if all(receipt["edge_id"] != kept["edge_id"] for kept in selected):
                selected.append(receipt)
    selected.sort(key=_edge_sort_key)
    edge_ids = [receipt["edge_id"] for receipt in selected]

    run_ids: list[str] = []
    for receipt in selected:
        for run_id in (receipt.get("parent_run_id"), receipt.get("child_run_id")):
            run_id = str(run_id)
            if run_id not in run_ids:
                run_ids.append(run_id)
    bearing = sorted(
        (
            record
            for record in _records_by_id(ledger).values()
            if record.get("status") in TERMINAL_STATUSES
            and _run_bears_target(record, target_kind=target_kind, target_id=target_id)
        ),
        key=_run_sort_key,
        reverse=True,
    )
    for record in bearing:
        run_id = str(record.get("run_id"))
        if run_id not in run_ids:
            run_ids.append(run_id)
    run_ids = run_ids[:MAX_RUNS_PER_TARGET]

    coverage = comparator_coverage(
        ledger, edge_ids, target_kind=target_kind, target_id=target_id
    )
    available = comparator_coverage(
        ledger,
        [receipt["edge_id"] for receipt in touching],
        target_kind=target_kind,
        target_id=target_id,
    )
    omitted = {key: available[key] - coverage[key] for key in COVERAGE_KEYS}
    return {
        "target_id": target_id,
        "evaluation_state": target_evaluation_state(
            ledger,
            target_kind=target_kind,
            target_id=target_id,
            evidence_run_ids=run_ids,
            evidence_edge_ids=edge_ids,
        ),
        "evidence_run_ids": run_ids,
        "evidence_edge_ids": edge_ids,
        "comparator_coverage": coverage,
        "available_comparator_coverage": available,
        "omitted_edge_counts": omitted,
        "edges": [edge_observation(ledger, edge_id) for edge_id in edge_ids],
    }


def render_target_evidence(
    registry: dict[str, Any],
    ledger: dict[str, Any],
    *,
    max_dimensions: int = 16,
    max_hypotheses: int = 32,
    max_edges_per_target: int = 5,
    target_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Scan the persisted edge index and render bounded per-target evidence.

    The extractor never reconstructs comparator coverage from the Top/Bottom
    graph window: this deterministic scan cites exact persisted edge ids, and
    discloses bounded-view loss through ``available_comparator_coverage`` and
    ``omitted_edge_counts``.
    """
    _require_cap("max_dimensions", max_dimensions, 1, MAX_DIMENSION_TARGETS)
    _require_cap("max_hypotheses", max_hypotheses, 1, MAX_HYPOTHESIS_TARGETS)
    _require_cap(
        "max_edges_per_target", max_edges_per_target, MIN_EDGES_PER_TARGET, MAX_EDGES_PER_TARGET
    )

    dimensions = dimension_map(registry)
    hypotheses = hypothesis_map(registry)
    index = edge_index(ledger)
    records = _records_by_id(ledger)
    experience = ledger.get("experience")
    experience = experience if isinstance(experience, dict) else {}
    experience_dag_revision = _record_dag_revision(experience)

    def build(target_kind: str, target_id: str) -> dict[str, Any]:
        touching = [
            receipt
            for receipt in index.values()
            if _edge_touches(receipt, target_kind=target_kind, target_id=target_id)
        ]
        return _target_block(
            ledger,
            touching,
            target_kind=target_kind,
            target_id=target_id,
            max_edges_per_target=max_edges_per_target,
        )

    dimension_blocks: list[dict[str, Any]] = []
    hypothesis_blocks: list[dict[str, Any]] = []
    omitted_dimensions: list[str] = []
    omitted_hypotheses: list[str] = []

    if target_ids is not None:
        requested = list(dict.fromkeys(str(item) for item in target_ids))
        unknown = sorted(set(requested) - set(dimensions) - set(hypotheses))
        if unknown:
            raise SemanticEvidenceError(f"unknown --target-id values {unknown}")
        for target_id in requested:
            if target_id in dimensions:
                dimension_blocks.append(build("dimension", target_id))
            else:
                hypothesis_blocks.append(build("hypothesis", target_id))
    else:
        preserved = {
            item.get("target_id")
            for field in ("dimension_evidence", "hypothesis_evidence")
            for item in experience.get(field, []) or []
            if isinstance(item, dict)
        }

        def has_new_dag_evidence(touching: list[dict[str, Any]]) -> bool:
            for receipt in touching:
                for run_id in (receipt.get("parent_run_id"), receipt.get("child_run_id")):
                    record = records.get(str(run_id))
                    if record is not None and (
                        _record_dag_revision(record) > experience_dag_revision
                    ):
                        return True
            return False

        candidates: dict[str, list[tuple[int, int, str]]] = {
            "dimension": [],
            "hypothesis": [],
        }
        registry_order = {
            target_id: position
            for position, target_id in enumerate([*dimensions, *hypotheses])
        }
        for target_kind, target_ids_known in (
            ("dimension", dimensions),
            ("hypothesis", hypotheses),
        ):
            for target_id in target_ids_known:
                touching = [
                    receipt
                    for receipt in index.values()
                    if _edge_touches(receipt, target_kind=target_kind, target_id=target_id)
                ]
                if not touching:
                    continue
                if has_new_dag_evidence(touching):
                    priority = 0
                elif target_id in preserved:
                    priority = 1
                else:
                    priority = 2
                candidates[target_kind].append(
                    (priority, registry_order[target_id], target_id)
                )
        for entries, cap, blocks, omitted in (
            (candidates["dimension"], max_dimensions, dimension_blocks, omitted_dimensions),
            (candidates["hypothesis"], max_hypotheses, hypothesis_blocks, omitted_hypotheses),
        ):
            entries.sort()
            for _, _, target_id in entries[:cap]:
                kind = "dimension" if target_id in dimensions else "hypothesis"
                blocks.append(build(kind, target_id))
            omitted.extend(target_id for _, _, target_id in entries[cap:])

    return {
        "schema_version": 2,
        "space_revision": space_revision(registry),
        "dag_revision": ledger.get("dag_revision", 0),
        "experience_dag_revision": experience_dag_revision,
        "direct_comparator_capability": direct_comparator_capability(ledger),
        "bounds": {
            "max_dimensions": max_dimensions,
            "max_hypotheses": max_hypotheses,
            "max_edges_per_target": max_edges_per_target,
            "max_runs_per_target": MAX_RUNS_PER_TARGET,
        },
        "dimension_targets": dimension_blocks,
        "hypothesis_targets": hypothesis_blocks,
        "omitted_target_ids": {
            "dimensions": omitted_dimensions,
            "hypotheses": omitted_hypotheses,
        },
    }
