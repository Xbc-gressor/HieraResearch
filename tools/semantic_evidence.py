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

MAX_DIMENSION_TARGETS = 16
MAX_HYPOTHESIS_TARGETS = 32
MIN_EDGES_PER_TARGET = 2
MAX_EDGES_PER_TARGET = 5
MAX_RUNS_PER_TARGET = 5
MIN_EXPERIENCE_ADJUSTMENT = 0.01

COVERAGE_KEYS = ("direct_noncrash_edges", "confounded_noncrash_edges", "crash_edges")


class SemanticEvidenceError(ValueError):
    """A malformed semantic edge receipt or evidence-view request."""


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


def edge_observation(ledger: dict[str, Any], edge_id: str) -> dict[str, Any]:
    """Report one persisted edge with statuses and a terminal non-crash delta."""
    receipt = edge_index(ledger).get(str(edge_id))
    if receipt is None:
        return {}
    records = _records_by_id(ledger)
    parent = records.get(str(receipt.get("parent_run_id")), {})
    child = records.get(str(receipt.get("child_run_id")), {})
    parent_score = _terminal_score(parent)
    child_score = _terminal_score(child)
    delta = None
    if parent_score is not None and child_score is not None:
        delta = round(child_score - parent_score, 4)
    return {
        "edge_id": receipt["edge_id"],
        "change_class": receipt.get("change_class"),
        "parent_run_id": receipt.get("parent_run_id"),
        "child_run_id": receipt.get("child_run_id"),
        "parent_status": parent.get("status"),
        "child_status": child.get("status"),
        "parent_score": parent_score,
        "child_score": child_score,
        "delta": delta,
    }


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
    if receipt.get("change_class") == "single_dimension":
        return "direct_noncrash_edges"
    return "confounded_noncrash_edges"


def comparator_coverage(
    ledger: dict[str, Any], edge_ids: list[str], *, target_kind: str, target_id: str
) -> dict[str, int]:
    """Classify only cited receipts that touch the requested target."""
    coverage = {key: 0 for key in COVERAGE_KEYS}
    index = edge_index(ledger)
    for edge_id in edge_ids:
        receipt = index.get(str(edge_id))
        if receipt is None or not _edge_touches(
            receipt, target_kind=target_kind, target_id=target_id
        ):
            continue
        category = _coverage_category(ledger, receipt)
        if category is not None:
            coverage[category] += 1
    return coverage


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
    if coverage["direct_noncrash_edges"] >= 2:
        return "comparator_covered"
    noncrash_observation = (
        coverage["direct_noncrash_edges"] + coverage["confounded_noncrash_edges"] > 0
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
    direct = pools["direct_noncrash_edges"]
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
        "schema_version": 1,
        "space_revision": space_revision(registry),
        "dag_revision": ledger.get("dag_revision", 0),
        "experience_dag_revision": experience_dag_revision,
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
