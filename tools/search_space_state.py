#!/usr/bin/env python3
"""Append-only runtime search-space state overlay.

``ledger.search_space_state`` is the P2 selection-policy overlay over the
frozen schema-3 background registry: an append-only log of helper-validated
runtime decisions that deprioritize, prune, or reopen individual dimensions
and hypotheses.  The frozen registry itself never changes, and structural
point validity stays with :func:`semantic_space.validate_point`, so historical
observations remain valid.  The overlay controls only the eligibility of
FUTURE candidate points.

Reopening appends a new decision; pruning never deletes an id, a record, a
point, an observation, or a prior decision.  Automated pruning is two-stage:
``active -> deprioritized`` in one experience generation, then
``deprioritized -> pruned`` in a later one; ``active -> pruned`` is forbidden.
"""

from __future__ import annotations

import re
from typing import Any

from semantic_space import dimension_map, hypothesis_map, selected_assignments


STATE_SCHEMA_VERSION = 1
DECISION_SCHEMA_VERSION = 1
RUNTIME_STATUSES = {"active", "deprioritized", "pruned"}
LEGAL_TRANSITIONS = {
    ("active", "deprioritized"),
    ("deprioritized", "active"),
    ("deprioritized", "pruned"),
    ("pruned", "active"),
}
DECISION_ID_RE = re.compile(r"^sdec-[0-9]{6}$")

TARGET_KINDS = {"dimension", "hypothesis"}
GUIDANCE_STATUSES = {"active", "deprioritized", "excluded"}

STATE_FIELDS = {"schema_version", "revision", "decisions"}
DECISION_FIELDS = {
    "schema_version",
    "decision_id",
    "revision",
    "target",
    "from_status",
    "to_status",
    "experience_generation",
    "experience_dag_revision",
    "assessment",
    "confidence",
    "claim",
    "uncertainty",
    "reopen_when",
    "evidence_edge_ids",
    "comparator_coverage",
    "evidence_observations",
}
TARGET_FIELDS = {"kind", "dimension_id", "id"}
COVERAGE_KEYS = {"direct_noncrash_edges", "confounded_noncrash_edges", "crash_edges"}
OBSERVATION_FIELDS = {
    "edge_id",
    "parent_status",
    "child_status",
    "parent_score",
    "child_score",
    "delta",
}


def empty_search_space_state() -> dict[str, Any]:
    return {"schema_version": STATE_SCHEMA_VERSION, "revision": 0, "decisions": []}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _decision_revisions(decisions: list[dict[str, Any]]) -> list[int]:
    return [
        decision["revision"]
        for decision in decisions
        if isinstance(decision.get("revision"), int)
        and not isinstance(decision.get("revision"), bool)
    ]


def _final_statuses(
    decisions: list[dict[str, Any]], cutoff: int
) -> tuple[dict[str, str], dict[str, str]]:
    """Apply decisions up to ``cutoff`` blindly; validation is the gate."""
    dimensions: dict[str, str] = {}
    hypotheses: dict[str, str] = {}
    for decision in decisions:
        revision = decision.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool):
            continue
        if revision > cutoff:
            continue
        to_status = decision.get("to_status")
        if to_status not in RUNTIME_STATUSES:
            continue
        target = decision.get("target")
        if not isinstance(target, dict) or not isinstance(target.get("id"), str):
            continue
        if target.get("kind") == "dimension":
            dimensions[target["id"]] = to_status
        elif target.get("kind") == "hypothesis":
            hypotheses[target["id"]] = to_status
    return dimensions, hypotheses


def replay_search_space_state(
    registry: dict[str, Any], state: dict[str, Any], *, revision: int | None = None
) -> dict[str, Any]:
    """Replay dimensions and hypotheses separately, defaulting all to active.

    The returned maps cover every registry dimension and hypothesis, so a
    pruned id keeps its identity instead of disappearing.  Expects a state
    that passes :func:`validate_search_space_state`; decisions are applied in
    log order up to the inclusive ``revision`` cutoff (default: the state's
    own revision).
    """
    dimensions = {dimension_id: "active" for dimension_id in dimension_map(registry)}
    hypotheses = {hypothesis_id: "active" for hypothesis_id in hypothesis_map(registry)}
    decisions = [
        decision
        for decision in (state.get("decisions") or [])
        if isinstance(decision, dict)
    ]
    cutoff = revision
    if cutoff is None:
        cutoff = state.get("revision")
        if not isinstance(cutoff, int) or isinstance(cutoff, bool):
            cutoff = max(_decision_revisions(decisions), default=0)
    final_dimensions, final_hypotheses = _final_statuses(decisions, cutoff)
    dimensions.update(
        {key: value for key, value in final_dimensions.items() if key in dimensions}
    )
    hypotheses.update(
        {key: value for key, value in final_hypotheses.items() if key in hypotheses}
    )
    return {"dimensions": dimensions, "hypotheses": hypotheses}


def runtime_status_counts(state: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Compact deprioritized/pruned counts for the ledger brief view.

    Registry-free: only targets touched by decisions can be non-active, so the
    counts never need the full registry universe (or the decision log itself).
    """
    decisions = []
    if isinstance(state, dict) and isinstance(state.get("decisions"), list):
        decisions = [
            decision
            for decision in state["decisions"]
            if isinstance(decision, dict)
        ]
    cutoff = max(_decision_revisions(decisions), default=0)
    final_dimensions, final_hypotheses = _final_statuses(decisions, cutoff)

    def count(final: dict[str, str]) -> dict[str, int]:
        return {
            "deprioritized": sum(1 for status in final.values() if status == "deprioritized"),
            "pruned": sum(1 for status in final.values() if status == "pruned"),
        }

    return {"dimensions": count(final_dimensions), "hypotheses": count(final_hypotheses)}


def _validate_target(
    registry: dict[str, Any],
    target: Any,
    where: str,
) -> list[str]:
    """Check exact target fields, ownership, and baseline protection."""
    errors: list[str] = []
    if not isinstance(target, dict) or set(target) != TARGET_FIELDS:
        return [f"{where}.target must contain exactly {sorted(TARGET_FIELDS)}"]
    kind = target.get("kind")
    dimension_id = target.get("dimension_id")
    target_id = target.get("id")
    if kind == "dimension":
        dimensions = dimension_map(registry)
        if target_id != dimension_id:
            errors.append(
                f"{where}.target.id must equal target.dimension_id for a dimension decision"
            )
        elif dimension_id not in dimensions:
            errors.append(f"{where}.target.id is not a selected dimension: {dimension_id!r}")
        elif dimensions[dimension_id].get("mode") == "baseline_only":
            errors.append(
                f"{where} cannot target baseline_only dimension {dimension_id}; "
                "its baseline is the only legal choice"
            )
    elif kind == "hypothesis":
        hypotheses = hypothesis_map(registry)
        owner = hypotheses.get(target_id) if isinstance(target_id, str) else None
        if owner is None:
            errors.append(f"{where}.target.id is not a registry hypothesis: {target_id!r}")
        elif owner[0] != dimension_id:
            errors.append(
                f"{where}.target.dimension_id does not own hypothesis {target_id}: "
                f"{dimension_id!r}"
            )
        elif owner[1].get("kind") == "baseline":
            errors.append(
                f"{where} cannot target baseline hypothesis {target_id}; "
                "the frozen space keeps an unconditional valid baseline"
            )
    else:
        errors.append(f"{where}.target.kind must be one of {sorted(TARGET_KINDS)}")
    return errors


def _validate_belief_copy(decision: dict[str, Any], where: str) -> list[str]:
    """Shape-check the immutable belief/evidence copy inside one decision."""
    errors: list[str] = []
    for field in ("assessment", "confidence", "claim", "uncertainty", "reopen_when"):
        if not _nonempty(decision.get(field)):
            errors.append(f"{where}.{field} must be a non-empty string")
    for field in ("experience_generation", "experience_dag_revision"):
        if not _nonnegative_int(decision.get(field)):
            errors.append(f"{where}.{field} must be a non-negative integer")
    edge_ids = decision.get("evidence_edge_ids")
    if not isinstance(edge_ids, list) or any(not _nonempty(item) for item in edge_ids):
        errors.append(f"{where}.evidence_edge_ids must be a list of edge id strings")
    coverage = decision.get("comparator_coverage")
    if not isinstance(coverage, dict) or set(coverage) != COVERAGE_KEYS or any(
        not _nonnegative_int(coverage.get(key)) for key in COVERAGE_KEYS
    ):
        errors.append(
            f"{where}.comparator_coverage must hold exactly the non-negative "
            f"counts {sorted(COVERAGE_KEYS)}"
        )
    observations = decision.get("evidence_observations")
    if not isinstance(observations, list):
        errors.append(f"{where}.evidence_observations must be a list")
        return errors
    for index, observation in enumerate(observations):
        item_where = f"{where}.evidence_observations[{index}]"
        if not isinstance(observation, dict) or set(observation) != OBSERVATION_FIELDS:
            errors.append(f"{item_where} must contain exactly {sorted(OBSERVATION_FIELDS)}")
            continue
        if not _nonempty(observation.get("edge_id")):
            errors.append(f"{item_where}.edge_id must be a non-empty string")
        for field in ("parent_status", "child_status"):
            if not _nonempty(observation.get(field)):
                errors.append(f"{item_where}.{field} must be a non-empty string")
        for field in ("parent_score", "child_score", "delta"):
            value = observation.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                errors.append(f"{item_where}.{field} must be a number or null")
    return errors


def validate_search_space_state(registry: dict[str, Any], ledger: dict[str, Any]) -> list[str]:
    """Require and validate the append-only overlay whenever records exist.

    Decision ids and revisions are helper-owned, derived, and contiguous;
    ``from_status`` must equal the replayed live status, transitions must be
    legal, and baselines / ``baseline_only`` dimensions are protected.
    """
    where = "ledger.search_space_state"
    records = ledger.get("records") if isinstance(ledger, dict) else None
    state = ledger.get("search_space_state") if isinstance(ledger, dict) else None
    if state is None:
        if isinstance(records, list) and records:
            return [f"{where} is required once records exist"]
        return []
    if not isinstance(state, dict):
        return [f"{where} must be an object"]

    errors: list[str] = []
    unknown = sorted(set(state) - STATE_FIELDS)
    if unknown:
        errors.append(f"{where} has unknown fields {unknown}")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        errors.append(f"{where}.schema_version must be {STATE_SCHEMA_VERSION}")
    revision = state.get("revision")
    if not _nonnegative_int(revision):
        errors.append(f"{where}.revision must be a non-negative integer")
        revision = None
    decisions = state.get("decisions")
    if not isinstance(decisions, list):
        errors.append(f"{where}.decisions must be a list")
        decisions = []
    if revision is not None and revision != len(decisions):
        errors.append(
            f"{where}.revision must equal the number of append-only decisions"
        )

    live: dict[str, dict[str, str]] = {"dimension": {}, "hypothesis": {}}
    for index, decision in enumerate(decisions):
        item_where = f"{where}.decisions[{index}]"
        if not isinstance(decision, dict):
            errors.append(f"{item_where} must be an object")
            continue
        unknown = sorted(set(decision) - DECISION_FIELDS)
        if unknown:
            errors.append(f"{item_where} has unknown fields {unknown}")
        missing = sorted(DECISION_FIELDS - set(decision))
        if missing:
            errors.append(f"{item_where} is missing fields {missing}")
        if decision.get("schema_version") != DECISION_SCHEMA_VERSION:
            errors.append(f"{item_where}.schema_version must be {DECISION_SCHEMA_VERSION}")
        expected_revision = index + 1
        if decision.get("revision") != expected_revision:
            errors.append(
                f"{item_where}.revision must be the sequential {expected_revision}"
            )
        expected_id = f"sdec-{expected_revision:06d}"
        if decision.get("decision_id") != expected_id:
            errors.append(f"{item_where}.decision_id must be the derived {expected_id}")

        target = decision.get("target")
        target_errors = _validate_target(registry, target, item_where)
        errors.extend(target_errors)
        tracked = not target_errors and isinstance(target, dict)
        kind = target.get("kind") if isinstance(target, dict) else None
        target_id = target.get("id") if isinstance(target, dict) else None
        current = live[kind].get(target_id, "active") if tracked else "active"

        from_status = decision.get("from_status")
        to_status = decision.get("to_status")
        if from_status not in RUNTIME_STATUSES:
            errors.append(
                f"{item_where}.from_status must be one of {sorted(RUNTIME_STATUSES)}"
            )
        elif tracked and from_status != current:
            errors.append(
                f"{item_where}.from_status must equal the replayed status "
                f"{current!r}"
            )
        if to_status not in RUNTIME_STATUSES:
            errors.append(
                f"{item_where}.to_status must be one of {sorted(RUNTIME_STATUSES)}"
            )
        if (
            from_status in RUNTIME_STATUSES
            and to_status in RUNTIME_STATUSES
            and (from_status, to_status) not in LEGAL_TRANSITIONS
        ):
            errors.append(
                f"{item_where} transition {from_status} -> {to_status} is not legal; "
                "automated pruning is two-stage and reopening appends a new decision"
            )
        if (
            tracked
            and from_status == current
            and (from_status, to_status) in LEGAL_TRANSITIONS
        ):
            live[kind][target_id] = to_status
        errors.extend(_validate_belief_copy(decision, item_where))
    return errors


def compose_effective_selection(
    registry: dict[str, Any], guidance: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any]:
    """Compose external guidance and runtime state per hypothesis.

    Returns every component status rather than a single lossy label; the
    effective precedence is ``guidance excluded > runtime hypothesis/dimension
    pruned > guidance or runtime hypothesis/dimension deprioritized > active``.
    A dimension's runtime status pins its non-baseline hypotheses; the
    explicit baseline stays eligible so a pruned dimension remains selectable
    at its baseline.  ``excluded`` and ``pruned`` stay distinguishable.
    """
    dimensions = dimension_map(registry)
    runtime_dimensions = runtime.get("dimensions", {}) if isinstance(runtime, dict) else {}
    runtime_hypotheses = runtime.get("hypotheses", {}) if isinstance(runtime, dict) else {}
    result: dict[str, dict[str, Any]] = {}
    for hypothesis_id, (dimension_id, _hypothesis) in hypothesis_map(registry).items():
        entry = guidance.get(hypothesis_id, {}) if isinstance(guidance, dict) else {}
        guidance_status = entry.get("selection_status", "active")
        if guidance_status not in GUIDANCE_STATUSES:
            guidance_status = "active"
        dimension_runtime = runtime_dimensions.get(dimension_id, "active")
        if dimension_runtime not in RUNTIME_STATUSES:
            dimension_runtime = "active"
        hypothesis_runtime = runtime_hypotheses.get(hypothesis_id, "active")
        if hypothesis_runtime not in RUNTIME_STATUSES:
            hypothesis_runtime = "active"
        is_baseline = dimensions[dimension_id].get("baseline_hypothesis_id") == hypothesis_id
        binding_dimension = "active" if is_baseline else dimension_runtime
        if guidance_status == "excluded":
            effective = "excluded"
        elif hypothesis_runtime == "pruned" or binding_dimension == "pruned":
            effective = "pruned"
        elif "deprioritized" in (guidance_status, hypothesis_runtime, binding_dimension):
            effective = "deprioritized"
        else:
            effective = "active"
        result[hypothesis_id] = {
            "guidance_status": guidance_status,
            "dimension_runtime_status": dimension_runtime,
            "hypothesis_runtime_status": hypothesis_runtime,
            "effective_status": effective,
            "binding_guidance": list(entry.get("binding_guidance") or []),
            "matched_guidance": list(entry.get("matched_guidance") or []),
        }
    return result


def validate_point_eligibility(
    point: dict[str, Any], registry: dict[str, Any], effective: dict[str, Any]
) -> list[str]:
    """Check selection-time policy eligibility of one candidate point.

    Only policy gates live here: guidance-excluded hypotheses, runtime-pruned
    hypotheses (directly or via a pruned dimension), and pinning a
    deprioritized dimension to its explicit baseline.  Structural validity
    stays with :func:`semantic_space.validate_point`, so historical points
    remain valid after later pruning.
    """
    errors: list[str] = []
    if not isinstance(point, dict):
        return ["semantic_point must be an object"]
    dimensions = dimension_map(registry)
    for dimension_id, hypothesis_id in selected_assignments(point).items():
        entry = effective.get(hypothesis_id) if isinstance(effective, dict) else None
        if not isinstance(entry, dict):
            continue  # unknown hypotheses are a structural concern, not eligibility
        status = entry.get("effective_status")
        if status == "excluded":
            errors.append(
                f"semantic_point selects guidance-excluded hypothesis {hypothesis_id}"
            )
        elif status == "pruned":
            errors.append(
                f"semantic_point selects runtime-pruned hypothesis {hypothesis_id}; "
                "pruning bars new proposals but never erases the id"
            )
        dimension = dimensions.get(dimension_id, {})
        if (
            entry.get("dimension_runtime_status") == "deprioritized"
            and hypothesis_id != dimension.get("baseline_hypothesis_id")
        ):
            errors.append(
                f"semantic_point must pin deprioritized dimension {dimension_id} "
                f"to its baseline {dimension.get('baseline_hypothesis_id')}"
            )
    return errors
