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
``deprioritized -> pruned`` in a later one with changed target evidence;
``active -> pruned`` is forbidden.

:func:`derive_experience_transitions` deterministically turns the current
validated experience snapshot into the next decision receipts, and
:func:`append_experience_transitions` appends them, advancing only
``search_space_state.revision`` (never ``dag_revision``).  Recommendation
gates recompute comparator coverage and evaluation state from the cited
receipts rather than trusting belief prose; baseline and externally excluded
targets never receive a runtime transition.
"""

from __future__ import annotations

import re
from typing import Any

from semantic_evidence import (
    COVERAGE_KEYS,
    TERMINAL_STATUSES,
    _contradiction_depth_bar,
    comparator_coverage,
    edge_observation,
    hypothesis_carriers,
    mechanical_gain_direction,
    normalize_coverage,
    target_evaluation_state,
)
from semantic_space import dimension_map, hypothesis_map, selected_assignments


STATE_SCHEMA_VERSION = 1
DECISION_SCHEMA_VERSION = 4
# Decision schema 4 adds `carrier_contexts` (the independent negative/positive
# carrier contexts behind a carrier-rule demotion); schema 1–3 receipts are
# append-only history and stay valid — the field is simply absent there.
# Decision schema 3 carries the five-key `comparator_coverage` that split
# `direct_lightly_tuned_edges` out of the direct bucket (schema 2 had four
# keys, schema 1 three). Schema-1/2 receipts are append-only history and stay
# valid: their coverage normalizes forward on read with the newer
# direct-depth buckets at 0. New decisions are always written at the current
# version.
READABLE_DECISION_SCHEMA_VERSIONS = {1, 2, 3, 4}
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
    "carrier_contexts",
}
TARGET_FIELDS = {"kind", "dimension_id", "id"}
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
    coverage = normalize_coverage(decision.get("comparator_coverage"))
    if coverage is None:
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
        if (
            decision.get("schema_version") in {1, 2, 3}
            and "carrier_contexts" in missing
        ):
            missing.remove("carrier_contexts")
        if missing:
            errors.append(f"{item_where} is missing fields {missing}")
        if decision.get("schema_version") not in READABLE_DECISION_SCHEMA_VERSIONS:
            errors.append(
                f"{item_where}.schema_version must be "
                f"{sorted(READABLE_DECISION_SCHEMA_VERSIONS)}"
            )
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

    Only policy gates live here: guidance-excluded hypotheses and runtime-pruned
    hypotheses (directly or via a pruned dimension; a pruned dimension's
    non-baseline hypotheses already carry ``pruned`` effective status, so
    pinning it to its explicit baseline needs no separate check).  Runtime-
    deprioritized dimensions and hypotheses stay eligible but enter the
    selection helper's limited admission-budget lane.  Structural validity stays with
    :func:`semantic_space.validate_point`, so historical points remain valid
    after later pruning.
    """
    errors: list[str] = []
    if not isinstance(point, dict):
        return ["semantic_point must be an object"]
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
    return errors


# ---------- deterministic experience-to-pruning transitions ----------


def _effective_recommendation(
    belief: dict[str, Any],
    evaluation_state: str,
    coverage: dict[str, int],
    *,
    target_kind: str,
    mechanical_direction: str,
    carrier_demote: bool = False,
    carrier_prune: bool = False,
) -> str | None:
    """Gate the authored recommendation on mechanically recomputed evidence.

    Demotion passes either the strict comparator path or the carrier rule
    (repeated independent negative contexts, zero positive).  A ``pruned``
    recommendation that only meets the deprioritize gate is carried out as a
    deprioritization; a recommendation whose gates fail yields no transition
    at all (``None``), never a reopening.
    """
    recommended = belief.get("recommended_status")
    if recommended == "active":
        return "active"
    if recommended not in {"deprioritized", "pruned"}:
        return None
    strict = evaluation_state == "comparator_covered" and _contradiction_depth_bar(
        coverage
    )
    deprioritize_ok = (
        belief.get("assessment") == "unpromising"
        and belief.get("confidence") in {"med", "high"}
        and (strict or carrier_demote)
        and (
            target_kind != "hypothesis"
            or mechanical_direction == "negative"
            or carrier_demote
        )
        and _nonempty(belief.get("reopen_when"))
    )
    if not deprioritize_ok:
        return None
    if recommended == "deprioritized":
        return "deprioritized"
    prune_ok = belief.get("confidence") == "high" and (strict or carrier_prune)
    return "pruned" if prune_ok else "deprioritized"


def _normalized_beliefs(
    ledger: dict[str, Any],
    experience: dict[str, Any],
    generation: int,
    dag_revision: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Re-derive each target belief's coverage/state and gate its recommendation.

    The normalized belief carries ``experience_generation`` and
    ``experience_dag_revision`` from the snapshot so downstream comparisons
    never read the raw experience again.
    """
    beliefs: dict[tuple[str, str], dict[str, Any]] = {}
    for field, target_kind in (
        ("dimension_evidence", "dimension"),
        ("hypothesis_evidence", "hypothesis"),
    ):
        items = experience.get(field)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            target_id = item.get("target_id")
            if not _nonempty(target_id):
                continue
            edge_ids = [
                str(edge_id)
                for edge_id in item.get("evidence_edge_ids") or []
                if isinstance(edge_id, str)
            ]
            run_ids = [
                str(run_id)
                for run_id in item.get("evidence_run_ids") or []
                if isinstance(run_id, str)
            ]
            coverage = comparator_coverage(
                ledger, edge_ids, target_kind=target_kind, target_id=target_id
            )
            raw_observations = [
                edge_observation(ledger, edge_id) for edge_id in edge_ids
            ]
            observations = [
                {
                    key: observation.get(key)
                    for key in (
                        "edge_id",
                        "parent_status",
                        "child_status",
                        "parent_score",
                        "child_score",
                        "delta",
                    )
                }
                for observation in raw_observations
            ]
            evaluation_state = target_evaluation_state(
                ledger,
                target_kind=target_kind,
                target_id=target_id,
                evidence_run_ids=run_ids,
                evidence_edge_ids=edge_ids,
            )
            direction = mechanical_gain_direction(
                ledger,
                target_kind=target_kind,
                target_id=target_id,
                evidence_edge_ids=edge_ids,
            )
            carriers = (
                hypothesis_carriers(ledger, target_id=target_id, edge_ids=edge_ids)
                if target_kind == "hypothesis"
                else {
                    "negative": 0,
                    "positive": 0,
                    "negative_contexts": [],
                    "positive_contexts": [],
                }
            )
            beliefs[(target_kind, target_id)] = {
                "assessment": item.get("assessment"),
                "confidence": item.get("confidence"),
                "claim": item.get("claim"),
                "uncertainty": item.get("uncertainty"),
                "reopen_when": item.get("reopen_when"),
                "evidence_edge_ids": edge_ids,
                "evidence_observations": observations,
                # Internal-only full observations let the staged transition
                # gate distinguish a new/corrected inherited control from a
                # crash, a legacy final-score edge, or inner-tuning movement.
                "_direct_observations": [
                    basic
                    for basic, raw in zip(observations, raw_observations)
                    if raw.get("score_basis") == "paired_semantic_control"
                ],
                "_carriers": carriers,
                "comparator_coverage": coverage,
                "evaluation_state": evaluation_state,
                "recommended_status": _effective_recommendation(
                    item,
                    evaluation_state,
                    coverage,
                    target_kind=target_kind,
                    mechanical_direction=direction,
                    carrier_demote=(
                        carriers["negative"] >= 2 and carriers["positive"] == 0
                    ),
                    carrier_prune=(
                        carriers["negative"] >= 3 and carriers["positive"] == 0
                    ),
                ),
                "experience_generation": generation,
                "experience_dag_revision": dag_revision,
            }
    return beliefs


def _has_advancing_evidence(belief: dict[str, Any], last: dict | None) -> bool:
    """True only when a later snapshot changes evidence for this target."""
    if last is None:
        return False
    if belief["experience_generation"] <= last["experience_generation"]:
        return False
    if belief["experience_dag_revision"] <= last["experience_dag_revision"]:
        return False
    prior_edge_ids = last.get("evidence_edge_ids")
    if not isinstance(prior_edge_ids, list):
        return False
    prior_observations = last.get("evidence_observations")
    current_observations = belief.get("evidence_observations")
    if not isinstance(prior_observations, list) or not isinstance(
        current_observations, list
    ):
        return False
    direct_current = [
        item
        for item in belief.get("_direct_observations", [])
        if isinstance(item, dict)
        and {
            item.get("parent_status"),
            item.get("child_status"),
        }.issubset(TERMINAL_STATUSES)
    ]
    if any(item.get("edge_id") not in prior_edge_ids for item in direct_current):
        return True
    prior_by_edge = {
        item.get("edge_id"): item
        for item in prior_observations
        if isinstance(item, dict) and isinstance(item.get("edge_id"), str)
    }
    if any(
        isinstance(item, dict)
        and isinstance(item.get("edge_id"), str)
        and prior_by_edge.get(item["edge_id"]) != item
        for item in direct_current
    ):
        return True
    # Carrier path: a changed set of independent contexts is advancing
    # evidence — a new negative context advances demotion, a new positive
    # context advances reopening.  Legacy receipts without carrier_contexts
    # fail closed.
    last_carriers = last.get("carrier_contexts")
    if isinstance(last_carriers, dict):
        current_carriers = belief.get("_carriers") or {}
        return (
            list(current_carriers.get("negative_contexts") or [])
            != list(last_carriers.get("negative") or [])
            or list(current_carriers.get("positive_contexts") or [])
            != list(last_carriers.get("positive") or [])
        )
    return False


def _recommended_transition(current: str, belief: dict[str, Any], last: dict | None) -> str | None:
    recommendation = belief["recommended_status"]
    if current == "active" and recommendation in {"deprioritized", "pruned"}:
        return "deprioritized"
    if current == "deprioritized" and recommendation == "pruned":
        if _has_advancing_evidence(belief, last):
            return "pruned"
        return None
    if current in {"deprioritized", "pruned"} and recommendation == "active":
        if _has_advancing_evidence(belief, last):
            return "active"
    return None


def _dimension_contraction_ready(
    dimension: dict[str, Any],
    generation: int,
    runtime: dict[str, Any],
    guidance: dict[str, Any],
    beliefs: dict[tuple[str, str], dict[str, Any]],
    to_status: str,
) -> bool:
    """Scoped guard: dimension contraction never suppresses adjacent mechanisms.

    Every non-baseline hypothesis must be externally excluded, already
    at least as contracted as the requested dimension status, or covered by
    its own same-generation belief that independently satisfies the matching
    comparator-covered recommendation gate.
    """
    acceptable = (
        {"deprioritized", "pruned"}
        if to_status == "deprioritized"
        else {"pruned"}
    )
    for hypothesis in dimension.get("hypotheses", []):
        if not isinstance(hypothesis, dict):
            continue
        if hypothesis.get("kind") == "baseline":
            continue
        hypothesis_id = hypothesis.get("id")
        entry = guidance.get(hypothesis_id) if isinstance(guidance, dict) else None
        if isinstance(entry, dict) and entry.get("selection_status") == "excluded":
            continue
        if runtime["hypotheses"].get(hypothesis_id) in acceptable:
            continue
        belief = beliefs.get(("hypothesis", hypothesis_id))
        if belief is None:
            return False
        if belief["experience_generation"] != generation:
            return False
        if belief["recommended_status"] not in acceptable:
            return False
    return True


def _decision_receipt(
    ledger: dict[str, Any],
    revision: int,
    kind: str,
    dimension_id: str,
    target_id: str,
    from_status: str,
    to_status: str,
    belief: dict[str, Any],
    last: dict | None,
) -> dict[str, Any]:
    """Copy the belief and the current edge observations into an immutable receipt."""
    reopen_when = belief.get("reopen_when")
    if not _nonempty(reopen_when) and isinstance(last, dict):
        reopen_when = last.get("reopen_when")
    observations = [dict(item) for item in belief["evidence_observations"]]
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision_id": f"sdec-{revision:06d}",
        "revision": revision,
        "target": {"kind": kind, "dimension_id": dimension_id, "id": target_id},
        "from_status": from_status,
        "to_status": to_status,
        "experience_generation": belief["experience_generation"],
        "experience_dag_revision": belief["experience_dag_revision"],
        "assessment": belief["assessment"],
        "confidence": belief["confidence"],
        "claim": belief["claim"],
        "uncertainty": belief["uncertainty"],
        "reopen_when": reopen_when,
        "evidence_edge_ids": list(belief["evidence_edge_ids"]),
        "comparator_coverage": dict(belief["comparator_coverage"]),
        "evidence_observations": observations,
        "carrier_contexts": {
            "negative": list(belief["_carriers"]["negative_contexts"]),
            "positive": list(belief["_carriers"]["positive_contexts"]),
        },
    }


def derive_experience_transitions(
    registry: dict[str, Any], ledger: dict[str, Any]
) -> list[dict[str, Any]]:
    """Derive the next append-only decisions from the current experience.

    Pure: returns the receipts that would be appended without mutating the
    ledger.  Decisions follow registry dimension order, a dimension before
    its hypotheses; baseline hypotheses, ``baseline_only`` dimensions, and
    externally excluded hypotheses never receive a runtime transition.
    """
    experience = ledger.get("experience") if isinstance(ledger, dict) else None
    if not isinstance(experience, dict):
        return []
    generation = experience.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        return []
    dag_revision = experience.get("dag_revision")
    if not isinstance(dag_revision, int) or isinstance(dag_revision, bool) or dag_revision < 0:
        dag_revision = 0
    state = ledger.get("search_space_state")
    if not isinstance(state, dict):
        state = empty_search_space_state()
    decisions = [
        decision
        for decision in state.get("decisions") or []
        if isinstance(decision, dict)
    ]
    runtime = replay_search_space_state(registry, state)
    beliefs = _normalized_beliefs(ledger, experience, generation, dag_revision)
    if not beliefs:
        return []
    # Lazy import: background_contract already imports this module.
    from background_contract import derive_hypothesis_selection

    guidance = derive_hypothesis_selection(registry)
    last_decisions: dict[tuple[Any, Any], dict[str, Any]] = {}
    for decision in decisions:
        target = decision.get("target")
        if isinstance(target, dict):
            last_decisions[(target.get("kind"), target.get("id"))] = decision
    base_revision = state.get("revision")
    if not isinstance(base_revision, int) or isinstance(base_revision, bool):
        base_revision = len(decisions)

    transitions: list[dict[str, Any]] = []
    for dimension_id, dimension in dimension_map(registry).items():
        belief = beliefs.get(("dimension", dimension_id))
        if belief is not None and dimension.get("mode") != "baseline_only":
            current = runtime["dimensions"].get(dimension_id, "active")
            last = last_decisions.get(("dimension", dimension_id))
            to_status = _recommended_transition(current, belief, last)
            if to_status in {"deprioritized", "pruned"} and not _dimension_contraction_ready(
                dimension,
                generation,
                runtime,
                guidance,
                beliefs,
                to_status,
            ):
                to_status = None
            if to_status is not None:
                base_revision += 1
                transitions.append(
                    _decision_receipt(
                        ledger,
                        base_revision,
                        "dimension",
                        dimension_id,
                        dimension_id,
                        current,
                        to_status,
                        belief,
                        last,
                    )
                )
        for hypothesis in dimension.get("hypotheses", []):
            if not isinstance(hypothesis, dict):
                continue
            hypothesis_id = hypothesis.get("id")
            belief = beliefs.get(("hypothesis", hypothesis_id))
            if belief is None:
                continue
            if hypothesis.get("kind") == "baseline":
                continue  # the frozen space keeps an unconditional valid baseline
            entry = guidance.get(hypothesis_id)
            if isinstance(entry, dict) and entry.get("selection_status") == "excluded":
                continue  # runtime decisions never override external exclusion
            current = runtime["hypotheses"].get(hypothesis_id, "active")
            last = last_decisions.get(("hypothesis", hypothesis_id))
            to_status = _recommended_transition(current, belief, last)
            if to_status is not None:
                base_revision += 1
                transitions.append(
                    _decision_receipt(
                        ledger,
                        base_revision,
                        "hypothesis",
                        dimension_id,
                        hypothesis_id,
                        current,
                        to_status,
                        belief,
                        last,
                    )
                )
    return transitions


def append_experience_transitions(
    registry: dict[str, Any], ledger: dict[str, Any]
) -> list[dict[str, Any]]:
    """Append the derived decisions and advance only the overlay revision.

    An empty transition set is a successful no-op.  ``dag_revision`` is never
    touched: it tracks graph-visible score/status changes only.
    """
    transitions = derive_experience_transitions(registry, ledger)
    if not transitions:
        return []
    state = ledger.get("search_space_state")
    if not isinstance(state, dict):
        state = empty_search_space_state()
        ledger["search_space_state"] = state
    decisions = state.setdefault("decisions", [])
    decisions.extend(transitions)
    state["revision"] = len(decisions)
    return transitions
