#!/usr/bin/env python3
"""P1 semantic-point proposal and replaceable acquisition policies.

``got_select`` continues to choose the structural graph action and parents.
This module independently turns that assignment into a bounded set of valid
semantic points, then selects one with one of three policies:

* ``coverage``: deterministic exploration without any model score;
* ``gain``: predicted gain with explicit cost and a small coverage tie-break;
* ``gain_uncertainty``: predicted gain plus a separate uncertainty bonus,
  explicit cost, and coverage.

Predictions are rubric inputs, not calibrated Bayesian posteriors.  Every
selection writes the components separately in a policy receipt; neither the
registry nor observation history is mutated.

Formally (see ``docs/search-space.md``): policies rank fibers (equivalence
classes of implementations), and every observed score is an upper bound on the
fiber objective ``F(s)``, never its value.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from background_contract import (
    ContractError,
    derive_hypothesis_selection,
    load_registry,
    validate_background_markdown,
    validate_registry,
)
from semantic_space import (
    SemanticSpaceError,
    complete_point,
    coverage_from_records,
    digest,
    dimension_map,
    incoming_activation_relations,
    point_diff,
    point_id,
    selected_assignments,
    space_receipt,
    validate_point,
)


PROPOSAL_SCHEMA_VERSION = 1
PREDICTION_SCHEMA_VERSION = 1
POLICY_RECEIPT_SCHEMA_VERSION = 1
POLICIES = {"coverage", "gain", "gain_uncertainty"}
DEFAULT_POLICY_CONFIG = {
    "coverage_weight": 0.10,
    "cost_weight": 0.20,
    "uncertainty_weight": 0.50,
}
MAX_PROPOSALS = 128
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def _write_object(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _parent_records(ledger: dict[str, Any], parents: list[str]) -> list[dict[str, Any]]:
    by_id = {
        str(record.get("run_id")): record
        for record in ledger.get("records", [])
        if isinstance(record, dict)
    }
    missing = [parent for parent in parents if parent not in by_id]
    if missing:
        raise ContractError(f"semantic action references missing parents {missing}")
    return [by_id[parent] for parent in parents]


def _validate_action(op: str, parents: list[str]) -> None:
    expected = {"fresh": 0, "improve": 1, "crossover": 2}.get(op)
    if expected is None:
        raise ContractError("op must be fresh, improve, or crossover")
    if len(parents) != expected or len(parents) != len(set(parents)):
        raise ContractError(f"{op} requires {expected} distinct numeric parents")
    if any(not parent.isdigit() for parent in parents):
        raise ContractError("semantic action parents must be numeric run ids")


def _eligible_hypotheses(registry: dict[str, Any]) -> dict[str, list[str]]:
    selection = derive_hypothesis_selection(registry)
    result: dict[str, list[str]] = {}
    for dimension in registry.get("dimensions", []):
        if not isinstance(dimension, dict):
            continue
        choices = [
            hypothesis.get("id")
            for hypothesis in dimension.get("hypotheses", [])
            if isinstance(hypothesis, dict)
            and selection.get(hypothesis.get("id"), {}).get("selection_status") != "excluded"
        ]
        result[dimension["id"]] = [str(item) for item in choices]
    return result


def _activation_overrides(
    registry: dict[str, Any], dimension_id: str, eligible: dict[str, list[str]]
) -> list[dict[str, str]]:
    incoming = incoming_activation_relations(registry, dimension_id)
    if not incoming:
        return [{}]
    options: list[dict[str, str]] = []
    for relation in incoming:
        when = relation.get("when", {})
        source_dimension = when.get("dimension_id")
        for hypothesis_id in when.get("hypothesis_ids", []):
            if hypothesis_id in eligible.get(source_dimension, []):
                options.append({source_dimension: hypothesis_id})
    return options


def _add_point(
    registry: dict[str, Any],
    values: dict[str, dict[str, Any]],
    point: dict[str, Any] | None,
) -> None:
    if point is None or validate_point(point, registry):
        return
    values.setdefault(point["point_id"], point)


def _round_robin(groups: list[list[dict[str, str]]]) -> Iterable[dict[str, str]]:
    """Yield one intervention per dimension before taking its next alternative."""
    width = max((len(group) for group in groups), default=0)
    for offset in range(width):
        for group in groups:
            if offset < len(group):
                yield group[offset]


def _fresh_points(
    registry: dict[str, Any], eligible: dict[str, list[str]], max_points: int
) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    _add_point(registry, values, complete_point(registry))
    if len(values) >= max_points:
        return list(values.values())
    dimensions = dimension_map(registry)
    intervention_groups: list[list[dict[str, str]]] = []
    for dimension_id, dimension in dimensions.items():
        group: list[dict[str, str]] = []
        baseline = dimension.get("baseline_hypothesis_id")
        for hypothesis_id in eligible.get(dimension_id, []):
            if hypothesis_id == baseline:
                continue
            for activation in _activation_overrides(registry, dimension_id, eligible):
                override = dict(activation)
                override[dimension_id] = hypothesis_id
                group.append(override)
        intervention_groups.append(group[:max_points])
    interventions = list(_round_robin(intervention_groups))
    for override in interventions:
        _add_point(registry, values, complete_point(registry, override))
        if len(values) >= max_points:
            return list(values.values())
    # Pairwise points make interactions searchable while the deterministic cap
    # prevents a full Cartesian explosion.
    for left, right in itertools.combinations(interventions, 2):
        merged = dict(left)
        conflict = any(key in merged and merged[key] != value for key, value in right.items())
        if conflict:
            continue
        merged.update(right)
        _add_point(registry, values, complete_point(registry, merged))
        if len(values) >= max_points:
            break
    return list(values.values())


def _improve_points(
    registry: dict[str, Any],
    parent: dict[str, Any],
    eligible: dict[str, list[str]],
    max_points: int,
) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    parent_point = parent.get("semantic_point")
    _add_point(registry, values, parent_point)
    if len(values) >= max_points:
        return list(values.values())
    base = selected_assignments(parent_point)
    intervention_groups: list[list[dict[str, str]]] = []
    for dimension_id, choices in eligible.items():
        group: list[dict[str, str]] = []
        for hypothesis_id in choices:
            if base.get(dimension_id) == hypothesis_id:
                continue
            activations = _activation_overrides(registry, dimension_id, eligible)
            for activation in activations:
                overrides = dict(base)
                overrides.update(activation)
                overrides[dimension_id] = hypothesis_id
                group.append(overrides)
        intervention_groups.append(group[:max_points])
    for overrides in _round_robin(intervention_groups):
        _add_point(registry, values, complete_point(registry, overrides))
        if len(values) >= max_points:
            return list(values.values())
    return list(values.values())


def _crossover_points(
    registry: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    eligible: dict[str, list[str]],
    max_points: int,
) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    left_point = left.get("semantic_point")
    right_point = right.get("semantic_point")
    left_selected = selected_assignments(left_point)
    right_selected = selected_assignments(right_point)
    _add_point(registry, values, left_point)
    _add_point(registry, values, right_point)
    dimension_ids = list(dimension_map(registry))
    differing = [
        dimension_id
        for dimension_id in dimension_ids
        if left_selected.get(dimension_id) != right_selected.get(dimension_id)
    ]
    # Enumerate true recombinations first.  For large diffs, deterministic
    # single-dimension swaps plus alternating masks stay bounded.
    masks: list[tuple[int, ...]] = []
    if len(differing) <= 10:
        masks.extend(itertools.product((0, 1), repeat=len(differing)))
    else:
        masks.append(tuple(index % 2 for index in range(len(differing))))
        masks.append(tuple((index + 1) % 2 for index in range(len(differing))))
        for index in range(len(differing)):
            mask = [0] * len(differing)
            mask[index] = 1
            masks.append(tuple(mask))
    for mask in masks:
        overrides = dict(left_selected)
        for index, dimension_id in enumerate(differing):
            if mask[index] and dimension_id in right_selected:
                overrides[dimension_id] = right_selected[dimension_id]
            elif mask[index] and dimension_id not in right_selected:
                overrides.pop(dimension_id, None)
        _add_point(registry, values, complete_point(registry, overrides))
        if len(values) >= max_points:
            return list(values.values())
    # If parents occupy the same or nearby point, implementation-level
    # recombination is still valid; add one-hop semantic alternatives as useful
    # neighbors without claiming that the point determines the implementation.
    if len(values) < max_points:
        for point in _improve_points(registry, left, eligible, max_points):
            _add_point(registry, values, point)
            if len(values) >= max_points:
                break
    return list(values.values())


def _coverage_score(
    point: dict[str, Any], coverage: dict[str, Any], registry: dict[str, Any]
) -> float:
    counts = {
        item["hypothesis_id"]: item["count"]
        for dimension in coverage["dimensions"]
        for item in dimension["hypotheses"]
    }
    selected = list(selected_assignments(point).values())
    hypothesis_term = (
        sum(1.0 / (1.0 + counts.get(hypothesis_id, 0)) for hypothesis_id in selected)
        / max(1, len(selected))
    )
    point_term = 1.0 / (1.0 + coverage["point_counts"].get(point["point_id"], 0))
    return round(0.5 * hypothesis_term + 0.5 * point_term, 8)


def build_proposal_set(
    registry: dict[str, Any],
    ledger: dict[str, Any],
    *,
    op: str,
    parents: list[str],
    max_points: int = 128,
) -> dict[str, Any]:
    if (
        not isinstance(max_points, int)
        or isinstance(max_points, bool)
        or not 1 <= max_points <= MAX_PROPOSALS
    ):
        raise ContractError(f"max_points must be an integer in [1, {MAX_PROPOSALS}]")
    _validate_action(op, parents)
    parent_records = _parent_records(ledger, parents)
    eligible = _eligible_hypotheses(registry)
    if any(not choices for choices in eligible.values()):
        empty = [dimension_id for dimension_id, choices in eligible.items() if not choices]
        raise ContractError(f"selected dimensions have no eligible hypotheses: {empty}")
    if op == "fresh":
        points = _fresh_points(registry, eligible, max_points)
    elif op == "improve":
        points = _improve_points(registry, parent_records[0], eligible, max_points)
    else:
        points = _crossover_points(
            registry, parent_records[0], parent_records[1], eligible, max_points
        )
    if not points:
        raise ContractError(f"no valid semantic points can satisfy action {op}")
    coverage = coverage_from_records(registry, ledger.get("records", []))
    selection = derive_hypothesis_selection(registry)
    proposals: list[dict[str, Any]] = []
    for point in points:
        selected = selected_assignments(point)
        if any(
            selection.get(hypothesis_id, {}).get("selection_status") == "excluded"
            for hypothesis_id in selected.values()
        ):
            continue
        parent_diffs = [
            {
                "parent_run_id": parent,
                "changes": point_diff(record["semantic_point"], point),
            }
            for parent, record in zip(parents, parent_records)
        ]
        proposals.append(
            {
                "point_id": point["point_id"],
                "point": point,
                "coverage": _coverage_score(point, coverage, registry),
                "parent_diffs": parent_diffs,
                "deprioritized_hypotheses": sorted(
                    hypothesis_id
                    for hypothesis_id in selected.values()
                    if selection.get(hypothesis_id, {}).get("selection_status")
                    == "deprioritized"
                ),
            }
        )
    proposals.sort(
        key=lambda item: (
            -item["coverage"],
            bool(item["deprioritized_hypotheses"]),
            item["point_id"],
        )
    )
    value = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "space": space_receipt(registry),
        "action": {"op": op, "parents": parents},
        "coverage_snapshot": {
            "n_valid_records": coverage["n_valid_records"],
            "n_unique_points": coverage["n_unique_points"],
        },
        "proposals": proposals[:max_points],
    }
    value["proposal_set_revision"] = digest(value)
    return value


def validate_proposal_set(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["proposal set must be an object"]
    errors: list[str] = []
    allowed_top = {
        "schema_version",
        "space",
        "action",
        "coverage_snapshot",
        "proposals",
        "proposal_set_revision",
    }
    unknown_top = sorted(set(value) - allowed_top)
    if unknown_top:
        errors.append(f"proposal set has unknown fields {unknown_top}")
    if value.get("schema_version") != PROPOSAL_SCHEMA_VERSION:
        errors.append(f"proposal set schema_version must be {PROPOSAL_SCHEMA_VERSION}")
    payload = dict(value)
    revision = payload.pop("proposal_set_revision", None)
    if not isinstance(revision, str) or DIGEST_RE.fullmatch(revision) is None:
        errors.append("proposal_set_revision must be a sha256 digest")
    elif revision != digest(payload):
        errors.append("proposal_set_revision does not match proposal contents")

    space = value.get("space")
    if not isinstance(space, dict):
        errors.append("proposal set space must be an object")
        space = {}
    else:
        if set(space) != {"space_id", "space_revision", "catalog", "dimension_ids"}:
            errors.append("proposal set space must be an exact frozen-space receipt")
        if not isinstance(space.get("space_id"), str) or not space.get("space_id"):
            errors.append("proposal set space.space_id must be non-empty")
        if (
            not isinstance(space.get("space_revision"), str)
            or DIGEST_RE.fullmatch(space["space_revision"]) is None
        ):
            errors.append("proposal set space.space_revision must be a sha256 digest")
        catalog = space.get("catalog")
        if (
            not isinstance(catalog, dict)
            or catalog.get("id") != "semantic-dimensions/v1"
            or not isinstance(catalog.get("revision"), str)
            or DIGEST_RE.fullmatch(catalog["revision"]) is None
            or set(catalog) != {"id", "revision"}
        ):
            errors.append("proposal set space.catalog must be a semantic-dimensions/v1 receipt")
        dimension_ids = space.get("dimension_ids")
        if (
            not isinstance(dimension_ids, list)
            or any(not isinstance(item, str) for item in dimension_ids)
            or len(dimension_ids) != len(set(dimension_ids))
        ):
            errors.append("proposal set space.dimension_ids must be a unique string list")

    action = value.get("action")
    if not isinstance(action, dict) or set(action) != {"op", "parents"}:
        errors.append("proposal set action must contain only op and parents")
        action = {}
    op = action.get("op")
    parents = action.get("parents")
    if not isinstance(parents, list) or any(not isinstance(item, str) for item in parents):
        errors.append("proposal set action.parents must be a string list")
        parents = []
    else:
        try:
            _validate_action(str(op), parents)
        except ContractError as exc:
            errors.append(str(exc))

    snapshot = value.get("coverage_snapshot")
    if not isinstance(snapshot, dict) or set(snapshot) != {"n_valid_records", "n_unique_points"}:
        errors.append("coverage_snapshot must contain valid-record and unique-point counts")
    else:
        for key in ("n_valid_records", "n_unique_points"):
            count = snapshot.get(key)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                errors.append(f"coverage_snapshot.{key} must be a non-negative integer")

    proposals = value.get("proposals")
    if not isinstance(proposals, list) or not proposals:
        errors.append("proposal set must contain at least one proposal")
    else:
        if len(proposals) > MAX_PROPOSALS:
            errors.append(f"proposal set may contain at most {MAX_PROPOSALS} proposals")
        ids = [item.get("point_id") for item in proposals if isinstance(item, dict)]
        if (
            len(ids) != len(proposals)
            or any(not isinstance(item, str) for item in ids)
            or len(ids) != len(set(ids))
        ):
            errors.append("proposal point ids must be present and unique")
        for index, proposal in enumerate(proposals):
            where = f"proposals[{index}]"
            if not isinstance(proposal, dict):
                errors.append(f"{where} must be an object")
                continue
            if set(proposal) != {
                "point_id",
                "point",
                "coverage",
                "parent_diffs",
                "deprioritized_hypotheses",
            }:
                errors.append(f"{where} has an unexpected shape")
            proposal_point = proposal.get("point")
            proposal_id = proposal.get("point_id")
            if not isinstance(proposal_point, dict):
                errors.append(f"{where}.point must be an object")
            else:
                if proposal_point.get("point_id") != proposal_id or point_id(proposal_point) != proposal_id:
                    errors.append(f"{where}.point_id does not match the point contents")
                if proposal_point.get("space_id") != space.get("space_id"):
                    errors.append(f"{where}.point space_id does not match the proposal set")
                if proposal_point.get("space_revision") != space.get("space_revision"):
                    errors.append(f"{where}.point space_revision does not match the proposal set")
                if not isinstance(proposal_point.get("assignments"), list):
                    errors.append(f"{where}.point.assignments must be a list")
            coverage = proposal.get("coverage")
            if (
                not isinstance(coverage, (int, float))
                or isinstance(coverage, bool)
                or not math.isfinite(float(coverage))
                or not 0.0 <= float(coverage) <= 1.0
            ):
                errors.append(f"{where}.coverage must be a finite number in [0, 1]")
            parent_diffs = proposal.get("parent_diffs")
            if not isinstance(parent_diffs, list) or len(parent_diffs) != len(parents):
                errors.append(f"{where}.parent_diffs must align with action parents")
            else:
                actual_parents = [
                    item.get("parent_run_id") if isinstance(item, dict) else None
                    for item in parent_diffs
                ]
                if actual_parents != parents:
                    errors.append(f"{where}.parent_diffs must preserve parent order")
                if any(
                    not isinstance(item, dict)
                    or set(item) != {"parent_run_id", "changes"}
                    or not isinstance(item.get("changes"), list)
                    for item in parent_diffs
                ):
                    errors.append(f"{where}.parent_diffs entries have an unexpected shape")
            deprioritized = proposal.get("deprioritized_hypotheses")
            if (
                not isinstance(deprioritized, list)
                or any(not isinstance(item, str) for item in deprioritized)
                or len(deprioritized) != len(set(deprioritized))
            ):
                errors.append(f"{where}.deprioritized_hypotheses must be a unique string list")
        if all(
            isinstance(item, dict)
            and isinstance(item.get("coverage"), (int, float))
            and not isinstance(item.get("coverage"), bool)
            and math.isfinite(float(item["coverage"]))
            and isinstance(item.get("point_id"), str)
            for item in proposals
        ):
            expected_ids = [
                item["point_id"]
                for item in sorted(
                    proposals,
                    key=lambda item: (
                        -float(item["coverage"]),
                        bool(item["deprioritized_hypotheses"]),
                        item["point_id"],
                    ),
                )
            ]
            if ids != expected_ids:
                errors.append("proposals must be ordered by coverage then point id")
    return errors


def _prediction_map(
    value: dict[str, Any] | None, proposal_set: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if value is None:
        return {}, ["gain policies require a predictions JSON object"]
    errors: list[str] = []
    if value.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        errors.append(f"predictions.schema_version must be {PREDICTION_SCHEMA_VERSION}")
    if value.get("proposal_set_revision") != proposal_set.get("proposal_set_revision"):
        errors.append("predictions.proposal_set_revision does not match proposals")
    predictions = value.get("predictions")
    if not isinstance(predictions, list):
        return {}, errors + ["predictions.predictions must be a list"]
    result: dict[str, dict[str, Any]] = {}
    proposal_ids = {item["point_id"] for item in proposal_set["proposals"]}
    for index, prediction in enumerate(predictions):
        where = f"predictions[{index}]"
        if not isinstance(prediction, dict):
            errors.append(f"{where} must be an object")
            continue
        point_id_value = prediction.get("point_id")
        if point_id_value not in proposal_ids:
            errors.append(f"{where}.point_id is not in the proposal set")
            continue
        if point_id_value in result:
            errors.append(f"duplicate prediction for {point_id_value}")
            continue
        for field in ("predicted_gain", "uncertainty", "cost"):
            score = prediction.get(field)
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                errors.append(f"{where}.{field} must be a number in [0, 1]")
        evidence = prediction.get("evidence")
        if (
            not isinstance(evidence, list)
            or not 1 <= len(evidence) <= 5
            or any(
                not isinstance(item, str)
                or not item.strip()
                or len(item) > 240
                for item in evidence
            )
        ):
            errors.append(
                f"{where}.evidence must contain 1–5 non-empty strings of at most 240 characters"
            )
        unknown = sorted(
            set(prediction) - {"point_id", "predicted_gain", "uncertainty", "cost", "evidence"}
        )
        if unknown:
            errors.append(f"{where} has unknown fields {unknown}")
        result[point_id_value] = prediction
    missing = sorted(proposal_ids - set(result))
    if missing:
        errors.append(f"predictions are missing proposal ids {missing}")
    return result, errors


def select_proposal(
    proposal_set: dict[str, Any],
    *,
    policy: str,
    predictions: dict[str, Any] | None = None,
    config: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors = validate_proposal_set(proposal_set)
    if errors:
        raise ContractError("invalid proposal set: " + "; ".join(errors))
    if policy not in POLICIES:
        raise ContractError(f"policy must be one of {sorted(POLICIES)}")
    cfg = dict(DEFAULT_POLICY_CONFIG)
    if config:
        unknown = sorted(set(config) - set(cfg))
        if unknown:
            raise ContractError(f"unknown semantic policy config keys {unknown}")
        for key, value in config.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ContractError(f"semantic policy config {key} must be a non-negative number")
            cfg[key] = float(value)

    prediction_by_id: dict[str, dict[str, Any]] = {}
    if policy != "coverage":
        prediction_by_id, prediction_errors = _prediction_map(predictions, proposal_set)
        if prediction_errors:
            raise ContractError("invalid policy predictions: " + "; ".join(prediction_errors))

    ranked: list[tuple[float, int, str, dict[str, Any], dict[str, Any]]] = []
    for proposal in proposal_set["proposals"]:
        point_id_value = proposal["point_id"]
        prediction = prediction_by_id.get(point_id_value)
        coverage = float(proposal["coverage"])
        predicted_gain = None if prediction is None else float(prediction["predicted_gain"])
        uncertainty = None if prediction is None else float(prediction["uncertainty"])
        cost = None if prediction is None else float(prediction["cost"])
        if policy == "coverage":
            score = coverage
        elif policy == "gain":
            score = (
                predicted_gain
                + cfg["coverage_weight"] * coverage
                - cfg["cost_weight"] * cost
            )
        else:
            score = (
                predicted_gain
                + cfg["uncertainty_weight"] * uncertainty
                + cfg["coverage_weight"] * coverage
                - cfg["cost_weight"] * cost
            )
        components = {
            "coverage": coverage,
            "predicted_gain": predicted_gain,
            "uncertainty": uncertainty,
            "cost": cost,
        }
        guidance_tier = 1 if proposal["deprioritized_hypotheses"] else 0
        ranked.append(
            (round(score, 10), guidance_tier, point_id_value, proposal, components)
        )
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    score, _, selected_id, selected, components = ranked[0]
    prediction = prediction_by_id.get(selected_id)
    receipt = {
        "schema_version": POLICY_RECEIPT_SCHEMA_VERSION,
        "space_revision": proposal_set["space"]["space_revision"],
        "proposal_set_revision": proposal_set["proposal_set_revision"],
        "policy": {"name": policy, "config": cfg},
        "action": proposal_set["action"],
        "selected_point_id": selected_id,
        "components": components,
        "acquisition_score": score,
        "evidence": [] if prediction is None else prediction["evidence"],
        "ranked_point_ids": [item[2] for item in ranked],
    }
    return selected["point"], receipt


def _framework_policy_config(ledger_path: Path | None) -> tuple[str | None, dict[str, float]]:
    if ledger_path is None:
        return None, {}
    path = ledger_path.parent / "framework_cfg.json"
    if not path.is_file():
        return None, {}
    try:
        root = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read semantic policy config {path}: {exc}") from exc
    if not isinstance(root, dict):
        raise ContractError(f"semantic policy config {path} must be a JSON object")
    section = root.get("semantic_search", {})
    if not isinstance(section, dict):
        raise ContractError(f"{path}: semantic_search must be an object")
    unknown = sorted(set(section) - ({"policy"} | set(DEFAULT_POLICY_CONFIG)))
    if unknown:
        raise ContractError(f"{path}: unknown semantic_search keys {unknown}")
    policy = section.get("policy")
    if policy is not None and policy not in POLICIES:
        raise ContractError(f"{path}: semantic_search.policy must be one of {sorted(POLICIES)}")
    config = {
        key: value
        for key, value in section.items()
        if key in DEFAULT_POLICY_CONFIG
    }
    return policy, config


def cmd_propose(args: argparse.Namespace) -> int:
    registry = load_registry(args.background)
    ledger = _load_object(args.ledger) if args.ledger and args.ledger.exists() else {"records": []}
    errors = validate_registry(registry, ledger=ledger)
    errors.extend(validate_background_markdown(args.background, registry))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    parents = [item.strip() for item in (args.parents or "").split(",") if item.strip()]
    value = build_proposal_set(
        registry,
        ledger,
        op=args.op,
        parents=parents,
        max_points=args.max_points,
    )
    _write_object(args.output, value)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "proposal_set_revision": value["proposal_set_revision"],
                "n_proposals": len(value["proposals"]),
                "action": value["action"],
            },
            separators=(",", ":"),
        )
    )
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    proposals = _load_object(args.proposals)
    predictions = _load_object(args.predictions) if args.predictions else None
    configured_policy, configured_weights = _framework_policy_config(args.ledger)
    policy = args.policy or configured_policy or "coverage"
    config = dict(configured_weights)
    if args.cfg:
        override = json.loads(args.cfg)
        if not isinstance(override, dict):
            raise ContractError("--cfg must be a JSON object")
        config.update(override)
    point, receipt = select_proposal(
        proposals, policy=policy, predictions=predictions, config=config
    )
    _write_object(args.point_output, point)
    _write_object(args.receipt_output, receipt)
    print(
        json.dumps(
            {
                "ok": True,
                "policy": policy,
                "selected_point_id": point["point_id"],
                "point_output": str(args.point_output),
                "receipt_output": str(args.receipt_output),
                "components": receipt["components"],
            },
            separators=(",", ":"),
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    propose = sub.add_parser("propose", help="build bounded valid points for one graph action")
    propose.add_argument("--background", type=Path, required=True)
    propose.add_argument("--ledger", type=Path)
    propose.add_argument("--op", choices=["fresh", "improve", "crossover"], required=True)
    propose.add_argument("--parents", default="", help="comma-separated numeric parents")
    propose.add_argument("--max-points", type=int, default=128)
    propose.add_argument("--output", type=Path, required=True)
    propose.set_defaults(func=cmd_propose)

    select = sub.add_parser("select", help="apply a replaceable acquisition policy")
    select.add_argument("--proposals", type=Path, required=True)
    select.add_argument("--policy", choices=sorted(POLICIES))
    select.add_argument(
        "--predictions",
        type=Path,
        help=(
            "required for gain policies; each point gets separate [0,1] predicted_gain, "
            "uncertainty, cost, and non-empty evidence"
        ),
    )
    select.add_argument("--ledger", type=Path, help="read run-local semantic_search config")
    select.add_argument("--cfg", help="JSON acquisition-weight overrides")
    select.add_argument("--point-output", type=Path, required=True)
    select.add_argument("--receipt-output", type=Path, required=True)
    select.set_defaults(func=cmd_select)
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
