#!/usr/bin/env python3
"""P2 semantic-point proposal and replaceable acquisition policies.

``got_select`` continues to choose the structural graph action and parents.
This module independently turns that assignment into a bounded set of valid
semantic points eligible under the ledger's revisioned ``search_space_state``
overlay, then selects one with one of four policies:

* ``coverage``: deterministic exploration without any model score;
* ``gain``: predicted gain with explicit cost and a small coverage tie-break;
* ``gain_uncertainty``: predicted gain plus a separate uncertainty bonus,
  explicit cost, and coverage;
* ``gain_uncertainty_nocost``: like ``gain_uncertainty`` but without any cost
  prediction, for settings where pre-implementation cost estimates are noise.

Predictions are rubric inputs, not calibrated Bayesian posteriors.  Model-scored
policies separate a background/mechanism prior from an experience-conditioned
adjustment, and the helper checks that the final gain and uncertainty are the
exact adjusted values.  Every selection writes those components separately in
a policy receipt; neither the registry nor observation history is mutated.
Runtime-deprioritized points use a separate deterministic admission-budget
lane: every configured Nth selection is reserved for that lane, and acquisition
scores rank only within the scheduled lane.

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
)
from search_space_state import (
    compose_effective_selection,
    empty_search_space_state,
    replay_search_space_state,
    validate_point_eligibility,
)
from semantic_evidence import (
    acquisition_conditioning,
    conditioning_cited_ids,
    hypothesis_carriers,
    mechanical_gain_directions,
    validate_conditioned_adjustment,
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


PROPOSAL_SCHEMA_VERSION = 4
GAIN_CONTEXT_SCHEMA_VERSION = 3
PREDICTION_SCHEMA_VERSION = 3
CONDITIONED_PREDICTION_SCHEMA_VERSION = 2
LEGACY_PREDICTION_SCHEMA_VERSION = 1
POLICY_RECEIPT_SCHEMA_VERSION = 7
POLICIES = {"coverage", "coverage_experience", "gain", "gain_uncertainty", "gain_uncertainty_nocost"}
DEFAULT_POLICY_CONFIG = {
    "coverage_weight": 0.10,
    "cost_weight": 0.20,
    "uncertainty_weight": 0.50,
    "deprioritized_budget_interval": 5,
    "llm_intelligence_score": 100.0,
    "carrier_pos_weight": 0.05,
    "carrier_pos_cap": 2,
    "carrier_neg_weight": 0.20,
    "carrier_neg_cap": 3,
}
MAX_PROPOSALS = 128
MAX_EXPERIENCE_RUN_IDS = 5
MAX_EXPERIENCE_EDGE_IDS = 5
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


def _experience_snapshot_receipt(experience: Any) -> dict[str, Any]:
    """Return the exact replaceable-belief revision used for one prediction.

    The full experience remains in the ledger.  Predictions copy this compact
    receipt so ``select`` can reject a stale or hand-waved history adjustment.
    """
    if experience in (None, {}):
        return {
            "generation": None,
            "updated_at_run": None,
            "revision": None,
        }
    if not isinstance(experience, dict):
        raise ContractError("ledger.experience must be an object when present")
    generation = experience.get("generation")
    updated_at_run = experience.get("updated_at_run")
    if (
        experience.get("schema_version") not in {3, 4}
        or not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 0
        or not isinstance(updated_at_run, str)
        or not updated_at_run.isdigit()
    ):
        raise ContractError(
            "ledger.experience must be a valid schema-3/4 snapshot with generation "
            "and numeric updated_at_run before gain prediction"
        )
    return {
        "generation": generation,
        "updated_at_run": updated_at_run,
        "revision": digest(experience),
    }


def _proposal_acquisition_target_relations(
    proposal: dict[str, Any],
) -> dict[str, str]:
    """Targets and exact move relations relevant to one proposal."""
    parent_diffs = proposal.get("parent_diffs")
    if isinstance(parent_diffs, list) and parent_diffs:
        relation_sets: dict[str, set[str]] = {}
        for parent_diff in parent_diffs:
            changes = (
                parent_diff.get("changes")
                if isinstance(parent_diff, dict)
                else None
            )
            for change in changes if isinstance(changes, list) else []:
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
    point = proposal.get("point")
    if not isinstance(point, dict):
        return {}
    # A fresh proposal has no relative move.  Hypothesis-level evidence about
    # the concrete selected mechanisms is relevant; broad dimension narratives
    # are not allowed to condition every complete point.
    try:
        return {
            target_id: "selected_fresh"
            for target_id in selected_assignments(point).values()
        }
    except Exception:
        return {}


def build_gain_context(
    proposal_set: dict[str, Any], ledger: dict[str, Any]
) -> dict[str, Any]:
    """Build the bounded, revisioned and prose-free experience scoring input."""
    # Import lazily to keep the helper dependency one-way at module load time.
    from ledger import _experience_refresh_status

    try:
        refresh = _experience_refresh_status(ledger)
    except ValueError as exc:
        raise ContractError(f"invalid experience refresh state: {exc}") from None
    if refresh["semantic_admission_blocked"]:
        raise ContractError(
            "stale experience: process the terminal DAG delta before building "
            "another semantic acquisition context"
        )
    errors = validate_proposal_set(proposal_set)
    if errors:
        raise ContractError("invalid proposal set: " + "; ".join(errors))
    experience = ledger.get("experience")
    receipt = _experience_snapshot_receipt(experience)
    gain_directions = mechanical_gain_directions(ledger, experience)
    conditioning_by_point = {
        proposal["point_id"]: acquisition_conditioning(
            experience,
            proposal.get("point"),
            target_relations=_proposal_acquisition_target_relations(proposal),
            gain_directions=gain_directions,
        )
        for proposal in proposal_set["proposals"]
    }
    cited_ids: set[str] = set()
    cited_edge_ids: set[str] = set()
    for conditioning in conditioning_by_point.values():
        run_ids, edge_ids = conditioning_cited_ids(conditioning)
        cited_ids.update(run_ids)
        cited_edge_ids.update(edge_ids)
    return {
        "schema_version": GAIN_CONTEXT_SCHEMA_VERSION,
        "proposal_set_revision": proposal_set["proposal_set_revision"],
        "experience_receipt": receipt,
        # Deliberately omit summary/promising_regions/lessons/bottlenecks and
        # target-entry prose.  Only helper-normalized structured receipts may
        # condition acquisition.
        "conditioning_by_point": conditioning_by_point,
        # IDs are audit handles only.  Scores and signed edge deltas are omitted
        # so uncertainty-only/confounded evidence cannot be laundered into the
        # model-authored background prior.
        "experience_evidence_run_ids": sorted(cited_ids),
        "experience_evidence_edge_ids": sorted(cited_edge_ids),
    }


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


def _eligible_hypotheses(
    registry: dict[str, Any], ledger: dict[str, Any]
) -> tuple[dict[str, list[str]], dict[str, Any], int]:
    """Selectable hypotheses and effective statuses at the ledger's state revision.

    A missing bootstrap ledger or an absent top-level state replays as
    :func:`empty_search_space_state` (the first proposal); a persisted P2
    ledger is validated upstream to carry the overlay object.  ``excluded``
    and ``pruned`` content is dropped, ``deprioritized`` content stays
    eligible, and a protected baseline always remains: under a validated
    registry and state the explicit baseline's effective status is ``active``,
    so a runtime-pruned dimension exposes only its baseline.
    """
    state = ledger.get("search_space_state") if isinstance(ledger, dict) else None
    if not isinstance(state, dict):
        state = empty_search_space_state()
    state_revision = state.get("revision")
    if (
        not isinstance(state_revision, int)
        or isinstance(state_revision, bool)
        or state_revision < 0
    ):
        raise ContractError(
            "ledger.search_space_state.revision must be a non-negative integer"
        )
    runtime = replay_search_space_state(registry, state)
    effective = compose_effective_selection(
        registry, derive_hypothesis_selection(registry), runtime
    )
    result: dict[str, list[str]] = {}
    for dimension in registry.get("dimensions", []):
        if not isinstance(dimension, dict):
            continue
        choices = [
            hypothesis.get("id")
            for hypothesis in dimension.get("hypotheses", [])
            if isinstance(hypothesis, dict)
            and effective.get(hypothesis.get("id"), {}).get("effective_status")
            not in {"excluded", "pruned"}
        ]
        result[dimension["id"]] = [str(item) for item in choices]
    return result, effective, state_revision


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
    effective: dict[str, Any],
    values: dict[str, dict[str, Any]],
    point: dict[str, Any] | None,
) -> None:
    if point is None or validate_point(point, registry):
        return
    # Revision-current eligibility gate: deterministic completion of a
    # `requires` relation can select a hypothesis absent from the sparse
    # overrides, so filtering only the input choice lists is insufficient.
    if validate_point_eligibility(point, registry, effective):
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
    registry: dict[str, Any],
    eligible: dict[str, list[str]],
    effective: dict[str, Any],
    max_points: int,
) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    _add_point(registry, effective, values, complete_point(registry))
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
        _add_point(registry, effective, values, complete_point(registry, override))
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
        _add_point(registry, effective, values, complete_point(registry, merged))
        if len(values) >= max_points:
            break
    return list(values.values())


def _improve_points(
    registry: dict[str, Any],
    parent: dict[str, Any],
    eligible: dict[str, list[str]],
    effective: dict[str, Any],
    max_points: int,
) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    parent_point = parent.get("semantic_point")
    _add_point(registry, effective, values, parent_point)
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
        _add_point(registry, effective, values, complete_point(registry, overrides))
        if len(values) >= max_points:
            return list(values.values())
    return list(values.values())


def _crossover_points(
    registry: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    eligible: dict[str, list[str]],
    effective: dict[str, Any],
    max_points: int,
) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    left_point = left.get("semantic_point")
    right_point = right.get("semantic_point")
    left_selected = selected_assignments(left_point)
    right_selected = selected_assignments(right_point)
    _add_point(registry, effective, values, left_point)
    _add_point(registry, effective, values, right_point)
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
        _add_point(registry, effective, values, complete_point(registry, overrides))
        if len(values) >= max_points:
            return list(values.values())
    # If parents occupy the same or nearby point, implementation-level
    # recombination is still valid; add one-hop semantic alternatives as useful
    # neighbors without claiming that the point determines the implementation.
    if len(values) < max_points:
        for point in _improve_points(registry, left, eligible, effective, max_points):
            _add_point(registry, effective, values, point)
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
    eligible, effective, state_revision = _eligible_hypotheses(registry, ledger)
    if any(not choices for choices in eligible.values()):
        empty = [dimension_id for dimension_id, choices in eligible.items() if not choices]
        raise ContractError(f"selected dimensions have no eligible hypotheses: {empty}")
    if op == "fresh":
        points = _fresh_points(registry, eligible, effective, max_points)
    elif op == "improve":
        points = _improve_points(registry, parent_records[0], eligible, effective, max_points)
    else:
        points = _crossover_points(
            registry, parent_records[0], parent_records[1], eligible, effective, max_points
        )
    if not points:
        raise ContractError(f"no valid semantic points can satisfy action {op}")
    coverage = coverage_from_records(registry, ledger.get("records", []))
    proposals: list[dict[str, Any]] = []
    for point in points:
        selected = selected_assignments(point)
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
                    if effective.get(hypothesis_id, {}).get("effective_status")
                    == "deprioritized"
                ),
            }
        )
        proposals[-1]["budget_lane"] = (
            "deprioritized"
            if proposals[-1]["deprioritized_hypotheses"]
            else "active"
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
        "search_space_state_revision": state_revision,
        "baseline_hypothesis_ids": sorted(
            str(dimension["baseline_hypothesis_id"])
            for dimension in registry.get("dimensions", [])
            if isinstance(dimension, dict)
            and dimension.get("baseline_hypothesis_id") is not None
        ),
        "action": {"op": op, "parents": parents},
        "coverage_snapshot": {
            "n_valid_records": coverage["n_valid_records"],
            "n_unique_points": coverage["n_unique_points"],
        },
        "proposals": proposals[:max_points],
    }
    value["proposal_set_revision"] = digest(value)
    return value


def build_baseline_proposal_set(
    registry: dict[str, Any],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    """Build the single all-baselines proposal for an empty run.

    A task-provided entrypoint is an observed control, not an acquisition
    choice.  Keep its semantic admission on the normal proposal/receipt path,
    while making the two invariants that matter deterministic: it is the first
    record and it maps to the registry's complete all-baselines point.
    """
    records = ledger.get("records", [])
    if not isinstance(records, list):
        raise ContractError("ledger.records must be a list")
    if records:
        raise ContractError(
            "the provided baseline must be admitted before any other record"
        )
    value = build_proposal_set(
        registry,
        ledger,
        op="fresh",
        parents=[],
        max_points=1,
    )
    baseline = complete_point(registry)
    proposals = value.get("proposals", [])
    if (
        len(proposals) != 1
        or proposals[0].get("point_id") != baseline.get("point_id")
    ):
        raise ContractError(
            "baseline-only proposal did not resolve to the all-baselines point"
        )
    return value


def validate_proposal_set(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["proposal set must be an object"]
    errors: list[str] = []
    allowed_top = {
        "schema_version",
        "space",
        "search_space_state_revision",
        "baseline_hypothesis_ids",
        "action",
        "coverage_snapshot",
        "proposals",
        "proposal_set_revision",
    }
    unknown_top = sorted(set(value) - allowed_top)
    if unknown_top:
        errors.append(f"proposal set has unknown fields {unknown_top}")
    if value.get("schema_version") not in {3, 4}:
        errors.append("proposal set schema_version must be 3 or 4")
    if value.get("schema_version") == 4:
        baseline_ids = value.get("baseline_hypothesis_ids")
        if (
            not isinstance(baseline_ids, list)
            or any(not isinstance(item, str) or not item for item in baseline_ids)
        ):
            errors.append(
                "proposal set baseline_hypothesis_ids must be a list of hypothesis ids"
            )
    state_revision = value.get("search_space_state_revision")
    if (
        not isinstance(state_revision, int)
        or isinstance(state_revision, bool)
        or state_revision < 0
    ):
        errors.append("proposal set search_space_state_revision must be a non-negative integer")
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
            or not isinstance(catalog.get("id"), str)
            or not catalog.get("id")
            or not isinstance(catalog.get("revision"), str)
            or DIGEST_RE.fullmatch(catalog["revision"]) is None
            or set(catalog) != {"id", "revision"}
        ):
            errors.append("proposal set space.catalog must be a resolved-catalog receipt")
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
                "budget_lane",
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
            lane = proposal.get("budget_lane")
            if lane not in {"active", "deprioritized"}:
                errors.append(f"{where}.budget_lane must be active or deprioritized")
            elif lane != ("deprioritized" if deprioritized else "active"):
                errors.append(
                    f"{where}.budget_lane must match deprioritized_hypotheses"
                )
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
    value: dict[str, Any] | None,
    proposal_set: dict[str, Any],
    policy: str,
    *,
    experience: Any = None,
    ledger: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if value is None:
        return {}, ["gain policies require a predictions JSON object"]
    errors: list[str] = []
    schema_version = value.get("schema_version")
    if schema_version not in {
        LEGACY_PREDICTION_SCHEMA_VERSION,
        CONDITIONED_PREDICTION_SCHEMA_VERSION,
        PREDICTION_SCHEMA_VERSION,
    }:
        errors.append(
            "predictions.schema_version must be 1/2 (legacy bootstrap only) or 3"
        )
    if value.get("proposal_set_revision") != proposal_set.get("proposal_set_revision"):
        errors.append("predictions.proposal_set_revision does not match proposals")
    try:
        expected_experience = _experience_snapshot_receipt(experience)
    except ContractError as exc:
        errors.append(str(exc))
        expected_experience = {
            "generation": None,
            "updated_at_run": None,
            "revision": None,
        }
    has_experience_snapshot = expected_experience["revision"] is not None
    gain_directions = mechanical_gain_directions(ledger or {}, experience)
    if schema_version == LEGACY_PREDICTION_SCHEMA_VERSION:
        if set(value) != {"schema_version", "proposal_set_revision", "predictions"}:
            errors.append("legacy predictions must contain only schema, proposal revision, and predictions")
        if has_experience_snapshot:
            errors.append(
                "schema-3 predictions are required when ledger.experience exists"
            )
    elif schema_version in {
        CONDITIONED_PREDICTION_SCHEMA_VERSION,
        PREDICTION_SCHEMA_VERSION,
    }:
        if set(value) != {
            "schema_version",
            "proposal_set_revision",
            "experience",
            "predictions",
        }:
            errors.append(
                "conditioned predictions must contain exactly schema_version, "
                "proposal_set_revision, experience, and predictions"
            )
        if value.get("experience") != expected_experience:
            errors.append(
                "predictions.experience must match the current gain-context "
                "experience receipt"
            )
        if (
            schema_version == CONDITIONED_PREDICTION_SCHEMA_VERSION
            and has_experience_snapshot
        ):
            errors.append(
                "schema-2 experience conditioning is no longer admissible; "
                "regenerate schema-3 predictions from the gated gain-context"
            )
    predictions = value.get("predictions")
    if not isinstance(predictions, list):
        return {}, errors + ["predictions.predictions must be a list"]
    result: dict[str, dict[str, Any]] = {}
    proposal_by_id = {
        item["point_id"]: item for item in proposal_set["proposals"]
    }
    proposal_ids = set(proposal_by_id)
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
        if schema_version in {
            CONDITIONED_PREDICTION_SCHEMA_VERSION,
            PREDICTION_SCHEMA_VERSION,
        }:
            score_fields = (
                "prior_gain",
                "predicted_gain",
                "prior_uncertainty",
                "uncertainty",
            )
            adjustment_fields = (
                "experience_gain_adjustment",
                "experience_uncertainty_adjustment",
            )
            allowed_fields = {
                "point_id",
                *score_fields,
                *adjustment_fields,
                "experience_run_ids",
                "experience_edge_ids",
                "experience_rationale",
                "evidence",
            }
            if schema_version == PREDICTION_SCHEMA_VERSION:
                allowed_fields.add("experience_target_ids")
        else:
            score_fields = ("predicted_gain", "uncertainty")
            adjustment_fields = ()
            allowed_fields = {"point_id", "predicted_gain", "uncertainty", "evidence"}
        if policy != "gain_uncertainty_nocost":
            score_fields += ("cost",)
            allowed_fields.add("cost")
        for field in score_fields:
            score = prediction.get(field)
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                errors.append(f"{where}.{field} must be a number in [0, 1]")
        for field in adjustment_fields:
            score = prediction.get(field)
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
                or not -1.0 <= float(score) <= 1.0
            ):
                errors.append(f"{where}.{field} must be a number in [-1, 1]")
        if schema_version in {
            CONDITIONED_PREDICTION_SCHEMA_VERSION,
            PREDICTION_SCHEMA_VERSION,
        }:
            gain_parts = (
                prediction.get("prior_gain"),
                prediction.get("experience_gain_adjustment"),
                prediction.get("predicted_gain"),
            )
            uncertainty_parts = (
                prediction.get("prior_uncertainty"),
                prediction.get("experience_uncertainty_adjustment"),
                prediction.get("uncertainty"),
            )
            for label, parts in (
                ("predicted_gain", gain_parts),
                ("uncertainty", uncertainty_parts),
            ):
                if all(
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and math.isfinite(float(item))
                    for item in parts
                ) and not math.isclose(
                    float(parts[0]) + float(parts[1]),
                    float(parts[2]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    errors.append(
                        f"{where}.{label} must equal its prior plus experience adjustment"
                    )
            raw_run_ids = prediction.get("experience_run_ids")
            experience_run_ids = (
                raw_run_ids
                if isinstance(raw_run_ids, list)
                and all(isinstance(item, str) for item in raw_run_ids)
                else []
            )
            if (
                not isinstance(raw_run_ids, list)
                or len(experience_run_ids) > MAX_EXPERIENCE_RUN_IDS
                or len(experience_run_ids) != len(set(experience_run_ids))
            ):
                errors.append(
                    f"{where}.experience_run_ids must contain 0–"
                    f"{MAX_EXPERIENCE_RUN_IDS} unique run ids"
                )
            raw_edge_ids = prediction.get("experience_edge_ids")
            experience_edge_ids = (
                raw_edge_ids
                if isinstance(raw_edge_ids, list)
                and all(isinstance(item, str) for item in raw_edge_ids)
                else []
            )
            if (
                not isinstance(raw_edge_ids, list)
                or len(experience_edge_ids) > MAX_EXPERIENCE_EDGE_IDS
                or len(experience_edge_ids) != len(set(experience_edge_ids))
            ):
                errors.append(
                    f"{where}.experience_edge_ids must contain 0–"
                    f"{MAX_EXPERIENCE_EDGE_IDS} unique edge ids"
                )
            rationale = prediction.get("experience_rationale")
            if (
                not isinstance(rationale, str)
                or not rationale.strip()
                or len(rationale) > 240
            ):
                errors.append(
                    f"{where}.experience_rationale must be non-empty and at most "
                    "240 characters"
                )
            gain_adjustment = prediction.get("experience_gain_adjustment")
            uncertainty_adjustment = prediction.get(
                "experience_uncertainty_adjustment"
            )
            if schema_version == PREDICTION_SCHEMA_VERSION:
                raw_target_ids = prediction.get("experience_target_ids")
                target_ids = (
                    raw_target_ids
                    if isinstance(raw_target_ids, list)
                    and all(isinstance(item, str) for item in raw_target_ids)
                    else []
                )
                if (
                    not isinstance(raw_target_ids, list)
                    or len(target_ids) != len(set(target_ids))
                ):
                    errors.append(
                        f"{where}.experience_target_ids must be a unique string list"
                    )
                available = acquisition_conditioning(
                    experience,
                    proposal_by_id[point_id_value].get("point"),
                    target_relations=_proposal_acquisition_target_relations(
                        proposal_by_id[point_id_value]
                    ),
                    gain_directions=gain_directions,
                )
                by_target = {item["target_id"]: item for item in available}
                unknown_targets = sorted(set(target_ids) - set(by_target))
                if unknown_targets:
                    errors.append(
                        f"{where}.experience_target_ids are not relevant gated "
                        f"targets {unknown_targets}"
                    )
                selected_conditioning = [
                    by_target[target_id]
                    for target_id in target_ids
                    if target_id in by_target
                ]
                expected_run_ids, expected_edge_ids = conditioning_cited_ids(
                    selected_conditioning
                )
                if set(experience_run_ids) != expected_run_ids:
                    errors.append(
                        f"{where}.experience_run_ids must equal the helper-derived "
                        "union for experience_target_ids"
                    )
                if set(experience_edge_ids) != expected_edge_ids:
                    errors.append(
                        f"{where}.experience_edge_ids must equal the helper-derived "
                        "union for experience_target_ids"
                    )
                errors.extend(
                    f"{where}: {error}"
                    for error in validate_conditioned_adjustment(
                        selected_conditioning,
                        proposal_by_id[point_id_value].get("point"),
                        evidence_run_ids=experience_run_ids,
                        evidence_edge_ids=experience_edge_ids,
                        gain_adjustment=gain_adjustment,
                        uncertainty_adjustment=uncertainty_adjustment,
                    )
                )
            else:
                selected_conditioning = []
                if experience_run_ids or experience_edge_ids:
                    errors.append(
                        f"{where} legacy schema-2 predictions cannot cite experience"
                    )
                if any(
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and not math.isclose(float(item), 0.0, abs_tol=1e-12)
                    for item in (gain_adjustment, uncertainty_adjustment)
                ):
                    errors.append(
                        f"{where} legacy schema-2 experience adjustments must be zero"
                    )
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
        unknown = sorted(set(prediction) - allowed_fields)
        if unknown:
            errors.append(f"{where} has unknown fields {unknown}")
        normalized = dict(prediction)
        if schema_version == LEGACY_PREDICTION_SCHEMA_VERSION:
            normalized.update(
                {
                    "prior_gain": prediction.get("predicted_gain"),
                    "experience_gain_adjustment": 0.0,
                    "prior_uncertainty": prediction.get("uncertainty"),
                    "experience_uncertainty_adjustment": 0.0,
                    "experience_run_ids": [],
                    "experience_edge_ids": [],
                    "experience_target_ids": [],
                    "_experience_conditioning": [],
                    "experience_rationale": (
                        "legacy bootstrap prediction without an experience snapshot"
                    ),
                }
            )
        elif schema_version == CONDITIONED_PREDICTION_SCHEMA_VERSION:
            normalized.update(
                {
                    "experience_target_ids": [],
                    "_experience_conditioning": [],
                }
            )
        else:
            normalized["_experience_conditioning"] = selected_conditioning
        result[point_id_value] = normalized
    missing = sorted(proposal_ids - set(result))
    if missing:
        errors.append(f"predictions are missing proposal ids {missing}")
    return result, errors


def _carrier_priors(
    proposal_set: dict[str, Any],
    ledger: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, tuple[float, dict[str, dict[str, int]]]]:
    """Deterministic per-point experience prior from hypothesis carriers.

    Only non-baseline hypotheses contribute: a reversion to baseline is the
    failing treatment's evidence, and counting it again as a baseline bonus
    would double-count the edge and bias selection toward baseline points.
    """
    baselines = set(proposal_set.get("baseline_hypothesis_ids") or [])
    cache: dict[str, dict[str, Any]] = {}
    priors: dict[str, tuple[float, dict[str, dict[str, int]]]] = {}
    for proposal in proposal_set["proposals"]:
        prior = 0.0
        detail: dict[str, dict[str, int]] = {}
        deprioritized = set(proposal.get("deprioritized_hypotheses") or [])
        for hypothesis_id in sorted(
            set(selected_assignments(proposal["point"]).values()) - baselines
        ):
            if hypothesis_id not in cache:
                cache[hypothesis_id] = hypothesis_carriers(
                    ledger, target_id=hypothesis_id
                )
            carriers = cache[hypothesis_id]
            # A deprioritized hypothesis (runtime overlay or external
            # guidance) counts as at least one negative context even without
            # carrier history — demotion must not be selection-inert.
            negative = carriers["negative"]
            if hypothesis_id in deprioritized:
                negative = max(negative, 1)
            prior += min(carriers["positive"], int(cfg["carrier_pos_cap"])) * float(
                cfg["carrier_pos_weight"]
            ) - min(negative, int(cfg["carrier_neg_cap"])) * float(
                cfg["carrier_neg_weight"]
            )
            detail[hypothesis_id] = {
                "negative": negative,
                "positive": carriers["positive"],
            }
        priors[proposal["point_id"]] = (round(prior, 10), detail)
    return priors


def select_proposal(
    proposal_set: dict[str, Any],
    *,
    policy: str,
    predictions: dict[str, Any] | None = None,
    config: dict[str, float | int] | None = None,
    selection_index: int = 1,
    experience: Any = None,
    ledger: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors = validate_proposal_set(proposal_set)
    if errors:
        raise ContractError("invalid proposal set: " + "; ".join(errors))
    if policy not in POLICIES:
        raise ContractError(f"policy must be one of {sorted(POLICIES)}")
    if (
        not isinstance(selection_index, int)
        or isinstance(selection_index, bool)
        or selection_index < 1
    ):
        raise ContractError("selection_index must be a positive integer")
    cfg = dict(DEFAULT_POLICY_CONFIG)
    if config:
        unknown = sorted(set(config) - set(cfg))
        if unknown:
            raise ContractError(f"unknown semantic policy config keys {unknown}")
        for key, value in config.items():
            if key == "deprioritized_budget_interval":
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 2 <= value <= 1000
                ):
                    raise ContractError(
                        "semantic policy config deprioritized_budget_interval "
                        "must be an integer in [2, 1000]"
                    )
                cfg[key] = value
                continue
            if key in {"carrier_pos_cap", "carrier_neg_cap"}:
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 1 <= value <= 100
                ):
                    raise ContractError(
                        f"semantic policy config {key} must be an integer in [1, 100]"
                    )
                cfg[key] = value
                continue
            if key == "llm_intelligence_score":
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 100.0
                ):
                    raise ContractError(
                        "semantic policy config llm_intelligence_score "
                        "must be a finite number in [0, 100]"
                    )
                cfg[key] = float(value)
                continue
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ContractError(f"semantic policy config {key} must be a non-negative number")
            cfg[key] = float(value)

    prediction_by_id: dict[str, dict[str, Any]] = {}
    if policy not in {"coverage", "coverage_experience"}:
        prediction_by_id, prediction_errors = _prediction_map(
            predictions,
            proposal_set,
            policy,
            experience=experience,
            ledger=ledger,
        )
        if prediction_errors:
            raise ContractError("invalid policy predictions: " + "; ".join(prediction_errors))

    carrier_priors: dict[str, tuple[float, dict[str, dict[str, int]]]] = {}
    if policy == "coverage_experience":
        if not isinstance(ledger, dict):
            raise ContractError("coverage_experience selection requires the ledger")
        carrier_priors = _carrier_priors(proposal_set, ledger, cfg)

    ranked: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
    llm_judgment_weight = float(cfg["llm_intelligence_score"]) / 100.0
    for proposal in proposal_set["proposals"]:
        point_id_value = proposal["point_id"]
        prediction = prediction_by_id.get(point_id_value)
        coverage = float(proposal["coverage"])
        predicted_gain = None if prediction is None else float(prediction["predicted_gain"])
        uncertainty = None if prediction is None else float(prediction["uncertainty"])
        cost = (
            None
            if prediction is None or "cost" not in prediction
            else float(prediction["cost"])
        )
        if policy == "coverage":
            score = coverage
        elif policy == "coverage_experience":
            score = coverage + carrier_priors[point_id_value][0]
        elif policy == "gain":
            score = (
                llm_judgment_weight
                * (predicted_gain - cfg["cost_weight"] * cost)
                + cfg["coverage_weight"] * coverage
            )
        elif policy == "gain_uncertainty":
            score = (
                llm_judgment_weight
                * (
                    predicted_gain
                    + cfg["uncertainty_weight"] * uncertainty
                    - cfg["cost_weight"] * cost
                )
                + cfg["coverage_weight"] * coverage
            )
        else:
            score = (
                llm_judgment_weight
                * (
                    predicted_gain
                    + cfg["uncertainty_weight"] * uncertainty
                )
                + cfg["coverage_weight"] * coverage
            )
        components = {
            "coverage": coverage,
            "llm_judgment_weight": (
                None if prediction is None else llm_judgment_weight
            ),
            "prior_gain": (
                None if prediction is None else float(prediction["prior_gain"])
            ),
            "experience_gain_adjustment": (
                None
                if prediction is None
                else float(prediction["experience_gain_adjustment"])
            ),
            "predicted_gain": predicted_gain,
            "prior_uncertainty": (
                None if prediction is None else float(prediction["prior_uncertainty"])
            ),
            "experience_uncertainty_adjustment": (
                None
                if prediction is None
                else float(prediction["experience_uncertainty_adjustment"])
            ),
            "uncertainty": uncertainty,
            "cost": cost,
            "experience_prior": (
                None
                if policy != "coverage_experience"
                else carrier_priors[point_id_value][0]
            ),
            "carriers": (
                None
                if policy != "coverage_experience"
                else carrier_priors[point_id_value][1]
            ),
        }
        ranked.append((round(score, 10), point_id_value, proposal, components))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    # Lane scheduling was removed: deprioritized content stays eligible and is
    # penalized through the carrier prior instead of a forced budget lane.
    # `budget_lane` is still computed in build_proposal_set for backward
    # readability but no longer affects selection.
    selected_item = ranked[0]
    base_rank = 1
    final_ranked = ranked
    score, selected_id, selected, components = selected_item
    prediction = prediction_by_id.get(selected_id)
    snapshot = _experience_snapshot_receipt(experience)
    if prediction is not None:
        rationale = prediction["experience_rationale"]
    elif policy == "coverage_experience":
        rationale = (
            "deterministic carrier prior over recorded edges; "
            "no model-scored experience"
        )
    else:
        rationale = "coverage policy does not use model-scored experience"
    experience_receipt = {
        **snapshot,
        "conditioning": (
            [] if prediction is None else prediction["_experience_conditioning"]
        ),
        "evidence_run_ids": (
            [] if prediction is None else prediction["experience_run_ids"]
        ),
        "evidence_edge_ids": (
            [] if prediction is None else prediction["experience_edge_ids"]
        ),
        "rationale": rationale,
    }
    receipt = {
        "schema_version": POLICY_RECEIPT_SCHEMA_VERSION,
        "space_revision": proposal_set["space"]["space_revision"],
        "search_space_state_revision": proposal_set["search_space_state_revision"],
        "proposal_set_revision": proposal_set["proposal_set_revision"],
        "policy": {"name": policy, "config": cfg},
        "action": proposal_set["action"],
        "selected_point_id": selected_id,
        "components": components,
        "acquisition_score": score,
        "evidence": [] if prediction is None else prediction["evidence"],
        "experience": experience_receipt,
        "budget": {
            "selection_index": selection_index,
            "deprioritized_interval": None,
            "scheduled_lane": None,
            "selected_lane": None,
            "fallback": "lanes_removed",
            "base_rank": base_rank,
        },
        "ranked_point_ids": [item[1] for item in final_ranked],
    }
    return selected["point"], receipt


def _framework_policy_config(
    ledger_path: Path | None,
) -> tuple[str | None, dict[str, float | int]]:
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
    # Setup validates and freezes background/catalog/source evidence.  Proposal
    # generation is a high-frequency runtime projection and validates the
    # changing action, ancestry, point eligibility, and state it consumes.
    registry = load_registry(args.background)
    ledger = _load_object(args.ledger) if args.ledger and args.ledger.exists() else {"records": []}
    parents = [item.strip() for item in (args.parents or "").split(",") if item.strip()]
    if args.baseline_only:
        if args.op != "fresh" or parents:
            raise ContractError("--baseline-only requires --op fresh and no parents")
        value = build_baseline_proposal_set(registry, ledger)
    else:
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
                "search_space_state_revision": value["search_space_state_revision"],
                "n_proposals": len(value["proposals"]),
                "action": value["action"],
            },
            separators=(",", ":"),
        )
    )
    return 0


def cmd_gain_context(args: argparse.Namespace) -> int:
    proposals = _load_object(args.proposals)
    ledger = _load_object(args.ledger) if args.ledger.exists() else {"records": []}
    state = ledger.get("search_space_state")
    current_revision = state.get("revision") if isinstance(state, dict) else 0
    if proposals.get("search_space_state_revision") != current_revision:
        raise ContractError(
            "stale proposal set: regenerate it against the current search-space state "
            "before building gain context"
        )
    context = build_gain_context(proposals, ledger)
    _write_object(args.output, context)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "experience_receipt": context["experience_receipt"],
                "n_conditioned_points": sum(
                    bool(entries)
                    for entries in context["conditioning_by_point"].values()
                ),
                "n_conditioning_entries": sum(
                    len(entries)
                    for entries in context["conditioning_by_point"].values()
                ),
            },
            separators=(",", ":"),
        )
    )
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    proposals = _load_object(args.proposals)
    predictions = _load_object(args.predictions) if args.predictions else None
    configured_policy, configured_weights = _framework_policy_config(args.ledger)
    policy = args.policy or configured_policy or "coverage_experience"
    config = dict(configured_weights)
    if args.cfg:
        override = json.loads(args.cfg)
        if not isinstance(override, dict):
            raise ContractError("--cfg must be a JSON object")
        config.update(override)
    ledger = _load_object(args.ledger) if args.ledger and args.ledger.exists() else {}
    configured_llm_score = config.get(
        "llm_intelligence_score",
        DEFAULT_POLICY_CONFIG["llm_intelligence_score"],
    )
    if (
        isinstance(configured_llm_score, (int, float))
        and not isinstance(configured_llm_score, bool)
        and math.isfinite(float(configured_llm_score))
    ):
        prior_records = ledger.get("records", [])
        for prior_record in prior_records if isinstance(prior_records, list) else []:
            prior_receipt = (
                prior_record.get("policy_receipt")
                if isinstance(prior_record, dict)
                else None
            )
            prior_policy = (
                prior_receipt.get("policy")
                if isinstance(prior_receipt, dict)
                and prior_receipt.get("schema_version") in {6, 7}
                else None
            )
            prior_config = (
                prior_policy.get("config")
                if isinstance(prior_policy, dict)
                else None
            )
            prior_score = (
                prior_config.get("llm_intelligence_score")
                if isinstance(prior_config, dict)
                else None
            )
            prior_score_valid = (
                isinstance(prior_score, (int, float))
                and not isinstance(prior_score, bool)
                and math.isfinite(float(prior_score))
            )
            if prior_score_valid and not math.isclose(
                float(configured_llm_score),
                float(prior_score),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ContractError(
                    "semantic_search.llm_intelligence_score is frozen at "
                    f"{float(prior_score):g} after the first schema-6 admission"
                )
            if prior_score_valid:
                break
    from ledger import _experience_refresh_status

    try:
        refresh = _experience_refresh_status(ledger)
    except ValueError as exc:
        raise ContractError(f"invalid experience refresh state: {exc}") from None
    if refresh["semantic_admission_blocked"]:
        raise ContractError(
            "stale experience: process the terminal DAG delta before selecting "
            "another semantic point"
        )
    records = ledger.get("records", [])
    if not isinstance(records, list):
        raise ContractError("ledger.records must be a list")
    state = ledger.get("search_space_state")
    current_revision = state.get("revision") if isinstance(state, dict) else 0
    if (
        not isinstance(current_revision, int)
        or isinstance(current_revision, bool)
        or current_revision < 0
    ):
        raise ContractError(
            "ledger.search_space_state.revision must be a non-negative integer"
        )
    if proposals.get("search_space_state_revision") != current_revision:
        raise ContractError(
            "stale proposal set: search_space_state_revision "
            f"{proposals.get('search_space_state_revision')!r} does not equal the "
            f"ledger's current search space state revision {current_revision}; "
            "re-propose against the current overlay before selecting"
        )
    point, receipt = select_proposal(
        proposals,
        policy=policy,
        predictions=predictions,
        config=config,
        selection_index=len(records) + 1,
        experience=ledger.get("experience"),
        ledger=ledger,
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
                "budget": receipt["budget"],
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
    propose.add_argument(
        "--baseline-only",
        action="store_true",
        help=(
            "for an empty run, emit only the complete all-baselines point; "
            "requires --op fresh and no parents"
        ),
    )
    propose.add_argument("--output", type=Path, required=True)
    propose.set_defaults(func=cmd_propose)

    context = sub.add_parser(
        "gain-context",
        help="render the revisioned bounded experience used by gain predictions",
    )
    context.add_argument("--proposals", type=Path, required=True)
    context.add_argument("--ledger", type=Path, required=True)
    context.add_argument("--output", type=Path, required=True)
    context.set_defaults(func=cmd_gain_context)

    select = sub.add_parser("select", help="apply a replaceable acquisition policy")
    select.add_argument("--proposals", type=Path, required=True)
    select.add_argument("--policy", choices=sorted(POLICIES))
    select.add_argument(
        "--predictions",
        type=Path,
        help=(
            "required for gain policies; schema 2 separates background priors, "
            "experience adjustments, final gain/uncertainty, cost, and evidence "
            "(gain_uncertainty_nocost omits cost)"
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
