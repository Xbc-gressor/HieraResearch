"""Candidate-admission rules for the run ledger.

Admission validates the frozen semantic space and mutates only the supplied
in-memory ledger.  The CLI facade in ``tools/ledger.py`` remains responsible
for loading and committing ``ledger.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from background_contract import (
    load_registry,
    validate_background_markdown,
    validate_registry,
)
from ledger_core import (
    experience_refresh_status,
    get_record,
    new_record,
)
from search_space_state import empty_search_space_state
from semantic_evidence import (
    SemanticEvidenceError,
    acquisition_conditioning,
    acquisition_target_relations,
    build_semantic_edges,
    mechanical_gain_directions,
    validate_conditioned_adjustment,
    validate_conditioning_against_ledger,
)
from semantic_space import (
    SemanticSpaceError,
    digest,
    resolve_dimension_catalog,
    resolve_dimension_strategy,
    space_receipt,
)


class AdmissionError(ValueError):
    """A candidate cannot be admitted without violating ledger contracts."""


@dataclass(frozen=True)
class AdmissionRequest:
    run_id: str
    kind: str
    idea: str
    change: str
    source_run_ids: str
    op: str
    background_path: Path
    catalog_path: Path | None
    semantic_point_path: Path
    policy_receipt_path: Path
    candidate_name_hint: str
    description: str | None = None


def _resolve_space(request: AdmissionRequest, data: dict) -> tuple[dict, dict, str]:
    registry = load_registry(request.background_path)
    try:
        dimension_strategy = resolve_dimension_strategy(request.background_path)
        catalog = resolve_dimension_catalog(
            request.background_path,
            explicit_path=request.catalog_path,
        )
    except SemanticSpaceError as exc:
        raise AdmissionError(f"invalid dimension catalog: {exc}") from exc
    errors = validate_registry(
        registry,
        ledger=data,
        catalog=catalog,
        dimension_strategy=dimension_strategy,
    )
    errors.extend(validate_background_markdown(request.background_path, registry))
    if errors:
        raise AdmissionError("invalid P2 background/ledger: " + "; ".join(errors))
    return registry, catalog, dimension_strategy


def _parent_ids(request: AdmissionRequest) -> list[str]:
    parents = [
        value.strip()
        for value in (request.source_run_ids or "").split(",")
        if value.strip()
    ]
    if any(not parent.isdigit() for parent in parents):
        raise AdmissionError(
            "--source-run-ids accepts numeric parents only; hypothesis attribution "
            "belongs in --semantic-point"
        )
    expected = {"fresh": 0, "improve": 1, "crossover": 2}.get(request.op)
    if expected is None:
        raise AdmissionError("--op is required")
    if len(parents) != expected or len(set(parents)) != len(parents):
        raise AdmissionError(
            f"{request.op} requires {expected} distinct numeric parent ids"
        )
    return parents


def _validate_experience_binding(
    data: dict,
    record: dict,
    policy_receipt: dict,
    parent_ids: list[str],
) -> None:
    current_experience = data.get("experience")
    if isinstance(current_experience, dict) and current_experience:
        expected = {
            "generation": current_experience.get("generation"),
            "updated_at_run": current_experience.get("updated_at_run"),
            "revision": digest(current_experience),
        }
    else:
        expected = {
            "generation": None,
            "updated_at_run": None,
            "revision": None,
        }
    receipt_experience = policy_receipt.get("experience")
    actual = (
        {
            "generation": receipt_experience.get("generation"),
            "updated_at_run": receipt_experience.get("updated_at_run"),
            "revision": receipt_experience.get("revision"),
        }
        if isinstance(receipt_experience, dict)
        else None
    )
    if actual != expected:
        raise AdmissionError(
            "stale policy receipt: experience generation/revision does not match "
            "the ledger's current bounded belief; rebuild gain-context and re-select"
        )
    if not isinstance(receipt_experience, dict):
        return

    by_run_id = {
        str(item.get("run_id")): item
        for item in data.get("records", [])
        if isinstance(item, dict)
    }
    parent_points = [
        by_run_id[parent_id].get("semantic_point")
        for parent_id in parent_ids
        if parent_id in by_run_id
    ]
    target_relations = acquisition_target_relations(
        record["semantic_point"],
        None
        if record["op"] == "fresh"
        else [point for point in parent_points if isinstance(point, dict)],
    )
    available = acquisition_conditioning(
        current_experience,
        record["semantic_point"],
        target_relations=target_relations,
        gain_directions=mechanical_gain_directions(data, current_experience),
    )
    available_by_target = {item["target_id"]: item for item in available}
    conditioning = receipt_experience.get("conditioning")
    conditioning = conditioning if isinstance(conditioning, list) else []
    if any(
        not isinstance(item, dict)
        or item.get("target_id") not in available_by_target
        or item != available_by_target[item.get("target_id")]
        for item in conditioning
    ):
        raise AdmissionError(
            "invalid policy receipt: experience conditioning must be an "
            "exact proposal-relevant helper rendering"
        )
    errors = validate_conditioning_against_ledger(conditioning, data)
    if errors:
        raise AdmissionError("invalid policy receipt: " + "; ".join(errors))

    policy = policy_receipt.get("policy")
    policy_name = policy.get("name") if isinstance(policy, dict) else None
    if policy_name not in {"gain", "gain_uncertainty", "gain_uncertainty_nocost"}:
        return
    components = policy_receipt.get("components")
    errors = validate_conditioned_adjustment(
        conditioning,
        record["semantic_point"],
        evidence_run_ids=receipt_experience.get("evidence_run_ids"),
        evidence_edge_ids=receipt_experience.get("evidence_edge_ids"),
        gain_adjustment=(
            components.get("experience_gain_adjustment")
            if isinstance(components, dict)
            else None
        ),
        uncertainty_adjustment=(
            components.get("experience_uncertainty_adjustment")
            if isinstance(components, dict)
            else None
        ),
    )
    if errors:
        raise AdmissionError("invalid policy receipt: " + "; ".join(errors))


def admit_record(data: dict, request: AdmissionRequest) -> dict:
    """Validate and append one candidate, returning the new record."""
    try:
        refresh = experience_refresh_status(data)
    except ValueError as exc:
        raise AdmissionError(f"invalid experience refresh state: {exc}") from None
    if refresh["semantic_admission_blocked"]:
        raise AdmissionError(
            "stale experience: terminal DAG evidence must be refreshed before "
            "another semantic candidate is admitted; resolve any already-pending "
            "siblings first"
        )

    registry, catalog, dimension_strategy = _resolve_space(request, data)
    if get_record(data, request.run_id) is not None:
        raise AdmissionError(f"record already exists for run_id {request.run_id}")
    parent_ids = _parent_ids(request)
    semantic_point = json.loads(request.semantic_point_path.read_text())
    policy_receipt = json.loads(request.policy_receipt_path.read_text())
    record = new_record(request.run_id)
    record.update(
        kind=request.kind,
        idea=request.idea,
        change=request.change,
        source_run_ids=parent_ids,
        op=request.op,
        semantic_point=semantic_point,
        semantic_edges=[],
        policy_receipt=policy_receipt,
        candidate_name=request.candidate_name_hint,
        description=request.description or request.idea,
        metric=data["metric"],
    )
    try:
        record["semantic_edges"] = build_semantic_edges(data["records"], record)
    except SemanticEvidenceError as exc:
        raise AdmissionError(f"invalid candidate semantic contract: {exc}") from exc

    data["search_space"] = data.get("search_space") or space_receipt(registry)
    data["search_space_state"] = (
        data.get("search_space_state") or empty_search_space_state()
    )
    revision = data["search_space_state"].get("revision", 0)
    if policy_receipt.get("search_space_state_revision") != revision:
        raise AdmissionError(
            "stale policy receipt: search_space_state_revision "
            f"{policy_receipt.get('search_space_state_revision')!r} does not equal the "
            f"current search space state revision {revision}; re-propose and "
            "re-select against the current overlay before admission"
        )
    if policy_receipt.get("schema_version") not in {6, 7}:
        raise AdmissionError(
            "new candidate admission requires policy receipt schema 6 or 7 with "
            "proposal-relevant gated experience conditioning and an auditable "
            "LLM-judgment reliability prior"
        )
    _validate_experience_binding(data, record, policy_receipt, parent_ids)

    budget = policy_receipt.get("budget")
    expected_selection_index = len(data["records"]) + 1
    if (
        not isinstance(budget, dict)
        or budget.get("selection_index") != expected_selection_index
    ):
        raise AdmissionError(
            "policy receipt budget selection_index must equal the next one-based "
            f"admission index {expected_selection_index}"
        )
    data["records"].append(record)
    errors = validate_registry(
        registry,
        ledger=data,
        catalog=catalog,
        dimension_strategy=dimension_strategy,
    )
    if errors:
        data["records"].pop()
        raise AdmissionError(
            "invalid candidate semantic contract: " + "; ".join(errors)
        )
    return record
