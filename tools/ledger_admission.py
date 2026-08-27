"""Candidate-admission rules for the run ledger.

Admission validates the frozen semantic space and mutates only the supplied
in-memory ledger.  The CLI facade in ``tools/ledger.py`` remains responsible
for loading and committing ``ledger.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from background_contract import (
    ATTEMPT_POLICY_NAMES,
    load_registry,
    validate_background_markdown,
    validate_registry,
)
from ledger_core import (
    experience_receipt,
    experience_refresh_status,
    get_record,
    new_record,
    records_prefix_digest,
    search_space_state_revision,
)
from run_cfg import load_run_cfg
from search_space_state import empty_search_space_state
from semantic_attempts import (
    DEFAULT_ATTEMPT_CONFIG,
    attempt_adjustment,
    classify_attempts,
)
from semantic_evidence import (
    SemanticEvidenceError,
    acquisition_conditioning,
    acquisition_target_relations,
    build_semantic_edges,
    mechanical_gain_directions,
    validate_conditioned_adjustment,
    validate_conditioning_against_ledger,
)
from semantic_routes import (
    RouteError,
    build_route_memory,
    is_not_applicable,
    route_arm_active,
    route_config,
    validate_route_provenance,
)
from semantic_space import (
    SemanticSpaceError,
    digest,
    point_id,
    resolve_dimension_catalog,
    resolve_dimension_strategy,
    space_receipt,
)
from slate import replay_aggregation, verify_manifest


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
    role: str | None = None
    description: str | None = None
    route_provenance_path: Path | None = None
    task_config: dict | None = None


def _resolve_space(
    request: "AdmissionRequest | SlateAdmissionRequest", data: dict
) -> tuple[dict, dict, str]:
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


def _validate_route_provenance(
    data: dict, request: AdmissionRequest, semantic_point: dict
) -> dict | None:
    """Resolve the route arm and validate the candidate's planned provenance.

    When the arm is on, a missing or inconsistent route record is an admission
    failure rather than a silent fall back to the no-memory arm — otherwise a
    route experiment could report the wrong arm for part of its own run.
    """
    try:
        cfg = route_config(load_run_cfg(request.background_path, "semantic_search"))
    except RouteError as exc:
        raise AdmissionError(str(exc)) from None
    active = route_arm_active(cfg)
    if request.route_provenance_path is None:
        if active:
            raise AdmissionError(
                "the configured route arm requires --route-provenance "
                f"(n_route_sketches={cfg['n_route_sketches']}, "
                f"route_memory={cfg['route_memory']})"
            )
        return None
    provenance = json.loads(request.route_provenance_path.read_text())
    if is_not_applicable(provenance):
        # Only a candidate nobody generated may opt out, and only a task that
        # actually ships a baseline entrypoint has one.
        seed = (request.task_config or {}).get("seed")
        provided = seed.get("provided") if isinstance(seed, dict) else None
        if not provided or data.get("records") or request.op != "fresh":
            raise AdmissionError(
                "route provenance may be marked not_applicable only for a "
                "task-provided baseline admitted as the run's first record"
            )
    memory = build_route_memory(data, semantic_point, request.op, cfg)
    errors = validate_route_provenance(provenance, memory=memory)
    if errors:
        raise AdmissionError("invalid route provenance: " + "; ".join(errors))
    return provenance


def _validate_attempt_binding(data: dict, record: dict, policy_receipt: dict) -> None:
    """Rebind the receipt's attempt channel to the pre-admission ledger.

    The receipt's shape is already validated upstream; what is not otherwise
    checkable is whether the numbers describe *this* ledger.  Recomputing the
    selected point's adjustment here makes a hand-edited counts / run_ids /
    screen / crash statistic an admission failure instead of a persisted claim.
    """
    policy = policy_receipt.get("policy")
    if not isinstance(policy, dict) or policy.get("name") not in ATTEMPT_POLICY_NAMES:
        return
    config = policy.get("config")
    if not isinstance(config, dict) or not set(DEFAULT_ATTEMPT_CONFIG) <= set(config):
        raise AdmissionError(
            "invalid policy receipt: an attempt policy must record the attempt "
            "configuration it scored with"
        )
    components = policy_receipt.get("components")
    if not isinstance(components, dict):
        raise AdmissionError("invalid policy receipt: components must be an object")
    rows = classify_attempts(
        data, noise_threshold=float(config["attempt_noise_threshold"])
    )
    expected_prior, expected_detail = attempt_adjustment(
        rows,
        point_id=point_id(record["semantic_point"]),
        op=record["op"],
        cfg=config,
    )
    if components.get("attempts") != expected_detail:
        raise AdmissionError(
            "invalid policy receipt: components.attempts does not equal the "
            "attempt statistic recomputed from the ledger's own observations"
        )
    if components.get("attempt_prior") != expected_prior:
        raise AdmissionError(
            "invalid policy receipt: components.attempt_prior "
            f"{components.get('attempt_prior')!r} does not equal the recomputed "
            f"adjustment {expected_prior!r}"
        )


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
        # Recomputed against the pre-admission ledger, so the memory the
        # generator was shown is exactly the memory validated here.
        route_provenance=_validate_route_provenance(data, request, semantic_point),
        role=request.role,
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
    if policy_receipt.get("schema_version") not in {6, 7, 8}:
        raise AdmissionError(
            "new candidate admission requires policy receipt schema 6, 7, or 8 with "
            "proposal-relevant gated experience conditioning and an auditable "
            "LLM-judgment reliability prior"
        )
    _validate_experience_binding(data, record, policy_receipt, parent_ids)
    _validate_attempt_binding(data, record, policy_receipt)

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


# ---------- judged-slate atomic batch admission (policy receipt schema 8) ----------


@dataclass(frozen=True)
class SlateAdmissionRequest:
    """One judged-slate generation's atomic admission input.

    ``manifest_path`` is the immutable ``generation.json``; each seat's
    ``idea``/``change``/``candidate_name`` (and optional ``route_provenance``)
    come from ``plans/slot-N.json``.  The point/op/parents and the whole
    schema-8 policy receipt are derived from the manifest, never from the
    plans.
    """

    background_path: Path
    catalog_path: Path | None
    manifest_path: Path
    plans_dir: Path
    run_dir: Path


def _load_json_object(path: Path, what: str) -> dict:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot read {what} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"{path}: {what} must be a JSON object")
    return value


def _load_slate_plan(request: SlateAdmissionRequest, slot: int) -> dict:
    plan = _load_json_object(
        Path(request.plans_dir) / f"slot-{slot}.json", "slate plan"
    )
    if plan.get("slot") != slot:
        raise AdmissionError(f"slate plan slot-{slot}.json does not belong to slot {slot}")
    for field in ("idea", "change", "candidate_name"):
        if not isinstance(plan.get(field), str) or not plan[field].strip():
            raise AdmissionError(f"slate plan slot {slot} needs a non-empty {field}")
    return plan


def _validate_judge_binding(
    data: dict, request: SlateAdmissionRequest, registry: dict
) -> tuple[dict, str]:
    """Bind the generation manifest to this run and the pre-admission ledger.

    Returns ``(manifest, run-relative manifest path)``.
    Everything the seats will claim is checked here, before any record is
    constructed: the manifest's location and content id, the artifact digest
    chain, the judge replay, and the generation-start ledger snapshot.
    """
    manifest_path = Path(request.manifest_path)
    manifest = _load_json_object(manifest_path, "slate manifest")
    if manifest.get("schema_version") != 1:
        raise AdmissionError("slate manifest schema_version must be 1")
    gen_no = manifest.get("gen_no")
    if not isinstance(gen_no, int) or isinstance(gen_no, bool) or gen_no < 1:
        raise AdmissionError("slate manifest gen_no must be a positive integer")
    policy = manifest.get("policy")
    if not isinstance(policy, dict) or policy.get("name") != "judged_slate":
        raise AdmissionError("slate manifest policy.name must be judged_slate")
    slate_slots = manifest.get("slate")
    if not isinstance(slate_slots, list) or not slate_slots:
        raise AdmissionError("slate manifest slate must be a non-empty list")
    aggregation = manifest.get("aggregation")
    if not isinstance(aggregation, dict) or not isinstance(aggregation.get("path"), str):
        raise AdmissionError("slate manifest aggregation must record the aggregation path")
    budget = manifest.get("budget")
    admission_cap = budget.get("admission_cap") if isinstance(budget, dict) else None
    if admission_cap is not None and (
        not isinstance(admission_cap, int)
        or isinstance(admission_cap, bool)
        or admission_cap < len(slate_slots)
    ):
        raise AdmissionError(
            "slate manifest budget.admission_cap must be null or cover the slate"
        )
    snapshot = manifest.get("ledger_snapshot")
    if not isinstance(snapshot, dict):
        raise AdmissionError("slate manifest ledger_snapshot must be an object")
    revisions = manifest.get("proposal_set_revisions")
    if not isinstance(revisions, dict):
        raise AdmissionError("slate manifest proposal_set_revisions must be an object")

    # The manifest is the generation commit point: it lives at the fixed
    # run-relative generation path.
    manifest_rel = f".semantic/gen-{gen_no:04d}/generation.json"
    resolved = (Path(request.run_dir) / manifest_rel).resolve()
    if resolved != manifest_path.resolve():
        raise AdmissionError(
            f"slate manifest must live at {manifest_rel} under the run directory"
        )

    run_ids = [
        slot.get("run_id") if isinstance(slot, dict) else None for slot in slate_slots
    ]
    if manifest.get("reserved_run_ids") != run_ids:
        raise AdmissionError(
            "slate manifest reserved_run_ids must equal the slate's run ids"
        )
    if len(set(run_ids)) != len(run_ids) or any(
        not isinstance(run_id, str) or not run_id.isdigit() for run_id in run_ids
    ):
        raise AdmissionError("slate run ids must be distinct numeric strings")
    for run_id in run_ids:
        if get_record(data, run_id) is not None:
            raise AdmissionError(f"record already exists for run_id {run_id}")
    for index, slot in enumerate(slate_slots):
        if not isinstance(slot, dict) or slot.get("slot") != index:
            raise AdmissionError(f"slate manifest slot {index} is missing or misnumbered")
        carrier = slot.get("carrier")
        op = carrier.get("op") if isinstance(carrier, dict) else None
        expected = {"fresh": 0, "improve": 1, "crossover": 2}.get(op)
        parents = carrier.get("parents") if isinstance(carrier, dict) else None
        if (
            expected is None
            or not isinstance(parents, list)
            or len(parents) != expected
            or len(set(parents)) != len(parents)
            or any(
                not isinstance(parent, str) or not parent.isdigit()
                for parent in parents
            )
        ):
            raise AdmissionError(
                f"slate slot {index} carrier has an invalid op/parents shape"
            )
        if not isinstance(slot.get("point"), dict):
            raise AdmissionError(f"slate slot {index} must carry the full point")
        if slot.get("point_id") != point_id(slot["point"]):
            raise AdmissionError(f"slate slot {index} point_id does not match its point")
        if carrier.get("proposal_set_revision") != revisions.get(carrier.get("lane_id")):
            raise AdmissionError(
                f"slate slot {index} carrier proposal revision is not one of the "
                "manifest's recorded proposal sets"
            )

    gen_dir = manifest_path.parent
    pool_doc = _load_json_object(gen_dir / "pool.json", "slate pool")
    context_doc = _load_json_object(gen_dir / "context.json", "slate context")
    judge_doc = _load_json_object(gen_dir / "judge.json", "slate judge")
    errors = verify_manifest(manifest, pool_doc, context_doc, judge_doc)
    if errors:
        raise AdmissionError(
            "invalid judged-slate generation artifacts: " + "; ".join(errors)
        )
    if pool_doc.get("space") != space_receipt(registry):
        raise AdmissionError(
            "the slate generation was constructed against a different search space"
        )
    errors = replay_aggregation(pool_doc, judge_doc)
    if errors:
        raise AdmissionError("; ".join(errors))

    # The generation-start snapshot must still be the current pre-admission
    # ledger: nothing may have moved between the manifest and the admission.
    records = data.get("records", [])
    dag_revision = int(data.get("dag_revision", 0) or 0)
    if snapshot.get("record_count") != len(records):
        raise AdmissionError(
            "slate manifest ledger_snapshot.record_count does not match the "
            "pre-admission ledger"
        )
    if snapshot.get("records_digest") != records_prefix_digest(records):
        raise AdmissionError(
            "slate manifest records_digest does not match the pre-admission "
            "ledger prefix"
        )
    if snapshot.get("dag_revision") != dag_revision:
        raise AdmissionError(
            "slate manifest dag_revision does not match the pre-admission ledger"
        )
    if snapshot.get("search_space_state_revision") != search_space_state_revision(data):
        raise AdmissionError(
            "slate manifest search_space_state_revision does not match the "
            "pre-admission ledger"
        )
    if snapshot.get("experience") != experience_receipt(data):
        raise AdmissionError(
            "slate manifest experience snapshot does not match the pre-admission "
            "ledger"
        )
    return manifest, manifest_rel


def _slate_route_provenance(
    data: dict, route_cfg: dict, route_active: bool, slot: dict, plan: dict
) -> dict | None:
    """Validate one seat's planned route provenance against the shared prefix."""
    provenance = plan.get("route_provenance")
    if provenance is None:
        if route_active:
            raise AdmissionError(
                "the configured route arm requires route_provenance in each "
                "slate plan"
            )
        return None
    if is_not_applicable(provenance):
        raise AdmissionError(
            "a judged-slate candidate is never the task-provided baseline; "
            "route provenance cannot be not_applicable"
        )
    carrier = slot["carrier"]
    memory = build_route_memory(data, slot["point"], carrier["op"], route_cfg)
    errors = validate_route_provenance(provenance, memory=memory)
    if errors:
        raise AdmissionError("invalid route provenance: " + "; ".join(errors))
    return provenance


def admit_slate_atomic(data: dict, request: SlateAdmissionRequest) -> list[dict]:
    """Validate and append one judged-slate generation's slate as one batch.

    Every seat is constructed against the same pre-admission ledger prefix —
    route memory, semantic edges, and the experience binding never see a
    sibling seat — then the batch is appended once and the whole ledger is
    revalidated.  Any failure rolls the in-memory ledger back; the caller must
    not persist after an ``AdmissionError``.
    """
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
    data["search_space"] = data.get("search_space") or space_receipt(registry)
    data["search_space_state"] = (
        data.get("search_space_state") or empty_search_space_state()
    )
    records = data["records"]
    pre_count = len(records)

    manifest, manifest_rel = _validate_judge_binding(
        data, request, registry
    )
    slate_slots = manifest["slate"]
    admission_cap = manifest["budget"]["admission_cap"]
    plans = [_load_slate_plan(request, slot["slot"]) for slot in slate_slots]
    try:
        route_cfg = route_config(
            load_run_cfg(request.background_path, "semantic_search")
        )
    except RouteError as exc:
        raise AdmissionError(str(exc)) from None
    route_active = route_arm_active(route_cfg)

    space = space_receipt(registry)
    state_revision = search_space_state_revision(data)
    experience = experience_receipt(data)
    admitted = []
    for slot, plan in zip(slate_slots, plans):
        index = slot["slot"]
        carrier = slot["carrier"]
        op = carrier["op"]
        parents = [str(parent) for parent in carrier["parents"]]
        receipt = {
            "schema_version": 8,
            "space": space,
            "search_space_state_revision": state_revision,
            "policy": {
                "name": "judged_slate",
                "config": dict(manifest["policy"].get("config") or {}),
            },
            "generation_id": manifest["generation_id"],
            "judge": {
                "manifest_path": manifest_rel,
                "slate_index": index,
                "candidate_id": slot["candidate_id"],
                "aggregation": manifest["aggregation"]["path"],
            },
            "carrier_proposal_set_revision": carrier["proposal_set_revision"],
            "budget": {
                "selection_index": pre_count + index + 1,
                "admission_cap": admission_cap,
            },
            "experience": dict(experience),
        }
        record = new_record(slot["run_id"])
        record.update(
            kind="optimization",
            idea=plan["idea"],
            change=plan["change"],
            source_run_ids=parents,
            op=op,
            semantic_point=slot["point"],
            semantic_edges=[],
            policy_receipt=receipt,
            # Recomputed against the pre-admission ledger, so the memory the
            # plan writer was shown is exactly the memory validated here.
            route_provenance=_slate_route_provenance(
                data, route_cfg, route_active, slot, plan
            ),
            role=None,
            candidate_name=plan["candidate_name"],
            description=plan["idea"],
            metric=data["metric"],
        )
        try:
            record["semantic_edges"] = build_semantic_edges(records, record)
        except SemanticEvidenceError as exc:
            raise AdmissionError(f"invalid candidate semantic contract: {exc}") from exc
        _validate_experience_binding(data, record, receipt, parents)
        _validate_attempt_binding(data, record, receipt)
        admitted.append(record)

    records.extend(admitted)
    errors = validate_registry(
        registry,
        ledger=data,
        catalog=catalog,
        dimension_strategy=dimension_strategy,
    )
    if errors:
        del records[pre_count:]
        raise AdmissionError("invalid judged-slate batch: " + "; ".join(errors))
    return admitted
