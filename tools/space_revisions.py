"""Same-run, append-only semantic registry revisions.

The driver owns admission and ledger mutation. Publication here atomically binds
an admitted review, its new registry/catalog and its pending probe. background.md
and historical points remain untouched. Runtime consumers use load_registry and
pass load_registry_history to historical views/validation.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from semantic_evidence import LIFECYCLE_TERMINAL_STATUSES
from semantic_space import (
    SemanticSpaceError, catalog_receipt, complete_point, dimension_map,
    hypothesis_map, load_catalog, selected_assignments, space_receipt,
    space_revision,
)

STATE_PATH = Path(".semantic/space-revisions.json")


def load_revision_state(background: Path) -> dict | None:
    path = Path(background).parent / STATE_PATH
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise SemanticSpaceError(f"invalid space revision state: {path}")
    versions = state.get("versions")
    if not isinstance(versions, list) or not versions:
        raise SemanticSpaceError("space revision state needs non-empty versions")
    for version in versions:
        if (not isinstance(version, dict) or not isinstance(version.get("registry"), dict)
                or version.get("space") != space_receipt(version["registry"])
                or version["registry"].get("catalog") != catalog_receipt(version["catalog"])):
            raise SemanticSpaceError("space revision snapshot does not match its receipt")
    if state.get("current_revision") != versions[-1]["space"]["space_revision"]:
        raise SemanticSpaceError("current space revision does not match the published tail")
    return state


def load_registry_history(background: Path) -> dict[str, dict]:
    state = load_revision_state(background)
    if state is None:
        return {}
    return {v["space"]["space_revision"]: v["registry"] for v in state["versions"]}


def apply_expansion(
    registry: dict, delta: dict, *, catalog: dict | None = None,
    dimension_strategy: str = "catalog_subset", records: list[dict] | None = None,
) -> tuple[dict, dict]:
    """Build and validate an additive registry; no I/O or budget decisions."""
    from background_contract import validate_registry

    fields = {"hypotheses", "dimensions", "relations", "sources", "guidance", "catalog_dimensions"}
    if not isinstance(delta, dict) or set(delta) - fields:
        raise SemanticSpaceError("invalid expansion delta fields")
    new = copy.deepcopy(registry)
    catalog = copy.deepcopy(load_catalog() if catalog is None else catalog)
    catalog_additions = delta.get("catalog_dimensions", [])
    if catalog_additions:
        if dimension_strategy != "llm_induced":
            raise SemanticSpaceError("catalog_subset cannot extend the dimension catalog")
        catalog["dimensions"].extend(copy.deepcopy(catalog_additions))
        new["catalog"] = catalog_receipt(catalog)
    additions = delta.get("hypotheses", {})
    if not isinstance(additions, dict):
        raise SemanticSpaceError("delta.hypotheses must map existing dimension ids to lists")
    dimensions = dimension_map(new)
    for dimension_id, hypotheses in additions.items():
        if dimension_id not in dimensions or dimensions[dimension_id]["mode"] != "searchable":
            raise SemanticSpaceError(f"cannot add hypotheses to {dimension_id}")
        if not isinstance(hypotheses, list):
            raise SemanticSpaceError("hypothesis additions must be lists")
        dimensions[dimension_id]["hypotheses"].extend(copy.deepcopy(hypotheses))
    for field in ("dimensions", "relations", "sources", "guidance"):
        items = delta.get(field, [])
        if not isinstance(items, list):
            raise SemanticSpaceError(f"delta.{field} must be a list")
        new[field].extend(copy.deepcopy(items))
    errors = validate_registry(new, catalog=catalog, dimension_strategy=dimension_strategy)
    if errors:
        raise SemanticSpaceError("invalid expanded registry: " + "; ".join(errors))
    new_dimensions = set(dimension_map(new)) - set(dimension_map(registry))
    new_hypotheses = set(hypothesis_map(new)) - set(hypothesis_map(registry))
    if not new_dimensions and not new_hypotheses:
        raise SemanticSpaceError("expansion must add a dimension or hypothesis")
    # Relations cannot restrict the old product space. Activation of an existing
    # dimension changes its meaning even if the trigger happens to be new.
    for relation in delta.get("relations", []):
        kind = relation["type"]
        if kind == "activates":
            safe = relation["target_dimension_id"] in new_dimensions
        elif kind == "requires":
            when = relation["when"]
            safe = (when["dimension_id"] in new_dimensions or
                    set(when["hypothesis_ids"]).issubset(new_hypotheses))
        else:  # excludes: at least one member must be impossible in the old space
            safe = any(m["dimension_id"] in new_dimensions or
                       set(m["hypothesis_ids"]).issubset(new_hypotheses)
                       for m in relation["members"])
        if not safe:
            raise SemanticSpaceError(f"relation {relation['id']} retroactively constrains old choices")
    # Keep explicit historical choices: completion is a proposal, never a
    # rewrite of the historical point or evidence that a new baseline was tried.
    for record in records or []:
        old_choices = selected_assignments(record.get("semantic_point") or {})
        projected = complete_point(new, old_choices)
        if projected is None or any(selected_assignments(projected).get(k) != v
                                    for k, v in old_choices.items()):
            raise SemanticSpaceError(f"expansion invalidates historical choices for {record.get('run_id')}")
    return new, catalog


def publish_expansion(background: Path, review: dict, *, ledger: dict, admission: dict) -> dict:
    """Called by the serial driver *after* reserving review + complete next slate.

    admission is the budget owner's receipt (requires reservation_id); this
    helper does not certify budgets or mutate ledger.json. The integrator must
    update ledger.search_space through tools/ledger.py before further work.
    """
    from background_contract import load_registry, validate_registry
    from semantic_space import resolve_dimension_catalog, resolve_dimension_strategy
    from space_review import validate_review

    background = Path(background)
    if not isinstance(admission, dict) or not isinstance(admission.get("reservation_id"), str) or not admission["reservation_id"].strip():
        raise SemanticSpaceError("publication requires a budget reservation receipt")
    current = load_registry(background)
    strategy = resolve_dimension_strategy(background)
    catalog = resolve_dimension_catalog(background)
    errors = validate_review(review, current, ledger=ledger, catalog=catalog, dimension_strategy=strategy,
                             registry_history=load_registry_history(background))
    if errors or review.get("decision") != "expand":
        raise SemanticSpaceError("invalid expansion review: " + "; ".join(errors))
    new, new_catalog = apply_expansion(current, review["delta"], catalog=catalog,
                                     dimension_strategy=strategy, records=ledger.get("records", []))
    manifest_path = background.parent / "background_retrieval.json"
    manifest = json.loads(manifest_path.read_text())
    errors = validate_registry(new, catalog=new_catalog, dimension_strategy=strategy,
                               retrieval_manifest=manifest, manifest_dir=manifest_path.parent,
                               manifest_path=manifest_path, number_gate=True)
    if errors:
        raise SemanticSpaceError("expansion evidence failed: " + "; ".join(errors))
    state = load_revision_state(background)
    if state is None:
        state = {"schema_version": 1, "versions": [{
            "space": space_receipt(current), "registry": current, "catalog": catalog,
        }]}
    elif state.get("pending_probe") is not None:
        from space_expansion import pending_expansion
        previous = state["pending_probe"]["point"]["point_id"]
        if pending_expansion(background.parent) is not None and not any(r.get("semantic_point", {}).get("point_id") == previous and
                   r.get("status") in LIFECYCLE_TERMINAL_STATUSES
                   for r in ledger.get("records", [])):
            raise SemanticSpaceError("previous probe is still pending")
    state["versions"].append({"space": space_receipt(new), "registry": new, "catalog": new_catalog,
                              "review": copy.deepcopy(review), "admission": copy.deepcopy(admission)})
    state["current_revision"] = space_revision(new)
    state["pending_probe"] = copy.deepcopy(review["probe"])
    path = background.parent / STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            json.dump(state, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return {"space": space_receipt(new), "pending_probe": state["pending_probe"],
            "state_path": str(path)}
