"""Shared no-network fixtures over a frozen toy semantic search space.

Every test builds its registry, background markdown, policy receipts, and
ledger records from here so that a contract change surfaces in one place
rather than in each test's own hand-rolled copy.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from semantic_evidence import build_semantic_edges  # noqa: E402
from semantic_space import catalog_receipt, complete_point, load_catalog  # noqa: E402


SCOPE_AXES = (
    "model_families",
    "data_regimes",
    "metrics",
    "interventions",
    "evaluation_protocols",
)
TOY_SOURCE = {
    "id": "src-01",
    "type": "paper",
    "title": "Toy mechanism study",
    "url": "https://example.test/toy-mechanism-study",
    "publication_status": "preprint_only",
    "validation_status": "artifact_available",
    "studied_scope": {axis: ["*"] for axis in SCOPE_AXES},
}


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def scope(intervention: str) -> dict:
    return {
        "model_families": ["toy-models"],
        "data_regimes": ["toy-data"],
        "metrics": ["validation-loss"],
        "interventions": [intervention],
        "evaluation_protocols": ["heldout-split"],
    }


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def hypothesis(
    hypothesis_id: str, title: str, *, kind: str, intervention: str
) -> dict:
    baseline = kind == "baseline"
    return {
        "id": hypothesis_id,
        "title": title,
        "claim": f"Use {title} as one attributed semantic choice.",
        "kind": kind,
        "status": "active",
        "provenance": [
            {
                "kind": "task_contract" if baseline else "literature",
                "ref": "tasks/toy/TASK.md" if baseline else "src-01",
            }
        ],
        "claim_scope": f"Toy scope for {title}.",
        "scope": scope(intervention),
        "required_comparisons": [
            "Matched budget and split against the dimension baseline."
        ],
        "reopen_when": "A materially different mechanism or task regime is tested.",
        "literature_credibility": "unverified" if baseline else "preliminary",
        "credibility_rationale": (
            "The task contract fixes this baseline."
            if baseline
            else "One directly inspected primary source supports a local test."
        ),
        "testable_expectation": (
            "Produce a valid candidate and compare its lower-is-better score."
        ),
        "evidence": [] if baseline else [{"source_id": "src-01", "role": "supports"}],
    }


def dimension(
    catalog: dict, dimension_id: str, *hypotheses: dict, mode: str = "searchable"
) -> dict:
    entry = {item["id"]: item for item in catalog["dimensions"]}[dimension_id]
    return {
        "id": dimension_id,
        "definition": entry["definition"],
        "boundary": entry["boundary"],
        "catalog_provenance": catalog["provenance"],
        "selection_reason": f"{dimension_id} has a legal material task choice.",
        "evidence": [{"kind": "task_contract", "ref": "tasks/toy/TASK.md"}],
        "mode": mode,
        "status": "active",
        "baseline_hypothesis_id": hypotheses[0]["id"],
        "hypotheses": list(hypotheses),
    }


def fixture_registry() -> dict:
    """The canonical five-dimension toy space used across the suite."""
    catalog = load_catalog()
    dimensions = [
        dimension(
            catalog,
            "dim-data-curation",
            hypothesis(
                "hyp-data-raw",
                "all legal raw examples",
                kind="baseline",
                intervention="raw-pool",
            ),
            hypothesis(
                "hyp-data-filtered",
                "quality-filtered examples",
                kind="evidence_prior",
                intervention="filtering",
            ),
        ),
        dimension(
            catalog,
            "dim-model-architecture",
            hypothesis(
                "hyp-model-linear",
                "linear predictor",
                kind="baseline",
                intervention="linear-model",
            ),
            hypothesis(
                "hyp-model-multibranch",
                "multi-branch predictor",
                kind="evidence_prior",
                intervention="multi-branch",
            ),
        ),
        dimension(
            catalog,
            "dim-validation-selection",
            hypothesis(
                "hyp-valid-holdout",
                "single holdout",
                kind="baseline",
                intervention="holdout",
            ),
            hypothesis(
                "hyp-valid-cv",
                "cross-validation",
                kind="evidence_prior",
                intervention="cross-validation",
            ),
        ),
        dimension(
            catalog,
            "dim-ensemble",
            hypothesis(
                "hyp-ensemble-identity",
                "identity single-predictor combination",
                kind="baseline",
                intervention="identity-ensemble",
            ),
            hypothesis(
                "hyp-ensemble-stacking",
                "stacked combination",
                kind="evidence_prior",
                intervention="stacking",
            ),
        ),
        dimension(
            catalog,
            "dim-initialization-adaptation",
            hypothesis(
                "hyp-adapt-fixed",
                "task-fixed initialization",
                kind="baseline",
                intervention="fixed-initialization",
            ),
            mode="baseline_only",
        ),
    ]
    return {
        "schema_version": 3,
        "kind": "semantic_search_space",
        "space_id": "toy-p1-space",
        "catalog": catalog_receipt(catalog),
        "dimensions": dimensions,
        "relations": [
            {
                "id": "rel-ensemble-activation",
                "type": "activates",
                "status": "active",
                "provenance": [{"kind": "agent_synthesis", "ref": "toy fixture"}],
                "evidence": [],
                "when": {
                    "dimension_id": "dim-model-architecture",
                    "hypothesis_ids": ["hyp-model-multibranch"],
                },
                "target_dimension_id": "dim-ensemble",
            },
            {
                "id": "rel-stacking-requires-cv",
                "type": "requires",
                "status": "active",
                "provenance": [{"kind": "agent_synthesis", "ref": "toy fixture"}],
                "evidence": [],
                "when": {
                    "dimension_id": "dim-ensemble",
                    "hypothesis_ids": ["hyp-ensemble-stacking"],
                },
                "then": {
                    "dimension_id": "dim-validation-selection",
                    "hypothesis_ids": ["hyp-valid-cv"],
                },
            },
            {
                "id": "rel-filter-multibranch-exclusion",
                "type": "excludes",
                "status": "active",
                "provenance": [{"kind": "task_contract", "ref": "tasks/toy/TASK.md"}],
                "evidence": [],
                "members": [
                    {
                        "dimension_id": "dim-data-curation",
                        "hypothesis_ids": ["hyp-data-filtered"],
                    },
                    {
                        "dimension_id": "dim-model-architecture",
                        "hypothesis_ids": ["hyp-model-multibranch"],
                    },
                ],
            },
        ],
        "guidance": [
            {
                "id": "g-01",
                "section": "pitfall",
                "effect": "caution",
                "claim": "Filtering can remove rare legal examples in this exact regime.",
                "scope": scope("filtering"),
                "literature_credibility": "preliminary",
                "credibility_rationale": "One direct primary study; caution only.",
                "reopen_when": "Rare-example retention is measured explicitly.",
                "evidence": [{"source_id": "src-01", "role": "supports"}],
            }
        ],
        "sources": [dict(TOY_SOURCE)],
    }


def shape_registry(catalog: dict, mapping: dict, space_id: str) -> dict:
    """One toy registry per benchmark shape, built only from its ownership map.

    Dimensions follow catalog order, so two shapes differ solely in display
    names and ``space_id``; any behavioral divergence between them proves an
    estimator-specific branch leaked into the shared helpers.
    """
    owned = set(mapping.values())
    dimensions = [
        dimension(
            catalog,
            entry["id"],
            hypothesis(
                f"hyp-{entry['id'][4:]}-base",
                f"{entry['id'][4:]} baseline choice",
                kind="baseline",
                intervention=f"{entry['id'][4:]}-base",
            ),
            hypothesis(
                f"hyp-{entry['id'][4:]}-variant",
                f"{entry['id'][4:]} variant choice",
                kind="evidence_prior",
                intervention=f"{entry['id'][4:]}-variant",
            ),
        )
        for entry in catalog["dimensions"]
        if entry["id"] in owned
    ]
    return {
        "schema_version": 3,
        "kind": "semantic_search_space",
        "space_id": space_id,
        "catalog": catalog_receipt(catalog),
        "dimensions": dimensions,
        "relations": [],
        "guidance": [],
        "sources": [dict(TOY_SOURCE)],
    }


def background_text(registry: dict) -> str:
    """Render the registry as a background.md the real loader accepts."""
    lines = [
        "# Background — toy fixture",
        "",
        "## Dimension coverage",
        "",
        "| Dimension | Mode | Baseline | Hypotheses |",
        "|---|---|---|---|",
    ]
    for item in registry["dimensions"]:
        ids = ", ".join(f"`{h['id']}`" for h in item["hypotheses"])
        lines.append(
            f"| `{item['id']}` | {item['mode']} | "
            f"`{item['baseline_hypothesis_id']}` | {ids} |"
        )
    lines += ["", "## Dimensions", ""]
    for item in registry["dimensions"]:
        lines.append(f"### `{item['id']}`")
        lines += [f"- `{h['id']}` — {h['title']}" for h in item["hypotheses"]]
        lines.append("")
    lines += ["## Relations", ""]
    lines += [f"- `{r['id']}` — {r['type']}" for r in registry["relations"]]
    lines += [
        "",
        "## Pitfalls",
        "",
        "- `g-01` — Filtering can remove rare legal examples in this exact regime.",
        "",
        "## Search space registry",
        "```json",
        json.dumps(registry, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# receipts and records
# --------------------------------------------------------------------------


def policy_receipt(
    op: str,
    parents: list[str],
    point: dict,
    *,
    state_revision: int = 0,
    selection_index: int = 1,
    selected_lane: str = "active",
    deprioritized_interval: int = 5,
    schema_version: int = 6,
) -> dict:
    """A coverage-policy selection receipt with a consistent budget lane.

    Schema 6 pins the LLM weight and an empty conditioning receipt.
    """
    scheduled_lane = (
        "deprioritized"
        if selection_index % deprioritized_interval == 0
        else "active"
    )
    receipt = {
        "schema_version": schema_version,
        "space_revision": point["space_revision"],
        "search_space_state_revision": state_revision,
        "proposal_set_revision": "sha256:" + "0" * 64,
        "policy": {
            "name": "coverage",
            "config": {
                "coverage_weight": 0.1,
                "cost_weight": 0.2,
                "uncertainty_weight": 0.5,
                "deprioritized_budget_interval": deprioritized_interval,
            },
        },
        "action": {"op": op, "parents": parents},
        "selected_point_id": point["point_id"],
        "components": {
            "coverage": 1.0,
            "prior_gain": None,
            "experience_gain_adjustment": None,
            "predicted_gain": None,
            "prior_uncertainty": None,
            "experience_uncertainty_adjustment": None,
            "uncertainty": None,
            "cost": None,
        },
        "acquisition_score": 1.0,
        "evidence": [],
        "experience": {
            "generation": None,
            "updated_at_run": None,
            "revision": None,
            "evidence_run_ids": [],
            "evidence_edge_ids": [],
            "rationale": "coverage policy does not use model-scored experience",
        },
        "budget": {
            "selection_index": selection_index,
            "deprioritized_interval": deprioritized_interval,
            "scheduled_lane": scheduled_lane,
            "selected_lane": selected_lane,
            "fallback": {
                ("active", "active"): "none",
                ("deprioritized", "deprioritized"): "none",
                ("active", "deprioritized"): "no_active_proposals",
                ("deprioritized", "active"): "no_deprioritized_proposals",
            }[(scheduled_lane, selected_lane)],
            "base_rank": 1,
        },
        "ranked_point_ids": [point["point_id"]],
    }
    if schema_version >= 6:
        receipt["policy"]["config"]["llm_intelligence_score"] = 100.0
        receipt["components"]["llm_judgment_weight"] = None
        receipt["experience"]["conditioning"] = []
    return receipt


def attach_matched_transfer(
    parent: dict,
    child: dict,
    *,
    control_score: float | None = None,
    reset: bool = False,
) -> None:
    """Attach an internally complete paired-control tuner receipt to ``child``.

    The receipt carries a matched inherited control plus a single-parameter
    semantic treatment, which is what makes the child's edge a *direct*
    comparator. ``reset=True`` instead models a parameter-kind change, whose
    projection resets rather than copies and so is never a direct comparator.
    """
    parent_score = float(parent["final_best_score"])
    child_score = (
        float(child["final_best_score"])
        if control_score is None
        else float(control_score)
    )
    parent_params = {"shared": 1.0}
    child_defaults = {"shared": 2.0 if reset else 1.0}
    projected = dict(child_defaults if reset else parent_params)
    parent_schema = {"shared": "float"}
    child_schema = {"shared": "int" if reset else "float"}
    treatment = {**projected, "shared": 3 if reset else 2.0}

    parent["applied_incumbent"] = {
        "schema_version": 1,
        "source": "applied_phase_a",
        "score": parent_score,
        "params": parent_params,
        "param_schema": parent_schema,
        "entrypoint_sha256": "sha256:" + "3" * 64,
        "tune_report_sha256": "sha256:" + "4" * 64,
    }
    change = next(
        (
            item
            for edge in child.get("semantic_edges", [])
            if isinstance(edge, dict)
            for item in edge.get("changes", [])
            if isinstance(item, dict)
        ),
        {},
    )
    receipt = {
        "schema_version": 2,
        "kind": "primary_parent_parameter_transfer",
        "candidate": {
            "run_id": child["run_id"],
            "path": f"candidates/{child['run_id']}/train.py",
            "brief_path": f"candidates/{child['run_id']}/_candidate_brief.json",
            "brief_sha256": "sha256:" + "1" * 64,
            "structure_snapshot": "fixture",
            "param_schema": child_schema,
            "defaults": child_defaults,
        },
        "primary_parent": {
            "run_id": parent["run_id"],
            "path": f"candidates/{parent['run_id']}/train.py",
            "entrypoint_sha256": "sha256:" + "3" * 64,
            "tune_report_path": f"candidates/{parent['run_id']}/tune_report.json",
            "tune_report_sha256": "sha256:" + "4" * 64,
            "ledger_path": "ledger.json",
            "ledger_record_sha256": json_sha256(parent),
            "incumbent_source": "applied_phase_a",
            "incumbent_score": parent_score,
            "incumbent_params": parent_params,
            "param_schema": parent_schema,
        },
        "projection": {
            "params": projected,
            "params_sha256": json_sha256(projected),
            "copied": [] if reset else [{"key": "shared", "value": 1.0}],
            "reset": (
                [
                    {
                        "key": "shared",
                        "parent_value": 1.0,
                        "child_value": 2.0,
                        "reason": "kind_changed",
                    }
                ]
                if reset
                else []
            ),
            "new": [],
            "dropped": [],
        },
        "semantic_control": {
            "status": "paired",
            "method": "same_child_code_single_parameter",
            "parameter": "shared",
            "target_dimension_id": change.get("dimension_id", "fixture-dimension"),
            "target_hypothesis_id": change.get(
                "to_hypothesis_id", "fixture-hypothesis"
            ),
            "control_params_sha256": json_sha256(projected),
            "treatment_params_sha256": json_sha256(treatment),
        },
    }
    receipt["receipt_sha256"] = json_sha256(receipt)

    child["op"] = child.get("op") or "improve"
    policy = child.setdefault("policy_receipt", {"schema_version": 6})
    if isinstance(policy, dict):
        # Fill in what a schema-6 receipt needs, but never downgrade a real one:
        # an attempt-family policy is only valid at schema 7.
        policy.setdefault("schema_version", 6)
        config = policy.get("policy", {}).get("config")
        if isinstance(config, dict):
            config.setdefault("llm_intelligence_score", 100.0)
        components = policy.get("components")
        if isinstance(components, dict):
            components.setdefault(
                "llm_judgment_weight",
                None if policy.get("policy", {}).get("name") == "coverage" else 1.0,
            )
        experience = policy.get("experience")
        if isinstance(experience, dict):
            experience.setdefault("conditioning", [])
    child["parameter_transfer"] = {
        "receipt": receipt,
        "inherited_control": {
            "warm_config_index": 0,
            "selected": True,
            "primary_parent_run_id": parent["run_id"],
            "parent_incumbent_score": parent_score,
            "params_sha256": receipt["projection"]["params_sha256"],
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "warm_start_observations": [
            {
                "params": projected,
                "score": parent_score,
                "proposed_index": 0,
                "role": "inherited_control",
                "parameter_transfer_receipt_sha256": receipt["receipt_sha256"],
                "params_sha256": receipt["projection"]["params_sha256"],
            },
            {
                "params": treatment,
                "score": child_score,
                "proposed_index": 1,
                "role": "semantic_treatment",
                "parameter_transfer_receipt_sha256": receipt["receipt_sha256"],
                "params_sha256": receipt["semantic_control"][
                    "treatment_params_sha256"
                ],
            },
        ],
    }


def record(
    run_id: str,
    op: str,
    parents: list[str],
    point: dict,
    *,
    score: float,
    status: str,
    prior_records: list[dict] | None = None,
    depth: str = "tuned",
) -> dict:
    """A full ledger record: attribution, ancestry, receipts, and observation."""
    entry = {
        "run_id": run_id,
        "kind": "optimization",
        "op": op,
        "source_run_ids": parents,
        "idea": f"Complete concrete implementation {run_id} at the attributed point.",
        "change": f"Concrete implementation process for {op} run {run_id}.",
        "candidate_name": f"fixture_{run_id}",
        "description": f"Fixture candidate {run_id}.",
        "semantic_point": point,
        "policy_receipt": policy_receipt(
            op, parents, point, selection_index=int(run_id) + 1
        ),
        "status": status,
        "final_best_score": score,
        "evaluation_depth": depth,
    }
    entry["semantic_edges"] = build_semantic_edges(prior_records or [], entry)
    if parents and math.isfinite(float(score)):
        parent = next(
            item for item in prior_records or [] if item.get("run_id") == parents[0]
        )
        attach_matched_transfer(parent, entry)
    return entry


def belief_ledger(registry: dict) -> dict:
    """Two independent matched comparisons against ``hyp-data-filtered``.

    This is the minimum shape that gives a target *comparator coverage* — two
    direct non-crash edges — so belief and pruning gates can be exercised.
    """
    baseline = complete_point(registry)
    filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    records = [
        {
            "run_id": "000", "source_run_ids": [], "semantic_point": baseline,
            "semantic_edges": [], "status": "keep", "final_best_score": 0.40,
            "evaluation_depth": "tuned", "dag_revision": 1,
        },
        {
            "run_id": "001", "source_run_ids": ["000"], "semantic_point": filtered,
            "status": "discard", "final_best_score": 0.50,
            "evaluation_depth": "tuned", "dag_revision": 2,
        },
        {
            "run_id": "002", "source_run_ids": [], "semantic_point": baseline,
            "semantic_edges": [], "status": "keep", "final_best_score": 0.41,
            "evaluation_depth": "tuned", "dag_revision": 3,
        },
        {
            "run_id": "003", "source_run_ids": ["002"], "semantic_point": filtered,
            "status": "discard", "final_best_score": 0.52,
            "evaluation_depth": "tuned", "dag_revision": 4,
        },
    ]
    records[1]["semantic_edges"] = build_semantic_edges(records[:1], records[1])
    records[3]["semantic_edges"] = build_semantic_edges(records[:3], records[3])
    attach_matched_transfer(records[0], records[1])
    attach_matched_transfer(records[2], records[3])
    return {"records": records, "dag_revision": 4}
