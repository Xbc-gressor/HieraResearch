#!/usr/bin/env python3
"""Focused deterministic checks for the P2 semantic-search-space contract."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from background_contract import (
    ContractError,
    derive_hypothesis_selection,
    load_registry,
    render_space,
    scope_relation,
    validate_background_markdown,
    validate_experience,
    validate_registry,
)
from search_backends import add_visit, new_manifest
from search_space_state import (
    compose_effective_selection,
    derive_experience_transitions,
    empty_search_space_state,
    replay_search_space_state,
)
from semantic_evidence import build_semantic_edges, comparator_coverage
from semantic_search import build_proposal_set, select_proposal, validate_proposal_set
from semantic_space import (
    catalog_receipt,
    complete_point,
    derive_semantic_lineage,
    digest,
    load_catalog,
    point_id,
    selected_assignments,
    space_receipt,
    space_revision,
    validate_point,
)


ROOT = Path(__file__).resolve().parents[1]


def _scope(intervention: str) -> dict:
    return {
        "model_families": ["toy-models"],
        "data_regimes": ["toy-data"],
        "metrics": ["validation-loss"],
        "interventions": [intervention],
        "evaluation_protocols": ["heldout-split"],
    }


def _hypothesis(
    hypothesis_id: str,
    title: str,
    *,
    kind: str,
    intervention: str,
    probe_for: list[str] | None = None,
) -> dict:
    baseline = kind == "baseline"
    value = {
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
        "scope": _scope(intervention),
        "required_comparisons": ["Matched budget and split against the dimension baseline."],
        "reopen_when": "A materially different mechanism or task regime is tested.",
        "literature_credibility": "unverified" if baseline else "preliminary",
        "credibility_rationale": (
            "The task contract fixes this baseline."
            if baseline
            else "One directly inspected primary source supports a local test."
        ),
        "testable_expectation": "Produce a valid candidate and compare its lower-is-better score.",
        "evidence": [] if baseline else [{"source_id": "src-01", "role": "supports"}],
    }
    if probe_for:
        value["probe_for"] = probe_for
    return value


def _dimension(
    catalog_by_id: dict,
    catalog_provenance: str,
    dimension_id: str,
    baseline: dict,
    *hypotheses: dict,
    mode: str = "searchable",
) -> dict:
    catalog_entry = catalog_by_id[dimension_id]
    return {
        "id": dimension_id,
        "definition": catalog_entry["definition"],
        "boundary": catalog_entry["boundary"],
        "catalog_provenance": catalog_provenance,
        "selection_reason": f"{dimension_id} has a legal material task choice.",
        "evidence": [{"kind": "task_contract", "ref": "tasks/toy/TASK.md"}],
        "mode": mode,
        "status": "active",
        "baseline_hypothesis_id": baseline["id"],
        "hypotheses": [baseline, *hypotheses],
    }


def fixture_registry() -> dict:
    catalog = load_catalog()
    catalog_by_id = {item["id"]: item for item in catalog["dimensions"]}
    dimensions = [
        _dimension(
            catalog_by_id,
            catalog["provenance"],
            "dim-data-curation",
            _hypothesis(
                "hyp-data-raw", "all legal raw examples", kind="baseline", intervention="raw-pool"
            ),
            _hypothesis(
                "hyp-data-filtered", "quality-filtered examples", kind="evidence_prior", intervention="filtering"
            ),
        ),
        _dimension(
            catalog_by_id,
            catalog["provenance"],
            "dim-model-architecture",
            _hypothesis(
                "hyp-model-linear", "linear predictor", kind="baseline", intervention="linear-model"
            ),
            _hypothesis(
                "hyp-model-multibranch", "multi-branch predictor", kind="evidence_prior", intervention="multi-branch"
            ),
        ),
        _dimension(
            catalog_by_id,
            catalog["provenance"],
            "dim-validation-selection",
            _hypothesis(
                "hyp-valid-holdout", "single holdout", kind="baseline", intervention="holdout"
            ),
            _hypothesis(
                "hyp-valid-cv", "cross-validation", kind="evidence_prior", intervention="cross-validation"
            ),
        ),
        _dimension(
            catalog_by_id,
            catalog["provenance"],
            "dim-ensemble",
            _hypothesis(
                "hyp-ensemble-identity", "identity single-predictor combination", kind="baseline", intervention="identity-ensemble"
            ),
            _hypothesis(
                "hyp-ensemble-stacking", "stacked combination", kind="evidence_prior", intervention="stacking"
            ),
        ),
        _dimension(
            catalog_by_id,
            catalog["provenance"],
            "dim-initialization-adaptation",
            _hypothesis(
                "hyp-adapt-fixed", "task-fixed initialization", kind="baseline", intervention="fixed-initialization"
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
                "scope": _scope("filtering"),
                "literature_credibility": "preliminary",
                "credibility_rationale": "One direct primary study; caution only.",
                "reopen_when": "Rare-example retention is measured explicitly.",
                "evidence": [{"source_id": "src-01", "role": "supports"}],
            }
        ],
        "sources": [
            {
                "id": "src-01",
                "type": "paper",
                "title": "Toy mechanism study",
                "url": "https://example.test/toy-mechanism-study",
                "publication_status": "preprint_only",
                "validation_status": "artifact_available",
                "studied_scope": {
                    axis: ["*"]
                    for axis in (
                        "model_families",
                        "data_regimes",
                        "metrics",
                        "interventions",
                        "evaluation_protocols",
                    )
                },
            }
        ],
    }


def shape_registry(catalog: dict, mapping: dict, space_id: str) -> dict:
    """One toy registry per benchmark shape, built only from its ownership map.

    Dimensions follow catalog order so the two shapes differ only in display
    names and ``space_id``; any behavioral divergence then proves an
    estimator-specific branch in the shared helpers.
    """
    catalog_by_id = {item["id"]: item for item in catalog["dimensions"]}
    mapped = set(mapping.values())
    dimensions = []
    for catalog_entry in catalog["dimensions"]:
        dimension_id = catalog_entry["id"]
        if dimension_id not in mapped:
            continue
        slug = dimension_id[4:]
        dimensions.append(
            _dimension(
                catalog_by_id,
                catalog["provenance"],
                dimension_id,
                _hypothesis(
                    f"hyp-{slug}-base",
                    f"{slug} baseline choice",
                    kind="baseline",
                    intervention=f"{slug}-base",
                ),
                _hypothesis(
                    f"hyp-{slug}-variant",
                    f"{slug} variant choice",
                    kind="evidence_prior",
                    intervention=f"{slug}-variant",
                ),
            )
        )
    return {
        "schema_version": 3,
        "kind": "semantic_search_space",
        "space_id": space_id,
        "catalog": catalog_receipt(catalog),
        "dimensions": dimensions,
        "relations": [],
        "guidance": [],
        "sources": [
            {
                "id": "src-01",
                "type": "paper",
                "title": "Toy mechanism study",
                "url": "https://example.test/toy-mechanism-study",
                "publication_status": "preprint_only",
                "validation_status": "artifact_available",
                "studied_scope": {
                    axis: ["*"]
                    for axis in (
                        "model_families",
                        "data_regimes",
                        "metrics",
                        "interventions",
                        "evaluation_protocols",
                    )
                },
            }
        ],
    }


def background_text(registry: dict) -> str:
    lines = [
        "# Background — P2 fixture",
        "",
        "## Dimension coverage",
        "",
        "| Dimension | Mode | Baseline | Hypotheses |",
        "|---|---|---|---|",
    ]
    for dimension in registry["dimensions"]:
        hypothesis_ids = ", ".join(f"`{item['id']}`" for item in dimension["hypotheses"])
        lines.append(
            f"| `{dimension['id']}` | {dimension['mode']} | "
            f"`{dimension['baseline_hypothesis_id']}` | {hypothesis_ids} |"
        )
    lines.extend(["", "## Dimensions", ""])
    for dimension in registry["dimensions"]:
        lines.append(f"### `{dimension['id']}`")
        for hypothesis in dimension["hypotheses"]:
            lines.append(f"- `{hypothesis['id']}` — {hypothesis['title']}")
        lines.append("")
    lines.extend(["## Relations", ""])
    for relation in registry["relations"]:
        lines.append(f"- `{relation['id']}` — {relation['type']}")
    lines.extend(
        [
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
    )
    return "\n".join(lines)


def policy_receipt(
    op: str,
    parents: list[str],
    point: dict,
    *,
    state_revision: int = 0,
    selection_index: int = 1,
    selected_lane: str = "active",
    deprioritized_interval: int = 5,
) -> dict:
    scheduled_lane = (
        "deprioritized"
        if selection_index % deprioritized_interval == 0
        else "active"
    )
    fallback = {
        ("active", "active"): "none",
        ("deprioritized", "deprioritized"): "none",
        ("active", "deprioritized"): "no_active_proposals",
        ("deprioritized", "active"): "no_deprioritized_proposals",
    }[(scheduled_lane, selected_lane)]
    return {
        "schema_version": 3,
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
            "predicted_gain": None,
            "uncertainty": None,
            "cost": None,
        },
        "acquisition_score": 1.0,
        "evidence": [],
        "budget": {
            "selection_index": selection_index,
            "deprioritized_interval": deprioritized_interval,
            "scheduled_lane": scheduled_lane,
            "selected_lane": selected_lane,
            "fallback": fallback,
            "base_rank": 1,
        },
        "ranked_point_ids": [point["point_id"]],
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
) -> dict:
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
            op,
            parents,
            point,
            selection_index=int(run_id) + 1,
        ),
        "status": status,
        "final_best_score": score,
    }
    entry["semantic_edges"] = build_semantic_edges(prior_records or [], entry)
    return entry


def main() -> int:
    registry = fixture_registry()
    assert validate_registry(registry) == [], validate_registry(registry)

    # Static ownership audits for both requested benchmark shapes: every named
    # material choice resolves to the built-in catalog and there is no catch-all.
    catalog = load_catalog()
    catalog_ids = {item["id"] for item in catalog["dimensions"]}
    coverage_fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "semantic-space-coverage.json").read_text()
    )
    for task_shape, mapping in coverage_fixture.items():
        assert set(mapping.values()) == catalog_ids, (task_shape, mapping)
        assert len(mapping) == len(catalog_ids), (task_shape, mapping)
        assert all("misc" not in dimension_id for dimension_id in mapping.values())

    # Cross-shape acceptance: the same state/proposal helpers drive both the
    # MLE-bench-shaped and the PostTrainBench-shaped ownership maps.  Any
    # estimator-specific branch would surface as a divergent signature.
    shape_signatures: dict[str, dict] = {}
    for task_shape, mapping in coverage_fixture.items():
        shape = shape_registry(catalog, mapping, f"toy-{task_shape}")
        assert validate_registry(shape) == [], validate_registry(shape)
        shape_state = empty_search_space_state()
        shape_runtime = replay_search_space_state(shape, shape_state)
        shape_effective = compose_effective_selection(
            shape, derive_hypothesis_selection(shape), shape_runtime
        )
        shape_ledger = {"records": [], "search_space_state": shape_state}
        shape_proposals = build_proposal_set(
            shape, shape_ledger, op="fresh", parents=[], max_points=16
        )
        assert validate_proposal_set(shape_proposals) == []
        assert shape_proposals["proposals"]
        shape_point, shape_receipt = select_proposal(shape_proposals, policy="coverage")
        assert validate_point(shape_point, shape) == []
        shape_transitions = derive_experience_transitions(shape, shape_ledger)
        assert shape_transitions == []
        shape_signatures[task_shape] = {
            "dimension_statuses": list(shape_runtime["dimensions"].values()),
            "hypothesis_statuses": list(shape_runtime["hypotheses"].values()),
            "effective_statuses": [
                entry["effective_status"] for entry in shape_effective.values()
            ],
            "n_proposals": len(shape_proposals["proposals"]),
            "proposal_state_revision": shape_proposals["search_space_state_revision"],
            "receipt_state_revision": shape_receipt["search_space_state_revision"],
        }
    assert set(shape_signatures) == {"mle_bench_shaped", "posttrain_bench_shaped"}
    (shape_signature, *other_signatures) = shape_signatures.values()
    assert all(other == shape_signature for other in other_signatures), shape_signatures

    with tempfile.TemporaryDirectory() as tmp:
        background_path = Path(tmp) / "background.md"
        background_path.write_text(background_text(registry))
        assert load_registry(background_path) == registry
        assert validate_background_markdown(background_path, registry) == []

        malformed_human = background_text(registry).replace("`hyp-data-filtered`", "filtered", 2)
        background_path.write_text(malformed_human)
        errors = validate_background_markdown(background_path, registry)
        assert any("hyp-data-filtered" in error for error in errors), errors
        background_path.write_text(background_text(registry))

        legacy_path = Path(tmp) / "legacy.md"
        legacy_path.write_text(
            "# old\n\n## Direction registry\n```json\n"
            '{"schema_version":2,"directions":[]}\n```\n'
        )
        try:
            load_registry(legacy_path)
        except ContractError as exc:
            assert "legacy flat" in str(exc)
        else:
            raise AssertionError("legacy flat background was silently accepted")

    # Retrieval provenance still gates every literature source.
    manifest = new_manifest()
    manifest["coverage_exemptions"] = [
        {
            "dimension_id": dimension["id"],
            "rationale": "Synthetic contract fixture does not run literature search.",
        }
        for dimension in registry["dimensions"]
        if dimension["mode"] == "searchable"
    ]
    add_visit(
        manifest,
        url=registry["sources"][0]["url"],
        lane="grounding",
        backend="fixture",
        view="full_text",
        status="success",
        content="inspected primary source" * 30,
    )
    assert validate_registry(registry, retrieval_manifest=manifest) == []

    baseline = complete_point(registry)
    assert baseline is not None and validate_point(baseline, registry) == []
    assert complete_point(registry, {"dim-miscellaneous": "hyp-unknown"}) is None
    assert len(baseline["assignments"]) == len(registry["dimensions"])
    ensemble_assignment = next(
        item for item in baseline["assignments"] if item["dimension_id"] == "dim-ensemble"
    )
    assert ensemble_assignment["state"] == "inactive"

    stacked = complete_point(
        registry,
        {
            "dim-model-architecture": "hyp-model-multibranch",
            "dim-ensemble": "hyp-ensemble-stacking",
        },
    )
    assert stacked is not None
    selected = selected_assignments(stacked)
    assert selected["dim-ensemble"] == "hyp-ensemble-stacking"
    assert selected["dim-validation-selection"] == "hyp-valid-cv"
    assert selected["dim-initialization-adaptation"] == "hyp-adapt-fixed"

    invalid_exclusion = copy.deepcopy(stacked)
    for assignment in invalid_exclusion["assignments"]:
        if assignment["dimension_id"] == "dim-data-curation":
            assignment["hypothesis_id"] = "hyp-data-filtered"
    invalid_exclusion["point_id"] = point_id(invalid_exclusion)
    errors = validate_point(invalid_exclusion, registry)
    assert any("rel-filter-multibranch-exclusion" in error for error in errors), errors

    unknown_dimension = copy.deepcopy(registry)
    unknown_dimension["dimensions"][0]["id"] = "dim-miscellaneous"
    errors = validate_registry(unknown_dimension)
    assert any("is not in catalog" in error for error in errors), errors

    legacy_registry = {"schema_version": 2, "directions": [], "sources": []}
    errors = validate_registry(legacy_registry)
    assert any("legacy flat background registry" in error for error in errors), errors

    changed = copy.deepcopy(registry)
    changed["dimensions"][0]["hypotheses"][0]["claim"] += " Changed."
    assert space_revision(changed) != space_revision(registry)
    assert any(
        "space_revision" in error for error in validate_point(baseline, changed)
    )

    selection = derive_hypothesis_selection(registry)
    assert selection["hyp-data-filtered"]["selection_status"] == "active"
    assert selection["hyp-data-filtered"]["matched_guidance"] == [
        {"id": "g-01", "effect": "caution"}
    ]
    assert scope_relation(_scope("filtering"), _scope("filtering")) == "direct"
    assert scope_relation(_scope("filtering"), _scope("stacking")) == "mismatch"

    # Distinct implementations may occupy the same semantic point.  Numeric
    # ancestry and attribution remain separate and mechanical diffs stay empty.
    records: list[dict] = []
    for run_id, op, parents, point, score, status in [
        ("000", "fresh", [], baseline, 0.50, "keep"),
        ("001", "improve", ["000"], baseline, 0.45, "keep"),
        ("002", "fresh", [], stacked, 0.60, "discard"),
        ("003", "crossover", ["001", "002"], stacked, 0.40, "keep"),
    ]:
        records.append(
            record(
                run_id, op, parents, point,
                score=score, status=status, prior_records=records,
            )
        )
    ledger = {
        "search_space": space_receipt(registry),
        "search_space_state": empty_search_space_state(),
        "records": records,
    }
    assert validate_registry(registry, ledger=ledger) == [], validate_registry(
        registry, ledger=ledger
    )
    lineage = derive_semantic_lineage(registry, ledger)
    run_one = next(item for item in lineage["runs"] if item["run_id"] == "001")
    (edge,) = run_one["semantic_edges"]
    assert edge["parent_run_id"] == "000"
    assert edge["changes"] == []
    run_three = next(item for item in lineage["runs"] if item["run_id"] == "003")
    assert len(run_three["semantic_edges"]) == 2
    assert "attribution only" in lineage["attribution_notice"].lower()
    assert "do not claim" in lineage["attribution_notice"].lower()

    bad_ancestry = copy.deepcopy(ledger)
    bad_ancestry["records"][0]["source_run_ids"] = ["tf-01"]
    errors = validate_registry(registry, ledger=bad_ancestry)
    assert any("numeric parent ids" in error for error in errors), errors

    missing_mapping = copy.deepcopy(ledger)
    missing_mapping["records"][0].pop("semantic_point")
    errors = validate_registry(registry, ledger=missing_mapping)
    assert any("semantic_point must be an object" in error for error in errors), errors

    forged_policy = copy.deepcopy(ledger)
    forged_policy["records"][0]["policy_receipt"]["acquisition_score"] = 0.25
    errors = validate_registry(registry, ledger=forged_policy)
    assert any("does not match its separate components" in error for error in errors), errors

    duplicate_run = copy.deepcopy(ledger)
    duplicate_run["records"][1]["run_id"] = "000"
    errors = validate_registry(registry, ledger=duplicate_run)
    assert any("duplicates an earlier record" in error for error in errors), errors

    # P1 ledgers without persisted mechanical receipts are intentionally
    # unsupported: a missing or forged semantic_edges field fails validation.
    missing_edges = copy.deepcopy(ledger)
    del missing_edges["records"][1]["semantic_edges"]
    errors = validate_registry(registry, ledger=missing_edges)
    assert any("semantic_edges" in error for error in errors), errors

    forged_edges = copy.deepcopy(ledger)
    forged_edges["records"][1]["semantic_edges"][0]["change_class"] = "multi_dimension"
    errors = validate_registry(registry, ledger=forged_edges)
    assert any("semantic_edges" in error for error in errors), errors

    # All graph actions repeatedly produce valid mapped points. Hypotheses are
    # coverage-counted, not consumed after first use.
    empty_ledger = {"records": []}
    fresh_proposals = build_proposal_set(
        registry, empty_ledger, op="fresh", parents=[], max_points=64
    )
    bounded_fresh = build_proposal_set(
        registry, empty_ledger, op="fresh", parents=[], max_points=5
    )
    bounded_choices = {
        hypothesis_id
        for proposal in bounded_fresh["proposals"]
        for hypothesis_id in selected_assignments(proposal["point"]).values()
    }
    assert {
        "hyp-data-filtered",
        "hyp-model-multibranch",
        "hyp-valid-cv",
        "hyp-ensemble-stacking",
    }.issubset(bounded_choices), bounded_choices
    improve_proposals = build_proposal_set(
        registry, ledger, op="improve", parents=["000"], max_points=64
    )
    crossover_proposals = build_proposal_set(
        registry, ledger, op="crossover", parents=["001", "002"], max_points=64
    )
    for proposal_set in (fresh_proposals, improve_proposals, crossover_proposals):
        assert validate_proposal_set(proposal_set) == [], validate_proposal_set(proposal_set)
        assert proposal_set["proposals"]
        for proposal in proposal_set["proposals"]:
            assert validate_point(proposal["point"], registry) == []
    malformed_proposals = copy.deepcopy(fresh_proposals)
    malformed_proposals["proposals"][0]["coverage"] = 2.0
    proposal_payload = dict(malformed_proposals)
    proposal_payload.pop("proposal_set_revision")
    malformed_proposals["proposal_set_revision"] = digest(proposal_payload)
    assert any("coverage" in error for error in validate_proposal_set(malformed_proposals))
    try:
        build_proposal_set(registry, empty_ledger, op="fresh", parents=[], max_points=0)
    except ContractError as exc:
        assert "max_points" in str(exc)
    else:
        raise AssertionError("an unbounded/empty proposal cap was accepted")
    assert any(
        proposal["point_id"] == baseline["point_id"]
        for proposal in improve_proposals["proposals"]
    )

    coverage_point, coverage_receipt = select_proposal(
        fresh_proposals, policy="coverage"
    )
    assert validate_point(coverage_point, registry) == []
    assert coverage_receipt["components"]["predicted_gain"] is None
    assert coverage_receipt["components"]["uncertainty"] is None

    # Exercise the real mutation boundary: add-record copies the exact space
    # receipt and persists ancestry, attribution, and policy separately.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        background_path = tmp_path / "background.md"
        point_path = tmp_path / "point.json"
        policy_path = tmp_path / "policy.json"
        ledger_path = tmp_path / "ledger.json"
        background_path.write_text(background_text(registry))
        point_path.write_text(json.dumps(coverage_point))
        policy_path.write_text(json.dumps(coverage_receipt))
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "ledger.py"),
                "add-record",
                "--ledger",
                str(ledger_path),
                "--task",
                "hard-interactions",
                "--run-id",
                "000",
                "--op",
                "fresh",
                "--source-run-ids",
                "",
                "--background",
                str(background_path),
                "--semantic-point",
                str(point_path),
                "--policy-receipt",
                str(policy_path),
                "--idea",
                "Complete fixture solution at the selected point.",
                "--change",
                "from scratch at the selected point",
                "--candidate-name-hint",
                "fixture_solution",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        stored = json.loads(ledger_path.read_text())
        assert stored["search_space"] == space_receipt(registry)
        assert stored["records"][0]["source_run_ids"] == []
        assert stored["records"][0]["semantic_point"] == coverage_point
        assert stored["records"][0]["policy_receipt"] == coverage_receipt

        # The real result path keeps finite improvements lower-is-better and
        # turns a missing/non-finite result into the worst crash sentinel.
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "ledger.py"),
                "record-run",
                "--ledger",
                str(ledger_path),
                "--task",
                "hard-interactions",
                "--run-id",
                "000",
                "--final-best-score",
                "0.5",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        policy_path.write_text(
            json.dumps(
                policy_receipt(
                    "improve", ["000"], coverage_point, selection_index=2
                )
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "ledger.py"),
                "add-record",
                "--ledger",
                str(ledger_path),
                "--task",
                "hard-interactions",
                "--run-id",
                "001",
                "--op",
                "improve",
                "--source-run-ids",
                "000",
                "--background",
                str(background_path),
                "--semantic-point",
                str(point_path),
                "--policy-receipt",
                str(policy_path),
                "--idea",
                "A distinct implementation at the same attributed semantic point.",
                "--change",
                "implementation-only refactor at the unchanged point",
                "--candidate-name-hint",
                "fixture_same_point",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "ledger.py"),
                "record-run",
                "--ledger",
                str(ledger_path),
                "--task",
                "hard-interactions",
                "--run-id",
                "001",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        experience_path = tmp_path / "experience.json"
        experience_path.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "updated_at_run": "001",
                    "generation": 0,
                    "summary": "A same-point implementation completed and a later attempt crashed.",
                    "promising_regions": [],
                    "lessons": [
                        {
                            "kind": "feasibility",
                            "claim": "The second implementation did not produce a finite result.",
                            "evidence": ["001"],
                            "confidence": "high",
                        }
                    ],
                    "bottlenecks": [],
                    "dimension_evidence": [],
                    "hypothesis_evidence": [],
                }
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "ledger.py"),
                "set-experience",
                "--ledger",
                str(ledger_path),
                "--task",
                "hard-interactions",
                "--background",
                str(background_path),
                "--from-json",
                str(experience_path),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        stored = json.loads(ledger_path.read_text())
        assert stored["records"][0]["status"] == "keep"
        assert stored["records"][0]["final_best_score"] == 0.5
        assert stored["records"][1]["status"] == "crash"
        assert math.isinf(stored["records"][1]["final_best_score"])
        assert stored["experience"]["dag_revision"] == stored["dag_revision"]

        # The append-only pruning pass follows set-experience; without target
        # beliefs it is a successful no-op that saves once and moves nothing.
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "ledger.py"),
                "apply-space-state",
                "--ledger",
                str(ledger_path),
                "--background",
                str(background_path),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        applied = json.loads(completed.stdout)
        assert applied == {
            "ok": True,
            "prior_revision": 0,
            "revision": 0,
            "decision_ids": [],
        }, applied
        stored = json.loads(ledger_path.read_text())
        assert stored["search_space_state"] == empty_search_space_state()

        # P2 selection lifecycle through real CLIs: two direct comparator
        # edges against hyp-data-filtered drive the two-stage pruning
        # (revision 1 deprioritized, revision 2 pruned); revision-stamped
        # proposals and receipts track the overlay; stale artifacts are
        # rejected; a later belief with an advanced DAG cursor reopens it.
        prune_target = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})

        def cli_add_record(run_id: str, op: str, parents: list[str], point: dict, receipt: dict) -> None:
            point_path.write_text(json.dumps(point))
            receipt = copy.deepcopy(receipt)
            current_records = (
                json.loads(ledger_path.read_text()).get("records", [])
                if ledger_path.exists()
                else []
            )
            budget = receipt["budget"]
            budget["selection_index"] = len(current_records) + 1
            interval = budget["deprioritized_interval"]
            budget["scheduled_lane"] = (
                "deprioritized"
                if budget["selection_index"] % interval == 0
                else "active"
            )
            budget["fallback"] = {
                ("active", "active"): "none",
                ("deprioritized", "deprioritized"): "none",
                ("active", "deprioritized"): "no_active_proposals",
                ("deprioritized", "active"): "no_deprioritized_proposals",
            }[(budget["scheduled_lane"], budget["selected_lane"])]
            policy_path.write_text(json.dumps(receipt))
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "ledger.py"),
                    "add-record",
                    "--ledger", str(ledger_path),
                    "--task", "hard-interactions",
                    "--run-id", run_id,
                    "--op", op,
                    "--source-run-ids", ",".join(parents),
                    "--background", str(background_path),
                    "--semantic-point", str(point_path),
                    "--policy-receipt", str(policy_path),
                    "--idea", f"Complete fixture solution {run_id} at the selected point.",
                    "--change", f"fixture change for {op} run {run_id}",
                    "--candidate-name-hint", f"fixture_{run_id}",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        def cli_record_run(run_id: str, score: float | None = None) -> None:
            command = [
                sys.executable,
                str(ROOT / "tools" / "ledger.py"),
                "record-run",
                "--ledger", str(ledger_path),
                "--task", "hard-interactions",
                "--run-id", run_id,
            ]
            if score is not None:
                command += ["--final-best-score", str(score)]
            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        def cli_set_experience(
            generation: int,
            updated_at_run: str,
            beliefs: list[dict],
            dimension_beliefs: list[dict] | None = None,
        ) -> None:
            experience_path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "updated_at_run": updated_at_run,
                        "generation": generation,
                        "summary": "Comparator evidence against the filtered hypothesis.",
                        "promising_regions": [],
                        "lessons": [],
                        "bottlenecks": [],
                        "dimension_evidence": dimension_beliefs or [],
                        "hypothesis_evidence": beliefs,
                    }
                )
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "ledger.py"),
                    "set-experience",
                    "--ledger", str(ledger_path),
                    "--task", "hard-interactions",
                    "--background", str(background_path),
                    "--from-json", str(experience_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        def cli_apply_space_state() -> dict:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "ledger.py"),
                    "apply-space-state",
                    "--ledger", str(ledger_path),
                    "--background", str(background_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(completed.stdout)

        def cli_propose(output_path: Path) -> dict:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "semantic_search.py"),
                    "propose",
                    "--background", str(background_path),
                    "--ledger", str(ledger_path),
                    "--op", "fresh",
                    "--output", str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(output_path.read_text())

        def selects_target(proposal: dict) -> bool:
            return (
                selected_assignments(proposal["point"]).get("dim-data-curation")
                == "hyp-data-filtered"
            )

        # Two matched comparisons against a fresh baseline parent, both worse.
        baseline_point = complete_point(registry)
        cli_add_record("002", "fresh", [], baseline_point, policy_receipt("fresh", [], baseline_point))
        cli_record_run("002", 0.45)
        cli_add_record("003", "improve", ["002"], prune_target, policy_receipt("improve", ["002"], prune_target))
        cli_record_run("003", 0.6)
        cli_add_record("004", "improve", ["002"], prune_target, policy_receipt("improve", ["002"], prune_target))
        cli_record_run("004", 0.62)

        prune_belief = {
            "target_id": "hyp-data-filtered",
            "evaluation_state": "comparator_covered",
            "assessment": "unpromising",
            "recommended_status": "pruned",
            "claim": (
                "Repeated matched comparisons show the filtering mechanism removes "
                "useful signal without reducing downstream cost, so another outer "
                "evaluation has low expected marginal value."
            ),
            "evidence_run_ids": ["002", "003", "004"],
            "evidence_edge_ids": ["sedge-002-003", "sedge-002-004"],
            "comparator_coverage": {
                "direct_noncrash_edges": 2,
                "confounded_noncrash_edges": 0,
                "crash_edges": 0,
            },
            "confidence": "high",
            "uncertainty": "Implementation differences remain confounded with each semantic change.",
            "reopen_when": "A later direct comparison improves over its parent.",
        }
        cli_set_experience(1, "004", [prune_belief])
        applied = cli_apply_space_state()
        assert applied == {
            "ok": True,
            "prior_revision": 0,
            "revision": 1,
            "decision_ids": ["sdec-000001"],
        }, applied

        # Deprioritization has real budget semantics. At admission 6 with an
        # interval of 2, a new direct comparison is admitted from the reserved
        # deprioritized lane and supplies target-specific advancing evidence.
        cli_add_record(
            "005",
            "improve",
            ["002"],
            prune_target,
            policy_receipt(
                "improve",
                ["002"],
                prune_target,
                state_revision=1,
                selected_lane="deprioritized",
                deprioritized_interval=2,
            ),
        )
        cli_record_run("005", 0.61)
        prune_belief = copy.deepcopy(prune_belief)
        prune_belief["evidence_run_ids"] = ["002", "003", "004", "005"]
        prune_belief["evidence_edge_ids"] = [
            "sedge-002-003",
            "sedge-002-004",
            "sedge-002-005",
        ]
        prune_belief["comparator_coverage"]["direct_noncrash_edges"] = 3
        cli_set_experience(2, "005", [prune_belief])
        applied = cli_apply_space_state()
        assert applied == {
            "ok": True,
            "prior_revision": 1,
            "revision": 2,
            "decision_ids": ["sdec-000002"],
        }, applied

        # Later pruning must not invalidate the revision-0 records.
        stored = json.loads(ledger_path.read_text())
        assert stored["search_space_state"]["revision"] == 2
        assert validate_registry(registry, ledger=stored) == [], validate_registry(
            registry, ledger=stored
        )

        # Revision-current proposals omit the pruned hypothesis and carry the stamp.
        proposals_path = tmp_path / "proposals.json"
        current_proposals = cli_propose(proposals_path)
        assert current_proposals["schema_version"] == 3
        assert current_proposals["search_space_state_revision"] == 2
        assert current_proposals["proposals"]
        assert not any(selects_target(item) for item in current_proposals["proposals"])

        # Selection stamps the same revision into the policy receipt.
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "semantic_search.py"),
                "select",
                "--proposals", str(proposals_path),
                "--ledger", str(ledger_path),
                "--policy", "coverage",
                "--point-output", str(point_path),
                "--receipt-output", str(policy_path),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        selected_receipt = json.loads(policy_path.read_text())
        assert selected_receipt["schema_version"] == 3
        assert selected_receipt["search_space_state_revision"] == 2

        # A proposal set stamped at another revision is rejected as stale.
        stale_proposals = copy.deepcopy(current_proposals)
        stale_proposals["search_space_state_revision"] = 1
        proposals_path.write_text(json.dumps(stale_proposals))
        rejected = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "semantic_search.py"),
                "select",
                "--proposals", str(proposals_path),
                "--ledger", str(ledger_path),
                "--policy", "coverage",
                "--point-output", str(point_path),
                "--receipt-output", str(policy_path),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0
        assert "stale" in rejected.stderr, rejected.stderr

        # add-record rejects a stale revision-0 receipt against revision 2.
        point_path.write_text(json.dumps(prune_target))
        policy_path.write_text(
            json.dumps(policy_receipt("fresh", [], prune_target, state_revision=0))
        )
        rejected = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "ledger.py"),
                "add-record",
                "--ledger", str(ledger_path),
                "--task", "hard-interactions",
                "--run-id", "006",
                "--op", "fresh",
                "--source-run-ids", "",
                "--background", str(background_path),
                "--semantic-point", str(point_path),
                "--policy-receipt", str(policy_path),
                "--idea", "A forged late arrival at the pruned point.",
                "--change", "from scratch at the pruned point",
                "--candidate-name-hint", "fixture_stale",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0
        assert "stale" in rejected.stderr, rejected.stderr
        stored = json.loads(ledger_path.read_text())
        assert [item["run_id"] for item in stored["records"]] == [
            "000", "001", "002", "003", "004", "005",
        ]

        # A changed observation on a cited target edge advances target-specific
        # evidence and can support reopening; an unrelated run would not.
        cli_record_run("005", 0.40)

        reopen_belief = {
            "target_id": "hyp-data-filtered",
            "evaluation_state": "comparator_covered",
            "assessment": "promising",
            "recommended_status": "active",
            "claim": "The updated direct comparison now improves over its matched parent.",
            "evidence_run_ids": ["002", "003", "004", "005"],
            "evidence_edge_ids": [
                "sedge-002-003",
                "sedge-002-004",
                "sedge-002-005",
            ],
            "comparator_coverage": {
                "direct_noncrash_edges": 3,
                "confounded_noncrash_edges": 0,
                "crash_edges": 0,
            },
            "confidence": "high",
            "uncertainty": "The improvement signal remains implementation-confounded.",
        }
        cli_set_experience(3, "005", [reopen_belief])
        applied = cli_apply_space_state()
        assert applied == {
            "ok": True,
            "prior_revision": 2,
            "revision": 3,
            "decision_ids": ["sdec-000003"],
        }, applied

        # Revision 3 proposals re-admit the reopened hypothesis.
        reopened_proposals = cli_propose(proposals_path)
        assert reopened_proposals["search_space_state_revision"] == 3
        assert any(selects_target(item) for item in reopened_proposals["proposals"])
        stored = json.loads(ledger_path.read_text())
        assert stored["search_space_state"]["revision"] == 3
        assert len(stored["search_space_state"]["decisions"]) == 3
        assert validate_registry(registry, ledger=stored) == [], validate_registry(
            registry, ledger=stored
        )

        # The overlay revision is exactly the number of append-only decisions.
        state = stored["search_space_state"]
        assert state["revision"] == len(state["decisions"]) > 0

        # Later decisions never invalidate records admitted at earlier state
        # revisions: the pre-pruning records at the once-pruned point still
        # validate against the frozen registry.
        pre_pruning = [
            item for item in stored["records"] if item["run_id"] in {"003", "004"}
        ]
        assert len(pre_pruning) == 2
        for item in pre_pruning:
            assert validate_point(item["semantic_point"], registry) == []

        # `target-evidence` is the extractor's authoritative bounded source: an
        # old direct comparator whose child run fell out of a one-node graph
        # window is still returned, with selected counts equal to validator
        # recomputation.  The experience cursor is current, so the incremental
        # delta is empty and --top 1 --bottom 0 leaves only the best node.
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "got_graph.py"),
                "render",
                "--ledger", str(ledger_path),
                "--incremental",
                "--top", "1",
                "--bottom", "0",
                "--format", "json",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        window = json.loads(completed.stdout)
        window_ids = {
            node["id"]
            for key in ("delta_nodes", "top_nodes", "bottom_nodes")
            for node in window[key]
        }
        assert window_ids == {"005"}, window_ids
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "background_contract.py"),
                "target-evidence",
                "--background", str(background_path),
                "--ledger", str(ledger_path),
                "--target-id", "hyp-data-filtered",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        view = json.loads(completed.stdout)
        (block,) = view["hypothesis_targets"]
        assert {
            "sedge-002-003",
            "sedge-002-004",
            "sedge-002-005",
        } <= set(block["evidence_edge_ids"])
        assert {"003", "004"}.isdisjoint(window_ids)
        assert block["comparator_coverage"] == comparator_coverage(
            stored,
            block["evidence_edge_ids"],
            target_kind="hypothesis",
            target_id="hyp-data-filtered",
        )

        # A crash-only belief informs feasibility but never becomes
        # contradiction evidence: applying it appends no decisions and moves
        # neither the overlay revision nor the DAG cursor.  The CLI helpers
        # close over ledger_path, so rebinding redirects them to a fresh ledger.
        ledger_path = tmp_path / "ledger-crash.json"
        cli_add_record(
            "000", "fresh", [], baseline_point, policy_receipt("fresh", [], baseline_point)
        )
        cli_record_run("000", 0.45)
        cli_add_record(
            "001", "improve", ["000"], prune_target,
            policy_receipt("improve", ["000"], prune_target),
        )
        cli_record_run("001")  # no score: the run crashes
        crash_belief = {
            "target_id": "hyp-data-filtered",
            "evaluation_state": "failed",
            "assessment": "unknown",
            "recommended_status": "active",
            "claim": "The only comparator attempt crashed; a crash informs feasibility, not contradiction.",
            "evidence_run_ids": ["001"],
            "evidence_edge_ids": ["sedge-000-001"],
            "comparator_coverage": {
                "direct_noncrash_edges": 0,
                "confounded_noncrash_edges": 0,
                "crash_edges": 1,
            },
            "confidence": "low",
            "uncertainty": "No non-crash observation exists for this target.",
        }
        dag_before = json.loads(ledger_path.read_text())["dag_revision"]
        cli_set_experience(0, "001", [crash_belief])
        applied = cli_apply_space_state()
        assert applied == {
            "ok": True,
            "prior_revision": 0,
            "revision": 0,
            "decision_ids": [],
        }, applied
        stored = json.loads(ledger_path.read_text())
        assert stored["dag_revision"] == dag_before
        assert stored["search_space_state"] == empty_search_space_state()

        # A runtime-pruned dimension is pinned to its explicit baseline in new
        # proposals; point arity and the frozen space_revision never change.
        ledger_path = tmp_path / "ledger-dimension.json"
        cli_add_record(
            "000", "fresh", [], baseline_point, policy_receipt("fresh", [], baseline_point)
        )
        cli_record_run("000", 0.45)
        cli_add_record(
            "001", "improve", ["000"], prune_target,
            policy_receipt("improve", ["000"], prune_target),
        )
        cli_record_run("001", 0.6)
        cli_add_record(
            "002", "improve", ["000"], prune_target,
            policy_receipt("improve", ["000"], prune_target),
        )
        cli_record_run("002", 0.62)

        def cli_apply_preserving_dag() -> dict:
            dag_before = json.loads(ledger_path.read_text())["dag_revision"]
            result = cli_apply_space_state()
            dag_after = json.loads(ledger_path.read_text())["dag_revision"]
            assert dag_after == dag_before, (dag_before, dag_after)
            return result

        hypothesis_prune = {
            "target_id": "hyp-data-filtered",
            "evaluation_state": "comparator_covered",
            "assessment": "unpromising",
            "recommended_status": "pruned",
            "claim": (
                "Matched comparisons consistently show the filtering mechanism "
                "removes useful signal without offsetting cost."
            ),
            "evidence_run_ids": ["000", "001", "002"],
            "evidence_edge_ids": ["sedge-000-001", "sedge-000-002"],
            "comparator_coverage": {
                "direct_noncrash_edges": 2,
                "confounded_noncrash_edges": 0,
                "crash_edges": 0,
            },
            "confidence": "high",
            "uncertainty": "Implementation differences remain confounded with each semantic change.",
            "reopen_when": "A later direct comparison improves over its parent.",
        }
        dimension_prune = {
            "target_id": "dim-data-curation",
            "evaluation_state": "comparator_covered",
            "assessment": "unpromising",
            "recommended_status": "pruned",
            "claim": "Every selectable non-baseline hypothesis in the dimension is already pruned.",
            "evidence_run_ids": ["000", "001", "002"],
            "evidence_edge_ids": ["sedge-000-001", "sedge-000-002"],
            "comparator_coverage": {
                "direct_noncrash_edges": 2,
                "confounded_noncrash_edges": 0,
                "crash_edges": 0,
            },
            "confidence": "high",
            "uncertainty": "Dimension-level attribution stays weaker than its single-dimension edges.",
            "reopen_when": "A non-baseline hypothesis in the dimension is reopened.",
        }
        cli_set_experience(0, "002", [hypothesis_prune])
        assert cli_apply_preserving_dag() == {
            "ok": True, "prior_revision": 0, "revision": 1,
            "decision_ids": ["sdec-000001"],
        }

        cli_add_record(
            "003",
            "improve",
            ["000"],
            prune_target,
            policy_receipt(
                "improve",
                ["000"],
                prune_target,
                state_revision=1,
                selected_lane="deprioritized",
                deprioritized_interval=2,
            ),
        )
        cli_record_run("003", 0.61)
        hypothesis_prune = copy.deepcopy(hypothesis_prune)
        hypothesis_prune["evidence_run_ids"] = ["000", "001", "002", "003"]
        hypothesis_prune["evidence_edge_ids"] = [
            "sedge-000-001",
            "sedge-000-002",
            "sedge-000-003",
        ]
        hypothesis_prune["comparator_coverage"]["direct_noncrash_edges"] = 3
        dimension_prune = copy.deepcopy(dimension_prune)
        dimension_prune["evidence_run_ids"] = ["000", "001", "002", "003"]
        dimension_prune["evidence_edge_ids"] = [
            "sedge-000-001",
            "sedge-000-002",
            "sedge-000-003",
        ]
        dimension_prune["comparator_coverage"]["direct_noncrash_edges"] = 3

        cli_set_experience(1, "003", [hypothesis_prune])
        assert cli_apply_preserving_dag() == {
            "ok": True, "prior_revision": 1, "revision": 2,
            "decision_ids": ["sdec-000002"],
        }
        cli_set_experience(2, "003", [], dimension_beliefs=[dimension_prune])
        assert cli_apply_preserving_dag() == {
            "ok": True, "prior_revision": 2, "revision": 3,
            "decision_ids": ["sdec-000003"],
        }

        # A later changed observation on an edge touching this dimension is
        # required for its second-stage contraction.
        cli_record_run("003", 0.60)
        cli_set_experience(3, "003", [], dimension_beliefs=[dimension_prune])
        assert cli_apply_preserving_dag() == {
            "ok": True, "prior_revision": 3, "revision": 4,
            "decision_ids": ["sdec-000004"],
        }

        stored = json.loads(ledger_path.read_text())
        state = stored["search_space_state"]
        assert state["revision"] == len(state["decisions"]) == 4
        assert validate_registry(registry, ledger=stored) == [], validate_registry(
            registry, ledger=stored
        )
        replayed = replay_search_space_state(registry, state)
        assert replayed["dimensions"]["dim-data-curation"] == "pruned"
        assert replayed["hypotheses"]["hyp-data-filtered"] == "pruned"

        dimension_proposals = cli_propose(proposals_path)
        assert dimension_proposals["search_space_state_revision"] == 4
        assert dimension_proposals["proposals"]
        dimension_baseline = next(
            item["baseline_hypothesis_id"]
            for item in registry["dimensions"]
            if item["id"] == "dim-data-curation"
        )
        for proposal in dimension_proposals["proposals"]:
            point = proposal["point"]
            assert validate_point(point, registry) == []
            assert point["space_revision"] == space_revision(registry)
            assert len(point["assignments"]) == len(registry["dimensions"])
            assert not selects_target(proposal)
            for assignment in point["assignments"]:
                if (
                    assignment["dimension_id"] == "dim-data-curation"
                    and assignment["state"] == "selected"
                ):
                    assert assignment["hypothesis_id"] == dimension_baseline

    proposals = fresh_proposals["proposals"]
    assert len(proposals) >= 2
    predictions = {
        "schema_version": 1,
        "proposal_set_revision": fresh_proposals["proposal_set_revision"],
        "predictions": [],
    }
    for index, proposal in enumerate(proposals):
        if index == 0:
            gain, uncertainty = 0.95, 0.05
        elif index == 1:
            gain, uncertainty = 0.65, 1.0
        else:
            gain, uncertainty = 0.05, 0.05
        predictions["predictions"].append(
            {
                "point_id": proposal["point_id"],
                "predicted_gain": gain,
                "uncertainty": uncertainty,
                "cost": 0.1,
                "evidence": ["fixture rubric over background and bounded ledger view"],
            }
        )
    gain_point, gain_receipt = select_proposal(
        fresh_proposals, policy="gain", predictions=predictions
    )
    explore_point, explore_receipt = select_proposal(
        fresh_proposals,
        policy="gain_uncertainty",
        predictions=predictions,
        config={"uncertainty_weight": 0.8},
    )
    assert gain_point["point_id"] == proposals[0]["point_id"]
    assert explore_point["point_id"] == proposals[1]["point_id"]
    assert gain_receipt["components"]["uncertainty"] == 0.05
    assert explore_receipt["components"]["predicted_gain"] == 0.65
    assert explore_receipt["components"]["uncertainty"] == 1.0
    assert gain_receipt["policy"]["name"] != explore_receipt["policy"]["name"]

    rendered = render_space(registry, ledger, max_hypotheses=2)
    assert rendered["coverage"]["n_valid_records"] == 4
    assert len(rendered["dimensions"]) == len(registry["dimensions"])

    # P2 experience snapshots are schema 3: the two-level target evidence
    # collections are required and the P1 schema is rejected without migration.
    errors = validate_experience(
        {
            "schema_version": 2,
            "updated_at_run": "003",
            "generation": 0,
            "summary": "A P1 snapshot has no place in a P2 ledger.",
            "promising_regions": [],
            "lessons": [],
            "bottlenecks": [],
        },
        registry,
        ledger,
    )
    assert any("schema_version must be 3" in error for error in errors), errors
    assert validate_experience(
        {
            "schema_version": 3,
            "updated_at_run": "003",
            "generation": 0,
            "summary": "One same-point implementation change improved the observed score.",
            "promising_regions": [],
            "lessons": [
                {
                    "kind": "lever",
                    "claim": "Implementation-level improvement at the same point.",
                    "evidence": ["000", "001"],
                    "confidence": "low",
                }
            ],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [],
        },
        registry,
        ledger,
    ) == []
    stale_view_ledger = copy.deepcopy(ledger)
    stale_view_ledger["dag_revision"] = 5
    assert validate_experience(
        {
            "schema_version": 3,
            "updated_at_run": "001",
            "generation": 0,
            "summary": "A valid bounded snapshot may lag a newer DAG delta.",
            "promising_regions": [],
            "lessons": [
                {
                    "kind": "lever",
                    "claim": "The earlier same-point implementation improved.",
                    "evidence": ["000", "001"],
                    "confidence": "low",
                }
            ],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [],
            "dag_revision": 2,
        },
        registry,
        stale_view_ledger,
    ) == []

    # P2 target-evidence CLI: the real subprocess renders mechanical comparator
    # coverage identical to the in-process function over persisted receipts.
    filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    evidence_records: list[dict] = []
    for run_id, op, parents, point, score, status in [
        ("000", "fresh", [], baseline, 0.40, "keep"),
        ("001", "improve", ["000"], filtered, 0.50, "discard"),
        ("002", "fresh", [], baseline, 0.41, "keep"),
        ("003", "improve", ["002"], filtered, 0.52, "discard"),
    ]:
        evidence_records.append(
            record(
                run_id, op, parents, point,
                score=score, status=status, prior_records=evidence_records,
            )
        )
    evidence_ledger = {
        "search_space": space_receipt(registry),
        "search_space_state": empty_search_space_state(),
        "records": evidence_records,
        "dag_revision": 4,
    }
    assert validate_registry(registry, ledger=evidence_ledger) == [], validate_registry(
        registry, ledger=evidence_ledger
    )
    with tempfile.TemporaryDirectory() as tmp:
        background_path = Path(tmp) / "background.md"
        ledger_path = Path(tmp) / "ledger.json"
        background_path.write_text(background_text(registry))
        ledger_path.write_text(json.dumps(evidence_ledger))
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "background_contract.py"),
                "target-evidence",
                "--background",
                str(background_path),
                "--ledger",
                str(ledger_path),
                "--target-id",
                "hyp-data-filtered",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    view = json.loads(completed.stdout)
    (block,) = view["hypothesis_targets"]
    assert block["evidence_edge_ids"] == ["sedge-000-001", "sedge-002-003"]
    assert block["comparator_coverage"] == comparator_coverage(
        evidence_ledger,
        block["evidence_edge_ids"],
        target_kind="hypothesis",
        target_id="hyp-data-filtered",
    )
    assert block["comparator_coverage"] == {
        "direct_noncrash_edges": 2,
        "confounded_noncrash_edges": 0,
        "crash_edges": 0,
    }

    # A schema-3 belief snapshot cites the same persisted receipts: the
    # recomputed comparator counts and evaluation state must match exactly,
    # and the conservative pruning gates are enforced on both levels.
    covered_belief = {
        "target_id": "hyp-data-filtered",
        "evaluation_state": "comparator_covered",
        "assessment": "unpromising",
        "recommended_status": "pruned",
        "claim": (
            "Matched comparisons consistently support a signal-removal failure "
            "mechanism with low expected value from another outer evaluation."
        ),
        "evidence_run_ids": ["000", "001", "002", "003"],
        "evidence_edge_ids": ["sedge-000-001", "sedge-002-003"],
        "comparator_coverage": {
            "direct_noncrash_edges": 2,
            "confounded_noncrash_edges": 0,
            "crash_edges": 0,
        },
        "confidence": "high",
        "uncertainty": "Implementation differences remain confounded with each semantic change.",
        "reopen_when": "A later direct comparison improves over its parent.",
    }
    assert validate_experience(
        {
            "schema_version": 3,
            "updated_at_run": "003",
            "generation": 0,
            "summary": "Two direct comparisons cover the filtered hypothesis.",
            "promising_regions": [],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [covered_belief],
        },
        registry,
        evidence_ledger,
    ) == []
    weak_belief = copy.deepcopy(covered_belief)
    weak_belief["confidence"] = "med"
    errors = validate_experience(
        {
            "schema_version": 3,
            "updated_at_run": "003",
            "generation": 0,
            "summary": "Pruning requires high-confidence comparator coverage.",
            "promising_regions": [],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [weak_belief],
        },
        registry,
        evidence_ledger,
    )
    assert any("pruned requires" in error for error in errors), errors

    print("P2 semantic background, edge, belief, and policy-state checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
