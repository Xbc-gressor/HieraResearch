#!/usr/bin/env python3
"""Focused deterministic checks for the P1 semantic-search-space contract."""

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
from search_space_state import empty_search_space_state
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


def background_text(registry: dict) -> str:
    lines = [
        "# Background — P1 fixture",
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


def policy_receipt(op: str, parents: list[str], point: dict) -> dict:
    return {
        "schema_version": 1,
        "space_revision": point["space_revision"],
        "proposal_set_revision": "sha256:" + "0" * 64,
        "policy": {
            "name": "coverage",
            "config": {
                "coverage_weight": 0.1,
                "cost_weight": 0.2,
                "uncertainty_weight": 0.5,
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
        "policy_receipt": policy_receipt(op, parents, point),
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
        policy_path.write_text(json.dumps(policy_receipt("improve", ["000"], coverage_point)))
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
                    "schema_version": 2,
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

    # P1 does not smuggle P2 dimension/hypothesis belief extraction into the
    # experience snapshot.
    errors = validate_experience(
        {"dimension_evidence": []}, registry, ledger
    )
    assert any("P2" in error for error in errors), errors
    assert validate_experience(
        {
            "schema_version": 2,
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
        },
        registry,
        ledger,
    ) == []
    stale_view_ledger = copy.deepcopy(ledger)
    stale_view_ledger["dag_revision"] = 5
    assert validate_experience(
        {
            "schema_version": 2,
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

    print("P1 semantic background, point, lineage, and policy checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
