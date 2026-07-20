#!/usr/bin/env python3
"""Focused checks for the background registry and tf-* lineage join."""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import tempfile
from pathlib import Path

from background_contract import (
    cmd_directions,
    cmd_preflight,
    derive_direction_selection,
    derive_lineage,
    load_registry,
    scope_relation,
    validate_background_markdown,
    validate_experience,
    validate_registry,
)
from search_backends import add_visit, new_manifest


def _scope(model: str, data: str, metric: str, intervention: str, protocol: str) -> dict:
    return {
        "model_families": [model],
        "data_regimes": [data],
        "metrics": [metric],
        "interventions": [intervention],
        "evaluation_protocols": [protocol],
    }


TREE_SCOPE = _scope(
    "regularized_trees", "small_binary_tabular", "validation_loss", "l2_regularization", "heldout_split"
)
FEATURE_SCOPE = _scope(
    "linear_models", "dense_redundant_tabular", "validation_loss", "stable_feature_selection", "repeated_splits"
)


REGISTRY = {
    "schema_version": 2,
    "directions": [
        {
            "id": "tf-01",
            "title": "regularized trees",
            "claim": "Regularized trees are competitive on this task.",
            "kind": "evidence_prior",
            "claim_scope": "Small binary tabular datasets scored by validation loss.",
            "scope": TREE_SCOPE,
            "required_comparisons": ["Same split and budget against the linear baseline."],
            "reopen_when": "A different regularization regime or dataset class is tested.",
            "literature_credibility": "preliminary",
            "credibility_rationale": "One direct preprint and released code.",
            "testable_expectation": "Beat the linear baseline without widening the gap.",
            "evidence": [
                {"source_id": "src-01", "role": "supports"},
                {"source_id": "src-02", "role": "context"},
            ],
        },
        {
            "id": "tf-02",
            "title": "feature selection",
            "claim": "Stable feature selection reduces validation loss.",
            "kind": "evidence_prior",
            "claim_scope": "Dense tabular datasets with redundant numeric features.",
            "scope": FEATURE_SCOPE,
            "required_comparisons": ["Selected features versus all features on matched splits."],
            "reopen_when": "A stability-aware selector or a different feature regime is tested.",
            "literature_credibility": "corroborated",
            "credibility_rationale": "Two independent benchmark studies agree.",
            "testable_expectation": "Reduce loss across more than one split.",
            "evidence": [{"source_id": "src-03", "role": "supports"}],
        },
    ],
    "guidance": [],
    "sources": [
        {
            "id": "src-01",
            "type": "paper",
            "title": "Tree study",
            "url": "https://example.test/tree-paper",
            "publication_status": "preprint_only",
            "validation_status": "claim_only",
            "studied_scope": TREE_SCOPE,
        },
        {
            "id": "src-02",
            "type": "official_code",
            "title": "Tree implementation",
            "url": "https://example.test/tree-code",
            "publication_status": "not_applicable",
            "validation_status": "artifact_available",
            "studied_scope": TREE_SCOPE,
        },
        {
            "id": "src-03",
            "type": "benchmark",
            "title": "Feature benchmark",
            "url": "https://example.test/feature-benchmark",
            "publication_status": "peer_reviewed",
            "validation_status": "independently_reproduced",
            "studied_scope": FEATURE_SCOPE,
        },
    ],
}


def _background_text(registry: dict) -> str:
    guidance = [item for item in registry.get("guidance", []) if isinstance(item, dict)]
    pitfalls = [
        f"- `{item['id']}` — {item['claim']}"
        for item in guidance
        if item.get("section") == "pitfall"
    ] or ["- `operational` — No additional literature-derived pitfall."]
    parts = ["# Background — toy", "", "## Pitfalls", *pitfalls, ""]
    deprioritize = [
        f"- `{item['id']}` — {item['claim']}"
        for item in guidance
        if item.get("section") == "deprioritize"
    ]
    if deprioritize:
        parts.extend(["## Deprioritize", *deprioritize, ""])
    parts.extend(
        [
            "## Direction registry",
            "```json",
            json.dumps(registry, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(parts)


def main() -> int:
    assert validate_registry(REGISTRY) == []

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "background.md"
        path.write_text(_background_text(REGISTRY))
        assert validate_background_markdown(path, REGISTRY) == []
        path.write_text(
            _background_text(REGISTRY).replace(
                "- `operational` — No additional literature-derived pitfall.",
                "- Resampling universally hurts.",
            )
        )
        errors = validate_background_markdown(path, REGISTRY)
        assert any("must start with a registered" in error for error in errors), errors

    legacy = copy.deepcopy(REGISTRY)
    legacy["schema_version"] = 1
    legacy.pop("guidance")
    for direction in legacy["directions"]:
        for field in (
            "kind",
            "claim_scope",
            "scope",
            "required_comparisons",
            "reopen_when",
        ):
            direction.pop(field)
    for source in legacy["sources"]:
        source.pop("studied_scope")
    assert validate_registry(legacy) == []

    missing_scope = copy.deepcopy(REGISTRY)
    missing_scope["sources"][0].pop("studied_scope")
    missing_scope["directions"][0]["scope"].pop("metrics")
    errors = validate_registry(missing_scope)
    assert any("studied_scope" in error for error in errors), errors
    assert any("scope.metrics" in error for error in errors), errors

    assert scope_relation(TREE_SCOPE, TREE_SCOPE) == "direct"
    broader_tree = copy.deepcopy(TREE_SCOPE)
    broader_tree["model_families"].append("unregularized_trees")
    assert scope_relation(TREE_SCOPE, broader_tree) == "partial"
    assert scope_relation(TREE_SCOPE, FEATURE_SCOPE) == "mismatch"

    mixed_wildcard = copy.deepcopy(REGISTRY)
    mixed_wildcard["directions"][0]["scope"]["model_families"] = ["*", "xgboost"]
    errors = validate_registry(mixed_wildcard)
    assert any("wildcard must be the only tag" in error for error in errors), errors

    misplaced_probe = copy.deepcopy(REGISTRY)
    misplaced_probe["directions"][0]["probe_for"] = ["g-01"]
    errors = validate_registry(misplaced_probe)
    assert any("only valid for a scope_probe" in error for error in errors), errors

    binding_without_probe = copy.deepcopy(REGISTRY)
    binding_without_probe["guidance"] = [
        {
            "id": "g-01",
            "section": "deprioritize",
            "effect": "deprioritize",
            "claim": "This exact regularized-tree setup underperformed.",
            "scope": TREE_SCOPE,
            "literature_credibility": "preliminary",
            "credibility_rationale": "One directly matched study.",
            "reopen_when": "The model, intervention, data, metric, or protocol changes.",
            "evidence": [{"source_id": "src-01", "role": "supports"}],
        }
    ]
    errors = validate_registry(binding_without_probe)
    assert any("requires an out-of-scope scope_probe" in error for error in errors), errors

    weak_binding = copy.deepcopy(binding_without_probe)
    weak_binding["guidance"][0]["literature_credibility"] = "unverified"
    errors = validate_registry(weak_binding)
    assert any("may only caution" in error for error in errors), errors

    nonprimary_binding = copy.deepcopy(binding_without_probe)
    nonprimary_binding["sources"][0]["type"] = "web_lead"
    errors = validate_registry(nonprimary_binding)
    assert any("primary empirical source" in error for error in errors), errors

    duplicate_source = copy.deepcopy(REGISTRY)
    duplicate_source["sources"][1]["url"] = duplicate_source["sources"][0]["url"]
    errors = validate_registry(duplicate_source)
    assert any("duplicate canonical work" in error for error in errors), errors

    binding_with_probe = copy.deepcopy(binding_without_probe)
    binding_with_probe["directions"][1]["kind"] = "scope_probe"
    binding_with_probe["directions"][1]["probe_for"] = ["g-01"]
    assert validate_registry(binding_with_probe) == []
    selection = derive_direction_selection(binding_with_probe)
    assert selection["tf-01"]["selection_status"] == "deprioritized", selection
    assert selection["tf-02"]["selection_status"] == "active", selection

    caution_only = copy.deepcopy(binding_with_probe)
    caution_only["guidance"][0]["effect"] = "caution"
    caution_selection = derive_direction_selection(caution_only)
    assert caution_selection["tf-01"]["selection_status"] == "active"
    assert caution_selection["tf-01"]["binding_guidance"] == []
    assert caution_selection["tf-01"]["matched_guidance"] == [
        {"id": "g-01", "effect": "caution"}
    ]

    matched_exclusion = copy.deepcopy(binding_with_probe)
    matched_exclusion["sources"][1]["type"] = "benchmark"
    matched_exclusion["sources"][1]["publication_status"] = "peer_reviewed"
    matched_exclusion["sources"][1]["validation_status"] = "independently_reproduced"
    matched_exclusion["guidance"][0]["effect"] = "exclude"
    matched_exclusion["guidance"][0]["literature_credibility"] = "replicated"
    matched_exclusion["guidance"][0]["evidence"].append(
        {"source_id": "src-02", "role": "supports"}
    )
    assert validate_registry(matched_exclusion) == []
    assert derive_direction_selection(matched_exclusion)["tf-01"]["selection_status"] == "excluded"
    reopened_ledger = {
        "experience": {
            "direction_evidence": [
                {
                    "direction_id": "tf-01",
                    "run_status": "supported_here",
                    "claim_coverage": "direct",
                }
            ]
        }
    }
    reopened = derive_direction_selection(matched_exclusion, reopened_ledger)
    assert reopened["tf-01"]["selection_status"] == "active", reopened
    assert reopened["tf-01"]["reopened_by_run_status"] == "supported_here"

    overbroad_guidance = copy.deepcopy(binding_with_probe)
    overbroad_guidance["guidance"][0]["scope"]["model_families"] = ["*"]
    errors = validate_registry(overbroad_guidance)
    assert any("directly contains the guidance scope" in error for error in errors), errors

    # Frozen scope-boundary fixture: a negative result for one learner and
    # global intervention may constrain an exact replication, but it cannot
    # bind a neighboring ensemble using component-level intervention in a
    # different task regime.
    narrow_scope = _scope(
        "learner_a",
        "binary_domain_data",
        "metric_a",
        "global_rebalancing",
        "cross_validation",
    )
    neighboring_scope = _scope(
        "ensemble_b",
        "multiclass_tabular",
        "aggregate_balanced_metric",
        "component_level_rebalancing",
        "heldout_aggregate",
    )
    replay = {
        "schema_version": 2,
        "directions": [
            {
                "id": "tf-01",
                "title": "Exact narrow-scope replication",
                "claim": "Replicate the paper-scoped learner and intervention.",
                "kind": "evidence_prior",
                "claim_scope": "Learner A on the paper's domain data and protocol.",
                "scope": narrow_scope,
                "required_comparisons": ["Intervention versus no intervention for learner A."],
                "reopen_when": "A materially different learner or intervention protocol is used.",
                "literature_credibility": "preliminary",
                "credibility_rationale": "One directly scoped primary study.",
                "testable_expectation": "The intervention should not improve the matched metric.",
                "evidence": [{"source_id": "src-01", "role": "context"}],
            },
            {
                "id": "tf-02",
                "title": "Neighboring ensemble intervention",
                "claim": "Component-level intervention may improve the task's aggregate metric.",
                "kind": "scope_probe",
                "probe_for": ["g-01"],
                "claim_scope": "Ensemble B on the target multiclass task.",
                "scope": neighboring_scope,
                "required_comparisons": ["Ensemble B versus a matched non-intervention baseline."],
                "reopen_when": "A different component-level intervention is available.",
                "literature_credibility": "unverified",
                "credibility_rationale": "The negative result does not cover this mechanism.",
                "testable_expectation": "Ensemble B improves the aggregate metric over the matched baseline.",
                "evidence": [{"source_id": "src-02", "role": "context"}],
            },
        ],
        "guidance": [
            {
                "id": "g-01",
                "section": "deprioritize",
                "effect": "deprioritize",
                "claim": "Global rebalancing reduced recognition for learner A.",
                "scope": narrow_scope,
                "literature_credibility": "preliminary",
                "credibility_rationale": "Direct within one learner and domain study only.",
                "reopen_when": "The model, sampling mechanism, data, metric, or protocol differs.",
                "evidence": [
                    {"source_id": "src-01", "role": "supports"},
                    {"source_id": "src-02", "role": "context"},
                ],
            }
        ],
        "sources": [
            {
                "id": "src-01",
                "type": "paper",
                "title": "Narrow intervention study",
                "url": "https://example.test/narrow-intervention-study",
                "publication_status": "preprint_only",
                "validation_status": "artifact_available",
                "studied_scope": narrow_scope,
            },
            {
                "id": "src-02",
                "type": "paper",
                "title": "Neighboring mechanisms survey",
                "url": "https://example.test/neighboring-mechanisms-survey",
                "publication_status": "preprint_only",
                "validation_status": "claim_only",
                "studied_scope": _scope(
                    "learner_c",
                    "synthetic_domain_data",
                    "metric_b",
                    "multiple_interventions",
                    "train_validation_test_split",
                ),
            },
        ],
    }
    assert validate_registry(replay) == []
    assert scope_relation(narrow_scope, neighboring_scope) == "mismatch"
    replay_selection = derive_direction_selection(replay)
    assert replay_selection["tf-01"]["selection_status"] == "deprioritized"
    assert replay_selection["tf-02"]["selection_status"] == "active"
    with tempfile.TemporaryDirectory() as tmp:
        replay_path = Path(tmp) / "background.md"
        replay_path.write_text(_background_text(replay))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = cmd_directions(
                argparse.Namespace(background=replay_path, ledger=None, unconsumed=True)
            )
        rendered = json.loads(stdout.getvalue())
        assert rc == 0, rendered
        assert [item["id"] for item in rendered["directions"]] == ["tf-02", "tf-01"]
        assert rendered["directions"][0]["binding_guidance"] == []

    retrieval = new_manifest()
    for source in REGISTRY["sources"]:
        add_visit(
            retrieval,
            url=source["url"],
            lane="grounding",
            backend="fixture",
            view="full_text",
            status="success",
            content="inspected primary source content" * 20,
        )
    assert validate_registry(REGISTRY, retrieval_manifest=retrieval) == []

    mixed_retrieval = copy.deepcopy(retrieval)
    mixed_retrieval["retrieval_condition"] = "mixed"
    errors = validate_registry(REGISTRY, retrieval_manifest=mixed_retrieval)
    assert any("cannot mix frozen and live retrieval" in error for error in errors), errors

    missing_visit = new_manifest()
    add_visit(
        missing_visit,
        url=REGISTRY["sources"][0]["url"],
        lane="novelty",
        backend="fixture",
        view="abstract",
        status="success",
        content="novelty-only source content",
    )
    errors = validate_registry(REGISTRY, retrieval_manifest=missing_visit)
    assert any("was not successfully visited in the grounding lane" in error for error in errors)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "background.md"
        path.write_text(_background_text(REGISTRY))
        assert load_registry(path) == REGISTRY
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = cmd_directions(
                argparse.Namespace(background=path, ledger=None, unconsumed=True)
            )
        rendered = json.loads(stdout.getvalue())
        assert rc == 0 and rendered["scope_contract"] == "explicit", rendered
        assert "refresh_required" not in rendered, rendered
        assert rendered["directions"][0]["claim_scope"], rendered

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "background.md"
        ledger_path = Path(tmp) / "ledger.json"
        path.write_text(_background_text(legacy))
        ledger_path.write_text(json.dumps({
            "records": [
                {"run_id": "001", "source_run_ids": ["tf-01"]},
                {"run_id": "002", "source_run_ids": ["tf-02"]},
            ]
        }))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = cmd_preflight(
                argparse.Namespace(background=path, ledger=ledger_path)
            )
        rendered = json.loads(stdout.getvalue())
        assert rc == 0 and rendered["action"] == "refresh_background", rendered
        assert rendered["reason"] == "legacy_directions_exhausted", rendered

        ledger_path.write_text(json.dumps({
            "records": [{"run_id": "001", "source_run_ids": ["tf-01"]}]
        }))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = cmd_preflight(
                argparse.Namespace(background=path, ledger=ledger_path)
            )
        rendered = json.loads(stdout.getvalue())
        assert rc == 0 and rendered["action"] == "none", rendered

    renumbered = copy.deepcopy(REGISTRY)
    renumbered["directions"][0]["id"] = "tf-02"
    errors = validate_registry(renumbered)
    assert any("must be tf-01" in error for error in errors), errors

    unknown_source = copy.deepcopy(REGISTRY)
    unknown_source["directions"][0]["evidence"] = [
        {"source_id": "src-99", "role": "supports"}
    ]
    errors = validate_registry(unknown_source)
    assert any("unknown source id" in error for error in errors), errors

    unsupported_replication = copy.deepcopy(REGISTRY)
    unsupported_replication["directions"][0]["literature_credibility"] = "replicated"
    errors = validate_registry(unsupported_replication)
    assert any("no independently reproduced supporting source" in error for error in errors), errors

    unsupported_contest = copy.deepcopy(REGISTRY)
    unsupported_contest["directions"][0]["literature_credibility"] = "contested"
    errors = validate_registry(unsupported_contest)
    assert any("cites no contradicting evidence" in error for error in errors), errors

    valid_replication = copy.deepcopy(REGISTRY)
    valid_replication["directions"][1]["literature_credibility"] = "replicated"
    assert validate_registry(valid_replication) == []

    valid_contest = copy.deepcopy(REGISTRY)
    valid_contest["directions"][0]["literature_credibility"] = "contested"
    valid_contest["directions"][0]["evidence"].append(
        {"source_id": "src-03", "role": "contradicts"}
    )
    assert validate_registry(valid_contest) == []

    ledger = {
        "records": [
            {
                "run_id": "000",
                "op": "fresh",
                "source_run_ids": ["tf-01"],
                "status": "keep",
                "final_best_score": 0.5,
            },
            {
                "run_id": "001",
                "op": "improve",
                "source_run_ids": ["000"],
                "status": "keep",
                "final_best_score": 0.4,
            },
            {
                "run_id": "002",
                "op": "fresh",
                "source_run_ids": ["tf-02"],
                "status": "discard",
                "final_best_score": 0.6,
            },
            {
                "run_id": "003",
                "op": "crossover",
                "source_run_ids": ["001", "002"],
                "status": "keep",
                "final_best_score": 0.35,
            },
        ]
    }
    assert validate_registry(REGISTRY, ledger=ledger) == []
    lineage = derive_lineage(REGISTRY, ledger)
    assert lineage["run_origins"] == {
        "000": ["tf-01"],
        "001": ["tf-01"],
        "002": ["tf-02"],
        "003": ["tf-01", "tf-02"],
    }
    tf1 = lineage["directions"]["tf-01"]
    assert [item["run_id"] for item in tf1["direct_runs"]] == ["000"]
    assert [item["run_id"] for item in tf1["single_origin_descendants"]] == ["001"]
    assert [item["run_id"] for item in tf1["combination_runs"]] == ["003"]

    experience = {
        "direction_evidence": [
            {
                "direction_id": "tf-01",
                "literature_credibility": "preliminary",
                "run_status": "supported_here",
                "claim_coverage": "direct",
                "comparison_runs": ["000", "001"],
                "missing_comparisons": [],
                "confidence": "med",
                "direct_runs": ["000"],
                "descendant_runs": ["001"],
                "combination_runs": ["003"],
                "evidence_edges": ["000->001"],
                "rationale": "The direct root survived and its isolated descendant improved.",
            },
            {
                "direction_id": "tf-02",
                "literature_credibility": "corroborated",
                "run_status": "inconclusive",
                "claim_coverage": "partial",
                "comparison_runs": ["002"],
                "missing_comparisons": ["Matched all-features baseline on a second split."],
                "confidence": "low",
                "direct_runs": ["002"],
                "descendant_runs": [],
                "combination_runs": ["003"],
                "evidence_edges": [],
                "rationale": "One discarded implementation and an ambiguous crossover are insufficient.",
            },
        ]
    }
    assert validate_experience(experience, REGISTRY, ledger) == []

    one_run_deadend = copy.deepcopy(experience)
    one_run_deadend["lessons"] = [
        {
            "kind": "deadend",
            "subject": "all tree models",
            "claim": "One tree run lagged.",
            "evidence": ["002"],
            "confidence": "low",
        }
    ]
    errors = validate_experience(one_run_deadend, REGISTRY, ledger)
    assert any("lessons[0].scope must be an object" in error for error in errors), errors
    assert any("requires at least two scored non-crash runs" in error for error in errors)

    scoped_deadend = copy.deepcopy(experience)
    scoped_deadend["lessons"] = [
        {
            "kind": "deadend",
            "subject": "fixed regularized-tree variant",
            "claim": "The fixed variant lagged twice under the same budget.",
            "evidence": ["000", "002"],
            "confidence": "med",
            "scope": TREE_SCOPE,
            "reopen_when": "A different regularizer or budget is available.",
        }
    ]
    assert validate_experience(scoped_deadend, REGISTRY, ledger) == []

    decisive_without_coverage = copy.deepcopy(experience)
    decisive_without_coverage["direction_evidence"][0]["claim_coverage"] = "partial"
    decisive_without_coverage["direction_evidence"][0]["missing_comparisons"] = [
        "Matched linear baseline."
    ]
    errors = validate_experience(decisive_without_coverage, REGISTRY, ledger)
    assert any("requires direct claim coverage" in error for error in errors), errors

    direct_with_crash = copy.deepcopy(experience)
    direct_with_crash["direction_evidence"][0]["comparison_runs"] = ["000", "004"]
    crash_ledger = copy.deepcopy(ledger)
    crash_ledger["records"].append(
        {
            "run_id": "004",
            "op": "improve",
            "source_run_ids": ["001"],
            "status": "crash",
            "final_best_score": None,
        }
    )
    direct_with_crash["direction_evidence"][0]["descendant_runs"].append("004")
    errors = validate_experience(direct_with_crash, REGISTRY, crash_ledger)
    assert any("scored non-crashes" in error for error in errors), errors

    wrong_stamp = copy.deepcopy(experience)
    wrong_stamp["direction_evidence"][0]["literature_credibility"] = "replicated"
    errors = validate_experience(wrong_stamp, REGISTRY, ledger)
    assert any("does not match background.md" in error for error in errors), errors

    wrong_lineage = copy.deepcopy(experience)
    wrong_lineage["direction_evidence"][0]["combination_runs"] = ["002"]
    errors = validate_experience(wrong_lineage, REGISTRY, ledger)
    assert any("is not a DAG-lineage subset" in error for error in errors), errors

    bad_ledger = copy.deepcopy(ledger)
    bad_ledger["records"][0]["source_run_ids"] = ["tf-99"]
    errors = validate_registry(REGISTRY, ledger=bad_ledger)
    assert any("unknown direction tf-99" in error for error in errors), errors

    missing_parent = copy.deepcopy(ledger)
    missing_parent["records"][1]["source_run_ids"] = ["999"]
    lineage = derive_lineage(REGISTRY, missing_parent)
    assert "run 001 references missing parent 999" in lineage["warnings"]
    errors = validate_experience(experience, REGISTRY, missing_parent)
    assert any("lineage: run 001 references missing parent 999" in error for error in errors)

    print("Background contract and lineage checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
