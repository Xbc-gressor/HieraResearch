from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from background_contract import (  # noqa: E402
    derive_hypothesis_selection,
    validate_experience,
    validate_registry,
)
from semantic_evidence import build_semantic_edges  # noqa: E402
from semantic_search import (  # noqa: E402
    build_proposal_set,
    validate_proposal_set,
)
from semantic_space import (  # noqa: E402
    SemanticSpaceError,
    catalog_receipt,
    complete_point,
    load_catalog,
    resolve_dimension_catalog,
    resolve_dimension_strategy,
    validate_catalog,
)
from tests.p2_fixtures import belief_ledger  # noqa: E402
from validate_background import fixture_registry  # noqa: E402


def _hypothesis(registry: dict, hypothesis_id: str) -> dict:
    for dimension in registry["dimensions"]:
        for hypothesis in dimension["hypotheses"]:
            if hypothesis["id"] == hypothesis_id:
                return hypothesis
    raise AssertionError(f"missing fixture hypothesis {hypothesis_id}")


def _binding_registry(*, with_probe: bool = True) -> dict:
    registry = fixture_registry()
    guidance = registry["guidance"][0]
    guidance.update(
        {
            "section": "deprioritize",
            "effect": "deprioritize",
            "claim": "Filtering under this exact toy scope underperformed.",
            "literature_credibility": "preliminary",
            "credibility_rationale": "One directly matched primary study.",
            "reopen_when": "The intervention or any other scope axis changes.",
        }
    )
    registry["sources"][0]["studied_scope"] = copy.deepcopy(guidance["scope"])
    if with_probe:
        probe = _hypothesis(registry, "hyp-model-multibranch")
        probe["kind"] = "scope_probe"
        probe["probe_for"] = ["g-01"]
    return registry


def _custom_catalog_registry() -> tuple[dict, dict]:
    registry = fixture_registry()
    provenance = "llm-induced:test-fixture"
    renamed = next(
        dimension
        for dimension in registry["dimensions"]
        if dimension["id"] == "dim-initialization-adaptation"
    )
    renamed.update(
        {
            "id": "dim-llm-runtime-regime",
            "definition": "An LLM-induced task-specific decision boundary.",
            "boundary": "Owns only the choices assigned by this generated catalog.",
        }
    )
    for dimension in registry["dimensions"]:
        dimension["catalog_provenance"] = provenance
    catalog = {
        "schema_version": 1,
        "catalog_id": "llm-induced/toy-v1",
        "provenance": provenance,
        "dimensions": [
            {
                "id": dimension["id"],
                "definition": dimension["definition"],
                "boundary": dimension["boundary"],
            }
            for dimension in registry["dimensions"]
        ],
    }
    registry["catalog"] = catalog_receipt(catalog)
    return catalog, registry


class CatalogInjectionTests(unittest.TestCase):
    def test_custom_catalog_can_own_run_dimensions(self) -> None:
        catalog, registry = _custom_catalog_registry()

        self.assertEqual(validate_catalog(catalog), [])
        self.assertEqual(validate_registry(registry, catalog=catalog), [])
        self.assertEqual(
            validate_registry(
                registry,
                catalog=catalog,
                dimension_strategy="llm_induced",
            ),
            [],
        )
        self.assertTrue(validate_registry(registry))

    def test_catalog_resolver_uses_configured_dimension_strategy(self) -> None:
        custom_catalog, _ = _custom_catalog_registry()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            background_path = run_dir / "background.md"
            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {"space_initialization": {"dimension_strategy": "llm_induced"}}
                )
            )
            (run_dir / "dimension_catalog.json").write_text(json.dumps(custom_catalog))

            self.assertEqual(resolve_dimension_strategy(background_path), "llm_induced")
            self.assertEqual(resolve_dimension_catalog(background_path), custom_catalog)

    def test_explicit_catalog_path_takes_priority_over_strategy(self) -> None:
        custom_catalog, _ = _custom_catalog_registry()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            background_path = run_dir / "background.md"
            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {"space_initialization": {"dimension_strategy": "llm_induced"}}
                )
            )
            explicit = run_dir / "override_catalog.json"
            explicit.write_text(json.dumps(custom_catalog))

            # No run-local dimension_catalog.json exists; the explicit path must
            # still resolve without raising the induced-catalog requirement.
            self.assertEqual(
                resolve_dimension_catalog(background_path, explicit_path=explicit),
                custom_catalog,
            )

    def test_catalog_resolver_defaults_and_rejects_missing_induced_catalog(self) -> None:
        template = json.loads((ROOT / "tasks" / "framework_cfg.example.json").read_text())
        self.assertEqual(
            template["space_initialization"]["dimension_strategy"], "catalog_subset"
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            background_path = run_dir / "background.md"
            self.assertEqual(resolve_dimension_strategy(background_path), "catalog_subset")
            self.assertEqual(resolve_dimension_catalog(background_path), load_catalog())

            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {"space_initialization": {"dimension_strategy": "llm_induced"}}
                )
            )
            with self.assertRaisesRegex(SemanticSpaceError, "llm_induced requires"):
                resolve_dimension_catalog(background_path)

    def test_induced_registry_must_use_every_catalog_dimension_in_order(self) -> None:
        catalog, registry = _custom_catalog_registry()

        omitted = copy.deepcopy(registry)
        omitted["dimensions"].pop()
        errors = validate_registry(
            omitted,
            catalog=catalog,
            dimension_strategy="llm_induced",
        )
        self.assertTrue(any("exactly match" in error for error in errors), errors)

        reordered = copy.deepcopy(registry)
        reordered["dimensions"][0], reordered["dimensions"][1] = (
            reordered["dimensions"][1],
            reordered["dimensions"][0],
        )
        errors = validate_registry(
            reordered,
            catalog=catalog,
            dimension_strategy="llm_induced",
        )
        self.assertTrue(any("ids and order" in error for error in errors), errors)

    def test_registry_rejects_unknown_dimension_strategy(self) -> None:
        catalog, registry = _custom_catalog_registry()

        errors = validate_registry(
            registry,
            catalog=catalog,
            dimension_strategy="unknown",
        )

        self.assertTrue(any("dimension_strategy must be" in error for error in errors), errors)

    def test_catalog_subset_still_accepts_a_catalog_subset(self) -> None:
        catalog, registry = _custom_catalog_registry()
        subset = copy.deepcopy(registry)
        removed = subset["dimensions"].pop()
        removed_hypotheses = {
            hypothesis["id"] for hypothesis in removed["hypotheses"]
        }
        subset["relations"] = [
            relation
            for relation in subset["relations"]
            if not any(
                hypothesis_id in json.dumps(relation)
                for hypothesis_id in removed_hypotheses
            )
            and removed["id"] not in json.dumps(relation)
        ]

        errors = validate_registry(
            subset,
            catalog=catalog,
            dimension_strategy="catalog_subset",
        )
        self.assertEqual(errors, [])

    def test_induced_space_receipt_flows_through_proposal_set(self) -> None:
        catalog, registry = _custom_catalog_registry()

        proposal_set = build_proposal_set(
            registry, {"records": []}, op="fresh", parents=[], max_points=3
        )

        self.assertEqual(proposal_set["space"]["catalog"], catalog_receipt(catalog))
        self.assertEqual(validate_proposal_set(proposal_set), [])


class ProbeReferenceTests(unittest.TestCase):
    def test_scope_probe_rejects_unknown_guidance_reference(self) -> None:
        registry = fixture_registry()
        probe = _hypothesis(registry, "hyp-model-multibranch")
        probe["kind"] = "scope_probe"
        probe["probe_for"] = ["g-99"]

        errors = validate_registry(registry)

        self.assertTrue(
            any("probe_for references unknown guidance id 'g-99'" in error for error in errors),
            errors,
        )

    def test_probe_for_is_rejected_on_non_probe_hypothesis(self) -> None:
        registry = fixture_registry()
        _hypothesis(registry, "hyp-data-filtered")["probe_for"] = ["g-01"]

        errors = validate_registry(registry)

        self.assertTrue(any("only valid for a scope_probe" in error for error in errors), errors)


class GuidanceGateTests(unittest.TestCase):
    def test_binding_guidance_requires_out_of_scope_probe(self) -> None:
        errors = validate_registry(_binding_registry(with_probe=False))
        self.assertTrue(any("requires an out-of-scope scope_probe" in error for error in errors), errors)

    def test_unverified_negative_may_only_caution(self) -> None:
        registry = _binding_registry()
        registry["guidance"][0]["literature_credibility"] = "unverified"
        errors = validate_registry(registry)
        self.assertTrue(any("may only caution" in error for error in errors), errors)

    def test_binding_guidance_requires_primary_empirical_source(self) -> None:
        registry = _binding_registry()
        registry["sources"][0]["type"] = "web_lead"
        errors = validate_registry(registry)
        self.assertTrue(any("primary empirical source" in error for error in errors), errors)

    def test_withdrawn_source_cannot_bind(self) -> None:
        registry = _binding_registry()
        registry["sources"][0]["publication_status"] = "withdrawn_or_retracted"
        errors = validate_registry(registry)
        self.assertTrue(any("primary empirical source" in error for error in errors), errors)

    def test_duplicate_canonical_source_is_rejected(self) -> None:
        registry = fixture_registry()
        duplicate = copy.deepcopy(registry["sources"][0])
        duplicate["id"] = "src-02"
        registry["sources"].append(duplicate)
        errors = validate_registry(registry)
        self.assertTrue(any("duplicate canonical work" in error for error in errors), errors)

    def test_overbroad_guidance_must_be_contained_by_source(self) -> None:
        registry = _binding_registry()
        registry["guidance"][0]["scope"]["model_families"] = ["*"]
        errors = validate_registry(registry)
        self.assertTrue(any("directly contains the guidance scope" in error for error in errors), errors)

    def test_exclusion_requires_corroborated_or_replicated_evidence(self) -> None:
        registry = _binding_registry()
        registry["guidance"][0]["effect"] = "exclude"
        errors = validate_registry(registry)
        self.assertTrue(any("requires corroborated or replicated" in error for error in errors), errors)

    def test_exclusion_requires_two_direct_sources(self) -> None:
        registry = _binding_registry()
        registry["guidance"][0].update(
            {"effect": "exclude", "literature_credibility": "corroborated"}
        )
        errors = validate_registry(registry)
        self.assertTrue(any("requires two directly scoped" in error for error in errors), errors)

    def test_exclusion_requires_independent_reproduction(self) -> None:
        registry = _binding_registry()
        second = copy.deepcopy(registry["sources"][0])
        second.update({"id": "src-02", "url": "https://example.test/second-study"})
        registry["sources"].append(second)
        registry["guidance"][0].update(
            {
                "effect": "exclude",
                "literature_credibility": "corroborated",
                "evidence": [
                    {"source_id": "src-01", "role": "supports"},
                    {"source_id": "src-02", "role": "supports"},
                ],
            }
        )
        errors = validate_registry(registry)
        self.assertTrue(any("independent reproduction" in error for error in errors), errors)

    def test_replicated_hypothesis_requires_reproduced_support(self) -> None:
        registry = fixture_registry()
        _hypothesis(registry, "hyp-data-filtered")["literature_credibility"] = "replicated"
        errors = validate_registry(registry)
        self.assertTrue(any("no independently reproduced" in error for error in errors), errors)

    def test_valid_binding_only_deprioritizes_direct_match(self) -> None:
        registry = _binding_registry()
        self.assertEqual(validate_registry(registry), [])
        selection = derive_hypothesis_selection(registry)
        self.assertEqual(selection["hyp-data-filtered"]["selection_status"], "deprioritized")
        self.assertEqual(selection["hyp-model-multibranch"]["selection_status"], "active")

    def test_caution_does_not_create_binding_guidance(self) -> None:
        registry = _binding_registry()
        registry["guidance"][0].update(
            {"section": "pitfall", "effect": "caution", "literature_credibility": "unverified"}
        )
        self.assertEqual(validate_registry(registry), [])
        selection = derive_hypothesis_selection(registry)["hyp-data-filtered"]
        self.assertEqual(selection["selection_status"], "active")
        self.assertEqual(selection["binding_guidance"], [])
        self.assertEqual(selection["matched_guidance"], [{"id": "g-01", "effect": "caution"}])


def _base_experience() -> dict:
    return {
        "schema_version": 3,
        "updated_at_run": "003",
        "generation": 0,
        "summary": "Bounded two-level interpretation of the comparator ledger.",
        "promising_regions": [],
        "lessons": [],
        "bottlenecks": [],
        "dimension_evidence": [],
        "hypothesis_evidence": [],
    }


def _comparator_covered_entry() -> dict:
    return {
        "target_id": "hyp-data-filtered",
        "evaluation_state": "comparator_covered",
        "assessment": "unpromising",
        "recommended_status": "pruned",
        "claim": "Both direct comparisons were worse than their matched baseline parents.",
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


def _observed_entry() -> dict:
    return {
        "target_id": "hyp-data-filtered",
        "evaluation_state": "observed",
        "assessment": "unpromising",
        "recommended_status": "active",
        "claim": "One direct comparison was worse than its matched baseline parent.",
        "evidence_run_ids": ["000", "001"],
        "evidence_edge_ids": ["sedge-000-001"],
        "comparator_coverage": {
            "direct_noncrash_edges": 1,
            "confounded_noncrash_edges": 0,
            "crash_edges": 0,
        },
        "confidence": "med",
        "uncertainty": "A second matched comparison is still missing.",
    }


def _crash_ledger(registry: dict) -> dict:
    ledger = belief_ledger(registry)
    record = {
        "run_id": "004",
        "source_run_ids": ["002"],
        "semantic_point": complete_point(
            registry, {"dim-data-curation": "hyp-data-filtered"}
        ),
        "status": "crash",
        "final_best_score": None,
        "dag_revision": 5,
    }
    record["semantic_edges"] = build_semantic_edges(ledger["records"], record)
    ledger["records"].append(record)
    ledger["dag_revision"] = 5
    return ledger


def _failed_entry() -> dict:
    return {
        "target_id": "hyp-data-filtered",
        "evaluation_state": "failed",
        "assessment": "unknown",
        "recommended_status": "active",
        "claim": "The only cited comparison attempt crashed before a score.",
        "evidence_run_ids": ["004"],
        "evidence_edge_ids": ["sedge-002-004"],
        "comparator_coverage": {
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 0,
            "crash_edges": 1,
        },
        "confidence": "low",
        "uncertainty": "A crash cannot contradict the hypothesis by itself.",
    }


class ExperienceSchema3Tests(unittest.TestCase):
    def test_accepts_comparator_covered_hypothesis_belief(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        experience = {
            "schema_version": 3,
            "updated_at_run": "003",
            "generation": 0,
            "summary": "Two direct comparisons make the filtered hypothesis eligible for a conservative recommendation.",
            "promising_regions": [],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [{
                "target_id": "hyp-data-filtered",
                "evaluation_state": "comparator_covered",
                "assessment": "unpromising",
                "recommended_status": "pruned",
                "claim": "Both direct comparisons were worse than their matched baseline parents.",
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
            }],
        }
        self.assertEqual(validate_experience(experience, registry, ledger), [])

    def test_accepts_valid_entries_for_each_evaluation_state(self) -> None:
        registry = fixture_registry()
        unevaluated = {
            "target_id": "hyp-valid-cv",
            "evaluation_state": "unevaluated",
            "assessment": "unknown",
            "recommended_status": "active",
            "claim": "No completed observation cites this target yet.",
            "evidence_run_ids": [],
            "evidence_edge_ids": [],
            "comparator_coverage": {
                "direct_noncrash_edges": 0,
                "confounded_noncrash_edges": 0,
                "crash_edges": 0,
            },
            "confidence": "low",
            "uncertainty": "The target has no attributed observation.",
        }
        dimension = _comparator_covered_entry()
        dimension["target_id"] = "dim-data-curation"
        dimension["claim"] = "Both matched data-curation changes worsened the score."
        cases = [
            ("unevaluated hypothesis", belief_ledger(registry), "hypothesis_evidence", unevaluated, "003"),
            ("failed hypothesis", _crash_ledger(registry), "hypothesis_evidence", _failed_entry(), "004"),
            ("observed hypothesis", belief_ledger(registry), "hypothesis_evidence", _observed_entry(), "003"),
            (
                "comparator_covered hypothesis",
                belief_ledger(registry),
                "hypothesis_evidence",
                _comparator_covered_entry(),
                "003",
            ),
            (
                "comparator_covered dimension",
                belief_ledger(registry),
                "dimension_evidence",
                dimension,
                "003",
            ),
        ]
        for name, ledger, field, entry, updated_at_run in cases:
            with self.subTest(case=name):
                experience = _base_experience()
                experience["updated_at_run"] = updated_at_run
                experience[field] = [entry]
                self.assertEqual(validate_experience(experience, registry, ledger), [])

    def test_accepts_single_direct_edge_deprioritized_entry(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        entry = _observed_entry()
        entry.update(
            {
                "recommended_status": "deprioritized",
                "reopen_when": "A later direct comparison improves over its parent.",
            }
        )
        experience = _base_experience()
        experience["hypothesis_evidence"] = [entry]
        self.assertEqual(validate_experience(experience, registry, ledger), [])

    def _reject(self, entry: dict, needle: str, *, ledger: dict | None = None, field: str = "hypothesis_evidence") -> None:
        registry = fixture_registry()
        experience = _base_experience()
        experience[field] = [entry]
        errors = validate_experience(
            experience, registry, ledger if ledger is not None else belief_ledger(registry)
        )
        self.assertTrue(any(needle in error for error in errors), errors)

    def test_rejects_schema_version_2(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        experience = _base_experience()
        experience["schema_version"] = 2
        errors = validate_experience(experience, registry, ledger)
        self.assertTrue(any("schema_version must be 3" in error for error in errors), errors)

    def test_rejects_unknown_target(self) -> None:
        entry = _comparator_covered_entry()
        entry["target_id"] = "hyp-unknown"
        self._reject(entry, "is not a known hypothesis id")

    def test_rejects_edge_that_does_not_touch_target(self) -> None:
        entry = _failed_entry()
        entry.update(
            {
                "target_id": "hyp-valid-cv",
                "evidence_run_ids": [],
                "evidence_edge_ids": ["sedge-000-001"],
                "comparator_coverage": {
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
            }
        )
        self._reject(entry, "does not touch the target")

    def test_rejects_forged_comparator_counts(self) -> None:
        entry = _comparator_covered_entry()
        entry["comparator_coverage"] = {
            "direct_noncrash_edges": 1,
            "confounded_noncrash_edges": 0,
            "crash_edges": 0,
        }
        self._reject(entry, "must equal the recomputed coverage")

    def test_rejects_crash_only_contradiction(self) -> None:
        registry = fixture_registry()
        ledger = _crash_ledger(registry)
        for mutate in (
            {"assessment": "unpromising"},
            {"recommended_status": "deprioritized"},
        ):
            with self.subTest(mutate=mutate):
                entry = _failed_entry()
                entry.update(mutate)
                experience = _base_experience()
                experience["updated_at_run"] = "004"
                experience["hypothesis_evidence"] = [entry]
                errors = validate_experience(experience, registry, ledger)
                self.assertTrue(
                    any("must keep assessment unknown" in error for error in errors),
                    errors,
                )

    def test_rejects_high_confidence_without_comparator_coverage(self) -> None:
        for assessment in ("promising", "unpromising"):
            with self.subTest(assessment=assessment):
                entry = _observed_entry()
                entry.update({"assessment": assessment, "confidence": "high"})
                self._reject(entry, "requires comparator_covered")

    def test_rejects_deprioritized_without_direct_edge(self) -> None:
        entry = _observed_entry()
        entry.update(
            {
                "evidence_run_ids": ["001"],
                "evidence_edge_ids": [],
                "comparator_coverage": {
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
                "recommended_status": "deprioritized",
                "reopen_when": "A later direct comparison improves over its parent.",
            }
        )
        self._reject(entry, "at least one direct non-crash edge")

    def test_rejects_deprioritized_gate_violations(self) -> None:
        base = _observed_entry()
        base.update(
            {
                "recommended_status": "deprioritized",
                "reopen_when": "A later direct comparison improves over its parent.",
            }
        )
        mutations = []
        mixed = copy.deepcopy(base)
        mixed["assessment"] = "mixed"
        mutations.append(("assessment not unpromising", mixed))
        low = copy.deepcopy(base)
        low["confidence"] = "low"
        mutations.append(("confidence not med or high", low))
        no_reopen = copy.deepcopy(base)
        del no_reopen["reopen_when"]
        mutations.append(("missing reopen_when", no_reopen))
        for name, entry in mutations:
            with self.subTest(case=name):
                self._reject(entry, "deprioritized requires")

    def test_rejects_pruned_gate_violations(self) -> None:
        mutations = []
        med = _comparator_covered_entry()
        med["confidence"] = "med"
        mutations.append(("confidence not high", med))
        mixed = _comparator_covered_entry()
        mixed["assessment"] = "mixed"
        mutations.append(("assessment not unpromising", mixed))
        observed = _comparator_covered_entry()
        observed["evaluation_state"] = "observed"
        mutations.append(("state not comparator_covered", observed))
        for name, entry in mutations:
            with self.subTest(case=name):
                self._reject(entry, "pruned requires")

    def test_rejects_duplicate_target(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        experience = _base_experience()
        experience["hypothesis_evidence"] = [
            _comparator_covered_entry(),
            _comparator_covered_entry(),
        ]
        errors = validate_experience(experience, registry, ledger)
        self.assertTrue(any("duplicates an earlier" in error for error in errors), errors)

    def test_rejects_pruned_without_reopen_condition(self) -> None:
        entry = _comparator_covered_entry()
        del entry["reopen_when"]
        self._reject(entry, "reopen_when")

    def test_rejects_more_than_16_dimension_entries(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        experience = _base_experience()
        experience["dimension_evidence"] = [
            {"target_id": f"dim-extra-{index}"} for index in range(17)
        ]
        errors = validate_experience(experience, registry, ledger)
        self.assertTrue(any("at most 16" in error for error in errors), errors)

    def test_rejects_more_than_32_hypothesis_entries(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        experience = _base_experience()
        experience["hypothesis_evidence"] = [
            {"target_id": f"hyp-extra-{index}"} for index in range(33)
        ]
        errors = validate_experience(experience, registry, ledger)
        self.assertTrue(any("at most 32" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
