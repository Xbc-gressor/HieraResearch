from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from background_contract import derive_hypothesis_selection, validate_registry  # noqa: E402
from semantic_search import (  # noqa: E402
    build_proposal_set,
    validate_proposal_set,
)
from semantic_space import (  # noqa: E402
    SemanticSpaceError,
    catalog_receipt,
    load_catalog,
    resolve_dimension_catalog,
    resolve_dimension_strategy,
    validate_catalog,
)
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


if __name__ == "__main__":
    unittest.main()
