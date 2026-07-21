from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from background_contract import derive_hypothesis_selection, validate_registry  # noqa: E402
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
