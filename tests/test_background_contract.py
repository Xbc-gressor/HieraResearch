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
    validate_experience_replacement,
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
from tests.fixtures import (  # noqa: E402
    attach_matched_transfer,
    belief_ledger,
    fixture_registry,
)


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


class BaselineMechanismTests(unittest.TestCase):
    """A mechanism the baseline already applies is not a searchable contrast."""

    @staticmethod
    def _inventory(registry: dict, **overrides: object) -> dict:
        dimensions = {}
        for dimension in registry["dimensions"]:
            baseline = _hypothesis(registry, dimension["baseline_hypothesis_id"])
            dimensions[dimension["id"]] = {
                "interventions": list(baseline["scope"]["interventions"]),
                "citations": ["train.py:1"],
            }
        inventory = {
            "schema_version": 1,
            "kind": "baseline_mechanism_inventory",
            "entrypoint": {"path": "tasks/toy/train.py", "sha256": "sha256:" + "a" * 64},
            "dimensions": dimensions,
        }
        inventory.update(overrides)
        return inventory

    def test_alternative_sharing_a_baseline_mechanism_is_rejected(self) -> None:
        registry = fixture_registry()
        baseline = _hypothesis(registry, "hyp-data-raw")
        alternative = _hypothesis(registry, "hyp-data-filtered")
        alternative["scope"]["interventions"] = list(baseline["scope"]["interventions"])

        errors = validate_registry(registry)

        self.assertTrue(
            any(
                "hyp-data-filtered" in error and "is not a contrast" in error
                for error in errors
            ),
            errors,
        )

    def test_cross_dimension_baseline_collision_is_rejected(self) -> None:
        registry = fixture_registry()
        owner = registry["dimensions"][0]
        baseline = _hypothesis(registry, owner["baseline_hypothesis_id"])
        stolen = baseline["scope"]["interventions"][0]
        alternative = _hypothesis(registry, "hyp-model-multibranch")
        alternative["scope"]["interventions"] = [stolen]

        errors = validate_registry(registry)

        self.assertTrue(
            any(
                f"the {owner['id']} baseline" in error and stolen in error
                for error in errors
            ),
            errors,
        )

    def test_registry_without_an_inventory_is_unaffected(self) -> None:
        self.assertEqual(validate_registry(fixture_registry()), [])

    def test_consistent_inventory_validates(self) -> None:
        registry = fixture_registry()

        self.assertEqual(
            validate_registry(registry, baseline_mechanisms=self._inventory(registry)),
            [],
        )

    def test_inventory_must_agree_with_the_registry_and_cite_lines(self) -> None:
        registry = fixture_registry()
        dimension_id = registry["dimensions"][0]["id"]

        undeclared = self._inventory(registry)
        undeclared["dimensions"][dimension_id]["interventions"].append("qk-layernorm")
        uncovered = self._inventory(registry)
        uncovered["dimensions"].pop(dimension_id)
        uncited = self._inventory(registry)
        uncited["dimensions"][dimension_id]["citations"] = ["train.py"]
        unknown = self._inventory(registry)
        unknown["dimensions"]["dim-not-real"] = {
            "interventions": ["whatever"],
            "citations": ["train.py:1"],
        }

        cases = [
            # Every mechanism the entrypoint already applies must be declared,
            # or a "new" hypothesis could silently restate the baseline.
            ("does not declare interventions ['qk-layernorm']", undeclared),
            ("does not cover dimensions", uncovered),
            ("<file>:<line>", uncited),
            ("unknown dimensions ['dim-not-real']", unknown),
            ("schema_version must be 1", self._inventory(registry, schema_version=2)),
            ("kind must be", self._inventory(registry, kind="something_else")),
            (
                "entrypoint.sha256 must be a sha256 digest",
                self._inventory(
                    registry,
                    entrypoint={"path": "tasks/toy/train.py", "sha256": "nope"},
                ),
            ),
        ]
        for needle, inventory in cases:
            with self.subTest(needle=needle):
                errors = validate_registry(registry, baseline_mechanisms=inventory)
                self.assertTrue(any(needle in error for error in errors), errors)


class GuidanceGateTests(unittest.TestCase):
    """Literature may only bind selection when its evidence actually earns it."""

    def test_valid_binding_only_deprioritizes_direct_match(self) -> None:
        registry = _binding_registry()
        self.assertEqual(validate_registry(registry), [])
        selection = derive_hypothesis_selection(registry)
        self.assertEqual(selection["hyp-data-filtered"]["selection_status"], "deprioritized")
        # Binding is scoped: an adjacent mechanism is untouched.
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

    def test_binding_requires_probe_credibility_and_contained_scope(self) -> None:
        def build(**changes: object) -> dict:
            registry = _binding_registry(with_probe=changes.pop("with_probe", True))
            registry["guidance"][0].update(changes.pop("guidance", {}))
            registry["sources"][0].update(changes.pop("source", {}))
            for key, value in changes.items():
                registry[key] = value
            return registry

        overbroad = build()
        overbroad["guidance"][0]["scope"]["model_families"] = ["*"]
        cases = [
            # A binding claim must stay falsifiable via an out-of-scope probe.
            ("requires an out-of-scope scope_probe", build(with_probe=False)),
            # Unverified evidence may warn, never restrict.
            (
                "may only caution",
                build(guidance={"literature_credibility": "unverified"}),
            ),
            # Only inspected primary empirical work can bind.
            ("primary empirical source", build(source={"type": "web_lead"})),
            (
                "primary empirical source",
                build(source={"publication_status": "withdrawn_or_retracted"}),
            ),
            # Guidance may not claim more scope than its source studied.
            ("directly contains the guidance scope", overbroad),
        ]
        for needle, registry in cases:
            with self.subTest(needle=needle):
                errors = validate_registry(registry)
                self.assertTrue(any(needle in error for error in errors), errors)

    def test_exclusion_requires_independently_reproduced_evidence(self) -> None:
        """Excluding a mechanism outright is the strongest claim available."""
        corroborated = _binding_registry()
        corroborated["guidance"][0].update(
            {"effect": "exclude", "literature_credibility": "corroborated"}
        )
        two_sources = copy.deepcopy(corroborated)
        second = copy.deepcopy(two_sources["sources"][0])
        second.update({"id": "src-02", "url": "https://example.test/second-study"})
        two_sources["sources"].append(second)
        two_sources["guidance"][0]["evidence"] = [
            {"source_id": "src-01", "role": "supports"},
            {"source_id": "src-02", "role": "supports"},
        ]
        preliminary = _binding_registry()
        preliminary["guidance"][0]["effect"] = "exclude"

        cases = [
            ("requires corroborated or replicated", preliminary),
            ("requires two directly scoped", corroborated),
            ("independent reproduction", two_sources),
        ]
        for needle, registry in cases:
            with self.subTest(needle=needle):
                errors = validate_registry(registry)
                self.assertTrue(any(needle in error for error in errors), errors)

    def test_credibility_claims_must_be_backed_by_sources(self) -> None:
        duplicate = fixture_registry()
        second = copy.deepcopy(duplicate["sources"][0])
        second["id"] = "src-02"
        duplicate["sources"].append(second)

        replicated = fixture_registry()
        _hypothesis(replicated, "hyp-data-filtered")["literature_credibility"] = "replicated"

        for needle, registry in (
            ("duplicate canonical work", duplicate),
            ("no independently reproduced", replicated),
        ):
            with self.subTest(needle=needle):
                errors = validate_registry(registry)
                self.assertTrue(any(needle in error for error in errors), errors)


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
            "direct_tuned_edges": 2,
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 0,
            "crash_edges": 0,
        },
        "confidence": "high",
        "uncertainty": "Implementation differences remain confounded with each semantic change.",
        "reopen_when": "A later direct comparison improves over its parent.",
    }


def _lightly_tuned_covered_entry() -> dict:
    """Cites the three lightly tuned comparisons of `_lightly_tuned_ledger`."""
    return {
        "target_id": "hyp-data-filtered",
        "evaluation_state": "comparator_covered",
        "assessment": "unpromising",
        "recommended_status": "pruned",
        "claim": "All three lightly tuned comparisons were worse than their matched baseline parents.",
        "evidence_run_ids": ["000", "001", "002", "003", "004"],
        "evidence_edge_ids": ["sedge-000-001", "sedge-002-003", "sedge-004-005"],
        "comparator_coverage": {
            "direct_tuned_edges": 0,
            "direct_lightly_tuned_edges": 3,
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 0,
            "crash_edges": 0,
        },
        "confidence": "high",
        "uncertainty": "Each child stopped short of full tuning, so residual tuning headroom remains.",
        "reopen_when": "A later direct comparison improves over its parent.",
    }


def _observed_entry() -> dict:
    return {
        "target_id": "hyp-data-filtered",
        "evaluation_state": "observed",
        "assessment": "mixed",
        "recommended_status": "active",
        "claim": "One comparison is insufficient for a directional belief.",
        "evidence_run_ids": ["000", "001"],
        "evidence_edge_ids": ["sedge-000-001"],
        "comparator_coverage": {
            "direct_tuned_edges": 1,
            "direct_noncrash_edges": 0,
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


def _lightly_tuned_ledger(registry: dict) -> dict:
    """Three matched comparisons against ``hyp-data-filtered``, lightly tuned.

    Same shape as ``belief_ledger`` but with three pairs whose children carry
    ``evaluation_depth: tuned_lightly``: zero direct tuned edges and three
    direct edges at tuned_lightly or deeper, so the contradiction depth bar
    clears on the lightly-tuned side alone.
    """
    baseline = complete_point(registry)
    filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    records = []
    for index in range(3):
        base = {
            "run_id": f"{2 * index:03d}", "source_run_ids": [],
            "semantic_point": baseline, "semantic_edges": [], "status": "keep",
            "final_best_score": 0.40 + index * 0.01,
            "evaluation_depth": "tuned", "dag_revision": 2 * index + 1,
        }
        child = {
            "run_id": f"{2 * index + 1:03d}", "source_run_ids": [base["run_id"]],
            "semantic_point": filtered, "status": "discard",
            "final_best_score": 0.50 + index * 0.01,
            "evaluation_depth": "tuned_lightly", "dag_revision": 2 * index + 2,
        }
        records.append(base)
        child["semantic_edges"] = build_semantic_edges(records, child)
        records.append(child)
        attach_matched_transfer(base, child)
    return {"records": records, "dag_revision": 6}


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
            "direct_tuned_edges": 0,
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 0,
            "crash_edges": 1,
        },
        "confidence": "low",
        "uncertainty": "A crash cannot contradict the hypothesis by itself.",
    }


class ExperienceSchema3Tests(unittest.TestCase):
    """A belief may claim only as much as its comparator receipts support."""

    def _reject(
        self,
        entry: dict,
        needle: str,
        *,
        ledger: dict | None = None,
        field: str = "hypothesis_evidence",
        updated_at_run: str | None = None,
    ) -> None:
        registry = fixture_registry()
        experience = _base_experience()
        experience[field] = [entry]
        if updated_at_run is not None:
            experience["updated_at_run"] = updated_at_run
        errors = validate_experience(
            experience, registry, ledger if ledger is not None else belief_ledger(registry)
        )
        self.assertTrue(any(needle in error for error in errors), errors)

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
                "direct_tuned_edges": 0,
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
        deprioritized = _comparator_covered_entry()
        deprioritized["recommended_status"] = "deprioritized"

        cases = [
            ("unevaluated", belief_ledger(registry), "hypothesis_evidence", unevaluated, "003"),
            ("failed", _crash_ledger(registry), "hypothesis_evidence", _failed_entry(), "004"),
            ("observed", belief_ledger(registry), "hypothesis_evidence", _observed_entry(), "003"),
            (
                "comparator_covered pruned",
                belief_ledger(registry),
                "hypothesis_evidence",
                _comparator_covered_entry(),
                "003",
            ),
            (
                "comparator_covered deprioritized",
                belief_ledger(registry),
                "hypothesis_evidence",
                deprioritized,
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

    def test_accepts_depth_bar_via_lightly_tuned_edges(self) -> None:
        """Three direct edges at tuned_lightly (zero tuned) clear the bar."""
        registry = fixture_registry()
        experience = _base_experience()
        experience["updated_at_run"] = "005"
        experience["hypothesis_evidence"] = [_lightly_tuned_covered_entry()]
        self.assertEqual(
            validate_experience(experience, registry, _lightly_tuned_ledger(registry)),
            [],
        )

    def test_rejects_claims_beyond_their_cited_receipts(self) -> None:
        """Citations must resolve, touch the target, and match recomputation."""
        unknown_target = _comparator_covered_entry()
        unknown_target["target_id"] = "hyp-unknown"

        wrong_edge = _failed_entry()
        wrong_edge.update(
            {
                "target_id": "hyp-valid-cv",
                "evidence_run_ids": [],
                "evidence_edge_ids": ["sedge-000-001"],
                "comparator_coverage": {
                    "direct_tuned_edges": 0,
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
            }
        )

        forged_counts = _comparator_covered_entry()
        forged_counts["comparator_coverage"] = {
            "direct_tuned_edges": 1,
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 0,
            "crash_edges": 0,
        }

        for needle, entry in (
            ("is not a known hypothesis id", unknown_target),
            ("does not touch the target", wrong_edge),
            ("must equal the recomputed coverage", forged_counts),
        ):
            with self.subTest(needle=needle):
                self._reject(entry, needle)

    def test_direction_requires_comparator_coverage(self) -> None:
        """A single or confounded observation cannot establish a direction."""
        for assessment in ("promising", "unpromising"):
            for confidence in ("low", "med", "high"):
                with self.subTest(assessment=assessment, confidence=confidence):
                    entry = _observed_entry()
                    entry.update({"assessment": assessment, "confidence": confidence})
                    self._reject(entry, "requires comparator_covered")

    def test_rejects_crash_only_contradiction(self) -> None:
        """A crash informs feasibility; it never contradicts a hypothesis."""
        registry = fixture_registry()
        ledger = _crash_ledger(registry)
        for mutate in (
            {"assessment": "unpromising"},
            {"recommended_status": "deprioritized"},
        ):
            with self.subTest(mutate=mutate):
                entry = _failed_entry()
                entry.update(mutate)
                self._reject(
                    entry,
                    "must keep assessment unknown",
                    ledger=ledger,
                    updated_at_run="004",
                )

    def test_rejects_contraction_gate_violations(self) -> None:
        """Deprioritizing and pruning each need their own evidence threshold."""
        deprioritized = _comparator_covered_entry()
        deprioritized["recommended_status"] = "deprioritized"

        def variant(base: dict, **changes: object) -> dict:
            entry = copy.deepcopy(base)
            for key, value in changes.items():
                if value is None:
                    del entry[key]
                else:
                    entry[key] = value
            return entry

        single_edge = variant(
            _observed_entry(),
            recommended_status="deprioritized",
            reopen_when="A later direct comparison improves over its parent.",
        )
        no_edge = variant(
            single_edge,
            evidence_run_ids=["001"],
            evidence_edge_ids=[],
            comparator_coverage={
                "direct_tuned_edges": 0,
                "direct_noncrash_edges": 0,
                "confounded_noncrash_edges": 0,
                "crash_edges": 0,
            },
        )

        cases = [
            # Contraction needs the depth bar, not one direct edge.
            ("comparator_covered", single_edge),
            (
                "at least two direct tuned edges, or at least three direct "
                "edges at tuned_lightly or deeper",
                no_edge,
            ),
            ("deprioritized requires", variant(deprioritized, assessment="mixed")),
            ("deprioritized requires", variant(deprioritized, confidence="low")),
            ("deprioritized requires", variant(deprioritized, reopen_when=None)),
            # Pruning is stricter: high confidence and a covered state.
            ("pruned requires", variant(_comparator_covered_entry(), confidence="med")),
            ("pruned requires", variant(_comparator_covered_entry(), assessment="mixed")),
            (
                "pruned requires",
                variant(_comparator_covered_entry(), evaluation_state="observed"),
            ),
            # Every contraction must remain reversible.
            ("reopen_when", variant(_comparator_covered_entry(), reopen_when=None)),
        ]
        for needle, entry in cases:
            with self.subTest(needle=needle, status=entry.get("recommended_status")):
                self._reject(entry, needle)

    def test_rejects_unbounded_or_malformed_snapshots(self) -> None:
        """The snapshot stays bounded; only schema 3 and 4 are readable."""
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        cases = [
            ("schema_version must be [3, 4]", {"schema_version": 2}),
            (
                "duplicates an earlier",
                {
                    "hypothesis_evidence": [
                        _comparator_covered_entry(),
                        _comparator_covered_entry(),
                    ]
                },
            ),
            (
                "at most 16",
                {
                    "dimension_evidence": [
                        {"target_id": f"dim-extra-{index}"} for index in range(17)
                    ]
                },
            ),
            (
                "at most 32",
                {
                    "hypothesis_evidence": [
                        {"target_id": f"hyp-extra-{index}"} for index in range(33)
                    ]
                },
            ),
        ]
        for needle, changes in cases:
            with self.subTest(needle=needle):
                experience = _base_experience()
                experience.update(changes)
                errors = validate_experience(experience, registry, ledger)
                self.assertTrue(any(needle in error for error in errors), errors)

    def test_replacement_generation_changes_only_when_belief_payload_changes(
        self,
    ) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        prior = _base_experience()
        prior.update({"generation": 7, "dag_revision": ledger["dag_revision"]})
        ledger["experience"] = copy.deepcopy(prior)

        # Processing another DAG cursor with the same bounded belief is an
        # epistemic no-op: run/cursor metadata may advance, but generation does
        # not pretend that a new belief was learned.
        same_belief = copy.deepcopy(prior)
        same_belief.pop("dag_revision")
        self.assertEqual(
            validate_experience_replacement(same_belief, ledger),
            [],
        )

        changed_belief = copy.deepcopy(same_belief)
        changed_belief["summary"] = "The structured bounded belief changed."
        errors = validate_experience_replacement(changed_belief, ledger)
        self.assertTrue(
            any("generation must be 8" in error for error in errors), errors
        )
        changed_belief["generation"] = 8
        self.assertEqual(
            validate_experience_replacement(changed_belief, ledger),
            [],
        )


if __name__ == "__main__":
    unittest.main()
