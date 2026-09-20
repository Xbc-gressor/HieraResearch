from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from background_contract import (  # noqa: E402
    _validated_inputs,
    audit_text,
    derive_hypothesis_selection,
    extract_result_numbers,
    item_number_presence,
    mapping_number_presence,
    result_number_matches,
    validate_experience,
    validate_experience_replacement,
    validate_registry,
)
from competition_policy import (  # noqa: E402
    COMPETITION_POLICY_VERSION,
    derive_profile,
)
from search_backends import (  # noqa: E402
    add_visit,
    canonical_key,
    new_manifest,
    verification_statuses,
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
    selected_assignments,
    validate_catalog,
)
from tests.fixtures import (  # noqa: E402
    TOY_SOURCE,
    attach_matched_transfer,
    background_text,
    belief_ledger,
    fixture_registry,
    retrieval_hit_manifest,
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

    def test_exclude_is_rejected_and_deprioritize_stays_eligible(self) -> None:
        """Background guidance cannot delete space; it only reweights it."""
        registry = _binding_registry()
        registry["guidance"][0]["effect"] = "exclude"
        errors = validate_registry(registry)
        self.assertTrue(any("effect must be one of" in error for error in errors), errors)

        registry = _binding_registry()
        self.assertEqual(validate_registry(registry), [])
        proposals = build_proposal_set(
            registry, {"records": []}, op="fresh", parents=[], max_points=16
        )
        self.assertTrue(
            any(
                selected_assignments(item["point"]).get("dim-data-curation")
                == "hyp-data-filtered"
                for item in proposals["proposals"]
            )
        )

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


class RetrievalReceiptTests(unittest.TestCase):
    """The receipt gate and the verification tiers derived from it."""

    def test_source_without_a_receipt_is_rejected(self) -> None:
        registry = fixture_registry()
        errors = validate_registry(registry, retrieval_manifest=new_manifest())
        self.assertTrue(
            any("no retrieval receipt" in error for error in errors), errors
        )

    def test_search_hit_alone_satisfies_the_gate(self) -> None:
        registry = fixture_registry()
        manifest = retrieval_hit_manifest()
        self.assertEqual(validate_registry(registry, retrieval_manifest=manifest), [])

    def test_verification_status_upgrades_with_substantive_visits(self) -> None:
        key = canonical_key(TOY_SOURCE["url"])
        with tempfile.TemporaryDirectory() as tmp:
            manifest = retrieval_hit_manifest()
            self.assertEqual(verification_statuses(manifest)[key], "snippet_only")
            add_visit(
                manifest,
                Path(tmp),
                url=TOY_SOURCE["url"],
                backend="deepxiv",
                view="section",
                section="Method",
                status="success",
                content="inspected primary source" * 30,
            )
            self.assertEqual(verification_statuses(manifest)[key], "section")


MLSP_TASK = ("mle-mlsp-birds", "mlsp-2013-birds")


class CompetitionPolicyTests(unittest.TestCase):
    """Artifact-side competition-policy gates in the validate flow."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name) / "runs" / MLSP_TASK[0] / "t1"
        self.run_dir.mkdir(parents=True)
        profile = derive_profile(*MLSP_TASK)
        self.assertIsNotNone(profile)
        (self.run_dir / "task_identity_profile.json").write_text(json.dumps(profile))
        self.manifest_path = self.run_dir / "background_retrieval.json"
        self.registry = fixture_registry()
        self.url = TOY_SOURCE["url"]

    def _write_manifest(self, manifest: dict) -> dict:
        self.manifest_path.write_text(json.dumps(manifest))
        return manifest

    def _stamp(self, manifest: dict) -> dict:
        manifest["competition_id"] = MLSP_TASK[1]
        manifest["competition_policy_version"] = COMPETITION_POLICY_VERSION
        return manifest

    def _preseeded_manifest(self, *, stamp: bool = True) -> dict:
        manifest = retrieval_hit_manifest()
        add_visit(
            manifest,
            self.run_dir,
            url=self.url,
            backend="web",
            view="full_text",
            status="success",
            content="retained primary text " * 10,
        )
        manifest["visits"][-1]["policy_status"] = "allowed"
        if stamp:
            self._stamp(manifest)
        return self._write_manifest(manifest)

    def _errors(self, manifest: dict) -> list[str]:
        return validate_registry(
            self.registry,
            retrieval_manifest=manifest,
            manifest_dir=self.run_dir,
            manifest_path=self.manifest_path,
        )

    def _validate_run(self, run_dir: Path, *, with_manifest: bool = False) -> list[str]:
        args = argparse.Namespace(
            background=run_dir / "background.md",
            catalog=None,
            ledger=None,
            retrieval_manifest=(
                run_dir / "background_retrieval.json" if with_manifest else None
            ),
            baseline_mechanisms=None,
            number_gate=False,
        )
        _, _, errors = _validated_inputs(args)
        return errors

    def test_preseeded_manifest_without_policy_metadata_fails_closed(self) -> None:
        errors = self._errors(self._preseeded_manifest(stamp=False))
        self.assertTrue(
            any(
                "competition policy: retrieval manifest competition_id is missing"
                in error
                for error in errors
            ),
            errors,
        )
        self.assertTrue(
            any("competition_policy_version" in error for error in errors),
            errors,
        )

        manifest = self._preseeded_manifest(stamp=False)
        manifest["competition_id"] = "spooky-author-identification"
        manifest["competition_policy_version"] = COMPETITION_POLICY_VERSION
        errors = self._errors(self._write_manifest(manifest))
        self.assertTrue(
            any("does not match the resolved competition" in error for error in errors),
            errors,
        )

        self.assertEqual(self._errors(self._preseeded_manifest()), [])

    def test_older_policy_version_fails(self) -> None:
        manifest = self._preseeded_manifest()
        manifest["competition_policy_version"] = COMPETITION_POLICY_VERSION - 1
        errors = self._errors(self._write_manifest(manifest))
        self.assertTrue(
            any(
                "competition_policy_version" in error and "predates" in error
                for error in errors
            ),
            errors,
        )

    def test_receipts_backed_only_by_blocked_or_undetermined_visits_fail(self) -> None:
        for label, status, policy_status in (
            ("blocked visit", "blocked", "blocked"),
            ("undetermined visit", "success", None),
        ):
            with self.subTest(receipt=label):
                manifest = self._stamp(new_manifest())
                add_visit(
                    manifest,
                    self.run_dir,
                    url=self.url,
                    backend="web",
                    view="full_text",
                    status=status,
                    content=(
                        "retained primary text " * 10 if status == "success" else None
                    ),
                )
                if policy_status is not None:
                    manifest["visits"][-1]["policy_status"] = policy_status
                errors = self._errors(self._write_manifest(manifest))
                self.assertEqual(len(errors), 1, errors)
                (error,) = errors
                self.assertIn("competition policy:", error)
                self.assertIn("src-01", error)
                self.assertIn("visit", error)

        # Unreferenced blocked_results are coverage records, not errors.
        manifest = self._preseeded_manifest()
        manifest["rounds"][0]["blocked_results"] = [
            {
                "url": "https://example.test/mlsp-winning-solution",
                "rule_categories": ["slug_sequence", "phrase:winning solution"],
                "basis": "direct",
                "pointer": "audit-0001",
            }
        ]
        self.assertEqual(self._errors(self._write_manifest(manifest)), [])

    def test_background_scan_flags_competition_solution_prose(self) -> None:
        (self.run_dir / "background.md").write_text(
            background_text(self.registry)
            + "\nThe MLSP 2013 birds winning solution used a first-place "
            "ensemble that reached 0.954 private leaderboard.\n"
        )
        errors = self._validate_run(self.run_dir)
        self.assertTrue(
            any(
                "competition policy: background.md matches" in error
                and "winning solution" in error
                for error in errors
            ),
            errors,
        )

    def test_background_scan_allows_generic_leaderboard_discussion(self) -> None:
        (self.run_dir / "background.md").write_text(
            background_text(self.registry)
            + "\nFor the MLSP 2013 birds task we review how the leaderboard "
            "mechanism aggregates per-recording validation metrics into a "
            "single ordering across species.\n"
        )
        self.assertEqual(self._validate_run(self.run_dir), [])

    def test_non_mle_paths_skip_all_policy_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = new_manifest()
            add_visit(
                manifest,
                run_dir,
                url=self.url,
                backend="web",
                view="full_text",
                status="success",
                content="retained primary text " * 10,
            )
            (run_dir / "background_retrieval.json").write_text(json.dumps(manifest))
            (run_dir / "background.md").write_text(
                background_text(self.registry)
                + "\nThe MLSP 2013 birds winning solution used a first-place "
                "ensemble that reached 0.954 private leaderboard.\n"
            )
            self.assertEqual(self._validate_run(run_dir, with_manifest=True), [])


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
            "direct_lightly_tuned_edges": 0,
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
            "direct_lightly_tuned_edges": 0,
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
            "direct_lightly_tuned_edges": 0,
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
                "direct_lightly_tuned_edges": 0,
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
                    "direct_lightly_tuned_edges": 0,
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
            }
        )

        forged_counts = _comparator_covered_entry()
        forged_counts["comparator_coverage"] = {
            "direct_tuned_edges": 1,
            "direct_lightly_tuned_edges": 0,
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
                "direct_lightly_tuned_edges": 0,
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



def _confounded_ledger(registry: dict) -> dict:
    """belief_ledger's shape WITHOUT matched transfers: confounded edges only.

    Both children adding hyp-data-filtered are worse than their baseline
    parents (0.50 > 0.40, 0.52 > 0.41) — two independent negative carrier
    contexts, zero comparator coverage.
    """
    baseline = complete_point(registry)
    filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    records = [
        {
            "run_id": "000", "source_run_ids": [], "semantic_point": baseline,
            "semantic_edges": [], "status": "keep", "final_best_score": 0.40,
            "evaluation_depth": "screening", "dag_revision": 1,
        },
        {
            "run_id": "001", "source_run_ids": ["000"], "semantic_point": filtered,
            "status": "discard", "final_best_score": 0.50,
            "evaluation_depth": "screening", "dag_revision": 2,
        },
        {
            "run_id": "002", "source_run_ids": [], "semantic_point": baseline,
            "semantic_edges": [], "status": "keep", "final_best_score": 0.41,
            "evaluation_depth": "screening", "dag_revision": 3,
        },
        {
            "run_id": "003", "source_run_ids": ["002"], "semantic_point": filtered,
            "status": "discard", "final_best_score": 0.52,
            "evaluation_depth": "screening", "dag_revision": 4,
        },
    ]
    records[1]["semantic_edges"] = build_semantic_edges(records[:1], records[1])
    records[3]["semantic_edges"] = build_semantic_edges(records[:3], records[3])
    return {"records": records, "dag_revision": 4}


def _carrier_demote_entry() -> dict:
    return {
        "target_id": "hyp-data-filtered",
        "evaluation_state": "observed",
        "assessment": "unpromising",
        "recommended_status": "deprioritized",
        "claim": "Two independent confounded contexts were both strictly worse.",
        "evidence_run_ids": ["000", "001", "002", "003"],
        "evidence_edge_ids": ["sedge-000-001", "sedge-002-003"],
        "comparator_coverage": {
            "direct_tuned_edges": 0,
            "direct_lightly_tuned_edges": 0,
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 2,
            "crash_edges": 0,
        },
        "confidence": "med",
        "uncertainty": "Implementation drift is an alternative explanation.",
        "reopen_when": "Any independent context where adding it improves the parent.",
    }


class CarrierDemotionGateTests(unittest.TestCase):
    def _validate(self, ledger: dict, entry: dict) -> list:
        registry = fixture_registry()
        experience = _base_experience()
        experience["hypothesis_evidence"] = [entry]
        return validate_experience(experience, registry, ledger)

    def test_deprioritize_accepted_on_two_negative_contexts(self) -> None:
        registry = fixture_registry()
        errors = self._validate(_confounded_ledger(registry), _carrier_demote_entry())
        self.assertEqual(errors, [])

    def test_prune_rejected_on_only_two_negative_contexts(self) -> None:
        registry = fixture_registry()
        entry = _carrier_demote_entry()
        entry["recommended_status"] = "pruned"
        entry["confidence"] = "high"
        errors = self._validate(_confounded_ledger(registry), entry)
        self.assertTrue(any("pruned" in error for error in errors), errors)

    def test_positive_context_blocks_demotion(self) -> None:
        registry = fixture_registry()
        ledger = _confounded_ledger(registry)
        ledger["records"][3]["final_best_score"] = 0.39  # 003 now BEATS parent 002
        errors = self._validate(ledger, _carrier_demote_entry())
        self.assertTrue(any("deprioritized" in error for error in errors), errors)

    def test_promising_still_requires_comparator_coverage(self) -> None:
        registry = fixture_registry()
        entry = _carrier_demote_entry()
        entry["assessment"] = "promising"
        entry["recommended_status"] = "active"
        errors = self._validate(_confounded_ledger(registry), entry)
        self.assertTrue(any("promising" in error for error in errors), errors)

    def test_uncited_negative_contexts_do_not_count(self) -> None:
        # The ledger holds two negative contexts, but the entry cites only
        # one edge: beliefs must be justified by their cited evidence.
        registry = fixture_registry()
        entry = _carrier_demote_entry()
        entry["evidence_run_ids"] = ["000", "001"]
        entry["evidence_edge_ids"] = ["sedge-000-001"]
        entry["comparator_coverage"] = {
            "direct_tuned_edges": 0,
            "direct_lightly_tuned_edges": 0,
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 1,
            "crash_edges": 0,
        }
        errors = self._validate(_confounded_ledger(registry), entry)
        self.assertTrue(any("deprioritized" in error for error in errors), errors)


class NumberPrecheckTests(unittest.TestCase):
    """The audit text builder and the deterministic result-number precheck."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.manifest_dir = Path(self._tmp.name)

    def _manifest(self, visits: list[tuple[str, str, str, bool]]) -> dict:
        manifest = new_manifest()
        for url, view, content, store_cap_hit in visits:
            add_visit(
                manifest,
                self.manifest_dir,
                url=url,
                backend="web",
                view=view,
                status="success",
                content=content,
                store_cap_hit=store_cap_hit,
            )
        return manifest

    @staticmethod
    def _registry() -> dict:
        return {
            "sources": [
                {"id": "src-01", "url": "https://example.test/study-a", "title": "Study A"},
                {"id": "src-02", "url": "https://example.test/study-b", "title": "Study B"},
            ]
        }

    @staticmethod
    def _item(claim: str, *source_ids: str) -> dict:
        return {
            "id": "g-01",
            "claim": claim,
            "evidence": [
                {"source_id": source_id, "role": "supports"} for source_id in source_ids
            ],
        }

    def test_audit_text_takes_present_fields_with_names(self) -> None:
        item = {
            "claim": "Filtering lifts accuracy.",
            "scope": {"metrics": ["accuracy"], "data_regimes": ["toy-data"]},
            "credibility_rationale": "One study reports it (2003.11545).",
            "title": "not part of the audit surface",
        }
        self.assertEqual(
            audit_text(item),
            "claim: Filtering lifts accuracy.\n"
            "scope: data_regimes: toy-data; metrics: accuracy\n"
            "credibility_rationale: One study reports it (2003.11545).",
        )

    def test_result_number_matches_percent_decimal_and_truncation(self) -> None:
        for claim, source, expected in (
            ("0.3843", "38.43%", True),  # % ↔ decimal variant
            ("38.43%", "0.3843", True),
            ("94%", "0.94", True),
            ("0.380", "0.38", True),  # trailing-zero equivalence
            ("0.38", "0.3843", True),  # one-direction rounding truncation
            ("38.4%", "0.3843", True),
            ("0.3843", "0.38", False),  # fabricated precision
            ("0.4", "0.3843", False),  # rounding up is not truncation
            ("5%", "0.50", False),  # magnitude differs
        ):
            with self.subTest(claim=claim, source=source):
                self.assertIs(result_number_matches(claim, source), expected)

    def test_extract_result_numbers_excludes_arxiv_and_splits_ranges(self) -> None:
        text = (
            "Reported (2003.11545) and 2003.11545v2 with 38.43% accuracy, "
            "a 94-95% range, 3 plain integers, and 0.3843."
        )
        self.assertEqual(
            extract_result_numbers(text), ["38.43%", "94%", "95%", "0.3843"]
        )
        self.assertEqual(extract_result_numbers("no result numbers, 42 plain"), [])
        self.assertEqual(
            extract_result_numbers(
                "<table><tr><td>0.839</td><td>0.816</td></tr></table> and 3.1"
            ),
            ["0.839", "0.816", "3.1"],
        )
        self.assertEqual(
            extract_result_numbers("<0.05; RUS: -0.004, p>"),
            ["0.05", "0.004"],
        )

    def test_mapping_number_presence_present_absent_none(self) -> None:
        registry = self._registry()
        url = registry["sources"][0]["url"]
        link = {"source_id": "src-01", "role": "supports"}
        item = self._item("Filtering lifts accuracy to 0.3843.", "src-01")

        manifest = self._manifest([(url, "full_text", "We report 38.43% accuracy.", False)])
        result = mapping_number_presence(item, link, registry, manifest, self.manifest_dir)
        self.assertEqual(
            result["number_presence"], {"item": "present", "this_source": "present"}
        )
        self.assertEqual(result["missing_tokens"], {"item": [], "this_source": []})
        coverage = result["coverage"]
        self.assertEqual(coverage["tier"], "full_text")
        self.assertEqual(coverage["routing"], "sufficient")
        self.assertFalse(coverage["store_cap_hit"])
        self.assertEqual(coverage["retained_chars"], len("We report 38.43% accuracy."))
        self.assertEqual(coverage["view"], "full_text")
        self.assertTrue(coverage["content_file"])

        manifest = self._manifest([(url, "full_text", "No numbers retained here.", True)])
        result = mapping_number_presence(item, link, registry, manifest, self.manifest_dir)
        self.assertEqual(
            result["number_presence"], {"item": "absent", "this_source": "absent"}
        )
        self.assertEqual(
            result["missing_tokens"],
            {"item": ["0.3843"], "this_source": ["0.3843"]},
        )
        self.assertEqual(result["coverage"]["routing"], "partial")  # store cap hit
        self.assertTrue(result["coverage"]["store_cap_hit"])

        # Citation-shaped tokens are not result numbers: nothing to check.
        rationale_only = self._item("Supported by prior work (2003.11545).", "src-01")
        result = mapping_number_presence(
            rationale_only, link, registry, manifest, self.manifest_dir
        )
        self.assertEqual(
            result["number_presence"], {"item": "none", "this_source": "none"}
        )

    def test_latex_escaped_percent_matches(self) -> None:
        # deepxiv retained content is LaTeX-ish, stored JSON-wrapped: the raw
        # file carries a backslash run before the % (single 97\%, or the
        # JSON-escaped double 97\\% actually seen on disk).
        registry = self._registry()
        url = registry["sources"][0]["url"]
        link = {"source_id": "src-01", "role": "supports"}
        item = self._item("Accuracy reaches 97%, 98.5%, or 0.985.", "src-01")
        for content in (
            r"Reports $(97\%)$ and $98.5\%$ accuracy.",
            r"Followed by $n$-grams $(97\\%)$, $97\\%$, and $98.5\\%$ accuracy.",
        ):
            with self.subTest(content=content):
                manifest = self._manifest([(url, "section", content, False)])
                result = mapping_number_presence(
                    item, link, registry, manifest, self.manifest_dir
                )
                self.assertEqual(
                    result["number_presence"],
                    {"item": "present", "this_source": "present"},
                )
        self.assertEqual(
            extract_result_numbers(r"$(97\%)$ and $98.5\\%$"),
            [r"97\%", r"98.5\\%"],
        )

    def test_all_substantive_visits_count_for_presence(self) -> None:
        # A source's retained content is every substantive visit receipt: the
        # number may live in a later same-tier visit than the first/best one.
        registry = self._registry()
        url = registry["sources"][0]["url"]
        manifest = new_manifest()
        first = "Introduction section without decimals."
        for content in (first, r"Followed by $n$-grams $(97\\%)$ accuracy."):
            add_visit(
                manifest,
                self.manifest_dir,
                url=url,
                backend="deepxiv",
                view="section",
                status="success",
                content=content,
            )
        item = self._item("Char n-grams reach 97% accuracy.", "src-01")
        result = mapping_number_presence(
            item,
            {"source_id": "src-01", "role": "supports"},
            registry,
            manifest,
            self.manifest_dir,
        )
        self.assertEqual(
            result["number_presence"], {"item": "present", "this_source": "present"}
        )
        # Coverage facts still describe the best (first highest-tier) visit.
        self.assertEqual(result["coverage"]["retained_chars"], len(first))

    def test_range_token_matches_by_endpoints(self) -> None:
        registry = self._registry()
        url = registry["sources"][0]["url"]
        item = self._item("Accuracy lands in the 94-95% range.", "src-01")
        link = {"source_id": "src-01", "role": "supports"}
        for content, expected, missing in (
            ("We reach 94-95% accuracy.", "present", []),  # verbatim range hits
            ("A 94% floor and a 95% peak.", "present", []),  # endpoints each hit
            ("The peak is 95%.", "absent", ["94%"]),  # one endpoint missing
        ):
            with self.subTest(content=content):
                manifest = self._manifest([(url, "full_text", content, False)])
                result = mapping_number_presence(
                    item, link, registry, manifest, self.manifest_dir
                )
                self.assertEqual(result["number_presence"]["this_source"], expected)
                self.assertEqual(result["missing_tokens"]["this_source"], missing)

    def test_item_level_presence_spans_sibling_sources(self) -> None:
        # The number rides on a sibling link: this source lacks it, the item
        # still has it (the judge must not read this_source absence as a lie).
        registry = self._registry()
        url_a, url_b = (source["url"] for source in registry["sources"])
        manifest = self._manifest(
            [
                (url_a, "full_text", "A qualitative discussion without decimals.", False),
                (url_b, "section", "The ablation reaches 0.3843 accuracy.", False),
            ]
        )
        item = self._item("Filtering lifts accuracy to 0.3843.", "src-01", "src-02")
        result = mapping_number_presence(
            item,
            {"source_id": "src-01", "role": "supports"},
            registry,
            manifest,
            self.manifest_dir,
        )
        self.assertEqual(
            result["number_presence"], {"item": "present", "this_source": "absent"}
        )
        self.assertEqual(result["coverage"]["routing"], "sufficient")

        item_result = item_number_presence(item, registry, manifest, self.manifest_dir)
        self.assertEqual(item_result["presence"], "present")
        self.assertEqual(item_result["tokens"], ["0.3843"])
        self.assertEqual(item_result["missing"], [])

        # A snippet-only citation is not a qualifying carrier for the gate.
        snippet_only = self._item("Filtering lifts accuracy to 0.3843.", "src-02")
        result = mapping_number_presence(
            item,
            {"source_id": "src-02", "role": "supports"},
            registry,
            new_manifest(),
            self.manifest_dir,
        )
        self.assertEqual(result["coverage"]["tier"], "none")
        self.assertEqual(result["coverage"]["routing"], "partial")
        item_result = item_number_presence(
            snippet_only, registry, new_manifest(), self.manifest_dir
        )
        self.assertEqual(item_result["presence"], "absent")
        self.assertEqual(item_result["missing"], ["0.3843"])


class NumberGateTests(unittest.TestCase):
    """The validate-time item-level number gate (generation path only)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.manifest_dir = Path(self._tmp.name)
        self.registry = fixture_registry()
        self.url = self.registry["sources"][0]["url"]

    def _manifest(self, *visits: tuple[str, str, str]) -> dict:
        manifest = new_manifest()
        for url, view, content in visits:
            add_visit(
                manifest,
                self.manifest_dir,
                url=url,
                backend="web",
                view=view,
                status="success",
                content=content,
            )
        return manifest

    def _errors(self, manifest: dict, **kwargs) -> list[str]:
        return validate_registry(
            self.registry,
            retrieval_manifest=manifest,
            manifest_dir=self.manifest_dir,
            **kwargs,
        )

    def test_gate_passes_when_a_cited_source_retains_the_number(self) -> None:
        self.registry["guidance"][0]["claim"] = (
            "Filtering lifts accuracy to 0.3843 in this exact regime."
        )
        manifest = self._manifest(
            (self.url, "full_text", "We report 38.43% accuracy on the split.")
        )
        self.assertEqual(self._errors(manifest, number_gate=True), [])

    def test_gate_flags_a_number_no_cited_source_retains(self) -> None:
        self.registry["guidance"][0]["credibility_rationale"] = (
            "One direct primary study reports 0.3843."
        )
        manifest = self._manifest(
            (self.url, "full_text", "A qualitative discussion without decimals.")
        )
        # The flag defaults off: pre-seeded backgrounds stay ungated.
        self.assertEqual(self._errors(manifest), [])
        errors = self._errors(manifest, number_gate=True)
        self.assertEqual(len(errors), 1)
        (error,) = errors
        self.assertIn("guidance g-01", error)
        self.assertIn("0.3843", error)
        self.assertIn("src-01", error)
        self.assertIn("visit a cited source containing the number", error)
        self.assertIn("cite a different source", error)
        self.assertIn("downgrade the claim to a qualitative statement", error)

    def test_gate_passes_when_no_item_has_result_numbers(self) -> None:
        manifest = self._manifest(
            (self.url, "full_text", "A qualitative discussion without decimals.")
        )
        self.assertEqual(self._errors(manifest, number_gate=True), [])

    def test_gate_passes_when_a_sibling_source_carries_the_number(self) -> None:
        sibling = {
            **self.registry["sources"][0],
            "id": "src-02",
            "url": "https://example.test/toy-mechanism-followup",
        }
        self.registry["sources"].append(sibling)
        guidance = self.registry["guidance"][0]
        guidance["claim"] = "Filtering lifts accuracy to 0.3843 in this exact regime."
        guidance["evidence"].append({"source_id": "src-02", "role": "supports"})
        manifest = self._manifest(
            (self.url, "full_text", "A qualitative discussion without decimals."),
            (sibling["url"], "preview", "The ablation reaches 0.3843 accuracy."),
        )
        self.assertEqual(self._errors(manifest, number_gate=True), [])

    def test_gate_passes_when_the_number_sits_inside_an_html_table_cell(self) -> None:
        self.registry["guidance"][0]["claim"] = (
            "GloVe reaches 0.839 accuracy on this split."
        )
        manifest = self._manifest(
            (self.url, "full_text", "<table><tr><td>0.839</td><td>0.816</td></tr></table>")
        )
        self.assertEqual(self._errors(manifest, number_gate=True), [])

    def test_gate_ignores_citation_shaped_tokens(self) -> None:
        self.registry["guidance"][0]["credibility_rationale"] = (
            "One direct primary study (2003.11545); caution only."
        )
        manifest = self._manifest(
            (self.url, "full_text", "A qualitative discussion without decimals.")
        )
        self.assertEqual(self._errors(manifest, number_gate=True), [])
