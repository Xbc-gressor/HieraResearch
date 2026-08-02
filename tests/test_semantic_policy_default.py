from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from background_contract import ContractError  # noqa: E402
from semantic_evidence import (  # noqa: E402
    acquisition_conditioning,
    validate_conditioned_adjustment,
)
from semantic_search import (  # noqa: E402
    build_baseline_proposal_set,
    build_gain_context,
    build_proposal_set,
    cmd_select,
    select_proposal,
)
from semantic_space import complete_point, selected_assignments  # noqa: E402
from tests.fixtures import belief_ledger, fixture_registry  # noqa: E402


class SemanticPolicyDefaultTests(unittest.TestCase):
    @staticmethod
    def _schema3_predictions(
        proposal_set: dict,
        receipt: dict,
        *,
        overrides: dict[str, dict] | None = None,
    ) -> dict:
        overrides = overrides or {}
        rows = []
        for proposal in proposal_set["proposals"]:
            values = {
                "prior_gain": 0.5,
                "gain_adjustment": 0.0,
                "prior_uncertainty": 0.3,
                "uncertainty_adjustment": 0.0,
                "run_ids": [],
                "edge_ids": [],
                "target_ids": [],
            }
            values.update(overrides.get(proposal["point_id"], {}))
            rows.append(
                {
                    "point_id": proposal["point_id"],
                    "prior_gain": values["prior_gain"],
                    "experience_gain_adjustment": values["gain_adjustment"],
                    "predicted_gain": (
                        values["prior_gain"] + values["gain_adjustment"]
                    ),
                    "prior_uncertainty": values["prior_uncertainty"],
                    "experience_uncertainty_adjustment": (
                        values["uncertainty_adjustment"]
                    ),
                    "uncertainty": (
                        values["prior_uncertainty"]
                        + values["uncertainty_adjustment"]
                    ),
                    "experience_run_ids": values["run_ids"],
                    "experience_edge_ids": values["edge_ids"],
                    "experience_target_ids": values["target_ids"],
                    "experience_rationale": (
                        "Only helper-rendered structured conditioning is used."
                    ),
                    "evidence": ["bounded gain-context regression fixture"],
                }
            )
        return {
            "schema_version": 3,
            "proposal_set_revision": proposal_set["proposal_set_revision"],
            "experience": receipt,
            "predictions": rows,
        }

    def test_empty_evidence_snapshot_is_pinned_but_does_not_force_adjustment(self) -> None:
        proposal_set = build_proposal_set(
            fixture_registry(), {"records": []}, op="fresh", parents=[], max_points=3
        )
        experience = {
            "schema_version": 3,
            "updated_at_run": "000",
            "generation": 0,
            "summary": "No bounded belief carries evidence yet.",
            "promising_regions": [],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [],
        }
        context = build_gain_context(
            proposal_set,
            {"records": [], "experience": experience},
        )
        predictions = self._schema3_predictions(
            proposal_set,
            context["experience_receipt"],
        )
        _, receipt = select_proposal(
            proposal_set,
            policy="gain_uncertainty_nocost",
            predictions=predictions,
            experience=experience,
        )
        self.assertEqual(receipt["experience"]["generation"], 0)
        self.assertEqual(receipt["experience"]["evidence_run_ids"], [])
        self.assertEqual(receipt["experience"]["conditioning"], [])
        self.assertEqual(receipt["components"]["experience_gain_adjustment"], 0.0)

    def test_provided_baseline_is_the_only_first_point(self) -> None:
        registry = fixture_registry()
        proposal_set = build_baseline_proposal_set(registry, {"records": []})

        self.assertEqual(len(proposal_set["proposals"]), 1)
        self.assertEqual(
            proposal_set["proposals"][0]["point"],
            complete_point(registry),
        )
        point, receipt = select_proposal(
            proposal_set,
            policy="coverage",
            selection_index=1,
        )
        self.assertEqual(point, complete_point(registry))
        self.assertEqual(receipt["budget"]["selection_index"], 1)
        self.assertEqual(receipt["policy"]["name"], "coverage")

        with self.assertRaisesRegex(ContractError, "before any other record"):
            build_baseline_proposal_set(
                registry,
                {"records": [{"run_id": "000"}]},
            )

    def test_gain_context_excludes_ungated_prose_and_signed_observations(self) -> None:
        proposal_set = build_proposal_set(
            fixture_registry(), {"records": []}, op="fresh", parents=[], max_points=3
        )
        experience = {
            "schema_version": 3,
            "updated_at_run": "000",
            "generation": 0,
            "summary": "SHOCK_CLAIM reverses the prior generation.",
            "promising_regions": [{
                "claim": "SHOCK_CLAIM is the strongest signal.",
                "evidence": ["000"],
            }],
            "lessons": [{"claim": "SHOCK_CLAIM", "evidence": ["000"]}],
            "bottlenecks": [{"claim": "SHOCK_CLAIM", "evidence": ["000"]}],
            "dimension_evidence": [],
            "hypothesis_evidence": [],
        }
        context = build_gain_context(
            proposal_set,
            {
                "records": [{
                    "run_id": "000",
                    "status": "keep",
                    "final_best_score": 0.01,
                }],
                "experience": experience,
            },
        )
        self.assertEqual(context["schema_version"], 3)
        self.assertEqual(
            set(context),
            {
                "schema_version",
                "proposal_set_revision",
                "experience_receipt",
                "conditioning_by_point",
                "experience_evidence_run_ids",
                "experience_evidence_edge_ids",
            },
        )
        rendered = json.dumps(context)
        self.assertNotIn("SHOCK_CLAIM", rendered)
        self.assertNotIn("final_best_score", rendered)
        self.assertNotIn('"delta"', rendered)
        self.assertEqual(context["experience_evidence_run_ids"], [])

    def test_weak_evidence_rejects_gain_but_allows_uncertainty_and_abstention(
        self,
    ) -> None:
        registry = fixture_registry()
        proposal_set = build_proposal_set(
            registry, {"records": []}, op="fresh", parents=[], max_points=64
        )
        filtered = next(
            proposal
            for proposal in proposal_set["proposals"]
            if selected_assignments(proposal["point"]).get("dim-data-curation")
            == "hyp-data-filtered"
        )
        experience = {
            "schema_version": 3,
            "updated_at_run": "001",
            "generation": 0,
            "summary": "One confounded rewrite is not a gain signal.",
            "promising_regions": [],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [{
                "target_id": "hyp-data-filtered",
                "evaluation_state": "observed",
                "assessment": "mixed",
                "recommended_status": "active",
                "claim": "The observed delta is implementation-confounded.",
                "evidence_run_ids": ["000", "001"],
                "evidence_edge_ids": ["sedge-000-001"],
                "comparator_coverage": {
                    "direct_tuned_edges": 0,
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 1,
                    "crash_edges": 0,
                },
                "confidence": "low",
                "uncertainty": "No direct matched comparator exists.",
            }],
        }
        context = build_gain_context(
            proposal_set, {"records": [], "experience": experience}
        )
        conditioning = context["conditioning_by_point"][filtered["point_id"]]
        self.assertEqual(conditioning[0]["acquisition_role"], "uncertainty_only")
        cited = {
            "target_ids": ["hyp-data-filtered"],
            "run_ids": ["000", "001"],
            "edge_ids": ["sedge-000-001"],
        }

        signed_gain = self._schema3_predictions(
            proposal_set,
            context["experience_receipt"],
            overrides={
                filtered["point_id"]: {
                    **cited,
                    "gain_adjustment": 0.03,
                }
            },
        )
        with self.assertRaisesRegex(
            ContractError, "nonzero experience gain requires cited comparator"
        ):
            select_proposal(
                proposal_set,
                policy="gain_uncertainty_nocost",
                predictions=signed_gain,
                experience=experience,
            )

        partial_citations = self._schema3_predictions(
            proposal_set,
            context["experience_receipt"],
            overrides={
                filtered["point_id"]: {
                    **cited,
                    "run_ids": ["001"],
                    "uncertainty_adjustment": 0.05,
                }
            },
        )
        with self.assertRaisesRegex(ContractError, "helper-derived union"):
            select_proposal(
                proposal_set,
                policy="gain_uncertainty_nocost",
                predictions=partial_citations,
                experience=experience,
            )

        uncertainty_only = self._schema3_predictions(
            proposal_set,
            context["experience_receipt"],
            overrides={
                filtered["point_id"]: {
                    **cited,
                    "prior_gain": 0.9,
                    "uncertainty_adjustment": 0.05,
                }
            },
        )
        _, uncertainty_receipt = select_proposal(
            proposal_set,
            policy="gain_uncertainty_nocost",
            predictions=uncertainty_only,
            experience=experience,
        )
        self.assertEqual(uncertainty_receipt["selected_point_id"], filtered["point_id"])
        self.assertEqual(
            uncertainty_receipt["components"]["experience_gain_adjustment"], 0.0
        )
        self.assertEqual(
            uncertainty_receipt["components"]["experience_uncertainty_adjustment"],
            0.05,
        )
        self.assertEqual(
            uncertainty_receipt["experience"]["conditioning"][0]["acquisition_role"],
            "uncertainty_only",
        )

        abstention = self._schema3_predictions(
            proposal_set,
            context["experience_receipt"],
            overrides={
                filtered["point_id"]: {
                    **cited,
                    "prior_gain": 0.9,
                }
            },
        )
        _, abstention_receipt = select_proposal(
            proposal_set,
            policy="gain_uncertainty_nocost",
            predictions=abstention,
            experience=experience,
        )
        self.assertEqual(abstention_receipt["selected_point_id"], filtered["point_id"])
        self.assertEqual(
            abstention_receipt["components"]["experience_gain_adjustment"], 0.0
        )
        self.assertEqual(
            abstention_receipt["components"]["experience_uncertainty_adjustment"],
            0.0,
        )
        self.assertEqual(
            abstention_receipt["experience"]["evidence_edge_ids"],
            ["sedge-000-001"],
        )

    def test_conditioning_is_proposal_relevant_and_same_point_is_empty(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        proposal_set = build_proposal_set(
            registry,
            {"records": [{"run_id": "000", "semantic_point": baseline}]},
            op="improve",
            parents=["000"],
            max_points=64,
        )
        experience = {
            "schema_version": 3,
            "updated_at_run": "003",
            "generation": 0,
            "summary": "Structured target evidence only.",
            "promising_regions": [],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [{
                "target_id": "dim-data-curation",
                "evaluation_state": "comparator_covered",
                "evidence_run_ids": ["000", "001", "002", "003"],
                "evidence_edge_ids": ["sedge-000-001", "sedge-002-003"],
                "comparator_coverage": {
                    "direct_tuned_edges": 2,
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
            }],
            "hypothesis_evidence": [{
                "target_id": "hyp-data-filtered",
                "evaluation_state": "comparator_covered",
                "evidence_run_ids": ["000", "001", "002", "003"],
                "evidence_edge_ids": ["sedge-000-001", "sedge-002-003"],
                "comparator_coverage": {
                    "direct_tuned_edges": 2,
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
            }],
        }
        context = build_gain_context(
            proposal_set, {"records": [], "experience": experience}
        )
        same_point = next(
            proposal
            for proposal in proposal_set["proposals"]
            if proposal["parent_diffs"][0]["changes"] == []
        )
        filtered = next(
            proposal
            for proposal in proposal_set["proposals"]
            if selected_assignments(proposal["point"]).get("dim-data-curation")
            == "hyp-data-filtered"
        )
        unrelated = next(
            proposal
            for proposal in proposal_set["proposals"]
            if proposal["point_id"] not in {
                same_point["point_id"],
                filtered["point_id"],
            }
            and all(
                change.get("dimension_id") != "dim-data-curation"
                for change in proposal["parent_diffs"][0]["changes"]
            )
        )
        self.assertEqual(
            context["conditioning_by_point"][same_point["point_id"]], []
        )
        self.assertEqual(
            context["conditioning_by_point"][unrelated["point_id"]], []
        )
        self.assertEqual(
            {
                item["target_id"]
                for item in context["conditioning_by_point"][filtered["point_id"]]
            },
            {"dim-data-curation", "hyp-data-filtered"},
        )

    def test_signed_gain_requires_helper_marked_directional_comparator(self) -> None:
        registry = fixture_registry()
        point = complete_point(
            registry, {"dim-data-curation": "hyp-data-filtered"}
        )
        experience = {
            "hypothesis_evidence": [{
                "target_id": "hyp-data-filtered",
                "evaluation_state": "comparator_covered",
                "evidence_run_ids": ["000", "001", "002", "003"],
                "evidence_edge_ids": ["sedge-000-001", "sedge-002-003"],
                "comparator_coverage": {
                    "direct_tuned_edges": 2,
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
            }]
        }
        without_direction = acquisition_conditioning(
            experience,
            point,
            target_ids={"hyp-data-filtered"},
        )
        self.assertEqual(
            without_direction[0]["acquisition_role"], "uncertainty_only"
        )
        self.assertTrue(
            any(
                "requires cited comparator-covered" in error
                for error in validate_conditioned_adjustment(
                    without_direction,
                    point,
                    evidence_run_ids=["000", "001", "002", "003"],
                    evidence_edge_ids=["sedge-000-001", "sedge-002-003"],
                    gain_adjustment=-0.05,
                    uncertainty_adjustment=0.0,
                )
            )
        )
        directed = acquisition_conditioning(
            experience,
            point,
            target_ids={"hyp-data-filtered"},
            gain_directions={"hyp-data-filtered": "negative"},
        )
        self.assertEqual(directed[0]["acquisition_role"], "comparator_gain")
        self.assertEqual(directed[0]["gain_direction"], "negative")
        self.assertEqual(
            validate_conditioned_adjustment(
                directed,
                point,
                evidence_run_ids=["000", "001", "002", "003"],
                evidence_edge_ids=["sedge-000-001", "sedge-002-003"],
                gain_adjustment=-0.05,
                uncertainty_adjustment=0.0,
            ),
            [],
        )
        self.assertTrue(
            any(
                "sign conflicts" in error
                for error in validate_conditioned_adjustment(
                    directed,
                    point,
                    evidence_run_ids=["000", "001", "002", "003"],
                    evidence_edge_ids=["sedge-000-001", "sedge-002-003"],
                    gain_adjustment=0.05,
                    uncertainty_adjustment=0.0,
                )
            )
        )

    def test_gain_context_derives_direction_and_inverts_it_for_removal(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        experience = {
            "schema_version": 3,
            "updated_at_run": "003",
            "generation": 0,
            "dag_revision": ledger["dag_revision"],
            "summary": "",
            "promising_regions": [],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [{
                "target_id": "hyp-data-filtered",
                "evaluation_state": "comparator_covered",
                "evidence_run_ids": ["000", "001", "002", "003"],
                "evidence_edge_ids": ["sedge-000-001", "sedge-002-003"],
                "comparator_coverage": {
                    "direct_tuned_edges": 2,
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
            }],
        }
        ledger["experience"] = experience
        fresh = build_proposal_set(
            registry, ledger, op="fresh", parents=[], max_points=64
        )
        filtered = next(
            proposal
            for proposal in fresh["proposals"]
            if selected_assignments(proposal["point"]).get("dim-data-curation")
            == "hyp-data-filtered"
        )
        context = build_gain_context(fresh, ledger)
        (conditioning,) = context["conditioning_by_point"][
            filtered["point_id"]
        ]
        self.assertEqual(conditioning["proposal_relation"], "selected_fresh")
        self.assertEqual(conditioning["acquisition_role"], "comparator_gain")
        self.assertEqual(conditioning["gain_direction"], "negative")

        predictions = self._schema3_predictions(
            fresh,
            context["experience_receipt"],
            overrides={
                filtered["point_id"]: {
                    "prior_gain": 0.9,
                    "gain_adjustment": -0.05,
                    "run_ids": ["000", "001", "002", "003"],
                    "edge_ids": ["sedge-000-001", "sedge-002-003"],
                    "target_ids": ["hyp-data-filtered"],
                }
            },
        )
        _, receipt = select_proposal(
            fresh,
            policy="gain_uncertainty_nocost",
            predictions=predictions,
            experience=experience,
            ledger=ledger,
        )
        self.assertEqual(
            receipt["components"]["experience_gain_adjustment"], -0.05
        )

        improve = build_proposal_set(
            registry, ledger, op="improve", parents=["003"], max_points=64
        )
        baseline = next(
            proposal
            for proposal in improve["proposals"]
            if selected_assignments(proposal["point"]).get("dim-data-curation")
            == "hyp-data-raw"
        )
        removal_context = build_gain_context(improve, ledger)
        removed = next(
            item
            for item in removal_context["conditioning_by_point"][
                baseline["point_id"]
            ]
            if item["target_id"] == "hyp-data-filtered"
        )
        self.assertEqual(removed["proposal_relation"], "removed")
        self.assertEqual(removed["gain_direction"], "positive")

    def test_default_policy_is_pure_coverage_in_template_and_cli(self) -> None:
        template = json.loads((ROOT / "tasks" / "framework_cfg.example.json").read_text())
        self.assertEqual(template["semantic_search"]["policy"], "coverage")
        self.assertEqual(
            template["semantic_search"]["deprioritized_budget_interval"], 5
        )
        self.assertEqual(
            template["semantic_search"]["llm_intelligence_score"], 100
        )

        proposal_set = build_proposal_set(
            fixture_registry(), {"records": []}, op="fresh", parents=[], max_points=3
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proposals_path = tmp_path / "proposals.json"
            point_path = tmp_path / "point.json"
            receipt_path = tmp_path / "policy.json"
            proposals_path.write_text(json.dumps(proposal_set))

            result = cmd_select(
                SimpleNamespace(
                    proposals=proposals_path,
                    predictions=None,
                    ledger=None,
                    policy=None,
                    cfg=None,
                    point_output=point_path,
                    receipt_output=receipt_path,
                )
            )

            self.assertEqual(result, 0)
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["policy"]["name"], "coverage")
            self.assertEqual(receipt["schema_version"], 6)
            self.assertIsNone(receipt["components"]["llm_judgment_weight"])
            self.assertIsNone(receipt["components"]["predicted_gain"])
            self.assertIsNone(receipt["components"]["uncertainty"])
            self.assertEqual(receipt["evidence"], [])
            self.assertEqual(receipt["experience"]["conditioning"], [])
            self.assertEqual(receipt["budget"]["selected_lane"], "active")

    def test_llm_intelligence_score_weights_only_model_judgments(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        proposal_set = build_proposal_set(
            registry,
            {"records": [{"run_id": "000", "semantic_point": baseline}]},
            op="improve",
            parents=["000"],
            max_points=64,
        )
        highest_coverage = max(
            proposal_set["proposals"],
            key=lambda proposal: (proposal["coverage"], proposal["point_id"]),
        )
        model_favorite = min(
            proposal_set["proposals"],
            key=lambda proposal: (proposal["coverage"], proposal["point_id"]),
        )
        predictions = {
            "schema_version": 1,
            "proposal_set_revision": proposal_set["proposal_set_revision"],
            "predictions": [
                {
                    "point_id": proposal["point_id"],
                    "predicted_gain": (
                        1.0 if proposal["point_id"] == model_favorite["point_id"] else 0.0
                    ),
                    "uncertainty": (
                        0.2 if proposal["point_id"] == model_favorite["point_id"] else 0.0
                    ),
                    "evidence": ["fixed reliability-prior regression fixture"],
                }
                for proposal in proposal_set["proposals"]
            ],
        }

        _, default_receipt = select_proposal(
            proposal_set,
            policy="gain_uncertainty_nocost",
            predictions=predictions,
        )
        self.assertEqual(default_receipt["selected_point_id"], model_favorite["point_id"])
        self.assertEqual(default_receipt["components"]["llm_judgment_weight"], 1.0)
        self.assertAlmostEqual(
            default_receipt["acquisition_score"],
            1.0 + 0.5 * 0.2 + 0.1 * model_favorite["coverage"],
        )

        _, weighted_receipt = select_proposal(
            proposal_set,
            policy="gain_uncertainty_nocost",
            predictions=predictions,
            config={"llm_intelligence_score": 44},
        )
        self.assertEqual(weighted_receipt["selected_point_id"], model_favorite["point_id"])
        self.assertEqual(
            weighted_receipt["policy"]["config"]["llm_intelligence_score"], 44.0
        )
        self.assertEqual(weighted_receipt["components"]["llm_judgment_weight"], 0.44)
        self.assertAlmostEqual(
            weighted_receipt["acquisition_score"],
            0.44 * (1.0 + 0.5 * 0.2) + 0.1 * model_favorite["coverage"],
        )
        self.assertEqual(weighted_receipt["components"]["predicted_gain"], 1.0)
        self.assertEqual(weighted_receipt["components"]["uncertainty"], 0.2)

        _, coverage_fallback = select_proposal(
            proposal_set,
            policy="gain_uncertainty_nocost",
            predictions=predictions,
            config={"llm_intelligence_score": 0},
        )
        self.assertEqual(
            coverage_fallback["selected_point_id"], highest_coverage["point_id"]
        )
        self.assertEqual(
            coverage_fallback["components"]["llm_judgment_weight"], 0.0
        )

    def test_llm_intelligence_score_is_finite_and_bounded(self) -> None:
        proposal_set = build_proposal_set(
            fixture_registry(), {"records": []}, op="fresh", parents=[], max_points=3
        )
        predictions = {
            "schema_version": 1,
            "proposal_set_revision": proposal_set["proposal_set_revision"],
            "predictions": [
                {
                    "point_id": proposal["point_id"],
                    "predicted_gain": 0.5,
                    "uncertainty": 0.5,
                    "evidence": ["bounded-config regression fixture"],
                }
                for proposal in proposal_set["proposals"]
            ],
        }
        for invalid in (-0.1, 100.1, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ContractError, r"finite number in \[0, 100\]"):
                    select_proposal(
                        proposal_set,
                        policy="gain_uncertainty_nocost",
                        predictions=predictions,
                        config={"llm_intelligence_score": invalid},
                    )

    def test_select_rejects_a_midrun_llm_intelligence_score_change(self) -> None:
        proposal_set = build_proposal_set(
            fixture_registry(), {"records": []}, op="fresh", parents=[], max_points=3
        )
        predictions = {
            "schema_version": 1,
            "proposal_set_revision": proposal_set["proposal_set_revision"],
            "predictions": [
                {
                    "point_id": proposal["point_id"],
                    "predicted_gain": 0.5,
                    "uncertainty": 0.5,
                    "evidence": ["frozen-config regression fixture"],
                }
                for proposal in proposal_set["proposals"]
            ],
        }
        _, prior_receipt = select_proposal(
            proposal_set,
            policy="gain_uncertainty_nocost",
            predictions=predictions,
            config={"llm_intelligence_score": 44},
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proposals_path = tmp_path / "proposals.json"
            predictions_path = tmp_path / "predictions.json"
            ledger_path = tmp_path / "ledger.json"
            proposals_path.write_text(json.dumps(proposal_set))
            predictions_path.write_text(json.dumps(predictions))
            ledger_path.write_text(
                json.dumps({"records": [{"policy_receipt": prior_receipt}]})
            )
            (tmp_path / "framework_cfg.json").write_text(
                json.dumps(
                    {"semantic_search": {"llm_intelligence_score": 61}}
                )
            )
            with self.assertRaisesRegex(ContractError, "is frozen at 44"):
                cmd_select(
                    SimpleNamespace(
                        proposals=proposals_path,
                        predictions=predictions_path,
                        ledger=ledger_path,
                        policy=None,
                        cfg=None,
                        point_output=tmp_path / "point.json",
                        receipt_output=tmp_path / "policy.json",
                    )
                )


if __name__ == "__main__":
    unittest.main()
