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
from semantic_search import (  # noqa: E402
    build_gain_context,
    build_proposal_set,
    cmd_select,
    select_proposal,
)
from validate_background import fixture_registry  # noqa: E402


class SemanticPolicyDefaultTests(unittest.TestCase):
    @staticmethod
    def _schema2_predictions(
        proposal_set: dict,
        receipt: dict,
        *,
        gain_adjustment: float,
        uncertainty_adjustment: float,
        run_ids: list[str],
    ) -> dict:
        return {
            "schema_version": 2,
            "proposal_set_revision": proposal_set["proposal_set_revision"],
            "experience": receipt,
            "predictions": [
                {
                    "point_id": proposal["point_id"],
                    "prior_gain": 0.5,
                    "experience_gain_adjustment": gain_adjustment,
                    "predicted_gain": 0.5 + gain_adjustment,
                    "prior_uncertainty": 0.3,
                    "experience_uncertainty_adjustment": uncertainty_adjustment,
                    "uncertainty": 0.3 + uncertainty_adjustment,
                    "experience_run_ids": run_ids,
                    "experience_edge_ids": [],
                    "experience_rationale": (
                        "The bounded experience has the stated numerical effect."
                    ),
                    "evidence": ["bounded gain-context regression fixture"],
                }
                for proposal in proposal_set["proposals"]
            ],
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
        predictions = self._schema2_predictions(
            proposal_set,
            context["experience_receipt"],
            gain_adjustment=0.0,
            uncertainty_adjustment=0.0,
            run_ids=[],
        )
        _, receipt = select_proposal(
            proposal_set,
            policy="gain_uncertainty_nocost",
            predictions=predictions,
            experience=experience,
        )
        self.assertEqual(receipt["experience"]["generation"], 0)
        self.assertEqual(receipt["experience"]["evidence_run_ids"], [])
        self.assertEqual(receipt["components"]["experience_gain_adjustment"], 0.0)

    def test_conditioning_adjustment_has_nontrivial_floor(self) -> None:
        proposal_set = build_proposal_set(
            fixture_registry(), {"records": []}, op="fresh", parents=[], max_points=3
        )
        experience = {
            "schema_version": 3,
            "updated_at_run": "000",
            "generation": 0,
            "summary": "One bounded belief cites a terminal run.",
            "promising_regions": [{"evidence": ["000"]}],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [],
        }
        context = build_gain_context(
            proposal_set,
            {"records": [], "experience": experience},
        )
        predictions = self._schema2_predictions(
            proposal_set,
            context["experience_receipt"],
            gain_adjustment=0.001,
            uncertainty_adjustment=0.0,
            run_ids=["000"],
        )
        with self.assertRaisesRegex(ContractError, "at least 0.01"):
            select_proposal(
                proposal_set,
                policy="gain_uncertainty_nocost",
                predictions=predictions,
                experience=experience,
            )

    def test_gain_context_enriches_only_cited_terminal_records(self) -> None:
        proposal_set = build_proposal_set(
            fixture_registry(), {"records": []}, op="fresh", parents=[], max_points=3
        )
        experience = {
            "schema_version": 3,
            "updated_at_run": "000",
            "generation": 0,
            "summary": "The bounded belief cites one same-point retry.",
            "promising_regions": [{"evidence": ["000"]}],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [{
                "evidence_run_ids": [],
                "evidence_edge_ids": ["sedge-999-000"],
            }],
        }
        context = build_gain_context(
            proposal_set,
            {
                "experience": experience,
                "records": [
                    {
                        "run_id": "999",
                        "status": "keep",
                        "source_run_ids": [],
                        "semantic_point": proposal_set["proposals"][0]["point"],
                        "semantic_edges": [],
                        "tune": False,
                        "best_warm_score": 0.6,
                        "final_best_score": 0.6,
                    },
                    {
                        "run_id": "000",
                        "status": "keep",
                        "source_run_ids": ["999"],
                        "semantic_point": proposal_set["proposals"][0]["point"],
                        "semantic_edges": [{
                            "edge_id": "sedge-999-000",
                            "parent_run_id": "999",
                            "child_run_id": "000",
                            "change_class": "same_point",
                        }],
                        "tune": True,
                        "best_warm_score": 0.5,
                        "final_best_score": 0.4,
                        "policy_receipt": {
                            "components": {
                                "predicted_gain": 0.3,
                                "uncertainty": 0.2,
                            }
                        },
                    },
                    {"run_id": "001", "status": "keep"},
                ],
            },
        )
        self.assertEqual(context["schema_version"], 2)
        self.assertEqual(len(context["cited_records"]), 2)
        by_id = {item["run_id"]: item for item in context["cited_records"]}
        cited = by_id["000"]
        self.assertTrue(cited["tuned"])
        self.assertEqual(cited["warm_to_final_delta"], -0.1)
        self.assertEqual(cited["source_run_ids"], ["999"])
        self.assertEqual(cited["same_point_parent_run_ids"], ["999"])
        self.assertEqual(
            cited["citation_roles"],
            ["edge_child:sedge-999-000", "run"],
        )
        self.assertEqual(
            by_id["999"]["citation_roles"],
            ["edge_parent:sedge-999-000"],
        )
        self.assertEqual(context["cited_edges"][0]["delta"], -0.2)

    def test_default_policy_is_gain_uncertainty_nocost_in_template_and_cli(self) -> None:
        template = json.loads((ROOT / "tasks" / "framework_cfg.example.json").read_text())
        self.assertEqual(
            template["semantic_search"]["policy"], "gain_uncertainty_nocost"
        )
        self.assertEqual(
            template["semantic_search"]["deprioritized_budget_interval"], 5
        )

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
                    "evidence": ["regression fixture"],
                }
                for proposal in proposal_set["proposals"]
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proposals_path = tmp_path / "proposals.json"
            predictions_path = tmp_path / "predictions.json"
            point_path = tmp_path / "point.json"
            receipt_path = tmp_path / "policy.json"
            proposals_path.write_text(json.dumps(proposal_set))
            predictions_path.write_text(json.dumps(predictions))

            result = cmd_select(
                SimpleNamespace(
                    proposals=proposals_path,
                    predictions=predictions_path,
                    ledger=None,
                    policy=None,
                    cfg=None,
                    point_output=point_path,
                    receipt_output=receipt_path,
                )
            )

            self.assertEqual(result, 0)
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["policy"]["name"], "gain_uncertainty_nocost")
            self.assertEqual(receipt["schema_version"], 4)
            self.assertEqual(receipt["budget"]["selected_lane"], "active")


if __name__ == "__main__":
    unittest.main()
