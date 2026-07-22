from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from semantic_search import build_proposal_set, cmd_select  # noqa: E402
from validate_background import fixture_registry  # noqa: E402


class SemanticPolicyDefaultTests(unittest.TestCase):
    def test_default_policy_is_gain_uncertainty_in_template_and_cli(self) -> None:
        template = json.loads((ROOT / "tasks" / "framework_cfg.example.json").read_text())
        self.assertEqual(template["semantic_search"]["policy"], "gain_uncertainty")

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
                    "cost": 0.5,
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
            self.assertEqual(receipt["policy"]["name"], "gain_uncertainty")


if __name__ == "__main__":
    unittest.main()
