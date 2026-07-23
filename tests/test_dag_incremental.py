from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from background_contract import validate_registry  # noqa: E402
from got_graph import CRASH, Graph, render_incremental  # noqa: E402
from ledger import _set_experience, _touch_dag_record  # noqa: E402
from semantic_evidence import build_semantic_edges  # noqa: E402
from semantic_space import complete_point, derive_semantic_lineage  # noqa: E402
from validate_background import (  # noqa: E402
    background_text,
    fixture_registry,
    policy_receipt,
)


def _records() -> list[dict]:
    rows = [
        ("000", [], "fresh", "keep", 0.5),
        ("001", ["000"], "improve", "keep", 0.4),
        ("002", [], "fresh", "discard", 0.8),
        ("003", ["001", "002"], "crossover", "keep", 0.3),
        ("004", ["003"], "improve", "crash", None),
        ("005", ["002"], "improve", "discard", 0.9),
    ]
    return [
        {
            "run_id": run_id,
            "source_run_ids": sources,
            "op": op,
            "status": status,
            "final_best_score": score,
            "idea": run_id,
            "change": op,
            "dag_revision": revision,
        }
        for revision, (run_id, sources, op, status, score) in enumerate(rows, 1)
    ]


class IncrementalDagTests(unittest.TestCase):
    def test_delta_replays_edges_affected_by_an_old_node_update(self) -> None:
        records = _records()
        ledger = {"records": records, "dag_revision": 6, "experience": {"dag_revision": 4}}

        view = render_incremental(ledger, top=2, bottom=2)
        self.assertEqual(view["cursor"], {"from_revision": 4, "to_revision": 6})
        self.assertEqual([node["id"] for node in view["delta_nodes"]], ["005", "004"])
        self.assertEqual([node["id"] for node in view["top_nodes"]], ["003", "001"])
        self.assertEqual([node["id"] for node in view["bottom_nodes"]], ["002", "000"])

        records[1]["final_best_score"] = 0.2
        records[1]["dag_revision"] = 7
        ledger["dag_revision"] = 7
        ledger["experience"]["dag_revision"] = 6
        changed = render_incremental(ledger, top=1, bottom=1)
        self.assertEqual([node["id"] for node in changed["delta_nodes"]], ["001"])
        self.assertEqual(
            {(edge["parent"], edge["child"]) for edge in changed["delta_edges"]},
            {("000", "001"), ("001", "003")},
        )

    def test_revision_cursor_and_compact_lineage(self) -> None:
        records = _records()
        data = {"records": records, "dag_revision": 6}
        self.assertEqual(_touch_dag_record(data, records[0]), 7)

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            ledger_path.write_text(json.dumps(data))
            _set_experience(ledger_path, {"summary": "current"})
            stored = json.loads(ledger_path.read_text())
        self.assertEqual(stored["experience"]["dag_revision"], 7)

        registry = fixture_registry()
        baseline = complete_point(registry)
        stacked = complete_point(
            registry,
            {
                "dim-model-architecture": "hyp-model-multibranch",
                "dim-ensemble": "hyp-ensemble-stacking",
            },
        )
        for index, record in enumerate(stored["records"]):
            record["semantic_point"] = stacked if record["run_id"] in {"002", "003", "004", "005"} else baseline
            record["semantic_edges"] = build_semantic_edges(stored["records"][:index], record)
        compact = derive_semantic_lineage(registry, stored, limit=2)
        self.assertEqual([item["run_id"] for item in compact["runs"]], ["004", "005"])
        self.assertEqual(compact["coverage"]["n_valid_records"], 6)
        self.assertLessEqual(len(compact["hypothesis_runs"]["hyp-model-multibranch"]), 2)

        lineage = derive_semantic_lineage(registry, stored)
        run_three = next(item for item in lineage["runs"] if item["run_id"] == "003")
        diffs = {item["parent_run_id"]: item["changes"] for item in run_three["semantic_edges"]}
        self.assertEqual(diffs["002"], [])
        self.assertEqual(
            diffs["001"],
            [
                {
                    "dimension_id": "dim-model-architecture",
                    "operation": "hypothesis_changed",
                    "from_hypothesis_id": "hyp-model-linear",
                    "to_hypothesis_id": "hyp-model-multibranch",
                },
                {
                    "dimension_id": "dim-validation-selection",
                    "operation": "hypothesis_changed",
                    "from_hypothesis_id": "hyp-valid-holdout",
                    "to_hypothesis_id": "hyp-valid-cv",
                },
                {
                    "dimension_id": "dim-ensemble",
                    "operation": "dimension_activated",
                    "from_hypothesis_id": None,
                    "to_hypothesis_id": "hyp-ensemble-stacking",
                },
            ],
        )

    def test_bulk_graph_stats_match_incremental_construction(self) -> None:
        ledger = {"records": _records()}
        bulk = Graph.from_ledger(ledger)
        online = Graph()
        for record in ledger["records"]:
            score = CRASH if record["status"] == "crash" else record["final_best_score"]
            online._add(
                record["run_id"], record["op"], record["source_run_ids"], None,
                score, record["status"],
            )
        for run_id in bulk.nodes:
            self.assertEqual(
                (bulk.N(run_id), bulk.nodes[run_id].ec, bulk.V_max(run_id)),
                (online.N(run_id), online.nodes[run_id].ec, online.V_max(run_id)),
            )


class SemanticEdgePersistenceTests(unittest.TestCase):
    """The real CLI persists helper-derived receipts and the graph reuses them."""

    def test_real_cli_persists_and_renders_semantic_edges(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            background_path = tmp_path / "background.md"
            point_path = tmp_path / "point.json"
            policy_path = tmp_path / "policy.json"
            ledger_path = tmp_path / "ledger.json"
            background_path.write_text(background_text(registry))

            def add_record(run_id: str, op: str, parents: list[str], point: dict) -> None:
                point_path.write_text(json.dumps(point))
                policy_path.write_text(json.dumps(policy_receipt(op, parents, point)))
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tools" / "ledger.py"),
                        "add-record",
                        "--ledger", str(ledger_path),
                        "--task", "hard-interactions",
                        "--run-id", run_id,
                        "--op", op,
                        "--source-run-ids", ",".join(parents),
                        "--background", str(background_path),
                        "--semantic-point", str(point_path),
                        "--policy-receipt", str(policy_path),
                        "--idea", f"Complete fixture solution {run_id} at the selected point.",
                        "--change", f"fixture change for {op} run {run_id}",
                        "--candidate-name-hint", f"fixture_{run_id}",
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            def record_run(run_id: str, score: float) -> None:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tools" / "ledger.py"),
                        "record-run",
                        "--ledger", str(ledger_path),
                        "--task", "hard-interactions",
                        "--run-id", run_id,
                        "--final-best-score", str(score),
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            add_record("000", "fresh", [], baseline)
            record_run("000", 0.5)
            add_record("001", "improve", ["000"], filtered)
            record_run("001", 0.4)

            stored = json.loads(ledger_path.read_text())
            self.assertEqual(stored["records"][0]["semantic_edges"], [])
            edge = stored["records"][1]["semantic_edges"][0]
            self.assertEqual(edge["edge_id"], "sedge-000-001")
            self.assertEqual(edge["parent_run_id"], "000")
            self.assertEqual(edge["child_run_id"], "001")

            view = render_incremental(stored, top=1, bottom=1)
            rendered = next(item for item in view["delta_edges"] if item["child"] == "001")
            self.assertEqual(rendered["semantic_edge"], edge)

            removed = copy.deepcopy(stored)
            del removed["records"][1]["semantic_edges"]
            errors = validate_registry(registry, ledger=removed)
            self.assertTrue(any("semantic_edges" in error for error in errors), errors)

            forged = copy.deepcopy(stored)
            forged["records"][1]["semantic_edges"][0]["change_class"] = "same_point"
            errors = validate_registry(registry, ledger=forged)
            self.assertTrue(any("semantic_edges" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
