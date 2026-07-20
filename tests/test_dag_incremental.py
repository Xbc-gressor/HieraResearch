from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from background_contract import derive_compact_lineage  # noqa: E402
from got_graph import CRASH, Graph, render_incremental  # noqa: E402
from ledger import _set_experience, _touch_dag_record  # noqa: E402


def _records() -> list[dict]:
    rows = [
        ("000", ["tf-01"], "fresh", "keep", 0.5),
        ("001", ["000"], "improve", "keep", 0.4),
        ("002", ["tf-02"], "fresh", "discard", 0.8),
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

        registry = {
            "schema_version": 2,
            "directions": [
                {"id": "tf-01", "literature_credibility": "preliminary"},
                {"id": "tf-02", "literature_credibility": "preliminary"},
            ],
        }
        stored["experience"]["dag_revision"] = 4
        compact = derive_compact_lineage(registry, stored, limit=2)
        self.assertEqual(compact["cursor"], {"from_revision": 4, "to_revision": 7})
        self.assertEqual(
            [item["run_id"] for item in compact["directions"]["tf-02"]["delta_runs"]["descendant_runs"]],
            ["005"],
        )
        self.assertLessEqual(
            len(compact["directions"]["tf-01"]["representative_runs"]["combination_runs"]),
            2,
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


if __name__ == "__main__":
    unittest.main()
