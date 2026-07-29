"""Minimal contract checks for the deterministic GoT calculation layer.

Keep this validator small. It protects only stable boundaries shared by
``got_graph``, ``got_cdag``, and ``got_select``; policy quality and search
trajectories belong in ``tools/got_benchmark``.

Run: ``python3 tools/validate_got.py`` (exit code 0 means all checks passed).
"""
from __future__ import annotations

from got_cdag import c_dag
from got_graph import Graph
from got_select import DEFAULT_CFG, decide


def graph_contract() -> None:
    ledger = {
        "records": [
            {
                "run_id": "000",
                "source_run_ids": [],
                "op": "fresh",
                "status": "keep",
                "final_best_score": 0.5,
            },
            {
                "run_id": "001",
                "source_run_ids": ["000"],
                "op": "improve",
                "status": "keep",
                "final_best_score": 0.4,
            },
            {
                "run_id": "002",
                "source_run_ids": ["001"],
                "op": "improve",
                "status": "crash",
                "final_best_score": None,
            },
            {
                "run_id": "003",
                "source_run_ids": ["001"],
                "op": "improve",
                "status": "pending",
                "final_best_score": None,
            },
        ]
    }

    graph = Graph.from_ledger(ledger)

    assert set(graph.nodes) == {"000", "001", "002"}
    assert graph.parents("001") == ["000"]
    assert max(graph.r_map(), key=graph.r_map().get) == "001"
    assert graph.nodes["001"].ec == 1
    assert graph.N("001") == 1
    assert "002" not in graph.F()


def ancestry_contract() -> None:
    graph = Graph()
    root = graph.add("fresh", [], score=0.5)
    left = graph.add("improve", [root], score=0.4)
    right = graph.add("improve", [root], score=0.4)
    independent = graph.add("fresh", [], score=0.6)

    assert abs(c_dag(graph, left, right, 0.6)) < 1e-9
    assert abs(c_dag(graph, left, independent, 0.6) - 1.0) < 1e-9


def select_smoke() -> None:
    decision = decide(Graph(), dict(DEFAULT_CFG))

    assert decision["kind"] == "fresh"
    assert decision["actions"]
    assert all(action["op"] == "fresh" for action in decision["actions"])


def main() -> None:
    graph_contract()
    ancestry_contract()
    select_smoke()
    print("✓ GoT graph, ancestry, and SELECT contracts passed")


if __name__ == "__main__":
    main()
