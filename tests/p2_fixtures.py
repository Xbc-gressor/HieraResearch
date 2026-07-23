"""Shared no-I/O P2 ledger fixtures built on the frozen toy registry."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from semantic_evidence import build_semantic_edges  # noqa: E402
from semantic_space import complete_point  # noqa: E402


def belief_ledger(registry: dict) -> dict:
    baseline = complete_point(registry)
    filtered = complete_point(
        registry, {"dim-data-curation": "hyp-data-filtered"}
    )
    records = [
        {
            "run_id": "000", "source_run_ids": [], "semantic_point": baseline,
            "semantic_edges": [], "status": "keep", "final_best_score": 0.40,
            "dag_revision": 1,
        },
        {
            "run_id": "001", "source_run_ids": ["000"], "semantic_point": filtered,
            "status": "discard", "final_best_score": 0.50, "dag_revision": 2,
        },
        {
            "run_id": "002", "source_run_ids": [], "semantic_point": baseline,
            "semantic_edges": [], "status": "keep", "final_best_score": 0.41,
            "dag_revision": 3,
        },
        {
            "run_id": "003", "source_run_ids": ["002"], "semantic_point": filtered,
            "status": "discard", "final_best_score": 0.52, "dag_revision": 4,
        },
    ]
    records[1]["semantic_edges"] = build_semantic_edges(records[:1], records[1])
    records[3]["semantic_edges"] = build_semantic_edges(records[:3], records[3])
    return {"records": records, "dag_revision": 4}
