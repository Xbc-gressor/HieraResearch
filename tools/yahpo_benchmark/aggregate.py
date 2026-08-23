from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

from .io import read_json, write_json


def aggregate(root: Path, *, write: bool = True) -> dict[str, Any]:
    root = Path(root)
    results = [read_json(path) for path in sorted(root.glob("cells/*/*/result.json"))]
    rows = [_result_row(result) for result in results]
    usage = {
        "calls": sum(int(row["usage"].get("calls", 0)) for row in results),
        "input_tokens": sum(
            int(row["usage"].get("input_tokens", 0)) for row in results
        ),
        "output_tokens": sum(
            int(row["usage"].get("output_tokens", 0)) for row in results
        ),
        "total_cost_usd": sum(
            float(row["usage"].get("total_cost_usd", 0.0)) for row in results
        ),
    }
    by_optimizer = {}
    for optimizer in sorted({row["optimizer"] for row in rows}):
        selected = [row for row in rows if row["optimizer"] == optimizer]
        endpoint = [row["endpoint_normalized_regret"] for row in selected]
        aucs = [row["normalized_regret_auc"] for row in selected]
        by_optimizer[optimizer] = {
            "runs": len(selected),
            "complete": sum(row["status"] == "complete" for row in selected),
            "median_endpoint_normalized_regret": _median(endpoint),
            "median_normalized_regret_auc": _median(aucs),
            "llm_calls": sum(row["llm_calls"] for row in selected),
            "total_cost_usd": sum(row["total_cost_usd"] for row in selected),
        }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": results[0].get("protocol_id") if results else None,
        "stage": results[0]["stage"] if results else None,
        "seed": results[0].get("seed") if results else None,
        "initial_count": results[0].get("initial_count") if results else None,
        "bo_trials": results[0].get("bo_trials") if results else None,
        "model": next(
            (result.get("model") for result in results if result.get("model")),
            None,
        ),
        "result_count": len(results),
        "rows": rows,
        "by_optimizer": by_optimizer,
        "paired": {
            "ours_vs_hebo_only": _paired(
                rows, "hebo_mace_llm_pool", "hebo_only"
            ),
            "ours_vs_llambo": _paired(
                rows, "hebo_mace_llm_pool", "llambo_modern_batched"
            ),
        },
        "usage": usage,
    }
    if summary["stage"] == "smoke":
        ours = by_optimizer.get("hebo_mace_llm_pool", {}).get(
            "total_cost_usd", 0.0
        )
        llambo = by_optimizer.get("llambo_modern_batched", {}).get(
            "total_cost_usd", 0.0
        )
        summary["cost_projection"] = {
            "formula": "1.20 * 40 * (smoke_ours_usd + smoke_llambo_usd)",
            "smoke_ours_usd": ours,
            "smoke_llambo_usd": llambo,
            "stage1_projected_usd": 1.2 * 40.0 * (ours + llambo),
        }
    if write:
        write_json(root / "summary.json", summary)
    return summary


def _result_row(result: dict[str, Any]) -> dict[str, Any]:
    complete = result.get("status") == "complete"
    endpoint = (
        float(result["terminal_normalized_regret"])
        if complete
        else float("inf")
    )
    bo_regrets = [
        float(row["normalized_regret"])
        for row in result.get("trials", [])
        if int(row["trial_no"]) > int(result["initial_count"])
    ]
    expected = int(result["bo_trials"])
    auc = (
        statistics.fmean(bo_regrets)
        if complete and len(bo_regrets) == expected
        else float("inf")
    )
    usage = result.get("usage") or {}
    return {
        "task_key": result["task"]["key"],
        "seed": int(result["seed"]),
        "optimizer": result["optimizer"],
        "status": result["status"],
        "endpoint_normalized_regret": endpoint,
        "normalized_regret_auc": auc,
        "llm_calls": int(usage.get("calls", 0)),
        "total_cost_usd": float(usage.get("total_cost_usd", 0.0)),
    }


def _paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    indexed = {
        (row["task_key"], row["seed"], row["optimizer"]): row for row in rows
    }
    keys = sorted({(row["task_key"], row["seed"]) for row in rows})
    pairs = []
    wins = ties = losses = 0
    finite_deltas = []
    for task_key, seed in keys:
        lhs = indexed.get((task_key, seed, left))
        rhs = indexed.get((task_key, seed, right))
        if lhs is None or rhs is None:
            continue
        a = lhs["endpoint_normalized_regret"]
        b = rhs["endpoint_normalized_regret"]
        if math.isinf(a) and math.isinf(b):
            outcome = "tie"
            delta = None
            ties += 1
        elif math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12):
            outcome = "tie"
            delta = a - b
            ties += 1
        elif a < b:
            outcome = "win"
            delta = a - b
            wins += 1
        else:
            outcome = "loss"
            delta = a - b
            losses += 1
        if delta is not None and math.isfinite(delta):
            finite_deltas.append(delta)
        pairs.append(
            {
                "task_key": task_key,
                "seed": seed,
                "left": a,
                "right": b,
                "delta": delta,
                "outcome": outcome,
            }
        )
    return {
        "left": left,
        "right": right,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "median_delta": _median(finite_deltas),
        "pairs": pairs,
    }


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None
