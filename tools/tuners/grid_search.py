#!/usr/bin/env python3
"""Grid hyperparameter search for one autoresearch candidate.

Reads the candidate's SEARCH_SPACE, expands each entry to a discrete grid,
shuffles the combos with a fixed seed for fair patience early-stopping,
and evaluates each combo via prepare.evaluate_config_for_tuning. Each
trial is appended incrementally to tune_report.json under
phase_c.stages[grid].

Patience early-stopping: when the configured number of consecutive
non-improving combos is reached, the search stops. The patience monitor
is seeded with the best score from prior trials in tune_report.json
(phase_a warm-start + any earlier phase_c stages).

Invoked by the tuner-orchestrator agent when n_dims ≤ 2. Reports
rejection via stdout JSON if total combos exceed --max-trials so the
orchestrator can fall back to a different method.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    PatienceMonitor,
    append_trial,
    cast_params_to_search_space,
    load_candidate_modules,
    prior_best_score,
    read_prior_trials,
    search_space_for_json,
    write_json,
)


def expand_entry(entry, resolution: int) -> list:
    kind = entry[0]
    if kind == "categorical":
        return list(entry[1])
    if kind == "int":
        low, high = int(entry[1]), int(entry[2])
        if high - low + 1 <= resolution:
            return list(range(low, high + 1))
        return [int(round(v)) for v in np.linspace(low, high, resolution)]
    if kind == "float":
        low, high = float(entry[1]), float(entry[2])
        if len(entry) >= 4 and entry[3] == "log":
            return list(np.logspace(math.log10(low), math.log10(high), resolution))
        return list(np.linspace(low, high, resolution))
    raise ValueError(f"unknown SEARCH_SPACE entry: {entry}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--tune-report-json", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=5)
    parser.add_argument("--max-trials", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_module, prepare_module = load_candidate_modules(args.candidate_path)
    search_space = train_module.SEARCH_SPACE
    make_model = train_module.make_model
    evaluate = prepare_module.evaluate_config_for_tuning

    keys = list(search_space.keys())
    grids = [expand_entry(search_space[k], args.resolution) for k in keys]
    total = 1
    for g in grids:
        total *= len(g)

    if total > args.max_trials:
        write_json({
            "method": "grid",
            "status": "rejected",
            "reason": (
                f"grid size {total} exceeds max_trials {args.max_trials}; "
                "orchestrator should fall back to bo or cmaes"
            ),
            "grid_size": total,
            "max_trials": args.max_trials,
            "search_space": search_space_for_json(search_space),
        })
        return 0

    combos = list(itertools.product(*grids))
    rng = random.Random(args.seed)
    rng.shuffle(combos)

    prior_trials = read_prior_trials(args.tune_report_json)
    monitor = PatienceMonitor(
        patience=args.patience,
        lower_is_better=args.lower_is_better,
        start_best=prior_best_score(prior_trials, args.lower_is_better),
    )

    started = time.time()
    best_params = None
    best_score = math.inf if args.lower_is_better else -math.inf
    trials_done = 0
    early_stopped = False
    early_stop_reason = "none"

    for combo in combos:
        params = cast_params_to_search_space(dict(zip(keys, combo)), search_space)
        score = evaluate(make_model, params)
        append_trial(args.tune_report_json, "grid", {"params": params, "score": score})
        trials_done += 1
        improved = score < best_score if args.lower_is_better else score > best_score
        if improved:
            best_score = score
            best_params = params
        if monitor.update(score):
            early_stopped = True
            early_stop_reason = "patience"
            break

    elapsed = time.time() - started

    write_json({
        "method": "grid",
        "status": "ok",
        "best_params": best_params,
        "best_score": best_score if trials_done > 0 else None,
        "trials_completed": trials_done,
        "trials_planned": total,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "elapsed_seconds": round(elapsed, 1),
        "search_space": search_space_for_json(search_space),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
