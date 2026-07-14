#!/usr/bin/env python3
"""Grid hyperparameter search for one autoresearch candidate.

Reads the candidate's SEARCH_SPACE, expands each entry to a discrete grid,
shuffles the combos with a fixed seed for fair patience early-stopping,
and evaluates each combo via the task's `score_fn`. Each
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
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    EvaluationFailure,
    resolve_score_fn,
    timed_eval,
    PatienceMonitor,
    append_trial,
    cast_params_to_search_space,
    load_candidate_modules,
    load_run_cfg,
    new_search_budget,
    prior_best_score,
    read_deferred_configs,
    read_global_eval_budget,
    read_prior_trials,
    read_stage_trials,
    search_space_for_json,
    set_stage_meta,
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
    parser.add_argument("--tune-report-json", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=5)
    parser.add_argument("--max-trials", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_cfg = load_run_cfg(args.candidate_path, "tuner")
    max_consecutive_failures = max(1, int(run_cfg.get("max_consecutive_failures", 5)))

    train_module, prepare_module = load_candidate_modules(args.candidate_path)
    search_space = train_module.SEARCH_SPACE
    make_model = train_module.make_model
    evaluate = resolve_score_fn(prepare_module, args.candidate_path)

    keys = list(search_space.keys())
    grids = [expand_entry(search_space[k], args.resolution) for k in keys]
    total = 1
    for g in grids:
        total *= len(g)

    if total > args.max_trials:
        set_stage_meta(args.tune_report_json, "grid", status="rejected")
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

    # Evaluate the deferred warm configs FIRST (proposed at step 0+1 but not
    # evaluated there), then the grid sweep. They count as normal trials.
    deferred = [cast_params_to_search_space(dict(p), search_space)
                for p in read_deferred_configs(args.tune_report_json)]
    proposed = deferred + [cast_params_to_search_space(dict(zip(keys, combo)), search_space)
                           for combo in combos]
    param_dicts = []
    proposed_keys = set()
    for params in proposed:
        key = json.dumps(params, sort_keys=True, default=str)
        if key not in proposed_keys:
            proposed_keys.add(key)
            param_dicts.append(params)

    prior_trials = read_prior_trials(args.tune_report_json)
    monitor = PatienceMonitor(
        patience=args.patience,
        start_best=prior_best_score(prior_trials),
    )

    existing_trials = read_stage_trials(args.tune_report_json, "grid")
    eval_budget = read_global_eval_budget(args.candidate_path)
    allowed = eval_budget["remaining"]
    target_trials = len(param_dicts) if allowed is None else min(len(param_dicts), int(allowed))
    target_trials = max(len(existing_trials), target_trials)
    existing_keys = {
        json.dumps(t.get("params"), sort_keys=True, default=str)
        for t in existing_trials if isinstance(t.get("params"), dict)
    }
    pending_params = [
        p for p in param_dicts
        if json.dumps(p, sort_keys=True, default=str) not in existing_keys
    ][:max(0, target_trials - len(existing_trials))]
    successful_existing = [
        t for t in existing_trials
        if isinstance(t.get("score"), (int, float)) and math.isfinite(float(t["score"]))
    ]

    started = time.time()
    budget = new_search_budget(args.candidate_path)
    if successful_existing:
        prior_best = min(successful_existing, key=lambda t: t["score"])
        best_params = prior_best["params"]
        best_score = float(prior_best["score"])
    else:
        best_params = None
        best_score = math.inf
    if target_trials == 0:
        finite_priors = [
            t for t in prior_trials
            if isinstance(t.get("score"), (int, float)) and math.isfinite(float(t["score"]))
        ]
        if finite_priors:
            prior_best = min(finite_priors, key=lambda t: t["score"])
            best_params = prior_best["params"]
            best_score = float(prior_best["score"])
    trials_done = len(existing_trials)
    early_stopped = False
    early_stop_reason = "none"
    failure_streak = 0

    for params in pending_params:
        if budget.exhausted:
            early_stop_reason = "wall_clock_budget"
            break
        try:
            score = timed_eval(evaluate, make_model, params, args.candidate_path)
        except Exception as exc:
            # A bad param combo must not kill the sweep: record it and skip.
            append_trial(args.tune_report_json, "grid",
                         {"params": params, "score": None,
                          "status": exc.status if isinstance(exc, EvaluationFailure) else "error",
                          "error": repr(exc)})
            trials_done += 1
            failure_streak += 1
            if failure_streak >= max_consecutive_failures:
                early_stopped = True
                early_stop_reason = "failure_circuit_breaker"
                break
            continue
        failure_streak = 0
        append_trial(args.tune_report_json, "grid", {"params": params, "score": score})
        trials_done += 1
        improved = score < best_score
        if improved:
            best_score = score
            best_params = params
        if monitor.update(score):
            early_stopped = True
            early_stop_reason = "patience"
            break

    elapsed = time.time() - started

    if best_params is None:
        # Every combo errored — surface a failed stage instead of "ok" with a null best.
        set_stage_meta(args.tune_report_json, "grid", status="failed",
                       elapsed_seconds=round(elapsed, 1), early_stopped=early_stopped,
                       stop_reason=early_stop_reason, target_trials=target_trials,
                       trials_attempted=trials_done)
        write_json({
            "method": "grid",
            "status": "failed",
            "reason": "all grid trials errored; no completed trial",
            "trials_completed": trials_done,
            "trials_planned": total,
            "early_stopped": early_stopped,
            "early_stop_reason": early_stop_reason,
            "elapsed_seconds": round(elapsed, 1),
            "search_space": search_space_for_json(search_space),
        })
        return 0

    partial = trials_done < target_trials and not early_stopped
    if not partial and target_trials < len(param_dicts):
        early_stop_reason = "global_eval_budget"
    status = "partial" if partial else "ok"
    set_stage_meta(args.tune_report_json, "grid", status=status,
                   elapsed_seconds=round(elapsed, 1), early_stopped=early_stopped,
                   stop_reason=early_stop_reason, target_trials=target_trials,
                   trials_attempted=trials_done, wall_budget_seconds=budget.seconds)

    write_json({
        "method": "grid",
        "status": status,
        "best_params": best_params,
        "best_score": best_score,
        "trials_completed": trials_done,
        "trials_planned": target_trials,
        "requested_trials": len(param_dicts),
        "remaining_trials": max(0, target_trials - trials_done),
        "global_eval_budget": eval_budget,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "elapsed_seconds": round(elapsed, 1),
        "wall_budget_seconds": budget.seconds,
        "search_space": search_space_for_json(search_space),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
