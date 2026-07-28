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
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    EvaluationBudgetExhausted,
    resolve_score_fn,
    resolve_preflight_fn,
    timed_eval,
    timed_preflight,
    PatienceMonitor,
    append_preflight_attempt,
    append_trial,
    cast_params_to_search_space,
    load_candidate_modules,
    prior_best_score,
    read_deferred_configs,
    read_prior_trials,
    search_space_for_json,
    set_stage_meta,
    write_json,
)
from failure_artifacts import record_failure  # noqa: E402


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

    train_module, prepare_module = load_candidate_modules(args.candidate_path)
    search_space = train_module.SEARCH_SPACE
    make_model = train_module.make_model
    evaluate = resolve_score_fn(prepare_module, args.candidate_path)
    preflight_enabled = resolve_preflight_fn(prepare_module, args.candidate_path) is not None

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
    param_dicts = deferred + [cast_params_to_search_space(dict(zip(keys, combo)), search_space)
                              for combo in combos]

    prior_trials = read_prior_trials(args.tune_report_json)
    monitor = PatienceMonitor(
        patience=args.patience,
        start_best=prior_best_score(prior_trials),
    )

    started = time.time()
    best_params = None
    best_score = math.inf
    trials_done = 0
    trials_attempted = 0
    early_stopped = False
    early_stop_reason = "none"
    budget_exhausted = False
    preflight_rejections = 0
    failure_refs = []

    for params in param_dicts:
        if preflight_enabled:
            try:
                preflight_result = timed_preflight(params, args.candidate_path)
            except Exception as exc:
                failure = record_failure(
                    report_path=args.tune_report_json,
                    candidate_path=args.candidate_path,
                    phase="preflight",
                    method="grid",
                    params=params,
                    error=exc,
                    traceback_text=traceback.format_exc(),
                )
                append_preflight_attempt(
                    args.tune_report_json,
                    source="grid",
                    params=params,
                    status="failed",
                    failure=failure,
                )
                append_trial(
                    args.tune_report_json,
                    "grid",
                    {
                        "params": params,
                        "score": None,
                        "status": "preflight_rejected",
                        **failure,
                    },
                )
                preflight_rejections += 1
                continue
            append_preflight_attempt(
                args.tune_report_json,
                source="grid",
                params=params,
                status="ok",
                result=preflight_result or {"status": "ok"},
            )
        try:
            score = timed_eval(
                evaluate,
                make_model,
                params,
                args.candidate_path,
                phase="phase_c",
                method="grid",
            )
        except EvaluationBudgetExhausted:
            budget_exhausted = True
            early_stopped = True
            early_stop_reason = "evaluation_budget"
            break
        except Exception as exc:
            trials_attempted += 1
            # A bad param combo must not kill the sweep: record it and skip.
            failure = record_failure(
                report_path=args.tune_report_json,
                candidate_path=args.candidate_path,
                phase="phase_c",
                method="grid",
                params=params,
                error=exc,
                traceback_text=traceback.format_exc(),
            )
            append_trial(args.tune_report_json, "grid",
                         {"params": params, "score": None, "status": "failed", **failure})
            if failure["failure_ref"] not in failure_refs:
                failure_refs.append(failure["failure_ref"])
            continue
        trials_attempted += 1
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

    if best_params is None and budget_exhausted:
        set_stage_meta(
            args.tune_report_json,
            "grid",
            status="budget_exhausted",
            elapsed_seconds=round(elapsed, 1),
            early_stopped=True,
            preflight_rejections=preflight_rejections,
        )
        write_json({
            "method": "grid",
            "status": "budget_exhausted",
            "reason": "global evaluation budget exhausted before score_fn",
            "trials_completed": trials_done,
            "trials_attempted": trials_attempted,
            "preflight_rejections": preflight_rejections,
            "elapsed_seconds": round(elapsed, 1),
        })
        return 0

    if best_params is None:
        # Every combo errored — surface a failed stage instead of "ok" with a null best.
        set_stage_meta(args.tune_report_json, "grid", status="failed",
                       elapsed_seconds=round(elapsed, 1), early_stopped=early_stopped)
        write_json({
            "method": "grid",
            "status": "failed",
            "reason": "all grid trials errored; no completed trial",
            "trials_completed": trials_done,
            "trials_attempted": trials_attempted,
            "preflight_rejections": preflight_rejections,
            "trials_planned": total,
            "early_stopped": early_stopped,
            "early_stop_reason": early_stop_reason,
            "failure_refs": failure_refs[-3:],
            "elapsed_seconds": round(elapsed, 1),
            "search_space": search_space_for_json(search_space),
        })
        return 0

    set_stage_meta(
        args.tune_report_json,
        "grid",
        status="ok",
        elapsed_seconds=round(elapsed, 1),
        early_stopped=early_stopped,
        preflight_rejections=preflight_rejections,
        budget_exhausted=budget_exhausted,
    )

    write_json({
        "method": "grid",
        "status": "ok",
        "best_params": best_params,
        "best_score": best_score,
        "trials_completed": trials_done,
        "trials_attempted": trials_attempted,
        "preflight_rejections": preflight_rejections,
        "budget_exhausted": budget_exhausted,
        "trials_planned": total,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "elapsed_seconds": round(elapsed, 1),
        "search_space": search_space_for_json(search_space),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
