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

Invoked by the deterministic `DeepTuner` when n_dims ≤ 2. Reports
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
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    EvaluationBudgetExhausted,
    DeepTuneTimeExhausted,
    resolve_score_fn,
    resolve_preflight_fn,
    timed_eval,
    timed_preflight,
    PatienceMonitor,
    append_preflight_attempt,
    append_trial,
    attempted_config_identities,
    bind_phase_c_objective_reservation,
    cancel_phase_c_objective_attempt,
    cast_params_to_search_space,
    clamp_search_space_to_preflight,
    commit_phase_c_objective_trial,
    deduplicate_configs,
    deep_tune_stage_elapsed,
    deep_tune_time_budget,
    deep_tune_time_remaining,
    ensure_deep_tune_time_remaining,
    is_config_infeasible_error,
    load_candidate_modules,
    objective_attempt_admitted,
    prepare_phase_c_objective_attempt,
    prior_patience_state,
    read_deferred_configs,
    search_space_for_json,
    set_stage_meta,
    split_configs_by_space,
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
        if low == high:
            return [low]
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
    time_budget = deep_tune_time_budget(
        args.candidate_path,
        args.tune_report_json,
        "grid",
    )

    def close_time_exhausted(
        *,
        trials_completed: int = 0,
        trials_attempted: int = 0,
        preflight_rejections: int = 0,
    ) -> int:
        elapsed_seconds = deep_tune_stage_elapsed(time_budget)
        set_stage_meta(
            args.tune_report_json,
            "grid",
            status="time_exhausted",
            elapsed_seconds=elapsed_seconds,
            early_stopped=True,
            early_stop_reason="time_budget",
            preflight_rejections=preflight_rejections,
            time_limit_seconds=time_budget["limit_seconds"],
        )
        write_json({
            "method": "grid",
            "status": "time_exhausted",
            "reason": "candidate deep-tune wall-clock allocation exhausted",
            "trials_completed": trials_completed,
            "trials_attempted": trials_attempted,
            "preflight_rejections": preflight_rejections,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "time_limit_seconds": time_budget["limit_seconds"],
        })
        return 0

    if time_budget["remaining_seconds"] <= 0:
        return close_time_exhausted()

    try:
        ensure_deep_tune_time_remaining(time_budget)
    except DeepTuneTimeExhausted:
        return close_time_exhausted()
    train_module, prepare_module = load_candidate_modules(
        args.candidate_path,
        expected_execution_revision=time_budget[
            "candidate_execution_revision"
        ],
    )
    search_space = train_module.SEARCH_SPACE
    make_model = train_module.make_model
    evaluate = resolve_score_fn(prepare_module, args.candidate_path)
    preflight_enabled = resolve_preflight_fn(prepare_module, args.candidate_path) is not None
    if preflight_enabled:
        # Clamp the box to the preflight-feasible region before searching.
        # Anything residual that still fails is rejected by preflight.
        try:
            search_space = clamp_search_space_to_preflight(
                search_space,
                getattr(train_module, "BASE_PARAMS", None),
                args.candidate_path,
                args.tune_report_json,
                admission_check=lambda: ensure_deep_tune_time_remaining(
                    time_budget
                ),
                phase_time_limit_seconds=lambda: deep_tune_time_remaining(
                    time_budget
                ),
                expected_execution_revision=time_budget[
                    "candidate_execution_revision"
                ],
            )
        except DeepTuneTimeExhausted:
            return close_time_exhausted()

    try:
        ensure_deep_tune_time_remaining(time_budget)
    except DeepTuneTimeExhausted:
        return close_time_exhausted()
    keys = list(search_space.keys())
    grids = [expand_entry(search_space[k], args.resolution) for k in keys]
    total = 1
    for g in grids:
        total *= len(g)
    report = json.loads(args.tune_report_json.read_text())
    existing_grid_trials = [
        trial
        for stage in report.get("phase_c", {}).get("stages", [])
        if isinstance(stage, dict) and stage.get("method") == "grid"
        for trial in stage.get("trials", [])
        if isinstance(trial, dict)
    ]

    try:
        ensure_deep_tune_time_remaining(time_budget)
    except DeepTuneTimeExhausted:
        return close_time_exhausted()
    # A fresh oversized grid rejects into the deterministic fallback. A resumed
    # grid already passed that admission once; continue from its unseen points
    # and let the atomic remaining allocation stop it exactly.
    if total > args.max_trials and not existing_grid_trials:
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

    try:
        ensure_deep_tune_time_remaining(time_budget)
    except DeepTuneTimeExhausted:
        return close_time_exhausted()
    combos = list(itertools.product(*grids))
    rng = random.Random(args.seed)
    rng.shuffle(combos)

    # Evaluate the deferred warm configs FIRST (proposed at step 0+1 but not
    # evaluated there), then the grid sweep. They count as normal trials.
    # Deferred configs outside the (possibly clamped) box are skipped — never
    # attempted, no budget, no patience effect — and accounted via
    # deferred_skipped_outside_space.
    deferred_in_space, deferred_outside = split_configs_by_space(
        read_deferred_configs(args.tune_report_json), search_space
    )
    attempted_identities = attempted_config_identities(
        args.tune_report_json,
        search_space,
    )
    deferred = [
        cast_params_to_search_space(dict(p), search_space)
        for p in deferred_in_space
    ]
    deferred, deferred_skipped_seen, seen = deduplicate_configs(
        deferred,
        seen=attempted_identities,
    )
    grid_configs = [
        cast_params_to_search_space(dict(zip(keys, combo)), search_space)
        for combo in combos
    ]
    grid_configs, grid_skipped_seen, _ = deduplicate_configs(
        grid_configs,
        seen=seen,
    )
    param_dicts = deferred + grid_configs

    # Seed best AND streak from the persisted trial history: a restarted search
    # continues the patience window instead of getting a fresh one.
    prior_best, prior_streak = prior_patience_state(args.tune_report_json)
    monitor = PatienceMonitor(
        patience=args.patience,
        start_best=prior_best,
        start_since=prior_streak,
    )
    set_stage_meta(args.tune_report_json, "grid", status="running",
                   deferred_skipped_outside_space=len(deferred_outside),
                   deferred_skipped_already_seen=deferred_skipped_seen,
                   grid_skipped_already_seen=grid_skipped_seen)

    prior_grid_scores = [
        trial
        for trial in existing_grid_trials
        if isinstance(trial, dict)
        and isinstance(trial.get("params"), dict)
        and isinstance(trial.get("score"), (int, float))
        and not isinstance(trial.get("score"), bool)
        and math.isfinite(float(trial["score"]))
    ]
    prior_grid_best = (
        min(prior_grid_scores, key=lambda trial: float(trial["score"]))
        if prior_grid_scores
        else None
    )
    best_params = (
        dict(prior_grid_best["params"]) if prior_grid_best is not None else None
    )
    best_score = (
        float(prior_grid_best["score"])
        if prior_grid_best is not None
        else math.inf
    )
    trials_done = 0
    trials_attempted = 0
    early_stopped = False
    early_stop_reason = "none"
    budget_exhausted = False
    budget_exhausted_scope = None
    time_exhausted = False
    preflight_rejections = 0
    failure_refs = []

    for params in param_dicts:
        try:
            ensure_deep_tune_time_remaining(time_budget)
        except DeepTuneTimeExhausted:
            time_exhausted = True
            early_stopped = True
            early_stop_reason = "time_budget"
            break
        if preflight_enabled:
            try:
                preflight_result = timed_preflight(
                    params,
                    args.candidate_path,
                    phase_time_limit_seconds=lambda: deep_tune_time_remaining(
                        time_budget
                    ),
                    expected_execution_revision=time_budget[
                        "candidate_execution_revision"
                    ],
                )
            except DeepTuneTimeExhausted:
                time_exhausted = True
                early_stopped = True
                early_stop_reason = "time_budget"
                break
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
                try:
                    ensure_deep_tune_time_remaining(time_budget)
                except DeepTuneTimeExhausted:
                    time_exhausted = True
                    early_stopped = True
                    early_stop_reason = "time_budget"
                    break
                # A scoreless trial is a non-improvement: count it toward
                # patience so failure streaks cannot sidestep early stopping.
                if monitor.update_failed():
                    early_stopped = True
                    early_stop_reason = "patience"
                    break
                continue
            append_preflight_attempt(
                args.tune_report_json,
                source="grid",
                params=params,
                status="ok",
                result=preflight_result or {"status": "ok"},
            )
            try:
                ensure_deep_tune_time_remaining(time_budget)
            except DeepTuneTimeExhausted:
                time_exhausted = True
                early_stopped = True
                early_stop_reason = "time_budget"
                break
        objective_intent = None
        try:
            ensure_deep_tune_time_remaining(time_budget)
            objective_intent = prepare_phase_c_objective_attempt(
                args.tune_report_json,
                args.candidate_path,
                "grid",
                params,
                time_budget["candidate_execution_revision"],
            )
            score = timed_eval(
                evaluate,
                make_model,
                params,
                args.candidate_path,
                phase="phase_c",
                method="grid",
                phase_time_limit_seconds=lambda: deep_tune_time_remaining(
                    time_budget
                ),
                expected_execution_revision=time_budget[
                    "candidate_execution_revision"
                ],
                on_objective_reserved=lambda receipt: (
                    bind_phase_c_objective_reservation(
                        args.tune_report_json,
                        args.candidate_path,
                        "grid",
                        objective_intent,
                        receipt,
                    )
                ),
            )
        except DeepTuneTimeExhausted as exc:
            if exc.attempt_reserved:
                trials_attempted += 1
                failure = record_failure(
                    report_path=args.tune_report_json,
                    candidate_path=args.candidate_path,
                    phase="phase_c",
                    method="grid",
                    params=params,
                    error=exc,
                    traceback_text=traceback.format_exc(),
                )
                commit_phase_c_objective_trial(
                    args.tune_report_json,
                    args.candidate_path,
                    "grid",
                    objective_intent,
                    {
                        "params": params,
                        "score": None,
                        "status": "failed",
                        "time_exhausted": True,
                        "config_infeasible": False,
                        **failure,
                    },
                )
                if failure["failure_ref"] not in failure_refs:
                    failure_refs.append(failure["failure_ref"])
            elif objective_intent is not None:
                cancel_phase_c_objective_attempt(
                    args.tune_report_json,
                    args.candidate_path,
                    "grid",
                    objective_intent,
                )
            time_exhausted = True
            early_stopped = True
            early_stop_reason = "time_budget"
            break
        except EvaluationBudgetExhausted as exc:
            if objective_intent is not None:
                cancel_phase_c_objective_attempt(
                    args.tune_report_json,
                    args.candidate_path,
                    "grid",
                    objective_intent,
                )
            budget_exhausted = True
            budget_exhausted_scope = exc.scope
            early_stopped = True
            early_stop_reason = "evaluation_budget"
            break
        except Exception as exc:
            if not objective_attempt_admitted(exc):
                if objective_intent is not None:
                    cancel_phase_c_objective_attempt(
                        args.tune_report_json,
                        args.candidate_path,
                        "grid",
                        objective_intent,
                    )
                raise
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
            commit_phase_c_objective_trial(
                args.tune_report_json,
                args.candidate_path,
                "grid",
                objective_intent,
                {
                    "params": params,
                    "score": None,
                    "status": "failed",
                    "config_infeasible": is_config_infeasible_error(exc),
                    **failure,
                },
            )
            if failure["failure_ref"] not in failure_refs:
                failure_refs.append(failure["failure_ref"])
            if monitor.update_failed():
                early_stopped = True
                early_stop_reason = "patience"
                break
            continue
        trials_attempted += 1
        commit_phase_c_objective_trial(
            args.tune_report_json,
            args.candidate_path,
            "grid",
            objective_intent,
            {"params": params, "score": score},
        )
        trials_done += 1
        improved = score < best_score
        if improved:
            best_score = score
            best_params = params
        if monitor.update(score):
            early_stopped = True
            early_stop_reason = "patience"
            break

    stage_elapsed = deep_tune_stage_elapsed(time_budget)

    if best_params is None and budget_exhausted:
        set_stage_meta(
            args.tune_report_json,
            "grid",
            status="budget_exhausted",
            elapsed_seconds=stage_elapsed,
            early_stopped=True,
            preflight_rejections=preflight_rejections,
            budget_exhausted_scope=budget_exhausted_scope,
        )
        write_json({
            "method": "grid",
            "status": "budget_exhausted",
            "reason": "evaluation allocation exhausted before score_fn",
            "budget_exhausted_scope": budget_exhausted_scope,
            "trials_completed": trials_done,
            "trials_attempted": trials_attempted,
            "preflight_rejections": preflight_rejections,
            "elapsed_seconds": round(stage_elapsed, 1),
        })
        return 0

    if best_params is None and time_exhausted:
        return close_time_exhausted(
            trials_completed=trials_done,
            trials_attempted=trials_attempted,
            preflight_rejections=preflight_rejections,
        )

    if best_params is None:
        # Every combo errored — surface a failed stage instead of "ok" with a null best.
        set_stage_meta(args.tune_report_json, "grid", status="failed",
                       elapsed_seconds=stage_elapsed, early_stopped=early_stopped)
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
            "elapsed_seconds": round(stage_elapsed, 1),
            "search_space": search_space_for_json(search_space),
        })
        return 0

    set_stage_meta(
        args.tune_report_json,
        "grid",
        status="ok",
        elapsed_seconds=stage_elapsed,
        early_stopped=early_stopped,
        preflight_rejections=preflight_rejections,
        budget_exhausted=budget_exhausted,
        time_limit_seconds=time_budget["limit_seconds"],
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
        "deferred_skipped_outside_space": len(deferred_outside),
        "deferred_skipped_already_seen": deferred_skipped_seen,
        "grid_skipped_already_seen": grid_skipped_seen,
        "trials_planned": total,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "elapsed_seconds": round(stage_elapsed, 1),
        "time_limit_seconds": time_budget["limit_seconds"],
        "search_space": search_space_for_json(search_space),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
