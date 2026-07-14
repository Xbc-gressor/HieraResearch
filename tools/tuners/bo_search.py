#!/usr/bin/env python3
"""Bayesian-optimization (Optuna TPE) hyperparameter search for one candidate.

Reads prior trials from tune_report.json (phase_a warm-start + any earlier
phase_c stages) and injects them into the Optuna study as completed trials
so TPE can use them as a prior. Each new trial is also appended back to
tune_report.json under phase_c.stages[bo] as it completes.

Evaluates configurations via the task's one `config → score` function
(`score_fn`). Requires `optuna` in the task's uv environment.

Invoked by the tuner-orchestrator agent when 3 ≤ n_dims ≤ 15.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    EvaluationFailure,
    resolve_score_fn,
    timed_eval,
    load_run_cfg,
    PatienceMonitor,
    append_trial,
    cast_params_to_search_space,
    load_candidate_modules,
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


def suggest(trial, key: str, entry) -> object:
    kind = entry[0]
    if kind == "float":
        low, high = float(entry[1]), float(entry[2])
        log = len(entry) >= 4 and entry[3] == "log"
        return trial.suggest_float(key, low, high, log=log)
    if kind == "int":
        return trial.suggest_int(key, int(entry[1]), int(entry[2]))
    if kind == "categorical":
        return trial.suggest_categorical(key, list(entry[1]))
    raise ValueError(f"unknown SEARCH_SPACE entry: {entry}")


def build_distributions(search_space: dict):
    """Translate SEARCH_SPACE into Optuna distribution objects so prior
    trials can be added via study.add_trial(create_trial(...))."""
    import optuna
    dists = {}
    for key, entry in search_space.items():
        kind = entry[0]
        if kind == "float":
            low, high = float(entry[1]), float(entry[2])
            log = len(entry) >= 4 and entry[3] == "log"
            dists[key] = optuna.distributions.FloatDistribution(low, high, log=log)
        elif kind == "int":
            dists[key] = optuna.distributions.IntDistribution(int(entry[1]), int(entry[2]))
        elif kind == "categorical":
            dists[key] = optuna.distributions.CategoricalDistribution(list(entry[1]))
        else:
            raise ValueError(f"unknown SEARCH_SPACE entry: {entry}")
    return dists


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--tune-report-json", required=True, type=Path)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # per-run framework overrides (Phase-3 OFAT): <run_dir>/framework_cfg.json tuner.*
    # explicit flag wins; else framework_cfg.json; else the historical defaults.
    _rc = load_run_cfg(args.candidate_path, "tuner")
    # Defaults are data-driven (dev_plan/hpo-benchmark-report.md): patience=6 was
    # the main HPO suppressor; budget ~40.
    n_trials = args.n_trials if args.n_trials is not None else int(_rc.get("bo_n_trials", 40))
    max_consecutive_failures = max(1, int(_rc.get("max_consecutive_failures", 5)))
    # patience: explicit flag > framework_cfg.tuner.bo_patience (fixed) > ADAPTIVE.
    # Adaptive (computed after SEARCH_SPACE loads, needs n_dims): min(cap, max(floor,
    # 1.5*n_dims)) — low/mid-dim save budget, high-dim ramps to the cap. Benchmark:
    # ~94% of fixed-20's improvement at ~64% of the trial cost.
    patience_override = args.patience if args.patience is not None else _rc.get("bo_patience")

    try:
        import optuna
    except ImportError:
        set_stage_meta(args.tune_report_json, "bo", status="rejected")
        write_json({
            "method": "bo",
            "status": "rejected",
            "reason": "optuna not installed in the task uv environment",
        })
        return 0

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_module, prepare_module = load_candidate_modules(args.candidate_path)
    search_space = train_module.SEARCH_SPACE
    make_model = train_module.make_model
    evaluate = resolve_score_fn(prepare_module, args.candidate_path)

    n_dims = len(search_space)
    if patience_override is not None:
        patience = int(patience_override)
    else:  # adaptive: cap at bo_patience_cap (20), floor at bo_patience_floor (12)
        cap = int(_rc.get("bo_patience_cap", 20))
        floor = int(_rc.get("bo_patience_floor", 12))
        patience = int(min(cap, max(floor, round(1.5 * n_dims))))

    # multivariate TPE ("tpe+") was the top optimizer in the benchmark — it models
    # parameter interactions, beating plain TPE/cmaes esp. at high dims.
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True, n_startup_trials=10)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    distributions = build_distributions(search_space)
    prior_trials = read_prior_trials(args.tune_report_json)
    for prior in prior_trials:
        prior_params = prior.get("params")
        prior_score = prior.get("score")
        if (prior_params is None or prior_score is None
                or not math.isfinite(float(prior_score))):
            continue
        prior_params = {k: v for k, v in prior_params.items() if k in distributions}
        if set(prior_params.keys()) != set(distributions.keys()):
            continue
        try:
            study.add_trial(
                optuna.trial.create_trial(
                    params=prior_params,
                    distributions=distributions,
                    value=float(prior_score),
                )
            )
        except Exception:
            continue

    # ``--n-trials`` is a durable TOTAL target, not "N more every invocation".
    # Deferred warm configs are extra target points, but only unevaluated ones are
    # enqueued after a partial invocation resumes.
    existing_stage_trials = read_stage_trials(args.tune_report_json, "bo")
    existing_keys = {
        json.dumps(t.get("params"), sort_keys=True, default=str)
        for t in existing_stage_trials if isinstance(t.get("params"), dict)
    }
    valid_deferred = []
    for raw in read_deferred_configs(args.tune_report_json):
        d_params = {k: v for k, v in raw.items() if k in distributions}
        if set(d_params.keys()) == set(distributions.keys()):
            valid_deferred.append(d_params)

    n_enqueued = 0
    for d_params in valid_deferred:
        if json.dumps(d_params, sort_keys=True, default=str) in existing_keys:
            continue
        try:
            study.enqueue_trial(d_params)
            n_enqueued += 1
        except Exception:
            continue
    requested_target_trials = n_trials + len(valid_deferred)
    eval_budget = read_global_eval_budget(args.candidate_path)
    allowed = eval_budget["remaining"]
    target_trials = requested_target_trials if allowed is None else min(
        requested_target_trials, int(allowed)
    )
    target_trials = max(len(existing_stage_trials), target_trials)
    remaining_trials = max(0, target_trials - len(existing_stage_trials))

    monitor = PatienceMonitor(
        patience=patience,
        start_best=prior_best_score(prior_trials),
    )
    early_stopped = {"flag": False, "reason": "none"}
    failure_streak = {"count": 0}

    started = time.time()
    budget = new_search_budget(args.candidate_path)

    def objective(trial):
        params = {k: suggest(trial, k, search_space[k]) for k in search_space}
        params = cast_params_to_search_space(params, search_space)
        try:
            score = timed_eval(evaluate, make_model, params, args.candidate_path)
        except Exception as exc:
            # Record the failure so it is auditable in tune_report, then re-raise
            # so Optuna (catch= below) marks this trial FAILED and moves on.
            append_trial(
                args.tune_report_json, "bo",
                {"params": params, "score": None,
                 "status": exc.status if isinstance(exc, EvaluationFailure) else "error",
                 "error": repr(exc)},
            )
            failure_streak["count"] += 1
            raise
        failure_streak["count"] = 0
        append_trial(
            args.tune_report_json, "bo", {"params": params, "score": score}
        )
        return score

    def patience_callback(study, trial):
        if failure_streak["count"] >= max_consecutive_failures:
            early_stopped["flag"] = True
            early_stopped["reason"] = "failure_circuit_breaker"
            study.stop()
            return
        if trial.value is None:
            return
        if monitor.update(float(trial.value)):
            early_stopped["flag"] = True
            early_stopped["reason"] = "patience"
            study.stop()

    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=budget.seconds,
            show_progress_bar=False,
            callbacks=[patience_callback],
            catch=(Exception,),
        )

    elapsed = time.time() - started
    stage_trials = read_stage_trials(args.tune_report_json, "bo")
    total_attempts = len(stage_trials)
    wall_stopped = (total_attempts < target_trials and not early_stopped["flag"])

    completed_trials = [
        t
        for t in study.trials
        if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        # Every trial (and any injected prior) errored: surface a failed stage
        # instead of crashing on study.best_value or mislabeling this "ok".
        set_stage_meta(args.tune_report_json, "bo", status="failed",
                       elapsed_seconds=round(elapsed, 1), early_stopped=early_stopped["flag"],
                       stop_reason=early_stopped["reason"], target_trials=target_trials,
                       trials_attempted=total_attempts)
        write_json({
            "method": "bo",
            "status": "failed",
            "reason": "all BO trials errored; no completed trial",
            "early_stopped": early_stopped["flag"],
            "early_stop_reason": early_stopped["reason"],
            "elapsed_seconds": round(elapsed, 1),
            "search_space": search_space_for_json(search_space),
        })
        return 0

    status = "partial" if wall_stopped else "ok"
    budget_capped = target_trials < requested_target_trials
    stop_reason = ("wall_clock_budget" if wall_stopped else
                   "global_eval_budget" if budget_capped else early_stopped["reason"])
    set_stage_meta(args.tune_report_json, "bo", status=status,
                   elapsed_seconds=round(elapsed, 1), early_stopped=early_stopped["flag"],
                   stop_reason=stop_reason, target_trials=target_trials,
                   trials_attempted=total_attempts,
                   wall_budget_seconds=budget.seconds)

    best_params = cast_params_to_search_space(dict(study.best_params), search_space)
    best_score = float(study.best_value)

    write_json({
        "method": "bo",
        "status": status,
        "best_params": best_params,
        "best_score": best_score,
        "trials_completed": total_attempts,
        "target_trials": target_trials,
        "requested_target_trials": requested_target_trials,
        "remaining_trials": max(0, target_trials - total_attempts),
        "global_eval_budget": eval_budget,
        "prior_trials_injected": len(prior_trials),
        "n_dims": n_dims,
        "patience": patience,
        "early_stopped": early_stopped["flag"],
        "early_stop_reason": stop_reason,
        "wall_budget_seconds": budget.seconds,
        "elapsed_seconds": round(elapsed, 1),
        "search_space": search_space_for_json(search_space),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
