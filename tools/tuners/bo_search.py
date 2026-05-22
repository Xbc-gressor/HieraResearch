#!/usr/bin/env python3
"""Bayesian-optimization (Optuna TPE) hyperparameter search for one candidate.

Reads prior trials from tune_report.json (phase_a warm-start + any earlier
phase_c stages) and injects them into the Optuna study as completed trials
so TPE can use them as a prior. Each new trial is also appended back to
tune_report.json under phase_c.stages[bo] as it completes.

Evaluates configurations via prepare.evaluate_config_for_tuning (which
internally uses the no-lock test_score_for_tuning helper).
Requires `optuna` in the task's uv environment.

Invoked by the tuner-orchestrator agent when 3 ≤ n_dims ≤ 15.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--tune-report-json", required=True, type=Path)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
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
    evaluate = prepare_module.evaluate_config_for_tuning

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    direction = "minimize" if args.lower_is_better else "maximize"
    study = optuna.create_study(direction=direction, sampler=sampler)

    distributions = build_distributions(search_space)
    prior_trials = read_prior_trials(args.tune_report_json)
    for prior in prior_trials:
        prior_params = prior.get("params")
        prior_score = prior.get("score")
        if prior_params is None or prior_score is None:
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

    monitor = PatienceMonitor(
        patience=args.patience,
        lower_is_better=args.lower_is_better,
        start_best=prior_best_score(prior_trials, args.lower_is_better),
    )
    early_stopped = {"flag": False, "reason": "none"}

    started = time.time()

    def objective(trial):
        params = {k: suggest(trial, k, search_space[k]) for k in search_space}
        params = cast_params_to_search_space(params, search_space)
        score = evaluate(make_model, params)
        append_trial(
            args.tune_report_json, "bo", {"params": params, "score": score}
        )
        return score

    def patience_callback(study, trial):
        if trial.value is None:
            return
        if monitor.update(float(trial.value)):
            early_stopped["flag"] = True
            early_stopped["reason"] = "patience"
            study.stop()

    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=False,
        callbacks=[patience_callback],
    )

    elapsed = time.time() - started

    completed_trials = [
        t
        for t in study.trials
        if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE
    ]
    best_params = cast_params_to_search_space(dict(study.best_params), search_space)
    best_score = float(study.best_value)

    write_json({
        "method": "bo",
        "status": "ok",
        "best_params": best_params,
        "best_score": best_score,
        "trials_completed": len(completed_trials) - len(prior_trials),
        "prior_trials_injected": len(prior_trials),
        "early_stopped": early_stopped["flag"],
        "early_stop_reason": early_stopped["reason"],
        "elapsed_seconds": round(elapsed, 1),
        "search_space": search_space_for_json(search_space),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
