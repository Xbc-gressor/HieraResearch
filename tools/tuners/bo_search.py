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
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    EvaluationBudgetExhausted,
    resolve_score_fn,
    resolve_preflight_fn,
    timed_eval,
    timed_preflight,
    load_run_cfg,
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


PREFLIGHT_CONSTRAINT_ATTR = "hiera_preflight_constraint"


class DeferredConfigError(ValueError):
    """One or more promised deferred configs could not be queued."""

    def __init__(self, rejections: list[dict]):
        self.rejections = rejections
        super().__init__(
            f"{len(rejections)} deferred config(s) could not be queued"
        )


def _params_key(params: dict) -> str:
    """Canonical identity for one fully cast BO configuration."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def _input_rejection(
    *,
    index: int,
    params,
    reason: str,
    error: BaseException | None = None,
) -> dict:
    rejection = {"index": index, "params": params, "reason": reason}
    if error is not None:
        rejection["error_type"] = type(error).__name__
        rejection["error"] = str(error)[:300]
    return rejection


def _inject_prior_trials(
    study,
    prior_trials: list[dict],
    distributions: dict,
    create_trial,
    prior_constraint_attrs: dict,
) -> tuple[int, list[dict]]:
    """Inject compatible priors and return explicit receipts for rejections."""
    injected = 0
    rejections: list[dict] = []
    expected_keys = set(distributions)
    for index, prior in enumerate(prior_trials):
        prior_params = prior.get("params")
        prior_score = prior.get("score")
        if not isinstance(prior_params, dict) or prior_score is None:
            rejections.append(
                _input_rejection(
                    index=index,
                    params=prior_params,
                    reason="missing_params_or_score",
                )
            )
            continue
        prior_params = {k: v for k, v in prior_params.items() if k in distributions}
        if set(prior_params) != expected_keys:
            rejections.append(
                _input_rejection(
                    index=index,
                    params=prior_params,
                    reason="parameter_set_mismatch",
                )
            )
            continue
        try:
            study.add_trial(
                create_trial(
                    params=prior_params,
                    distributions=distributions,
                    value=float(prior_score),
                    **prior_constraint_attrs,
                )
            )
        except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
            rejections.append(
                _input_rejection(
                    index=index,
                    params=prior_params,
                    reason="backend_rejected",
                    error=exc,
                )
            )
            continue
        injected += 1
    return injected, rejections


def _enqueue_unique_deferred(
    study,
    deferred_configs: list[dict],
    search_space: dict,
    distributions: dict,
) -> int:
    """Enqueue every novel deferred config or raise with rejection receipts."""
    seen = {_params_key(trial.params) for trial in study.trials}
    n_enqueued = 0
    rejections: list[dict] = []
    for index, raw_params in enumerate(deferred_configs):
        if not isinstance(raw_params, dict):
            rejections.append(
                _input_rejection(
                    index=index,
                    params=raw_params,
                    reason="params_must_be_object",
                )
            )
            continue
        d_params = {k: v for k, v in raw_params.items() if k in distributions}
        if set(d_params) != set(distributions):
            rejections.append(
                _input_rejection(
                    index=index,
                    params=raw_params,
                    reason="parameter_set_mismatch",
                )
            )
            continue
        try:
            d_params = cast_params_to_search_space(d_params, search_space)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            rejections.append(
                _input_rejection(
                    index=index,
                    params=raw_params,
                    reason="parameter_cast_failed",
                    error=exc,
                )
            )
            continue
        key = _params_key(d_params)
        if key in seen:
            continue
        try:
            study.enqueue_trial(d_params, skip_if_exists=True)
        except (TypeError, ValueError, RuntimeError) as exc:
            rejections.append(
                _input_rejection(
                    index=index,
                    params=d_params,
                    reason="backend_rejected",
                    error=exc,
                )
            )
            continue
        seen.add(key)
        n_enqueued += 1
    if rejections:
        raise DeferredConfigError(rejections)
    return n_enqueued


def _preflight_constraints(trial) -> tuple[float, ...]:
    """Optuna constraint vector: positive means preflight-infeasible."""
    values = trial.user_attrs.get(PREFLIGHT_CONSTRAINT_ATTR, (0.0,))
    return tuple(float(value) for value in values)


def _set_preflight_feasibility(trial, *, feasible: bool) -> None:
    trial.set_user_attr(PREFLIGHT_CONSTRAINT_ATTR, [0.0 if feasible else 1.0])


def _is_preflight_infeasible(trial) -> bool:
    return any(value > 0.0 for value in _preflight_constraints(trial))


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
    preflight_enabled = resolve_preflight_fn(prepare_module, args.candidate_path) is not None

    n_dims = len(search_space)
    if patience_override is not None:
        patience = int(patience_override)
    else:  # adaptive: cap at bo_patience_cap (20), floor at bo_patience_floor (12)
        cap = int(_rc.get("bo_patience_cap", 20))
        floor = int(_rc.get("bo_patience_floor", 12))
        patience = int(min(cap, max(floor, round(1.5 * n_dims))))

    # multivariate TPE ("tpe+") was the top optimizer in the benchmark — it models
    # parameter interactions, beating plain TPE/cmaes esp. at high dims.
    sampler_kwargs = {
        "seed": args.seed,
        "multivariate": True,
        "group": True,
        "n_startup_trials": 10,
    }
    if preflight_enabled:
        sampler_kwargs["constraints_func"] = _preflight_constraints
    sampler = optuna.samplers.TPESampler(
        **sampler_kwargs,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    distributions = build_distributions(search_space)
    prior_trials = read_prior_trials(args.tune_report_json)
    prior_constraint_attrs = {}
    if preflight_enabled:
        prior_constraint_attrs = {
            # add_trial() does not invoke constraints_func, so explicitly mark
            # already-successful warm/Phase-C priors as feasible.
            "user_attrs": {PREFLIGHT_CONSTRAINT_ATTR: [0.0]},
            "system_attrs": {"constraints": (0.0,)},
        }
    n_priors_injected, rejected_priors = _inject_prior_trials(
        study,
        prior_trials,
        distributions,
        optuna.trial.create_trial,
        prior_constraint_attrs,
    )

    # Deferred warm configs (proposed at step 0+1 but not evaluated there): enqueue
    # them as the FIRST trials so BO evaluates them before TPE. They are EXTRA points
    # on top of the TPE budget (n_trials += n_enqueued), so deep-search depth is
    # unchanged — the saving was purely the evals skipped on un-promoted candidates.
    try:
        n_enqueued = _enqueue_unique_deferred(
            study,
            read_deferred_configs(args.tune_report_json),
            search_space,
            distributions,
        )
    except DeferredConfigError as exc:
        set_stage_meta(
            args.tune_report_json,
            "bo",
            status="failed",
            rejected_priors=rejected_priors,
            deferred_rejections=exc.rejections,
        )
        write_json(
            {
                "method": "bo",
                "status": "failed",
                "reason": str(exc),
                "prior_trials_injected": n_priors_injected,
                "rejected_priors": rejected_priors,
                "deferred_rejections": exc.rejections,
            }
        )
        return 0
    set_stage_meta(
        args.tune_report_json,
        "bo",
        prior_trials_injected=n_priors_injected,
        rejected_priors=rejected_priors,
        deferred_rejections=[],
    )
    n_trials = n_trials + n_enqueued

    monitor = PatienceMonitor(
        patience=patience,
        start_best=prior_best_score(prior_trials),
    )
    early_stopped = {"flag": False, "reason": "none"}
    counters = {
        "objective_attempts": 0,
        "objective_completed": 0,
        "preflight_rejections": 0,
        "budget_exhausted": False,
    }
    failure_refs = []
    infeasible_value = max(
        (
            float(trial.value)
            for trial in study.trials
            if trial.value is not None and math.isfinite(float(trial.value))
        ),
        default=0.0,
    )

    started = time.time()

    def objective(trial):
        params = {k: suggest(trial, k, search_space[k]) for k in search_space}
        params = cast_params_to_search_space(params, search_space)
        if preflight_enabled:
            try:
                preflight_result = timed_preflight(params, args.candidate_path)
            except Exception as exc:
                _set_preflight_feasibility(trial, feasible=False)
                failure = record_failure(
                    report_path=args.tune_report_json,
                    candidate_path=args.candidate_path,
                    phase="preflight",
                    method="bo",
                    params=params,
                    error=exc,
                    traceback_text=traceback.format_exc(),
                )
                append_preflight_attempt(
                    args.tune_report_json,
                    source="bo",
                    params=params,
                    status="failed",
                    failure=failure,
                )
                append_trial(
                    args.tune_report_json,
                    "bo",
                    {
                        "params": params,
                        "score": None,
                        "status": "preflight_rejected",
                        **failure,
                    },
                )
                counters["preflight_rejections"] += 1
                # Complete this Optuna trial as constrained-infeasible instead of
                # FAILED: built-in samplers ignore failed trials, while constrained
                # TPE can use this receipt to avoid nearby infeasible proposals.
                return infeasible_value
            _set_preflight_feasibility(trial, feasible=True)
            append_preflight_attempt(
                args.tune_report_json,
                source="bo",
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
                method="bo",
            )
        except EvaluationBudgetExhausted:
            counters["budget_exhausted"] = True
            early_stopped["flag"] = True
            early_stopped["reason"] = "evaluation_budget"
            study.stop()
            raise
        except Exception as exc:
            counters["objective_attempts"] += 1
            # Record the failure so it is auditable in tune_report, then re-raise
            # so Optuna (catch= below) marks this trial FAILED and moves on.
            failure = record_failure(
                report_path=args.tune_report_json,
                candidate_path=args.candidate_path,
                phase="phase_c",
                method="bo",
                params=params,
                error=exc,
                traceback_text=traceback.format_exc(),
            )
            append_trial(
                args.tune_report_json, "bo",
                {"params": params, "score": None, "status": "failed", **failure},
            )
            if failure["failure_ref"] not in failure_refs:
                failure_refs.append(failure["failure_ref"])
            raise
        counters["objective_attempts"] += 1
        counters["objective_completed"] += 1
        append_trial(
            args.tune_report_json, "bo", {"params": params, "score": score}
        )
        return score

    def patience_callback(study, trial):
        if trial.value is None or _is_preflight_infeasible(trial):
            return
        if monitor.update(float(trial.value)):
            early_stopped["flag"] = True
            early_stopped["reason"] = "patience"
            study.stop()

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[patience_callback],
        catch=(Exception,),
    )

    elapsed = time.time() - started

    if counters["objective_completed"] == 0 and counters["budget_exhausted"]:
        set_stage_meta(
            args.tune_report_json,
            "bo",
            status="budget_exhausted",
            elapsed_seconds=round(elapsed, 1),
            early_stopped=True,
            preflight_rejections=counters["preflight_rejections"],
        )
        write_json({
            "method": "bo",
            "status": "budget_exhausted",
            "reason": "global evaluation budget exhausted before score_fn",
            "trials_completed": 0,
            "trials_attempted": counters["objective_attempts"],
            "preflight_rejections": counters["preflight_rejections"],
            "elapsed_seconds": round(elapsed, 1),
        })
        return 0

    if counters["objective_completed"] == 0:
        # Every newly attempted trial errored. Injected priors do not make this
        # search stage successful because they were evaluated before it began.
        set_stage_meta(args.tune_report_json, "bo", status="failed",
                       elapsed_seconds=round(elapsed, 1), early_stopped=early_stopped["flag"])
        write_json({
            "method": "bo",
            "status": "failed",
            "reason": "all BO trials errored; no completed trial",
            "trials_completed": 0,
            "trials_attempted": counters["objective_attempts"],
            "preflight_rejections": counters["preflight_rejections"],
            "early_stopped": early_stopped["flag"],
            "early_stop_reason": early_stopped["reason"],
            "failure_refs": failure_refs[-3:],
            "elapsed_seconds": round(elapsed, 1),
            "search_space": search_space_for_json(search_space),
        })
        return 0

    set_stage_meta(
        args.tune_report_json,
        "bo",
        status="ok",
        elapsed_seconds=round(elapsed, 1),
        early_stopped=early_stopped["flag"],
        preflight_rejections=counters["preflight_rejections"],
        budget_exhausted=counters["budget_exhausted"],
    )

    best_params = cast_params_to_search_space(dict(study.best_params), search_space)
    best_score = float(study.best_value)

    write_json({
        "method": "bo",
        "status": "ok",
        "best_params": best_params,
        "best_score": best_score,
        "trials_completed": counters["objective_completed"],
        "trials_attempted": counters["objective_attempts"],
        "preflight_rejections": counters["preflight_rejections"],
        "budget_exhausted": counters["budget_exhausted"],
        "prior_trials_injected": n_priors_injected,
        "n_dims": n_dims,
        "patience": patience,
        "early_stopped": early_stopped["flag"],
        "early_stop_reason": early_stopped["reason"],
        "elapsed_seconds": round(elapsed, 1),
        "search_space": search_space_for_json(search_space),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
