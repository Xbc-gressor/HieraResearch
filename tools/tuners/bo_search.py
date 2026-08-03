#!/usr/bin/env python3
"""Bayesian-optimization (Optuna TPE) hyperparameter search for one candidate.

Reads prior trials from tune_report.json (phase_a warm-start + any earlier
phase_c stages) and injects them into the Optuna study as completed trials
so TPE can use them as a prior. Each new trial is also appended back to
tune_report.json under phase_c.stages[bo] as it completes.

Evaluates configurations via the task's one `config → score` function
(`score_fn`). Requires `optuna` in the task's uv environment.

Invoked by the deterministic `DeepTuner` when 3 ≤ n_dims ≤ 15.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    EvaluationBudgetExhausted,
    DeepTuneTimeExhausted,
    resolve_score_fn,
    resolve_preflight_fn,
    timed_eval,
    timed_preflight,
    load_run_cfg,
    PatienceMonitor,
    append_preflight_attempt,
    append_trial,
    attempted_config_identities,
    bind_phase_c_objective_reservation,
    cancel_phase_c_objective_attempt,
    cast_params_to_search_space,
    clamp_search_space_to_preflight,
    commit_phase_c_objective_trial,
    deep_tune_stage_elapsed,
    deep_tune_time_budget,
    deep_tune_time_remaining,
    ensure_deep_tune_time_remaining,
    is_config_infeasible_error,
    load_candidate_modules,
    objective_attempt_admitted,
    params_identity,
    prepare_phase_c_objective_attempt,
    prior_patience_state,
    read_deferred_configs,
    read_prior_infeasible_trials,
    read_prior_trials,
    search_space_for_json,
    set_stage_meta,
    split_configs_by_space,
    write_json,
)
from failure_artifacts import record_failure  # noqa: E402


INFEASIBLE_ATTR = "hiera_infeasible"


class DeferredConfigError(ValueError):
    """One or more promised deferred configs could not be queued."""

    def __init__(self, rejections: list[dict]):
        self.rejections = rejections
        super().__init__(
            f"{len(rejections)} deferred config(s) could not be queued"
        )


def _params_key(params: dict) -> str:
    """Canonical identity for one fully cast BO configuration."""
    return params_identity(params)


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


def _inject_infeasible_trials(
    study,
    infeasible_trials: list[dict],
    distributions: dict,
    create_trial,
    penalty_value: float,
) -> tuple[int, list[dict]]:
    """Re-inject persisted config-infeasible trials (preflight rejections and
    config-infeasible crashes from earlier sessions) as constrained-infeasible
    points, so a restarted study does not re-propose regions it already knows
    are doomed. Each becomes a completed trial with the penalty value and the
    infeasible constraint marking. Out-of-distribution params (e.g. outside a
    clamped space) are skipped with a rejection receipt.
    """
    expected_keys = set(distributions)
    injected = 0
    rejections: list[dict] = []
    for index, trial in enumerate(infeasible_trials):
        raw_params = trial.get("params")
        params = {k: v for k, v in raw_params.items() if k in distributions} if isinstance(raw_params, dict) else {}
        if set(params) != expected_keys:
            rejections.append(
                _input_rejection(
                    index=index,
                    params=raw_params,
                    reason="parameter_set_mismatch",
                )
            )
            continue
        try:
            study.add_trial(
                create_trial(
                    params=params,
                    distributions=distributions,
                    value=penalty_value,
                    user_attrs={INFEASIBLE_ATTR: [1.0]},
                    system_attrs={"constraints": (1.0,)},
                )
            )
        except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
            rejections.append(
                _input_rejection(
                    index=index,
                    params=params,
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
    *,
    blocked_identities: set[str] | None = None,
) -> int:
    """Enqueue every novel deferred config or raise with rejection receipts."""
    seen = {_params_key(trial.params) for trial in study.trials}
    seen.update(blocked_identities or ())
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


def _infeasible_constraints(trial) -> tuple[float, ...]:
    """Optuna constraint vector: positive means infeasible (preflight-rejected
    or crashed during score_fn)."""
    values = trial.user_attrs.get(INFEASIBLE_ATTR, (0.0,))
    return tuple(float(value) for value in values)


def _set_feasibility(trial, *, feasible: bool) -> None:
    trial.set_user_attr(INFEASIBLE_ATTR, [0.0 if feasible else 1.0])


def _is_infeasible(trial) -> bool:
    return any(value > 0.0 for value in _infeasible_constraints(trial))


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
    time_budget = deep_tune_time_budget(
        args.candidate_path,
        args.tune_report_json,
        "bo",
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
            "bo",
            status="time_exhausted",
            elapsed_seconds=elapsed_seconds,
            early_stopped=True,
            early_stop_reason="time_budget",
            preflight_rejections=preflight_rejections,
            time_limit_seconds=time_budget["limit_seconds"],
        )
        write_json({
            "method": "bo",
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
        ensure_deep_tune_time_remaining(time_budget)
        import optuna
        ensure_deep_tune_time_remaining(time_budget)
    except DeepTuneTimeExhausted:
        return close_time_exhausted()
    except ImportError:
        set_stage_meta(
            args.tune_report_json,
            "bo",
            status="rejected",
            elapsed_seconds=deep_tune_stage_elapsed(time_budget),
        )
        write_json({
            "method": "bo",
            "status": "rejected",
            "reason": "optuna not installed in the task uv environment",
        })
        return 0

    optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        # Anything residual that still fails is fed to TPE as a constraint.
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
        # Always constrained: crashed trials (not only preflight rejections)
        # are marked infeasible so TPE steers away from crashing regions.
        "constraints_func": _infeasible_constraints,
    }
    sampler = optuna.samplers.TPESampler(
        **sampler_kwargs,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    distributions = build_distributions(search_space)
    prior_trials = read_prior_trials(args.tune_report_json)
    # add_trial() does not invoke constraints_func, so explicitly mark
    # already-successful warm/Phase-C priors as feasible.
    prior_constraint_attrs = {
        "user_attrs": {INFEASIBLE_ATTR: [0.0]},
        "system_attrs": {"constraints": (0.0,)},
    }
    n_priors_injected, rejected_priors = _inject_prior_trials(
        study,
        prior_trials,
        distributions,
        optuna.trial.create_trial,
        prior_constraint_attrs,
    )

    # Re-inject persisted config-infeasible trials before deferred configs are
    # queued. `_enqueue_unique_deferred` de-duplicates against study.trials, so
    # this order prevents a deferred config that already crashed in an earlier
    # session from becoming a WAITING trial and being retried on restart.
    infeasible_value = max(
        (
            float(trial.value)
            for trial in study.trials
            if trial.value is not None and math.isfinite(float(trial.value))
        ),
        default=0.0,
    )
    prior_infeasible_trials = read_prior_infeasible_trials(
        args.tune_report_json
    )
    n_infeasible_injected, infeasible_rejections = _inject_infeasible_trials(
        study,
        prior_infeasible_trials,
        distributions,
        optuna.trial.create_trial,
        infeasible_value,
    )

    # Deferred warm configs (proposed at step 0+1 but not evaluated there): enqueue
    # them as the FIRST trials so BO evaluates them before TPE. They are EXTRA points
    # on top of the TPE budget (n_trials += n_enqueued), so deep-search depth is
    # unchanged — the saving was purely the evals skipped on un-promoted candidates.
    # Deferred warm configs outside the (possibly clamped) box are skipped —
    # never attempted, no budget, no patience effect; the clamp marked that
    # region infeasible. The skip is accounted via deferred_skipped_outside_space.
    deferred_in_space, deferred_outside = split_configs_by_space(
        read_deferred_configs(args.tune_report_json), search_space
    )
    attempted_identities = attempted_config_identities(
        args.tune_report_json,
        search_space,
    )
    try:
        n_enqueued = _enqueue_unique_deferred(
            study,
            deferred_in_space,
            search_space,
            distributions,
            blocked_identities=attempted_identities,
        )
    except DeferredConfigError as exc:
        set_stage_meta(
            args.tune_report_json,
            "bo",
            status="failed",
            rejected_priors=rejected_priors,
            infeasible_priors_injected=n_infeasible_injected,
            infeasible_prior_rejections=len(infeasible_rejections),
            deferred_rejections=exc.rejections,
        )
        write_json(
            {
                "method": "bo",
                "status": "failed",
                "reason": str(exc),
                "prior_trials_injected": n_priors_injected,
                "rejected_priors": rejected_priors,
                "infeasible_priors_injected": n_infeasible_injected,
                "infeasible_prior_rejections": infeasible_rejections,
                "deferred_rejections": exc.rejections,
            }
        )
        return 0
    set_stage_meta(
        args.tune_report_json,
        "bo",
        status="running",
        prior_trials_injected=n_priors_injected,
        rejected_priors=rejected_priors,
        infeasible_priors_injected=n_infeasible_injected,
        infeasible_prior_rejections=len(infeasible_rejections),
        deferred_rejections=[],
        deferred_skipped_outside_space=len(deferred_outside),
    )
    n_trials = n_trials + n_enqueued

    # Seed best AND streak from the persisted trial history: a restarted study
    # continues the patience window instead of getting a fresh one.
    prior_best, prior_streak = prior_patience_state(args.tune_report_json)
    monitor = PatienceMonitor(
        patience=patience,
        start_best=prior_best,
        start_since=prior_streak,
    )
    early_stopped = {"flag": False, "reason": "none"}
    counters = {
        "objective_attempts": 0,
        "objective_completed": 0,
        "preflight_rejections": 0,
        "duplicates_skipped": 0,
        "budget_exhausted": False,
        "budget_exhausted_scope": None,
        "time_exhausted": False,
    }
    failure_refs = []
    known_scores: dict[str, float] = {}
    known_score_params: dict[str, dict] = {}
    for prior in prior_trials:
        if set(prior.get("params", {})) != set(search_space):
            continue
        try:
            normalized = cast_params_to_search_space(
                dict(prior["params"]),
                search_space,
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        identity = params_identity(normalized)
        score = float(prior["score"])
        if score < known_scores.get(identity, math.inf):
            known_scores[identity] = score
            known_score_params[identity] = normalized
    known_infeasible: set[str] = set()
    for prior in prior_infeasible_trials:
        if set(prior.get("params", {})) != set(search_space):
            continue
        try:
            normalized = cast_params_to_search_space(
                dict(prior["params"]),
                search_space,
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        known_infeasible.add(params_identity(normalized))
    known_infeasible.difference_update(known_scores)
    known_unscored = (
        attempted_identities - set(known_scores) - known_infeasible
    )

    def admit_timed_work() -> None:
        try:
            ensure_deep_tune_time_remaining(time_budget)
        except DeepTuneTimeExhausted:
            counters["time_exhausted"] = True
            early_stopped["flag"] = True
            early_stopped["reason"] = "time_budget"
            study.stop()
            raise

    def objective(trial):
        params = {k: suggest(trial, k, search_space[k]) for k in search_space}
        params = cast_params_to_search_space(params, search_space)
        admit_timed_work()
        identity = params_identity(params)
        if identity in known_scores:
            counters["duplicates_skipped"] += 1
            _set_feasibility(trial, feasible=True)
            return known_scores[identity]
        if identity in known_infeasible:
            counters["duplicates_skipped"] += 1
            _set_feasibility(trial, feasible=False)
            return infeasible_value
        if identity in known_unscored:
            counters["duplicates_skipped"] += 1
            raise RuntimeError(
                "optimizer reproposed a previously consumed scoreless config"
            )
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
                counters["time_exhausted"] = True
                early_stopped["flag"] = True
                early_stopped["reason"] = "time_budget"
                study.stop()
                raise
            except Exception as exc:
                _set_feasibility(trial, feasible=False)
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
                known_infeasible.add(identity)
                admit_timed_work()
                # Complete this Optuna trial as constrained-infeasible instead of
                # FAILED: built-in samplers ignore failed trials, while constrained
                # TPE can use this receipt to avoid nearby infeasible proposals.
                return infeasible_value
            _set_feasibility(trial, feasible=True)
            append_preflight_attempt(
                args.tune_report_json,
                source="bo",
                params=params,
                status="ok",
                result=preflight_result or {"status": "ok"},
            )
            admit_timed_work()
        objective_intent = None
        try:
            admit_timed_work()
            objective_intent = prepare_phase_c_objective_attempt(
                args.tune_report_json,
                args.candidate_path,
                "bo",
                params,
                time_budget["candidate_execution_revision"],
            )
            score = timed_eval(
                evaluate,
                make_model,
                params,
                args.candidate_path,
                phase="phase_c",
                method="bo",
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
                        "bo",
                        objective_intent,
                        receipt,
                    )
                ),
            )
        except DeepTuneTimeExhausted as exc:
            if exc.attempt_reserved:
                counters["objective_attempts"] += 1
                failure = record_failure(
                    report_path=args.tune_report_json,
                    candidate_path=args.candidate_path,
                    phase="phase_c",
                    method="bo",
                    params=params,
                    error=exc,
                    traceback_text=traceback.format_exc(),
                )
                commit_phase_c_objective_trial(
                    args.tune_report_json,
                    args.candidate_path,
                    "bo",
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
                    "bo",
                    objective_intent,
                )
            counters["time_exhausted"] = True
            early_stopped["flag"] = True
            early_stopped["reason"] = "time_budget"
            study.stop()
            raise
        except EvaluationBudgetExhausted as exc:
            if objective_intent is not None:
                cancel_phase_c_objective_attempt(
                    args.tune_report_json,
                    args.candidate_path,
                    "bo",
                    objective_intent,
                )
            counters["budget_exhausted"] = True
            counters["budget_exhausted_scope"] = exc.scope
            early_stopped["flag"] = True
            early_stopped["reason"] = "evaluation_budget"
            study.stop()
            raise
        except Exception as exc:
            if not objective_attempt_admitted(exc):
                if objective_intent is not None:
                    cancel_phase_c_objective_attempt(
                        args.tune_report_json,
                        args.candidate_path,
                        "bo",
                        objective_intent,
                    )
                raise
            counters["objective_attempts"] += 1
            failure = record_failure(
                report_path=args.tune_report_json,
                candidate_path=args.candidate_path,
                phase="phase_c",
                method="bo",
                params=params,
                error=exc,
                traceback_text=traceback.format_exc(),
            )
            # Only clearly config-specific failures (OOM, timeout) feed the
            # sampler's constraint model. Anything else (infra flake, candidate
            # bug, malformed result) is not a parameter-region signal: it stays
            # an ordinary FAILED trial rather than teaching a false boundary.
            config_infeasible = is_config_infeasible_error(exc)
            commit_phase_c_objective_trial(
                args.tune_report_json,
                args.candidate_path,
                "bo",
                objective_intent,
                {
                    "params": params,
                    "score": None,
                    "status": "failed",
                    "config_infeasible": config_infeasible,
                    **failure,
                },
            )
            if failure["failure_ref"] not in failure_refs:
                failure_refs.append(failure["failure_ref"])
            if not config_infeasible:
                known_unscored.add(identity)
                raise
            # Complete this Optuna trial as constrained-infeasible instead of
            # FAILED: built-in samplers ignore failed trials, while constrained
            # TPE can use this receipt to avoid nearby infeasible proposals.
            _set_feasibility(trial, feasible=False)
            known_infeasible.add(identity)
            known_unscored.discard(identity)
            return infeasible_value
        counters["objective_attempts"] += 1
        counters["objective_completed"] += 1
        commit_phase_c_objective_trial(
            args.tune_report_json,
            args.candidate_path,
            "bo",
            objective_intent,
            {"params": params, "score": score},
        )
        known_scores[identity] = float(score)
        known_score_params[identity] = params
        known_infeasible.discard(identity)
        known_unscored.discard(identity)
        return score

    def patience_callback(study, trial):
        # Failed (value None) and infeasible trials (preflight-rejected or
        # crashed) are non-improvements: they must count toward patience, or a
        # failure streak dilutes the counter and early stopping never fires.
        if trial.value is None or _is_infeasible(trial):
            stop = monitor.update_failed()
        else:
            stop = monitor.update(float(trial.value))
        if stop and not early_stopped["flag"]:
            early_stopped["flag"] = True
            early_stopped["reason"] = "patience"
            study.stop()

    try:
        ensure_deep_tune_time_remaining(time_budget)
    except DeepTuneTimeExhausted:
        return close_time_exhausted(
            trials_completed=counters["objective_completed"],
            trials_attempted=counters["objective_attempts"],
            preflight_rejections=counters["preflight_rejections"],
        )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=deep_tune_time_remaining(time_budget),
        show_progress_bar=False,
        callbacks=[patience_callback],
        catch=(Exception,),
    )

    stage_elapsed = deep_tune_stage_elapsed(time_budget)
    hit_time_limit = (
        (
            counters["time_exhausted"]
            or deep_tune_time_remaining(time_budget) <= 0
        )
        and not counters["budget_exhausted"]
    )
    if hit_time_limit:
        early_stopped["flag"] = True
        early_stopped["reason"] = "time_budget"

    # TPE startup accounting. Optuna's sampler silently falls back to random
    # draws until the study holds n_startup_trials COMPLETE/PRUNED trials, so a
    # stage can report method "bo" while never engaging TPE (run 0730-ds-ex100-1:
    # every stage ended below startup). Reconstruct the split from the trial
    # sequence so the receipt shows it. Order in study.trials: injected score
    # priors, injected infeasible priors, enqueued deferred, then sampler draws;
    # injected trials are all COMPLETE (feasible with scores, infeasible with
    # the penalty value) and count toward startup. Exact for one invocation;
    # across a crash resume it describes the final invocation only.
    n_startup = sampler_kwargs["n_startup_trials"]
    completes = n_priors_injected + n_infeasible_injected
    model_driven_trials = 0
    random_fallback_trials = 0
    for index, run_trial in enumerate(study.trials[completes:]):
        if index >= n_enqueued:  # enqueued deferred are not sampler draws
            if completes >= n_startup:
                model_driven_trials += 1
            else:
                random_fallback_trials += 1
        # COMPLETE/PRUNED count toward TPE startup. Here that is exactly the
        # finite-value trials: no pruner is configured, crashes leave value
        # None, and infeasible completions carry the finite penalty value.
        if run_trial.value is not None and math.isfinite(float(run_trial.value)):
            completes += 1

    if counters["objective_completed"] == 0 and counters["budget_exhausted"]:
        set_stage_meta(
            args.tune_report_json,
            "bo",
            status="budget_exhausted",
            elapsed_seconds=stage_elapsed,
            early_stopped=True,
            preflight_rejections=counters["preflight_rejections"],
            duplicates_skipped=counters["duplicates_skipped"],
            budget_exhausted_scope=counters["budget_exhausted_scope"],
        )
        write_json({
            "method": "bo",
            "status": "budget_exhausted",
            "reason": "evaluation allocation exhausted before score_fn",
            "budget_exhausted_scope": counters["budget_exhausted_scope"],
            "trials_completed": 0,
            "trials_attempted": counters["objective_attempts"],
            "preflight_rejections": counters["preflight_rejections"],
            "duplicates_skipped": counters["duplicates_skipped"],
            "elapsed_seconds": round(stage_elapsed, 1),
        })
        return 0

    if counters["objective_completed"] == 0 and hit_time_limit:
        return close_time_exhausted(
            trials_completed=0,
            trials_attempted=counters["objective_attempts"],
            preflight_rejections=counters["preflight_rejections"],
        )

    if (
        counters["objective_completed"] == 0
        and counters["objective_attempts"] == 0
        and counters["preflight_rejections"] == 0
        and counters["duplicates_skipped"] > 0
        and known_scores
    ):
        best_identity = min(known_scores, key=known_scores.get)
        set_stage_meta(
            args.tune_report_json,
            "bo",
            status="failed",
            elapsed_seconds=stage_elapsed,
            early_stopped=True,
            early_stop_reason="proposal_space_exhausted",
            duplicates_skipped=counters["duplicates_skipped"],
            trials_completed=0,
            trials_attempted=0,
        )
        write_json({
            "method": "bo",
            "status": "failed",
            "reason": (
                "optimizer proposed only previously attempted configurations; "
                "no new objective trial was admitted"
            ),
            "best_params": known_score_params[best_identity],
            "best_score": known_scores[best_identity],
            "trials_completed": 0,
            "trials_attempted": 0,
            "preflight_rejections": 0,
            "duplicates_skipped": counters["duplicates_skipped"],
            "early_stopped": True,
            "early_stop_reason": "proposal_space_exhausted",
            "elapsed_seconds": round(stage_elapsed, 1),
            "search_space": search_space_for_json(search_space),
        })
        return 0

    if counters["objective_completed"] == 0:
        # Every newly attempted trial errored. Injected priors do not make this
        # search stage successful because they were evaluated before it began.
        set_stage_meta(
            args.tune_report_json,
            "bo",
            status="failed",
            elapsed_seconds=stage_elapsed,
            early_stopped=early_stopped["flag"],
            duplicates_skipped=counters["duplicates_skipped"],
        )
        write_json({
            "method": "bo",
            "status": "failed",
            "reason": "all BO trials errored; no completed trial",
            "trials_completed": 0,
            "trials_attempted": counters["objective_attempts"],
            "preflight_rejections": counters["preflight_rejections"],
            "duplicates_skipped": counters["duplicates_skipped"],
            "early_stopped": early_stopped["flag"],
            "early_stop_reason": early_stopped["reason"],
            "failure_refs": failure_refs[-3:],
            "elapsed_seconds": round(stage_elapsed, 1),
            "search_space": search_space_for_json(search_space),
        })
        return 0

    set_stage_meta(
        args.tune_report_json,
        "bo",
        status="ok",
        elapsed_seconds=stage_elapsed,
        early_stopped=early_stopped["flag"],
        preflight_rejections=counters["preflight_rejections"],
        duplicates_skipped=counters["duplicates_skipped"],
        budget_exhausted=counters["budget_exhausted"],
        time_limit_seconds=time_budget["limit_seconds"],
        n_startup_trials=n_startup,
        model_driven_trials=model_driven_trials,
        random_fallback_trials=random_fallback_trials,
    )

    # Rank feasible trials only: infeasible ones (preflight-rejected or
    # crashed) carry the penalty `infeasible_value`, not a real score — with no
    # priors that penalty is 0.0, which would look optimal to a minimize study.
    feasible_trials = [
        t
        for t in study.trials
        if t.value is not None
        and math.isfinite(float(t.value))
        and not _is_infeasible(t)
    ]
    best_trial = min(feasible_trials, key=lambda t: float(t.value))
    best_params = cast_params_to_search_space(dict(best_trial.params), search_space)
    best_score = float(best_trial.value)

    write_json({
        "method": "bo",
        "status": "ok",
        "best_params": best_params,
        "best_score": best_score,
        "trials_completed": counters["objective_completed"],
        "trials_attempted": counters["objective_attempts"],
        "preflight_rejections": counters["preflight_rejections"],
        "duplicates_skipped": counters["duplicates_skipped"],
        "budget_exhausted": counters["budget_exhausted"],
        "prior_trials_injected": n_priors_injected,
        "infeasible_priors_injected": n_infeasible_injected,
        "deferred_skipped_outside_space": len(deferred_outside),
        "n_dims": n_dims,
        "patience": patience,
        "n_startup_trials": n_startup,
        "model_driven_trials": model_driven_trials,
        "random_fallback_trials": random_fallback_trials,
        "early_stopped": early_stopped["flag"],
        "early_stop_reason": early_stopped["reason"],
        "elapsed_seconds": round(stage_elapsed, 1),
        "time_limit_seconds": time_budget["limit_seconds"],
        "search_space": search_space_for_json(search_space),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
