#!/usr/bin/env python3
"""CMA-ES hyperparameter search for one autoresearch candidate.

Encodes the candidate's SEARCH_SPACE into a continuous bounded vector
(int → rounded continuous, categorical → integer index, log-float → log
space), runs cma.CMAEvolutionStrategy with the configured budget.

Reads prior trials from tune_report.json; picks the best prior in the task's
metric direction as the CMA-ES initial mean (x0). Other priors are not directly
used by CMA-ES (it is a population method, not surrogate-based), but they
remain accessible in tune_report.json for downstream consumers.

Each evaluated point is appended to tune_report.json under
phase_c.stages[cmaes] as it completes.

Evaluates configurations via the task's one `config → score` function
(`score_fn`). Requires `cma` in the task's uv environment.

Invoked by the tuner-orchestrator as the deterministic fallback when BO rejects.
"""

from __future__ import annotations

import argparse
import math
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
    cast_params_to_search_space,
    clamp_search_space_to_preflight,
    deduplicate_configs,
    deep_tune_stage_elapsed,
    deep_tune_time_budget,
    deep_tune_time_remaining,
    ensure_deep_tune_time_remaining,
    is_config_infeasible_error,
    is_finite_score,
    load_candidate_modules,
    params_identity,
    params_within_search_space,
    prior_patience_state,
    read_deferred_configs,
    read_pending_proposals,
    read_prior_infeasible_trials,
    read_prior_trials,
    read_tune_report,
    search_space_for_json,
    set_stage_meta,
    split_configs_by_space,
    write_json,
)
from failure_artifacts import record_failure  # noqa: E402


def build_codec(search_space: dict, base_params: dict):
    """Build a CMA-ES vector over non-fixed coordinates.

    A clamp may collapse a numeric range to one value. Fixed numeric and
    single-choice categorical coordinates are omitted from CMA's vector (which
    requires every lower bound to be strictly smaller than its upper bound)
    and merged back into every decoded parameter dict.
    """
    keys = []
    lower = []
    upper = []
    x0 = []
    decoders = []
    fixed_params = {}

    for key, entry in search_space.items():
        kind = entry[0]
        base_value = base_params.get(key)

        if kind == "float":
            low, high = float(entry[1]), float(entry[2])
            if low == high:
                fixed_params[key] = low
                continue
            keys.append(key)
            if len(entry) >= 4 and entry[3] == "log":
                lo, hi = math.log10(low), math.log10(high)
                lower.append(lo)
                upper.append(hi)
                x0.append(math.log10(base_value) if base_value else (lo + hi) / 2)
                decoders.append(("float_log", key))
            else:
                lower.append(low)
                upper.append(high)
                x0.append(
                    float(base_value) if base_value is not None else (low + high) / 2
                )
                decoders.append(("float", key))
        elif kind == "int":
            low, high = int(entry[1]), int(entry[2])
            if low == high:
                fixed_params[key] = low
                continue
            keys.append(key)
            lower.append(low - 0.5)
            upper.append(high + 0.5)
            x0.append(
                float(base_value) if base_value is not None else (low + high) / 2
            )
            decoders.append(("int", key, low, high))
        elif kind == "categorical":
            choices = list(entry[1])
            n = len(choices)
            if n == 1:
                fixed_params[key] = choices[0]
                continue
            keys.append(key)
            lower.append(-0.5)
            upper.append(n - 0.5)
            if base_value in choices:
                x0.append(float(choices.index(base_value)))
            else:
                x0.append((n - 1) / 2)
            decoders.append(("categorical", key, choices))
        else:
            raise ValueError(f"unknown SEARCH_SPACE entry: {entry}")

    def encode(params: dict) -> np.ndarray:
        vec = []
        for dec in decoders:
            kind = dec[0]
            if kind == "float":
                vec.append(float(params[dec[1]]))
            elif kind == "float_log":
                vec.append(math.log10(float(params[dec[1]])))
            elif kind == "int":
                vec.append(float(params[dec[1]]))
            elif kind == "categorical":
                _, key, choices = dec
                vec.append(float(choices.index(params[key])))
        return np.asarray(vec, dtype=float)

    def decode(x: np.ndarray) -> dict:
        out = dict(fixed_params)
        for value, dec in zip(x, decoders):
            kind = dec[0]
            if kind == "float":
                out[dec[1]] = float(value)
            elif kind == "float_log":
                out[dec[1]] = float(10 ** value)
            elif kind == "int":
                _, key, low, high = dec
                clipped = max(low, min(high, int(round(float(value)))))
                out[key] = clipped
            elif kind == "categorical":
                _, key, choices = dec
                idx = max(0, min(len(choices) - 1, int(round(float(value)))))
                out[key] = choices[idx]
        return out

    return keys, lower, upper, x0, encode, decode


def _encode_prior_seed(
    best_prior: dict | None,
    *,
    encode,
    lower: list,
    upper: list,
    x0_default: list,
) -> tuple[list, dict | None]:
    """Encode a CMA prior or return the default with an explicit rejection."""
    if best_prior is None:
        return list(x0_default), None
    params = best_prior.get("params")
    try:
        if not isinstance(params, dict):
            raise TypeError("prior params must be an object")
        encoded = encode(params)
        if len(encoded) != len(lower):
            raise ValueError("encoded prior dimension does not match search space")
        x0 = []
        for low, high, raw in zip(lower, upper, encoded):
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("encoded prior contains a non-finite value")
            x0.append(max(low, min(high, value)))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return list(x0_default), {
            "params": params,
            "reason": "seed_encoding_failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
        }
    return x0, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--tune-report-json", required=True, type=Path)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--max-evals", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--sigma0", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    time_budget = deep_tune_time_budget(
        args.candidate_path,
        args.tune_report_json,
        "cmaes",
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
            "cmaes",
            bout_index=time_budget["bout_index"],
            status="time_exhausted",
            elapsed_seconds=elapsed_seconds,
            early_stopped=True,
            early_stop_reason="time_budget",
            preflight_rejections=preflight_rejections,
            time_limit_seconds=time_budget["limit_seconds"],
        )
        write_json({
            "method": "cmaes",
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
        import cma
        ensure_deep_tune_time_remaining(time_budget)
    except DeepTuneTimeExhausted:
        return close_time_exhausted()
    except ImportError:
        set_stage_meta(
            args.tune_report_json,
            "cmaes",
            bout_index=time_budget["bout_index"],
            status="rejected",
            elapsed_seconds=deep_tune_stage_elapsed(time_budget),
        )
        write_json({
            "method": "cmaes",
            "status": "rejected",
            "reason": "cma not installed in the task uv environment",
        })
        return 0

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
    base_params = dict(train_module.BASE_PARAMS)
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
                base_params,
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
    prior_trials = read_prior_trials(args.tune_report_json)
    best_prior = None
    if prior_trials:
        best_prior = min(prior_trials, key=lambda t: t.get("score", math.inf))
    seed_params = best_prior["params"] if best_prior else base_params

    keys, lower, upper, x0_default, encode, decode = build_codec(
        search_space, base_params
    )
    x0, rejected_prior = _encode_prior_seed(
        best_prior,
        encode=encode,
        lower=lower,
        upper=upper,
        x0_default=x0_default,
    )
    rejected_priors = [rejected_prior] if rejected_prior is not None else []
    if rejected_prior is not None:
        seed_params = base_params

    deferred_in_space, deferred_outside = split_configs_by_space(
        read_deferred_configs(args.tune_report_json), search_space
    )
    attempted_identities = attempted_config_identities(
        args.tune_report_json,
        search_space,
    )
    known_scores: dict[str, float] = {}
    known_score_params: dict[str, dict] = {}
    for prior in prior_trials:
        normalized = cast_params_to_search_space(
            dict(prior["params"]),
            search_space,
        )
        if set(normalized) != set(search_space):
            continue
        identity = params_identity(normalized)
        score = float(prior["score"])
        if score < known_scores.get(identity, math.inf):
            known_scores[identity] = score
            known_score_params[identity] = normalized
    known_infeasible: set[str] = set()
    for prior in read_prior_infeasible_trials(args.tune_report_json):
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
    # Validated LLM re-warm proposals (admitted by validate-proposals for a
    # continuation bout) are attempted ahead of the deferred warm configs; the
    # chained `seen` set keeps a proposal equal to a deferred config from
    # being attempted twice.
    proposals_in_space, proposals_outside = split_configs_by_space(
        read_pending_proposals(args.tune_report_json), search_space
    )
    proposals_in_space = [
        cast_params_to_search_space(dict(params), search_space)
        for params in proposals_in_space
    ]
    proposals_in_space, proposals_skipped_seen, seen = deduplicate_configs(
        proposals_in_space,
        seen=attempted_identities,
    )
    deferred_in_space = [
        cast_params_to_search_space(dict(params), search_space)
        for params in deferred_in_space
    ]
    deferred_in_space, deferred_skipped_seen, _ = deduplicate_configs(
        deferred_in_space,
        seen=seen,
    )

    # A clamp can collapse every coordinate to one value. CMA-ES cannot
    # construct a zero-dimensional strategy, and there is no search left to
    # perform. Phase A is required to have scored the incumbent, so close this
    # stage deterministically from the best compatible prior. If that invariant
    # is broken, fail closed instead of fabricating a successful CMA trial.
    if not keys:
        try:
            ensure_deep_tune_time_remaining(time_budget)
        except DeepTuneTimeExhausted:
            return close_time_exhausted()
        terminal_elapsed = deep_tune_stage_elapsed(time_budget)
        phase_a_trials = [
            trial
            for trial in read_tune_report(args.tune_report_json)
            .get("phase_a", {})
            .get("warm_start_configs", [])
            if isinstance(trial, dict)
            and isinstance(trial.get("params"), dict)
            and is_finite_score(trial.get("score"))
        ]
        compatible_priors = [
            trial
            for trial in phase_a_trials
            if params_within_search_space(trial.get("params", {}), search_space)
        ]
        if compatible_priors:
            fixed_best = min(
                compatible_priors, key=lambda trial: float(trial["score"])
            )
            set_stage_meta(
                args.tune_report_json,
                "cmaes",
                bout_index=time_budget["bout_index"],
                status="no_search_needed",
                elapsed_seconds=terminal_elapsed,
                early_stopped=True,
                early_stop_reason="fixed_search_space",
                fixed_search_space=True,
                fixed_incumbent_params=cast_params_to_search_space(
                    dict(fixed_best["params"]), search_space
                ),
                fixed_incumbent_score=float(fixed_best["score"]),
                effective_search_space=search_space_for_json(search_space),
                trials_completed=0,
                trials_attempted=0,
                prior_trials_seen=len(prior_trials),
                rejected_priors=rejected_priors,
                deferred_skipped_outside_space=len(deferred_outside),
                deferred_skipped_already_seen=deferred_skipped_seen,
            )
            write_json(
                {
                    "method": "cmaes",
                    "status": "no_search_needed",
                    "best_params": cast_params_to_search_space(
                        dict(fixed_best["params"]), search_space
                    ),
                    "best_score": float(fixed_best["score"]),
                    "trials_completed": 0,
                    "trials_attempted": 0,
                    "preflight_rejections": 0,
                    "budget_exhausted": False,
                    "deferred_skipped_outside_space": len(deferred_outside),
                    "deferred_skipped_already_seen": deferred_skipped_seen,
                    "prior_trials_seen": len(prior_trials),
                    "x0_from_prior": True,
                    "popsize": args.popsize,
                    "fixed_search_space": True,
                    "early_stopped": True,
                    "early_stop_reason": "fixed_search_space",
                    "elapsed_seconds": round(terminal_elapsed, 1),
                    "search_space": search_space_for_json(search_space),
                }
            )
            return 0

        reason = (
            "search space has no tunable dimensions and no compatible scored prior"
        )
        set_stage_meta(
            args.tune_report_json,
            "cmaes",
            bout_index=time_budget["bout_index"],
            status="failed",
            elapsed_seconds=terminal_elapsed,
            early_stopped=True,
            early_stop_reason="fixed_search_space_without_prior",
            fixed_search_space=True,
            prior_trials_seen=len(prior_trials),
            rejected_priors=rejected_priors,
            deferred_skipped_outside_space=len(deferred_outside),
            deferred_skipped_already_seen=deferred_skipped_seen,
        )
        write_json(
            {
                "method": "cmaes",
                "status": "failed",
                "reason": reason,
                "trials_completed": 0,
                "trials_attempted": 0,
                "preflight_rejections": 0,
                "prior_trials_seen": len(prior_trials),
                "fixed_search_space": True,
                "early_stopped": True,
                "early_stop_reason": "fixed_search_space_without_prior",
                "elapsed_seconds": round(terminal_elapsed, 1),
                "search_space": search_space_for_json(search_space),
            }
        )
        return 0

    set_stage_meta(
        args.tune_report_json,
        "cmaes",
        bout_index=time_budget["bout_index"],
        status="running",
        prior_trials_seen=len(prior_trials),
        rejected_priors=rejected_priors,
        deferred_skipped_already_seen=deferred_skipped_seen,
        rewarm_proposals_enqueued=len(proposals_in_space),
        rewarm_skipped_outside_space=len(proposals_outside),
        rewarm_skipped_already_seen=proposals_skipped_seen,
    )

    try:
        ensure_deep_tune_time_remaining(time_budget)
    except DeepTuneTimeExhausted:
        return close_time_exhausted()
    es = cma.CMAEvolutionStrategy(
        x0,
        args.sigma0,
        {
            "bounds": [lower, upper],
            "popsize": args.popsize,
            "seed": args.seed,
            "verbose": -9,
            "verb_disp": 0,
        },
    )
    try:
        ensure_deep_tune_time_remaining(time_budget)
    except DeepTuneTimeExhausted:
        return close_time_exhausted()

    best_params = dict(seed_params)
    best_score = math.inf
    evals = 0
    trials_attempted = 0
    trials_completed = 0
    any_success = False
    early_stopped = False
    early_stop_reason = "none"
    budget_exhausted = False
    budget_exhausted_scope = None
    time_exhausted = False
    preflight_rejections = 0
    duplicates_skipped = 0
    duplicate_scores_reused = 0
    failure_refs = []

    # Seed best AND streak from the persisted trial history: a restarted search
    # continues the patience window instead of getting a fresh one.
    prior_best, prior_streak = prior_patience_state(args.tune_report_json)
    monitor = PatienceMonitor(
        patience=args.patience,
        start_best=prior_best,
        start_since=prior_streak,
    )

    def preflight_passes(params: dict) -> bool:
        nonlocal preflight_rejections
        ensure_deep_tune_time_remaining(time_budget)
        if not preflight_enabled:
            return True
        try:
            result = timed_preflight(
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
            raise
        except Exception as exc:
            failure = record_failure(
                report_path=args.tune_report_json,
                candidate_path=args.candidate_path,
                phase="preflight",
                method="cmaes",
                params=params,
                error=exc,
                traceback_text=traceback.format_exc(),
            )
            append_preflight_attempt(
                args.tune_report_json,
                source="cmaes",
                params=params,
                status="failed",
                failure=failure,
            )
            append_trial(
                args.tune_report_json,
                "cmaes",
                {
                    "params": params,
                    "score": None,
                    "status": "preflight_rejected",
                    **failure,
                },
            )
            preflight_rejections += 1
            ensure_deep_tune_time_remaining(time_budget)
            return False
        append_preflight_attempt(
            args.tune_report_json,
            source="cmaes",
            params=params,
            status="ok",
            result=result or {"status": "ok"},
        )
        ensure_deep_tune_time_remaining(time_budget)
        return True

    # Validated LLM re-warm proposals first, then the deferred warm configs
    # (proposed at step 0+1, not evaluated there): evaluate them up front so
    # the rare cmaes path doesn't lose them. Recorded + considered
    # for best (select-best ranks the whole report). PROPOSALS are charged to
    # the cmaes `evals` budget per objective attempt — they displace search
    # trials inside the bout cap, never add to it (spec §6) — while DEFERRED
    # configs stay EXTRA (pre-paid step-0+1 savings), not charged. CMA-ES still
    # seeds x0 from the best evaluated prior. Configs outside the (possibly
    # clamped) box were skipped earlier — never attempted, no budget, no
    # patience effect — and accounted via rewarm_skipped_outside_space /
    # deferred_skipped_outside_space.
    upfront_configs = [(params, True) for params in proposals_in_space]
    upfront_configs += [(params, False) for params in deferred_in_space]
    for d_params, charge_to_budget in upfront_configs:
        if charge_to_budget and evals >= args.max_evals:
            # Proposals displace the evals budget; once it is spent the
            # remaining proposals are dropped (deferred extras still run).
            continue
        try:
            ensure_deep_tune_time_remaining(time_budget)
        except DeepTuneTimeExhausted:
            time_exhausted = True
            early_stopped = True
            early_stop_reason = "time_budget"
            break
        params = cast_params_to_search_space(dict(d_params), search_space)
        try:
            preflight_ok = preflight_passes(params)
        except DeepTuneTimeExhausted:
            time_exhausted = True
            early_stopped = True
            early_stop_reason = "time_budget"
            break
        if not preflight_ok:
            # No budget charge, for either kind: preflight reserves no slot in
            # evaluation_attempts.jsonl and never reaches score_fn, so a
            # rejected proposal was never an objective attempt and must not
            # displace one. `evals` is a pure budget gate in this finite loop
            # (unlike the search loop below, where it also bounds iteration),
            # so skipping the increment cannot stall progress here.
            identity = params_identity(params)
            attempted_identities.add(identity)
            known_infeasible.add(identity)
            # Scoreless trials are non-improvements and count toward patience.
            if monitor.update_failed():
                early_stopped = True
                early_stop_reason = "patience"
                break
            continue
        try:
            ensure_deep_tune_time_remaining(time_budget)
            score = timed_eval(
                evaluate,
                make_model,
                params,
                args.candidate_path,
                phase="phase_c",
                method="cmaes",
                phase_time_limit_seconds=lambda: deep_tune_time_remaining(
                    time_budget
                ),
            )
        except DeepTuneTimeExhausted as exc:
            if exc.attempt_reserved:
                trials_attempted += 1
                attempted_identities.add(params_identity(params))
                failure = record_failure(
                    report_path=args.tune_report_json,
                    candidate_path=args.candidate_path,
                    phase="phase_c",
                    method="cmaes",
                    params=params,
                    error=exc,
                    traceback_text=traceback.format_exc(),
                )
                append_trial(
                    args.tune_report_json,
                    "cmaes",
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
            time_exhausted = True
            early_stopped = True
            early_stop_reason = "time_budget"
            break
        except EvaluationBudgetExhausted as exc:
            budget_exhausted = True
            budget_exhausted_scope = exc.scope
            early_stopped = True
            early_stop_reason = "evaluation_budget"
            break
        except Exception as exc:
            trials_attempted += 1
            if charge_to_budget:
                evals += 1
            identity = params_identity(params)
            attempted_identities.add(identity)
            if is_config_infeasible_error(exc):
                known_infeasible.add(identity)
            failure = record_failure(
                report_path=args.tune_report_json,
                candidate_path=args.candidate_path,
                phase="phase_c",
                method="cmaes",
                params=params,
                error=exc,
                traceback_text=traceback.format_exc(),
            )
            append_trial(args.tune_report_json, "cmaes",
                         {"params": params, "score": None, "status": "failed",
                          "config_infeasible": is_config_infeasible_error(exc), **failure})
            if failure["failure_ref"] not in failure_refs:
                failure_refs.append(failure["failure_ref"])
            if monitor.update_failed():
                early_stopped = True
                early_stop_reason = "patience"
                break
            continue
        trials_attempted += 1
        if charge_to_budget:
            evals += 1
        append_trial(args.tune_report_json, "cmaes", {"params": params, "score": score})
        identity = params_identity(params)
        attempted_identities.add(identity)
        known_scores[identity] = float(score)
        known_score_params[identity] = params
        known_infeasible.discard(identity)
        any_success = True
        trials_completed += 1
        if score < best_score:
            best_score, best_params = score, params

    while evals < args.max_evals and not budget_exhausted and not early_stopped:
        try:
            ensure_deep_tune_time_remaining(time_budget)
        except DeepTuneTimeExhausted:
            time_exhausted = True
            early_stopped = True
            early_stop_reason = "time_budget"
            break
        if es.stop():
            early_stopped = True
            early_stop_reason = "cma_internal"
            break
        xs = es.ask()
        results = []  # [(x, fitness_or_None)] — None marks a failed evaluation
        stop_now = False
        for x in xs:
            try:
                ensure_deep_tune_time_remaining(time_budget)
            except DeepTuneTimeExhausted:
                time_exhausted = True
                early_stopped = True
                early_stop_reason = "time_budget"
                stop_now = True
                break
            if evals >= args.max_evals:
                break
            params = decode(np.asarray(x))
            params = cast_params_to_search_space(params, search_space)
            identity = params_identity(params)
            if identity in attempted_identities:
                duplicates_skipped += 1
                evals += 1
                cached_score = known_scores.get(identity)
                results.append((x, cached_score))
                if cached_score is None:
                    stop_duplicate = monitor.update_failed()
                else:
                    duplicate_scores_reused += 1
                    if cached_score < best_score:
                        best_score = cached_score
                        best_params = dict(
                            known_score_params.get(identity, params)
                        )
                    stop_duplicate = monitor.update(cached_score)
                if stop_duplicate:
                    early_stopped = True
                    early_stop_reason = "patience"
                    stop_now = True
                    break
                continue
            try:
                preflight_ok = preflight_passes(params)
            except DeepTuneTimeExhausted:
                time_exhausted = True
                early_stopped = True
                early_stop_reason = "time_budget"
                stop_now = True
                break
            if not preflight_ok:
                results.append((x, None))
                evals += 1
                attempted_identities.add(identity)
                known_infeasible.add(identity)
                # A preflight-rejected point is a non-improvement: count it
                # toward patience so failure streaks cannot sidestep stopping.
                if monitor.update_failed():
                    early_stopped = True
                    early_stop_reason = "patience"
                    stop_now = True
                    break
                continue
            try:
                ensure_deep_tune_time_remaining(time_budget)
                score = timed_eval(
                    evaluate,
                    make_model,
                    params,
                    args.candidate_path,
                    phase="phase_c",
                    method="cmaes",
                    phase_time_limit_seconds=lambda: deep_tune_time_remaining(
                        time_budget
                    ),
                )
            except DeepTuneTimeExhausted as exc:
                if exc.attempt_reserved:
                    trials_attempted += 1
                    attempted_identities.add(identity)
                    failure = record_failure(
                        report_path=args.tune_report_json,
                        candidate_path=args.candidate_path,
                        phase="phase_c",
                        method="cmaes",
                        params=params,
                        error=exc,
                        traceback_text=traceback.format_exc(),
                    )
                    append_trial(
                        args.tune_report_json,
                        "cmaes",
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
                time_exhausted = True
                early_stopped = True
                early_stop_reason = "time_budget"
                stop_now = True
                break
            except EvaluationBudgetExhausted as exc:
                budget_exhausted = True
                budget_exhausted_scope = exc.scope
                early_stopped = True
                early_stop_reason = "evaluation_budget"
                stop_now = True
                break
            except Exception as exc:
                trials_attempted += 1
                attempted_identities.add(identity)
                if is_config_infeasible_error(exc):
                    known_infeasible.add(identity)
                # One bad param region must not kill the whole search: record the
                # failure for audit and carry it as a penalty placeholder below.
                failure = record_failure(
                    report_path=args.tune_report_json,
                    candidate_path=args.candidate_path,
                    phase="phase_c",
                    method="cmaes",
                    params=params,
                    error=exc,
                    traceback_text=traceback.format_exc(),
                )
                append_trial(args.tune_report_json, "cmaes",
                             {"params": params, "score": None, "status": "failed",
                              "config_infeasible": is_config_infeasible_error(exc), **failure})
                if failure["failure_ref"] not in failure_refs:
                    failure_refs.append(failure["failure_ref"])
                results.append((x, None))
                evals += 1
                if monitor.update_failed():
                    early_stopped = True
                    early_stop_reason = "patience"
                    stop_now = True
                    break
                continue
            trials_attempted += 1
            append_trial(
                args.tune_report_json, "cmaes", {"params": params, "score": score}
            )
            attempted_identities.add(identity)
            known_scores[identity] = float(score)
            known_score_params[identity] = params
            known_infeasible.discard(identity)
            any_success = True
            trials_completed += 1
            # CMA-ES minimizes and scores are lower-is-better → fitness = score.
            fit = score
            results.append((x, fit))
            if score < best_score:
                best_score = score
                best_params = params
            evals += 1
            if monitor.update(score):
                early_stopped = True
                early_stop_reason = "patience"
                stop_now = True
                break
        # Tell CMA-ES only a full generation; failed points take the generation's
        # worst (finite) fitness so the distribution steers away — never inf/nan,
        # which would break the covariance update.
        succ = [f for _, f in results if f is not None]
        if succ and len(results) == len(xs):
            penalty = max(succ)
            es.tell(xs, [f if f is not None else penalty for _, f in results])
        if stop_now:
            break

    stage_elapsed = deep_tune_stage_elapsed(time_budget)

    if not any_success and budget_exhausted:
        set_stage_meta(
            args.tune_report_json,
            "cmaes",
            bout_index=time_budget["bout_index"],
            status="budget_exhausted",
            elapsed_seconds=stage_elapsed,
            early_stopped=True,
            preflight_rejections=preflight_rejections,
            duplicates_skipped=duplicates_skipped,
            duplicate_scores_reused=duplicate_scores_reused,
            budget_exhausted_scope=budget_exhausted_scope,
        )
        write_json({
            "method": "cmaes",
            "status": "budget_exhausted",
            "reason": "evaluation allocation exhausted before score_fn",
            "budget_exhausted_scope": budget_exhausted_scope,
            "trials_completed": trials_completed,
            "trials_attempted": trials_attempted,
            "preflight_rejections": preflight_rejections,
            "duplicates_skipped": duplicates_skipped,
            "duplicate_scores_reused": duplicate_scores_reused,
            "elapsed_seconds": round(stage_elapsed, 1),
        })
        return 0

    if not any_success and time_exhausted:
        return close_time_exhausted(
            trials_completed=trials_completed,
            trials_attempted=trials_attempted,
            preflight_rejections=preflight_rejections,
        )

    if (
        not any_success
        and trials_attempted == 0
        and preflight_rejections == 0
        and duplicate_scores_reused > 0
    ):
        set_stage_meta(
            args.tune_report_json,
            "cmaes",
            bout_index=time_budget["bout_index"],
            status="failed",
            elapsed_seconds=stage_elapsed,
            early_stopped=True,
            early_stop_reason="proposal_space_exhausted",
            duplicates_skipped=duplicates_skipped,
            duplicate_scores_reused=duplicate_scores_reused,
            trials_completed=0,
            trials_attempted=0,
        )
        write_json({
            "method": "cmaes",
            "status": "failed",
            "reason": (
                "optimizer proposed only previously attempted configurations; "
                "no new objective trial was admitted"
            ),
            "best_params": best_params,
            "best_score": best_score,
            "trials_completed": 0,
            "trials_attempted": 0,
            "preflight_rejections": 0,
            "duplicates_skipped": duplicates_skipped,
            "duplicate_scores_reused": duplicate_scores_reused,
            "early_stopped": True,
            "early_stop_reason": "proposal_space_exhausted",
            "elapsed_seconds": round(stage_elapsed, 1),
            "search_space": search_space_for_json(search_space),
        })
        return 0

    if not any_success:
        # Every evaluated trial errored — surface a failed stage instead of
        # writing the seed defaults as if they were a real "ok" best.
        set_stage_meta(args.tune_report_json, "cmaes", status="failed",
                       bout_index=time_budget["bout_index"],
                       elapsed_seconds=stage_elapsed, early_stopped=early_stopped)
        write_json({
            "method": "cmaes",
            "status": "failed",
            "reason": "all CMA-ES trials errored; no completed trial",
            "trials_completed": trials_completed,
            "trials_attempted": trials_attempted,
            "preflight_rejections": preflight_rejections,
            "duplicates_skipped": duplicates_skipped,
            "duplicate_scores_reused": duplicate_scores_reused,
            "prior_trials_seen": len(prior_trials),
            "popsize": args.popsize,
            "early_stopped": early_stopped,
            "early_stop_reason": early_stop_reason,
            "failure_refs": failure_refs[-3:],
            "elapsed_seconds": round(stage_elapsed, 1),
            "search_space": search_space_for_json(search_space),
        })
        return 0

    set_stage_meta(
        args.tune_report_json,
        "cmaes",
        bout_index=time_budget["bout_index"],
        status="ok",
        elapsed_seconds=stage_elapsed,
        early_stopped=early_stopped,
        preflight_rejections=preflight_rejections,
        duplicates_skipped=duplicates_skipped,
        duplicate_scores_reused=duplicate_scores_reused,
        budget_exhausted=budget_exhausted,
        time_limit_seconds=time_budget["limit_seconds"],
    )

    write_json({
        "method": "cmaes",
        "status": "ok",
        "best_params": best_params,
        "best_score": best_score,
        "trials_completed": trials_completed,
        "trials_attempted": trials_attempted,
        "preflight_rejections": preflight_rejections,
        "duplicates_skipped": duplicates_skipped,
        "duplicate_scores_reused": duplicate_scores_reused,
        "budget_exhausted": budget_exhausted,
        "deferred_skipped_outside_space": len(deferred_outside),
        "deferred_skipped_already_seen": deferred_skipped_seen,
        "prior_trials_seen": len(prior_trials),
        "x0_from_prior": best_prior is not None,
        "popsize": args.popsize,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "elapsed_seconds": round(stage_elapsed, 1),
        "time_limit_seconds": time_budget["limit_seconds"],
        "search_space": search_space_for_json(search_space),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
