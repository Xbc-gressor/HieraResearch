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

Invoked by the tuner-orchestrator agent when n_dims ≥ 16.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    resolve_score_fn,
    timed_eval,
    PatienceMonitor,
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


def build_codec(search_space: dict, base_params: dict):
    """Build (keys, lower, upper, x0, decode) for a continuous CMA-ES vector."""
    keys = list(search_space.keys())
    lower = []
    upper = []
    x0 = []
    decoders = []

    for key in keys:
        entry = search_space[key]
        kind = entry[0]
        base_value = base_params.get(key)

        if kind == "float":
            low, high = float(entry[1]), float(entry[2])
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
            lower.append(low - 0.5)
            upper.append(high + 0.5)
            x0.append(
                float(base_value) if base_value is not None else (low + high) / 2
            )
            decoders.append(("int", key, low, high))
        elif kind == "categorical":
            choices = list(entry[1])
            n = len(choices)
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
                vec.append(float(choices.index(params[key])) if params[key] in choices else 0.0)
        return np.asarray(vec, dtype=float)

    def decode(x: np.ndarray) -> dict:
        out = {}
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--tune-report-json", required=True, type=Path)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--max-evals", type=int, default=64)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--sigma0", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import cma
    except ImportError:
        set_stage_meta(args.tune_report_json, "cmaes", status="rejected")
        write_json({
            "method": "cmaes",
            "status": "rejected",
            "reason": "cma not installed in the task uv environment",
        })
        return 0

    train_module, prepare_module = load_candidate_modules(args.candidate_path)
    base_params = dict(train_module.BASE_PARAMS)
    search_space = train_module.SEARCH_SPACE
    make_model = train_module.make_model
    evaluate = resolve_score_fn(prepare_module, args.candidate_path)

    prior_trials = read_prior_trials(args.tune_report_json)
    best_prior = None
    if prior_trials:
        best_prior = min(prior_trials, key=lambda t: t.get("score", math.inf))
    seed_params = best_prior["params"] if best_prior else base_params

    keys, lower, upper, x0_default, encode, decode = build_codec(
        search_space, seed_params
    )
    try:
        x0 = encode(seed_params)
        x0 = [max(low, min(high, float(v))) for low, high, v in zip(lower, upper, x0)]
    except Exception:
        x0 = x0_default

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

    started = time.time()
    best_params = dict(seed_params)
    best_score = math.inf
    evals = 0
    any_success = False
    early_stopped = False
    early_stop_reason = "none"
    failure_refs = []

    monitor = PatienceMonitor(
        patience=args.patience,
        start_best=prior_best_score(prior_trials),
    )

    # Deferred warm configs (proposed at step 0+1, not evaluated there): evaluate
    # them up front so the rare cmaes path doesn't lose them. Recorded + considered
    # for best (select-best ranks the whole report); they are EXTRA — not charged to
    # the cmaes `evals` budget (cmaes still seeds x0 from the best evaluated prior).
    for d_params in read_deferred_configs(args.tune_report_json):
        params = cast_params_to_search_space(dict(d_params), search_space)
        try:
            score = timed_eval(evaluate, make_model, params, args.candidate_path)
        except Exception as exc:
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
                         {"params": params, "score": None, "status": "failed", **failure})
            if failure["failure_ref"] not in failure_refs:
                failure_refs.append(failure["failure_ref"])
            continue
        append_trial(args.tune_report_json, "cmaes", {"params": params, "score": score})
        any_success = True
        if score < best_score:
            best_score, best_params = score, params

    while evals < args.max_evals:
        if es.stop():
            early_stopped = True
            early_stop_reason = "cma_internal"
            break
        xs = es.ask()
        results = []  # [(x, fitness_or_None)] — None marks a failed evaluation
        stop_now = False
        for x in xs:
            if evals >= args.max_evals:
                break
            params = decode(np.asarray(x))
            params = cast_params_to_search_space(params, search_space)
            try:
                score = timed_eval(evaluate, make_model, params, args.candidate_path)
            except Exception as exc:
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
                             {"params": params, "score": None, "status": "failed", **failure})
                if failure["failure_ref"] not in failure_refs:
                    failure_refs.append(failure["failure_ref"])
                results.append((x, None))
                evals += 1
                continue
            append_trial(
                args.tune_report_json, "cmaes", {"params": params, "score": score}
            )
            any_success = True
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

    elapsed = time.time() - started

    if not any_success:
        # Every evaluated trial errored — surface a failed stage instead of
        # writing the seed defaults as if they were a real "ok" best.
        set_stage_meta(args.tune_report_json, "cmaes", status="failed",
                       elapsed_seconds=round(elapsed, 1), early_stopped=early_stopped)
        write_json({
            "method": "cmaes",
            "status": "failed",
            "reason": "all CMA-ES trials errored; no completed trial",
            "trials_completed": evals,
            "prior_trials_seen": len(prior_trials),
            "popsize": args.popsize,
            "early_stopped": early_stopped,
            "early_stop_reason": early_stop_reason,
            "failure_refs": failure_refs[-3:],
            "elapsed_seconds": round(elapsed, 1),
            "search_space": search_space_for_json(search_space),
        })
        return 0

    set_stage_meta(args.tune_report_json, "cmaes", status="ok",
                   elapsed_seconds=round(elapsed, 1), early_stopped=early_stopped)

    write_json({
        "method": "cmaes",
        "status": "ok",
        "best_params": best_params,
        "best_score": best_score,
        "trials_completed": evals,
        "prior_trials_seen": len(prior_trials),
        "x0_from_prior": best_prior is not None,
        "popsize": args.popsize,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "elapsed_seconds": round(elapsed, 1),
        "search_space": search_space_for_json(search_space),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
