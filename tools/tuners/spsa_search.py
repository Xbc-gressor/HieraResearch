#!/usr/bin/env python3
"""Two-sided SPSA deep-tune bout for one candidate (DEEP regime of inner
policy deferred-random8-hebo10-spsa10-v1; scheduler v3.2 design §2.1).

A DEEP bout is 10 objective evaluations = 5 complete perturbation pairs.
First-order two-sided simultaneous-perturbation stochastic approximation
over the NON-DEGENERATE FLOAT dimensions only, in their normalized (z)
coordinates — linear floats perturb on the linear scale, log floats on the
log scale. Integer, categorical, and degenerate float dimensions hold the
incumbent's values for the whole bout: a perturbation along them is either
impossible (degenerate) or undefined (discrete), and would only pollute the
gradient estimate.

The iterate starts from the current applied incumbent (BASE_PARAMS) at
k=0; calibration is frozen at bout admission from the candidate's finite
unique history scores (IQR at >= 4 observations, else best-worst range,
else the 0.05 default; a_0 = 2 * c_0 * target_step / s, target_step=0.05,
Spall 1998). Per pair k:

- c_k = 0.10 / (k+1)^0.101, a_k = a_0 / (k+1)^0.602 (classic exponents);
- delta_k iid Rademacher (+-1), derived deterministically from
  (seed, k, resample_count) so an interrupted bout resumes exactly;
- theta_k+- = clip(theta_k +- c_k * delta_k) to the (clamped) box;
- g_hat_j = (f(theta_k+) - f(theta_k-)) / (2 * c_k * delta_k,j);
- theta_(k+1) = clip(theta_k - a_k * g_hat).

Pair legality: the two sides must be mutually distinct and neither may
duplicate an already-attempted config; delta_k is resampled until legal,
32 consecutive failed draws close the stage as failed (never a silent
one-sided search). Outcome semantics: the gradient update applies only
when BOTH sides score finite; a config-infeasible crash or preflight
rejection on either side skips the update (iterate unchanged, k advances,
spent budget is not refunded). Non-config-specific evaluation errors are
infra failures and close the stage loudly (same convention as bo_search).

The full pair state (seed, a_0, s, k, theta) is persisted in the stage's
`spsa_state` after every pair, so an interrupted bout resumes the exact
pair it left; a NEW DEEP bout starts a fresh stage from the new incumbent
at k=0.

Reference: Spall (1992); Spall (1998). Ported from the inner-benchmark
`spsa` arm onto the production Phase-C stage machinery.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    EvaluationBudgetExhausted,
    append_preflight_attempt,
    append_trial,
    attempted_config_identities,
    cast_params_to_search_space,
    clamp_search_space_to_preflight,
    deep_tune_stage_elapsed,
    deep_tune_time_budget,
    is_config_infeasible_error,
    load_candidate_modules,
    params_identity,
    read_prior_trials,
    read_tune_report,
    resolve_preflight_fn,
    resolve_score_fn,
    search_space_for_json,
    set_stage_meta,
    timed_eval,
    timed_preflight,
    write_json,
)
from failure_artifacts import record_failure  # noqa: E402

C0 = 0.10
C_EXPONENT = 0.101
A_EXPONENT = 0.602
TARGET_STEP = 0.05
DEFAULT_SCORE_SCALE = 0.05
IQR_MIN_OBSERVATIONS = 4
MAX_CONSECUTIVE_PAIR_DRAWS = 32
MAX_SCORELESS_PAIRS = 32


def _movable_dims(search_space: dict) -> list[str]:
    """Non-degenerate float dimensions, in SEARCH_SPACE order."""
    return [
        key
        for key, entry in search_space.items()
        if entry[0] == "float" and float(entry[1]) < float(entry[2])
    ]


def _encode(params: dict, search_space: dict, movable: list[str]) -> list[float]:
    """Params -> z-vector over the movable dimensions (log floats in log space)."""
    z = []
    for key in movable:
        entry = search_space[key]
        lo, hi = float(entry[1]), float(entry[2])
        log = len(entry) >= 4 and entry[3] == "log"
        value = float(params[key])
        if log:
            lo, hi, value = math.log(lo), math.log(hi), math.log(value)
        z.append((value - lo) / (hi - lo))
    return z


def _decode(z: list[float], base_params: dict, search_space: dict, movable: list[str]) -> dict:
    """z-vector + held dimensions -> a full cast params dict."""
    params = dict(base_params)
    for key, z_value in zip(movable, z):
        entry = search_space[key]
        lo, hi = float(entry[1]), float(entry[2])
        log = len(entry) >= 4 and entry[3] == "log"
        clipped = min(1.0, max(0.0, z_value))
        if log:
            lo, hi = math.log(lo), math.log(hi)
        value = lo + clipped * (hi - lo)
        params[key] = math.exp(value) if log else value
    return cast_params_to_search_space(params, search_space)


def _calibrate(report_path: Path, search_space: dict) -> tuple[float, float]:
    """(s, a_0) from the candidate's finite unique history scores."""
    best_by_identity: dict[str, float] = {}
    for trial in read_prior_trials(report_path):
        params, score = trial.get("params"), trial.get("score")
        if not isinstance(params, dict) or not isinstance(score, (int, float)):
            continue
        if isinstance(score, bool) or not math.isfinite(float(score)):
            continue
        if set(params) != set(search_space):
            continue
        try:
            identity = params_identity(
                cast_params_to_search_space(dict(params), search_space)
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if float(score) < best_by_identity.get(identity, math.inf):
            best_by_identity[identity] = float(score)
    scores = sorted(best_by_identity.values())
    if len(scores) >= IQR_MIN_OBSERVATIONS:
        q25 = scores[len(scores) // 4]
        q75 = scores[(3 * len(scores)) // 4]
        scale = q75 - q25
    elif scores:
        scale = scores[-1] - scores[0]
    else:
        scale = 0.0
    if scale <= 0.0:
        scale = DEFAULT_SCORE_SCALE
    return scale, 2.0 * C0 * TARGET_STEP / scale


def _delta(seed: int, k: int, resample: int, size: int) -> list[int]:
    """Rademacher draw, deterministic in (seed, k, resample) so an
    interrupted bout resumes the exact pair sequence."""
    rng = random.Random(seed * 1000003 + k * 97 + resample)
    return [rng.choice((-1, 1)) for _ in range(size)]


def _current_spsa_stage(report: dict) -> dict | None:
    stages = report.get("phase_c", {}).get("stages", [])
    return next(
        (s for s in reversed(stages) if isinstance(s, dict) and s.get("method") == "spsa"),
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--tune-report-json", required=True, type=Path)
    parser.add_argument("--n-evals", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    time_budget = deep_tune_time_budget(
        args.candidate_path,
        args.tune_report_json,
        "spsa",
    )
    bout_index = time_budget["bout_index"]

    train_module, prepare_module = load_candidate_modules(
        args.candidate_path,
        expected_execution_revision=time_budget["candidate_execution_revision"],
    )
    search_space = train_module.SEARCH_SPACE
    make_model = train_module.make_model
    base_params = getattr(train_module, "BASE_PARAMS", None)
    if not isinstance(base_params, dict):
        raise ValueError("spsa requires BASE_PARAMS (the applied incumbent)")
    evaluate = resolve_score_fn(prepare_module, args.candidate_path)
    preflight_enabled = resolve_preflight_fn(prepare_module, args.candidate_path) is not None
    if preflight_enabled:
        search_space = clamp_search_space_to_preflight(
            search_space,
            base_params,
            args.candidate_path,
            args.tune_report_json,
            expected_execution_revision=time_budget["candidate_execution_revision"],
        )

    movable = _movable_dims(search_space)
    if not movable:
        # Selection should never route such a candidate to a DEEP bout; this
        # is the deterministic backstop, and it is honest (rejected, never a
        # silent TPE/grid bout labeled SPSA).
        set_stage_meta(
            args.tune_report_json,
            "spsa",
            bout_index=bout_index,
            status="rejected",
            elapsed_seconds=deep_tune_stage_elapsed(time_budget),
        )
        write_json(
            {
                "method": "spsa",
                "status": "rejected",
                "reason": "no non-degenerate continuous dimension to perturb",
            }
        )
        return 0

    # Resume or initialize the persisted pair state. Calibration is frozen at
    # bout admission: a resume must not recalibrate from the now-larger
    # history, so (s, a_0) live in the stage meta, not in a recomputation.
    stage = _current_spsa_stage(read_tune_report(args.tune_report_json)) or {}
    state = stage.get("spsa_state")
    if not isinstance(state, dict):
        scale, a_0 = _calibrate(args.tune_report_json, search_space)
        state = {
            "seed": int(args.seed),
            "s": scale,
            "a_0": a_0,
            "k": 0,
            "movable_dims": movable,
            "theta": _encode(base_params, search_space, movable),
            "updates_applied": 0,
        }
        set_stage_meta(
            args.tune_report_json,
            "spsa",
            bout_index=bout_index,
            spsa_state=state,
        )
    if state.get("movable_dims") != movable:
        raise ValueError("persisted spsa_state movable dims no longer match SEARCH_SPACE")
    theta = [float(v) for v in state["theta"]]
    k = int(state["k"])
    a_0 = float(state["a_0"])
    seed = int(state["seed"])
    updates_applied = int(state.get("updates_applied", 0))
    # Objective spend already charged to THIS stage by earlier (interrupted)
    # invocations: scored rows and config-infeasible crashes each consumed a
    # reservation; preflight rejections did not. The bout's pair budget is
    # cross-invocation — a resumed DEEP bout never exceeds its n_evals.
    prior_attempts = sum(
        1
        for row in stage.get("trials", [])
        if isinstance(row, dict)
        and (
            (
                isinstance(row.get("score"), (int, float))
                and not isinstance(row.get("score"), bool)
                and math.isfinite(float(row["score"]))
            )
            or row.get("status") == "failed"
        )
    )

    counters = {
        "objective_attempts": 0,
        "objective_completed": 0,
        "preflight_rejections": 0,
        "internal_resamples": 0,
        "budget_exhausted": False,
        "budget_exhausted_scope": None,
    }
    failure_refs: list[str] = []
    known_scores: dict[str, float] = {}
    known_score_params: dict[str, dict] = {}
    for prior in read_prior_trials(args.tune_report_json):
        params, score = prior.get("params"), prior.get("score")
        if not isinstance(params, dict) or set(params) != set(search_space):
            continue
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        if not math.isfinite(float(score)):
            continue
        try:
            normalized = cast_params_to_search_space(dict(params), search_space)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        identity = params_identity(normalized)
        if float(score) < known_scores.get(identity, math.inf):
            known_scores[identity] = float(score)
            known_score_params[identity] = normalized

    def run_leg(params: dict, side: str) -> dict:
        """Preflight + evaluate one pair leg. Returns its outcome receipt."""
        if preflight_enabled:
            try:
                preflight_result = timed_preflight(
                    params,
                    args.candidate_path,
                    expected_execution_revision=time_budget[
                        "candidate_execution_revision"
                    ],
                )
            except Exception as exc:
                failure = record_failure(
                    report_path=args.tune_report_json,
                    candidate_path=args.candidate_path,
                    phase="preflight",
                    method="spsa",
                    params=params,
                    error=exc,
                    traceback_text=traceback.format_exc(),
                )
                append_preflight_attempt(
                    args.tune_report_json,
                    source="spsa",
                    params=params,
                    status="failed",
                    failure=failure,
                )
                append_trial(
                    args.tune_report_json,
                    "spsa",
                    {
                        "params": params,
                        "score": None,
                        "status": "preflight_rejected",
                        "spsa_k": k,
                        "spsa_side": side,
                        **failure,
                    },
                )
                counters["preflight_rejections"] += 1
                if failure["failure_ref"] not in failure_refs:
                    failure_refs.append(failure["failure_ref"])
                return {"status": "preflight_rejected"}
            append_preflight_attempt(
                args.tune_report_json,
                source="spsa",
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
                method="spsa",
            )
        except EvaluationBudgetExhausted:
            raise
        except Exception as exc:
            counters["objective_attempts"] += 1
            failure = record_failure(
                report_path=args.tune_report_json,
                candidate_path=args.candidate_path,
                phase="phase_c",
                method="spsa",
                params=params,
                error=exc,
                traceback_text=traceback.format_exc(),
            )
            if failure["failure_ref"] not in failure_refs:
                failure_refs.append(failure["failure_ref"])
            config_infeasible = is_config_infeasible_error(exc)
            append_trial(
                args.tune_report_json,
                "spsa",
                {
                    "params": params,
                    "score": None,
                    "status": "failed",
                    "config_infeasible": config_infeasible,
                    "spsa_k": k,
                    "spsa_side": side,
                    **failure,
                },
            )
            if not config_infeasible:
                # Infra flake or candidate bug, not a parameter-region signal:
                # fail the stage loudly rather than silently skipping updates.
                raise
            return {"status": "failed"}
        counters["objective_attempts"] += 1
        counters["objective_completed"] += 1
        append_trial(
            args.tune_report_json,
            "spsa",
            {
                "params": params,
                "score": score,
                "spsa_k": k,
                "spsa_side": side,
            },
        )
        identity = params_identity(params)
        if float(score) < known_scores.get(identity, math.inf):
            known_scores[identity] = float(score)
            known_score_params[identity] = params
        return {"status": "ok", "score": float(score)}

    def persist_state() -> None:
        state.update({"k": k, "theta": theta, "updates_applied": updates_applied})
        set_stage_meta(
            args.tune_report_json,
            "spsa",
            bout_index=bout_index,
            spsa_state=state,
        )

    def recorded_leg_outcome(identity: str) -> dict | None:
        """Recover a pending-pair leg already evaluated by an earlier
        (interrupted) invocation of THIS bout, without re-spending budget."""
        report = read_tune_report(args.tune_report_json)
        rows = list(report.get("phase_a", {}).get("warm_start_configs", []))
        for stage_row in report.get("phase_c", {}).get("stages", []):
            if isinstance(stage_row, dict):
                rows.extend(stage_row.get("trials", []))
        for row in rows:
            params = row.get("params") if isinstance(row, dict) else None
            if not isinstance(params, dict) or set(params) != set(search_space):
                continue
            try:
                row_identity = params_identity(
                    cast_params_to_search_space(dict(params), search_space)
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if row_identity != identity:
                continue
            score = row.get("score")
            if (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
            ):
                return {"status": "ok", "score": float(score)}
            if row.get("status") in ("failed", "preflight_rejected"):
                return {"status": row["status"]}
        return None

    scoreless_pairs = 0
    close_status = "ok"
    close_reason = None
    try:
        # Pairs are atomic: a new pair starts only when both legs fit the
        # bout's remaining evaluation budget (counting earlier invocations).
        while prior_attempts + counters["objective_attempts"] + 2 <= args.n_evals:
            pending = state.get("pending_pair")
            if not (isinstance(pending, dict) and pending.get("k") == k):
                c_k = C0 / (k + 1) ** C_EXPONENT
                drawn = None
                for resample in range(MAX_CONSECUTIVE_PAIR_DRAWS):
                    delta = _delta(seed, k, resample, len(movable))
                    plus = [
                        min(1.0, max(0.0, z + c_k * d)) for z, d in zip(theta, delta)
                    ]
                    minus = [
                        min(1.0, max(0.0, z - c_k * d)) for z, d in zip(theta, delta)
                    ]
                    candidate_plus = _decode(plus, base_params, search_space, movable)
                    candidate_minus = _decode(minus, base_params, search_space, movable)
                    seen = attempted_config_identities(
                        args.tune_report_json, search_space
                    )
                    if (
                        params_identity(candidate_plus) != params_identity(candidate_minus)
                        and params_identity(candidate_plus) not in seen
                        and params_identity(candidate_minus) not in seen
                    ):
                        drawn = (delta, candidate_plus, candidate_minus)
                        break
                    counters["internal_resamples"] += 1
                if drawn is None:
                    close_status = "failed"
                    close_reason = (
                        f"{MAX_CONSECUTIVE_PAIR_DRAWS} consecutive pair draws hit "
                        "duplicates/boundary collapse"
                    )
                    break
                delta, params_plus, params_minus = drawn
                # Persist the pair BEFORE evaluating it: an interrupted bout
                # resumes this exact pair, recovering any leg the crashed
                # invocation already paid for (design §2.1).
                state["pending_pair"] = {
                    "k": k,
                    "c_k": c_k,
                    "delta": delta,
                    "plus": params_plus,
                    "minus": params_minus,
                }
                persist_state()
            pair = state["pending_pair"]
            c_k = float(pair["c_k"])
            a_k = a_0 / (k + 1) ** A_EXPONENT
            delta = [int(d) for d in pair["delta"]]
            params_plus = cast_params_to_search_space(dict(pair["plus"]), search_space)
            params_minus = cast_params_to_search_space(dict(pair["minus"]), search_space)
            outcome_plus = recorded_leg_outcome(params_identity(params_plus))
            if outcome_plus is None:
                outcome_plus = run_leg(params_plus, "plus")
            outcome_minus = recorded_leg_outcome(params_identity(params_minus))
            if outcome_minus is None:
                outcome_minus = run_leg(params_minus, "minus")
            if (
                outcome_plus.get("status") == "ok"
                and outcome_minus.get("status") == "ok"
            ):
                gradient = [
                    (outcome_plus["score"] - outcome_minus["score"]) / (2.0 * c_k * d)
                    for d in delta
                ]
                theta = [
                    min(1.0, max(0.0, z - a_k * g))
                    for z, g in zip(theta, gradient)
                ]
                updates_applied += 1
            # else: crash or preflight rejection on either side — no score
            # pair exists, so skip the update (iterate unchanged); the spent
            # budget is not refunded.
            if (
                outcome_plus.get("status") == "preflight_rejected"
                and outcome_minus.get("status") == "preflight_rejected"
            ):
                scoreless_pairs += 1
            else:
                scoreless_pairs = 0
            state.pop("pending_pair", None)
            if scoreless_pairs >= MAX_SCORELESS_PAIRS:
                close_status = "failed"
                close_reason = (
                    f"{MAX_SCORELESS_PAIRS} consecutive pairs were fully "
                    "preflight-rejected; no measurable pair remains"
                )
                persist_state()
                break
            k += 1
            persist_state()
    except EvaluationBudgetExhausted as exc:
        counters["budget_exhausted"] = True
        counters["budget_exhausted_scope"] = exc.scope
        persist_state()

    stage_elapsed = deep_tune_stage_elapsed(time_budget)

    if counters["budget_exhausted"] and counters["objective_completed"] == 0:
        set_stage_meta(
            args.tune_report_json,
            "spsa",
            bout_index=bout_index,
            status="budget_exhausted",
            elapsed_seconds=stage_elapsed,
            preflight_rejections=counters["preflight_rejections"],
            budget_exhausted_scope=counters["budget_exhausted_scope"],
        )
        write_json(
            {
                "method": "spsa",
                "status": "budget_exhausted",
                "reason": "evaluation allocation exhausted before score_fn",
                "budget_exhausted_scope": counters["budget_exhausted_scope"],
                "trials_completed": 0,
                "trials_attempted": counters["objective_attempts"],
                "preflight_rejections": counters["preflight_rejections"],
                "elapsed_seconds": round(stage_elapsed, 1),
            }
        )
        return 0

    if close_status == "ok" and counters["objective_completed"] == 0:
        close_status = "failed"
        close_reason = "all SPSA legs errored or were rejected; no completed trial"

    receipt = {
        "method": "spsa",
        "status": close_status,
        "trials_completed": counters["objective_completed"],
        "trials_attempted": counters["objective_attempts"],
        "preflight_rejections": counters["preflight_rejections"],
        "pairs_attempted": k,
        "updates_applied": updates_applied,
        "internal_resamples": counters["internal_resamples"],
        "budget_exhausted": counters["budget_exhausted"],
        "s": state["s"],
        "a_0": a_0,
        "n_movable_dims": len(movable),
        "failure_refs": failure_refs[-3:],
        "elapsed_seconds": round(stage_elapsed, 1),
        "search_space": search_space_for_json(search_space),
    }
    if close_reason is not None:
        receipt["reason"] = close_reason
    if known_scores:
        best_identity = min(known_scores, key=known_scores.get)
        receipt["best_params"] = known_score_params[best_identity]
        receipt["best_score"] = known_scores[best_identity]
    set_stage_meta(
        args.tune_report_json,
        "spsa",
        bout_index=bout_index,
        status=close_status,
        elapsed_seconds=stage_elapsed,
        preflight_rejections=counters["preflight_rejections"],
        pairs_attempted=k,
        updates_applied=updates_applied,
        internal_resamples=counters["internal_resamples"],
        budget_exhausted=counters["budget_exhausted"],
        budget_exhausted_scope=counters["budget_exhausted_scope"],
        time_limit_seconds=time_budget["limit_seconds"],
    )
    write_json(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
