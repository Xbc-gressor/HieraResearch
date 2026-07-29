#!/usr/bin/env python3
"""Step 1 — warm-config evaluator (sequential, resumable, stop-on-crash).

Run by `tunable-contract-extractor` (segment ③) after it has produced a
candidate `train.py` (PARAM_SCHEMA + SEARCH_SPACE + make_model, NO BASE_PARAMS
yet) and written `_warm_configs.json` (the K warm configs). It evaluates those
configs on the score fn (`prepare`'s `score_fn`) and seeds the candidate's
tuning. Sequential + resumable + stop-on-crash, so the extractor can diagnose +
fix ONE crash at a time without re-evaluating what already passed:

1. CREATE `BASE_PARAMS` (so the candidate is a complete contract that imports).
2. Evaluate the configs IN ORDER. **Resume**: any config already scored in a
   prior run's report is reused, not re-evaluated (passed configs are cached by
   params). On the FIRST not-yet-scored config that raises, record it (index +
   FULL traceback) and STOP — exit `3` (CRASHED). The caller diagnoses it
   (config-invalid → edit the config in `_warm_configs.json`; code-incompatible
   → edit `train.py`), then re-runs this to resume.
3. When every config has a score (no crash), pick best-of-K′ (= min over all K),
   write it into `BASE_PARAMS`, finalize `phase_a` (warm_start_configs +
   best_warm_score + best_warm_params + search_space), exit `0`.

An optional task-owned preflight runs before each score attempt in an isolated
subprocess. It is a real-shape feasibility check, not a smoke score, and never
reserves an objective slot. There is no `base_score`. Run from the task uv env:
`uv --project tasks/<task> run python tools/tuners/warmstart_eval.py ...`
(`--project` selects the task env without chdir, so repo-relative paths resolve;
`--directory` would chdir into the task dir and break them).

Every candidate requires a schema-3 `_candidate_brief.json` with a recognized
implementation origin. A brief stamped
`implementation_source.kind=provided_entrypoint` enables the observed-control
guard: exactly one warm config may run, `k_eval` must be one, and a literal
`DEFAULT_PARAMS` must match that config exactly.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # tools/ for apply_base_params
import apply_base_params  # noqa: E402
from _common import (  # noqa: E402
    EvaluationBudgetExhausted,
    resolve_score_fn,
    resolve_preflight_fn,
    timed_eval,
    timed_preflight,
    cast_params_to_search_space,
    is_finite_score,
    load_candidate_modules,
    read_tune_report,
    search_space_for_json,
    write_json,
    write_tune_report,
)
from failure_artifacts import record_failure  # noqa: E402

CRASHED = 3  # a not-yet-scored config raised; the caller diagnoses + fixes + resumes
BUDGET_EXHAUSTED = 4  # no score_fn call was started; coordinator ends the run


def _params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def _literal_default_params(candidate_path: Path) -> dict | None:
    tree = ast.parse(candidate_path.read_text())
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_PARAMS"
            for target in targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return None
        return value if isinstance(value, dict) else None
    return None


def validate_provided_baseline_configs(
    candidate_path: Path,
    configs: list,
    k_eval: int | None,
) -> None:
    """Validate candidate origin, then keep a provided control to one trial."""
    brief_path = candidate_path.parent / "_candidate_brief.json"
    try:
        brief = json.loads(brief_path.read_text())
    except OSError as exc:
        raise ValueError(f"warmstart requires candidate brief {brief_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid candidate brief {brief_path}: {exc}") from exc
    if not isinstance(brief, dict) or brief.get("schema_version") != 3:
        raise ValueError("warmstart requires a schema-3 candidate brief")
    source = brief.get("implementation_source")
    if not isinstance(source, dict):
        raise ValueError("candidate brief requires an implementation_source object")
    source_kind = source.get("kind")
    if source_kind not in {"generated", "legacy_generated", "provided_entrypoint"}:
        raise ValueError(
            "candidate brief implementation_source.kind must be generated, "
            "legacy_generated, or provided_entrypoint"
        )
    if source_kind != "provided_entrypoint":
        return
    source_path = source.get("path")
    source_sha256 = source.get("sha256")
    if (
        not isinstance(source_path, str)
        or not source_path
        or not isinstance(source_sha256, str)
        or len(source_sha256) != 71
        or not source_sha256.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in source_sha256[7:])
    ):
        raise ValueError(
            "provided_entrypoint implementation_source requires path and sha256 receipt"
        )
    if len(configs) != 1:
        raise ValueError("provided baseline requires exactly one warm config")
    if k_eval not in (None, 1):
        raise ValueError("provided baseline requires k_eval=1")
    defaults = _literal_default_params(candidate_path)
    if defaults is None:
        raise ValueError(
            "provided baseline requires a module-level literal DEFAULT_PARAMS"
        )
    if configs[0] != defaults:
        raise ValueError(
            "provided baseline warm config must equal its literal DEFAULT_PARAMS"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--configs-json", required=True, type=Path,
                        help="JSON list of K warm config dicts (_warm_configs.json)")
    parser.add_argument("--tune-report-json", required=True, type=Path)
    parser.add_argument("--k-eval", type=int, default=None,
                        help="evaluate only the FIRST k-eval configs now (best-of-k-eval = "
                             "screening score); the rest are DEFERRED — stored params-only and "
                             "evaluated later by the deep-tuner (BO enqueue / grid prepend) only if "
                             "this candidate is selected. Default = all (no deferral).")
    args = parser.parse_args()

    with open(args.configs_json) as f:
        all_configs = json.load(f)
    if not isinstance(all_configs, list) or not all_configs:
        parser.error(f"--configs-json must be a non-empty list, got {type(all_configs).__name__}")
    try:
        validate_provided_baseline_configs(
            args.candidate_path,
            all_configs,
            args.k_eval,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Split into eval-now (first k_eval) and deferred (the rest, evaluated by the
    # tuner only if promoted). k_eval=None / >=len → evaluate all (no deferral).
    k_eval = len(all_configs) if args.k_eval is None else max(1, min(args.k_eval, len(all_configs)))
    configs = all_configs[:k_eval]
    deferred = all_configs[k_eval:]

    # Ensure BASE_PARAMS exists (create on the first run, rewrite later) so
    # load_candidate_modules' REQUIRED_SYMBOLS check passes; AST reads SEARCH_SPACE
    # from the file, so this precedes the import.
    apply_base_params.apply(args.candidate_path, dict(configs[0]))

    train_module, prepare_module = load_candidate_modules(args.candidate_path)
    search_space = train_module.SEARCH_SPACE
    make_model = train_module.make_model
    evaluate = resolve_score_fn(prepare_module, args.candidate_path)
    preflight_enabled = resolve_preflight_fn(prepare_module, args.candidate_path) is not None

    previous_report = read_tune_report(args.tune_report_json)
    previous_phase_a = previous_report.get("phase_a", {})

    # Resume cache: configs already scored in a prior run, keyed by params. A
    # config the caller edited (config-invalid fix) gets new params → cache miss
    # → re-evaluated; a config that crashed has no score → re-evaluated; passed
    # configs are reused (not re-run, even after a code fix — fix-forward).
    prev = previous_phase_a.get("warm_start_configs", [])
    cache = {_params_key(t["params"]): t["score"] for t in prev
             if isinstance(t.get("params"), dict) and is_finite_score(t.get("score"))}
    trials_attempted = previous_phase_a.get("trials_attempted")
    if not isinstance(trials_attempted, int) or isinstance(trials_attempted, bool) \
            or trials_attempted < 0:
        # Backward-compatible recovery for reports written before the explicit
        # attempt counter: every persisted warm row came from one score_fn call.
        trials_attempted = len(prev)

    report = previous_report
    preflight_report = report.setdefault("preflight", {"attempts": [], "invocations": 0})
    if preflight_enabled:
        preflight_report["invocations"] = len(preflight_report.get("attempts", []))
        preflight_report["status"] = "running"
    report["phase_a"] = {
        "warm_start_configs": [],
        # deferred = proposed-but-not-evaluated-now; the deep-tuner evaluates these
        # first (BO enqueue / grid prepend) only if this candidate is promoted.
        "deferred_configs": [{"params": cast_params_to_search_space(dict(d), search_space)}
                             for d in deferred],
        "search_space": search_space_for_json(search_space),
        "trials_attempted": trials_attempted,
        "status": "running",
    }
    write_tune_report(args.tune_report_json, report)

    started = time.time()
    wsc: list[dict] = []
    for i, raw in enumerate(configs):
        params = cast_params_to_search_space(dict(raw), search_space)
        if preflight_enabled:
            try:
                result = timed_preflight(params, args.candidate_path)
            except Exception as exc:
                tb = traceback.format_exc()
                sys.stderr.write(tb)
                failure = record_failure(
                    report_path=args.tune_report_json,
                    candidate_path=args.candidate_path,
                    phase="preflight",
                    method="warmstart",
                    params=params,
                    error=exc,
                    traceback_text=tb,
                )
                preflight_report.setdefault("attempts", []).append(
                    {
                        "params": params,
                        "source": "warmstart",
                        "status": "failed",
                        **failure,
                    }
                )
                preflight_report["invocations"] = len(preflight_report["attempts"])
                preflight_report["status"] = "failed"
                report["phase_a"]["warm_start_configs"] = wsc
                report["phase_a"]["status"] = "preflight_failed"
                write_tune_report(args.tune_report_json, report)
                write_json(
                    {
                        "phase": "preflight",
                        "status": "crashed",
                        "crash_index": i,
                        "crash_params": params,
                        "objective_slot_consumed": False,
                        **failure,
                    }
                )
                return CRASHED
            preflight_report.setdefault("attempts", []).append(
                {
                    "params": params,
                    "source": "warmstart",
                    "status": "ok",
                    "result": result or {"status": "ok"},
                }
            )
            preflight_report["invocations"] = len(preflight_report["attempts"])
            write_tune_report(args.tune_report_json, report)

        key = _params_key(params)
        if key in cache:
            wsc.append({"params": params, "score": cache[key]})
        else:
            try:
                score = timed_eval(
                    evaluate,
                    make_model,
                    params,
                    args.candidate_path,
                    phase="phase_a",
                    method="warmstart",
                )
            except EvaluationBudgetExhausted as exc:
                if preflight_enabled:
                    preflight_report["status"] = "ok"
                report["phase_a"]["warm_start_configs"] = wsc
                if wsc:
                    unscored = configs[i:] + deferred
                    report["phase_a"]["deferred_configs"] = [
                        {
                            "params": cast_params_to_search_space(
                                dict(config),
                                search_space,
                            )
                        }
                        for config in unscored
                    ]
                    best_params, best_warm_score = min(
                        ((trial["params"], trial["score"]) for trial in wsc),
                        key=lambda item: item[1],
                    )
                    apply_base_params.apply(args.candidate_path, best_params)
                    elapsed = time.time() - started
                    report["phase_a"].update(
                        {
                            "best_warm_score": best_warm_score,
                            "best_warm_params": best_params,
                            "k_evaluated": len(wsc),
                            "k_survived": len(wsc),
                            "k_deferred": len(unscored),
                            "elapsed_seconds": round(elapsed, 1),
                            "status": "ok",
                            "budget_exhausted": True,
                        }
                    )
                    write_tune_report(args.tune_report_json, report)
                    write_json(
                        {
                            "phase": "a",
                            "status": "ok",
                            "budget_exhausted": True,
                            "k_evaluated": len(wsc),
                            "trials_attempted": trials_attempted,
                            "best_warm_score": best_warm_score,
                            "best_warm_params": best_params,
                            "elapsed_seconds": round(elapsed, 1),
                        }
                    )
                    return 0
                report["phase_a"]["status"] = "budget_exhausted"
                write_tune_report(args.tune_report_json, report)
                write_json(
                    {
                        "phase": "a",
                        "status": "budget_exhausted",
                        "reason": str(exc),
                        "objective_slot_consumed": False,
                    }
                )
                return BUDGET_EXHAUSTED
            except Exception as exc:
                trials_attempted += 1
                report["phase_a"]["trials_attempted"] = trials_attempted
                tb = traceback.format_exc()
                sys.stderr.write(tb)
                failure = record_failure(
                    report_path=args.tune_report_json,
                    candidate_path=args.candidate_path,
                    phase="phase_a",
                    method="warmstart",
                    params=params,
                    error=exc,
                    traceback_text=tb,
                )
                wsc.append({"params": params, "score": None, "status": "failed",
                            **failure})
                report["phase_a"]["warm_start_configs"] = wsc
                report["phase_a"]["status"] = "crashed"
                write_tune_report(args.tune_report_json, report)
                write_json({"phase": "a", "status": "crashed", "crash_index": i,
                            "crash_params": params,
                            **failure})
                return CRASHED
            trials_attempted += 1
            report["phase_a"]["trials_attempted"] = trials_attempted
            wsc.append({"params": params, "score": score})
            cache[key] = score
        report["phase_a"]["warm_start_configs"] = wsc
        write_tune_report(args.tune_report_json, report)

    # ---- every config scored → best-of-K′ → BASE_PARAMS ----
    best_params, best_warm_score = min(((t["params"], t["score"]) for t in wsc),
                                       key=lambda t: t[1])
    apply_base_params.apply(args.candidate_path, best_params)
    elapsed = time.time() - started

    report["phase_a"].update({
        "best_warm_score": best_warm_score,
        "best_warm_params": best_params,
        "k_evaluated": len(configs),
        "k_survived": len(wsc),
        "k_deferred": len(deferred),
        "elapsed_seconds": round(elapsed, 1),
        "status": "ok",
    })
    if preflight_enabled:
        preflight_report["status"] = "ok"
    write_tune_report(args.tune_report_json, report)

    write_json({
        "phase": "a",
        "status": "ok",
        "k_evaluated": len(configs),
        "k_survived": len(wsc),
        "trials_attempted": trials_attempted,
        "best_warm_score": best_warm_score,
        "best_warm_params": best_params,
        "elapsed_seconds": round(elapsed, 1),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
