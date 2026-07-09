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

No smoke/non-smoke distinction; no `base_score`. Run from the task uv env:
`uv --directory tasks/<task> run python tools/tuners/warmstart_eval.py ...`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # tools/ for apply_base_params
import apply_base_params  # noqa: E402
from _common import (  # noqa: E402
    resolve_score_fn,
    timed_eval,
    cast_params_to_search_space,
    load_candidate_modules,
    read_tune_report,
    search_space_for_json,
    write_json,
    write_tune_report,
)

CRASHED = 3  # a not-yet-scored config raised; the caller diagnoses + fixes + resumes


def _params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


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

    # Resume cache: configs already scored in a prior run, keyed by params. A
    # config the caller edited (config-invalid fix) gets new params → cache miss
    # → re-evaluated; a config that crashed has no score → re-evaluated; passed
    # configs are reused (not re-run, even after a code fix — fix-forward).
    prev = read_tune_report(args.tune_report_json).get("phase_a", {}).get("warm_start_configs", [])
    cache = {_params_key(t["params"]): t["score"] for t in prev
             if isinstance(t.get("score"), (int, float))}

    report = read_tune_report(args.tune_report_json)
    report["phase_a"] = {
        "warm_start_configs": [],
        # deferred = proposed-but-not-evaluated-now; the deep-tuner evaluates these
        # first (BO enqueue / grid prepend) only if this candidate is promoted.
        "deferred_configs": [{"params": cast_params_to_search_space(dict(d), search_space)}
                             for d in deferred],
        "search_space": search_space_for_json(search_space),
        "status": "running",
    }
    write_tune_report(args.tune_report_json, report)

    started = time.time()
    wsc: list[dict] = []
    for i, raw in enumerate(configs):
        params = cast_params_to_search_space(dict(raw), search_space)
        key = _params_key(params)
        if key in cache:
            wsc.append({"params": params, "score": cache[key]})
        else:
            try:
                score = timed_eval(evaluate, make_model, params, args.candidate_path)
            except Exception as exc:
                tb = traceback.format_exc()
                sys.stderr.write(tb)
                wsc.append({"params": params, "score": None, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}"[:300],
                            "error_traceback": tb})
                report["phase_a"]["warm_start_configs"] = wsc
                report["phase_a"]["status"] = "crashed"
                write_tune_report(args.tune_report_json, report)
                write_json({"phase": "a", "status": "crashed", "crash_index": i,
                            "crash_params": params,
                            "error": f"{type(exc).__name__}: {exc}"[:300]})
                return CRASHED
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
    write_tune_report(args.tune_report_json, report)

    write_json({
        "phase": "a",
        "status": "ok",
        "k_evaluated": len(configs),
        "k_survived": len(wsc),
        "best_warm_score": best_warm_score,
        "best_warm_params": best_params,
        "elapsed_seconds": round(elapsed, 1),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
