#!/usr/bin/env python3
"""Phase A warm-start evaluator.

Reads a JSON file containing K candidate hyperparameter configs proposed by
the hyperparam-tuner-llm skill, plus the candidate's BASE_PARAMS for the
baseline reference, and evaluates each on the test split via
prepare.test_score_for_tuning. Persists results to tune_report.json
incrementally (one append per evaluated config).

Invoked by the tuner-orchestrator agent in Phase A. The orchestrator passes
the candidate path, the path to a JSON file with the K proposed configs,
and the path to tune_report.json (which it has already created with an
empty phase_a section).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    append_warmstart_trial,
    cast_params_to_search_space,
    load_candidate_modules,
    read_tune_report,
    search_space_for_json,
    write_json,
    write_tune_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--configs-json", required=True, type=Path)
    parser.add_argument("--tune-report-json", required=True, type=Path)
    args = parser.parse_args()

    train_module, prepare_module = load_candidate_modules(args.candidate_path)
    base_params = dict(train_module.BASE_PARAMS)
    search_space = train_module.SEARCH_SPACE
    make_model = train_module.make_model
    evaluate = prepare_module.evaluate_config_for_tuning

    with open(args.configs_json) as f:
        proposed_configs = json.load(f)

    started = time.time()

    base_score = evaluate(make_model, base_params)
    report = read_tune_report(args.tune_report_json)
    phase_a = report.setdefault("phase_a", {"warm_start_configs": []})
    phase_a["base_score"] = base_score
    phase_a["base_params"] = cast_params_to_search_space(base_params, search_space)
    phase_a["search_space"] = search_space_for_json(search_space)
    write_tune_report(args.tune_report_json, report)

    warm_scores = []
    for raw_config in proposed_configs:
        params = cast_params_to_search_space(dict(raw_config), search_space)
        score = evaluate(make_model, params)
        trial = {"params": params, "score": score}
        append_warmstart_trial(args.tune_report_json, trial)
        warm_scores.append(score)

    best_warm_score = max(warm_scores)
    best_warm_idx = warm_scores.index(best_warm_score)
    best_warm = {
        "params": cast_params_to_search_space(
            dict(proposed_configs[best_warm_idx]), search_space
        ),
        "score": best_warm_score,
    }

    report = read_tune_report(args.tune_report_json)
    report["phase_a"]["best_warm_score"] = best_warm_score
    report["phase_a"]["best_warm_params"] = best_warm["params"]
    write_tune_report(args.tune_report_json, report)

    elapsed = time.time() - started

    write_json({
        "phase": "a",
        "status": "ok",
        "base_score": base_score,
        "warm_scores": warm_scores,
        "best_warm_score": best_warm_score,
        "best_warm_params": best_warm["params"],
        "k_configs_evaluated": len(proposed_configs),
        "elapsed_seconds": round(elapsed, 1),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
