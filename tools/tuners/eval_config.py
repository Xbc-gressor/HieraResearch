#!/usr/bin/env python3
"""Evaluate one diagnostic config through the framework's capped evaluator.

This is the only supported manual diagnostic surface. It inherits
``per_runtime_limit`` from the run's framework_cfg.json, emits one structured
JSON result, and uses the same process-group cleanup as every tuner.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    EvaluationFailure,
    cast_params_to_search_space,
    load_candidate_modules,
    resolve_score_fn,
    timed_eval,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--params-json", type=Path, help="path to one parameter dict")
    group.add_argument("--params", help="one parameter dict as JSON")
    args = parser.parse_args()

    raw = json.loads(args.params_json.read_text() if args.params_json else args.params)
    if not isinstance(raw, dict):
        parser.error("diagnostic params must be a JSON object")

    train_module, prepare_module = load_candidate_modules(args.candidate_path)
    params = cast_params_to_search_space(raw, train_module.SEARCH_SPACE)
    evaluate = resolve_score_fn(prepare_module, args.candidate_path)
    started = time.monotonic()
    try:
        score = timed_eval(evaluate, train_module.make_model, params, args.candidate_path)
    except Exception as exc:
        write_json({
            "status": exc.status if isinstance(exc, EvaluationFailure) else "error",
            "score": None,
            "params": params,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.monotonic() - started, 1),
        })
        return 3
    write_json({
        "status": "ok",
        "score": score,
        "params": params,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
