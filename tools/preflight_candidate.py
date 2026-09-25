#!/usr/bin/env python3
"""Run a standalone candidate's task-declared no-score preflight.

This is the hillclimb counterpart to the tuners' preflight path.  It reads the
params used by the standalone entrypoint, then delegates to the same isolated,
timeout-bounded task hook.  It never reserves an objective slot.

Exit codes: 0 ok / not declared, 3 preflight failed, 4 run budget exhausted,
5 the open round's quota exhausted (``--round-quota-exempt`` skips that gate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


TUNER_DIR = Path(__file__).resolve().parent / "tuners"
sys.path.insert(0, str(TUNER_DIR))

from _common import (  # noqa: E402
    EvaluationBudgetExhausted,
    _configured_preflight_name,
    load_candidate_modules,
    timed_preflight,
)


PARAM_NAMES = ("DEFAULT_PARAMS", "BASE_PARAMS", "PARAMS")


def read_standalone_params(candidate_path: Path) -> tuple[str, dict]:
    """Return the first standalone params mapping declared by a candidate."""
    train_module, _ = load_candidate_modules(
        candidate_path,
        required_symbols=("make_model",),
    )
    for name in PARAM_NAMES:
        if not hasattr(train_module, name):
            continue
        value = getattr(train_module, name)
        if not isinstance(value, dict):
            raise TypeError(f"{candidate_path}: {name} must be a dict")
        return name, dict(value)
    raise ValueError(
        f"{candidate_path}: no standalone params mapping found "
        f"(expected one of {', '.join(PARAM_NAMES)})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--round-quota-exempt", action="store_true")
    args = parser.parse_args()
    candidate_path = args.candidate_path.resolve()
    if not candidate_path.is_file():
        parser.error(f"candidate does not exist: {candidate_path}")

    if _configured_preflight_name(candidate_path) is None:
        print(json.dumps({"status": "not_declared", "objective_calls": 0}))
        return 0

    try:
        params_name, params = read_standalone_params(candidate_path)
        result = timed_preflight(
            params, candidate_path, round_quota_exempt=args.round_quota_exempt
        )
    except EvaluationBudgetExhausted as exc:
        print(json.dumps({"status": "budget_exhausted", "scope": exc.scope, "objective_calls": 0}))
        return 5 if exc.scope == "round_quota" else 4
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "objective_calls": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                default=str,
            ),
            file=sys.stderr,
        )
        return 3

    print(
        json.dumps(
            {
                "status": "ok",
                "objective_calls": 0,
                "params_source": params_name,
                "result": result or {"status": "ok"},
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
