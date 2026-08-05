#!/usr/bin/env python3
"""Run a standalone candidate's task-declared no-score preflight.

This is the hillclimb counterpart to the tuners' preflight path.  It reads the
params used by the standalone entrypoint, then delegates to the same isolated,
timeout-bounded task hook.  It never reserves an objective slot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


TUNER_DIR = Path(__file__).resolve().parent / "tuners"
sys.path.insert(0, str(TUNER_DIR))

from _common import (  # noqa: E402
    _configured_preflight_name,
    cast_params_to_search_space,
    load_candidate_modules,
    timed_preflight,
)
from warmstart_eval import (  # noqa: E402
    _validated_warm_configs,
    validate_provided_baseline_configs,
)
from tune_tools import (  # noqa: E402
    PARAMETER_TRANSFER_FILENAME,
    validate_parameter_transfer,
)


PARAM_NAMES = ("DEFAULT_PARAMS", "BASE_PARAMS", "PARAMS")


class CandidatePreflightRejected(ValueError):
    """The frozen candidate/config contract failed before no-score execution."""


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


def preflight_warm_configs(
    candidate_path: Path,
    configs_path: Path,
    *,
    k_eval: int | None,
) -> dict:
    """Validate and preflight the exact authored configs without scoring."""
    try:
        configs = json.loads(configs_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CandidatePreflightRejected(
            f"invalid warm configs {configs_path}: {exc}"
        ) from exc
    if not isinstance(configs, list) or not configs:
        raise CandidatePreflightRejected(
            "warm configs must be a non-empty JSON list"
        )
    try:
        control = validate_provided_baseline_configs(
            candidate_path, configs, k_eval
        )
    except (TypeError, ValueError, SyntaxError) as exc:
        raise CandidatePreflightRejected(str(exc)) from exc
    if control["requires_parameter_transfer"]:
        receipt_path = candidate_path.parent / PARAMETER_TRANSFER_FILENAME
        try:
            transfer = json.loads(receipt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CandidatePreflightRejected(
                f"invalid parameter-transfer receipt {receipt_path}: {exc}"
            ) from exc
        try:
            validate_parameter_transfer(candidate_path, configs, transfer)
        except (TypeError, ValueError, SyntaxError) as exc:
            raise CandidatePreflightRejected(str(exc)) from exc
    try:
        configs, search_space = _validated_warm_configs(candidate_path, configs)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise CandidatePreflightRejected(str(exc)) from exc
    if _configured_preflight_name(candidate_path) is None:
        return {
            "ok": True,
            "errors": [],
            "status": "not_declared",
            "objective_calls": 0,
            "configs_checked": len(configs),
            "attempts": [],
        }

    attempts = []
    for index, raw_params in enumerate(configs):
        params = cast_params_to_search_space(dict(raw_params), search_space)
        result = timed_preflight(params, candidate_path)
        attempts.append(
            {
                "index": index,
                "params": params,
                "status": "ok",
                "task_status": (
                    result.get("status")
                    if isinstance(result, dict) and isinstance(result.get("status"), str)
                    else None
                ),
            }
        )
    return {
        "ok": True,
        "errors": [],
        "status": "ok",
        "objective_calls": 0,
        "configs_checked": len(configs),
        "attempts": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument(
        "--configs-json",
        type=Path,
        help="preflight every exact authored warm config instead of standalone params",
    )
    parser.add_argument(
        "--k-eval",
        type=int,
        help="official warm-evaluation count, used by provided-control validation",
    )
    args = parser.parse_args()
    candidate_path = args.candidate_path.resolve()
    if not candidate_path.is_file():
        parser.error(f"candidate does not exist: {candidate_path}")

    try:
        if args.configs_json is not None:
            payload = preflight_warm_configs(
                candidate_path,
                args.configs_json.resolve(),
                k_eval=args.k_eval,
            )
            print(json.dumps(payload, default=str))
            return 0
        if args.k_eval is not None:
            parser.error("--k-eval requires --configs-json")
        if _configured_preflight_name(candidate_path) is None:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "errors": [],
                        "status": "not_declared",
                        "objective_calls": 0,
                    }
                )
            )
            return 0
        try:
            params_name, params = read_standalone_params(candidate_path)
        except (TypeError, ValueError, SyntaxError) as exc:
            raise CandidatePreflightRejected(str(exc)) from exc
        result = timed_preflight(params, candidate_path)
    except CandidatePreflightRejected as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "ok": False,
                    "failure_kind": "candidate_preflight_validation",
                    "errors": [str(exc)],
                    "objective_calls": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                default=str,
            ),
            file=sys.stderr,
        )
        return 1
    except TimeoutError as exc:
        # A no-score preflight timeout is side-effect-free and consumes no
        # objective slot.  Type it explicitly so the caller may apply its
        # bounded-retry policy instead of treating it as an opaque tool
        # failure.
        print(
            json.dumps(
                {
                    "status": "operational_failure",
                    "ok": False,
                    "failure_kind": "preflight_timeout",
                    "errors": [str(exc)],
                    "objective_calls": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                default=str,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "operational_failure",
                    "ok": False,
                    "errors": [str(exc)],
                    "objective_calls": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                default=str,
            ),
            file=sys.stderr,
        )
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "errors": [],
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
