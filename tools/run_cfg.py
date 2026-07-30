"""Shared reading of the run-local `framework_cfg.json`.

That file is the single source of the evaluation budget, the per-evaluation
wall-clock limit, and per-run meta-parameter overrides, and it is designed to
be hand-edited (`init_run.py` invites the user to edit it). A file that exists
but cannot be parsed is therefore a hard error, not "unconfigured": the guards
enforcing budget and timeout must fail fast instead of silently dropping the
limits they exist to enforce. A missing file remains a legitimate "no
overrides configured" state and yields the caller's default.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class RunConfigError(ValueError):
    """framework_cfg.json exists but cannot be read, parsed, or validated."""


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _validate_optional_positive_int(
    config: dict,
    key: str,
    path: Path,
    *,
    label: str | None = None,
) -> None:
    if key not in config or config[key] is None:
        return
    value = config[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RunConfigError(
            f"{path}: {label or key} must be a positive integer or null"
        )


def _validate_positive_int_override(
    config: dict,
    key: str,
    path: Path,
    *,
    label: str | None = None,
) -> None:
    """Validate an integer override whose explicit null is not meaningful."""
    if key not in config:
        return
    value = config[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RunConfigError(f"{path}: {label or key} must be a positive integer")


def _validate_optional_positive_number(config: dict, key: str, path: Path) -> None:
    if key not in config or config[key] is None:
        return
    value = config[key]
    if (
        not _is_finite_number(value)
        or value <= 0
    ):
        raise RunConfigError(f"{path}: {key} must be a positive finite number or null")


def _validate_positive_number_override(config: dict, key: str, path: Path) -> None:
    """Validate a numeric override whose explicit null is not meaningful."""
    if key not in config:
        return
    value = config[key]
    if not _is_finite_number(value) or value <= 0:
        raise RunConfigError(f"{path}: tuner.{key} must be a positive finite number")


def _validate_tuner_config(tuner: dict, path: Path) -> None:
    """Validate every tuner override consumed by deterministic Python code."""
    # These consumers treat null as "use the derived/default value".
    for key in ("K_eval", "n_min", "bo_patience"):
        _validate_optional_positive_int(
            tuner,
            key,
            path,
            label=f"tuner.{key}",
        )
    if (
        tuner.get("K_eval") is not None
        and int(tuner["K_eval"]) < 2
    ):
        raise RunConfigError(
            f"{path}: tuner.K_eval must be at least 2 so a non-fresh "
            "candidate has one selectable row beyond its fidelity control"
        )

    # These consumers call int(value) whenever the key is present, so an
    # explicit null is invalid rather than equivalent to omission.
    for key in (
        "bo_n_trials",
        "bo_patience_cap",
        "bo_patience_floor",
        "deep_tune_per_candidate_cap",
    ):
        _validate_positive_int_override(
            tuner,
            key,
            path,
            label=f"tuner.{key}",
        )

    if "top_percentile" in tuner:
        value = tuner["top_percentile"]
        if (
            not _is_finite_number(value)
            or not 0 <= float(value) < 100
        ):
            raise RunConfigError(
                f"{path}: tuner.top_percentile must be a finite number "
                "in [0, 100)"
            )

    if "deep_tune_budget_fraction" in tuner:
        value = tuner["deep_tune_budget_fraction"]
        if (
            not _is_finite_number(value)
            or not 0 <= float(value) <= 1
        ):
            raise RunConfigError(
                f"{path}: tuner.deep_tune_budget_fraction must be a finite "
                "number in [0, 1]"
            )

    _validate_positive_number_override(
        tuner,
        "deep_tune_time_limit_seconds",
        path,
    )

    # The adaptive rule in bo_search is
    # min(cap, max(floor, round(1.5 * n_dims))). Check the effective pair,
    # including its code defaults, whenever fixed patience is not selected.
    if tuner.get("bo_patience") is None:
        cap = tuner.get("bo_patience_cap", 20)
        floor = tuner.get("bo_patience_floor", 12)
        if floor > cap:
            raise RunConfigError(
                f"{path}: tuner.bo_patience_floor must be less than or equal "
                "to tuner.bo_patience_cap"
            )


def _validate_framework_cfg(config: dict, path: Path) -> None:
    """Validate the hard-limit fields shared by deterministic consumers."""
    _validate_optional_positive_int(config, "max_evaluations", path)
    _validate_optional_positive_number(config, "per_runtime_limit", path)
    _validate_optional_positive_number(config, "preflight_runtime_limit", path)

    tuner = config.get("tuner")
    if tuner is None:
        return
    if not isinstance(tuner, dict):
        raise RunConfigError(f"{path}: tuner must be an object")
    _validate_tuner_config(tuner, path)


def read_framework_cfg(path: Any) -> dict:
    """Parse one framework_cfg.json into a dict.

    Raises RunConfigError when the file is unreadable, is not valid JSON, or
    does not contain a JSON object.
    """
    path = Path(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunConfigError(f"cannot read framework config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunConfigError(f"{path}: framework config must be an object")
    _validate_framework_cfg(value, path)
    return value


def find_framework_cfg(ref_path: Any) -> Path | None:
    """Nearest ancestor (inclusive) of ref_path holding a framework_cfg.json."""
    p = Path(ref_path).resolve()
    for anc in (p, *p.parents):
        cfg = anc / "framework_cfg.json"
        if cfg.is_file():
            return cfg
    return None


def load_run_cfg(ref_path: Any, section: str) -> dict:
    """One section from the nearest framework_cfg.json ({} when none exists)."""
    cfg = find_framework_cfg(ref_path)
    if cfg is None:
        return {}
    return dict(read_framework_cfg(cfg).get(section, {}))
