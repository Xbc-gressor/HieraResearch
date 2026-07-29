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


def _validate_optional_positive_int(config: dict, key: str, path: Path) -> None:
    if key not in config or config[key] is None:
        return
    value = config[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RunConfigError(f"{path}: {key} must be a positive integer or null")


def _validate_optional_positive_number(config: dict, key: str, path: Path) -> None:
    if key not in config or config[key] is None:
        return
    value = config[key]
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise RunConfigError(f"{path}: {key} must be a positive finite number or null")


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
    if "K_eval" in tuner and tuner["K_eval"] is not None:
        value = tuner["K_eval"]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RunConfigError(
                f"{path}: tuner.K_eval must be a positive integer or null"
            )


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
