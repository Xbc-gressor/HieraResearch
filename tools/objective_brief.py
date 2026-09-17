#!/usr/bin/env python3
"""Objective brief: the single generator of the task's objective/target view.

Every role that proposes, implements, or tunes (outer loops, judged slate,
inner tuner arms) consumes the same bounded view of the task's metric and
its declared aspirational target, so no loop parses task.toml on its own.
The target is an anti-slop aspiration bar: it raises ambition and never
enters keep/revert thresholds, attribution, budget reservation, crash
adjudication, or scheduler decisions.

The target resolves from ``[result]`` in one place: the ``TARGET_TIER``
environment variable (per-run E2E override), then the ``target_tier``
selector over the ``target_tiers`` table, then the plain ``target_score``.
The brief always carries a single number plus the source it came from.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIEF_FILENAME = "objective_brief.json"
TARGET_SOURCE = "task.toml[result].target_score"
TIER_ENV_VAR = "TARGET_TIER"
TARGET_SEMANTICS = "anti_slop_aspiration"

SEMANTICS_TEXT = (
    "an ambition bar for proposing and tuning, not an official score, a "
    "comparability proof, or a stop line: pursue it with mechanisms that "
    "really move the metric; when it is missing, far away, or already met, "
    "keep making the most promising verifiable improvement"
)


def _is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _finite_or_none(value) -> float | None:
    return float(value) if _is_finite_number(value) else None


def _tiers_table(result_section: dict) -> dict | None:
    """``[result].target_tiers`` validated as a finite-number table."""
    tiers = result_section.get("target_tiers")
    if tiers is None:
        return None
    if not isinstance(tiers, dict) or not tiers:
        raise ValueError(
            "task.toml [result].target_tiers must be a non-empty table"
        )
    for name, value in tiers.items():
        if not _is_finite_number(value):
            raise ValueError(
                f"task.toml [result].target_tiers.{name} must be a finite number"
            )
    return tiers


def resolve_target(result_section: dict | None) -> tuple[float | None, str | None]:
    """The active aspiration target as ``(value, source)``; ``(None, None)`` unset.

    Resolution order: the ``TARGET_TIER`` environment variable (per-run
    override), then ``[result].target_tier`` over ``[result].target_tiers``,
    then the plain ``[result].target_score``. A declared tiers table without
    a resolvable selector is a contract error, never a silent fallback.
    """
    if result_section is None:
        return None, None
    if not isinstance(result_section, dict):
        raise ValueError("task.toml [result] must be a table")
    tiers = _tiers_table(result_section)
    env_tier = os.environ.get(TIER_ENV_VAR) or None
    if tiers is None:
        if env_tier:
            raise ValueError(
                f"{TIER_ENV_VAR}={env_tier} but task.toml declares no "
                "[result].target_tiers"
            )
        value = result_section.get("target_score")
        if value is None:
            return None, None
        if not _is_finite_number(value):
            raise ValueError(
                "task.toml [result].target_score must be a finite number"
            )
        return float(value), TARGET_SOURCE
    tier = env_tier or result_section.get("target_tier")
    if not isinstance(tier, str) or not tier:
        raise ValueError(
            "task.toml declares [result].target_tiers but sets no "
            "[result].target_tier selector"
        )
    if tier not in tiers:
        raise ValueError(
            f"target tier {tier!r} not in task.toml [result].target_tiers "
            f"(available: {sorted(tiers)})"
        )
    source = f"task.toml[result].target_tiers.{tier}"
    if env_tier:
        source += f" (env {TIER_ENV_VAR})"
    return float(tiers[tier]), source


def validated_target(result_section: dict | None) -> float | None:
    """The active aspiration target as a finite float; None when unset.

    Same finite-number rule as validate_tasks (bools and non-finite values
    are contract errors, not silently dropped).
    """
    return resolve_target(result_section)[0]


def build_brief(task_toml: dict, *, baseline_score=None, run_best=None,
                current_best=None) -> dict:
    """The objective brief from an already-parsed task.toml mapping.

    Dynamic scores (baseline/run best/current best) are optional caller
    inputs; scores are always lower-is-better.
    """
    result = task_toml.get("result") if isinstance(task_toml, dict) else None
    result = result if isinstance(result, dict) else {}
    metric = result.get("metric")
    target, target_source = resolve_target(result)
    return {
        "schema_version": 1,
        "metric": metric if isinstance(metric, str) and metric else None,
        "direction": "minimize",
        "aspirational_target_score": target,
        "target_source": target_source if target is not None else None,
        "target_semantics": TARGET_SEMANTICS if target is not None else None,
        "baseline_score": _finite_or_none(baseline_score),
        "run_best": _finite_or_none(run_best),
        "current_best": _finite_or_none(current_best),
    }


def compact_line(brief: dict) -> str:
    """One bounded line for invocation extras (key: value context lines)."""
    metric = brief.get("metric") or "score"
    parts = [f"minimize {metric} (lower is better)"]
    target = brief.get("aspirational_target_score")
    if target is not None:
        parts.append(
            f"aspirational target {target} ({brief.get('target_source')}; "
            "an ambition bar, not an official score or a stop line)"
        )
    else:
        parts.append("no declared target — keep improving against the "
                     "incumbent/run best")
    run_best = brief.get("run_best")
    if run_best is not None:
        parts.append(f"run_best {run_best}")
    return "; ".join(parts)


def render_block(brief: dict) -> str:
    """The bounded text block used as a fixed payload prefix."""
    lines = [
        "## Objective",
        f"metric: {brief.get('metric') or 'unknown'} "
        "(lower is better; the framework's lower-is-better convention)",
    ]
    target = brief.get("aspirational_target_score")
    if target is not None:
        lines.extend([
            f"aspirational_target_score: {target}",
            f"target_source: {brief.get('target_source')}",
            f"target_semantics: {brief.get('target_semantics')} — "
            + SEMANTICS_TEXT,
        ])
    else:
        lines.append(
            "aspirational_target_score: none declared — the bar is the "
            "current best; keep improving it"
        )
    for key in ("baseline_score", "run_best", "current_best"):
        value = brief.get(key)
        if value is not None:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def gap_to_target(brief: dict, score) -> float | None:
    """Lower-is-better gap: how far ``score`` still sits above the target."""
    target = brief.get("aspirational_target_score")
    if target is None or not _is_finite_number(score):
        return None
    return float(score) - float(target)


def ensure_brief(run_dir: Path, task_toml: dict) -> dict:
    """Load the run's objective_brief.json, writing it once when absent.

    The stored brief is the declared (static) view plus its generation
    time; dynamic scores are call-time inputs, never persisted here.
    """
    path = Path(run_dir) / BRIEF_FILENAME
    if path.is_file():
        try:
            brief = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            brief = None
        if isinstance(brief, dict) and brief.get("schema_version") == 1:
            return brief
    brief = build_brief(task_toml)
    brief["generated_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(brief, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return brief


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", help="task name under tasks/")
    group.add_argument("--task-toml", type=Path, help="path to a task.toml")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args()

    path = args.task_toml if args.task_toml else ROOT / "tasks" / args.task / "task.toml"
    try:
        with path.open("rb") as fh:
            task_toml = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 1
    try:
        brief = build_brief(task_toml)
    except ValueError as exc:
        print(f"ERROR: {path}: {exc}", file=sys.stderr)
        return 1
    brief.pop("generated_at", None)
    if args.format == "text":
        print(render_block(brief))
    else:
        print(json.dumps(brief, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
