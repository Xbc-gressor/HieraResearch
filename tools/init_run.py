#!/usr/bin/env python3
"""Initialize a new autoresearch run directory.

Creates the run directory structure and copies the framework_cfg.json template
if it exists, so the user has a local editable config with all framework
hyperparameters documented inline.

Usage:
    python tools/init_run.py <task_name> <tag>
      [--dimension-strategy <strategy>]
      [--llm-intelligence-score <score>]
      [--max-evaluations <count>]
      [--timeout <seconds>]

Example:
    python tools/init_run.py tabular-model-search exp-20260630 \
      --dimension-strategy llm_induced \
      --llm-intelligence-score 70 \
      --max-evaluations 200 \
      --timeout 60
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from semantic_space import DEFAULT_DIMENSION_STRATEGY, DIMENSION_STRATEGIES
from validate_tasks import parse_task_toml
from run_cfg import read_framework_cfg


SEMANTIC_ARTIFACTS = ("dimension_catalog.json", "background.md", "ledger.json")


def _read_framework_config(path: Path) -> dict:
    return read_framework_cfg(path)


def _task_runtime_limit(repo_root: Path, task_name: str) -> float | None:
    """Read the maintained task's default per-evaluation wall-clock limit."""
    task_toml = repo_root / "tasks" / task_name / "task.toml"
    if not task_toml.is_file():
        return None
    config = parse_task_toml(task_toml)
    run = config.get("run")
    if run is None:
        return None
    if not isinstance(run, dict):
        raise ValueError(f"{task_toml}: [run] must be a table")
    value = run.get("timeout_seconds")
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(
            f"{task_toml}: run.timeout_seconds must be a positive finite number"
        )
    return float(value)


def initialize_run(
    repo_root: Path,
    task_name: str,
    tag: str,
    *,
    dimension_strategy: str | None = None,
    llm_intelligence_score: float | None = None,
    max_evaluations: int | None = None,
    per_runtime_limit: float | None = None,
) -> Path:
    repo_root = Path(repo_root).resolve()
    run_dir = repo_root / "runs" / task_name / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir.relative_to(repo_root)}")

    template = repo_root / "tasks" / "framework_cfg.example.json"
    target = run_dir / "framework_cfg.json"
    target_existed = target.exists()
    if template.exists():
        if not target.exists():
            shutil.copy2(template, target)
            print(f"Copied framework config template to {target.relative_to(repo_root)}")
            print("  → Edit this file to override framework behavior for this run.")
        else:
            print(
                "Framework config already exists at "
                f"{target.relative_to(repo_root)}, skipping copy."
            )
    elif not target.exists():
        print(
            f"Template {template.relative_to(repo_root)} not found, "
            "skipping framework_cfg.json copy."
        )
        print("  → Framework will use code defaults.")

    # New runs inherit a task-appropriate limit instead of blindly retaining
    # the generic template's 60 seconds. Existing run-local choices remain
    # untouched, and an explicit --timeout still wins.
    if per_runtime_limit is None and not target_existed:
        per_runtime_limit = _task_runtime_limit(repo_root, task_name)

    if (
        dimension_strategy is not None
        and dimension_strategy not in DIMENSION_STRATEGIES
    ):
        raise ValueError(
            f"dimension strategy must be one of {sorted(DIMENSION_STRATEGIES)}"
        )
    if (
        llm_intelligence_score is not None
        and (
            isinstance(llm_intelligence_score, bool)
            or not isinstance(llm_intelligence_score, (int, float))
            or not math.isfinite(float(llm_intelligence_score))
            or not 0 <= float(llm_intelligence_score) <= 100
        )
    ):
        raise ValueError(
            "llm intelligence score must be a finite number in [0, 100]"
        )
    if (
        max_evaluations is not None
        and (
            not isinstance(max_evaluations, int)
            or isinstance(max_evaluations, bool)
            or max_evaluations <= 0
        )
    ):
        raise ValueError("max_evaluations must be a positive integer")
    if (
        per_runtime_limit is not None
        and (
            isinstance(per_runtime_limit, bool)
            or not isinstance(per_runtime_limit, (int, float))
            or not math.isfinite(per_runtime_limit)
            or per_runtime_limit <= 0
        )
    ):
        raise ValueError("timeout must be a positive number of seconds")

    if (
        dimension_strategy is None
        and llm_intelligence_score is None
        and max_evaluations is None
        and per_runtime_limit is None
    ):
        return run_dir

    config = _read_framework_config(target) if target.exists() else {}
    updates: list[str] = []

    if dimension_strategy is not None:
        section = config.get("space_initialization", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: space_initialization must be an object")
        current = section.get("dimension_strategy", DEFAULT_DIMENSION_STRATEGY)
        if current not in DIMENSION_STRATEGIES:
            raise ValueError(
                f"{target}: space_initialization.dimension_strategy must be one of "
                f"{sorted(DIMENSION_STRATEGIES)}"
            )
        existing_artifacts = [
            name for name in SEMANTIC_ARTIFACTS if (run_dir / name).exists()
        ]
        if current != dimension_strategy and existing_artifacts:
            raise ValueError(
                "cannot change dimension strategy after semantic artifacts exist: "
                + ", ".join(existing_artifacts)
            )
        if current != dimension_strategy or not target.exists():
            config["space_initialization"] = {
                **section,
                "dimension_strategy": dimension_strategy,
            }
            updates.append(f"dimension_strategy={dimension_strategy}")
        else:
            print(f"Dimension strategy already set to {dimension_strategy}.")

    if llm_intelligence_score is not None:
        section = config.get("semantic_search", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: semantic_search must be an object")
        current = section.get("llm_intelligence_score")
        if current is not None and (
            isinstance(current, bool)
            or not isinstance(current, (int, float))
            or not math.isfinite(float(current))
            or not 0 <= float(current) <= 100
        ):
            raise ValueError(
                f"{target}: semantic_search.llm_intelligence_score must be "
                "a finite number in [0, 100]"
            )
        normalized_score: int | float = llm_intelligence_score
        if (
            isinstance(normalized_score, float)
            and normalized_score.is_integer()
        ):
            normalized_score = int(normalized_score)
        existing_artifacts = [
            name for name in SEMANTIC_ARTIFACTS if (run_dir / name).exists()
        ]
        if (
            (current is None or float(current) != float(normalized_score))
            and existing_artifacts
        ):
            raise ValueError(
                "cannot change llm intelligence score after semantic artifacts "
                "exist: "
                + ", ".join(existing_artifacts)
            )
        if current is None or float(current) != float(normalized_score):
            config["semantic_search"] = {
                **section,
                "llm_intelligence_score": normalized_score,
            }
            updates.append(f"llm_intelligence_score={normalized_score}")
        else:
            print(
                "LLM intelligence score already set to "
                f"{normalized_score}."
            )

    if max_evaluations is not None:
        config["max_evaluations"] = max_evaluations
        updates.append(f"max_evaluations={max_evaluations}")

    if per_runtime_limit is not None:
        normalized_limit: int | float = per_runtime_limit
        if isinstance(normalized_limit, float) and normalized_limit.is_integer():
            normalized_limit = int(normalized_limit)
        config["per_runtime_limit"] = normalized_limit
        updates.append(f"per_runtime_limit={normalized_limit}")

    if updates:
        target.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        print(
            f"Set {', '.join(updates)} in "
            f"{target.relative_to(repo_root)}"
        )
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name", help="Task name (e.g., tabular-model-search)")
    parser.add_argument("tag", help="Run tag (e.g., exp-20260630)")
    parser.add_argument(
        "--dimension-strategy",
        choices=sorted(DIMENSION_STRATEGIES),
        help="run search-space initialization strategy (default: catalog_subset)",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        help="global experiment evaluation budget (must be positive)",
    )
    parser.add_argument(
        "--llm-intelligence-score",
        type=float,
        metavar="SCORE",
        help=(
            "LLM intelligence score used by semantic prediction calibration "
            "(finite number in [0, 100])"
        ),
    )
    parser.add_argument(
        "--timeout",
        "--per-runtime-limit",
        dest="per_runtime_limit",
        type=float,
        metavar="SECONDS",
        help="hard wall-clock limit for each evaluation (must be positive)",
    )
    args = parser.parse_args()
    try:
        initialize_run(
            Path.cwd(),
            args.task_name,
            args.tag,
            dimension_strategy=args.dimension_strategy,
            llm_intelligence_score=args.llm_intelligence_score,
            max_evaluations=args.max_evaluations,
            per_runtime_limit=args.per_runtime_limit,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
