#!/usr/bin/env python3
"""Initialize a new autoresearch run directory.

Creates the run directory structure and copies the framework_cfg.json template
if it exists, so the user has a local editable config with all framework
hyperparameters documented inline.

Usage:
    python tools/init_run.py <task_name> <tag> [--dimension-strategy <strategy>]

Example:
    python tools/init_run.py tabular-model-search exp-20260630 \
      --dimension-strategy llm_induced
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from semantic_space import DEFAULT_DIMENSION_STRATEGY, DIMENSION_STRATEGIES


SEMANTIC_ARTIFACTS = ("dimension_catalog.json", "background.md", "ledger.json")


def _read_framework_config(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read framework config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: framework config must be an object")
    return value


def initialize_run(
    repo_root: Path,
    task_name: str,
    tag: str,
    *,
    dimension_strategy: str | None = None,
) -> Path:
    repo_root = Path(repo_root).resolve()
    run_dir = repo_root / "runs" / task_name / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir.relative_to(repo_root)}")

    template = repo_root / "tasks" / "framework_cfg.example.json"
    target = run_dir / "framework_cfg.json"
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

    if dimension_strategy is None:
        return run_dir
    if dimension_strategy not in DIMENSION_STRATEGIES:
        raise ValueError(
            f"dimension strategy must be one of {sorted(DIMENSION_STRATEGIES)}"
        )

    config = _read_framework_config(target) if target.exists() else {}
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
    if current == dimension_strategy and target.exists():
        print(f"Dimension strategy already set to {dimension_strategy}.")
        return run_dir

    config["space_initialization"] = {
        **section,
        "dimension_strategy": dimension_strategy,
    }
    target.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Set dimension strategy to {dimension_strategy} in "
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
    args = parser.parse_args()
    try:
        initialize_run(
            Path.cwd(),
            args.task_name,
            args.tag,
            dimension_strategy=args.dimension_strategy,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
