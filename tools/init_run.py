#!/usr/bin/env python3
"""Initialize a new autoresearch run directory.

Creates the run directory structure and copies the framework_cfg.json template
if it exists, so the user has a local editable config with all framework
hyperparameters documented inline.

Usage:
    python tools/init_run.py <task_name> <tag>

Example:
    python tools/init_run.py tabular-model-search exp-20260630
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name", help="Task name (e.g., tabular-model-search)")
    parser.add_argument("tag", help="Run tag (e.g., exp-20260630)")
    args = parser.parse_args()

    # Create run directory
    run_dir = Path(f"runs/{args.task_name}/{args.tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created run directory: {run_dir}")

    # Copy framework_cfg.json template if it exists
    template = Path("tasks/framework_cfg.example.json")
    target = run_dir / "framework_cfg.json"

    if template.exists():
        if not target.exists():
            shutil.copy2(template, target)
            print(f"Copied framework config template to {target}")
            print("  → Edit this file to override framework hyperparameters (GoT/tuner knobs).")
        else:
            print(f"Framework config already exists at {target}, skipping copy.")
    else:
        print(f"Template {template} not found, skipping framework_cfg.json copy.")
        print("  → Framework will use code defaults.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
