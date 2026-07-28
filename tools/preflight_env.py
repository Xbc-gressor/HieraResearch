#!/usr/bin/env python3
"""Validate one task runtime before any candidate objective evaluation.

The command imports the fixed ``prepare.py``, verifies the declared score
surface, and optionally calls ``evaluation.environment_preflight_fn``.  It
never calls ``score_fn`` and writes an auditable run-level receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import traceback

from validate_tasks import ROOT, parse_task_toml


SCHEMA_VERSION = 1
RECEIPT_NAME = "environment_preflight.json"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_prepare(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("prepare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prepare module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare"] = module
    spec.loader.exec_module(module)
    return module


def run_preflight(task_name: str, run_dir: Path) -> dict:
    task_dir = ROOT / "tasks" / task_name
    task_toml = task_dir / "task.toml"
    prepare_path = task_dir / "prepare.py"
    if not task_toml.is_file():
        raise FileNotFoundError(f"missing task.toml: {task_toml}")
    if not prepare_path.is_file():
        raise FileNotFoundError(f"missing prepare.py: {prepare_path}")
    config = parse_task_toml(task_toml)
    evaluation = config.get("evaluation", {})
    score_name = evaluation.get("score_fn")
    if not isinstance(score_name, str) or not score_name:
        raise ValueError("task.toml evaluation.score_fn must be a non-empty string")

    prepare_module = _load_prepare(prepare_path)
    score_fn = getattr(prepare_module, score_name, None)
    if not callable(score_fn):
        raise RuntimeError(f"prepare.py score function is not callable: {score_name!r}")

    checks = [
        {"name": "task_toml", "status": "ok"},
        {"name": "prepare_import", "status": "ok"},
        {"name": "score_surface", "status": "ok", "symbol": score_name},
    ]
    hook_result = None
    hook_name = evaluation.get("environment_preflight_fn")
    if hook_name is not None:
        if not isinstance(hook_name, str) or not hook_name:
            raise ValueError(
                "task.toml evaluation.environment_preflight_fn must be a non-empty string"
            )
        hook = getattr(prepare_module, hook_name, None)
        if not callable(hook):
            raise RuntimeError(
                f"prepare.py environment preflight is not callable: {hook_name!r}"
            )
        hook_result = hook()
        checks.append(
            {
                "name": "task_environment",
                "status": "ok",
                "symbol": hook_name,
            }
        )
    else:
        checks.append({"name": "task_environment", "status": "not_declared"})

    return {
        "schema_version": SCHEMA_VERSION,
        "task": task_name,
        "status": "ok",
        "objective_calls": 0,
        "checks": checks,
        "hook_result": hook_result,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "contract": {
            "task_toml_sha256": _sha256(task_toml),
            "prepare_sha256": _sha256(prepare_path),
        },
        "run_dir": str(Path(run_dir).resolve()),
    }


def _write_receipt(run_dir: Path, payload: dict) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / RECEIPT_NAME
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = run_preflight(args.task, args.run_dir)
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "task": args.task,
            "status": "failed",
            "objective_calls": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_dir": str(args.run_dir.resolve()),
        }
        receipt = _write_receipt(args.run_dir, payload)
        print(json.dumps({"status": "failed", "receipt": str(receipt), "error": str(exc)}))
        return 3
    receipt = _write_receipt(args.run_dir, payload)
    print(json.dumps({"status": "ok", "receipt": str(receipt), "objective_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
