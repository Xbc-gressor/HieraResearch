#!/usr/bin/env python3
"""LEGACY log parser — NOT called by the current experiment loop.

The S-GoT single `config -> score` model has no run log: a candidate's score is
written straight to `ledger.json` by `tunable-contract-extractor` / `tuner-
orchestrator` (`record-run`), so this script is not invoked. It is kept only so
`validate_tasks` finds the file that `task.toml`'s `result.parser` points at, and
for the legacy `autoresearch-baseline` task (which still scores a full training
run). Original behavior: parse one candidate run log for the score + model name
and hand them to `tools/ledger.py` (which owns the schema, the keep/discard/crash
decision, and the derived `loop_state.md`).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

from ledger import infer_task_name, load_task_config, record_run


def _parse_summary(log_path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    line_re = re.compile(r"^([A-Za-z0-9_]+):\s+(.+?)\s*$")
    for line in log_path.read_text(errors="replace").splitlines():
        match = line_re.match(line)
        if match:
            key, value = match.groups()
            metrics[key] = value
    return metrics


def _required_patterns_present(log_path: Path, patterns: list[str]) -> bool:
    text = log_path.read_text(errors="replace")
    return all(re.search(pattern, text, flags=re.MULTILINE) for pattern in patterns)


def _primary_value(metrics: dict[str, str], metric: str) -> Optional[float]:
    for key in (metric, "score"):
        raw_value = metrics.get(key)
        if raw_value is None:
            continue
        try:
            return float(raw_value)
        except ValueError:
            continue
    return None


def _sanitize_description(description: str) -> str:
    return " ".join(description.replace("\t", " ").split())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--ledger", type=Path, required=True, help="Path to ledger.json.")
    parser.add_argument("--run-id", default="000")
    parser.add_argument("--task", help="Task name; inferred from --ledger/log when omitted.")
    parser.add_argument(
        "--status", choices=["auto", "keep", "discard", "crash"], default="auto"
    )
    parser.add_argument("--description", default="experiment")
    args = parser.parse_args()

    if not args.log_path.exists():
        parser.error(f"log file does not exist: {args.log_path}")

    task_name = args.task or infer_task_name([args.ledger, args.log_path])
    if not task_name:
        parser.error("could not infer task name; pass --task <task-name>")

    try:
        config = load_task_config(task_name)
    except ValueError as exc:
        parser.error(str(exc))

    result = config.get("result", {})
    metric = result.get("metric", "score")
    required_patterns = result.get("required_patterns", [])
    if not isinstance(required_patterns, list):
        required_patterns = []

    metrics = _parse_summary(args.log_path)
    value_float = _primary_value(metrics, metric)
    has_result = value_float is not None and _required_patterns_present(
        args.log_path, required_patterns
    )

    status = args.status
    if status == "auto" and not has_result:
        status = "crash"

    best_model = metrics.get("best_model", metrics.get("candidate"))
    # Keep every extra summary line (e.g. per-dataset scores, fit_seconds) so
    # the experience layer can read component-level signal. Task-agnostic: the
    # values stay as the raw strings the task printed; the consumer interprets
    # packed ones like `dataset_scores`.
    aux_metrics = {
        key: value for key, value in metrics.items()
        if key not in ("metric", "score", "best_model")
    } if has_result else None
    record = record_run(
        args.ledger,
        task_name,
        args.run_id,
        final_best_score=value_float if has_result else None,
        status=status,
        candidate_name=best_model,
        description=_sanitize_description(args.description),
        aux_metrics=aux_metrics,
    )
    print(f"run {record['run_id']}: status={record['status']} "
          f"score={record['final_best_score']} -> {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
