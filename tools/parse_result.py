#!/usr/bin/env python3
"""Generic task log parser for autoresearch results.tsv and loop_state.md."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Optional

from validate_tasks import ROOT, parse_task_toml


HEADER = (
    "run_id\tcommit\tmetric\tbaseline_score\tbest_warm_score\t"
    "final_best_score\tbest_model\tstatus\tdescription"
)


def git_commit() -> str:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            return "worktree"
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def infer_task_name(paths: list[Optional[Path]]) -> Optional[str]:
    for path in paths:
        if path is None:
            continue
        parts = path.resolve().parts
        for index, part in enumerate(parts[:-1]):
            if part == "runs" and index + 1 < len(parts):
                return parts[index + 1]
    return None


def load_task_config(task_name: str) -> dict:
    task_toml = ROOT / "tasks" / task_name / "task.toml"
    if not task_toml.exists():
        raise ValueError(f"missing task.toml for task: {task_name}")
    return parse_task_toml(task_toml)


def parse_summary(log_path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    line_re = re.compile(r"^([A-Za-z0-9_]+):\s+(.+?)\s*$")
    for line in log_path.read_text(errors="replace").splitlines():
        match = line_re.match(line)
        if match:
            key, value = match.groups()
            metrics[key] = value
    return metrics


def required_patterns_present(log_path: Path, patterns: list[str]) -> bool:
    text = log_path.read_text(errors="replace")
    return all(re.search(pattern, text, flags=re.MULTILINE) for pattern in patterns)


def primary_value(metrics: dict[str, str], metric: str) -> Optional[float]:
    for key in (metric, "score"):
        raw_value = metrics.get(key)
        if raw_value is None:
            continue
        try:
            return float(raw_value)
        except ValueError:
            continue
    return None


def sanitize_description(description: str) -> str:
    return " ".join(description.replace("\t", " ").split())


def iter_result_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    lines = path.read_text(errors="replace").splitlines()
    if not lines:
        return []
    header = lines[0]
    if header != HEADER:
        raise ValueError(f"{path}: header mismatch; expected {HEADER!r}, got {header!r}")
    columns = header.split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        if len(values) != len(columns):
            continue
        rows.append(dict(zip(columns, values)))
    return rows


def is_improvement(value: float, best_value: Optional[float], lower_is_better: bool) -> bool:
    if best_value is None:
        return True
    if lower_is_better:
        return value < best_value
    return value > best_value


def best_kept_value(path: Path, metric: str, lower_is_better: bool) -> Optional[float]:
    values: list[float] = []
    for row in iter_result_rows(path):
        if row.get("status") != "keep" or row.get("metric") != metric:
            continue
        try:
            values.append(float(row["final_best_score"]))
        except (KeyError, ValueError):
            continue
    if not values:
        return None
    return min(values) if lower_is_better else max(values)


def best_kept_row(
    rows: list[dict[str, str]],
    metric: str,
    lower_is_better: bool,
) -> Optional[dict[str, str]]:
    best_row: Optional[dict[str, str]] = None
    best_value: Optional[float] = None
    for row in rows:
        if row.get("status") != "keep" or row.get("metric") != metric:
            continue
        try:
            value = float(row["final_best_score"])
        except (KeyError, ValueError):
            continue
        if is_improvement(value, best_value, lower_is_better):
            best_value = value
            best_row = row
    return best_row


def next_run_id(rows: list[dict[str, str]]) -> str:
    numeric_ids: list[int] = []
    widths = [3]
    for row in rows:
        run_id = row.get("run_id", "")
        if run_id.isdigit():
            numeric_ids.append(int(run_id))
            widths.append(len(run_id))
    if not numeric_ids:
        return "000"
    return f"{max(numeric_ids) + 1:0{max(widths)}d}"


def infer_tag(results_path: Path) -> str:
    return results_path.parent.name or "unknown"


def candidate_dir_for(
    results_path: Path,
    task_name: str,
    tag: str,
    run_id: str,
    config: dict,
) -> str:
    candidate = config.get("candidate", {})
    if not isinstance(candidate, dict) or not candidate.get("enabled"):
        return "none"
    template = candidate.get("root_template", "runs/{task_name}/{tag}/candidates/{run_id}")
    if not isinstance(template, str):
        return "none"
    try:
        relative = template.format(task_name=task_name, tag=tag, run_id=run_id)
    except KeyError:
        return (results_path.parent / "candidates" / run_id).as_posix()
    return (ROOT / relative).as_posix()


def write_loop_state(results_path: Path, task_name: str, config: dict) -> None:
    rows = iter_result_rows(results_path)
    result = config.get("result", {})
    metric = result.get("metric", "score")
    lower_is_better = bool(result.get("lower_is_better", False))
    tag = infer_tag(results_path)
    last_row = rows[-1] if rows else {}
    best_row = best_kept_row(rows, metric, lower_is_better)

    best_run_id = best_row.get("run_id", "none") if best_row else "none"
    best_score = best_row.get("final_best_score", "none") if best_row else "none"
    best_candidate_dir = "none"
    if best_row:
        best_candidate_dir = candidate_dir_for(results_path, task_name, tag, best_run_id, config)

    candidate = config.get("candidate", {})
    candidate_mode = isinstance(candidate, dict) and bool(candidate.get("enabled"))

    lines = [
        f"task: {task_name}",
        f"tag: {tag}",
        "phase: running",
        f"next_run_id: {next_run_id(rows)}",
        f"best_run_id: {best_run_id}",
        f"best_score: {best_score}",
        f"metric: {metric}",
        f"lower_is_better: {str(lower_is_better).lower()}",
        f"candidate_mode: {str(candidate_mode).lower()}",
        f"best_candidate_dir: {best_candidate_dir}",
        f"last_run_id: {last_row.get('run_id', 'none')}",
        f"last_status: {last_row.get('status', 'none')}",
        f"last_score: {last_row.get('value', 'none')}",
        "active_stop_condition: none",
        f"notes: {last_row.get('description', 'none')}",
    ]
    (results_path.parent / "loop_state.md").write_text("\n".join(lines) + "\n")


def read_tune_scores(args: argparse.Namespace, task_name: str, config: dict) -> tuple[str, str]:
    """Look up baseline_score and best_warm_score from the candidate's
    tune_report.json. Returns ("n/a", "n/a") if the file is missing or
    cannot be parsed.
    """
    if not args.append:
        return ("n/a", "n/a")
    tag = infer_tag(args.append)
    candidate_dir = candidate_dir_for(args.append, task_name, tag, args.run_id, config)
    if candidate_dir == "none":
        return ("n/a", "n/a")
    report_path = Path(candidate_dir) / "tune_report.json"
    if not report_path.exists():
        return ("n/a", "n/a")
    try:
        with open(report_path) as f:
            report = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ("n/a", "n/a")
    phase_a = report.get("phase_a", {})
    baseline = phase_a.get("base_score")
    best_warm = phase_a.get("best_warm_score")
    baseline_str = f"{baseline:.6f}" if isinstance(baseline, (int, float)) else "n/a"
    best_warm_str = f"{best_warm:.6f}" if isinstance(best_warm, (int, float)) else "n/a"
    return (baseline_str, best_warm_str)


def build_row(args: argparse.Namespace, task_name: str, config: dict) -> str:
    result = config.get("result", {})
    metric = result.get("metric", "score")
    lower_is_better = bool(result.get("lower_is_better", False))
    required_patterns = result.get("required_patterns", [])
    if not isinstance(required_patterns, list):
        required_patterns = []

    metrics = parse_summary(args.log_path)
    value_float = primary_value(metrics, metric)
    has_result = (
        value_float is not None
        and required_patterns_present(args.log_path, required_patterns)
    )

    status = args.status
    if status == "auto":
        if not has_result:
            status = "crash"
        else:
            best_value = best_kept_value(args.append, metric, lower_is_better) if args.append else None
            status = "keep" if is_improvement(value_float, best_value, lower_is_better) else "discard"

    final_best = f"{value_float:.6f}" if has_result and value_float is not None else "0.000000"
    baseline_score, best_warm_score = read_tune_scores(args, task_name, config)
    best_model = metrics.get("best_model", metrics.get("candidate", "none"))
    commit = args.commit or git_commit()
    description = sanitize_description(args.description)
    return (
        f"{args.run_id}\t{commit}\t{metric}\t"
        f"{baseline_score}\t{best_warm_score}\t{final_best}\t"
        f"{best_model}\t{status}\t{description}"
    )


def append_row(path: Path, row: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(HEADER + "\n" + row + "\n")
        return
    text = path.read_text(errors="replace")
    iter_result_rows(path)
    prefix = "" if text.endswith("\n") else "\n"
    with path.open("a") as f:
        f.write(prefix + row + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--append", type=Path, help="Append the row to this TSV file.")
    parser.add_argument("--run-id", default="000", help="Run id inside the run directory.")
    parser.add_argument("--commit", help="Override the git commit column.")
    parser.add_argument("--task", help="Task name under tasks/. Inferred from --append when omitted.")
    parser.add_argument(
        "--status",
        choices=["auto", "keep", "discard", "crash"],
        default="auto",
    )
    parser.add_argument("--description", default="experiment", help="Short TSV-safe summary.")
    args = parser.parse_args()

    if not args.log_path.exists():
        parser.error(f"log file does not exist: {args.log_path}")

    task_name = args.task or infer_task_name([args.append, args.log_path])
    if not task_name:
        parser.error("could not infer task name; pass --task <task-name>")

    try:
        config = load_task_config(task_name)
        row = build_row(args, task_name, config)
        if args.append:
            append_row(args.append, row)
            write_loop_state(args.append, task_name, config)
    except ValueError as exc:
        parser.error(str(exc))

    print(HEADER)
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
