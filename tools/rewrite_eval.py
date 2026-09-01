#!/usr/bin/env python3
"""Measure one rewrite-edited candidate at its BASE_PARAMS, in the task env.

The rewrite loop's measurement half: after the editor changes train.py, this
CLI evaluates exactly the edited candidate and reports a three-way outcome on
stdout:

    {"attempt_id": str|null, "score": float|null, "error": str|null,
     "stage": "params"|"eval"}

- unreadable/non-literal BASE_PARAMS -> stage "params", score null + error,
  exit 0, and no budget spent (params are read before timed_eval reserves a
  slot);
- evaluation budget exhausted -> stage "eval", exit 4 (the loop's normal
  stop signal);
- evaluation crash/timeout/non-finite -> stage "eval", score null + error,
  exit 0 (the reservation is already spent; its attempt_id is still
  reported);
- success -> stage "eval", score set, error null.

The evaluation itself always runs in the task's own uv project via
timed_eval(..., python_cmd=...), never in this process; timed_eval writes the
bounded stdout/stderr trace to <candidate>/_traces/<attempt_id>.log itself.

Usage:
    python tools/rewrite_eval.py --candidate <dir>
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tuners"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import EvaluationBudgetExhausted, timed_eval  # noqa: E402
from apply_base_params import _find_base_params  # noqa: E402
from evaluation_budget import ATTEMPT_LOG, find_run_dir  # noqa: E402
from validate_tasks import ROOT, parse_task_toml  # noqa: E402


def read_base_params(train_py_path) -> dict:
    """Return the candidate's module-level BASE_PARAMS as a plain dict.

    Raises ValueError when the assignment is missing, malformed, or not a
    pure literal dict — the loop calls this before reserving any budget.
    """
    train_py_path = Path(train_py_path)
    tree = ast.parse(train_py_path.read_text(encoding="utf-8"))
    try:
        node = _find_base_params(tree)
    except SystemExit as exc:
        raise ValueError(f"{train_py_path}: {exc}") from None
    if node is None:
        raise ValueError(f"{train_py_path}: no module-level BASE_PARAMS assignment")
    if not isinstance(node, ast.Dict):
        raise ValueError(f"{train_py_path}: BASE_PARAMS value is not a dict literal")
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        raise ValueError(
            f"{train_py_path}: BASE_PARAMS is not a pure literal dict"
        ) from None


def _task_name(candidate_path: Path) -> str | None:
    parts = Path(candidate_path).resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "runs" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _task_section(candidate_path: Path, section: str) -> dict:
    name = _task_name(candidate_path)
    if name is None:
        return {}
    task_toml = ROOT / "tasks" / name / "task.toml"
    if not task_toml.is_file():
        return {}
    data = parse_task_toml(task_toml).get(section, {})
    return data if isinstance(data, dict) else {}


def _python_cmd(candidate_path: Path) -> list[str] | None:
    project = _task_section(candidate_path, "env").get("project")
    if not isinstance(project, str) or not project:
        return None
    return ["uv", "--project", str(ROOT / project), "run", "python"]


def _attempt_log_marker(ref_path: Path) -> tuple[Path | None, int]:
    """(attempt log path, rows already present) captured before evaluating.

    timed_eval reserves — and thereby logs — the attempt itself, so the
    before/after line-count delta identifies THIS evaluation's receipt
    without a second reservation. The rewrite loop evaluates serially, which
    makes the newly appended row the log's last line.
    """
    run_dir = find_run_dir(ref_path)
    if run_dir is None:
        return None, 0
    path = run_dir / ATTEMPT_LOG
    try:
        with path.open("r", encoding="utf-8") as handle:
            return path, sum(1 for line in handle if line.strip())
    except OSError:
        return path, 0


def _new_attempt_id(log_path: Path | None, rows_before: int) -> str | None:
    """The attempt_id this evaluation appended, or None when unrecoverable."""
    if log_path is None or not log_path.is_file():
        return None
    try:
        lines = [
            line
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(lines) <= rows_before:
            return None
        attempt_id = json.loads(lines[-1]).get("attempt_id")
    except (OSError, ValueError):
        return None
    return attempt_id if isinstance(attempt_id, str) else None


def _summarize_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if len(text) > 500:
        text = text[:500] + "...[truncated]"
    return text


def _emit(stage, attempt_id, score, error) -> None:
    print(json.dumps({"attempt_id": attempt_id, "score": score,
                      "error": error, "stage": stage}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    train_path = args.candidate / "train.py"
    try:
        params = read_base_params(train_path)
    except (OSError, SyntaxError, ValueError) as exc:
        _emit("params", None, None, str(exc))
        return 0
    log_path, rows_before = _attempt_log_marker(train_path)
    try:
        score = timed_eval(
            None,
            None,
            params,
            train_path,
            phase="rewrite",
            method="rewrite",
            python_cmd=_python_cmd(train_path),
        )
    except EvaluationBudgetExhausted as exc:
        _emit("eval", None, None, str(exc))
        return 4
    except Exception as exc:
        _emit("eval", _new_attempt_id(log_path, rows_before), None,
              _summarize_error(exc))
        return 0
    _emit("eval", _new_attempt_id(log_path, rows_before), score, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
