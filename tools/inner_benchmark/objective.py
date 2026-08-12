"""Objective + preflight evaluation path for the inner-tuner benchmark (PLAN §5.1).

THE single evaluation path shared by every benchmark arm and the checkpoint
history re-measurement:

- ``evaluate`` runs the task's ``score_fn(make_model, params)`` in a fresh
  subprocess (production tools/tuners/_eval_one.py mechanics), never
  in-process. Crash / timeout / non-finite / missing result after objective
  start returns ``EvalOutcome(status="crash", score=None)``; mapping that to a
  ``+inf`` raw score is the runner's job, not this module's.
- ``preflight`` runs the task's ``evaluation.preflight_fn`` (no-score runtime
  feasibility hook) in its own fresh subprocess. Any failure — preflight-fn
  exception, subprocess crash, timeout — is ``status="rejected"`` with detail.

This module is accounting-free: no evaluation_attempts.jsonl reservation, no
production budget caps — the benchmark runner does its own B accounting.

Sanctioned deviation from production: score-fn / preflight-fn names arrive
explicitly (from the checkpoint spec) instead of being resolved from
tasks/<task>/task.toml via candidate-path inference — frozen checkpoints are
moved to arbitrary benchmark paths. Everything else reuses production
mechanics: load_candidate_modules (revision-pinned, refuses mid-load
rewrites), _communicate_with_limit (own process group, hard SIGKILL of the
group at the limit), _candidate_execution_revision.

Scores are always lower-is-better. No RNG, no logging framework, no global
state.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tuners"))

import tune_tools  # noqa: E402
from _common import (  # noqa: E402
    DEFAULT_PREFLIGHT_LIMIT,
    _communicate_with_limit,
    is_finite_score,
)

_BENCH_EVAL_ONE = Path(__file__).resolve().parent / "_bench_eval_one.py"
_BENCH_PREFLIGHT_ONE = Path(__file__).resolve().parent / "_bench_preflight_one.py"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def python_cmd_for_project(project: str | None) -> list[str]:
    """Interpreter command prefix for evaluation/preflight subprocesses.

    ``project=None`` (test fixtures) -> ``[sys.executable]``: the cell process
    itself runs in an env that can load the candidate. A real checkpoint
    records ``task.project`` (e.g. ``tasks/autoresearch-baseline``), and the
    subprocess runs under that uv project — production's split
    (``uv --project tasks/<task> run python ...``), so cells can run in the
    repo root env (LLM session deps + optimizer libs) while evaluations get
    the task env (torch & friends).
    """
    if project is None:
        return [sys.executable]
    return ["uv", "--project", str(_REPO_ROOT / project), "run", "python"]


@dataclass(frozen=True)
class EvalOutcome:
    """Result of one objective evaluation. ``score`` is None iff crash."""

    status: str  # "ok" | "crash"
    score: float | None
    detail: str | None  # error class/message tail for logs; None on ok
    elapsed_seconds: float


@dataclass(frozen=True)
class PreflightOutcome:
    """Result of one no-score preflight probe (never consumes budget)."""

    status: str  # "ok" | "rejected"
    detail: str | None  # rejection reason for logs; None on ok


def evaluate(
    candidate_path,
    params: dict,
    *,
    score_fn: str,
    per_runtime_limit: float | None = None,
    python_cmd: list[str] | None = None,
) -> EvalOutcome:
    """Run one objective evaluation of ``params`` in a fresh subprocess.

    ``candidate_path`` is the candidate's train.py path (its directory must
    also contain prepare.py); a directory path is accepted and resolved to
    ``<dir>/train.py``. The candidate execution revision is computed here and
    pinned for the subprocess call, so a candidate mutated mid-evaluation
    fails loudly (crash with detail) instead of evaluating stale code.

    ``per_runtime_limit=None`` means no timeout — the subprocess runs to
    completion (per-evaluation limits are a task-layer choice; this module
    adds no wall-clock defensiveness of its own).

    ``python_cmd`` overrides the subprocess interpreter prefix (see
    ``python_cmd_for_project``); ``None`` means ``[sys.executable]``.
    """
    candidate_path = _train_path(candidate_path)
    started = time.monotonic()

    def finish(status: str, score, detail: str | None) -> EvalOutcome:
        return EvalOutcome(
            status=status,
            score=score,
            detail=detail,
            elapsed_seconds=time.monotonic() - started,
        )

    try:
        revision = tune_tools._candidate_execution_revision(candidate_path)
        out, err, returncode = _communicate_with_limit(
            [
                *(python_cmd or [sys.executable]),
                str(_BENCH_EVAL_ONE),
                str(candidate_path),
                json.dumps(params),
                json.dumps(revision),
                score_fn,
            ],
            limit=per_runtime_limit,
            label="evaluation exceeded per_runtime_limit",
        )
    except Exception as exc:
        return finish("crash", None, f"{type(exc).__name__}: {exc}")

    # Same classification order as production timed_eval: a RESULT line wins
    # over a nonzero exit code.
    for line in out.splitlines():
        if line.startswith("RESULT:"):
            try:
                score = float(line[len("RESULT:"):])
            except ValueError:
                return finish(
                    "crash",
                    None,
                    f"evaluation subprocess printed invalid result: {line!r}",
                )
            if not is_finite_score(score):
                return finish(
                    "crash", None, f"evaluation returned non-finite score: {score!r}"
                )
            return finish("ok", score, None)

    return finish("crash", None, _no_result_message("evaluation", returncode, err))


def preflight(
    candidate_path,
    params: dict,
    *,
    preflight_fn: str,
    per_runtime_limit: float | None = None,
    python_cmd: list[str] | None = None,
) -> PreflightOutcome:
    """Run the task's no-score preflight probe in a fresh subprocess.

    Same argument conventions as ``evaluate``, ``python_cmd`` included. The
    timeout mirrors production read_preflight_limit: ``preflight_runtime_limit``
    is not configurable at this layer, so it is
    ``min(DEFAULT_PREFLIGHT_LIMIT, per_runtime_limit)`` when a limit is given,
    else ``DEFAULT_PREFLIGHT_LIMIT`` (180 s).
    """
    candidate_path = _train_path(candidate_path)
    limit = DEFAULT_PREFLIGHT_LIMIT
    if per_runtime_limit is not None:
        limit = min(DEFAULT_PREFLIGHT_LIMIT, per_runtime_limit)

    try:
        revision = tune_tools._candidate_execution_revision(candidate_path)
        out, err, returncode = _communicate_with_limit(
            [
                *(python_cmd or [sys.executable]),
                str(_BENCH_PREFLIGHT_ONE),
                str(candidate_path),
                json.dumps(params),
                json.dumps(revision),
                preflight_fn,
            ],
            limit=limit,
            label="preflight exceeded preflight_runtime_limit",
        )
    except Exception as exc:
        return PreflightOutcome(
            status="rejected", detail=f"{type(exc).__name__}: {exc}"
        )

    for line in out.splitlines():
        if line.startswith("PREFLIGHT:"):
            try:
                json.loads(line[len("PREFLIGHT:"):])
            except json.JSONDecodeError:
                return PreflightOutcome(
                    status="rejected",
                    detail=f"preflight subprocess printed invalid result: {line!r}",
                )
            return PreflightOutcome(status="ok", detail=None)

    return PreflightOutcome(
        status="rejected",
        detail=_no_result_message("preflight", returncode, err),
    )


def _train_path(candidate_path) -> Path:
    """Accept a candidate train.py path or its directory (brief API wording)."""
    path = Path(candidate_path)
    return path / "train.py" if path.is_dir() else path


def _no_result_message(kind: str, returncode: int, err: str) -> str:
    """Production timed_eval/timed_preflight no-result message, verbatim."""
    detail = err.strip()
    if len(detail) > 4000:
        detail = "...[stderr truncated]...\n" + detail[-4000:]
    marker = "RESULT" if kind == "evaluation" else "PREFLIGHT"
    message = (
        f"{kind} subprocess exited with code {returncode} "
        f"without a {marker} line"
    )
    if detail:
        message += f"\nchild stderr:\n{detail}"
    return message
