"""Run lifecycle derivation + compact status output.

Phase semantics ported from the retired harness_watch._run_snapshot:
- blocked iff ledger.run_state.phase == "blocked" (loop_state.md may carry the
  same signal; compact_status folds it in);
- otherwise a configured, exhausted budget means completed ONLY when every
  ledger record is lifecycle-terminal and the experience cursor is current;
- anything else is running.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .roles import REPO_ROOT, ledger_brief

# Mirrors tools/semantic_evidence.LIFECYCLE_TERMINAL_STATUSES; driver/ stays
# self-contained (tools/ is not an importable package for the driver).
LIFECYCLE_TERMINAL_STATUSES = frozenset({"keep", "discard", "crash", "unevaluated"})


def _integer(value) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _attempted(records: list[dict], evaluations_done: int) -> int:
    # _run_snapshot: per-record trials sum, floored by the strict budget
    # counter (evaluation_budget.py status -> evaluations_done).
    total = 0
    for record in records:
        value = record.get("trials_attempted")
        if value is None:
            value = record.get("trials_completed")
        if value is None:
            value = record.get("warm_start_K")
        total += _integer(value)
    return max(total, evaluations_done)


def _derive_state(ledger: dict | None, framework_cfg: dict,
                  evaluations_done: int) -> tuple[str, str]:
    """Returns (phase, stop_condition) — the full _run_snapshot semantics."""
    ledger = ledger if isinstance(ledger, dict) else {}
    stored = ledger.get("run_state") if isinstance(ledger.get("run_state"), dict) else {}
    if stored.get("phase") == "blocked":
        return "blocked", stored.get("active_stop_condition") or "none"
    raw_records = ledger.get("records", [])
    records = (
        [r for r in raw_records if isinstance(r, dict)]
        if isinstance(raw_records, list) else []
    )
    budget = framework_cfg.get("max_evaluations")
    if not _is_int(budget):
        budget = stored.get("evaluation_budget")
    budget = budget if _is_int(budget) else None
    if budget is not None and _attempted(records, evaluations_done) >= budget:
        lifecycle_terminal = bool(records) and all(
            record.get("status") in LIFECYCLE_TERMINAL_STATUSES
            for record in records
        )
        dag_revision = ledger.get("dag_revision", 0)
        experience = (
            ledger.get("experience")
            if isinstance(ledger.get("experience"), dict) else {}
        )
        experience_cursor = experience.get("dag_revision", 0)
        stale_experience = (
            _is_int(dag_revision)
            and _is_int(experience_cursor)
            and dag_revision > experience_cursor
        )
        if not lifecycle_terminal:
            return "running", "budget_reached_pending_resolution"
        if stale_experience:
            return "running", "final_experience_refresh_required"
        return "completed", "evaluation_budget_reached"
    return "running", "none"


def derive_phase(ledger: dict | None, framework_cfg: dict,
                 evaluations_done: int) -> str:
    return _derive_state(ledger, framework_cfg, evaluations_done)[0]


def budget_status(run_dir: Path, repo_root: Path = REPO_ROOT, cmd=None) -> dict:
    """Wraps `evaluation_budget.py status`. Loops pass their injected cmd so
    tests can fake it; production uses the default real subprocess."""
    argv = [sys.executable, "tools/evaluation_budget.py", "status",
            "--run-dir", str(run_dir)]
    if cmd is not None:
        return json.loads(cmd(argv, repo_root).stdout)
    out = subprocess.run(
        argv, cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Same tolerance as _run_snapshot: a half-written file means "no data",
        # not a crashed status report.
        return {}


def _load_loop_state(path: Path) -> dict[str, str]:
    state: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator:
                state[key.strip()] = value.strip()
    return state


def compact_status(task: str, tag: str, run_dir: Path,
                   repo_root: Path = REPO_ROOT, cmd=None) -> dict:
    ledger = _load_json(run_dir / "ledger.json")
    cfg = _load_json(run_dir / "framework_cfg.json")
    state = _load_loop_state(run_dir / "loop_state.md")
    if state.get("phase") == "blocked":
        # _run_snapshot treats loop_state.md's blocked marker like the
        # ledger's run_state; fold it in so derive_phase sees one source.
        ledger = dict(ledger)
        run_state = dict(ledger.get("run_state") or {})
        run_state["phase"] = "blocked"
        if state.get("active_stop_condition"):
            run_state.setdefault("active_stop_condition",
                                 state["active_stop_condition"])
        ledger["run_state"] = run_state
    done = budget_status(run_dir, repo_root, cmd).get("evaluations_done", 0)
    phase, stop_condition = _derive_state(ledger or None, cfg, done)
    status: dict = {
        "task": task,
        "tag": tag,
        "run_dir": str(run_dir),
        "phase": phase,
        "stop_condition": stop_condition,
    }
    if ledger:
        if cmd is not None:
            brief = json.loads(cmd(
                [sys.executable, "tools/ledger.py", "brief",
                 "--ledger", str(run_dir / "ledger.json")],
                repo_root).stdout)
        else:
            brief = ledger_brief(run_dir)
        for key in ("next_run_id", "best_run_id", "best_score", "last_run_id",
                    "last_status", "active_stop_condition"):
            if key in brief:
                status[key] = brief[key]
    return status
