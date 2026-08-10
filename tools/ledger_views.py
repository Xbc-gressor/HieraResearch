"""Read-only projections of persisted ledger state.

These helpers may read run configuration and evaluation receipts, but never
write run artifacts.  ``tools/ledger.py`` owns the final ``loop_state.md`` and
``ledger.json`` writes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

from evaluation_budget import budget_status
from ledger_core import (
    best_kept_record,
    experience_refresh_status,
    format_score,
    next_run_id,
)
from run_cfg import read_framework_cfg
from search_space_state import empty_search_space_state, runtime_status_counts
from semantic_evidence import (
    DIRECT_COMPARATOR_CAPABILITY,
    DIRECT_COMPARATOR_CAPABILITY_KEY,
)
from validate_tasks import ROOT


def candidate_dir_for(
    ledger_path: Path,
    task_name: str,
    tag: str,
    run_id: str,
    config: dict,
) -> str:
    candidate = config.get("candidate", {})
    if not isinstance(candidate, dict):
        candidate = {}
    template = candidate.get(
        "root_template", "runs/{task_name}/{tag}/candidates/{run_id}"
    )
    if not isinstance(template, str):
        template = "runs/{task_name}/{tag}/candidates/{run_id}"
    try:
        relative = template.format(task_name=task_name, tag=tag, run_id=run_id)
    except KeyError:
        return (ledger_path.parent / "candidates" / run_id).as_posix()
    return (ROOT / relative).as_posix()


def evaluations_done(data: dict, ledger_path: Path) -> dict:
    """Return all admitted config-to-score attempts from strict receipts."""
    records = data.get("records", [])
    strict = budget_status(Path(ledger_path).parent)
    strict_per = {
        row["run_id"]: row["evals"]
        for row in strict.get("per_candidate", [])
    }
    per_candidate = []
    for record in records:
        run_id = record.get("run_id")
        per_candidate.append(
            {
                "run_id": run_id,
                "evals": strict_per.pop(run_id, 0),
                "tuned": bool(record.get("tune")),
                "status": record.get("status"),
            }
        )
    for run_id, attempted in sorted(strict_per.items()):
        per_candidate.append(
            {
                "run_id": run_id,
                "evals": attempted,
                "tuned": False,
                "status": "pending",
            }
        )
    return {
        "evaluations_done": strict["evaluations_done"],
        "n_candidates": len(records),
        "per_candidate": per_candidate,
    }


def framework_budget(ledger_path: Path) -> Optional[int]:
    cfg_path = Path(ledger_path).parent / "framework_cfg.json"
    if not cfg_path.is_file():
        return None
    config = read_framework_cfg(cfg_path)
    value = config.get("max_evaluations")
    return (
        value
        if isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        else None
    )


def run_phase(
    ledger_path: Path,
    data: dict,
    *,
    budget_override: Optional[int] = None,
) -> tuple[str, str]:
    """Return the truthful derived phase and stop condition for a run."""
    state = data.get("run_state") if isinstance(data.get("run_state"), dict) else {}
    budget = budget_override if budget_override is not None else framework_budget(ledger_path)
    if budget is None:
        saved_budget = state.get("evaluation_budget")
        if isinstance(saved_budget, int) and not isinstance(saved_budget, bool):
            budget = saved_budget
    attempted = evaluations_done(data, ledger_path)["evaluations_done"]
    if state.get("phase") == "blocked":
        return "blocked", str(state.get("active_stop_condition") or "unspecified_blocker")
    if budget is not None and attempted >= budget:
        refresh = experience_refresh_status(data)
        if not refresh["all_records_terminal"]:
            return "running", "budget_reached_pending_resolution"
        if refresh["semantic_admission_blocked"]:
            return "running", "final_experience_refresh_required"
        return "completed", "evaluation_budget_reached"
    return "running", "none"


def render_loop_state(ledger_path: Path, data: dict, config: dict) -> str:
    task_name = data.get("task") or "unknown"
    tag = data.get("tag") or ledger_path.parent.name
    metric = data.get("metric") or "score"
    records = data["records"]
    last = records[-1] if records else {}
    best = best_kept_record(data)
    phase, stop_condition = run_phase(ledger_path, data)
    best_run_id = best.get("run_id", "none") if best else "none"
    best_dir = (
        candidate_dir_for(ledger_path, task_name, tag, best_run_id, config)
        if best
        else "none"
    )
    lines = [
        f"task: {task_name}",
        f"tag: {tag}",
        f"phase: {phase}",
        f"next_run_id: {next_run_id(data)}",
        f"best_run_id: {best_run_id}",
        f"best_score: {format_score(best.get('final_best_score')) if best else 'none'}",
        f"metric: {metric}",
        f"best_candidate_dir: {best_dir}",
        f"last_run_id: {last.get('run_id', 'none')}",
        f"last_status: {last.get('status', 'none')}",
        f"last_score: {format_score(last.get('final_best_score'))}",
        f"active_stop_condition: {stop_condition}",
        f"notes: {last.get('description') or 'none'}",
    ]
    return "\n".join(lines) + "\n"


def brief(data: dict, ledger_path: Path, *, budget_override: int | None = None) -> dict:
    records = data.get("records", [])
    best = best_kept_record(data)
    last = records[-1] if records else None
    attempted = evaluations_done(data, ledger_path)["evaluations_done"]
    budget = budget_override if budget_override is not None else framework_budget(ledger_path)
    refresh = experience_refresh_status(data)
    phase, stop_condition = run_phase(
        ledger_path,
        data,
        budget_override=budget_override,
    )
    experience = data.get("experience") if isinstance(data.get("experience"), dict) else {}
    state = data.get("search_space_state")
    state = state if isinstance(state, dict) else empty_search_space_state()
    counts = runtime_status_counts(state)
    return {
        "task": data.get("task"),
        "tag": data.get("tag"),
        "metric": data.get("metric"),
        "phase": phase,
        "active_stop_condition": stop_condition,
        "next_run_id": next_run_id(data),
        "n_candidates": len(records),
        "status_counts": dict(Counter(record.get("status") for record in records)),
        "op_counts": dict(Counter(record.get("op") for record in records)),
        "pending_run_ids": [
            record.get("run_id")
            for record in records
            if record.get("status") == "pending"
        ],
        "best": None
        if best is None
        else {
            "run_id": best.get("run_id"),
            "score": best.get("final_best_score"),
            "candidate_name": best.get("candidate_name"),
        },
        "last": None
        if last is None
        else {
            "run_id": last.get("run_id"),
            "status": last.get("status"),
            "score": last.get("final_best_score"),
        },
        "evaluations_attempted": attempted,
        "preflight_attempts": sum(
            int(record.get("preflight_attempts") or 0) for record in records
        ),
        "preflight_failures": sum(
            int(record.get("preflight_failures") or 0) for record in records
        ),
        "feasibility_rejections": sum(
            int(record.get("feasibility_rejections") or 0) for record in records
        ),
        "budget": budget,
        "remaining": None if budget is None else max(0, budget - attempted),
        "reached": None if budget is None else attempted >= budget,
        "experience_updated_at_run": experience.get("updated_at_run"),
        "experience_generation": experience.get("generation"),
        "dag_revision": refresh["dag_revision"],
        "experience_dag_revision": refresh["experience_dag_revision"],
        "experience_dag_delta": refresh["experience_dag_delta"],
        "semantic_admission_blocked": refresh["semantic_admission_blocked"],
        "experience_refresh_required": refresh["experience_refresh_required"],
        "direct_comparator_capability": data.get(
            DIRECT_COMPARATOR_CAPABILITY_KEY,
            DIRECT_COMPARATOR_CAPABILITY,
        ),
        "search_space_state_revision": state.get("revision", 0),
        "runtime_deprioritized_dimensions": counts["dimensions"]["deprioritized"],
        "runtime_pruned_dimensions": counts["dimensions"]["pruned"],
        "runtime_deprioritized_hypotheses": counts["hypotheses"]["deprioritized"],
        "runtime_pruned_hypotheses": counts["hypotheses"]["pruned"],
    }
