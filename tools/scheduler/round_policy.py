"""Round scheduler ``round_v1``: candidate-count-triggered optimization rounds.

Every ``N`` newly produced candidates (the first cycle waits for the whole
seed set) the driver runs one optimization phase over the full candidate
pool: ``r`` rewrite bouts, then ``t`` tune bouts. The budget is wall clock:
a run-level ``deadline`` plus a per-phase quota ``round_seconds``. This
module owns the deterministic parts — the cycle state under
``<run_dir>/.scheduler/round_state.json``, the generation/optimization
switch, and both eligibility rankings; ``driver/loops`` only sequences.

Rewrite eligibility: finite reference score, no in-flight tuning bout,
under the rewrite bout cap, not stalled, top-``k`` by current best. Among
those the candidate with the fewest rewrite bouts (then the best score)
gets the bout, so attention rotates inside the top-``k``.

Tune eligibility: ``contract.eligible()`` plus a tune report that matches
the candidate on disk (a kept rewrite rebases the report; a failed rebase
leaves the candidate untunable rather than crashing the tuner). Ranking is
expected improvement per unit time: ``1 / (1 + score_rank)`` divided by the
bout's expected seconds, so a slower candidate needs a better rank to win.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from .policy import Decision
from .state import SchedulerState, candidate_score
from .store import STORE_DIRNAME

POLICY_ID = "round_v1"
POLICY_VERSION = "scheduler-round-v1"
ROUND_STATE_FILENAME = "round_state.json"
REWRITE = "REWRITE"
OVERHEAD_WINDOW = 20

DEFAULTS: dict = {
    "new_candidates": 4,
    "rewrite_bouts": 1,
    "tune_bouts": 1,
    "round_seconds": None,
    "rewrite_top_k": 3,
    "rewrite_max_bouts": 12,
    "rewrite_stall_after": 5,
    "noise_margin": 0.0,
    "session_overhead_seconds": 600.0,
}


# --- configuration and cycle state -------------------------------------------


def load_config(run_dir: Path) -> dict:
    from run_cfg import load_run_cfg

    section = load_run_cfg(Path(run_dir), "round")
    config = dict(DEFAULTS)
    for key, value in section.items():
        if key in config:
            config[key] = value
    return config


def _state_path(run_dir: Path) -> Path:
    return Path(run_dir) / STORE_DIRNAME / ROUND_STATE_FILENAME


def load_round_state(run_dir: Path) -> dict:
    path = _state_path(run_dir)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "kind": "round_state",
        "cycle": 0,
        "cycle_start_count": 0,
        "phase": "generate",
        "phase_started_at": None,
        "phase_deadline": None,
        "overhead_seconds": {"rewrite": [], "tune": []},
    }


def save_round_state(run_dir: Path, state: dict) -> None:
    path = _state_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def finite_candidate_count(ledger: dict) -> int:
    """Admitted candidates with a finite score; crashed slate seats do not count."""
    return sum(
        1
        for record in ledger.get("records", [])
        if record.get("status") != "crash" and candidate_score(record) is not None
    )


def generation_threshold(config: dict, state: dict, n_seed: int) -> int:
    """First cycle: the whole seed set; later cycles: ``N`` new candidates."""
    if int(state.get("cycle", 0)) == 0 and n_seed > 0:
        return int(n_seed)
    return int(config["new_candidates"])


def round_status(run_dir: Path, ledger: dict, *, now: float | None = None) -> dict:
    """The driver's generate-or-optimize switch for this iteration."""
    from evaluation_budget import time_budget
    from .state import seed_quota

    run_dir = Path(run_dir)
    config = load_config(run_dir)
    state = load_round_state(run_dir)
    count = finite_candidate_count(ledger)
    produced = count - int(state.get("cycle_start_count", 0))
    threshold = generation_threshold(config, state, seed_quota(run_dir))
    usable = time_budget(run_dir, now=now)["usable_seconds"]
    quota = config["round_seconds"]
    # Generation must not eat into the next optimization round's quota.
    final_round = usable is not None and quota is not None and usable <= quota
    return {
        "config": config,
        "state": state,
        "finite_candidates": count,
        "produced": produced,
        "threshold": threshold,
        "usable_seconds": usable,
        "final_round": final_round,
        "generate": produced < threshold and not final_round,
    }


def begin_optimization(run_dir: Path, *, now: float | None = None) -> dict:
    from evaluation_budget import time_budget

    run_dir = Path(run_dir)
    now = time.time() if now is None else now
    config = load_config(run_dir)
    state = load_round_state(run_dir)
    quota = config["round_seconds"]
    usable = time_budget(run_dir, now=now)["usable_seconds"]
    if usable is not None:
        quota = usable if quota is None else min(quota, usable)
    state["phase"] = "optimize"
    state["phase_started_at"] = now
    state["phase_deadline"] = None if quota is None else now + float(quota)
    save_round_state(run_dir, state)
    return state


def end_optimization(run_dir: Path, ledger: dict) -> dict:
    run_dir = Path(run_dir)
    state = load_round_state(run_dir)
    state["cycle"] = int(state.get("cycle", 0)) + 1
    state["cycle_start_count"] = finite_candidate_count(ledger)
    state["phase"] = "generate"
    state["phase_started_at"] = None
    state["phase_deadline"] = None
    save_round_state(run_dir, state)
    return state


def record_overhead(run_dir: Path, kind: str, seconds: float) -> dict:
    """Observed non-evaluation seconds of one bout (session + tooling)."""
    run_dir = Path(run_dir)
    state = load_round_state(run_dir)
    window = state.setdefault("overhead_seconds", {}).setdefault(kind, [])
    window.append(round(max(0.0, float(seconds)), 1))
    del window[:-OVERHEAD_WINDOW]
    save_round_state(run_dir, state)
    return state


def overhead_estimate(config: dict, state: dict, kind: str) -> float:
    observed = (state.get("overhead_seconds") or {}).get(kind) or []
    if observed:
        return sum(observed) / len(observed)
    return float(config["session_overhead_seconds"])


def quota_remaining(state: dict, *, now: float | None = None) -> float | None:
    deadline = state.get("phase_deadline")
    if state.get("phase") != "optimize" or deadline is None:
        return None
    return float(deadline) - (time.time() if now is None else now)


# --- per-candidate facts -----------------------------------------------------


def _eval_seconds(run_dir: Path) -> tuple[dict[str, float], float | None]:
    from evaluation_budget import budget_status

    status = budget_status(Path(run_dir))
    per_candidate = {
        row["run_id"]: row["mean_seconds"]
        for row in status.get("per_candidate", [])
        if row.get("mean_seconds") is not None
    }
    return per_candidate, status.get("mean_eval_seconds")


def _load_report(run_dir: Path, run_id: str) -> dict | None:
    path = Path(run_dir) / "candidates" / run_id / "tune_report.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def tuning_in_progress(run_dir: Path, run_id: str) -> bool:
    """A Phase-C bout has started and no finalize close covers every stage."""
    report = _load_report(run_dir, run_id)
    if report is None:
        return False
    stages = (report.get("phase_c") or {}).get("stages") or []
    if not stages:
        return False
    from tune_tools import has_validated_applied_close

    try:
        return not has_validated_applied_close(report)
    except ValueError:
        return True


def tune_report_current(run_dir: Path, run_id: str) -> bool:
    """The report's scores belong to the candidate now on disk."""
    report = _load_report(run_dir, run_id)
    if report is None:
        return False
    from tune_tools import validate_candidate_execution_revision

    candidate = Path(run_dir) / "candidates" / run_id / "train.py"
    try:
        validate_candidate_execution_revision(report, candidate)
    except (OSError, ValueError):
        return False
    return True


def rewrite_facts(run_dir: Path, run_id: str) -> dict:
    from rewrite_bout import consecutive_non_kept, load_bouts

    bouts = load_bouts(Path(run_dir) / "candidates" / run_id)
    return {
        "bouts": len(bouts),
        "consecutive_non_kept": consecutive_non_kept(bouts),
        "tuning_in_progress": tuning_in_progress(run_dir, run_id),
    }


# --- selection ---------------------------------------------------------------


def _fits(expected: float | None, quota: float | None) -> bool:
    return quota is None or expected is None or expected <= quota


def select_rewrite(
    state: SchedulerState,
    run_dir: Path,
    *,
    now: float | None = None,
) -> Decision:
    run_dir = Path(run_dir)
    config = load_config(run_dir)
    round_state = load_round_state(run_dir)
    quota = quota_remaining(round_state, now=now)
    per_candidate, overall = _eval_seconds(run_dir)
    overhead = overhead_estimate(config, round_state, "rewrite")
    top_k = int(config["rewrite_top_k"])

    rows = []
    for candidate in sorted(state.candidates, key=lambda c: (c.best_score, c.run_id)):
        if candidate.crashed:
            continue
        reasons = []
        if candidate.has_unresolved_descendant:
            reasons.append("unresolved primary descendant")
        facts = rewrite_facts(run_dir, candidate.run_id)
        if facts["tuning_in_progress"]:
            reasons.append("tuning in progress")
        if facts["bouts"] >= int(config["rewrite_max_bouts"]):
            reasons.append("rewrite bout cap")
        if facts["consecutive_non_kept"] >= int(config["rewrite_stall_after"]):
            reasons.append("stalled")
        mean = per_candidate.get(candidate.run_id, overall)
        # One adjudication eval plus the confirmation re-eval after a keep.
        expected = None if mean is None else 2.0 * float(mean) + overhead
        rows.append({
            "run_id": candidate.run_id,
            "best_score": candidate.best_score,
            "rewrite_bouts": facts["bouts"],
            "consecutive_non_kept": facts["consecutive_non_kept"],
            "expected_seconds": expected,
            "ineligible": reasons,
        })
    eligible = [row for row in rows if not row["ineligible"]][:top_k]
    mode = {
        "kind": "rewrite",
        "quota_remaining_seconds": quota,
        "overhead_seconds": overhead,
        "top_k": top_k,
        "ranked": rows,
    }
    if not eligible:
        return Decision("STOP", None, "no rewrite-eligible candidate", evidence_mode=mode)
    affordable = [row for row in eligible if _fits(row["expected_seconds"], quota)]
    if not affordable:
        return Decision(
            "STOP", None, "no rewrite bout fits the remaining round quota",
            evidence_mode=mode,
        )
    target = min(
        affordable,
        key=lambda row: (row["rewrite_bouts"], row["best_score"], row["run_id"]),
    )
    mode["reference"] = target["best_score"]
    return Decision(
        REWRITE,
        target["run_id"],
        f"fewest rewrite bouts ({target['rewrite_bouts']}) inside the "
        f"top-{top_k} by score",
        evidence_mode=mode,
    )


def select_tune(
    state: SchedulerState,
    run_dir: Path,
    *,
    now: float | None = None,
) -> Decision:
    run_dir = Path(run_dir)
    config = load_config(run_dir)
    round_state = load_round_state(run_dir)
    quota = quota_remaining(round_state, now=now)
    per_candidate, overall = _eval_seconds(run_dir)
    overhead = overhead_estimate(config, round_state, "tune")

    rows = []
    ranked = sorted(state.eligible(), key=lambda c: (c.best_score, c.run_id))
    for rank, candidate in enumerate(ranked):
        expected = state.contract.expected_bout_seconds(
            candidate,
            mean_eval_seconds=per_candidate.get(candidate.run_id, overall),
            session_overhead_seconds=overhead,
        )
        value = 1.0 / (1 + rank)
        rows.append({
            "run_id": candidate.run_id,
            "best_score": candidate.best_score,
            "rank": rank,
            "bouts_used": candidate.bouts_used,
            "bout_evals": state.contract.bout_cost_for(candidate),
            "expected_seconds": expected,
            "priority": value / max(1.0, expected if expected is not None else 1.0),
            "report_current": tune_report_current(run_dir, candidate.run_id),
        })
    mode = {
        "kind": "tune",
        "quota_remaining_seconds": quota,
        "overhead_seconds": overhead,
        "ranked": rows,
    }
    choices = [
        row for row in rows
        if row["report_current"] and _fits(row["expected_seconds"], quota)
    ]
    if not choices:
        return Decision(
            "STOP", None,
            "no tune-eligible candidate fits the remaining round quota",
            evidence_mode=mode,
        )
    target = max(choices, key=lambda row: (row["priority"], -row["rank"]))
    return Decision(
        "TUNE",
        target["run_id"],
        f"best improvement per second (score rank {target['rank']}, "
        f"expected {target['expected_seconds']} s)",
        evidence_mode=mode,
    )


def decide(state: SchedulerState, run_dir: Path) -> Decision:
    """The tune decision `select-candidate` executes; tuning only runs inside
    an open optimization phase."""
    round_state = load_round_state(run_dir)
    if round_state.get("phase") != "optimize":
        return Decision(
            "DEFER", None, "generation phase: no tuning outside a round",
            evidence_mode={"kind": "tune", "phase": round_state.get("phase")},
        )
    return select_tune(state, run_dir)


def receipt(
    decision: Decision,
    *,
    state: SchedulerState,
    evidence_cursor: int,
    snapshot_id: str,
    decision_id: str,
) -> dict:
    return {
        "schema_version": 1,
        "kind": "scheduler_decision",
        "decision_id": decision_id,
        "state_snapshot_id": snapshot_id,
        "policy_version": POLICY_VERSION,
        "prior_id": None,
        "reference_policy_version": None,
        "evidence_cursor": evidence_cursor,
        "paired_scenarios": 0,
        "crn_scheme": None,
        "selected_action": decision.action,
        "selected_run_id": decision.run_id,
        "reason": decision.reason,
        "tie_broken": False,
        "coverage_spent": 0,
        "evidence_mode": decision.evidence_mode,
        "q_hat": [],
        "margin": None,
        "global_best": state.global_best,
        "remaining_budget": state.remaining_budget,
    }
