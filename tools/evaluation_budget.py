#!/usr/bin/env python3
"""Durable, strict reservation ledger for objective-function evaluations.

Every objective runner calls :func:`reserve_evaluation` immediately before
entering the task's ``score_fn``.  The append-only JSONL file is the crash-safe
authority for the run-level evaluation cap; aggregate fields in ``ledger.json``
remain the bounded per-candidate summary.

Preflight calls never use this module.  Tuner preflights are tracked separately
in each candidate's ``tune_report.json``; the hillclimb protocol runs its
standalone preflight before calling this module's reservation CLI.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time
from typing import Any

try:  # POSIX is the supported experiment runtime; keep reads usable elsewhere.
    import fcntl
except ImportError:  # pragma: no cover - Windows compatibility fallback
    fcntl = None

try:
    from .run_cfg import read_framework_cfg
except ImportError:  # direct ``python tools/evaluation_budget.py`` execution
    from run_cfg import read_framework_cfg


SCHEMA_VERSION = 1
ATTEMPT_LOG = "evaluation_attempts.jsonl"
ATTEMPT_KIND = "score_attempt"
#: Appended after an admitted attempt finishes: its wall-clock duration.
#: Never counted against any cap; only the per-candidate eval-time estimate
#: the round scheduler prices bouts with reads it.
COMPLETION_KIND = "score_completion"
RESERVATION_KIND = "budget_reservation"
RESERVATION_SETTLEMENT_KIND = "budget_reservation_settlement"
#: Written by tools/scheduler/round_policy.py while an optimization phase is
#: open; ``phase_deadline`` bounds every reservation inside that phase.
ROUND_STATE_PATH = Path(".scheduler") / "round_state.json"


class EvaluationBudgetExhausted(RuntimeError):
    """Raised before ``score_fn`` when no objective slot remains."""

    def __init__(
        self,
        *,
        used: int,
        budget: int | None,
        run_dir: Path,
        scope: str = "global",
    ):
        self.used = used
        self.budget = budget
        self.run_dir = Path(run_dir)
        self.scope = scope
        super().__init__(
            f"evaluation cut off by the {scope} budget mid-run (run_dir={run_dir})"
            if scope.endswith("cutoff")
            else f"{scope} evaluation budget exhausted before score_fn "
            f"(used={used}, budget={budget}, run_dir={run_dir})"
        )


class BudgetReservationDenied(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        self.reason, self.detail = reason, detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


BudgetAdmissionDenied = BudgetReservationDenied


def find_run_dir(ref_path: Any) -> Path | None:
    """Return the enclosing run directory for a candidate/report path."""
    path = Path(ref_path).resolve()
    for ancestor in (path, *path.parents):
        if (
            (ancestor / "framework_cfg.json").is_file()
            and ancestor.parent.parent.name == "runs"
        ):
            return ancestor
    return None


def _framework_budget(run_dir: Path) -> int | None:
    path = Path(run_dir) / "framework_cfg.json"
    if not path.is_file():
        return None
    value = read_framework_cfg(path).get("max_evaluations")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def time_budget(run_dir: Path, *, now: float | None = None) -> dict:
    """The run-level wall-clock budget view.

    ``deadline`` is an absolute epoch timestamp and ``final_reserve_seconds``
    the tail kept free for export; ``usable_seconds`` is what new work may
    still spend. Every field is ``None`` when no deadline is configured.
    """
    run_dir = Path(run_dir)
    path = run_dir / "framework_cfg.json"
    config = read_framework_cfg(path) if path.is_file() else {}
    deadline = _finite_number(config.get("deadline"))
    reserve = _finite_number(config.get("final_reserve_seconds")) or 0.0
    if deadline is None:
        return {
            "deadline": None,
            "final_reserve_seconds": reserve,
            "remaining_seconds": None,
            "usable_seconds": None,
            "time_reached": None,
        }
    now = time.time() if now is None else now
    remaining = deadline - now
    usable = remaining - reserve
    return {
        "deadline": deadline,
        "final_reserve_seconds": reserve,
        "remaining_seconds": remaining,
        "usable_seconds": usable,
        "time_reached": usable <= 0,
    }


def phase_quota_remaining(run_dir: Path, *, now: float | None = None) -> float | None:
    """Seconds left in the open optimization phase, or None when none is open."""
    path = Path(run_dir) / ROUND_STATE_PATH
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    deadline = _finite_number(state.get("phase_deadline"))
    if deadline is None or state.get("phase") != "optimize":
        return None
    return deadline - (time.time() if now is None else now)


def time_remaining(run_dir: Path) -> float | None:
    """Seconds a new evaluation may still run: the tighter of the run's usable
    time and the open phase quota. None when neither is configured."""
    run_dir = Path(run_dir)
    values = [
        value
        for value in (
            time_budget(run_dir)["usable_seconds"],
            phase_quota_remaining(run_dir),
        )
        if value is not None
    ]
    return min(values) if values else None


def phase_c_attempts(run_dir: Path, run_id: str) -> int:
    """Admitted Phase-C score attempts on record for one candidate.

    Lenient by design: this counts consumption for failure accounting on
    recovery paths, where a missing or partly unreadable log must read as
    what it does show rather than raise — enforcement stays with the
    reservation log itself.
    """
    path = Path(run_dir) / ATTEMPT_LOG
    if not path.is_file():
        return 0
    run_id = str(run_id)
    total = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for raw in lines:
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(row, dict)
            and row.get("kind") == ATTEMPT_KIND
            and row.get("phase") == "phase_c"
            and row.get("run_id") == run_id
        ):
            total += 1
    return total


def _deep_tune_limits(run_dir: Path, budget: int | None) -> dict:
    """Phase-C allocation limits.

    The run-level share is OFF by default. It was a 0.4 fraction with no
    derivation, and run 0802-sonnet-ex125-1 showed what it actually bought:
    the cap bound at evaluation 87 of 124, and every one of the 37 evaluations
    after it went to screening, which produced *zero* score improvements all
    run. Both improvements in that run came from Phase C. Three candidates
    (013/016/018) then landed within noise of the incumbent and none could be
    tuned, while the one candidate that was tuned early gained 0.0406 — more
    than the 0.0145 gap to the hillclimb baseline. Same failure shape as the
    removed Phase-C wall clock: a ceiling on the productive mechanism, chosen
    without measurement, degrading the search silently.

    `deep_tune_per_candidate_cap` stays on: it bounds a real observed failure
    mode (one candidate consuming the whole budget) without capping the phase.
    An explicit `deep_tune_budget_fraction` is still honored — including 0 to
    disable deep tuning outright — so existing run configs keep working.
    """
    path = Path(run_dir) / "framework_cfg.json"
    config = read_framework_cfg(path) if path.is_file() else {}
    tuner = config.get("tuner", {})
    tuner = tuner if isinstance(tuner, dict) else {}
    configured = tuner.get("deep_tune_budget_fraction")
    fraction = None if configured is None else float(configured)
    return {
        "fraction": fraction,
        "total_cap": (
            None
            if fraction is None
            else (
                0
                if fraction == 0
                else (
                    None
                    if budget is None
                    else int(math.floor(budget * fraction))
                )
            )
        ),
        "per_candidate_cap": int(
            tuner.get(
                "deep_tune_per_candidate_cap",
                44
                if tuner.get("inner_policy")
                in (
                    "mixup24-turbo20-v1",
                    "hebo24-turbo20-v1",
                    "hebo24-hebo20",
                    # 24+10+10 ordinary / 10+10+10 donor-transferred both fit
                    # under the same 44-eval lifetime cap (design §5.2).
                    "hebo24-transfer10-hebo10",
                )
                else 40,
            )
        ),
        # Phase-C wall clock removed: trials (patience/n_trials/caps above)
        # denominate the budget. Kept as null so receipt shape is stable.
        "time_limit_seconds": None,
    }


def _deep_tune_usage(
    rows: list[dict],
) -> tuple[int, dict[str, int]]:
    logged: Counter[str] = Counter()
    for row in rows:
        if row.get("kind") != "score_attempt" or row.get("phase") != "phase_c":
            continue
        run_id = row.get("run_id")
        if isinstance(run_id, str):
            logged[run_id] += 1
    return sum(logged.values()), dict(logged)


def _read_rows(handle) -> list[dict]:
    handle.seek(0)
    rows: list[dict] = []
    for line_number, raw in enumerate(handle, start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid {ATTEMPT_LOG} line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"invalid {ATTEMPT_LOG} line {line_number}: expected a JSON object"
            )
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported {ATTEMPT_LOG} schema on line {line_number}: "
                f"{row.get('schema_version')!r}"
            )
        if row.get("kind") not in (ATTEMPT_KIND, COMPLETION_KIND, RESERVATION_KIND, RESERVATION_SETTLEMENT_KIND):
            raise ValueError(
                f"unsupported {ATTEMPT_LOG} row kind on line {line_number}: "
                f"{row.get('kind')!r}"
            )
        rows.append(row)
    return rows


def _attempts(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("kind") == ATTEMPT_KIND]


def _durations(rows: list[dict]) -> tuple[list[float], dict[str, list[float]]]:
    """Completed attempts' durations, overall and per candidate."""
    overall: list[float] = []
    per_candidate: dict[str, list[float]] = {}
    for row in rows:
        if row.get("kind") != COMPLETION_KIND:
            continue
        seconds = _finite_number(row.get("duration_seconds"))
        if seconds is None or seconds < 0:
            continue
        overall.append(seconds)
        run_id = row.get("run_id")
        if isinstance(run_id, str):
            per_candidate.setdefault(run_id, []).append(seconds)
    return overall, per_candidate


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize_rows(rows: list[dict]) -> tuple[int, dict[str, int]]:
    total = 0
    per_candidate: Counter[str] = Counter()
    for row in _attempts(rows):
        total += 1
        run_id = row.get("run_id")
        if isinstance(run_id, str):
            per_candidate[run_id] += 1
    return total, dict(per_candidate)


def attempt_log_summary(run_dir: Path) -> dict[str, Any] | None:
    """Read the append-only attempt log without mutating it.

    This is the public diagnostic view over the same strict parser and
    accounting used by budget admission. Every row is one admitted score
    attempt, so ``evaluations_done`` is exactly the number of canonical rows.
    """
    path = Path(run_dir) / ATTEMPT_LOG
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        rows = _read_rows(handle)

    total, per_candidate = _summarize_rows(rows)
    phase_counts: Counter[str] = Counter()
    score_attempts = 0
    unclassified_score_attempts = 0
    for row in _attempts(rows):
        score_attempts += 1
        phase = row.get("phase")
        if isinstance(phase, str) and phase:
            phase_counts[phase] += 1
        else:
            unclassified_score_attempts += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluations_done": total,
        "per_candidate": per_candidate,
        "phase_counts": dict(sorted(phase_counts.items())),
        "score_attempts": score_attempts,
        "unclassified_score_attempts": unclassified_score_attempts,
        "mean_eval_seconds": _mean(_durations(rows)[0]),
    }


def _append_row(handle, row: dict) -> None:
    handle.seek(0, os.SEEK_END)
    handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _locked_log(run_dir: Path):
    path = Path(run_dir) / ATTEMPT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def reserve_evaluation(
    ref_path: Any,
    *,
    params: dict,
    phase: str,
    method: str,
) -> dict | None:
    """Atomically reserve one objective slot and return its durable receipt.

    Paths outside ``runs/<task>/<tag>`` have no run-level budget and return
    ``None``; this keeps standalone/unit-test uses backward compatible.
    """
    run_dir = find_run_dir(ref_path)
    if run_dir is None:
        return None
    budget = _framework_budget(run_dir)
    run_id = Path(ref_path).resolve().parent.name
    with _locked_log(run_dir) as handle:
        rows = _read_rows(handle)
        used, _ = _summarize_rows(rows)
        holds = _reservation_outstanding(rows)
        held_evals = sum(row['evaluations'] for row in holds.values())
        if budget is not None and used + held_evals >= budget:
            raise EvaluationBudgetExhausted(
                used=used,
                budget=budget,
                run_dir=run_dir,
            )
        if time_budget(run_dir)["time_reached"]:
            raise EvaluationBudgetExhausted(
                used=used, budget=budget, run_dir=run_dir, scope="time"
            )
        quota = phase_quota_remaining(run_dir)
        if quota is not None and quota <= 0:
            raise EvaluationBudgetExhausted(
                used=used, budget=budget, run_dir=run_dir, scope="round_quota"
            )
        if str(phase) == "phase_c":
            limits = _deep_tune_limits(run_dir, budget)
            deep_used, deep_per_candidate = _deep_tune_usage(rows)
            total_cap = limits["total_cap"]
            if total_cap is not None and deep_used >= total_cap:
                raise EvaluationBudgetExhausted(
                    used=deep_used,
                    budget=total_cap,
                    run_dir=run_dir,
                    scope="deep_tune_total",
                )
            candidate_used = deep_per_candidate.get(run_id, 0)
            candidate_cap = limits["per_candidate_cap"]
            if candidate_used >= candidate_cap:
                raise EvaluationBudgetExhausted(
                    used=candidate_used,
                    budget=candidate_cap,
                    run_dir=run_dir,
                    scope=f"deep_tune_candidate:{run_id}",
                )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": ATTEMPT_KIND,
            "attempt_id": f"eval-{used + 1:06d}",
            "run_id": run_id,
            "phase": str(phase),
            "method": str(method),
            "params": params,
        }
        _append_row(handle, receipt)
        rows.append(receipt)
        return receipt


def record_evaluation_completion(
    ref_path: Any,
    *,
    attempt_id: str | None,
    duration_seconds: float,
    time_cutoff: bool = False,
) -> None:
    """Append the wall-clock duration of one admitted attempt (any outcome).

    ``time_cutoff`` marks an attempt the run's time budget killed before it
    produced a result (the limit was the budget's remainder, not the task's
    per_runtime_limit); deadline settlement tells such candidates apart from
    genuinely crashed ones.
    """
    run_dir = find_run_dir(ref_path)
    if run_dir is None or not isinstance(attempt_id, str):
        return
    row = {
        "schema_version": SCHEMA_VERSION,
        "kind": COMPLETION_KIND,
        "attempt_id": attempt_id,
        "run_id": Path(ref_path).resolve().parent.name,
        "duration_seconds": round(float(duration_seconds), 3),
    }
    if time_cutoff:
        row["time_cutoff"] = True
    with _locked_log(run_dir) as handle:
        _append_row(handle, row)


def _time_cutoffs(rows: list[dict]) -> dict[str, int]:
    """Per candidate, admitted attempts the time budget cut off."""
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("kind") == COMPLETION_KIND and row.get("time_cutoff") is True \
                and isinstance(row.get("run_id"), str):
            counts[row["run_id"]] += 1
    return dict(counts)


def _read_budget_rows(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / ATTEMPT_LOG
    if not path.is_file():
        return []
    with path.open() as stream:
        return _read_rows(stream)


def _reservation_outstanding(rows: list[dict]) -> dict[str, dict]:
    """Unstarted capacity, reduced by settlements and admitted attempts."""
    outstanding: dict[str, dict] = {}
    for row in rows:
        reservation_id = row.get("reservation_id")
        if row["kind"] in (RESERVATION_KIND,):
            outstanding[reservation_id] = {
                "reservation_id": reservation_id,
                "label": row.get("label"),
                "seconds": row["reserved_seconds"],
                "evaluations": row["reserved_evaluations"],
            }
        elif reservation_id in outstanding:
            entry = outstanding[reservation_id]
            if row["kind"] == RESERVATION_SETTLEMENT_KIND and row["outcome"] in ("released", "activated"):
                del outstanding[reservation_id]
            elif row["kind"] in (RESERVATION_SETTLEMENT_KIND, ATTEMPT_KIND):
                entry["seconds"] = max(0.0, entry["seconds"] - row.get("consumed_seconds", 0))
                entry["evaluations"] = max(0, entry["evaluations"] - row.get("consumed_evaluations", 0))
    return {key: value for key, value in outstanding.items()
            if value["seconds"] > 0 or value["evaluations"] > 0}


def outstanding_reservations(run_dir: Path) -> dict:
    """Time/objective capacity currently promised to future work."""
    entries = list(_reservation_outstanding(_read_budget_rows(run_dir)).values())
    return {
        "seconds": round(sum(entry["seconds"] for entry in entries), 3),
        "evaluations": sum(entry["evaluations"] for entry in entries),
        "rows": entries,
    }


def outstanding_reserved_seconds(run_dir: Path) -> float:
    return float(outstanding_reservations(run_dir)["seconds"])


def outstanding_reserved_evaluations(run_dir: Path) -> int:
    return int(outstanding_reservations(run_dir)["evaluations"])


def reserve_budget(
    ref_path: Any,
    *,
    label: str,
    seconds: float = 0.0,
    evaluations: int = 0,
    note: str = "",
) -> dict:
    """Promise capacity for future work (space review + probe slate).

    Admission checks remaining run capacity: the promised seconds and
    objective slots must fit what the run can still spend after existing
    reservations, or the request is denied outright — a reservation never
    overdraws the budget it is promising. ``settle_reservation`` drains it
    as the protected work consumes time/slots.
    """
    run_dir = find_run_dir(ref_path)
    if run_dir is None:
        raise BudgetReservationDenied(
            "outside_run", "no runs/<task>/<tag> encloses the reference path"
        )
    seconds = _finite_number(seconds)
    if seconds is None or seconds < 0:
        raise BudgetReservationDenied("invalid_request", "seconds must be finite and non-negative")
    if (
        not isinstance(evaluations, int)
        or isinstance(evaluations, bool)
        or evaluations < 0
    ):
        raise BudgetReservationDenied("invalid_request", "evaluations must be a non-negative integer")
    if seconds <= 0 and evaluations <= 0:
        raise BudgetReservationDenied("invalid_request", "reservation is empty")
    with _locked_log(run_dir) as handle:
        rows = _read_rows(handle)
        outstanding = _reservation_outstanding(rows)
        # Use the raw clock while holding the common budget lock.
        clock = time_budget(run_dir)
        reserved_seconds = sum(entry["seconds"] for entry in outstanding.values())
        reserved_evals = sum(entry["evaluations"] for entry in outstanding.values())
        if clock["usable_seconds"] is not None and (
            seconds + reserved_seconds > clock["usable_seconds"]
        ):
            raise BudgetReservationDenied(
                "insufficient_time",
                f"requested {seconds:g}s; usable {clock['usable_seconds']:g}s "
                f"with {reserved_seconds:g}s already reserved",
            )
        budget = _framework_budget(run_dir)
        if budget is not None:
            used, _ = _summarize_rows(rows)
            if budget - used - reserved_evals - evaluations < 0:
                raise BudgetReservationDenied(
                    "insufficient_evaluations",
                    f"used={used}, reserved={reserved_evals}, "
                    f"requested={evaluations}, budget={budget}",
                )
        reservation = {
            "schema_version": SCHEMA_VERSION,
            "kind": RESERVATION_KIND,
            "reservation_id": f"resv-{sum(1 for r in rows if r.get('kind') == RESERVATION_KIND) + 1:04d}",
            "run_id": Path(ref_path).resolve().parent.name,
            "label": str(label)[:200],
            "reserved_seconds": round(seconds, 3),
            "reserved_evaluations": evaluations,
            "note": str(note)[:500],
            "created_at_epoch": time.time(),
        }
        _append_row(handle, reservation)
        return reservation


def settle_reservation(
    ref_path: Any,
    reservation_id: str,
    *,
    consumed_seconds: float = 0.0,
    consumed_evaluations: int = 0,
    outcome: str = "consumed",
    reason: str = "",
) -> dict | None:
    """Drain one reservation by the capacity the protected work consumed.

    Partial settlements are append-only and accumulate; ``outcome``
    distinguishes normal consumption from an explicit release
    (cancelled/unevaluated promised work). Over-settling is a no-op beyond
    zero (the outstanding computation floors at zero). A release clears
    all remaining capacity; repeated releases are no-ops."""
    run_dir = find_run_dir(ref_path)
    if run_dir is None:
        return None
    seconds = _finite_number(consumed_seconds) or 0.0
    if seconds < 0:
        seconds = 0.0
    if not isinstance(consumed_evaluations, int) or isinstance(consumed_evaluations, bool):
        consumed_evaluations = 0
    row = {
        "schema_version": SCHEMA_VERSION,
        "kind": RESERVATION_SETTLEMENT_KIND,
        "reservation_id": str(reservation_id),
        "consumed_seconds": round(seconds, 3),
        "consumed_evaluations": max(0, consumed_evaluations),
        "outcome": str(outcome),
        "reason": str(reason)[:500],
        "created_at_epoch": time.time(),
    }
    with _locked_log(run_dir) as handle:
        if reservation_id not in _reservation_outstanding(_read_rows(handle)):
            return None
        _append_row(handle, row)
        return row


def activate_reservation(ref_path: Any, reservation_id: str) -> dict:
    """Transfer a whole slate's pending promise to the driver's active work.

    The caller starts the protected slate immediately and closes its publication
    afterwards. Objective calls then charge the ordinary run budget.
    """
    run_dir = find_run_dir(ref_path)
    if run_dir is None:
        raise BudgetReservationDenied("outside_run")
    with _locked_log(run_dir) as handle:
        rows = _read_rows(handle)
        outstanding = _reservation_outstanding(rows)
        own = outstanding.pop(reservation_id, None)
        if own is None:
            activated = next((row for row in reversed(rows)
                              if row["kind"] == RESERVATION_SETTLEMENT_KIND
                              and row.get("reservation_id") == reservation_id
                              and row.get("outcome") == "activated"), None)
            if activated is not None:
                return activated
            raise BudgetReservationDenied("reservation_unavailable", reservation_id)
        available = time_budget(run_dir)["usable_seconds"]
        if available is not None:
            available -= sum(row["seconds"] for row in outstanding.values())
            if available < own["seconds"]:
                raise BudgetReservationDenied("insufficient_time", "reserved work no longer fits")
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": RESERVATION_SETTLEMENT_KIND,
            "reservation_id": reservation_id,
            "outcome": "activated",
            "label": own["label"],
            "activated_seconds": own["seconds"],
            "activated_evaluations": own["evaluations"],
            "created_at_epoch": time.time(),
        }
        _append_row(handle, receipt)
        return receipt


def budget_status(run_dir: Path, *, create: bool = False) -> dict:
    """Return the strict objective usage view without mutating by default."""
    run_dir = Path(run_dir)
    path = run_dir / ATTEMPT_LOG
    rows: list[dict] = []
    if path.exists() or create:
        with _locked_log(run_dir) as handle:
            rows = _read_rows(handle)
    total, per_candidate = _summarize_rows(rows)
    budget = _framework_budget(run_dir)
    limits = _deep_tune_limits(run_dir, budget)
    deep_used, deep_per_candidate = _deep_tune_usage(rows)
    deep_total_cap = limits["total_cap"]
    clock = time_budget(run_dir)
    quota = phase_quota_remaining(run_dir)
    overall_seconds, seconds_by_candidate = _durations(rows)
    cutoffs = _time_cutoffs(rows)
    evals_reached = None if budget is None else total >= budget
    if clock["time_reached"] is None:
        reached = evals_reached
    else:
        reached = bool(evals_reached) or clock["time_reached"]
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluations_done": total,
        "objective_attempts": total,
        "budget": budget,
        "remaining": None if budget is None else max(0, budget - total),
        "reached": reached,
        "time": clock,
        "phase_quota_remaining_seconds": quota,
        "phase_quota_reached": quota is not None and quota <= 0,
        "mean_eval_seconds": _mean(overall_seconds),
        "per_candidate": [
            {
                "run_id": run_id,
                "evals": value,
                "mean_seconds": _mean(seconds_by_candidate.get(run_id, [])),
                "time_cutoff_evals": cutoffs.get(run_id, 0),
            }
            for run_id, value in sorted(per_candidate.items())
        ],
        "deep_tune": {
            **limits,
            "attempts": deep_used,
            "remaining": (
                None
                if deep_total_cap is None
                else max(0, deep_total_cap - deep_used)
            ),
            "per_candidate": [
                {"run_id": run_id, "evals": value}
                for run_id, value in sorted(deep_per_candidate.items())
            ],
        },
        "reservations": outstanding_reservations(run_dir),
        "attempt_log": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-dir", required=True, type=Path)
    status_parser.add_argument(
        "--initialize",
        action="store_true",
        help="create the append-only attempt log",
    )
    reserve_parser = subparsers.add_parser(
        "reserve",
        help="atomically reserve one objective attempt before a standalone run",
    )
    reserve_parser.add_argument("--ref-path", required=True, type=Path)
    reserve_parser.add_argument("--phase", default="hillclimb")
    reserve_parser.add_argument("--method", default="direct")
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(budget_status(args.run_dir, create=args.initialize)))
        return 0
    if args.command == "reserve":
        ref_path = args.ref_path.resolve()
        if not ref_path.is_file():
            parser.error(f"--ref-path does not exist: {ref_path}")
        params = {}
        try:
            receipt = reserve_evaluation(
                ref_path,
                params=params,
                phase=args.phase,
                method=args.method,
            )
        except EvaluationBudgetExhausted as exc:
            print(
                json.dumps(
                    {
                        "status": "exhausted",
                        "evaluations_done": exc.used,
                        "budget": exc.budget,
                        "run_dir": str(exc.run_dir),
                    }
                )
            )
            return 4
        if receipt is None:
            parser.error("--ref-path is not inside an initialized runs/<task>/<tag>")
        print(json.dumps({"status": "reserved", "receipt": receipt}))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
