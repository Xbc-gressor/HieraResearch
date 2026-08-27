"""Pure data model and state derivations for the run ledger.

This module deliberately performs no filesystem I/O.  ``tools/ledger.py`` is
the persistence boundary: it loads a ledger, applies these transformations,
and commits the result.  Keeping the helpers pure makes the record lifecycle
readable without pulling CLI, background-research, or tuner concerns into the
same module.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional, TypedDict

from semantic_evidence import LIFECYCLE_TERMINAL_STATUSES
from semantic_space import digest


class LedgerRecord(TypedDict, total=False):
    """The commonly accessed portion of one persisted candidate record."""

    run_id: str
    status: str
    op: str
    source_run_ids: list[str]
    metric: str
    tune: bool
    final_best_score: float
    best_warm_score: float
    dag_revision: int
    semantic_point: dict[str, Any]
    policy_receipt: dict[str, Any]
    parameter_transfer: dict[str, Any]
    applied_incumbent: dict[str, Any]


class LedgerData(TypedDict, total=False):
    """Top-level persisted ledger fields used by core state derivations."""

    task: str
    tag: str
    metric: str
    dag_revision: int
    records: list[LedgerRecord]
    experience: dict[str, Any]
    run_state: dict[str, Any]
    search_space_state: dict[str, Any]
    lineage_snapshots: list[dict[str, Any]]
    attempt_observations: list[dict[str, Any]]
    items: dict[str, dict[str, Any]]


# Field order is the on-disk record layout. Keep stable; do not rename keys.
RECORD_FIELDS = (
    "run_id",
    "kind",
    "idea",
    "change",
    "source_run_ids",
    "op",
    "semantic_point",
    "semantic_edges",
    "policy_receipt",
    "route_provenance",
    "role",
    "parameter_transfer",
    "applied_incumbent",
    "candidate_name",
    "description",
    "metric",
    "tune",
    "status",
    "unevaluated_receipt",
    "best_warm_score",
    "final_best_score",
    "evaluation_depth",
    "n_dims",
    "warm_start_K",
    "warm_percentile",
    "phase_b_decision",
    "phase_c_method",
    "trials_completed",
    "trials_attempted",
    "preflight_attempts",
    "preflight_failures",
    "feasibility_rejections",
    "elapsed_seconds",
    "applied",
    "dag_revision",
    "tuning_bouts",
    "last_bout_improved",
)

TUNING_FIELDS = (
    "best_warm_score",
    "evaluation_depth",
    "n_dims",
    "warm_start_K",
    "warm_percentile",
    "phase_b_decision",
    "phase_c_method",
    "trials_completed",
    "trials_attempted",
    "preflight_attempts",
    "preflight_failures",
    "feasibility_rejections",
    "elapsed_seconds",
    "applied",
    "parameter_transfer",
    "applied_incumbent",
    "tuning_bouts",
    "last_bout_improved",
)

# Scores are always lower-is-better, so a crash is the worst possible score.
CRASH_SENTINEL = float("inf")

# Retained for the existing percentile CLI. Candidate selection itself lives in
# tuners/tune_tools.py.
COLD_START_N = 10
TOP_PERCENTILE = 90


def get_record(data: dict, run_id: str) -> Optional[dict]:
    for record in data["records"]:
        if record.get("run_id") == run_id:
            return record
    return None


def require_current_record(record: dict, run_id: str) -> None:
    if not isinstance(record.get("semantic_point"), dict):
        raise ValueError(
            f"record {run_id} has no semantic_point; unsupported record format"
        )
    if not isinstance(record.get("policy_receipt"), dict):
        raise ValueError(
            f"record {run_id} has no policy_receipt; selection policy must remain traceable"
        )


def new_record(run_id: str) -> dict:
    return {field: None for field in RECORD_FIELDS} | {
        "run_id": run_id,
        "tune": False,
        "tuning_bouts": 0,
        "last_bout_improved": None,
        "status": "pending",
    }


def current_dag_revision(data: dict) -> int:
    return int(data.get("dag_revision", 0))


# ---------- judged-slate ledger snapshots (selection-safe projections) ----------
# The fields a selection layer may read from one prefix record. Notably absent:
# `final_best_score` and every tuning-lifecycle field, which legitimately change
# in later bouts and must not invalidate a generation's prefix binding.
SELECTION_SAFE_RECORD_FIELDS = (
    "run_id",
    "selection_index",
    "op",
    "source_run_ids",
    "status",
    "idea",
    "semantic_point",
    "best_warm_score",
)


def selection_safe_record_projection(record: dict, selection_index: int) -> dict:
    """Project one record to the fields judged-slate artifacts may bind to.

    ``selection_index`` is positional: admission enforces
    ``len(records) + 1`` (ledger_admission), so the projection derives it
    instead of trusting a mutable record field.
    """
    return {
        "run_id": record.get("run_id"),
        "selection_index": selection_index,
        "op": record.get("op"),
        "source_run_ids": record.get("source_run_ids"),
        "status": record.get("status"),
        "idea": record.get("idea"),
        "semantic_point": record.get("semantic_point"),
        "best_warm_score": record.get("best_warm_score"),
    }


def records_prefix_digest(records: list) -> str:
    """Digest the generation-start ledger prefix in admission order."""
    return digest(
        [
            selection_safe_record_projection(record, index + 1)
            for index, record in enumerate(records)
        ]
    )


def experience_receipt(data: dict) -> dict:
    """Compact receipt of the ledger's experience snapshot (Nones when absent)."""
    experience = data.get("experience") if isinstance(data, dict) else None
    if not isinstance(experience, dict) or not experience:
        return {"generation": None, "updated_at_run": None, "revision": None}
    return {
        "generation": experience.get("generation"),
        "updated_at_run": experience.get("updated_at_run"),
        "revision": digest(experience),
    }


def search_space_state_revision(data: dict) -> int:
    state = data.get("search_space_state") if isinstance(data, dict) else None
    if not isinstance(state, dict):
        return 0
    revision = state.get("revision")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < 0
    ):
        raise ValueError(
            "ledger.search_space_state.revision must be a non-negative integer"
        )
    return revision


def experience_refresh_status(data: dict) -> dict:
    """Return the deterministic experience refresh and admission state."""
    dag_revision = data.get("dag_revision", 0)
    if (
        not isinstance(dag_revision, int)
        or isinstance(dag_revision, bool)
        or dag_revision < 0
    ):
        raise ValueError("ledger.dag_revision must be a non-negative integer")
    experience = data.get("experience")
    experience = experience if isinstance(experience, dict) else {}
    raw_cursor = experience.get("dag_revision")
    if raw_cursor is None:
        cursor = 0
    elif (
        not isinstance(raw_cursor, int)
        or isinstance(raw_cursor, bool)
        or raw_cursor < 0
    ):
        raise ValueError(
            "ledger.experience.dag_revision must be a non-negative integer"
        )
    else:
        cursor = raw_cursor
    if cursor > dag_revision:
        raise ValueError(
            "ledger.experience.dag_revision cannot exceed ledger.dag_revision"
        )
    records = data.get("records")
    records = records if isinstance(records, list) else []
    all_terminal = bool(records) and all(
        isinstance(record, dict)
        and record.get("status") in LIFECYCLE_TERMINAL_STATUSES
        for record in records
    )
    delta = dag_revision - cursor
    return {
        "dag_revision": dag_revision,
        "experience_dag_revision": raw_cursor,
        "experience_cursor": cursor,
        "experience_dag_delta": delta,
        "all_records_terminal": all_terminal,
        "semantic_admission_blocked": delta > 0,
        "experience_refresh_required": all_terminal and delta > 0,
    }


def touch_dag_record(data: dict, record: dict) -> int:
    """Mark one new or score-updated DAG node with the next revision."""
    current = current_dag_revision(data) + 1
    data["dag_revision"] = current
    record["dag_revision"] = current
    return current


def is_improvement(value: float, best: Optional[float]) -> bool:
    if best is None:
        return True
    return value < best


def best_kept_value(
    data: dict, exclude_run_id: Optional[str] = None
) -> Optional[float]:
    values = [
        record["final_best_score"]
        for record in data["records"]
        if record.get("status") == "keep"
        and isinstance(record.get("final_best_score"), (int, float))
        and math.isfinite(float(record["final_best_score"]))
        and record.get("run_id") != exclude_run_id
    ]
    return min(values) if values else None


def best_kept_record(data: dict) -> Optional[dict]:
    best: Optional[dict] = None
    best_value: Optional[float] = None
    for record in data["records"]:
        if record.get("status") != "keep":
            continue
        value = record.get("final_best_score")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        if is_improvement(value, best_value):
            best_value = value
            best = record
    return best


def next_run_id(data: dict) -> str:
    numeric = [
        int(record["run_id"])
        for record in data["records"]
        if str(record.get("run_id", "")).isdigit()
    ]
    widths = [3] + [
        len(record["run_id"])
        for record in data["records"]
        if str(record.get("run_id", "")).isdigit()
    ]
    if not numeric:
        return "000"
    return f"{max(numeric) + 1:0{max(widths)}d}"


def percentile(data: dict, run_id: str, field: str) -> dict:
    target = get_record(data, run_id)
    value = target.get(field) if target else None
    priors = [
        record[field]
        for record in data["records"]
        if record.get("run_id") != run_id
        and isinstance(record.get(field), (int, float))
        and math.isfinite(float(record[field]))
    ]
    if (
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not priors
    ):
        return {"value": value, "count_prior": len(priors), "percentile": None}
    below = sum(1 for prior in priors if prior > value)
    return {
        "value": value,
        "count_prior": len(priors),
        "percentile": round(100.0 * below / len(priors)),
    }


def phase_b_decision(result: dict) -> dict:
    n_prior = result["count_prior"]
    if n_prior < COLD_START_N:
        return {
            "decision": "continue",
            "reason": f"cold-start: {n_prior} < {COLD_START_N} prior records with the field",
        }
    rank = result["percentile"]
    if rank is None:
        return {
            "decision": "continue",
            "reason": "current value not recorded; cannot rank, continuing",
        }
    if rank >= TOP_PERCENTILE:
        return {
            "decision": "continue",
            "reason": f"top {100 - TOP_PERCENTILE}%: percentile {rank} >= {TOP_PERCENTILE}",
        }
    return {
        "decision": "stop",
        "reason": f"below top {100 - TOP_PERCENTILE}%: percentile {rank} < {TOP_PERCENTILE}",
    }


def format_score(value: Any) -> str:
    return f"{value:.6f}" if isinstance(value, (int, float)) else "none"


def json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    # This digest binds persisted parent revisions and tuning reports.  It is
    # identity data, not generic defensive hashing.
    import hashlib

    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def json_native(value: Any) -> Any:
    """Normalize a durable receipt to exactly what JSON reload will return."""
    try:
        return json.loads(
            json.dumps(
                value,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"value is not durable JSON: {exc}") from exc
