#!/usr/bin/env python3
"""Structured per-run ledger for autoresearch experiments.

`ledger.json` is the single source of truth for one
`runs/<task>/<tag>/` directory. It replaces the older split between
`idea_log.md` (the per-candidate "why" + tuning summary) and `results.tsv`
(the numeric scores + keep/discard state). One JSON record per candidate
holds both halves, so there is no cross-file sync to drift.

The file is only ever written by this module — never hand-edited — so the
schema stays stable across long autonomous runs. The LLM/agents supply field
*values* as CLI arguments; this module owns the layout, the keep/discard
decision, `next_run_id`, the cross-idea percentile, and the derived
`loop_state.md`.

Shape:

    {
      "task": "tabular-model-search",
      "tag": "agent-main-smoke",
      "metric": "mean_test_accuracy",
      "direct_comparator_capability": { ... },  # explicit unavailable runtime gate
      "search_space": { ... },           # exact catalog + background revision
      "search_space_state": { ... },     # append-only P2 runtime eligibility overlay
      "lineage_snapshots": [ ... ],      # append-only parent revisions cited by children
      "dag_revision": 12,              # monotone graph-change cursor
      "records": [ {record}, ... ]   # ordered by run_id
    }

Record schema (see RECORD_FIELDS). Unavailable fields are `null`, never
omitted. `add-record` creates a record with idea fields; ordinary `set-tuning`
fills Phase-A metadata; `finalize_tuning` closes deep tuning atomically on the
ledger side; `record-run` fills an ordinary run result and computes `status`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

from evaluation_budget import budget_status
from ledger_core import (
    CRASH_SENTINEL,
    RECORD_FIELDS,
    TUNING_FIELDS,
    best_kept_value as _best_kept_value,
    current_dag_revision as _current_dag_revision,
    experience_refresh_status as _experience_refresh_status,
    get_record as _get_record,
    is_improvement as _is_improvement,
    new_record as _new_record,
    percentile as _percentile,
    phase_b_decision as _phase_b_decision,
    require_current_record as _require_p1_record,
    touch_dag_record as _touch_dag_record,
)
from ledger_tuning import (
    applied_incumbent_from_report as _applied_incumbent_from_report,
    capture_transfer_parent_snapshot as _capture_transfer_parent_snapshot,
    preserve_descendant_bindings as _preserve_descendant_bindings,
    prospective_finalized_record as _prospective_finalized_record,
    tuning_record_from_report as _tuning_record_from_report,
    validate_tuning_report_ownership as _validate_tuning_report_ownership,
)
from ledger_views import (
    brief as _brief_view,
    evaluations_done as _evaluations_done,
    framework_budget as _framework_budget,
    render_loop_state,
    run_phase as _run_phase,
)
from run_cfg import RunConfigError
from search_space_state import (
    append_experience_transitions,
)
from semantic_attempts import (
    capture_attempt_observation as _capture_attempt_observation,
)
from semantic_evidence import (
    DIRECT_COMPARATOR_CAPABILITY,
    DIRECT_COMPARATOR_CAPABILITY_KEY,
    validate_parameter_transfer_binding,
)
from validate_tasks import ROOT, parse_task_toml


# ---------- task config ----------


def infer_task_name(paths: list[Optional[Path]]) -> Optional[str]:
    for path in paths:
        if path is None:
            continue
        parts = Path(path).resolve().parts
        for index, part in enumerate(parts[:-1]):
            if part == "runs" and index + 1 < len(parts):
                return parts[index + 1]
    return None


def load_task_config(task_name: str) -> dict:
    task_toml = ROOT / "tasks" / task_name / "task.toml"
    if not task_toml.exists():
        raise ValueError(f"missing task.toml for task: {task_name}")
    return parse_task_toml(task_toml)


# ---------- ledger I/O ----------


def _load_ledger(path: Path) -> dict:
    if path.exists() and path.stat().st_size > 0:
        with open(path) as f:
            data = json.load(f)
        data.setdefault("records", [])
        if data["records"] and not isinstance(data.get("search_space_state"), dict):
            raise ValueError(
                f"{path}: record-bearing ledger requires search_space_state; "
                "run an explicit migration instead of rebuilding pruning history"
            )
        return data
    return {
        "task": None,
        "tag": None,
        "metric": None,
        "records": [],
        "lineage_snapshots": [],
        DIRECT_COMPARATOR_CAPABILITY_KEY: copy.deepcopy(
            DIRECT_COMPARATOR_CAPABILITY
        ),
    }


def _save_ledger(path: Path, data: dict) -> None:
    data.setdefault(
        DIRECT_COMPARATOR_CAPABILITY_KEY,
        copy.deepcopy(DIRECT_COMPARATOR_CAPABILITY),
    )
    data["records"].sort(key=lambda r: r.get("run_id", ""))
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _ensure_meta(data: dict, ledger_path: Path, task_name: str, config: dict) -> None:
    result = config.get("result", {})
    if data.get("task") is None:
        data["task"] = task_name
    if data.get("tag") is None:
        data["tag"] = ledger_path.parent.name
    if data.get("metric") is None:
        data["metric"] = result.get("metric", "score")


# ---------- loop_state.md (derived) ----------


def _write_loop_state(ledger_path: Path, data: dict, config: dict) -> Path:
    state_path = ledger_path.parent / "loop_state.md"
    state_path.write_text(render_loop_state(ledger_path, data, config))
    return state_path


# ---------- mutations ----------


def _coerce(value: Optional[str], kind: str):
    if value is None:
        return None
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "bool":
        return value.lower() in {"true", "1", "yes"}
    return value


def cmd_add_record(args) -> int:
    from ledger_admission import AdmissionError, AdmissionRequest, admit_record

    ledger_path = Path(args.ledger)
    task_name = args.task or infer_task_name([ledger_path])
    if not task_name:
        raise SystemExit("could not infer task; pass --task")
    config = load_task_config(task_name)
    data = _load_ledger(ledger_path)
    _ensure_meta(data, ledger_path, task_name, config)
    try:
        record = admit_record(
            data,
            AdmissionRequest(
                run_id=args.run_id,
                kind=args.kind,
                idea=args.idea,
                change=args.change,
                source_run_ids=args.source_run_ids,
                op=args.op,
                background_path=Path(args.background),
                catalog_path=Path(args.catalog) if args.catalog else None,
                semantic_point_path=Path(args.semantic_point),
                policy_receipt_path=Path(args.policy_receipt),
                candidate_name_hint=args.candidate_name_hint,
                description=args.description,
                route_provenance_path=(
                    Path(args.route_provenance) if args.route_provenance else None
                ),
                task_config=config,
            ),
        )
    except AdmissionError as exc:
        raise SystemExit(str(exc)) from None
    _save_ledger(ledger_path, data)
    _write_loop_state(ledger_path, data, config)
    print(json.dumps(record, indent=2))
    return 0


def _tuning_target(
    ledger_path: Path,
    task_name: str,
    run_id: str,
) -> tuple[dict, dict, dict]:
    """Load and validate the immutable preconditions for a tuning close."""
    config = load_task_config(task_name)
    data = _load_ledger(ledger_path)
    record = _get_record(data, run_id)
    if record is None:
        raise ValueError(f"no record for run_id {run_id}; add-record first")
    _require_p1_record(record, run_id)
    if record.get("status") == "unevaluated":
        raise ValueError(
            f"record {run_id} is terminal unevaluated and cannot be tuned"
        )
    return config, data, record


def validate_tuning_finalization(
    ledger_path: Path,
    task_name: str,
    run_id: str,
    prospective_report_path: Path,
    *,
    owned_report_path: Path,
) -> dict:
    """Validate an exact prospective close before any durable file mutation.

    ``prospective_report_path`` may be a temporary, byte-identical rendering of
    the final report.  ``owned_report_path`` binds that rendering to the real
    candidate/run path so previews cannot weaken ownership checks.
    """
    ledger_path = Path(ledger_path).resolve()
    candidate_path = _validate_tuning_report_ownership(
        ledger_path,
        run_id,
        owned_report_path,
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tuners"))
    from tune_tools import validate_candidate_execution_revision  # noqa: E402

    prospective_report = json.loads(Path(prospective_report_path).read_text())
    validate_candidate_execution_revision(prospective_report, candidate_path)
    _, data, record = _tuning_target(ledger_path, task_name, run_id)
    _preserve_descendant_bindings(data, run_id)
    strict = budget_status(ledger_path.parent)
    strict_attempts = next(
        (
            row["evals"]
            for row in strict.get("per_candidate", [])
            if row.get("run_id") == run_id
        ),
        0,
    )
    prospective = _prospective_finalized_record(
        data,
        record,
        prospective_report_path,
        strict_attempts=int(strict_attempts),
        validate_revision=False,
    )
    _capture_transfer_parent_snapshot(data, prospective)
    return prospective


def validate_tuning_target(
    ledger_path: Path,
    task_name: str,
    run_id: str,
) -> None:
    """Prove that a ledger can accept the close before candidate mutation."""
    _tuning_target(ledger_path, task_name, run_id)
    # Parse the strict admission log now as well.  The finalizer calls this
    # before applying params, so a damaged accounting receipt cannot produce a
    # closed report with no corresponding ledger transaction.
    budget_status(ledger_path.parent)


def finalize_tuning(
    ledger_path: Path,
    task_name: str,
    run_id: str,
    report_path: Path,
) -> dict:
    """Atomically close one candidate's ledger state from a completed report.

    Candidate/report mutation is owned by ``tools/finalize_tuning.py``.  This
    function is the ledger-side transaction: score/status, tuning metadata, and
    ``tune: true`` are written together only after the report proves that Phase
    C reached a terminal finalizable state and that the global best was applied.
    """
    ledger_path = Path(ledger_path).resolve()
    report_path = Path(report_path).resolve()
    _validate_tuning_report_ownership(ledger_path, run_id, report_path)
    config, data, record = _tuning_target(ledger_path, task_name, run_id)
    _preserve_descendant_bindings(data, run_id)

    # The append-only admission log is authoritative even when an external kill
    # happened after reservation but before a trial row could be appended.
    strict = budget_status(ledger_path.parent)
    strict_attempts = next(
        (
            row["evals"]
            for row in strict.get("per_candidate", [])
            if row.get("run_id") == run_id
        ),
        0,
    )
    prospective = _prospective_finalized_record(
        data,
        record,
        report_path,
        strict_attempts=int(strict_attempts),
    )

    before_graph_value = (record.get("status"), record.get("final_best_score"))
    record.clear()
    record.update(prospective)
    _capture_transfer_parent_snapshot(data, record)

    after_graph_value = (record.get("status"), record.get("final_best_score"))
    if after_graph_value != before_graph_value:
        _touch_dag_record(data, record)
    _save_ledger(ledger_path, data)
    _write_loop_state(ledger_path, data, config)
    return record


def cmd_set_tuning(args) -> int:
    ledger_path = Path(args.ledger)
    task_name = args.task or infer_task_name([ledger_path])
    if args.mark_tuned:
        raise SystemExit(
            "--mark-tuned is disabled; use tools/finalize_tuning.py so the "
            "candidate, report, and ledger are prevalidated and closed together"
        )

    config = load_task_config(task_name)
    data = _load_ledger(ledger_path)
    record = _get_record(data, args.run_id)
    if record is None:
        raise SystemExit(f"no record for run_id {args.run_id}; add-record first")
    _require_p1_record(record, args.run_id)
    if record.get("status") == "unevaluated":
        raise SystemExit(
            f"record {args.run_id} is terminal unevaluated and cannot receive tuning metadata"
        )
    try:
        _preserve_descendant_bindings(data, args.run_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    if args.from_report:
        try:
            _validate_tuning_report_ownership(
                ledger_path,
                args.run_id,
                Path(args.from_report),
            )
            updates = _tuning_record_from_report(Path(args.from_report))
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    else:
        receipt = record.get("policy_receipt")
        if (
            record.get("op") in {"improve", "crossover"}
            and isinstance(receipt, dict)
            and receipt.get("schema_version") in {6, 7}
        ):
            raise SystemExit(
                "non-fresh candidates require --from-report so the exact "
                "parameter-transfer control is persisted"
            )
        updates = {
            "best_warm_score": _coerce(args.best_warm_score, "float"),
            "n_dims": _coerce(args.n_dims, "int"),
            "warm_start_K": _coerce(args.warm_start_k, "int"),
            "warm_percentile": _coerce(args.warm_percentile, "int"),
            "phase_b_decision": args.phase_b_decision,
            "phase_c_method": args.phase_c_method,
            "trials_completed": _coerce(args.trials_completed, "int"),
            "trials_attempted": _coerce(args.trials_attempted, "int"),
            "preflight_attempts": _coerce(args.preflight_attempts, "int"),
            "preflight_failures": _coerce(args.preflight_failures, "int"),
            "feasibility_rejections": _coerce(args.feasibility_rejections, "int"),
            "elapsed_seconds": _coerce(args.elapsed_seconds, "float"),
            "applied": _coerce(args.applied, "bool"),
        }
    if args.from_report:
        # A report is the complete authority for the tuning projection.  Its
        # explicit nulls clear metadata left by an earlier partial/manual
        # update; preserving any old value would make the ledger disagree with
        # the report that was just admitted.
        for key in TUNING_FIELDS:
            record[key] = updates.get(key)
    else:
        # Individual flags remain patch semantics: an omitted flag does not
        # erase a field that the caller did not address.
        for key, value in updates.items():
            if value is not None:
                record[key] = value
    transfer_errors = validate_parameter_transfer_binding(data, record)
    if transfer_errors:
        raise SystemExit("; ".join(transfer_errors))
    try:
        _capture_transfer_parent_snapshot(data, record)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    _save_ledger(ledger_path, data)
    _write_loop_state(ledger_path, data, config)
    print(json.dumps(record, indent=2))
    return 0


def record_run(
    ledger_path: Path,
    task_name: str,
    run_id: str,
    *,
    final_best_score: Optional[float] = None,
    status: str = "auto",
    candidate_name: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """Fill one record's result score (the config-eval best — there is no separate
    official run) and the (auto) keep/discard/crash status. Owns `final_best_score`
    and `status`; tuning metadata is written by `set-tuning`. Every candidate
    requires a validated semantic mapping, so missing records are not
    synthesized by the result path. Returns the updated record.
    """
    config = load_task_config(task_name)
    data = _load_ledger(ledger_path)
    record = _get_record(data, run_id)
    if record is None:
        raise ValueError(
            f"no record for run_id {run_id}; add-record with a validated semantic point first"
        )
    _require_p1_record(record, run_id)
    if record.get("status") == "unevaluated":
        raise ValueError(
            f"record {run_id} is terminal unevaluated and cannot receive a score"
        )
    _preserve_descendant_bindings(data, run_id)

    before_graph_value = (record.get("status"), record.get("final_best_score"))

    # A non-finite score (+inf from a timeout / all-failed HPO, or nan) is NOT a
    # real result — it means the candidate could not be evaluated → crash, not
    # discard. Only a finite score is keep/discard.
    has_result = final_best_score is not None and math.isfinite(float(final_best_score))
    value = float(final_best_score) if has_result else None
    if status == "auto":
        status = "keep" if (has_result and _is_improvement(
            value, _best_kept_value(data, exclude_run_id=run_id)
        )) else ("discard" if has_result else "crash")

    record["metric"] = data.get("metric")
    record["final_best_score"] = value if has_result else CRASH_SENTINEL
    record["status"] = status
    if candidate_name:
        record["candidate_name"] = candidate_name
    if description:
        record["description"] = description
    transfer_errors = validate_parameter_transfer_binding(data, record)
    if transfer_errors:
        raise ValueError("; ".join(transfer_errors))
    _capture_transfer_parent_snapshot(data, record)
    # The screening observation is frozen HERE, where final_best_score still
    # equals the step 0+1 warm score. A repair may replace a provisional crash
    # with this finite result; a later Phase-C close uses a different path and
    # cannot rewrite the observation.
    _capture_attempt_observation(data, record)
    after_graph_value = (record.get("status"), record.get("final_best_score"))
    if after_graph_value != before_graph_value:
        _touch_dag_record(data, record)
    _save_ledger(ledger_path, data)
    _write_loop_state(ledger_path, data, config)
    return record


def resolve_unevaluated(
    ledger_path: Path,
    task_name: str,
    run_id: str,
) -> dict:
    """Resolve one unstarted admission when the strict objective cap is full.

    This is deliberately separate from ``record-run``: ``unevaluated`` is not
    a model-authored outcome and carries no score.  The helper proves both that
    the global budget is exhausted and that this candidate owns zero objective
    attempts, snapshots the strict accounting log, then records an
    evidence-neutral terminal lifecycle state.
    """
    config = load_task_config(task_name)
    ledger_path = Path(ledger_path)
    data = _load_ledger(ledger_path)
    record = _get_record(data, run_id)
    if record is None:
        raise ValueError(f"no record for run_id {run_id}; add-record first")
    _require_p1_record(record, run_id)
    if record.get("status") == "unevaluated":
        return record
    if record.get("status") != "pending":
        raise ValueError(
            f"record {run_id} must be pending before unevaluated resolution"
        )
    _preserve_descendant_bindings(data, run_id)

    strict = budget_status(ledger_path.parent, create=True)
    if strict.get("reached") is not True or not isinstance(
        strict.get("budget"), int
    ):
        raise ValueError(
            "unevaluated resolution requires a configured, exhausted "
            "objective budget"
        )
    candidate_attempts = next(
        (
            row.get("evals")
            for row in strict.get("per_candidate", [])
            if row.get("run_id") == run_id
        ),
        0,
    )
    if candidate_attempts != 0:
        raise ValueError(
            f"record {run_id} has {candidate_attempts} objective attempt(s); "
            "it cannot be marked unevaluated"
        )
    if record.get("final_best_score") is not None:
        raise ValueError(
            f"record {run_id} already carries a score and cannot be marked unevaluated"
        )

    attempt_log = Path(strict["attempt_log"])
    attempt_log_sha256 = (
        "sha256:" + hashlib.sha256(attempt_log.read_bytes()).hexdigest()
    )
    receipt = {
        "schema_version": 1,
        "kind": "budget_exhausted_before_candidate_attempt",
        "budget": strict["budget"],
        "evaluations_done": strict["evaluations_done"],
        "candidate_objective_attempts": 0,
        "attempt_log": attempt_log.name,
        "attempt_log_sha256": attempt_log_sha256,
    }
    record["metric"] = data.get("metric")
    record["status"] = "unevaluated"
    record["final_best_score"] = None
    record["unevaluated_receipt"] = receipt
    _touch_dag_record(data, record)
    _save_ledger(ledger_path, data)
    _write_loop_state(ledger_path, data, config)
    return record


def cmd_record_run(args) -> int:
    task_name = args.task or infer_task_name([Path(args.ledger)])
    record = record_run(
        Path(args.ledger),
        task_name,
        args.run_id,
        final_best_score=args.final_best_score,
        status=args.status,
        candidate_name=args.candidate_name,
        description=args.description,
    )
    print(json.dumps(record, indent=2))
    return 0


def cmd_resolve_unevaluated(args) -> int:
    task_name = args.task or infer_task_name([Path(args.ledger)])
    if not task_name:
        raise SystemExit("could not infer task; pass --task")
    try:
        record = resolve_unevaluated(
            Path(args.ledger),
            task_name,
            args.run_id,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(record, indent=2))
    return 0


def cmd_percentile(args) -> int:
    data = _load_ledger(Path(args.ledger))
    result = _percentile(data, args.run_id, args.field)
    result.update(_phase_b_decision(result))  # always carries the Phase B verdict
    print(json.dumps(result))
    return 0


def cmd_evaluations(args) -> int:
    ledger_path = Path(args.ledger)
    data = _load_ledger(ledger_path)
    result = _evaluations_done(data, ledger_path)
    if args.budget is not None:
        result["budget"] = args.budget
        result["remaining"] = max(0, args.budget - result["evaluations_done"])
        result["reached"] = result["evaluations_done"] >= args.budget
    print(json.dumps(result))
    return 0


def cmd_brief(args) -> int:
    """Emit the small coordinator view; omit records and large experience text."""
    ledger_path = Path(args.ledger)
    data = _load_ledger(ledger_path)
    try:
        result = _brief_view(data, ledger_path, budget_override=args.budget)
    except RunConfigError:
        raise
    except ValueError as exc:
        raise SystemExit(f"invalid experience refresh state: {exc}") from None
    print(json.dumps(result, separators=(",", ":")))
    return 0


def cmd_set_phase(args) -> int:
    """Persist a blocked/running state; completion is budget-derived only."""
    ledger_path = Path(args.ledger)
    data = _load_ledger(ledger_path)
    if args.phase == "completed":
        try:
            refresh = _experience_refresh_status(data)
        except ValueError as exc:
            raise SystemExit(f"invalid experience refresh state: {exc}") from None
        if refresh["semantic_admission_blocked"]:
            detail = (
                "pending records must be resolved before that refresh can run"
                if not refresh["all_records_terminal"]
                else "run the final experience refresh first"
            )
            raise SystemExit(
                "cannot mark completed while a terminal DAG delta remains "
                f"unprocessed; {detail}"
            )
        budget = args.budget if args.budget is not None else _framework_budget(ledger_path)
        attempted = _evaluations_done(data, ledger_path)["evaluations_done"]
        # Any positive remainder is still spendable: got_select's admission
        # cap governs only NEW candidate admission, while an admitted
        # candidate's Phase C reserves per trial and can consume the tail.
        # Permitting completion here would discard real evaluations, and
        # close_exhausted_stage already holds the matching strict line
        # (it refuses to close a running stage while remaining > 0).
        if budget is None or attempted < budget:
            raise SystemExit(
                "cannot mark completed before a configured evaluation budget is reached "
                f"(attempted={attempted}, budget={budget})"
            )
    if args.phase == "blocked" and not args.stop_condition:
        raise SystemExit("--stop-condition is required for phase=blocked")
    data["run_state"] = {
        "phase": args.phase,
        "active_stop_condition": args.stop_condition or (
            "evaluation_budget_reached" if args.phase == "completed" else "none"
        ),
    }
    if args.phase == "completed" and budget is not None:
        data["run_state"]["evaluation_budget"] = budget
    _save_ledger(ledger_path, data)
    task_name = args.task or infer_task_name([ledger_path]) or data.get("task")
    config = load_task_config(task_name) if task_name else {}
    _write_loop_state(ledger_path, data, config)
    print(json.dumps(data["run_state"], separators=(",", ":")))
    return 0


def cmd_loop_state(args) -> int:
    ledger_path = Path(args.ledger)
    task_name = args.task or infer_task_name([ledger_path]) or (_load_ledger(ledger_path).get("task"))
    config = load_task_config(task_name) if task_name else {}
    data = _load_ledger(ledger_path)
    state_path = _write_loop_state(ledger_path, data, config)
    print(f"wrote {state_path}")
    return 0


def _set_experience(
    ledger_path: Path,
    validated_ledger: dict,
    experience: dict,
    *,
    validated_dag_revision: int,
) -> None:
    """Overwrite the derived experience snapshot and advance its DAG cursor.

    The cursor is helper-owned rather than model-authored.  It is written only
    after the caller has validated the complete replacement snapshot, so a
    failed extraction cannot acknowledge graph changes it did not process.
    """
    data = _load_ledger(ledger_path)
    if data != validated_ledger or _current_dag_revision(data) != validated_dag_revision:
        raise ValueError(
            "ledger changed after experience validation; rerun extraction against "
            "the current DAG revision"
        )
    experience = dict(experience)
    experience["dag_revision"] = validated_dag_revision
    data["experience"] = experience
    _save_ledger(ledger_path, data)


def cmd_set_experience(args) -> int:
    from background_contract import (
        load_registry,
        validate_background_markdown,
        validate_experience,
        validate_experience_replacement,
        validate_registry,
    )
    from semantic_space import (
        SemanticSpaceError,
        resolve_dimension_catalog,
        resolve_dimension_strategy,
    )

    ledger_path = Path(args.ledger)
    background_path = Path(args.background)
    data = _load_ledger(ledger_path)
    try:
        refresh = _experience_refresh_status(data)
    except ValueError as exc:
        raise SystemExit(f"invalid experience refresh state: {exc}") from None
    if not refresh["experience_refresh_required"]:
        reason = (
            "a record is still pending"
            if data.get("records") and not refresh["all_records_terminal"]
            else "there is no unprocessed terminal DAG delta"
        )
        raise SystemExit(f"experience refresh rejected: {reason}")
    validated_dag_revision = refresh["dag_revision"]
    registry = load_registry(background_path)
    try:
        dimension_strategy = resolve_dimension_strategy(background_path)
        catalog = resolve_dimension_catalog(
            background_path,
            explicit_path=Path(args.catalog) if args.catalog else None,
        )
    except SemanticSpaceError as exc:
        raise SystemExit(f"invalid dimension catalog: {exc}") from exc
    experience = json.loads(Path(args.from_json).read_text())
    errors = validate_registry(
        registry,
        ledger=data,
        catalog=catalog,
        dimension_strategy=dimension_strategy,
    )
    errors.extend(validate_background_markdown(background_path, registry))
    if not errors:
        errors.extend(validate_experience(experience, registry, data))
        errors.extend(validate_experience_replacement(experience, data))
    if errors:
        raise SystemExit("invalid P2 experience replacement: " + "; ".join(errors))
    try:
        _set_experience(
            ledger_path,
            data,
            experience,
            validated_dag_revision=validated_dag_revision,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    keys = list(experience.keys()) if isinstance(experience, dict) else None
    prior = data.get("experience")
    metadata_fields = {"generation", "updated_at_run", "dag_revision"}
    belief_changed = not isinstance(prior, dict) or {
        key: value for key, value in prior.items() if key not in metadata_fields
    } != {
        key: value for key, value in experience.items() if key not in metadata_fields
    }
    print(
        json.dumps(
            {
                "ok": True,
                "experience_keys": keys,
                "dag_revision": validated_dag_revision,
                "belief_changed": belief_changed,
            }
        )
    )
    return 0


def cmd_apply_space_state(args) -> int:
    """Atomically apply the current experience's deterministic pruning transitions.

    Loads and validates the background, ledger, stored experience, and current
    overlay; derives and appends the transitions; re-validates the resulting
    ledger; and saves once.  An empty transition set is a successful no-op.
    Kept separate from ``set-experience`` so a failed policy transition never
    corrupts or hides the replaceable belief snapshot.
    """
    from background_contract import (
        load_registry,
        validate_background_markdown,
        validate_experience,
        validate_registry,
    )
    from semantic_space import (
        SemanticSpaceError,
        resolve_dimension_catalog,
        resolve_dimension_strategy,
    )

    ledger_path = Path(args.ledger)
    background_path = Path(args.background)
    data = _load_ledger(ledger_path)
    registry = load_registry(background_path)
    try:
        dimension_strategy = resolve_dimension_strategy(background_path)
        catalog = resolve_dimension_catalog(
            background_path,
            explicit_path=Path(args.catalog) if args.catalog else None,
        )
    except SemanticSpaceError as exc:
        raise SystemExit(f"invalid dimension catalog: {exc}") from exc
    errors = validate_registry(
        registry,
        ledger=data,
        catalog=catalog,
        dimension_strategy=dimension_strategy,
    )
    errors.extend(validate_background_markdown(background_path, registry))
    experience = data.get("experience")
    if isinstance(experience, dict):
        errors.extend(validate_experience(experience, registry, data))
    if errors:
        raise SystemExit("invalid P2 background/ledger/experience: " + "; ".join(errors))

    state = data.get("search_space_state")
    prior_revision = state.get("revision", 0) if isinstance(state, dict) else 0
    decisions = append_experience_transitions(registry, data)
    errors = validate_registry(
        registry,
        ledger=data,
        catalog=catalog,
        dimension_strategy=dimension_strategy,
    )
    if errors:
        raise SystemExit("invalid search-space transition: " + "; ".join(errors))
    _save_ledger(ledger_path, data)
    state = data.get("search_space_state")
    revision = state.get("revision", prior_revision) if isinstance(state, dict) else prior_revision
    print(
        json.dumps(
            {
                "ok": True,
                "prior_revision": prior_revision,
                "revision": revision,
                "decision_ids": [decision["decision_id"] for decision in decisions],
            }
        )
    )
    return 0


def cmd_show(args) -> int:
    data = _load_ledger(Path(args.ledger))
    if args.experience:
        print(json.dumps(data.get("experience"), indent=2))
    elif args.run_id:
        record = _get_record(data, args.run_id)
        print(json.dumps(record, indent=2) if record else "null")
    else:
        print(json.dumps(data, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ledger", required=True, help="Path to ledger.json")
    common.add_argument("--task", help="Task name; inferred from the ledger path when omitted")

    add = sub.add_parser("add-record", parents=[common])
    add.add_argument("--run-id", required=True)
    add.add_argument("--kind", default="optimization", choices=["optimization"])
    add.add_argument("--idea", required=True,
                     help="RESULT: self-contained description of this solution, NO parent references (DAG node label)")
    add.add_argument("--change", required=True,
                     help="PROCESS: how this candidate changes from its parent(s) (DAG edge label); "
                          "fresh -> 'from scratch at <point-id>'")
    add.add_argument("--source-run-ids", default="",
                     help="comma-separated numeric parent run_ids; fresh must be empty")
    add.add_argument("--op", required=True, choices=["fresh", "improve", "crossover"],
                     help="S-GoT op; inferable from parent count but stored for clarity")
    add.add_argument("--background", required=True,
                     help="hierarchical background.md that freezes this run's search space")
    add.add_argument("--catalog", help="explicit dimension catalog override")
    add.add_argument("--semantic-point", required=True, type=Path,
                     help="validated complete semantic-point JSON selected for this candidate")
    add.add_argument("--policy-receipt", required=True, type=Path,
                     help="semantic acquisition receipt JSON kept separate from observations")
    add.add_argument("--candidate-name-hint", required=True)
    add.add_argument("--description")
    add.add_argument("--route-provenance", type=Path,
                     help="planned route sketches, preference order, and chosen route; "
                          "required when the run's route arm is enabled")
    add.set_defaults(func=cmd_add_record)

    tune = sub.add_parser("set-tuning", parents=[common])
    tune.add_argument("--run-id", required=True)
    tune.add_argument("--from-report", type=Path,
                      help="Fill ALL tuning fields from a tune_report.json "
                           "(supersedes the individual flags below).")
    for name in ("best-warm-score",
                 "n-dims", "warm-start-k", "warm-percentile",
                 "phase-b-decision", "phase-c-method", "trials-completed",
                 "trials-attempted", "preflight-attempts",
                 "preflight-failures", "feasibility-rejections",
                 "elapsed-seconds", "applied"):
        tune.add_argument(f"--{name}")
    tune.add_argument("--mark-tuned", action="store_true",
                      help="disabled compatibility flag; deep tuning must close through "
                           "tools/finalize_tuning.py")
    tune.set_defaults(func=cmd_set_tuning)

    exp = sub.add_parser("set-experience", parents=[common])
    exp.add_argument("--background", required=True,
                     help="hierarchical background.md used to validate the ledger and belief view")
    exp.add_argument("--catalog", help="explicit dimension catalog override")
    exp.add_argument("--from-json", required=True, type=Path,
                     help="JSON file with the experience block to store (overwrites).")
    exp.set_defaults(func=cmd_set_experience)

    apply_state = sub.add_parser("apply-space-state", parents=[common])
    apply_state.add_argument("--background", required=True,
                             help="hierarchical background.md that freezes this run's search space")
    apply_state.add_argument("--catalog", help="explicit dimension catalog override")
    apply_state.set_defaults(func=cmd_apply_space_state)

    run = sub.add_parser("record-run", parents=[common])
    run.add_argument("--run-id", required=True)
    run.add_argument("--final-best-score")
    run.add_argument("--status", default="auto", choices=["auto", "keep", "discard", "crash"])
    run.add_argument("--candidate-name")
    run.add_argument("--description")
    run.set_defaults(func=cmd_record_run)

    unevaluated = sub.add_parser("resolve-unevaluated", parents=[common])
    unevaluated.add_argument("--run-id", required=True)
    unevaluated.set_defaults(func=cmd_resolve_unevaluated)

    pct = sub.add_parser("percentile", parents=[common])
    pct.add_argument("--run-id", required=True)
    pct.add_argument("--field", default="best_warm_score")
    pct.set_defaults(func=cmd_percentile)

    ev = sub.add_parser("evaluations", parents=[common])
    ev.add_argument("--budget", type=int, default=None,
                    help="optional max_evaluations; adds remaining/reached to the output")
    ev.set_defaults(func=cmd_evaluations)

    brief = sub.add_parser("brief", parents=[common],
                           help="Print the compact coordinator view without full records")
    brief.add_argument("--budget", type=int, default=None)
    brief.set_defaults(func=cmd_brief)

    phase = sub.add_parser("set-phase", parents=[common])
    phase.add_argument("--phase", required=True, choices=["running", "blocked", "completed"])
    phase.add_argument("--stop-condition")
    phase.add_argument("--budget", type=int, default=None,
                       help="explicit evaluation budget when framework_cfg.json has none")
    phase.set_defaults(func=cmd_set_phase)

    state = sub.add_parser("loop-state", parents=[common])
    state.set_defaults(func=cmd_loop_state)

    show = sub.add_parser("show", parents=[common])
    show.add_argument("--run-id")
    show.add_argument("--experience", action="store_true",
                      help="Print just the experience block.")
    show.set_defaults(func=cmd_show)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
