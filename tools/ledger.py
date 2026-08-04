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
from collections import Counter
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

from evaluation_budget import budget_status
from run_cfg import read_framework_cfg
from search_space_state import (
    append_experience_transitions,
    empty_search_space_state,
    runtime_status_counts,
)
from semantic_evidence import (
    DIRECT_COMPARATOR_CAPABILITY,
    DIRECT_COMPARATOR_CAPABILITY_KEY,
    LIFECYCLE_TERMINAL_STATUSES,
    acquisition_conditioning,
    acquisition_target_relations,
    mechanical_gain_directions,
    unbound_primary_descendants,
    validate_conditioning_against_ledger,
    validate_conditioned_adjustment,
    validate_parameter_transfer_binding,
)
from validate_tasks import ROOT, parse_task_toml


# Field order is the on-disk record layout. Keep stable; do not rename keys.
RECORD_FIELDS = (
    "run_id",
    "kind",              # always optimization; a provided baseline is an ordinary fresh root
    "idea",              # RESULT: self-contained description of THIS solution — no parent references (DAG node label)
    "change",            # PROCESS: parent-relative implementation change; fresh -> from scratch at the selected point
    "source_run_ids",    # numeric parent run_ids only; fresh=[]
    "op",                # S-GoT op: fresh | improve | crossover (derivable from resolvable-parent count; stored for clarity)
    "semantic_point",    # complete revisioned attribution to the frozen background search space
    "semantic_edges",    # helper-derived per-parent receipts; never model-authored
    "policy_receipt",    # derived point-selection inputs/config; separate from ancestry and observations
    "parameter_transfer",  # exact parent-incumbent projection + scored inherited control
    "applied_incumbent",  # exact params/schema/files represented by this record's score
    "candidate_name",    # stable name: hint at add-record, log's best_model after run
    "description",
    "metric",
    "tune",              # bool: has the candidate been through the tuner
    "status",            # pending | keep | discard | crash | unevaluated
    "unevaluated_receipt",  # helper proof: budget exhausted before this run attempted
    "best_warm_score",
    "final_best_score",
    "evaluation_depth",  # screening | tuned_lightly | tuned: graded by cumulative Phase-C attempts
    "n_dims",
    "warm_start_K",
    "warm_percentile",
    "phase_b_decision",  # continue | stop | null
    "phase_c_method",    # grid | bo | cmaes | null
    "trials_completed",  # finite score observations
    "trials_attempted",  # all config->score calls, including failures
    "preflight_attempts",  # no-score candidate feasibility calls
    "preflight_failures",  # failed no-score feasibility calls
    "feasibility_rejections",  # Phase-C proposals rejected before score_fn
    "elapsed_seconds",
    "applied",           # bool | null: tuned params applied to BASE_PARAMS
    "dag_revision",      # last score/status revision visible to the development DAG
    "tuning_bouts",      # int: completed progressive-tuning bouts (0 = never tuned)
    "last_bout_improved",  # bool | null: last bout beat its pre-bout incumbent
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

# Phase B continue/stop policy (single source of truth; no longer in prose).
COLD_START_N = 10       # fewer prior records with the field than this → continue
TOP_PERCENTILE = 90     # otherwise continue iff percentile >= this (top 10%)


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


def _candidate_dir_for(ledger_path: Path, task_name: str, tag: str, run_id: str, config: dict) -> str:
    candidate = config.get("candidate", {})
    if not isinstance(candidate, dict):
        candidate = {}
    template = candidate.get("root_template", "runs/{task_name}/{tag}/candidates/{run_id}")
    if not isinstance(template, str):
        template = "runs/{task_name}/{tag}/candidates/{run_id}"
    try:
        relative = template.format(task_name=task_name, tag=tag, run_id=run_id)
    except KeyError:
        return (ledger_path.parent / "candidates" / run_id).as_posix()
    return (ROOT / relative).as_posix()


# ---------- ledger I/O ----------


def _load_ledger(path: Path) -> dict:
    if path.exists() and path.stat().st_size > 0:
        with open(path) as f:
            data = json.load(f)
        data.setdefault("records", [])
        for record in data["records"]:
            if not isinstance(record, dict):
                continue
            # Progressive-tuning fields, additive: a legacy one-shot-tuned
            # record counts as one completed bout with unknown response.
            record.setdefault(
                "tuning_bouts", 1 if record.get("tune") else 0
            )
            record.setdefault("last_bout_improved", None)
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


def _get_record(data: dict, run_id: str) -> Optional[dict]:
    for record in data["records"]:
        if record.get("run_id") == run_id:
            return record
    return None


def _require_p1_record(record: dict, run_id: str) -> None:
    """Reject mixed/legacy records on every downstream mutation path."""
    if not isinstance(record.get("semantic_point"), dict):
        raise ValueError(
            f"record {run_id} has no semantic_point; legacy or mixed-mode records are unsupported"
        )
    if not isinstance(record.get("policy_receipt"), dict):
        raise ValueError(
            f"record {run_id} has no policy_receipt; selection policy must remain traceable"
)


def _new_record(run_id: str) -> dict:
    return {field: None for field in RECORD_FIELDS} | {
        "run_id": run_id,
        "tune": False,
        "tuning_bouts": 0,
        "last_bout_improved": None,
        "status": "pending",
    }


def _current_dag_revision(data: dict) -> int:
    return int(data.get("dag_revision", 0))


def _experience_refresh_status(data: dict) -> dict:
    """Return the single deterministic refresh/admission state."""
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
        # Admission/planning closes as soon as any terminal observation is
        # newer than the belief cursor, even when sibling records from the
        # already-admitted batch are still pending.  Pending siblings delay the
        # refresh itself; they do not authorize another semantic admission.
        "semantic_admission_blocked": delta > 0,
        "experience_refresh_required": all_terminal and delta > 0,
    }


def _touch_dag_record(data: dict, record: dict) -> int:
    """Mark one new or score-updated DAG node with the next revision."""
    current = _current_dag_revision(data) + 1
    data["dag_revision"] = current
    record["dag_revision"] = current
    return current


# ---------- derived computations ----------


def _is_improvement(value: float, best: Optional[float]) -> bool:
    """Scores are always lower-is-better: a value improves iff it is smaller."""
    if best is None:
        return True
    return value < best


def _best_kept_value(data: dict, exclude_run_id: Optional[str] = None) -> Optional[float]:
    # exclude_run_id: when re-recording a record in place (e.g. the decoupled
    # deep-tune updates a candidate's score), the keep/discard decision must
    # compare against OTHER kept records, not the record's own prior keep —
    # otherwise re-recording the best candidate with its own score is not a
    # strict improvement over itself and wrongly flips it to discard.
    values = [
        r["final_best_score"]
        for r in data["records"]
        if r.get("status") == "keep"
        and isinstance(r.get("final_best_score"), (int, float))
        and math.isfinite(float(r["final_best_score"]))
        and r.get("run_id") != exclude_run_id
    ]
    return min(values) if values else None


def _best_kept_record(data: dict) -> Optional[dict]:
    best: Optional[dict] = None
    best_value: Optional[float] = None
    for record in data["records"]:
        if record.get("status") != "keep":
            continue
        value = record.get("final_best_score")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        if _is_improvement(value, best_value):
            best_value = value
            best = record
    return best


def _next_run_id(data: dict) -> str:
    numeric = [int(r["run_id"]) for r in data["records"] if str(r.get("run_id", "")).isdigit()]
    widths = [3] + [len(r["run_id"]) for r in data["records"] if str(r.get("run_id", "")).isdigit()]
    if not numeric:
        return "000"
    return f"{max(numeric) + 1:0{max(widths)}d}"


def _percentile(data: dict, run_id: str, field: str) -> dict:
    """Percentile rank of run_id's `field` among prior records that have it.

    Scores are always lower-is-better, so the percentile is the fraction of
    prior values strictly worse (greater) than this one — a high percentile
    means among the best. Used by tuner Phase B (field=best_warm_score).
    Returns {value, count_prior, percentile|null}.
    """
    target = _get_record(data, run_id)
    value = target.get(field) if target else None
    priors = [
        r[field]
        for r in data["records"]
        if r.get("run_id") != run_id
        and isinstance(r.get(field), (int, float))
        and math.isfinite(float(r[field]))
    ]
    if (
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not priors
    ):
        return {"value": value, "count_prior": len(priors), "percentile": None}
    below = sum(1 for p in priors if p > value)
    return {
        "value": value,
        "count_prior": len(priors),
        "percentile": round(100.0 * below / len(priors)),
    }


def _phase_b_decision(pct: dict) -> dict:
    """Continue/stop verdict from a _percentile() result. Pure; the 10/90
    thresholds are the module constants, not prose. Cold-start is driven by
    count_prior (not by percentile being null), so a missing-priors case
    (count_prior == 0) continues, while a missing-current-value case with
    enough priors fails open (continue) rather than wrongly skipping.
    """
    n = pct["count_prior"]
    if n < COLD_START_N:
        return {"decision": "continue",
                "reason": f"cold-start: {n} < {COLD_START_N} prior records with the field"}
    p = pct["percentile"]
    if p is None:
        return {"decision": "continue",
                "reason": "current value not recorded; cannot rank, continuing"}
    if p >= TOP_PERCENTILE:
        return {"decision": "continue",
                "reason": f"top {100 - TOP_PERCENTILE}%: percentile {p} >= {TOP_PERCENTILE}"}
    return {"decision": "stop",
            "reason": f"below top {100 - TOP_PERCENTILE}%: percentile {p} < {TOP_PERCENTILE}"}


# ---------- loop_state.md (derived) ----------


def _fmt_score(value: Any) -> str:
    return f"{value:.6f}" if isinstance(value, (int, float)) else "none"


def _write_loop_state(ledger_path: Path, data: dict, config: dict) -> Path:
    task_name = data.get("task") or "unknown"
    tag = data.get("tag") or ledger_path.parent.name
    metric = data.get("metric") or "score"
    records = data["records"]
    last = records[-1] if records else {}
    best = _best_kept_record(data)
    phase, stop_condition = _run_phase(ledger_path, data)

    best_run_id = best.get("run_id", "none") if best else "none"
    best_dir = (
        _candidate_dir_for(ledger_path, task_name, tag, best_run_id, config)
        if best else "none"
    )
    lines = [
        f"task: {task_name}",
        f"tag: {tag}",
        f"phase: {phase}",
        f"next_run_id: {_next_run_id(data)}",
        f"best_run_id: {best_run_id}",
        f"best_score: {_fmt_score(best.get('final_best_score')) if best else 'none'}",
        f"metric: {metric}",
        f"best_candidate_dir: {best_dir}",
        f"last_run_id: {last.get('run_id', 'none')}",
        f"last_status: {last.get('status', 'none')}",
        f"last_score: {_fmt_score(last.get('final_best_score'))}",
        f"active_stop_condition: {stop_condition}",
        f"notes: {last.get('description') or 'none'}",
    ]
    state_path = ledger_path.parent / "loop_state.md"
    state_path.write_text("\n".join(lines) + "\n")
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
    # Import lazily so read-only ledger commands do not pay background-contract
    # setup cost and to keep the mutation boundary explicit.
    from background_contract import (
        load_registry,
        validate_background_markdown,
        validate_registry,
    )
    from semantic_evidence import SemanticEvidenceError, build_semantic_edges
    from semantic_space import (
        SemanticSpaceError,
        digest,
        resolve_dimension_catalog,
        resolve_dimension_strategy,
        space_receipt,
    )

    ledger_path = Path(args.ledger)
    task_name = args.task or infer_task_name([ledger_path])
    if not task_name:
        raise SystemExit("could not infer task; pass --task")
    config = load_task_config(task_name)
    data = _load_ledger(ledger_path)
    _ensure_meta(data, ledger_path, task_name, config)
    try:
        refresh = _experience_refresh_status(data)
    except ValueError as exc:
        raise SystemExit(f"invalid experience refresh state: {exc}") from None
    if refresh["semantic_admission_blocked"]:
        raise SystemExit(
            "stale experience: terminal DAG evidence must be refreshed before "
            "another semantic candidate is admitted; resolve any already-pending "
            "siblings first"
        )

    background_path = Path(args.background)
    registry = load_registry(background_path)
    try:
        dimension_strategy = resolve_dimension_strategy(background_path)
        catalog = resolve_dimension_catalog(
            background_path,
            explicit_path=Path(args.catalog) if args.catalog else None,
        )
    except SemanticSpaceError as exc:
        raise SystemExit(f"invalid dimension catalog: {exc}") from exc
    background_errors = validate_registry(
        registry,
        ledger=data,
        catalog=catalog,
        dimension_strategy=dimension_strategy,
    )
    background_errors.extend(validate_background_markdown(background_path, registry))
    if background_errors:
        raise SystemExit("invalid P2 background/ledger: " + "; ".join(background_errors))

    if _get_record(data, args.run_id) is not None:
        raise SystemExit(f"record already exists for run_id {args.run_id}")
    record = _new_record(args.run_id)
    source_run_ids = [x.strip() for x in (args.source_run_ids or "").split(",") if x.strip()]
    if any(not source.isdigit() for source in source_run_ids):
        raise SystemExit(
            "--source-run-ids accepts numeric parents only; hypothesis attribution "
            "belongs in --semantic-point"
        )
    expected_parent_count = {"fresh": 0, "improve": 1, "crossover": 2}.get(args.op)
    if expected_parent_count is None:
        raise SystemExit("--op is required")
    if len(source_run_ids) != expected_parent_count or len(set(source_run_ids)) != len(source_run_ids):
        raise SystemExit(
            f"{args.op} requires {expected_parent_count} distinct numeric parent ids"
        )
    semantic_point = json.loads(Path(args.semantic_point).read_text())
    policy_receipt = json.loads(Path(args.policy_receipt).read_text())
    record.update(
        kind=args.kind,
        idea=args.idea,
        change=args.change,
        source_run_ids=source_run_ids,
        op=args.op,
        semantic_point=semantic_point,
        semantic_edges=[],
        policy_receipt=policy_receipt,
        candidate_name=args.candidate_name_hint,
        description=args.description or args.idea,
        metric=data["metric"],
    )
    try:
        record["semantic_edges"] = build_semantic_edges(data["records"], record)
    except SemanticEvidenceError as exc:
        raise SystemExit(f"invalid candidate semantic contract: {exc}") from exc
    data["search_space"] = data.get("search_space") or space_receipt(registry)
    data["search_space_state"] = data.get("search_space_state") or empty_search_space_state()
    current_revision = data["search_space_state"].get("revision", 0)
    if policy_receipt.get("search_space_state_revision") != current_revision:
        raise SystemExit(
            "stale policy receipt: search_space_state_revision "
            f"{policy_receipt.get('search_space_state_revision')!r} does not equal the "
            f"current search space state revision {current_revision}; re-propose and "
            "re-select against the current overlay before admission"
        )
    if policy_receipt.get("schema_version") != 6:
        raise SystemExit(
            "new candidate admission requires policy receipt schema 6 with "
            "proposal-relevant gated experience conditioning and an auditable "
            "LLM-judgment reliability prior; historical schema-2/3/4/5 "
            "receipts remain readable"
        )
    current_experience = data.get("experience")
    if isinstance(current_experience, dict) and current_experience:
        expected_experience = {
            "generation": current_experience.get("generation"),
            "updated_at_run": current_experience.get("updated_at_run"),
            "revision": digest(current_experience),
        }
    else:
        expected_experience = {
            "generation": None,
            "updated_at_run": None,
            "revision": None,
        }
    receipt_experience = policy_receipt.get("experience")
    actual_experience = (
        {
            "generation": receipt_experience.get("generation"),
            "updated_at_run": receipt_experience.get("updated_at_run"),
            "revision": receipt_experience.get("revision"),
        }
        if isinstance(receipt_experience, dict)
        else None
    )
    if actual_experience != expected_experience:
        raise SystemExit(
            "stale policy receipt: experience generation/revision does not match "
            "the ledger's current bounded belief; rebuild gain-context and re-select"
        )
    if isinstance(receipt_experience, dict):
        by_run_id = {
            str(item.get("run_id")): item
            for item in data.get("records", [])
            if isinstance(item, dict)
        }
        parent_points = [
            by_run_id[parent_id].get("semantic_point")
            for parent_id in source_run_ids
            if parent_id in by_run_id
        ]
        target_relations = acquisition_target_relations(
            semantic_point,
            None
            if args.op == "fresh"
            else [
                point
                for point in parent_points
                if isinstance(point, dict)
            ],
        )
        available_conditioning = acquisition_conditioning(
            current_experience,
            semantic_point,
            target_relations=target_relations,
            gain_directions=mechanical_gain_directions(
                data, current_experience
            ),
        )
        available_by_target = {
            item["target_id"]: item for item in available_conditioning
        }
        conditioning = receipt_experience.get("conditioning")
        conditioning = conditioning if isinstance(conditioning, list) else []
        if any(
            not isinstance(item, dict)
            or item.get("target_id") not in available_by_target
            or item != available_by_target[item.get("target_id")]
            for item in conditioning
        ):
            raise SystemExit(
                "invalid policy receipt: experience conditioning must be an "
                "exact proposal-relevant helper rendering"
            )
        ledger_conditioning_errors = validate_conditioning_against_ledger(
            conditioning, data
        )
        if ledger_conditioning_errors:
            raise SystemExit(
                "invalid policy receipt: "
                + "; ".join(ledger_conditioning_errors)
            )
        cited_run_ids = receipt_experience.get("evidence_run_ids")
        cited_edge_ids = receipt_experience.get("evidence_edge_ids")
        policy = policy_receipt.get("policy")
        policy_name = policy.get("name") if isinstance(policy, dict) else None
        if policy_name in {"gain", "gain_uncertainty", "gain_uncertainty_nocost"}:
            components = policy_receipt.get("components")
            conditioning_errors = validate_conditioned_adjustment(
                conditioning,
                semantic_point,
                evidence_run_ids=cited_run_ids,
                evidence_edge_ids=cited_edge_ids,
                gain_adjustment=(
                    components.get("experience_gain_adjustment")
                    if isinstance(components, dict)
                    else None
                ),
                uncertainty_adjustment=(
                    components.get("experience_uncertainty_adjustment")
                    if isinstance(components, dict)
                    else None
                ),
            )
            if conditioning_errors:
                raise SystemExit(
                    "invalid policy receipt: "
                    + "; ".join(conditioning_errors)
                )
    budget = policy_receipt.get("budget")
    expected_selection_index = len(data["records"]) + 1
    if (
        not isinstance(budget, dict)
        or budget.get("selection_index") != expected_selection_index
    ):
        raise SystemExit(
            "policy receipt budget selection_index must equal the next one-based "
            f"admission index {expected_selection_index}"
        )
    data["records"].append(record)
    contract_errors = validate_registry(
        registry,
        ledger=data,
        catalog=catalog,
        dimension_strategy=dimension_strategy,
    )
    if contract_errors:
        data["records"].pop()
        raise SystemExit("invalid candidate semantic contract: " + "; ".join(contract_errors))
    _save_ledger(ledger_path, data)
    _write_loop_state(ledger_path, data, config)
    print(json.dumps(record, indent=2))
    return 0


def _tuning_record_from_report(report_path: Path) -> dict:
    """Derive the ledger tuning updates from a tune_report.json. The
    report-format knowledge lives in the tuner-side reducer; this module only
    consumes its dict and owns the ledger write."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tuners"))
    from tune_tools import (  # noqa: E402
        tuning_record,
        validate_phase_a_candidate_state,
    )

    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    validate_phase_a_candidate_state(
        report,
        report_path.parent / "train.py",
    )
    stages = report.get("phase_c", {}).get("stages", [])
    if stages:
        raise ValueError(
            "Phase-A tuning updates cannot consume a report with Phase-C "
            "stages; use tools/finalize_tuning.py after a finalizable close"
        )
    fields = tuning_record(report)
    phase_a = report.get("phase_a")
    fields["applied_incumbent"] = (
        _applied_incumbent_from_report(report_path)
        if isinstance(phase_a, dict) and phase_a.get("status") == "ok"
        else None
    )
    return {key: fields.get(key) for key in TUNING_FIELDS}


def _finalized_tuning_record_from_report(
    report_path: Path,
    *,
    validate_revision: bool = True,
) -> dict:
    """Read a report only after deterministic Phase-C completion validation."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tuners"))
    from tune_tools import (  # noqa: E402
        finalized_tuning_record,
        load_tuned_threshold,
        validate_report_trial_rows,
    )

    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    validate_report_trial_rows(report, report_path.parent / "train.py")
    # The report lives at <run_dir>/candidates/<run_id>/tune_report.json, so
    # parent.parent.parent is the run dir; load_tuned_threshold only reads
    # framework_cfg.json beside that ledger path and falls back to the default
    # when absent.
    ledger_guess = report_path.parent.parent.parent / "ledger.json"
    fields = finalized_tuning_record(
        report,
        tuned_threshold=load_tuned_threshold(ledger_guess),
    )
    fields["applied_incumbent"] = _applied_incumbent_from_report(
        report_path,
        require_final=True,
        validate_revision=validate_revision,
    )
    return fields


def _applied_incumbent_from_report(
    report_path: Path,
    *,
    require_final: bool = False,
    validate_revision: bool = True,
) -> dict:
    """Snapshot the exact applied candidate state represented by a report."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tuners"))
    from tune_tools import (  # noqa: E402
        _bounds_violations,
        _read_literal_mapping,
        _validate_schema_values,
        finalizable_tuning_result,
        lint_contract,
        validate_candidate_execution_revision,
        validated_phase_a_incumbent,
    )

    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    if require_final:
        result = finalizable_tuning_result(report, require_applied=True)
        params = result["best_params"]
        score = float(result["best_score"])
        source = "finalized_phase_c"
    else:
        closing_present = any(
            report.get(key) is not None
            for key in (
                "final_best_params",
                "final_best_score",
            )
        )
        if closing_present:
            result = finalizable_tuning_result(report, require_applied=True)
            params = result["best_params"]
            score = float(result["best_score"])
            source = "finalized_phase_c"
        else:
            phase_a_best = validated_phase_a_incumbent(report)
            params = phase_a_best["params"]
            score = phase_a_best["score"]
            source = "applied_phase_a"

    candidate_path = report_path.parent / "train.py"
    if not candidate_path.is_file():
        raise ValueError(
            f"cannot snapshot applied incumbent without candidate {candidate_path}"
        )
    contract = lint_contract(candidate_path)
    if not contract.get("ok"):
        details = "; ".join(
            str(error.get("detail", error))
            for error in contract.get("errors", [])
            if isinstance(error, dict)
        )
        raise ValueError(
            "candidate tuning contract is invalid"
            + (f": {details}" if details else "")
        )
    if validate_revision:
        validate_candidate_execution_revision(report, candidate_path)

    param_schema = _read_literal_mapping(candidate_path, "PARAM_SCHEMA")
    search_space = _read_literal_mapping(candidate_path, "SEARCH_SPACE")
    phase_a = report.get("phase_a")
    reported_search_space = (
        phase_a.get("search_space")
        if isinstance(phase_a, dict)
        else None
    )
    if not isinstance(reported_search_space, dict):
        raise ValueError(
            "tune report phase_a.search_space must be an object"
        )
    try:
        search_space_matches = (
            _json_sha256(reported_search_space)
            == _json_sha256(search_space)
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"tune report phase_a.search_space is not canonical JSON: {exc}"
        ) from exc
    if not search_space_matches:
        raise ValueError(
            "tune report phase_a.search_space does not match the candidate "
            "SEARCH_SPACE literal"
        )

    _validate_schema_values(
        params,
        param_schema,
        label="applied incumbent params",
    )
    violations = _bounds_violations(params, search_space)
    if violations:
        raise ValueError(
            "applied incumbent params violate the candidate SEARCH_SPACE: "
            + json.dumps(violations, ensure_ascii=False)
        )
    if _read_literal_mapping(candidate_path, "BASE_PARAMS") != params:
        raise ValueError(
            "candidate BASE_PARAMS do not equal the report's applied incumbent"
        )
    params_snapshot = _json_native(params)
    param_schema_snapshot = _json_native(param_schema)
    snapshot = {
        "schema_version": 1,
        "source": source,
        "score": score,
        "params": params_snapshot,
        "params_sha256": _json_sha256(params_snapshot),
        "param_schema": param_schema_snapshot,
        "param_schema_sha256": _json_sha256(param_schema_snapshot),
        "entrypoint_sha256": (
            "sha256:" + hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        ),
        "tune_report_sha256": (
            "sha256:" + hashlib.sha256(report_path.read_bytes()).hexdigest()
        ),
    }
    return snapshot


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


def _validate_tuning_report_ownership(
    ledger_path: Path,
    run_id: str,
    report_path: Path,
) -> Path:
    """Bind a tuning report to exactly one candidate in exactly one run."""
    ledger_path = Path(ledger_path).resolve()
    report_path = Path(report_path).resolve()
    expected_candidate_dir = ledger_path.parent / "candidates" / str(run_id)
    expected_report = expected_candidate_dir / "tune_report.json"
    if report_path != expected_report:
        raise ValueError(
            "tune report must be the candidate-owned "
            f"{expected_report}; got {report_path}"
        )
    candidate_path = expected_candidate_dir / "train.py"
    if not candidate_path.is_file():
        raise ValueError(f"candidate entrypoint does not exist: {candidate_path}")
    return candidate_path


def _capture_transfer_parent_snapshot(data: dict, child: dict) -> None:
    """Persist the exact parent revision cited by one transfer receipt."""
    transfer = child.get("parameter_transfer")
    receipt = transfer.get("receipt") if isinstance(transfer, dict) else None
    primary = receipt.get("primary_parent") if isinstance(receipt, dict) else None
    if not isinstance(primary, dict):
        return
    parent_run_id = str(primary.get("run_id"))
    record_hash = primary.get("ledger_record_sha256")
    snapshots = data.setdefault("lineage_snapshots", [])
    if not isinstance(snapshots, list):
        raise ValueError("ledger.lineage_snapshots must be a list")
    existing = [
        item
        for item in snapshots
        if isinstance(item, dict)
        and item.get("parent_run_id") == parent_run_id
        and item.get("ledger_record_sha256") == record_hash
    ]
    if existing:
        if len(existing) != 1:
            raise ValueError(
                f"parent revision {parent_run_id}/{record_hash} is duplicated"
            )
        return
    parent = _get_record(data, parent_run_id)
    if parent is None or _json_sha256(parent) != record_hash:
        raise ValueError(
            f"record {child.get('run_id')} cites a parent revision that is "
            "neither current nor durably snapshotted"
        )
    score = parent.get("final_best_score")
    applied = parent.get("applied_incumbent")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        or not isinstance(applied, dict)
    ):
        raise ValueError(
            f"record {child.get('run_id')} cannot snapshot parent {parent_run_id} "
            "without a finite score and exact applied incumbent"
        )
    entry = {
        "schema_version": 1,
        "kind": "parameter_transfer_parent_snapshot",
        "parent_run_id": parent_run_id,
        "ledger_record_sha256": record_hash,
        "final_best_score": float(score),
        "applied_incumbent": copy.deepcopy(applied),
        "captured_by_run_id": str(child.get("run_id")),
    }
    entry["receipt_sha256"] = _json_sha256(entry)
    snapshots.append(entry)


def _preserve_descendant_bindings(data: dict, run_id: str) -> None:
    """Snapshot every primary-child binding before mutating its parent.

    Which children block the mutation is decided by
    ``semantic_evidence.unbound_primary_descendants`` — the same predicate
    ``select_candidate`` uses for tuning eligibility, so a parent can never be
    selected for tuning and then rejected here (or vice versa).
    """
    target = str(run_id)
    unbound = set(unbound_primary_descendants(data, target))
    for child in data.get("records", []):
        if not isinstance(child, dict):
            continue
        child_run_id = str(child.get("run_id"))
        if child_run_id in unbound:
            continue
        parents = child.get("source_run_ids")
        if not isinstance(parents, list) or not parents or str(parents[0]) != target:
            continue
        transfer = child.get("parameter_transfer")
        receipt = transfer.get("receipt") if isinstance(transfer, dict) else None
        primary = receipt.get("primary_parent") if isinstance(receipt, dict) else None
        if not isinstance(primary, dict) or str(primary.get("run_id")) != target:
            # Settled with no binding to preserve (crash/unevaluated, no transfer).
            continue
        try:
            _capture_transfer_parent_snapshot(data, child)
        except ValueError:
            unbound.add(child_run_id)
    if unbound:
        raise ValueError(
            f"record {run_id} has in-flight or invalid primary descendants "
            f"{sorted(unbound)}; finish their parameter-transfer binding before "
            "mutating the parent"
        )


def _prospective_finalized_record(
    data: dict,
    record: dict,
    report_path: Path,
    *,
    strict_attempts: int,
    validate_revision: bool = True,
) -> dict:
    """Build and validate the exact record that a tuning close would persist."""
    finalized = _finalized_tuning_record_from_report(
        report_path,
        validate_revision=validate_revision,
    )
    updates = {key: finalized.get(key) for key in TUNING_FIELDS}
    updates["trials_attempted"] = max(
        int(updates.get("trials_attempted") or 0),
        int(strict_attempts),
    )
    score = float(finalized["final_best_score"])

    # Finalization owns the complete tuning projection, including explicit
    # nulls.  Filtering null values would preserve stale Phase-B/method fields
    # from an earlier partial close and make the ledger disagree with the
    # finalizer's proven result.
    applied_updates = updates
    already_closed = (
        record.get("tune") is True
        and record.get("metric") == data.get("metric")
        and isinstance(record.get("final_best_score"), (int, float))
        and not isinstance(record.get("final_best_score"), bool)
        and math.isfinite(float(record["final_best_score"]))
        and float(record["final_best_score"]) == score
        and all(record.get(key) == value for key, value in applied_updates.items())
    )
    if already_closed:
        prospective = copy.deepcopy(record)
    else:
        prospective = copy.deepcopy(record)
        prospective["metric"] = data.get("metric")
        prospective["final_best_score"] = score
        prospective["status"] = (
            "keep"
            if _is_improvement(
                score,
                _best_kept_value(data, exclude_run_id=str(record.get("run_id"))),
            )
            else "discard"
        )
        prospective.update(applied_updates)
        prospective["tune"] = True

    transfer_errors = validate_parameter_transfer_binding(data, prospective)
    if transfer_errors:
        raise ValueError("; ".join(transfer_errors))
    return prospective


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
            and receipt.get("schema_version") in {5, 6}
        ):
            raise SystemExit(
                "new non-fresh candidates require --from-report so the exact "
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
    after_graph_value = (record.get("status"), record.get("final_best_score"))
    if after_graph_value != before_graph_value:
        _touch_dag_record(data, record)
    _save_ledger(ledger_path, data)
    _write_loop_state(ledger_path, data, config)
    return record


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_native(value: Any) -> Any:
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
        raise ValueError(f"value is not durable canonical JSON: {exc}") from exc


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
    receipt["receipt_sha256"] = _json_sha256(receipt)
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


def _evaluations_done(data: dict, ledger_path: Path | None = None) -> dict:
    """Run-level evaluation budget used so far = all config->score attempts.

    New records carry ``trials_attempted``, including failed calls. Legacy
    records fall back to finite ``trials_completed``, then ``warm_start_K``.
    Never add those fields together: each is a successively older total.
    """
    records = data.get("records", [])
    per, total = [], 0
    for r in records:
        attempted = r.get("trials_attempted")
        if attempted is None:
            attempted = r.get("trials_completed")
        if attempted is None:
            attempted = r.get("warm_start_K") or 0
        attempted = int(attempted)
        total += attempted
        per.append({"run_id": r.get("run_id"), "evals": attempted,
                    "tuned": bool(r.get("tune")), "status": r.get("status")})
    if ledger_path is not None:
        strict = budget_status(Path(ledger_path).parent)
        total = max(total, strict["evaluations_done"])
        strict_per = {
            row["run_id"]: row["evals"]
            for row in strict.get("per_candidate", [])
        }
        for row in per:
            row["evals"] = max(row["evals"], strict_per.pop(row["run_id"], 0))
        for run_id, attempted in sorted(strict_per.items()):
            per.append(
                {
                    "run_id": run_id,
                    "evals": attempted,
                    "tuned": False,
                    "status": "pending",
                }
            )
    return {"evaluations_done": total, "n_candidates": len(records), "per_candidate": per}


def _framework_budget(ledger_path: Path) -> Optional[int]:
    """Read the run-local global evaluation budget, if one is configured."""
    path = ledger_path.parent / "framework_cfg.json"
    if not path.is_file():
        return None
    value = read_framework_cfg(path).get("max_evaluations")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _run_phase(
    ledger_path: Path,
    data: dict,
    *,
    budget_override: int | None = None,
) -> tuple[str, str]:
    """Return the truthful derived phase and stop condition for a run."""
    state = data.get("run_state") if isinstance(data.get("run_state"), dict) else {}
    budget = budget_override
    if budget is None:
        budget = _framework_budget(ledger_path)
    if budget is None:
        saved_budget = state.get("evaluation_budget")
        if isinstance(saved_budget, int) and not isinstance(saved_budget, bool):
            budget = saved_budget
    attempted = _evaluations_done(data, ledger_path)["evaluations_done"]
    if state.get("phase") == "blocked":
        return "blocked", str(state.get("active_stop_condition") or "unspecified_blocker")
    if budget is not None and attempted >= budget:
        refresh = _experience_refresh_status(data)
        if not refresh["all_records_terminal"]:
            return "running", "budget_reached_pending_resolution"
        if refresh["semantic_admission_blocked"]:
            return "running", "final_experience_refresh_required"
        return "completed", "evaluation_budget_reached"
    # Completion cannot survive new evidence that the configured budget has not
    # been reached. This prevents premature success claims.
    return "running", "none"


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
    records = data.get("records", [])
    best = _best_kept_record(data)
    last = records[-1] if records else None
    evals = _evaluations_done(data, ledger_path)
    attempted = evals["evaluations_done"]
    budget = args.budget if args.budget is not None else _framework_budget(ledger_path)
    # Derive the refresh state before `_run_phase`, which consults it too. Doing
    # it here means one corrupt cursor reports the actionable message instead of
    # surfacing the same ValueError as an uncaught traceback from phase derivation.
    try:
        refresh = _experience_refresh_status(data)
    except ValueError as exc:
        raise SystemExit(f"invalid experience refresh state: {exc}") from None
    phase, stop_condition = _run_phase(
        ledger_path,
        data,
        budget_override=args.budget,
    )
    experience = data.get("experience") if isinstance(data.get("experience"), dict) else {}
    state = data.get("search_space_state")
    state = state if isinstance(state, dict) else empty_search_space_state()
    state_counts = runtime_status_counts(state)
    result = {
        "task": data.get("task"), "tag": data.get("tag"), "metric": data.get("metric"),
        "phase": phase, "active_stop_condition": stop_condition,
        "next_run_id": _next_run_id(data), "n_candidates": len(records),
        "status_counts": dict(Counter(r.get("status") for r in records)),
        "op_counts": dict(Counter(r.get("op") for r in records)),
        "pending_run_ids": [r.get("run_id") for r in records if r.get("status") == "pending"],
        "best": None if best is None else {
            "run_id": best.get("run_id"), "score": best.get("final_best_score"),
            "candidate_name": best.get("candidate_name"),
        },
        "last": None if last is None else {
            "run_id": last.get("run_id"), "status": last.get("status"),
            "score": last.get("final_best_score"),
        },
        "evaluations_attempted": attempted,
        "preflight_attempts": sum(int(r.get("preflight_attempts") or 0) for r in records),
        "preflight_failures": sum(int(r.get("preflight_failures") or 0) for r in records),
        "feasibility_rejections": sum(
            int(r.get("feasibility_rejections") or 0) for r in records
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
        # The coordinator checks this only at a quiescent round boundary.  A
        # terminal DAG delta is refreshed before the next semantic admission;
        # pending records keep the boundary closed.
        "experience_refresh_required": refresh["experience_refresh_required"],
        "direct_comparator_capability": data.get(
            DIRECT_COMPARATOR_CAPABILITY_KEY,
            DIRECT_COMPARATOR_CAPABILITY,
        ),
        # Compact overlay summary only; the full decision log stays in show.
        "search_space_state_revision": state.get("revision", 0),
        "runtime_deprioritized_dimensions": state_counts["dimensions"]["deprioritized"],
        "runtime_pruned_dimensions": state_counts["dimensions"]["pruned"],
        "runtime_deprioritized_hypotheses": state_counts["hypotheses"]["deprioritized"],
        "runtime_pruned_hypotheses": state_counts["hypotheses"]["pruned"],
    }
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
