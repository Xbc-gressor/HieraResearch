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
      "search_space": { ... },           # exact catalog + background revision
      "dag_revision": 12,              # monotone graph-change cursor
      "records": [ {record}, ... ]   # ordered by run_id
    }

Record schema (see RECORD_FIELDS). Unavailable fields are `null`, never
omitted. `add-record` creates a record with idea fields; `set-tuning` fills
the tuning block; `record-run` fills the run result and computes `status`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

from validate_tasks import ROOT, parse_task_toml


# Field order is the on-disk record layout. Keep stable; do not rename keys.
RECORD_FIELDS = (
    "run_id",
    "kind",              # always optimization (seed species retired)
    "idea",              # RESULT: self-contained description of THIS solution — no parent references (DAG node label)
    "change",            # PROCESS: parent-relative implementation change; fresh -> from scratch at the selected point
    "source_run_ids",    # numeric parent run_ids only; fresh=[]
    "op",                # S-GoT op: fresh | improve | crossover (derivable from resolvable-parent count; stored for clarity)
    "semantic_point",    # complete revisioned attribution to the frozen background search space
    "policy_receipt",    # derived point-selection inputs/config; separate from ancestry and observations
    "candidate_name",    # stable name: hint at add-record, log's best_model after run
    "description",
    "metric",
    "tune",              # bool: has the candidate been through the tuner
    "status",            # pending | keep | discard | crash
    "best_warm_score",
    "final_best_score",
    "n_dims",
    "warm_start_K",
    "warm_percentile",
    "phase_b_decision",  # continue | stop | null
    "phase_c_method",    # grid | bo | cmaes | null
    "trials_completed",
    "elapsed_seconds",
    "applied",           # bool | null: tuned params applied to BASE_PARAMS
    "dag_revision",      # last score/status revision visible to the development DAG
)

TUNING_FIELDS = (
    "best_warm_score",
    "n_dims",
    "warm_start_K",
    "warm_percentile",
    "phase_b_decision",
    "phase_c_method",
    "trials_completed",
    "elapsed_seconds",
    "applied",
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
        return data
    return {"task": None, "tag": None, "metric": None, "records": []}


def _save_ledger(path: Path, data: dict) -> None:
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
        "status": "pending",
    }


def _current_dag_revision(data: dict) -> int:
    return int(data.get("dag_revision", 0))


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
        if r.get("status") == "keep" and isinstance(r.get("final_best_score"), (int, float))
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
        if not isinstance(value, (int, float)):
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
        if r.get("run_id") != run_id and isinstance(r.get(field), (int, float))
    ]
    if not isinstance(value, (int, float)) or not priors:
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
    from semantic_space import (
        SemanticSpaceError,
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
        raise SystemExit("invalid P1 background/ledger: " + "; ".join(background_errors))

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
    missing_parents = [
        source for source in source_run_ids if _get_record(data, source) is None
    ]
    if missing_parents:
        raise SystemExit(f"unknown parent run ids {missing_parents}")
    semantic_point = json.loads(Path(args.semantic_point).read_text())
    policy_receipt = json.loads(Path(args.policy_receipt).read_text())
    record.update(
        kind=args.kind,
        idea=args.idea,
        change=args.change,
        source_run_ids=source_run_ids,
        op=args.op,
        semantic_point=semantic_point,
        policy_receipt=policy_receipt,
        candidate_name=args.candidate_name_hint,
        description=args.description or args.idea,
        metric=data["metric"],
    )
    data["search_space"] = data.get("search_space") or space_receipt(registry)
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
    from tune_tools import tuning_record  # noqa: E402

    fields = tuning_record(json.loads(Path(report_path).read_text()))
    return {key: fields.get(key) for key in TUNING_FIELDS}


def cmd_set_tuning(args) -> int:
    ledger_path = Path(args.ledger)
    task_name = args.task or infer_task_name([ledger_path])
    config = load_task_config(task_name)
    data = _load_ledger(ledger_path)
    record = _get_record(data, args.run_id)
    if record is None:
        raise SystemExit(f"no record for run_id {args.run_id}; add-record first")
    _require_p1_record(record, args.run_id)

    if args.from_report:
        updates = _tuning_record_from_report(Path(args.from_report))
    else:
        updates = {
            "best_warm_score": _coerce(args.best_warm_score, "float"),
            "n_dims": _coerce(args.n_dims, "int"),
            "warm_start_K": _coerce(args.warm_start_k, "int"),
            "warm_percentile": _coerce(args.warm_percentile, "int"),
            "phase_b_decision": args.phase_b_decision,
            "phase_c_method": args.phase_c_method,
            "trials_completed": _coerce(args.trials_completed, "int"),
            "elapsed_seconds": _coerce(args.elapsed_seconds, "float"),
            "applied": _coerce(args.applied, "bool"),
        }
    for key, value in updates.items():
        if value is not None:
            record[key] = value
    if args.mark_tuned:                  # tune:true ONLY when the deep-tuner says so —
        record["tune"] = True            # step-1 warm recording must not, or select-candidate sees all as tuned
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
    and `status`; tuning metadata is written by `set-tuning`. P1 requires every
    candidate to have a validated semantic mapping, so missing records are not
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
    after_graph_value = (record.get("status"), record.get("final_best_score"))
    if after_graph_value != before_graph_value:
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


def cmd_percentile(args) -> int:
    data = _load_ledger(Path(args.ledger))
    result = _percentile(data, args.run_id, args.field)
    result.update(_phase_b_decision(result))  # always carries the Phase B verdict
    print(json.dumps(result))
    return 0


def _evaluations_done(data: dict) -> dict:
    """Run-level evaluation budget used so far = total config->score validations =
    Σ trials_completed over the records. `trials_completed` is a per-candidate
    TOTAL (the K warm evals + any new Phase-C trials), so summing it counts each
    real evaluation exactly once — never add warm_start_K on top. A candidate with
    no trials_completed yet (e.g. crashed before set-tuning) falls back to its
    warm_start_K, then 0."""
    records = data.get("records", [])
    per, total = [], 0
    for r in records:
        tc = r.get("trials_completed")
        if tc is None:
            tc = r.get("warm_start_K") or 0
        tc = int(tc)
        total += tc
        per.append({"run_id": r.get("run_id"), "evals": tc,
                    "tuned": bool(r.get("tune")), "status": r.get("status")})
    return {"evaluations_done": total, "n_candidates": len(records), "per_candidate": per}


def _framework_budget(ledger_path: Path) -> Optional[int]:
    """Read the run-local global evaluation budget, if one is configured."""
    path = ledger_path.parent / "framework_cfg.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text()).get("max_evaluations")
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _run_phase(ledger_path: Path, data: dict) -> tuple[str, str]:
    """Return the truthful derived phase and stop condition for a run."""
    state = data.get("run_state") if isinstance(data.get("run_state"), dict) else {}
    budget = _framework_budget(ledger_path)
    if budget is None:
        saved_budget = state.get("evaluation_budget")
        if isinstance(saved_budget, int) and not isinstance(saved_budget, bool):
            budget = saved_budget
    attempted = _evaluations_done(data)["evaluations_done"]
    if state.get("phase") == "blocked":
        return "blocked", str(state.get("active_stop_condition") or "unspecified_blocker")
    if budget is not None and attempted >= budget:
        return "completed", "evaluation_budget_reached"
    # Completion cannot survive new evidence that the configured budget has not
    # been reached. This prevents premature success claims.
    return "running", "none"


def cmd_evaluations(args) -> int:
    data = _load_ledger(Path(args.ledger))
    result = _evaluations_done(data)
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
    evals = _evaluations_done(data)
    attempted = evals["evaluations_done"]
    budget = args.budget if args.budget is not None else _framework_budget(ledger_path)
    phase, stop_condition = _run_phase(ledger_path, data)
    if budget is not None and attempted >= budget and phase != "blocked":
        phase, stop_condition = "completed", "evaluation_budget_reached"
    experience = data.get("experience") if isinstance(data.get("experience"), dict) else {}
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
        "budget": budget,
        "remaining": None if budget is None else max(0, budget - attempted),
        "reached": None if budget is None else attempted >= budget,
        "experience_updated_at_run": experience.get("updated_at_run"),
        "experience_generation": experience.get("generation"),
        "dag_revision": data.get("dag_revision", 0),
        "experience_dag_revision": experience.get("dag_revision"),
    }
    print(json.dumps(result, separators=(",", ":")))
    return 0


def cmd_set_phase(args) -> int:
    """Persist a blocked/running state; completion is budget-derived only."""
    ledger_path = Path(args.ledger)
    data = _load_ledger(ledger_path)
    if args.phase == "completed":
        budget = args.budget if args.budget is not None else _framework_budget(ledger_path)
        attempted = _evaluations_done(data)["evaluations_done"]
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


def _set_experience(ledger_path: Path, experience: dict) -> None:
    """Overwrite the derived experience snapshot and advance its DAG cursor.

    The cursor is helper-owned rather than model-authored.  It is written only
    after the caller has validated the complete replacement snapshot, so a
    failed extraction cannot acknowledge graph changes it did not process.
    """
    data = _load_ledger(ledger_path)
    experience = dict(experience)
    experience["dag_revision"] = _current_dag_revision(data)
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
        raise SystemExit("invalid P1 experience replacement: " + "; ".join(errors))
    _set_experience(ledger_path, experience)
    keys = list(experience.keys()) if isinstance(experience, dict) else None
    print(json.dumps({"ok": True, "experience_keys": keys}))
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
                 "elapsed-seconds", "applied"):
        tune.add_argument(f"--{name}")
    tune.add_argument("--mark-tuned", action="store_true",
                      help="set tune:true — ONLY the deep-tuner (step 2) passes this; step-1 warm "
                           "recording must not, else select-candidate sees every candidate as tuned")
    tune.set_defaults(func=cmd_set_tuning)

    exp = sub.add_parser("set-experience", parents=[common])
    exp.add_argument("--background", required=True,
                     help="hierarchical background.md used to validate the ledger and belief view")
    exp.add_argument("--catalog", help="explicit dimension catalog override")
    exp.add_argument("--from-json", required=True, type=Path,
                     help="JSON file with the experience block to store (overwrites).")
    exp.set_defaults(func=cmd_set_experience)

    run = sub.add_parser("record-run", parents=[common])
    run.add_argument("--run-id", required=True)
    run.add_argument("--final-best-score")
    run.add_argument("--status", default="auto", choices=["auto", "keep", "discard", "crash"])
    run.add_argument("--candidate-name")
    run.add_argument("--description")
    run.set_defaults(func=cmd_record_run)

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
