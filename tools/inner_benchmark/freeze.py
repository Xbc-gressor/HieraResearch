"""Checkpoint freezing tool for the inner-tuner benchmark (PLAN §七).

Turns a real experiment run directory (READ-ONLY source, e.g. under
/home/woden/Hidden) into a frozen benchmark checkpoint in the
``checkpoint.json`` schema_version=2 format that checkpoint.py loads.

Three subcommands (same names as the library entry points):

- ``inspect --run-dir <dir>`` — per-candidate summary table for checkpoint
  selection (§七 stratification aid): source kind, phase-a status, completed
  bout count, per-bout best score + strict-improvement flags, finite unique
  config counts at each boundary, eligible regimes. Read-only.
- ``create --run-dir <dir> --candidate <id> --bouts <N> --out <dir>`` — freeze
  the boundary after N completed bouts (N=0 -> regime "first", N=1 ->
  "continuation", N>=2 -> "deep").
- ``remeasure --checkpoint <dir> [--eval-limit K]`` — re-evaluate every unique
  frozen config on THIS machine via the Task-2 objective path and rewrite the
  checkpoint with local scores (§七 one-machine comparability rule).

Boundary definition (production semantics)
------------------------------------------

A bout is COMPLETED iff

1. every one of its stages has a terminal status
   (tune_tools._TERMINAL_STAGE_STATUSES), and
2. a validated applied close covers the bout's stages
   (tune_tools.has_applied_close + last_finalized_stage_index; production
   finalizes after every bout before the next bout is admitted, so the
   on-disk close covers a prefix of bouts).

These are production's own finished signals. Row counts deliberately do NOT
gate completion: two bo_search.objective early returns consume an n_trials
slot without producing an objective row (preflight rejection ->
infeasible_value, exact duplicate -> duplicates_skipped), and patience
early-stop / EvaluationBudgetExhausted end a bout early — production calls all
of these finished. A nominal size is still computed for ``inspect``
(``tuner.bout_trials`` from the source framework_cfg.json, default
tune_tools.DEFAULT_BOUT_TRIALS = 10, plus the bout's deferred extras) and
reported as ``nominal`` / ``rows_below_nominal``. Production stage meta records
NO deferred-enqueue counter, so the extras are DERIVED: distinct
phase_a.deferred_configs params identities appearing among the bout's objective
rows (deferred configs are enqueued at the first bout and re-injected as scored
priors afterwards, so each matches at most one bout); a deferred config that
was enqueued and then preflight-rejected is undercounted, which only affects
the descriptive field.

Continuation stratification (PLAN §七) reads the bout AFTER the boundary — the
already-observed Current cell on this checkpoint — not the last bout included
in it; that following bout must exist and be complete, with no fallback.

Boundary-time BASE_PARAMS
-------------------------

The copied candidate's BASE_PARAMS is restored to the production incumbent at
the boundary — not left at the source file's value, which may reflect LATER
bouts (finalize_tuning.py rewrites BASE_PARAMS to the global best after every
bout). The value is recomputed as the global best finite row up to that
boundary, including an inherited control when it wins. Historical controls
outside the frozen SEARCH_SPACE remain in history but cannot be materialized
as BASE_PARAMS. The rewrite goes through production apply_base_params.apply
and is verified by re-reading the train.py literals afterwards.

Frozen content
--------------

``<out>/candidate/`` receives train.py + prepare.py ONLY — no tune_report.json
and no ``_*.json`` sidecars, so transient state (phase_c.pending_proposals,
rewarm leftovers) is never carried into a checkpoint (PLAN §七: Current's new
bout must not inherit the real run's rewarm proposals). History rows are
re-derived instead: phase_a evaluated rows (role tags preserved) + per-bout
stage trial rows with origins "phase_a" / "bout_0" / ...; success ->
``{params, score, status: "ok"}``, failure -> ``{params, score: null,
status: "crash"}``; preflight_rejected rows are excluded entirely.

``deferred_configs`` = phase_a.deferred_configs minus configs already
attempted (by production cast + params identity) in the included bouts.

Incumbent (PLAN §5.2): argmin over finite history INCLUDING inherited_control
rows, with the guard that a control config participates only
when representable in the frozen SEARCH_SPACE (tune_tools._bounds_violations);
an excluded control is recorded in ``extra.inherited_control_excluded``.
Create-time scores are SOURCE scores (provisional); ``remeasure`` recomputes.

Stratification: "first" for N=0; for N=1 "cont_improved" /
"cont_not_improved" by whether the last included bout strictly improved its
starting incumbent under production口径 (mirrors
tune_tools._last_bout_improved). For N>=2 the loader couples regime "deep" to
stratum "deep", so the last-bout evidence is recorded in ``extra.last_bout``
instead; ``extra.incumbents`` carries both the production口径 and
benchmark口径 incumbents so the benchmark口径 recompute is trivial.

WARMUP: continuation/deep checkpoints require >= WARMUP=8 finite unique
history rows (arm_api.WARMUP; §七 hard condition) — enforced at create with
source scores and re-checked by remeasure with local scores. First regime has
no WARMUP guard.

Re-measurement
--------------

``remeasure`` re-evaluates ALL unique configs (contract.params_identity) among
history ∪ {incumbent} in fresh subprocesses (objective.evaluate; injectable as
``eval_fn`` for tests). Local crash -> status "crash", score null. The
incumbent is recomputed from re-measured finite scores under the same
benchmark口径 guard. It is idempotent/resumable: identities already holding a
local result (``extra.remeasure.remeasured_identities``) are skipped, and
checkpoint.json is rewritten atomically (tmp + rename) after EVERY
evaluation; ``--eval-limit K`` caps new evaluations per invocation. Guards
fire once re-measurement is complete: no finite re-measured config -> INVALID;
continuation/deep with finite unique local history < WARMUP -> INVALID (§七:
作废换实例). The verdict is printed and persisted under ``extra.remeasure``;
an INVALID checkpoint still loads (the incumbent keeps its last finite value)
so tooling can inspect it.

Operator same-machine attestation: ``remeasure --mark-same-machine LABEL``
records that the source run itself executed on this machine — the frozen
scores are already native, so re-measurement is skipped entirely (the same
field shape is written, with ``mode: same-machine``; it refuses checkpoints
already holding real re-measurement data).

stdlib + numpy only at import time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tomllib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "tuners"))

import apply_base_params  # noqa: E402
import arm_api  # noqa: E402
import checkpoint as checkpoint_mod  # noqa: E402
import objective as objective_mod  # noqa: E402
import space as space_mod  # noqa: E402
import tune_tools  # noqa: E402
from _common import _to_native, params_identity as _raw_identity  # noqa: E402
from _common import stages_by_bout  # noqa: E402

WARMUP = arm_api.WARMUP
CHECKPOINT_FILENAME = checkpoint_mod.CHECKPOINT_FILENAME
SCHEMA_VERSION = checkpoint_mod.SCHEMA_VERSION

PROVIDED_BASELINE_NAME = "provided_baseline"


# ---------------------------------------------------------------------------
# Source-run reading helpers
# ---------------------------------------------------------------------------


def _load_json_object(path: Path, label: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"missing {label}: {path}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return data


def _framework_cfg(run_dir: Path) -> dict:
    path = run_dir / "framework_cfg.json"
    if not path.is_file():
        return {}
    return _load_json_object(path, "framework_cfg.json")


def _bout_trials(framework_cfg: dict) -> int:
    tuner = framework_cfg.get("tuner", {})
    value = (
        tuner.get("bout_trials", tune_tools.DEFAULT_BOUT_TRIALS)
        if isinstance(tuner, dict)
        else tune_tools.DEFAULT_BOUT_TRIALS
    )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"tuner.bout_trials must be a positive integer, got {value!r}")
    return value


def _task_name_from_run_dir(run_dir: Path) -> str | None:
    """Task name from the run-dir path ``runs/<task>/<tag>/``.

    Mirrors ledger.infer_task_name (tools/ledger.py:93) without importing the
    ledger dependency chain.
    """
    parts = Path(run_dir).resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "runs" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _task_block(run_dir: Path, framework_cfg: dict) -> dict:
    """score_fn / preflight_fn from the repo's tasks/<task>/task.toml
    ([evaluation] section); per_runtime_limit from the source framework_cfg.
    ``project`` pins the uv project the benchmark's evaluation subprocesses
    run under (objective.python_cmd_for_project)."""
    task = _task_name_from_run_dir(run_dir)
    if not task:
        raise ValueError(
            f"cannot infer task name from run-dir path {run_dir} "
            "(expected .../runs/<task>/<tag>)"
        )
    toml_path = _HERE.parent.parent / "tasks" / task / "task.toml"
    if not toml_path.is_file():
        raise ValueError(f"missing task.toml for task {task!r}: {toml_path}")
    with toml_path.open("rb") as handle:
        task_toml = tomllib.load(handle)
    evaluation = task_toml.get("evaluation", {})
    score_fn = evaluation.get("score_fn")
    preflight_fn = evaluation.get("preflight_fn")
    if not isinstance(score_fn, str) or not score_fn:
        raise ValueError(f"{toml_path}: [evaluation].score_fn missing")
    if not isinstance(preflight_fn, str) or not preflight_fn:
        raise ValueError(f"{toml_path}: [evaluation].preflight_fn missing")
    # The authoritative source for the evaluation environment is the task's
    # own declaration (same field the production loops read), not the
    # run-dir layout.
    project = task_toml.get("env", {}).get("project")
    if not isinstance(project, str) or not project:
        raise ValueError(f"{toml_path}: [env].project missing")
    limit = framework_cfg.get("per_runtime_limit")
    if limit is not None and not tune_tools._is_finite_score(limit):
        raise ValueError(
            f"framework_cfg per_runtime_limit must be a finite number or null, "
            f"got {limit!r}"
        )
    relative_target = task_toml.get("goal", {}).get(
        "relative_improvement_over_baseline"
    )
    if relative_target is not None:
        if (
            isinstance(relative_target, bool)
            or not tune_tools._is_finite_score(relative_target)
            or not 0 <= float(relative_target) < 1
        ):
            raise ValueError(
                f"{toml_path}: [goal].relative_improvement_over_baseline "
                "must be a finite number in [0, 1)"
            )
    return {
        "score_fn": score_fn,
        "preflight_fn": preflight_fn,
        "per_runtime_limit": float(limit) if limit is not None else None,
        "project": project,
        "relative_improvement_over_baseline": (
            float(relative_target) if relative_target is not None else None
        ),
    }


def _legacy_task_baseline(ledger: dict) -> dict | None:
    """Derive the task_baseline item from a pre-items ledger.

    Runs recorded before the ledger items schema never persisted ``items``;
    their step-0+1 control observation lives in the provided-baseline
    record's ``best_warm_score`` — the screening score before any later
    tuning could lower the mutable ``final_best_score`` (same observation
    boundary and "first finite value wins" rule as
    ledger._capture_task_baseline_item). Returns the same item shape plus a
    ``derived_from`` provenance marker.
    """
    for record in ledger.get("records") or []:
        if not isinstance(record, dict):
            continue
        if record.get("candidate_name") != "provided_baseline":
            continue
        score = record.get("best_warm_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            continue
        return {
            "schema_version": 1,
            "kind": "observed_metric",
            "metric": ledger.get("metric"),
            "value": float(score),
            "direction": "minimize",
            "source": {
                "role": "task_provided_baseline",  # ledger.TASK_BASELINE_ROLE
                "run_id": str(record.get("run_id")),
                "stage": "screening",
            },
            "derived_from": "legacy ledger records[].best_warm_score",
        }
    return None


def _run_items(run_dir: Path, task: dict) -> dict:
    """Freeze run-global observations needed by benchmark policies."""
    ledger = _load_json_object(Path(run_dir) / "ledger.json", "ledger.json")
    items = ledger.get("items")
    if items is not None and not isinstance(items, dict):
        raise ValueError("ledger.json: items must be an object")
    baseline = (items or {}).get("task_baseline")
    if baseline is None and items is None:
        baseline = _legacy_task_baseline(ledger)
    if task.get("relative_improvement_over_baseline") is not None and not isinstance(
        baseline, dict
    ):
        raise ValueError(
            "ledger.json: configured baseline-relative goal requires "
            "items.task_baseline"
        )
    return {"task_baseline": dict(baseline)} if isinstance(baseline, dict) else {}


# ---------------------------------------------------------------------------
# Bout analysis (shared by inspect and create)
# ---------------------------------------------------------------------------


def _objective_rows(bout: list[dict]) -> list[dict]:
    """Objective-evaluation rows of one bout: stage trial rows excluding
    preflight_rejected (those were never objective evaluations)."""
    rows = []
    for stage in bout:
        trials = stage.get("trials") if isinstance(stage, dict) else None
        if not isinstance(trials, list):
            continue
        for row in trials:
            if not isinstance(row, dict) or not isinstance(row.get("params"), dict):
                continue
            if row.get("status") == "preflight_rejected":
                continue
            rows.append(row)
    return rows


def _close_coverage(report: dict, stages: list) -> tuple[int | None, str | None]:
    """Flat stage index through which a validated applied close reaches.

    Reuses production tune_tools.has_applied_close, which validates the close
    fields against the stage prefix up to last_finalized_stage_index (legacy
    closes without the field cover every stage). Returns (index, None), or
    (None, reason) when no validated applied close exists.
    """
    if not isinstance(report, dict) or report.get("applied_to_base_params") is not True:
        return None, "applied_to_base_params is not true"
    try:
        if not tune_tools.has_applied_close(report):
            return None, "no validated applied close"
    except ValueError as exc:
        return None, f"applied close failed validation: {exc}"
    last = tune_tools.last_finalized_stage_index(report)
    return (last if last is not None else len(stages) - 1), None


def _improvement_flags(phase_a: dict, bouts: list[list[dict]]) -> list[dict]:
    """Per-bout best score and strict-improvement flag, production口径.

    Mirrors tune_tools._last_bout_improved generalized to every bout: the
    starting incumbent is phase_a.best_warm_score (production's Phase-A best,
    including inherited_control when it wins), tightened by each earlier bout's
    finite trial rows; a bout improves iff its best finite trial is strictly lower.
    """
    warm_best = phase_a.get("best_warm_score")
    prior = float(warm_best) if tune_tools._is_finite_score(warm_best) else None
    flags = []
    for bout in bouts:
        current = None
        for stage in bout:
            trials = stage.get("trials") if isinstance(stage, dict) else None
            if not isinstance(trials, list):
                continue
            for row in trials:
                if isinstance(row, dict) and tune_tools._is_finite_score(row.get("score")):
                    score = float(row["score"])
                    if current is None or score < current:
                        current = score
        improved = (prior is None or current < prior) if current is not None else False
        flags.append(
            {
                "best_score": current,
                "improved": improved,
                "starting_incumbent_score": prior,
            }
        )
        if current is not None and (prior is None or current < prior):
            prior = current
    return flags


def _analyze_bouts(report: dict, *, bout_trials: int, identity) -> tuple[list[dict], list]:
    """Per-bout completion analysis. ``identity`` is contract.params_identity
    (production cast + stable JSON identity).

    Completion = every stage in a terminal status AND a validated applied close
    covering the bout — production's own finished signals. Row counts are NOT a
    completion criterion: bo_search.objective has two early-return paths that
    consume an n_trials slot without producing an objective row (preflight
    rejection -> infeasible_value, exact duplicate -> duplicates_skipped), and
    PatienceMonitor / EvaluationBudgetExhausted stop a bout early, all of which
    production treats as finished. Measured over the real corpus, the row-count
    rule rejected 28 of 112 genuinely-finished bouts — including both runs'
    candidate 000, the provided baseline PLAN §七 requires covering. The
    shortfall is kept as a descriptive field for inspect.
    """
    phase_a = report.get("phase_a") if isinstance(report.get("phase_a"), dict) else {}
    phase_c = report.get("phase_c") if isinstance(report.get("phase_c"), dict) else {}
    stages = phase_c.get("stages") if isinstance(phase_c.get("stages"), list) else []
    stages = [stage for stage in stages if isinstance(stage, dict)]
    bouts = stages_by_bout(stages) if stages else []
    deferred_raw = phase_a.get("deferred_configs")
    deferred_params = [
        entry["params"]
        for entry in (deferred_raw if isinstance(deferred_raw, list) else [])
        if isinstance(entry, dict) and isinstance(entry.get("params"), dict)
    ]
    coverage, coverage_note = _close_coverage(report, stages)
    flags = _improvement_flags(phase_a, bouts)
    analysis = []
    flat = 0
    for bout_index, bout in enumerate(bouts):
        objective = _objective_rows(bout)
        # Deferred extras: production stage meta records no deferred-enqueue
        # counter (bo_search grows n_trials by n_deferred_enqueued without
        # persisting it), so derive it by params identity. This UNDERCOUNTS a
        # deferred config that was enqueued (raising n_trials) and then
        # preflight-rejected, since such a config leaves no objective row —
        # tolerable now that nominal is descriptive rather than a gate.
        attempted = {identity(row["params"]) for row in objective}
        extras = len({identity(params) for params in deferred_params} & attempted)
        nominal = bout_trials + extras
        non_terminal = [
            stage.get("status")
            for stage in bout
            if stage.get("status") not in tune_tools._TERMINAL_STAGE_STATUSES
        ]
        last_flat = flat + len(bout) - 1
        close_covered = coverage is not None and coverage >= last_flat
        reasons = []
        if non_terminal:
            reasons.append(f"non-terminal stage statuses {non_terminal!r}")
        if not close_covered:
            reason = "no validated applied close covers this bout"
            if coverage_note:
                reason += f" ({coverage_note})"
            reasons.append(reason)
        analysis.append(
            {
                "bout_index": bout_index,
                "best_score": flags[bout_index]["best_score"],
                "improved": flags[bout_index]["improved"],
                "starting_incumbent_score": flags[bout_index][
                    "starting_incumbent_score"
                ],
                "objective_rows": len(objective),
                "nominal": nominal,
                # Descriptive only (see _analyze_bouts docstring): slots the
                # bout consumed without producing an objective row.
                "rows_below_nominal": max(0, nominal - len(objective)),
                "terminal": not non_terminal,
                "close_covered": close_covered,
                "complete": not reasons,
                "incomplete_reasons": reasons,
            }
        )
        flat += len(bout)
    return analysis, bouts


# ---------------------------------------------------------------------------
# History / incumbent construction
# ---------------------------------------------------------------------------


def _convert_row(row: dict, *, origin: str) -> dict | None:
    """One source row -> checkpoint history row, or None to drop it.

    preflight_rejected rows are dropped (never objective evaluations); finite
    scores become ok rows; anything else becomes a crash row (score null).
    Role tags (e.g. inherited_control) are preserved.
    """
    if not isinstance(row, dict) or not isinstance(row.get("params"), dict):
        return None
    if row.get("status") == "preflight_rejected":
        return None
    out = {"params": dict(row["params"])}
    if tune_tools._is_finite_score(row.get("score")):
        out["score"] = float(row["score"])
        out["status"] = "ok"
    else:
        out["score"] = None
        out["status"] = "crash"
    out["origin"] = origin
    role = row.get("role")
    if isinstance(role, str):
        out["role"] = role
    return out


def _history_rows(phase_a: dict, bouts: list[list[dict]], n_bouts: int) -> list[dict]:
    """All evaluated configs up to the boundary: phase_a evaluated rows (role
    tags preserved) + the included bouts' stage trial rows."""
    rows = []
    warm = phase_a.get("warm_start_configs")
    for row in warm if isinstance(warm, list) else []:
        converted = _convert_row(row, origin="phase_a")
        if converted is not None:
            rows.append(converted)
    for bout_index in range(n_bouts):
        for row in _objective_rows(bouts[bout_index]):
            converted = _convert_row(row, origin=f"bout_{bout_index}")
            if converted is not None:
                rows.append(converted)
    return rows


def _select_benchmark_incumbent(rows: list[dict], search_space: dict):
    """Benchmark口径 incumbent (PLAN §5.2): argmin over finite rows INCLUDING
    inherited_control, with the guard that a control config participates only
    when representable in the frozen SEARCH_SPACE (production
    tune_tools._bounds_violations). Returns (best_row, excluded_controls)."""
    best = None
    excluded = []
    for row in rows:
        if row.get("status") != "ok" or not tune_tools._is_finite_score(row.get("score")):
            continue
        if row.get("role") == "inherited_control":
            violations = tune_tools._bounds_violations(row["params"], search_space)
            if violations:
                excluded.append({"params": row["params"], "violations": violations})
                continue
        if best is None or float(row["score"]) < float(best["score"]):
            best = row
    return best, excluded


def _finite_unique_count(rows: list[dict], identity) -> int:
    """Finite unique config count (checkpoint.finite_unique_history semantics:
    the incumbent is always one of the history rows, so rows suffice)."""
    return len(
        {identity(row["params"]) for row in rows if row.get("status") == "ok"}
    )


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def _ledger_records(run_dir: Path) -> dict:
    """run_id -> ledger record, read-only ({} when the ledger is absent or
    unreadable — kind is descriptive, never a gate)."""
    records: dict = {}
    ledger_path = Path(run_dir) / "ledger.json"
    if not ledger_path.is_file():
        return records
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return records
    for record in ledger.get("records", []) if isinstance(ledger, dict) else []:
        if isinstance(record, dict) and record.get("run_id") is not None:
            records[str(record["run_id"])] = record
    return records


def _candidate_kind(candidate_dir: Path, records: dict) -> str:
    """fresh / improve / crossover / provided-baseline — from the ledger record
    (op field; the task-provided baseline is named provided_baseline), falling
    back to _candidate_brief.json."""
    record = records.get(candidate_dir.name)
    if record is not None:
        if record.get("candidate_name") == PROVIDED_BASELINE_NAME:
            return "provided-baseline"
        op = record.get("op")
        if isinstance(op, str) and op:
            return op
    brief_path = candidate_dir / "_candidate_brief.json"
    if brief_path.is_file():
        try:
            brief = json.loads(brief_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            brief = {}
        if isinstance(brief, dict) and isinstance(brief.get("op"), str) and brief["op"]:
            return brief["op"]
    return "unknown"


def _inspect_candidate(candidate_dir: Path, records: dict, bout_trials: int) -> dict:
    row = {
        "candidate_id": candidate_dir.name,
        "kind": _candidate_kind(candidate_dir, records),
        "phase_a_ok": False,
        "completed_bouts": 0,
        "bouts": [],
        "finite_unique_by_boundary": [],
        "eligible_regimes": [],
    }
    report_path = candidate_dir / "tune_report.json"
    if not report_path.is_file():
        row["note"] = "no tune_report.json"
        return row
    try:
        report = _load_json_object(report_path, "tune_report.json")
        phase_a = (
            report.get("phase_a") if isinstance(report.get("phase_a"), dict) else {}
        )
        row["phase_a_ok"] = phase_a.get("status") == "ok"
        train_path = candidate_dir / "train.py"
        try:
            identity = space_mod.read_contract(train_path).params_identity
        except (OSError, SyntaxError, ValueError):
            # Selection aid only: fall back to the raw (uncast) JSON identity.
            identity = _raw_identity
        analysis, bouts = _analyze_bouts(
            report, bout_trials=bout_trials, identity=identity
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        row["note"] = f"{type(exc).__name__}: {exc}"
        return row
    row["bouts"] = analysis
    if not row["phase_a_ok"]:
        row["note"] = f"phase_a status is {phase_a.get('status')!r}"
        return row
    completed = 0
    for info in analysis:
        if not info["complete"]:
            break
        completed += 1
    row["completed_bouts"] = completed
    row["finite_unique_by_boundary"] = [
        _finite_unique_count(_history_rows(phase_a, bouts, n), identity)
        for n in range(completed + 1)
    ]
    eligible = ["first"]
    if completed >= 1:
        eligible.append("continuation")
    if completed >= 2:
        eligible.append("deep")
    row["eligible_regimes"] = eligible
    return row


def inspect_run(run_dir) -> list[dict]:
    """Per-candidate summary rows for checkpoint selection (§七 stratification
    aid). Read-only."""
    run_dir = Path(run_dir)
    candidates_dir = run_dir / "candidates"
    if not candidates_dir.is_dir():
        raise ValueError(f"no candidates directory: {candidates_dir}")
    bout_trials = _bout_trials(_framework_cfg(run_dir))
    records = _ledger_records(run_dir)
    return [
        _inspect_candidate(candidate_dir, records, bout_trials)
        for candidate_dir in sorted(candidates_dir.iterdir())
        if candidate_dir.is_dir()
    ]


def _fmt_score(value) -> str:
    return f"{value:.6f}" if isinstance(value, float) else "-"


def _format_inspect_table(rows: list[dict]) -> str:
    header = (
        f"{'candidate':<10} {'kind':<17} {'phase_a':<8} {'bouts':<6} "
        f"{'per-bout best (improved)':<34} {'unique@boundary':<18} eligible"
    )
    lines = [header]
    for row in rows:
        per_bout = " ".join(
            f"b{info['bout_index']}:{_fmt_score(info['best_score'])}"
            f"({'+' if info['improved'] else '-'}{'' if info['complete'] else '!'})"
            for info in row["bouts"]
        )
        counts = ",".join(
            f"N{index}:{count}"
            for index, count in enumerate(row["finite_unique_by_boundary"])
        )
        note = row.get("note")
        lines.append(
            f"{row['candidate_id']:<10} {row['kind']:<17} "
            f"{('ok' if row['phase_a_ok'] else 'no'):<8} "
            f"{row['completed_bouts']:<6} {per_bout:<34} {counts:<18} "
            f"{','.join(row['eligible_regimes']) or '-'}"
            + (f"   # {note}" if note else "")
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def _regime(n_bouts: int) -> str:
    return "first" if n_bouts == 0 else "continuation" if n_bouts == 1 else "deep"


def create_checkpoint(run_dir, candidate_id, bouts: int, out_dir) -> dict:
    """Freeze one candidate at the boundary after ``bouts`` completed bouts.

    Writes ``<out_dir>/checkpoint.json`` (schema_version=2) plus
    ``<out_dir>/candidate/{train.py,prepare.py}`` with BASE_PARAMS restored to
    the boundary-time production incumbent. Raises ValueError with a specific
    message when the requested boundary does not exist or the source is
    malformed. Refuses to overwrite a non-empty out dir.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    candidate_id = str(candidate_id)
    if isinstance(bouts, bool) or not isinstance(bouts, int) or bouts < 0:
        raise ValueError(f"bouts must be a non-negative integer, got {bouts!r}")

    candidate_dir = run_dir / "candidates" / candidate_id
    train_src = candidate_dir / "train.py"
    prepare_src = candidate_dir / "prepare.py"
    for path, label in ((train_src, "train.py"), (prepare_src, "prepare.py")):
        if not path.is_file():
            raise ValueError(f"source candidate {candidate_id}: missing {label} ({path})")
    report = _load_json_object(candidate_dir / "tune_report.json", "tune_report.json")
    phase_a = report.get("phase_a") if isinstance(report.get("phase_a"), dict) else {}
    if phase_a.get("status") != "ok":
        raise ValueError(
            f"candidate {candidate_id}: phase_a status is {phase_a.get('status')!r}, "
            "expected 'ok' — no evaluated phase-a base to freeze"
        )

    # AST-only contract read (never imports the candidate); lints the contract.
    contract = space_mod.read_contract(train_src)
    identity = contract.params_identity
    framework_cfg = _framework_cfg(run_dir)
    bout_trials = _bout_trials(framework_cfg)
    analysis, bout_list = _analyze_bouts(report, bout_trials=bout_trials, identity=identity)

    if bouts > len(bout_list):
        raise ValueError(
            f"candidate {candidate_id}: boundary after {bouts} completed bout(s) "
            f"does not exist — tune_report has {len(bout_list)} bout(s)"
        )
    for info in analysis[:bouts]:
        if not info["complete"]:
            raise ValueError(
                f"candidate {candidate_id}: boundary after {bouts} completed bout(s) "
                f"does not exist — bout {info['bout_index']} is not completed: "
                + "; ".join(info["incomplete_reasons"])
            )

    # History, deferred configs, and incumbent — all computed BEFORE
    # any output write so a failure never leaves a partial checkpoint.
    history = _history_rows(phase_a, bout_list, bouts)
    attempted = {
        identity(row["params"]) for row in history if row["origin"] != "phase_a"
    }
    deferred_raw = phase_a.get("deferred_configs")
    deferred = [
        {"params": dict(entry["params"])}
        for entry in (deferred_raw if isinstance(deferred_raw, list) else [])
        if isinstance(entry, dict)
        and isinstance(entry.get("params"), dict)
        and identity(entry["params"]) not in attempted
    ]
    incumbent_row, excluded_controls = _select_benchmark_incumbent(
        history, contract.search_space
    )
    if incumbent_row is None:
        raise ValueError(
            f"candidate {candidate_id}: no finite history row can serve as incumbent"
        )
    prod_params = dict(incumbent_row["params"])
    prod_score = float(incumbent_row["score"])
    regime = _regime(bouts)
    if regime != "first":
        finite_unique = _finite_unique_count(history, identity)
        if finite_unique < WARMUP:
            raise ValueError(
                f"candidate {candidate_id}: {regime} checkpoint needs >= "
                f"WARMUP={WARMUP} finite unique history rows at the boundary, "
                f"got {finite_unique} (PLAN §七 hard condition)"
            )

    if bouts == 0:
        stratum = "first"
    elif bouts == 1:
        # PLAN §七 defines "no improvement" by the COMPLETE bout that FOLLOWS
        # the boundary — that bout is Current's already-observed cell on this
        # checkpoint. analysis[bouts - 1] is the bout included IN the
        # checkpoint, which answers a different question; using it inverted the
        # label on every real cont_not_improved instance in the corpus (5/5).
        # No fallback to the earlier bout: a checkpoint whose following bout is
        # missing or incomplete carries no evidence for this stratum.
        if len(analysis) <= bouts or not analysis[bouts]["complete"]:
            raise ValueError(
                f"candidate {candidate_id}: continuation stratification needs a "
                f"complete bout {bouts} AFTER the boundary (PLAN §七), but "
                + (
                    f"only {len(analysis)} bout(s) exist"
                    if len(analysis) <= bouts
                    else "it is not complete: "
                    + "; ".join(analysis[bouts]["incomplete_reasons"])
                )
            )
        # A following bout with no finite score at all is an infrastructure
        # failure, not evidence that Current could not improve. In the corpus
        # one such bout (every trial preflight_rejected on a kernel-trust
        # error) reached this branch, and it would have been 1 of only 3
        # members of the scarcest stratum.
        if analysis[bouts]["best_score"] is None:
            raise ValueError(
                f"candidate {candidate_id}: the bout {bouts} after the boundary "
                f"produced no finite objective score "
                f"({analysis[bouts]['objective_rows']} row(s), all crashed or "
                "preflight-rejected) — it carries no improvement evidence"
            )
        stratum = (
            "cont_improved" if analysis[bouts]["improved"] else "cont_not_improved"
        )
    else:
        # checkpoint.py couples regime "deep" to stratum "deep"; the last-bout
        # improvement evidence lives in extra.last_bout.
        stratum = "deep"

    extra = {
        "incumbents": {
            "production": {"params": prod_params, "score": prod_score},
            "benchmark": {
                "params": dict(incumbent_row["params"]),
                "score": float(incumbent_row["score"]),
            },
        },
        "remeasure": {
            "remeasured_identities": [],
            "incumbent_local": False,
            "complete": False,
            "valid": None,
        },
    }
    if bouts >= 1:
        last = analysis[bouts - 1]
        prior_origins = {"phase_a"} | {f"bout_{k}" for k in range(bouts - 1)}
        prior_rows = [row for row in history if row["origin"] in prior_origins]
        prior_best, _ = _select_benchmark_incumbent(prior_rows, contract.search_space)
        extra["last_bout"] = {
            "bout_index": bouts - 1,
            "best_score": last["best_score"],
            "improved_production": last["improved"],
            "starting_incumbent_score_production": last["starting_incumbent_score"],
            "starting_incumbent_score_benchmark": (
                float(prior_best["score"]) if prior_best is not None else None
            ),
        }
        # Both sides of the boundary, so a disagreement between "the last bout
        # inside the checkpoint improved" and "the bout after it improved" (the
        # one stratification actually uses) is visible in the artifact rather
        # than hidden behind one flag.
        following = analysis[bouts] if len(analysis) > bouts else None
        extra["bout_after_boundary"] = (
            {
                "bout_index": bouts,
                "best_score": following["best_score"],
                "improved_production": following["improved"],
                "starting_incumbent_score_production": following[
                    "starting_incumbent_score"
                ],
                "complete": following["complete"],
            }
            if following is not None
            else None
        )
    if excluded_controls:
        extra["inherited_control_excluded"] = {
            "reason": "inherited_control config is not representable in the "
            "frozen SEARCH_SPACE (tune_tools._bounds_violations)",
            "rows": excluded_controls,
        }

    run_metadata_path = run_dir / "run_metadata.json"
    run_metadata = None
    if run_metadata_path.is_file():
        try:
            run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            run_metadata = {"unreadable": f"{type(exc).__name__}: {exc}"}
    source = {
        "run_dir": str(run_dir.resolve()),
        "task": _task_name_from_run_dir(run_dir),
        "tag": run_dir.name,
        "candidate_id": candidate_id,
        # Persisted here because it is only derivable from the RUN (ledger op /
        # _candidate_brief.json), which the frozen checkpoint no longer points
        # into. llm.first_message_blocks renders it in the candidate block.
        "kind": _candidate_kind(candidate_dir, _ledger_records(run_dir)),
        "bouts_included": bouts,
        # Factual provenance / sensitivity covariate (PLAN-inner-arms-mixup-alt
        # §6): the production inner-tuner policy id whose FIRST/CONTINUE
        # kernels produced the rows before this boundary. Null when the
        # source run did not set one.
        "tuner_inner_policy": (
            tuner_cfg.get("inner_policy")
            if isinstance((tuner_cfg := framework_cfg.get("tuner")), dict)
            else None
        ),
        "train_sha256": "sha256:" + hashlib.sha256(train_src.read_bytes()).hexdigest(),
    }
    if run_metadata is not None:
        source["run_metadata"] = run_metadata

    task_block = _task_block(run_dir, framework_cfg)
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_id": f"{run_dir.name}-{candidate_id}-b{bouts}",
        "regime": regime,
        "stratum": stratum,
        "source": source,
        "candidate_relpath": "candidate",
        "task": task_block,
        "items": _run_items(run_dir, task_block),
        "incumbent": {
            "params": dict(incumbent_row["params"]),
            "score": float(incumbent_row["score"]),
        },
        "incumbent_is_inherited_control": incumbent_row.get("role")
        == "inherited_control",
        "history": history,
        "deferred_configs": deferred,
        "extra": extra,
    }

    if out_dir.exists():
        if not out_dir.is_dir() or any(out_dir.iterdir()):
            raise ValueError(f"refusing to overwrite non-empty out dir: {out_dir}")
    else:
        out_dir.mkdir(parents=True)
    candidate_out = out_dir / "candidate"
    candidate_out.mkdir()
    # train.py + prepare.py ONLY: no tune_report.json, no _*.json sidecars, so
    # transient state (phase_c.pending_proposals, rewarm leftovers) is never
    # carried into the checkpoint.
    shutil.copy2(train_src, candidate_out / "train.py")
    shutil.copy2(prepare_src, candidate_out / "prepare.py")
    try:
        apply_base_params.apply(candidate_out / "train.py", prod_params)
    except SystemExit as exc:
        raise ValueError(
            f"cannot restore boundary-time BASE_PARAMS in the copied train.py: "
            f"{exc.code}"
        ) from None
    rewritten = tune_tools._read_literal_mapping(candidate_out / "train.py", "BASE_PARAMS")
    if rewritten != prod_params:
        raise ValueError(
            f"post-rewrite verification failed: copied BASE_PARAMS {rewritten!r} "
            f"!= boundary incumbent {prod_params!r}"
        )
    # Prove the frozen candidate still lints as a valid contract.
    space_mod.read_contract(candidate_out / "train.py")

    _write_checkpoint_json(out_dir, checkpoint)
    # Prove the written checkpoint loads cleanly through the Task-3 loader.
    checkpoint_mod.load_checkpoint(out_dir)

    return {
        "checkpoint_id": checkpoint["checkpoint_id"],
        "out": str(out_dir),
        "regime": regime,
        "stratum": stratum,
        "bouts_included": bouts,
        "history_rows": len(history),
        "deferred_configs": len(deferred),
        "incumbent": checkpoint["incumbent"],
        "incumbent_is_inherited_control": checkpoint["incumbent_is_inherited_control"],
        "base_params_restored": prod_params,
        "task": checkpoint["task"],
    }


def _write_checkpoint_json(checkpoint_dir: Path, data: dict) -> None:
    """Atomic checkpoint.json rewrite (tmp + rename)."""
    path = Path(checkpoint_dir) / CHECKPOINT_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_to_native) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


# ---------------------------------------------------------------------------
# remeasure
# ---------------------------------------------------------------------------


def _mark_same_machine(directory: Path, data: dict, ckpt, contract, *, label: str) -> dict:
    """Operator attestation: the source run executed on THIS machine, so the
    frozen scores are already native and re-measurement would only re-noise
    them (PLAN §七 同机可比性 is satisfied by provenance, not by
    re-evaluation). Writes the same extra.remeasure field shape the real
    remeasure writes, plus provenance. Refuses to overwrite a checkpoint
    that already holds real re-measurement data."""
    extra = data.setdefault("extra", {})
    state = extra.setdefault("remeasure", {})
    if state.get("remeasured_identities"):
        raise ValueError(
            f"{directory}: checkpoint already holds re-measurement data; "
            "refusing to overwrite it with a same-machine mark"
        )
    if not tune_tools._is_finite_score(data["incumbent"].get("score")):
        raise ValueError(f"{directory}: incumbent score is not finite")
    finite_unique = len(ckpt.finite_unique_history(contract))
    if ckpt.regime != "first" and finite_unique < WARMUP:
        raise ValueError(
            f"{directory}: finite unique history {finite_unique} < WARMUP="
            f"{WARMUP} — INVALID regardless of machine (PLAN §七)"
        )
    state.update(
        {
            "mode": "same-machine",
            "machine": label,
            "remeasured_identities": [],
            "incumbent_local": True,
            "complete": True,
            "valid": True,
            "finite_unique_local": finite_unique,
            "evaluations": 0,
            "note": (
                "operator attestation: the source run executed on this "
                "machine; source scores are native and re-measurement was "
                "skipped, so no re-evaluation consumed objective budget"
            ),
        }
    )
    state.pop("invalid_reason", None)
    _write_checkpoint_json(directory, data)
    return {
        "checkpoint": str(directory),
        "mode": "same-machine",
        "machine": label,
        "evaluated": 0,
        "complete": True,
        "incumbent_after": data["incumbent"]["score"],
        "finite_unique_local": finite_unique,
        "verdict": "ok",
        "invalid_reason": None,
    }


def remeasure_checkpoint(checkpoint_dir, *, eval_fn=None, eval_limit=None, mark_same_machine=None) -> dict:
    """Re-measure every unique frozen config on THIS machine (PLAN §七).

    ``eval_fn(params) -> objective.EvalOutcome-compatible`` is the injection
    seam (tests); the CLI default is objective.evaluate wired with the
    checkpoint's score_fn / per_runtime_limit. ``eval_limit`` caps how many
    NEW evaluations this invocation performs. Resumable: configs whose
    identity is already in extra.remeasure.remeasured_identities are skipped,
    and checkpoint.json is rewritten atomically after every evaluation.
    """
    directory = Path(checkpoint_dir)
    ckpt = checkpoint_mod.load_checkpoint(directory)
    contract = space_mod.read_contract(ckpt.candidate_path)
    identity = contract.params_identity
    data = _load_json_object(directory / CHECKPOINT_FILENAME, "checkpoint.json")
    if mark_same_machine is not None:
        return _mark_same_machine(
            directory, data, ckpt, contract, label=mark_same_machine
        )
    extra = data.setdefault("extra", {})
    state = extra.setdefault("remeasure", {})
    done = {str(item) for item in state.get("remeasured_identities", [])}
    incumbent = {"local": bool(state.get("incumbent_local", False))}

    # Unique configs among history ∪ {incumbent}, first occurrence order.
    unique: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for entry in [*data["history"], {"params": data["incumbent"]["params"]}]:
        key = identity(entry["params"])
        if key not in seen:
            seen.add(key)
            unique.append((key, entry["params"]))

    if eval_fn is None:
        candidate_path = ckpt.candidate_path
        score_fn = ckpt.task.score_fn
        per_runtime_limit = ckpt.task.per_runtime_limit
        python_cmd = objective_mod.python_cmd_for_project(ckpt.task.project)

        def eval_fn(params):  # noqa: F811 — the CLI default evaluation path
            return objective_mod.evaluate(
                candidate_path,
                params,
                score_fn=score_fn,
                per_runtime_limit=per_runtime_limit,
                python_cmd=python_cmd,
            )

    if eval_limit is not None:
        if isinstance(eval_limit, bool) or not isinstance(eval_limit, int) or eval_limit < 0:
            raise ValueError(f"eval_limit must be a non-negative integer or None, got {eval_limit!r}")

    def persist() -> None:
        state["remeasured_identities"] = sorted(done)
        state["incumbent_local"] = incumbent["local"]
        _write_checkpoint_json(directory, data)

    def local_ok_rows() -> list[dict]:
        return [
            row
            for row in data["history"]
            if row.get("status") == "ok" and identity(row["params"]) in done
        ]

    evaluated = crashed = 0
    skipped = sum(1 for key, _ in unique if key in done)
    incumbent_before = data["incumbent"]["score"]
    for key, params in unique:
        if key in done:
            continue
        if eval_limit is not None and evaluated >= eval_limit:
            break
        outcome = eval_fn(dict(params))
        ok = getattr(outcome, "status", None) == "ok" and tune_tools._is_finite_score(
            getattr(outcome, "score", None)
        )
        score = float(outcome.score) if ok else None
        for row in data["history"]:
            if identity(row["params"]) == key:
                row["status"] = "ok" if ok else "crash"
                row["score"] = score
        done.add(key)
        evaluated += 1
        crashed += 0 if ok else 1
        # Recompute the incumbent from re-measured finite rows only (never mix
        # local and source scores), benchmark口径 with the control guard.
        best, excluded = _select_benchmark_incumbent(local_ok_rows(), contract.search_space)
        if best is not None:
            data["incumbent"] = {
                "params": dict(best["params"]),
                "score": float(best["score"]),
            }
            data["incumbent_is_inherited_control"] = best.get("role") == "inherited_control"
            incumbent["local"] = True
        _record_excluded_controls(extra, excluded)
        persist()

    remaining = sum(1 for key, _ in unique if key not in done)
    complete = remaining == 0
    finite_entries = [row["params"] for row in local_ok_rows()]
    if incumbent["local"]:
        finite_entries.append(data["incumbent"]["params"])
    finite_unique = len({identity(params) for params in finite_entries})

    valid = None
    invalid_reason = None
    if complete:
        if not incumbent["local"]:
            valid = False
            invalid_reason = (
                "no finite configuration survived re-measurement; the incumbent "
                "on file keeps its pre-remeasure score"
            )
        elif ckpt.regime != "first" and finite_unique < WARMUP:
            valid = False
            invalid_reason = (
                f"finite unique re-measured history {finite_unique} < "
                f"WARMUP={WARMUP} — checkpoint INVALID, pick another instance "
                "(PLAN §七)"
            )
        else:
            valid = True
    state.update(
        {
            "complete": complete,
            "valid": valid,
            "finite_unique_local": finite_unique,
            "evaluations": len(done),
        }
    )
    if invalid_reason is not None:
        state["invalid_reason"] = invalid_reason
    else:
        state.pop("invalid_reason", None)
    persist()

    return {
        "checkpoint": str(directory),
        "evaluated": evaluated,
        "skipped": skipped,
        "crashed": crashed,
        "remaining": remaining,
        "complete": complete,
        "incumbent_before": incumbent_before,
        "incumbent_after": data["incumbent"]["score"],
        "incumbent_is_inherited_control": data["incumbent_is_inherited_control"],
        "finite_unique_local": finite_unique,
        "verdict": "invalid" if valid is False else "ok" if valid else "incomplete",
        "invalid_reason": invalid_reason,
    }


def _record_excluded_controls(extra: dict, excluded: list[dict]) -> None:
    if excluded:
        extra["inherited_control_excluded"] = {
            "reason": "inherited_control config is not representable in the "
            "frozen SEARCH_SPACE (tune_tools._bounds_violations)",
            "rows": excluded,
        }
    else:
        extra.pop("inherited_control_excluded", None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="per-candidate boundary summary")
    p_inspect.add_argument("--run-dir", required=True, type=Path)

    p_create = sub.add_parser("create", help="freeze one candidate boundary")
    p_create.add_argument("--run-dir", required=True, type=Path)
    p_create.add_argument("--candidate", required=True)
    p_create.add_argument("--bouts", required=True, type=int)
    p_create.add_argument("--out", required=True, type=Path)

    p_remeasure = sub.add_parser("remeasure", help="re-measure scores locally")
    p_remeasure.add_argument("--checkpoint", required=True, type=Path)
    p_remeasure.add_argument("--eval-limit", type=int, default=None)
    p_remeasure.add_argument(
        "--mark-same-machine",
        metavar="LABEL",
        default=None,
        help="skip re-evaluation: attest the source run executed on THIS "
        "machine (LABEL names it), so frozen scores are already native",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            print(_format_inspect_table(inspect_run(args.run_dir)))
        elif args.command == "create":
            summary = create_checkpoint(args.run_dir, args.candidate, args.bouts, args.out)
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=_to_native))
        else:
            if args.mark_same_machine is not None and args.eval_limit is not None:
                raise SystemExit("--mark-same-machine and --eval-limit are mutually exclusive")
            summary = remeasure_checkpoint(
                args.checkpoint,
                eval_limit=args.eval_limit,
                mark_same_machine=args.mark_same_machine,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=_to_native))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
