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
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

try:  # POSIX is the supported experiment runtime; keep reads usable elsewhere.
    import fcntl
except ImportError:  # pragma: no cover - Windows compatibility fallback
    fcntl = None

from run_cfg import read_framework_cfg


SCHEMA_VERSION = 1
ATTEMPT_LOG = "evaluation_attempts.jsonl"


class EvaluationBudgetExhausted(RuntimeError):
    """Raised before ``score_fn`` when no objective slot remains."""

    def __init__(
        self,
        *,
        used: int,
        budget: int,
        run_dir: Path,
        scope: str = "global",
    ):
        self.used = used
        self.budget = budget
        self.run_dir = Path(run_dir)
        self.scope = scope
        super().__init__(
            f"{scope} evaluation budget exhausted before score_fn "
            f"(used={used}, budget={budget}, run_dir={run_dir})"
        )


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


def _deep_tune_limits(run_dir: Path, budget: int | None) -> dict:
    path = Path(run_dir) / "framework_cfg.json"
    config = read_framework_cfg(path) if path.is_file() else {}
    tuner = config.get("tuner", {})
    tuner = tuner if isinstance(tuner, dict) else {}
    fraction = float(tuner.get("deep_tune_budget_fraction", 0.4))
    return {
        "fraction": fraction,
        "total_cap": (
            0
            if fraction == 0
            else (
                None
                if budget is None
                else int(math.floor(budget * fraction))
            )
        ),
        "per_candidate_cap": int(
            tuner.get("deep_tune_per_candidate_cap", 20)
        ),
        # Phase-C wall clock removed: trials (patience/n_trials/caps above)
        # denominate the budget. Kept as null so receipt shape is stable.
        "time_limit_seconds": None,
    }


def _legacy_phase_c_per_candidate(run_dir: Path) -> dict[str, int]:
    per_candidate: dict[str, int] = {}
    for report_path in sorted(
        (Path(run_dir) / "candidates").glob("*/tune_report.json")
    ):
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(report, dict):
            per_candidate[report_path.parent.name] = _phase_c_objective_attempts(
                report
            )
    return per_candidate


def _deep_tune_usage(
    rows: list[dict],
    run_dir: Path,
) -> tuple[int, dict[str, int]]:
    logged: Counter[str] = Counter()
    for row in rows:
        if row.get("kind") != "score_attempt" or row.get("phase") != "phase_c":
            continue
        run_id = row.get("run_id")
        if isinstance(run_id, str):
            logged[run_id] += 1
    # Old reservation logs migrated only aggregate/per-candidate totals. Reports
    # are the best phase-specific receipt for those runs. For current runs the
    # append-only log can be ahead of a report after interruption, so take the
    # per-candidate maximum rather than summing two views of the same calls.
    legacy = _legacy_phase_c_per_candidate(run_dir)
    per_candidate = {
        run_id: max(logged.get(run_id, 0), legacy.get(run_id, 0))
        for run_id in set(logged) | set(legacy)
    }
    return sum(per_candidate.values()), per_candidate


def _phase_c_objective_attempts(report: dict) -> int:
    return sum(
        1
        for stage in report.get("phase_c", {}).get("stages", [])
        for trial in stage.get("trials", [])
        if trial.get("status") != "preflight_rejected"
    )


def _report_attempts(report: dict) -> int:
    phase_a = report.get("phase_a", {})
    warm = phase_a.get("warm_start_configs", [])
    attempted = phase_a.get("trials_attempted")
    if not isinstance(attempted, int) or isinstance(attempted, bool) or attempted < 0:
        attempted = len(warm)
    else:
        attempted = max(attempted, len(warm))
    return attempted + _phase_c_objective_attempts(report)


def _legacy_per_candidate(run_dir: Path) -> dict[str, int]:
    """Best backward-readable totals from framework or hillclimb artifacts."""
    per_candidate: dict[str, int] = {}
    ledger_path = Path(run_dir) / "ledger.json"
    try:
        records = json.loads(ledger_path.read_text()).get("records", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        records = []
    for record in records if isinstance(records, list) else []:
        run_id = record.get("run_id")
        if not isinstance(run_id, str):
            continue
        value = record.get("trials_attempted")
        if value is None:
            value = record.get("trials_completed")
        if value is None:
            value = record.get("warm_start_K")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            per_candidate[run_id] = max(0, int(value))

    for report_path in sorted((Path(run_dir) / "candidates").glob("*/tune_report.json")):
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        run_id = report_path.parent.name
        per_candidate[run_id] = max(
            per_candidate.get(run_id, 0),
            _report_attempts(report),
        )

    # Runs created by the deliberately-simple hillclimb have one root train.py
    # and no candidates/ tree.  Migrate their historical one-row-per-run TSV
    # into the strict attempt log on first reservation.
    results_path = Path(run_dir) / "results.tsv"
    if (
        (Path(run_dir) / "train.py").is_file()
        and not (Path(run_dir) / "candidates").exists()
        and results_path.is_file()
    ):
        lines = [line for line in results_path.read_text().splitlines() if line.strip()]
        if lines and lines[0].split("\t")[:3] == ["step", "score", "status"]:
            run_id = Path(run_dir).name
            per_candidate[run_id] = max(
                per_candidate.get(run_id, 0),
                len(lines) - 1,
            )
    return per_candidate


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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
        rows.append(row)
    return rows


def _summarize_rows(rows: list[dict]) -> tuple[int, dict[str, int]]:
    total = 0
    per_candidate: Counter[str] = Counter()
    for row in rows:
        kind = row.get("kind")
        if kind == "baseline":
            baseline = row.get("per_candidate", {})
            if isinstance(baseline, dict):
                for run_id, value in baseline.items():
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        per_candidate[str(run_id)] += value
                        total += value
        elif kind == "sync":
            synced = row.get("per_candidate")
            if isinstance(synced, dict):
                for run_id, value in synced.items():
                    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                        per_candidate[str(run_id)] += value
                        total += value
            else:
                # Backward-readable fallback for early schema-1 logs that only
                # carried an aggregate migration delta.
                value = row.get("evaluations", 0)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    total += value
        elif kind == "score_attempt":
            total += 1
            run_id = row.get("run_id")
            if isinstance(run_id, str):
                per_candidate[run_id] += 1
    return total, dict(per_candidate)


def attempt_log_summary(run_dir: Path) -> dict[str, Any] | None:
    """Read the append-only attempt log without mutating it.

    This is the public diagnostic view over the same strict parser and
    accounting used by budget admission.  ``baseline`` and ``sync`` rows may
    carry more than one evaluation, so callers must use ``evaluations_done``
    rather than count JSONL rows.  Valid but unfamiliar rows are surfaced for
    observability instead of disappearing into a phase-only tally.
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
    unrecognized_rows = 0
    for row in rows:
        kind = row.get("kind")
        if kind == "score_attempt":
            score_attempts += 1
            phase = row.get("phase")
            if isinstance(phase, str) and phase:
                phase_counts[phase] += 1
            else:
                unclassified_score_attempts += 1
        elif kind not in {"baseline", "sync"}:
            unrecognized_rows += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluations_done": total,
        "per_candidate": per_candidate,
        "phase_counts": dict(sorted(phase_counts.items())),
        "score_attempts": score_attempts,
        "carried_evaluations": total - score_attempts,
        "unclassified_score_attempts": unclassified_score_attempts,
        "unrecognized_rows": unrecognized_rows,
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


def _initialize_or_sync(handle, run_dir: Path) -> tuple[list[dict], int, dict[str, int]]:
    rows = _read_rows(handle)
    legacy = _legacy_per_candidate(run_dir)
    if not rows:
        baseline = {
            "schema_version": SCHEMA_VERSION,
            "kind": "baseline",
            "per_candidate": legacy,
            "evaluations": sum(legacy.values()),
        }
        _append_row(handle, baseline)
        rows.append(baseline)

    total, per_candidate = _summarize_rows(rows)
    missing_by_candidate = {
        run_id: value - per_candidate.get(run_id, 0)
        for run_id, value in legacy.items()
        if value > per_candidate.get(run_id, 0)
    }
    missing_total = sum(missing_by_candidate.values())
    if missing_total:
        sync = {
            "schema_version": SCHEMA_VERSION,
            "kind": "sync",
            "evaluations": missing_total,
            "per_candidate": missing_by_candidate,
            "reason": "legacy ledger/report advanced outside reservation log",
        }
        _append_row(handle, sync)
        rows.append(sync)
        total += missing_total
        for run_id, value in missing_by_candidate.items():
            per_candidate[run_id] = per_candidate.get(run_id, 0) + value
    return rows, total, per_candidate


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
        rows, used, _ = _initialize_or_sync(handle, run_dir)
        if budget is not None and used >= budget:
            raise EvaluationBudgetExhausted(
                used=used,
                budget=budget,
                run_dir=run_dir,
            )
        if str(phase) == "phase_c":
            limits = _deep_tune_limits(run_dir, budget)
            deep_used, deep_per_candidate = _deep_tune_usage(rows, run_dir)
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
            "kind": "score_attempt",
            "attempt_id": f"eval-{used + 1:06d}",
            "run_id": run_id,
            "phase": str(phase),
            "method": str(method),
            "params_sha256": _canonical_hash(params),
        }
        _append_row(handle, receipt)
        rows.append(receipt)
        return receipt


def budget_status(run_dir: Path, *, create: bool = False) -> dict:
    """Return the strict objective usage view without mutating by default."""
    run_dir = Path(run_dir)
    path = run_dir / ATTEMPT_LOG
    legacy = _legacy_per_candidate(run_dir)
    rows: list[dict] = []
    if not path.exists() and not create:
        total = sum(legacy.values())
        per_candidate = legacy
    else:
        with _locked_log(run_dir) as handle:
            if create:
                rows, total, per_candidate = _initialize_or_sync(handle, run_dir)
            else:
                rows = _read_rows(handle)
                total, per_candidate = _summarize_rows(rows)
                for run_id, value in legacy.items():
                    logged = per_candidate.get(run_id, 0)
                    if value > logged:
                        total += value - logged
                        per_candidate[run_id] = value
    budget = _framework_budget(run_dir)
    limits = _deep_tune_limits(run_dir, budget)
    deep_used, deep_per_candidate = _deep_tune_usage(rows, run_dir)
    deep_total_cap = limits["total_cap"]
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluations_done": total,
        "objective_attempts": total,
        "budget": budget,
        "remaining": None if budget is None else max(0, budget - total),
        "reached": None if budget is None else total >= budget,
        "per_candidate": [
            {"run_id": run_id, "evals": value}
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
        help="create/synchronize the append-only attempt log",
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
        params = {
            "candidate_sha256": "sha256:" + hashlib.sha256(ref_path.read_bytes()).hexdigest()
        }
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
