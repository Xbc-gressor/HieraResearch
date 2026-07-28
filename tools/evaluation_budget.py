#!/usr/bin/env python3
"""Durable, strict reservation ledger for objective-function evaluations.

Every tuner calls :func:`reserve_evaluation` immediately before entering the
task's ``score_fn``.  The append-only JSONL file is the crash-safe authority
for the run-level evaluation cap; aggregate fields in ``ledger.json`` remain
the bounded per-candidate summary.

Preflight calls never use this module.  They are tracked separately in each
candidate's ``tune_report.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
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

    def __init__(self, *, used: int, budget: int, run_dir: Path):
        self.used = used
        self.budget = budget
        self.run_dir = Path(run_dir)
        super().__init__(
            f"evaluation budget exhausted before score_fn "
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
    """Best backward-readable objective totals from ledger + tuner reports."""
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
    with _locked_log(run_dir) as handle:
        rows, used, _ = _initialize_or_sync(handle, run_dir)
        if budget is not None and used >= budget:
            raise EvaluationBudgetExhausted(
                used=used,
                budget=budget,
                run_dir=run_dir,
            )
        run_id = Path(ref_path).resolve().parent.name
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
    if not path.exists() and not create:
        total = sum(legacy.values())
        per_candidate = legacy
    else:
        with _locked_log(run_dir) as handle:
            if create:
                _, total, per_candidate = _initialize_or_sync(handle, run_dir)
            else:
                rows = _read_rows(handle)
                total, per_candidate = _summarize_rows(rows)
                for run_id, value in legacy.items():
                    logged = per_candidate.get(run_id, 0)
                    if value > logged:
                        total += value - logged
                        per_candidate[run_id] = value
    budget = _framework_budget(run_dir)
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
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(budget_status(args.run_dir, create=args.initialize)))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
