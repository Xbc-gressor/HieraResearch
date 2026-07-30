#!/usr/bin/env python3
"""Fail-closed Phase-C finalization for one selected candidate.

This is the only normal deep-tuning close path.  It validates that every search
stage is terminal, selects the best observation admissible under that terminal
state, applies it to ``BASE_PARAMS``,
closes the report, and writes the candidate's score/status/tuning ledger fields
together.  Every step is idempotent so a coordinator can safely retry this
short finalizer after interruption.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
from pathlib import Path
import sys
import tempfile

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS / "tuners"))

import apply_base_params  # noqa: E402
import ledger  # noqa: E402
from _common import write_tune_report  # noqa: E402
from tune_tools import (  # noqa: E402
    finalizable_tuning_result,
    has_validated_applied_close,
    validate_phase_a_candidate_state,
)


RECOVERY_JOURNAL_FILENAME = ".tuning_finalize_recovery.json"


def _replace_bytes(path: Path, content: bytes) -> None:
    """Atomically restore one durable input after a failed close."""
    tmp = path.with_suffix(path.suffix + ".rollback.tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def _write_recovery_journal(
    journal_path: Path,
    *,
    run_id: str,
    candidate_path: Path,
    report_path: Path,
    candidate_before: bytes,
    report_before: bytes,
) -> None:
    payload = {
        "schema_version": 1,
        "kind": "tuning_finalize_recovery",
        "run_id": str(run_id),
        "candidate_path": str(candidate_path),
        "report_path": str(report_path),
        "candidate_before_base64": base64.b64encode(candidate_before).decode("ascii"),
        "report_before_base64": base64.b64encode(report_before).decode("ascii"),
    }
    tmp = journal_path.with_suffix(journal_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    tmp.replace(journal_path)


def _ledger_committed_final_state(
    ledger_path: Path,
    report_path: Path,
    run_id: str,
) -> bool:
    """Whether the ledger already committed the candidate/report now on disk."""
    try:
        data = json.loads(ledger_path.read_text())
        records = [
            record
            for record in data.get("records", [])
            if isinstance(record, dict) and str(record.get("run_id")) == str(run_id)
        ]
        if len(records) != 1:
            return False
        expected = ledger._applied_incumbent_from_report(
            report_path,
            require_final=True,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    record = records[0]
    return (
        record.get("tune") is True
        and record.get("applied_incumbent") == expected
        and record.get("final_best_score") == expected["score"]
    )


def _recover_interrupted_finalization(
    journal_path: Path,
    *,
    ledger_path: Path,
    candidate_path: Path,
    report_path: Path,
    run_id: str,
) -> None:
    """Restore or forward-complete a close interrupted after journal creation."""
    if not journal_path.exists():
        return
    try:
        payload = json.loads(journal_path.read_text())
        expected_fields = {
            "schema_version",
            "kind",
            "run_id",
            "candidate_path",
            "report_path",
            "candidate_before_base64",
            "report_before_base64",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected_fields
            or payload.get("schema_version") != 1
            or payload.get("kind") != "tuning_finalize_recovery"
            or payload.get("run_id") != str(run_id)
            or payload.get("candidate_path") != str(candidate_path)
            or payload.get("report_path") != str(report_path)
        ):
            raise ValueError("tuning finalization recovery journal is invalid")
        candidate_before = base64.b64decode(
            payload["candidate_before_base64"],
            validate=True,
        )
        report_before = base64.b64decode(
            payload["report_before_base64"],
            validate=True,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        base64.binascii.Error,
    ) as exc:
        raise ValueError(f"cannot recover interrupted tuning finalization: {exc}") from exc

    if not _ledger_committed_final_state(ledger_path, report_path, run_id):
        _replace_bytes(report_path, report_before)
        _replace_bytes(candidate_path, candidate_before)
    journal_path.unlink()


def finalize(
    *,
    candidate_path: Path,
    report_path: Path,
    ledger_path: Path,
    run_id: str,
    task_name: str | None = None,
) -> dict:
    candidate_path = candidate_path.resolve()
    report_path = report_path.resolve()
    ledger_path = ledger_path.resolve()
    if candidate_path.parent != report_path.parent:
        raise ValueError("candidate path and tune report must share one candidate directory")
    if candidate_path.parent.name != run_id:
        raise ValueError(
            f"candidate directory {candidate_path.parent.name!r} does not match run_id {run_id!r}"
        )
    expected_ledger_path = candidate_path.parent.parent.parent / "ledger.json"
    if ledger_path != expected_ledger_path:
        raise ValueError(
            "ledger path must be the ledger.json owned by the candidate's run"
        )

    journal_path = candidate_path.parent / RECOVERY_JOURNAL_FILENAME
    _recover_interrupted_finalization(
        journal_path,
        ledger_path=ledger_path,
        candidate_path=candidate_path,
        report_path=report_path,
        run_id=run_id,
    )
    candidate_before = candidate_path.read_bytes()
    report_before = report_path.read_bytes()
    report = json.loads(report_before)
    if not isinstance(report, dict):
        raise ValueError("tune report must be a JSON object")
    applied_close = has_validated_applied_close(report)
    # Binds phase_a to the candidate on disk, including the SEARCH_SPACE
    # literal and execution revision.
    validate_phase_a_candidate_state(
        report,
        candidate_path,
        require_warm_base_applied=not applied_close,
    )
    result = finalizable_tuning_result(report)
    resolved_task = task_name or ledger.infer_task_name([ledger_path])
    if not resolved_task:
        raise ValueError("cannot infer task name from ledger path; pass --task")

    rendered_source, apply_receipt = apply_base_params.render(
        candidate_path,
        result["best_params"],
    )
    closed_report = copy.deepcopy(report)
    closed_report["final_best_params"] = result["best_params"]
    closed_report["final_best_score"] = result["best_score"]
    closed_report["applied_to_base_params"] = True

    # Render the exact target candidate/report off to the side, then validate
    # the complete prospective ledger record (including parameter-transfer and
    # lineage-revision rules) before the first durable mutation.
    with tempfile.TemporaryDirectory(prefix="tuning-finalize-preview-") as tmp:
        preview_dir = Path(tmp) / run_id
        preview_dir.mkdir()
        preview_candidate = preview_dir / "train.py"
        preview_report = preview_dir / "tune_report.json"
        preview_candidate.write_text(rendered_source)
        write_tune_report(preview_report, closed_report)
        ledger.validate_tuning_finalization(
            ledger_path,
            resolved_task,
            run_id,
            preview_report,
            owned_report_path=report_path,
        )

    # Reject a concurrent/stale close instead of applying a plan rendered from
    # different durable inputs.
    if candidate_path.read_bytes() != candidate_before:
        raise ValueError("candidate changed while tuning finalization was prepared")
    if report_path.read_bytes() != report_before:
        raise ValueError("tune report changed while tuning finalization was prepared")

    _write_recovery_journal(
        journal_path,
        run_id=run_id,
        candidate_path=candidate_path,
        report_path=report_path,
        candidate_before=candidate_before,
        report_before=report_before,
    )
    try:
        apply_base_params.commit_rendered(
            candidate_path,
            rendered_source,
            apply_receipt,
        )
        write_tune_report(report_path, closed_report)

        record = ledger.finalize_tuning(
            ledger_path,
            resolved_task,
            run_id,
            report_path,
        )
    except BaseException as exc:
        # A ledger-side rejection or write failure must not strand a candidate
        # and report that claim to be finalized.  If the ledger commit itself
        # landed and only later loop-state bookkeeping failed, keep the
        # consistent forward state so an idempotent retry can finish it.
        if not _ledger_committed_final_state(ledger_path, report_path, run_id):
            restore_errors = []
            for path, content in (
                (report_path, report_before),
                (candidate_path, candidate_before),
            ):
                try:
                    _replace_bytes(path, content)
                except OSError as restore_exc:
                    restore_errors.append(f"{path}: {restore_exc}")
            if restore_errors:
                raise RuntimeError(
                    "tuning finalization failed and rollback was incomplete: "
                    + "; ".join(restore_errors)
                ) from exc
        journal_path.unlink(missing_ok=True)
        raise
    journal_path.unlink()
    return {
        "status": "ok",
        "run_id": run_id,
        "phase_c_method": record.get("phase_c_method"),
        "best_warm_score": record.get("best_warm_score"),
        "final_best_score": record.get("final_best_score"),
        "trials_completed": record.get("trials_completed"),
        "trials_attempted": record.get("trials_attempted"),
        "preflight_attempts": record.get("preflight_attempts"),
        "preflight_failures": record.get("preflight_failures"),
        "feasibility_rejections": record.get("feasibility_rejections"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "applied": record.get("applied"),
        "ledger_updated": True,
        "apply_mode": apply_receipt["mode"],
        "report_path": str(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--tune-report-json", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task")
    args = parser.parse_args()
    try:
        result = finalize(
            candidate_path=args.candidate_path,
            report_path=args.tune_report_json,
            ledger_path=args.ledger,
            run_id=args.run_id,
            task_name=args.task,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
