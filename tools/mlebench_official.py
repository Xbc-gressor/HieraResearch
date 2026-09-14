"""Operator-side MLE-bench official grader adapter.

The adapter is intentionally independent of candidate execution.  It invokes
an installed ``grade``/``grade-sample`` command only after submission audit;
its output is written under an operator reporting root and never returned to
candidate-facing selection state.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Sequence

try:  # package import (``tools.mlebench_official``)
    from .evaluation_records import EvaluationRecord, append_record
    from .mlebench_isolation import IsolationError, audit_artifact_paths
except ImportError:  # direct script/test execution with ``tools/`` on PYTHONPATH
    from evaluation_records import EvaluationRecord, append_record
    from mlebench_isolation import IsolationError, audit_artifact_paths

class OfficialPreflightError(RuntimeError):
    pass

def validate_submission(path: Path, *, required_ids: set[str] | None = None) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise OfficialPreflightError(f"submission is missing: {path}")
    if path.name != "submission.csv":
        raise OfficialPreflightError("submission must be named submission.csv")
    text = path.read_text(encoding="utf-8", errors="strict")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or "," not in lines[0]:
        raise OfficialPreflightError("submission.csv has no usable header/rows")
    header = [part.strip() for part in lines[0].split(",")]
    if len(set(header)) != len(header) or not header[0]:
        raise OfficialPreflightError("submission.csv has invalid columns")
    if required_ids is not None:
        ids = {line.split(",", 1)[0].strip() for line in lines[1:]}
        if ids != required_ids:
            raise OfficialPreflightError("submission IDs do not match the task test IDs")
    return {"path": str(path), "rows": len(lines) - 1, "columns": header}

def run_official_grade(*, task_id: str, candidate_id: str, submission: Path,
                       grader: Sequence[str], reporting_root: Path,
                       input_revision: str, fidelity: str = "full",
                       timeout_seconds: float = 3600,
                       record_path: Path | None = None) -> EvaluationRecord:
    started = time.monotonic()
    try:
        audit_artifact_paths(submission.parent)
        manifest = validate_submission(submission)
        try:
            Path(reporting_root).resolve().relative_to(submission.parent.resolve())
        except ValueError:
            pass
        else:
            raise OfficialPreflightError("reporting root must be outside candidate submission tree")
        if not grader or shutil.which(grader[0]) is None:
            raise OfficialPreflightError("official grader command is unavailable")
        proc = subprocess.run(list(grader) + [str(submission)], capture_output=True,
                              text=True, timeout=timeout_seconds, check=False)
        if proc.returncode != 0:
            raise OfficialPreflightError(f"grader exited {proc.returncode}: {proc.stderr[-500:]}")
        payload = {"task_id": task_id, "candidate_id": candidate_id, "submission": manifest,
                   "grader": list(grader), "stdout": proc.stdout}
        root = Path(reporting_root).resolve() / task_id / candidate_id
        root.mkdir(parents=True, exist_ok=True)
        result_path = root / "official-grade.json"
        result_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        try:
            score = float(proc.stdout.strip().splitlines()[-1].split()[-1])
        except (ValueError, IndexError):
            raise OfficialPreflightError("grader returned no parseable numeric score")
        record = EvaluationRecord(task_id, candidate_id, input_revision, _digest(result_path), "official-1", "official", fidelity, "official_grade", score, output_kind="official_report", output_artifact=str(result_path), runtime_seconds=time.monotonic()-started, selection_visible=False, metadata={"gold_protection": "temporal_soft", "grader": list(grader)})
    except (OfficialPreflightError, IsolationError, subprocess.TimeoutExpired) as exc:
        kind = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "preflight"
        record = EvaluationRecord(task_id, candidate_id, input_revision, None, "official-1", "official", fidelity, "official_grade", None, status="failed", failure_kind=kind, runtime_seconds=time.monotonic()-started, selection_visible=False, metadata={"error": str(exc), "gold_protection": "temporal_soft"})
    if record_path is not None:
        append_record(record_path, record)
    return record

def _digest(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()
