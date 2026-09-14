"""Task-native evaluation records and same-domain score projections.

The evaluator is the only writer of per-candidate JSONL records.  Ledger and
scheduler code should consume :func:`selection_view`, never raw scores.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

STAGES = {"proxy", "protocol", "official"}
FAILURES = {None, "preflight", "runtime", "invalid-output", "resource", "evaluator", "timeout", "isolation_violation"}

@dataclass(frozen=True)
class EvaluationRecord:
    task_id: str
    candidate_id: str
    input_revision: str
    output_artifact_digest: str | None
    contract_version: str
    stage: str
    fidelity: str
    metric_name: str
    score: float | None
    direction: str = "min"
    unit: str = ""
    data_version: str = ""
    split: str = ""
    seed: int | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float | None = None
    status: str = "success"
    output_kind: str = "score_only"
    output_artifact: str | None = None
    failure_kind: str | None = None
    selection_visible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"invalid evaluation stage: {self.stage!r}")
        if not self.fidelity:
            raise ValueError("fidelity must be non-empty")
        if self.direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'")
        if self.failure_kind not in FAILURES:
            raise ValueError(f"invalid failure_kind: {self.failure_kind!r}")
        if self.stage == "official" and self.selection_visible:
            raise ValueError("official records must not be selection-visible")
        if self.status == "success" and self.score is None:
            raise ValueError("successful records require a score")
        if self.status == "success" and not self.input_revision:
            raise ValueError("successful records require input_revision")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["record_digest"] = digest_record(value)
        return value


def digest_record(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "record_digest"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256(payload).hexdigest()


def append_record(path: Path, record: EvaluationRecord | dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = record.to_dict() if isinstance(record, EvaluationRecord) else dict(record)
    if not value.get("record_digest"):
        value["record_digest"] = digest_record(value)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
    return value["record_digest"]


def read_records(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("record_digest") != digest_record(row):
                raise ValueError("evaluation record digest mismatch")
            rows.append(row)
    return rows


def selection_view(records: Iterable[dict[str, Any]], *, stage: str, fidelity: str) -> list[dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"invalid stage: {stage!r}")
    if stage == "official":
        raise PermissionError("official records are operator-only")
    view = []
    for row in records:
        if row.get("stage") != stage or row.get("fidelity") != fidelity:
            continue
        if not row.get("selection_visible", True) or row.get("status") != "success":
            continue
        if row.get("score") is None or not row.get("input_revision") or not row.get("output_artifact_digest"):
            continue
        view.append(row)
    return view


def reporting_view(records: Iterable[dict[str, Any]], *, include_selection: bool = True) -> list[dict[str, Any]]:
    """Return the complete operator view, including official measurements.

    This function is intentionally separate from ``selection_view`` so a
    caller cannot accidentally widen a scheduler query to include official
    results merely because they happen to be present in the ledger.
    """
    rows = [dict(row) for row in records]
    if include_selection:
        return rows
    return [row for row in rows if not row.get("selection_visible", True)]


def best_score(records: Iterable[dict[str, Any]], *, stage: str, fidelity: str) -> float | None:
    rows = selection_view(records, stage=stage, fidelity=fidelity)
    if not rows:
        return None
    direction = rows[0].get("direction", "min")
    scores = [float(r["score"]) for r in rows]
    return (min if direction == "min" else max)(scores)


def legacy_score_record(*, task_id: str, candidate_id: str, input_revision: str,
                        score: float, metric_name: str, fidelity: str = "fast",
                        output_artifact: Path | None = None) -> EvaluationRecord:
    """Wrap a scalar compatibility-adapter measurement in the new schema."""
    digest = None
    if output_artifact is not None and Path(output_artifact).is_file():
        digest = sha256(Path(output_artifact).read_bytes()).hexdigest()
    return EvaluationRecord(task_id, candidate_id, input_revision, digest,
                            "legacy-1", "proxy", fidelity, metric_name,
                            float(score), output_kind="score_only",
                            output_artifact=str(output_artifact) if output_artifact else None)
