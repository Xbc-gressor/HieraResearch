"""Small task-native evaluator runner.

Tasks may provide a callable ``run``/``proxy``/``protocol`` entry in their
contract.  The runner owns record construction and output protocol checks;
callables return a mapping containing predictions and optional test submission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import time
import inspect
from typing import Any, Callable, Mapping

try:  # package import (``tools.task_evaluator``)
    from .evaluation_records import EvaluationRecord, append_record
except ImportError:  # direct script/test execution with ``tools/`` on PYTHONPATH
    from evaluation_records import EvaluationRecord, append_record

@dataclass(frozen=True)
class EvaluationContract:
    task_id: str
    version: str
    metric_name: str
    direction: str
    stages: dict[str, str]
    resources: dict[str, Any]
    data_version: str = ""

    @classmethod
    def from_task(cls, task_id: str, task_toml: Path) -> "EvaluationContract":
        import tomllib
        with Path(task_toml).open("rb") as stream:
            doc = tomllib.load(stream)
        evaluation = doc.get("evaluation", {})
        result = doc.get("result", {})
        if not isinstance(evaluation, dict) or not isinstance(result, dict):
            raise ValueError("task contract sections must be tables")
        stages = {k: str(evaluation[k]) for k in ("run", "proxy", "protocol", "official") if k in evaluation}
        if "run" not in stages and "score_fn" in evaluation:
            stages["run"] = str(evaluation["score_fn"])
        if not stages:
            raise ValueError("evaluation contract needs run or score_fn")
        return cls(task_id, str(evaluation.get("version", "1")), str(result.get("metric", "score")), str(result.get("direction", "min")), stages, dict(doc.get("resources", {}) or {}), str(evaluation.get("data_version", "")))

class EvaluationFailure(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message); self.kind = kind

def _digest(value: Any) -> str:
    if isinstance(value, Path):
        h = hashlib.sha256()
        for p in sorted(value.rglob("*")):
            if p.is_file(): h.update(p.relative_to(value).as_posix().encode()); h.update(p.read_bytes())
        return h.hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def validate_predictions(predictions: Any, expected_ids: list[Any], *, name: str) -> list[Any]:
    if not isinstance(predictions, Mapping):
        raise EvaluationFailure("invalid-output", f"{name} predictions must be an id->value mapping")
    got = list(predictions)
    if got != expected_ids and set(got) != set(expected_ids):
        raise EvaluationFailure("invalid-output", f"{name} prediction IDs do not match expected IDs")
    if len(predictions) != len(expected_ids):
        raise EvaluationFailure("invalid-output", f"{name} prediction count mismatch")
    return [predictions[i] for i in expected_ids]

class EvaluationRunner:
    def __init__(self, contract: EvaluationContract, *, evaluator_module: Any,
                 reporting_root: Path | None = None, record_path: Path | None = None):
        self.contract = contract
        self.module = evaluator_module
        self.reporting_root = Path(reporting_root).resolve() if reporting_root else None
        self.record_path = Path(record_path) if record_path else None

    def evaluate(self, artifact: Path, *, candidate_id: str, input_revision: str,
                 stage: str = "proxy", fidelity: str = "fast", params: dict | None = None) -> EvaluationRecord:
        started = time.monotonic()
        record_stage = "proxy" if stage == "run" else stage
        try:
            fn_name = self.contract.stages.get(stage)
            if not fn_name: raise EvaluationFailure("preflight", f"stage {stage!r} is not declared")
            fn = getattr(self.module, fn_name, None)
            if not callable(fn): raise EvaluationFailure("evaluator", f"missing evaluator callable {fn_name!r}")
            supplied = params or {}
            legacy_adapter = False
            # Native artifact evaluators receive artifact/params and stage
            # metadata.  A legacy score_fn adapter may only accept
            # (make_model, params); callers must explicitly provide the
            # adapter in params so this fallback cannot silently load labels.
            try:
                signature = inspect.signature(fn)
                if "stage" in signature.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
                    result = fn(Path(artifact), supplied, stage=stage, fidelity=fidelity)
                elif len(signature.parameters) >= 2 and "make_model" in supplied:
                    legacy_adapter = True
                    score = fn(supplied["make_model"], supplied)
                    result = {"expected_holdout_ids": supplied.get("expected_holdout_ids", []), "holdout_predictions": supplied.get("holdout_predictions", {}), "score": score}
                else:
                    result = fn(Path(artifact), supplied)
            except (TypeError, ValueError) as exc:
                raise EvaluationFailure("evaluator", f"evaluator invocation failed: {exc}") from exc
            if not isinstance(result, Mapping): raise EvaluationFailure("invalid-output", "evaluator result must be a mapping")
            expected = list(result.get("expected_holdout_ids", []))
            if not expected and not legacy_adapter:
                raise EvaluationFailure("invalid-output", "evaluator did not provide holdout IDs")
            holdout = validate_predictions(result.get("holdout_predictions", {}), expected, name="holdout")
            test_expected = result.get("expected_test_ids")
            if record_stage in {"protocol", "official"} and not test_expected:
                raise EvaluationFailure("invalid-output", "protocol/official evaluation requires test IDs")
            if test_expected is not None:
                validate_predictions(result.get("test_submission", {}), list(test_expected), name="test")
            score = result.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise EvaluationFailure("evaluator", "evaluator did not return a numeric score")
            output = {"holdout_predictions": holdout, "test_submission": result.get("test_submission"), "metric": self.contract.metric_name}
            output_digest = _digest(output)
            output_path = None
            if self.reporting_root is not None:
                output_path = self.reporting_root / self.contract.task_id / candidate_id / f"{output_digest}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(output, sort_keys=True) + "\n", encoding="utf-8")
            record = EvaluationRecord(self.contract.task_id, candidate_id, input_revision, output_digest, self.contract.version, record_stage, fidelity, self.contract.metric_name, float(score), self.contract.direction, data_version=self.contract.data_version, output_kind="predictions", output_artifact=str(output_path or artifact), runtime_seconds=time.monotonic()-started, selection_visible=record_stage != "official", metadata={"isolation_mode": "soft"})
            if self.record_path is not None:
                append_record(self.record_path, record)
            return record
        except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
            record = EvaluationRecord(self.contract.task_id, candidate_id, input_revision, None, self.contract.version, record_stage, fidelity, self.contract.metric_name, None, self.contract.direction, data_version=self.contract.data_version, status="failed", failure_kind="invalid-output", runtime_seconds=time.monotonic()-started, selection_visible=record_stage != "official", metadata={"error": str(exc), "isolation_mode": "soft"})
            if self.record_path is not None:
                append_record(self.record_path, record)
            return record
        except EvaluationFailure as exc:
            record = EvaluationRecord(self.contract.task_id, candidate_id, input_revision, None, self.contract.version, record_stage, fidelity, self.contract.metric_name, None, self.contract.direction, data_version=self.contract.data_version, status="failed", failure_kind=exc.kind, runtime_seconds=time.monotonic()-started, selection_visible=record_stage != "official", metadata={"error": str(exc), "isolation_mode": "soft"})
            if self.record_path is not None:
                append_record(self.record_path, record)
            return record
