"""Deterministic gate and bounded state for LLM-assisted candidate debugging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactError, atomic_write_json
from .models import DebugDecision, DebugVerdict


@dataclass(frozen=True)
class FailureEvidence:
    run_id: str
    phase: str
    crash_index: int
    crash_params: dict[str, Any]
    failure_receipt: dict[str, Any]
    failure_ref: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        value = self.failure_ref.get("failure_id")
        if not isinstance(value, str) or not value:
            raise ValueError("failure_ref.failure_id must be a non-empty string")
        return value

    @classmethod
    def from_worker_payload(cls, run_id: str, payload: Any) -> "FailureEvidence":
        if not isinstance(payload, dict):
            raise ValueError("warmstart crash payload must be an object")
        if payload.get("status") != "crashed" or payload.get("phase") not in {"preflight", "a"}:
            raise ValueError("only evidenced preflight/evaluation crashes are debuggable")
        index = payload.get("crash_index")
        params = payload.get("crash_params")
        receipt = payload.get("failure_receipt")
        ref = payload.get("failure_ref")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("crash_index must be a non-negative integer")
        if not isinstance(params, dict) or not isinstance(receipt, dict) or not isinstance(ref, dict):
            raise ValueError("debuggable crash requires params, failure_receipt, and failure_ref")
        evidence = cls(
            run_id=run_id,
            phase=str(payload["phase"]),
            crash_index=index,
            crash_params=params,
            failure_receipt=receipt,
            failure_ref=ref,
        )
        _ = evidence.fingerprint
        return evidence


class DebugPolicy:
    """Reserve one analyzer call per candidate/failure fingerprint."""

    def __init__(self, run_dir: Path, *, max_code_repairs_per_candidate: int = 3):
        self.path = Path(run_dir) / ".orchestrator" / "debug_attempts.json"
        self.max_code_repairs_per_candidate = max_code_repairs_per_candidate

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "failures": {}, "code_repairs": {}}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"invalid debug attempt state {self.path}: {exc}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ArtifactError(f"unsupported debug attempt state: {self.path}")
        if not isinstance(value.get("failures"), dict) or not isinstance(value.get("code_repairs"), dict):
            raise ArtifactError(f"malformed debug attempt state: {self.path}")
        return value

    def reserve_analysis(self, evidence: FailureEvidence) -> None:
        state = self._load()
        key = f"{evidence.run_id}:{evidence.fingerprint}"
        if int(state["failures"].get(key, 0)) >= 1:
            raise ArtifactError(
                f"debug analyzer already invoked for candidate/failure {key}"
            )
        state["failures"][key] = 1
        atomic_write_json(self.path, state)

    def reserve_code_repair(self, run_id: str) -> None:
        state = self._load()
        count = int(state["code_repairs"].get(run_id, 0))
        if count >= self.max_code_repairs_per_candidate:
            raise ArtifactError(
                f"candidate {run_id} reached its code repair cap "
                f"({self.max_code_repairs_per_candidate})"
            )
        state["code_repairs"][run_id] = count + 1
        atomic_write_json(self.path, state)


def parse_debug_response(value: Any) -> DebugDecision:
    if not isinstance(value, dict):
        raise ValueError("debug response must be an object")
    verdict = value.get("verdict")
    corrected_entries = value.get("corrected_config")
    instructions = value.get("repair_instructions")
    if not isinstance(corrected_entries, list) or not isinstance(instructions, str):
        raise ValueError("debug repair payload fields have invalid types")
    corrected: dict[str, Any] | None = None
    if verdict == DebugVerdict.CONFIG_INVALID.value:
        corrected = {}
        for entry in corrected_entries:
            if not isinstance(entry, dict) or set(entry) != {"key", "value_json"}:
                raise ValueError("corrected_config entries require key and value_json")
            key = entry.get("key")
            encoded = entry.get("value_json")
            if not isinstance(key, str) or not key or not isinstance(encoded, str):
                raise ValueError("corrected_config entry values are invalid")
            if key in corrected:
                raise ValueError(f"duplicate corrected_config key: {key}")
            try:
                corrected[key] = json.loads(encoded)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrected_config value for {key} is not JSON") from exc
    normalized = {
        "verdict": verdict,
        "rationale": value.get("rationale"),
        "corrected_config": corrected,
        "repair_instructions": instructions or None,
    }
    return DebugDecision.from_dict(normalized)
