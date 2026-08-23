from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from driver.events import EventsLog
from driver.roles import InvocationContext, PROMPT_DIR, RoleDefinition
from driver.session import SDKSessionRunner


SAMPLER_ROLE = RoleDefinition(
    name="hpo-candidate-sampler",
    prompt_file="hpo-candidate-sampler.md",
    tools=(),
    disallowed=(),
    receipt_schema={"configs": "list", "rationale": "?str"},
    corrective_attempts=1,
    max_turns=4,
)

SCORER_ROLE = RoleDefinition(
    name="hpo-llambo-scorer",
    prompt_file="hpo-llambo-scorer.md",
    tools=(),
    disallowed=(),
    receipt_schema={"predictions": "list"},
    corrective_attempts=1,
    max_turns=4,
)

ROLES = {"sampler": SAMPLER_ROLE, "scorer": SCORER_ROLE}


@dataclass(frozen=True)
class LLMResponse:
    receipt: dict[str, Any]
    usage: dict[str, Any]


class LLMCallFailed(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]):
        self.usage = usage
        super().__init__(message)


class LLMProvider(Protocol):
    def call(self, kind: str, payload: dict[str, Any]) -> LLMResponse:
        ...


class SDKLLMProvider:
    """One SDK session per call; no transcript is resumed across BO trials."""

    def __init__(self, *, model: str, session_root: Path):
        self.model = model
        self.session_root = Path(session_root)
        self._counter = 0

    def call(self, kind: str, payload: dict[str, Any]) -> LLMResponse:
        try:
            role = ROLES[kind]
        except KeyError:
            raise ValueError(f"unknown LLM call kind: {kind}") from None
        self._counter += 1
        call_dir = self.session_root / f"call-{self._counter:04d}-{kind}"
        events = EventsLog(call_dir)
        runner = SDKSessionRunner(model=self.model, events=events)
        payload_json = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        ctx = InvocationContext(
            task="hyperparameter-optimization",
            tag="structured-call",
            run_dir=call_dir,
            invocation_id=1,
            extra={"payload_json": payload_json},
        )
        receipt = None
        problem = None
        try:
            receipt = runner.run(role, ctx)
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
        usage = _read_usage(
            events.path,
            role_name=role.name,
            invocation_id=1,
            model=self.model,
            payload_json=payload_json,
            prompt_path=PROMPT_DIR / role.prompt_file,
            status="ok" if receipt is not None else "failed",
            problem=problem,
        )
        if receipt is None:
            raise LLMCallFailed(problem or "LLM call failed", usage)
        return LLMResponse(receipt=receipt, usage=usage)


def _read_usage(
    path: Path,
    *,
    role_name: str,
    invocation_id: int,
    model: str,
    payload_json: str,
    prompt_path: Path,
    status: str,
    problem: str | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "role": role_name,
        "requested_model": model,
        "status": status,
        "problem": problem,
        "session_ends": 0,
        "num_turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "total_cost_usd": 0.0,
        "payload_sha256": hashlib.sha256(payload_json.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "events_path": str(path),
    }
    if not path.exists():
        return entry
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (
            row.get("kind") != "session_end"
            or row.get("role") != role_name
            or row.get("invocation_id") != invocation_id
        ):
            continue
        entry["session_ends"] += 1
        entry["num_turns"] += int(row.get("num_turns", 0) or 0)
        entry["total_cost_usd"] += float(row.get("total_cost_usd", 0.0) or 0.0)
        usage = row.get("usage") or {}
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            entry[key] += int(usage.get(key, 0) or 0)
    return entry
