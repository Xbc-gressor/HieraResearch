"""Small typed contracts shared by the coordinator boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Transition(str, Enum):
    INITIALIZE = "initialize"
    ENVIRONMENT_PREFLIGHT = "environment_preflight"
    BUILD_BACKGROUND = "build_background"
    ADMIT_BASELINE = "admit_baseline"
    ADMIT_ROUND = "admit_round"
    MATERIALIZE_CANDIDATE = "materialize_candidate"
    BUILD_TUNING_CONTRACT = "build_tuning_contract"
    EVALUATE_WARM_CONFIGS = "evaluate_warm_configs"
    DEBUG_CANDIDATE = "debug_candidate"
    DEEP_TUNE = "deep_tune"
    REFRESH_EXPERIENCE = "refresh_experience"
    COMPLETE = "complete"
    STOP = "stop"


class DebugVerdict(str, Enum):
    CONFIG_INVALID = "config_invalid"
    CODE_INCOMPATIBLE = "code_incompatible"
    ABANDON = "abandon"


class CoordinatorPhase(str, Enum):
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"


@dataclass(frozen=True)
class RunIdentity:
    repo_root: Path
    task_name: str
    tag: str

    @property
    def run_dir(self) -> Path:
        return self.repo_root / "runs" / self.task_name / self.tag

    @property
    def ledger_path(self) -> Path:
        return self.run_dir / "ledger.json"


@dataclass
class RoundAction:
    op: str
    parents: list[str] = field(default_factory=list)
    run_id: str | None = None
    admitted: bool = False
    materialized: bool = False
    implemented: bool = False
    contract_ready: bool = False
    resolved: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RoundAction":
        op = value.get("op")
        parents = value.get("parents", [])
        if op not in {"fresh", "improve", "crossover"}:
            raise ValueError(f"invalid round action op: {op!r}")
        if not isinstance(parents, list) or not all(isinstance(v, str) for v in parents):
            raise ValueError("round action parents must be a list of strings")
        return cls(
            op=op,
            parents=parents,
            run_id=value.get("run_id"),
            admitted=bool(value.get("admitted", False)),
            materialized=bool(value.get("materialized", False)),
            implemented=bool(value.get("implemented", False)),
            contract_ready=bool(value.get("contract_ready", False)),
            resolved=bool(value.get("resolved", False)),
        )


@dataclass
class ActiveRound:
    round_id: int
    actions: list[RoundAction]
    admission_complete: bool = False
    tuning_complete: bool = False
    evaluations_before: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActiveRound":
        actions = value.get("actions")
        if not isinstance(actions, list):
            raise ValueError("active round actions must be a list")
        return cls(
            round_id=int(value["round_id"]),
            actions=[RoundAction.from_dict(item) for item in actions],
            admission_complete=bool(value.get("admission_complete", False)),
            tuning_complete=bool(value.get("tuning_complete", False)),
            evaluations_before=int(value.get("evaluations_before", 0)),
        )


@dataclass
class CoordinatorState:
    task_name: str
    tag: str
    schema_version: int = 1
    phase: CoordinatorPhase = CoordinatorPhase.RUNNING
    next_round_id: int = 0
    active_round: ActiveRound | None = None
    last_transition: str | None = None
    stop_condition: str | None = None
    no_progress_cycles: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CoordinatorState":
        if value.get("schema_version") != 1:
            raise ValueError(
                f"unsupported coordinator state schema: {value.get('schema_version')!r}"
            )
        active = value.get("active_round")
        return cls(
            task_name=str(value["task_name"]),
            tag=str(value["tag"]),
            schema_version=1,
            phase=CoordinatorPhase(value.get("phase", "running")),
            next_round_id=int(value.get("next_round_id", 0)),
            active_round=ActiveRound.from_dict(active) if isinstance(active, dict) else None,
            last_transition=value.get("last_transition"),
            stop_condition=value.get("stop_condition"),
            no_progress_cycles=int(value.get("no_progress_cycles", 0)),
        )


@dataclass(frozen=True)
class IdeaProposal:
    idea: str
    change: str
    candidate_name_hint: str
    description: str

    @classmethod
    def from_dict(cls, value: Any) -> "IdeaProposal":
        if not isinstance(value, dict):
            raise ValueError("idea response must be an object")
        expected = {"idea", "change", "candidate_name_hint", "description"}
        if set(value) != expected:
            raise ValueError(f"idea response fields must be exactly {sorted(expected)}")
        fields: dict[str, str] = {}
        for key in expected:
            item = value[key]
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"idea response {key} must be a non-empty string")
            fields[key] = item.strip()
        return cls(**fields)


@dataclass(frozen=True)
class DebugDecision:
    verdict: DebugVerdict
    rationale: str
    corrected_config: dict[str, Any] | None = None
    repair_instructions: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "DebugDecision":
        if not isinstance(value, dict):
            raise ValueError("debug response must be an object")
        allowed = {"verdict", "rationale", "corrected_config", "repair_instructions"}
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"unknown debug response fields: {sorted(extra)}")
        verdict = DebugVerdict(value.get("verdict"))
        rationale = value.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("debug rationale must be a non-empty string")
        corrected = value.get("corrected_config")
        instructions = value.get("repair_instructions")
        if verdict is DebugVerdict.CONFIG_INVALID:
            if not isinstance(corrected, dict) or not corrected:
                raise ValueError("config_invalid requires a non-empty corrected_config")
            if instructions is not None:
                raise ValueError("config_invalid must not include repair_instructions")
        elif verdict is DebugVerdict.CODE_INCOMPATIBLE:
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError("code_incompatible requires repair_instructions")
            if corrected is not None:
                raise ValueError("code_incompatible must not include corrected_config")
        else:
            if corrected is not None or instructions is not None:
                raise ValueError("abandon must not include a repair payload")
        return cls(
            verdict=verdict,
            rationale=rationale.strip(),
            corrected_config=corrected,
            repair_instructions=instructions.strip() if isinstance(instructions, str) else None,
        )
