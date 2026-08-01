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
    PREFLIGHT_CANDIDATE = "preflight_candidate"
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
    preflight_ready: bool = False
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
            preflight_ready=bool(value.get("preflight_ready", False)),
            resolved=bool(value.get("resolved", False)),
        )


@dataclass(frozen=True)
class DeepTuneSelection:
    """The one population-level tuning decision reserved for a round.

    ``input_revision`` records the admission basis. It is intentionally not
    revalidated on resume because the selected worker may already have changed
    the attempt log, report, or ledger before interruption.
    """

    run_id: str | None
    reason: str
    trial_cap: int | None
    input_revision: str

    def __post_init__(self) -> None:
        if self.run_id is not None and (
            not isinstance(self.run_id, str) or not self.run_id.isdigit()
        ):
            raise ValueError(f"invalid selected tuning run id: {self.run_id!r}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("deep-tune selection reason must be a non-empty string")
        object.__setattr__(self, "reason", self.reason.strip())
        if (
            not isinstance(self.input_revision, str)
            or len(self.input_revision) != len("sha256:") + 64
            or not self.input_revision.startswith("sha256:")
            or any(
                char not in "0123456789abcdef"
                for char in self.input_revision.removeprefix("sha256:")
            )
        ):
            raise ValueError(
                "deep-tune selection requires a lowercase sha256 input revision"
            )
        if self.run_id is None and self.trial_cap is not None:
            raise ValueError("no-op deep-tune selection must not carry a trial cap")
        if self.run_id is not None and (
            not isinstance(self.trial_cap, int)
            or isinstance(self.trial_cap, bool)
            or self.trial_cap <= 0
        ):
            raise ValueError(
                "selected deep-tune candidate requires a positive trial_cap"
            )

    @classmethod
    def from_tool_result(
        cls,
        value: Any,
        *,
        input_revision: str,
    ) -> "DeepTuneSelection":
        if not isinstance(value, dict):
            raise ValueError("deep-tune selection must be an object")
        run_id = value.get("run_id")
        allocation = value.get("budget_allocation")
        trial_cap = allocation.get("trial_cap") if isinstance(allocation, dict) else None
        return cls(
            run_id=run_id,
            reason=value.get("reason"),
            trial_cap=trial_cap if run_id is not None else None,
            input_revision=input_revision,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "DeepTuneSelection":
        if not isinstance(value, dict):
            raise ValueError("deep-tune selection state must be an object")
        expected = {"run_id", "reason", "trial_cap", "input_revision"}
        if set(value) != expected:
            raise ValueError(
                f"deep-tune selection fields must be exactly {sorted(expected)}"
            )
        return cls(
            run_id=value["run_id"],
            reason=value["reason"],
            trial_cap=value["trial_cap"],
            input_revision=value["input_revision"],
        )


@dataclass(frozen=True)
class DeepTuneOutcome:
    tuned_run_id: str | None
    ledger_updated: bool
    reason: str

    def __post_init__(self) -> None:
        if self.tuned_run_id is not None and (
            not isinstance(self.tuned_run_id, str)
            or not self.tuned_run_id.isdigit()
        ):
            raise ValueError(f"invalid tuned run id: {self.tuned_run_id!r}")
        if not isinstance(self.ledger_updated, bool):
            raise ValueError("deep-tune outcome ledger_updated must be boolean")
        if self.ledger_updated and self.tuned_run_id is None:
            raise ValueError("ledger-updating deep-tune outcome requires a run id")
        if self.tuned_run_id is not None and not self.ledger_updated:
            raise ValueError("tuned run id requires a ledger-updating outcome")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("deep-tune outcome reason must be a non-empty string")
        object.__setattr__(self, "reason", self.reason.strip())

    @classmethod
    def from_dict(cls, value: Any) -> "DeepTuneOutcome":
        if not isinstance(value, dict):
            raise ValueError("deep-tune outcome state must be an object")
        expected = {"tuned_run_id", "ledger_updated", "reason"}
        if set(value) != expected:
            raise ValueError(
                f"deep-tune outcome fields must be exactly {sorted(expected)}"
            )
        return cls(
            tuned_run_id=value["tuned_run_id"],
            ledger_updated=value["ledger_updated"],
            reason=value["reason"],
        )


@dataclass
class ActiveRound:
    round_id: int
    actions: list[RoundAction]
    admission_complete: bool = False
    tuning_complete: bool = False
    evaluations_before: int = 0
    deep_tune_selection: DeepTuneSelection | None = None
    deep_tune_outcome: DeepTuneOutcome | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActiveRound":
        actions = value.get("actions")
        if not isinstance(actions, list):
            raise ValueError("active round actions must be a list")
        raw_selection = value.get("deep_tune_selection")
        raw_outcome = value.get("deep_tune_outcome")
        selection = (
            DeepTuneSelection.from_dict(raw_selection)
            if raw_selection is not None
            else None
        )
        outcome = (
            DeepTuneOutcome.from_dict(raw_outcome)
            if raw_outcome is not None
            else None
        )
        if outcome is not None and selection is None:
            raise ValueError("deep-tune outcome requires a reserved selection")
        if (
            outcome is not None
            and outcome.tuned_run_id is not None
            and outcome.tuned_run_id != selection.run_id
        ):
            raise ValueError("deep-tune outcome does not match reserved selection")
        tuning_complete = bool(value.get("tuning_complete", False))
        if outcome is not None and not tuning_complete:
            raise ValueError("deep-tune outcome requires a completed tuning transition")
        if tuning_complete and selection is not None and outcome is None:
            raise ValueError("reserved deep-tune selection requires an outcome on completion")
        return cls(
            round_id=int(value["round_id"]),
            actions=[RoundAction.from_dict(item) for item in actions],
            admission_complete=bool(value.get("admission_complete", False)),
            tuning_complete=tuning_complete,
            evaluations_before=int(value.get("evaluations_before", 0)),
            deep_tune_selection=selection,
            deep_tune_outcome=outcome,
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
