"""Candidate materialization, bounded authoring, Phase A, and debug escalation."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactError,
    atomic_write_json,
    atomic_write_text,
    file_revision,
    json_revision,
    paths_revision,
)
from .debug import (
    DebugAllowanceExhausted,
    DebugPolicy,
    FailureEvidence,
    parse_debug_response,
)
from .llm import (
    AgentEditSpec,
    InferenceContractError,
    InferenceError,
    InferenceRequestError,
    ModelGateway,
)
from .models import DebugDecision, DebugVerdict, RoundAction, RunIdentity
from .prompts import (
    CANDIDATE_WRITER_SYSTEM,
    CODE_REPAIR_SYSTEM,
    CONTRACT_BUILDER_SYSTEM,
    DEBUG_SYSTEM,
    TUNING_VALUES_SYSTEM,
)
from .schemas import (
    DEBUG_SCHEMA,
    TUNING_VALUES_SCHEMA_VERSION,
    tuning_values_schema,
)
from .toolchain import (
    PreflightTimeout,
    ToolFailure,
    Toolchain,
    ValidationRejected,
    parse_json_output,
)
from .upstream import is_retryable_upstream_failure, last_error_is_upstream_transport


MAX_TUNING_SCHEMA_REPAIRS = 2
MAX_TUNING_VALUES_CORRECTIONS = 1
MAX_TUNING_VALUES_TRANSPORT_FAILURES = 2
MAX_IMPLEMENTATION_ATTEMPTS = 2
PHASE_A_WORKER_RECOVERY_BACKOFF_SECONDS = 0.25
MAX_PHASE_A_WORKER_TIMEOUTS = 2

# Exact key set of the schema-3 execution-revision object produced by
# tools/tuners/tune_tools.py::_candidate_execution_revision.  The warmstart
# worker stamps it on phase_a.candidate_code_revision and on every
# terminal_failure.candidate_execution_revision; the consumer must accept the
# full producer shape, not a projection of it.
EXECUTION_REVISION_KEYS = frozenset(
    {
        "schema_version",
        "structure_sha256",
        "search_space",
        "search_space_keys",
        "search_space_sha256",
        "prepare_sha256",
        "evaluation_contract",
        "revision_sha256",
    }
)


class CandidateBuildError(RuntimeError):
    pass


class SourceValidationRejected(RuntimeError):
    """A staged model-authored Python file failed the local source gate."""


@dataclass(frozen=True)
class CandidateOutcome:
    status: str
    best_score: float | None = None


@dataclass(frozen=True)
class TuningValues:
    warm_configs: list[dict[str, Any]]
    search_space: dict[str, list[Any]]

    @classmethod
    def from_response(
        cls,
        value: Any,
        *,
        k: int,
        expected_keys: set[str],
    ) -> "TuningValues":
        if not isinstance(value, dict):
            raise ValueError("tuning-values response must be an object")
        expected_top = {"schema_version", "warm_configs", "search_space"}
        if set(value) != expected_top:
            raise ValueError(
                f"tuning-values fields must be exactly {sorted(expected_top)}"
            )
        if value.get("schema_version") != TUNING_VALUES_SCHEMA_VERSION:
            raise ValueError(
                "tuning-values response has an unsupported schema version"
            )
        raw_configs = value.get("warm_configs")
        if not isinstance(raw_configs, list) or len(raw_configs) != k:
            raise ValueError(f"tuning-values response must contain exactly {k} configs")

        configs: list[dict[str, Any]] = []
        seen_configs: set[str] = set()
        config_keys: set[str] | None = None
        for index, raw_config in enumerate(raw_configs):
            if not isinstance(raw_config, list) or not raw_config:
                raise ValueError(f"warm_configs[{index}] must be a non-empty entry list")
            config: dict[str, Any] = {}
            for entry in raw_config:
                if not isinstance(entry, dict) or set(entry) != {"key", "value"}:
                    raise ValueError(
                        f"warm_configs[{index}] entries require only key and value"
                    )
                key = entry.get("key")
                item = entry.get("value")
                if not isinstance(key, str) or not key:
                    raise ValueError(f"warm_configs[{index}] has an invalid key")
                if key in config:
                    raise ValueError(
                        f"warm_configs[{index}] repeats parameter {key!r}"
                    )
                _validate_json_primitive(item, where=f"warm_configs[{index}].{key}")
                config[key] = item
            keys = set(config)
            if config_keys is None:
                config_keys = keys
            elif keys != config_keys:
                raise ValueError("every warm config must contain the same parameter keys")
            revision = json.dumps(
                config,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            if revision in seen_configs:
                raise ValueError("warm configs must be distinct")
            seen_configs.add(revision)
            configs.append(config)

        raw_space = value.get("search_space")
        if not isinstance(raw_space, list) or not raw_space:
            raise ValueError("search_space must be a non-empty entry list")
        space: dict[str, list[Any]] = {}
        required_entry = {"key", "kind", "low", "high", "log", "options"}
        for index, entry in enumerate(raw_space):
            if not isinstance(entry, dict) or set(entry) != required_entry:
                raise ValueError(
                    f"search_space[{index}] fields must be {sorted(required_entry)}"
                )
            key = entry.get("key")
            kind = entry.get("kind")
            low = entry.get("low")
            high = entry.get("high")
            logarithmic = entry.get("log")
            options = entry.get("options")
            if not isinstance(key, str) or not key or key in space:
                raise ValueError(f"search_space[{index}] has an invalid or duplicate key")
            if not isinstance(logarithmic, bool) or not isinstance(options, list):
                raise ValueError(f"search_space[{index}] has invalid control fields")
            if kind == "int":
                if (
                    type(low) is not int
                    or type(high) is not int
                    or low > high
                    or logarithmic
                    or options
                ):
                    raise ValueError(
                        f"search_space[{index}] int requires ordered integer bounds, "
                        "log=false, and empty options"
                    )
                space[key] = ["int", low, high]
            elif kind == "float":
                if (
                    not _finite_number(low)
                    or not _finite_number(high)
                    or float(low) > float(high)
                    or (logarithmic and float(low) <= 0.0)
                    or options
                ):
                    raise ValueError(
                        f"search_space[{index}] float requires ordered finite bounds "
                        "and positive bounds for log scale"
                    )
                space[key] = ["float", low, high]
                if logarithmic:
                    space[key].append("log")
            elif kind == "categorical":
                if low is not None or high is not None or logarithmic or not options:
                    raise ValueError(
                        f"search_space[{index}] categorical requires null bounds, "
                        "log=false, and non-empty options"
                    )
                seen_options: list[Any] = []
                for option in options:
                    _validate_json_primitive(
                        option, where=f"search_space[{index}].options"
                    )
                    if any(option == prior for prior in seen_options):
                        raise ValueError(
                            f"search_space[{index}] categorical options must be unique"
                        )
                    seen_options.append(option)
                space[key] = ["categorical", options]
            else:
                raise ValueError(f"search_space[{index}] has unsupported kind {kind!r}")

        if set(space) != (config_keys or set()):
            raise ValueError("search-space keys must match every warm configuration")
        if set(space) != expected_keys:
            raise ValueError("tuning-value keys must match validated PARAM_SCHEMA")
        return cls(warm_configs=configs, search_space=space)


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _validate_json_primitive(value: Any, *, where: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float and math.isfinite(value):
        return
    raise ValueError(f"{where} must be a finite primitive JSON value")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _is_failure_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 21
        and value.startswith("fail-")
        and all(character in "0123456789abcdef" for character in value[5:])
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc


def _validation_error_messages(exc: ValidationRejected) -> str:
    """Rejection messages from a typed validator receipt, unwrapped."""
    try:
        payload = parse_json_output(exc.result.output)
    except ValueError:
        return str(exc)
    errors = payload.get("errors") if isinstance(payload, dict) else None
    if (
        isinstance(errors, list)
        and errors
        and all(isinstance(error, str) for error in errors)
    ):
        return "; ".join(errors)
    return str(exc)


def _bounded_text(path: Path, limit: int = 80_000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def _bounded_diagnostic_text(path: Path, *, head: int = 3000, tail: int = 3000) -> str:
    """Inline a durable diagnostic for a tool-less Messages correction call.

    The file remains the durable receipt; the prompt carries the same bounded
    head/tail shape as ToolFailure's rendered validator output because the
    correction model cannot read paths.
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n...[bounded diagnostic omitted]...\n" + text[-tail:]


class CandidatePipeline:
    IMPLEMENTATION_DRAFT = "_implementation_draft.py"
    IMPLEMENTATION_STATE = "_implementation_authoring.json"
    IMPLEMENTATION_RECEIPT = "_implementation_ready.json"
    TUNING_SCHEMA_DRAFT = "_tuning_schema_draft.py"
    TUNING_SCHEMA_STATE = "_tuning_schema_authoring.json"
    TUNING_SCHEMA_RECEIPT = "_tuning_schema_ready.json"
    TUNING_VALUES_STATE = "_tuning_values_authoring.json"
    TUNING_LINEAGE_EVIDENCE = "_tuning_lineage_evidence.json"
    TUNING_CONTRACT_DRAFT = "_tuning_contract_draft.py"
    TUNING_VALUES_RECEIPT = "_tuning_values_ready.json"
    CONTRACT_RECEIPT = "_contract_ready.json"
    PREFLIGHT_RECEIPT = "_preflight_ready.json"
    # Bounded retry allowance for side-effect-free preflight timeouts (the
    # initial attempt plus this many retries).
    PREFLIGHT_TIMEOUT_RETRIES = 1
    PHASE_A_RETRY_RECEIPT = "_phase_a_retry.json"
    PHASE_A_REPAIR_RECEIPT = "_phase_a_repair.json"
    DEBUG_CORRECTED_BASE = "_debug_corrected_base.json"

    def __init__(
        self,
        identity: RunIdentity,
        toolchain: Toolchain,
        models: ModelGateway,
        task_config: dict[str, Any],
    ):
        self.identity = identity
        self.toolchain = toolchain
        self.models = models
        self.task_config = task_config
        self.debug_policy = DebugPolicy(identity.run_dir)

    def candidate_dir(self, run_id: str) -> Path:
        return self.identity.run_dir / "candidates" / run_id

    def _candidate_authoring_context_paths(
        self,
        action: RoundAction,
    ) -> list[Path]:
        """Return frozen semantic/task inputs, excluding operational run state."""
        if action.run_id is None:
            raise ValueError("candidate authoring requires a run_id")
        candidate_dir = self.candidate_dir(action.run_id)
        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        paths = [
            candidate_dir / "_candidate_brief.json",
            candidate_dir / "prepare.py",
            task_dir / "TASK.md",
            task_dir / "task.toml",
            self.identity.run_dir / "framework_cfg.json",
        ]
        try:
            brief = _read_json(candidate_dir / "_candidate_brief.json")
        except ValueError:
            brief = None
        source = brief.get("implementation_source") if isinstance(brief, dict) else None
        source_path = source.get("path") if isinstance(source, dict) else None
        if isinstance(source, dict) and source.get("kind") == "provided_entrypoint":
            if not isinstance(source_path, str) or not source_path:
                raise ArtifactError("provided candidate brief lacks its frozen source path")
            provided_path = (self.identity.repo_root / source_path).resolve()
            try:
                provided_path.relative_to(task_dir.resolve())
            except ValueError as exc:
                raise ArtifactError(
                    "provided candidate source escapes its task directory"
                ) from exc
            paths.append(provided_path)
        for parent in action.parents:
            paths.extend(
                (
                    self.candidate_dir(parent) / "train.py",
                    self.candidate_dir(parent) / "tune_report.json",
                )
            )
        return paths

    def is_provided_baseline(self, run_id: str) -> bool:
        brief = _read_json(self.candidate_dir(run_id) / "_candidate_brief.json")
        source = brief.get("implementation_source") if isinstance(brief, dict) else None
        return isinstance(source, dict) and source.get("kind") == "provided_entrypoint"

    def materialize(self, action: RoundAction, *, provided_baseline: bool = False) -> Path:
        if action.run_id is None:
            raise ValueError("cannot materialize an action without run_id")
        candidate_dir = self.candidate_dir(action.run_id)
        if candidate_dir.exists():
            record = self.toolchain.ledger_record(
                self.identity.run_dir,
                action.run_id,
            )
            if not isinstance(record, dict) or not self.materialization_is_ready(
                action, record
            ):
                raise ArtifactError(
                    f"candidate directory exists without a valid materialization: {candidate_dir}"
                )
            return candidate_dir
        return self.toolchain.materialize_candidate(
            self.identity.task_name,
            self.identity.tag,
            action.run_id,
            provided_baseline=provided_baseline,
        )

    def materialization_is_ready(
        self,
        action: RoundAction,
        record: dict[str, Any] | None = None,
    ) -> bool:
        if action.run_id is None:
            return False
        candidate_dir = self.candidate_dir(action.run_id)
        try:
            brief = _read_json(candidate_dir / "_candidate_brief.json")
        except ValueError:
            return False
        if not isinstance(brief, dict) or (
            brief.get("schema_version") != 4
            or brief.get("run_id") != action.run_id
            or brief.get("op") != action.op
            or brief.get("source_run_ids") != action.parents
        ):
            return False
        if record is not None:
            for key in (
                "idea",
                "change",
                "source_run_ids",
                "op",
                "semantic_point",
                "policy_receipt",
            ):
                if brief.get(key) != record.get(key):
                    return False
        candidate_config = self.task_config.get("candidate", {})
        if not isinstance(candidate_config, dict):
            return False
        entrypoint = candidate_config.get("entrypoint", "train.py")
        copy_files = candidate_config.get("copy_files", ["prepare.py", "train.py"])
        if not isinstance(entrypoint, str) or not isinstance(copy_files, list):
            return False
        source = brief.get("implementation_source")
        source_kind = source.get("kind") if isinstance(source, dict) else None
        required = [path for path in copy_files if path != entrypoint]
        if source_kind != "generated":
            required.append(entrypoint)
        if not all(
            isinstance(relative, str) and (candidate_dir / relative).is_file()
            for relative in required
        ):
            return False
        if source_kind == "provided_entrypoint":
            return self._provided_entrypoint_is_current(
                candidate_dir,
                entrypoint=entrypoint,
                source=source,
            )
        return True

    def implement(self, action: RoundAction) -> Path:
        if action.run_id is None:
            raise ValueError("cannot implement an action without run_id")
        candidate_dir = self.candidate_dir(action.run_id)
        candidate_path = candidate_dir / "train.py"
        if self.implementation_is_ready(action):
            return candidate_path
        if self.is_provided_baseline(action.run_id):
            if not self.materialization_is_ready(action):
                raise ArtifactError(
                    "provided baseline candidate does not match its task seed receipt"
                )
            self._validate_python(candidate_path)
            self._write_implementation_receipt(
                action.run_id,
                candidate_path,
                "provided",
                input_revision=paths_revision(
                    self._candidate_authoring_context_paths(action)
                ),
            )
            return candidate_path

        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        brief_path = candidate_dir / "_candidate_brief.json"
        read_roots = [task_dir, candidate_dir]
        context_paths = self._candidate_authoring_context_paths(action)
        for parent in action.parents:
            parent_path = self.candidate_dir(parent) / "train.py"
            read_roots.append(parent_path)
        for purpose in (
            f"candidate_writer:{action.run_id}:repair",
            f"candidate_writer:{action.run_id}",
        ):
            if self._completed_edit_matches(
                purpose=purpose,
                output_path=candidate_path,
                immutable_input_paths=tuple(context_paths),
            ):
                self._validate_python(candidate_path)
                self._write_implementation_receipt(
                    action.run_id,
                    candidate_path,
                    "model_receipt",
                    input_revision=paths_revision(context_paths),
                )
                return candidate_path

        draft_path = candidate_dir / self.IMPLEMENTATION_DRAFT
        state_path = candidate_dir / self.IMPLEMENTATION_STATE
        context_revision = paths_revision(context_paths)
        source_revision = paths_revision((candidate_path,))
        state = self._read_implementation_state(
            state_path,
            run_id=action.run_id,
            context_revision=context_revision,
            source_revision=source_revision,
            candidate_path=candidate_path,
            draft_path=draft_path,
        )
        if state is None:
            atomic_write_text(
                draft_path,
                (
                    candidate_path.read_text(encoding="utf-8")
                    if candidate_path.is_file()
                    else ""
                ),
            )
            state = {
                "schema_version": 1,
                "kind": "candidate_implementation_authoring",
                "run_id": action.run_id,
                "status": "ready",
                "context_revision": context_revision,
                "source_candidate_revision": source_revision,
                "draft_revision": file_revision(draft_path),
                "attempts_admitted": 0,
            }
            atomic_write_json(state_path, state)
        elif state["status"] == "publishing":
            return self._publish_implementation(
                action.run_id,
                candidate_path=candidate_path,
                draft_path=draft_path,
                state_path=state_path,
                state=state,
            )
        elif state["status"] == "completed":
            raise ArtifactError(
                "completed implementation state has no matching implementation receipt"
            )

        diagnostic_path = (
            self.identity.run_dir
            / ".orchestrator"
            / "candidate_contract_diagnostics"
            / f"{action.run_id}-implementation.json"
        )
        last_error: BaseException | None = None
        retry_purpose: str | None = None
        repair_needed = False
        if state["status"] == "rejected":
            last_error = CandidateBuildError(str(state.get("last_error", "")))
            repair_needed = True
            if not diagnostic_path.is_file():
                raise ArtifactError(
                    "failed implementation state has no diagnostic artifact"
                )
        elif state["status"] == "infrastructure_failed":
            if not diagnostic_path.is_file():
                raise ArtifactError(
                    "failed implementation state has no diagnostic artifact"
                )
            # Coordinator outer backoff re-enters here after local transport
            # exhaustion. Re-open one local window when the cause was upstream
            # so the next attempt actually reaches the provider again.
            if (
                int(state["attempts_admitted"]) >= MAX_IMPLEMENTATION_ATTEMPTS
                and last_error_is_upstream_transport(state.get("last_error"))
            ):
                state = {
                    **state,
                    "attempts_admitted": 0,
                    "status": "infrastructure_failed",
                }
                atomic_write_json(state_path, state)
            prior_index = max(int(state["attempts_admitted"]) - 1, 0)
            retry_purpose = state.get("purpose")
            if not isinstance(retry_purpose, str) or not retry_purpose:
                retry_purpose = f"candidate_writer:{action.run_id}" + (
                    ":repair" if prior_index else ""
                )
            if retry_purpose.endswith(":repair"):
                last_error = CandidateBuildError(
                    str(state.get("last_error", ""))
                )
            atomic_write_text(
                draft_path,
                (
                    candidate_path.read_text(encoding="utf-8")
                    if candidate_path.is_file()
                    else ""
                ),
            )
        elif not diagnostic_path.is_file():
            diagnostic_path = self._write_contract_diagnostic(
                action.run_id,
                stage="implementation",
                error=CandidateBuildError(
                    "candidate implementation requires a completed bounded edit"
                ),
            )

        admitted = int(state["attempts_admitted"])
        if state["status"] == "started":
            prior_index = admitted - 1
            prior_purpose = state.get("purpose")
            if not isinstance(prior_purpose, str) or not prior_purpose:
                prior_purpose = f"candidate_writer:{action.run_id}" + (
                    ":repair" if prior_index else ""
                )
            immutable_paths = tuple(
                [*context_paths, candidate_path, diagnostic_path]
            )
            if self._completed_edit_matches(
                purpose=prior_purpose,
                output_path=draft_path,
                immutable_input_paths=immutable_paths,
            ):
                state = {
                    **state,
                    "status": "publishing",
                    "expected_candidate_revision": file_revision(draft_path),
                }
                atomic_write_json(state_path, state)
                return self._publish_implementation(
                    action.run_id,
                    candidate_path=candidate_path,
                    draft_path=draft_path,
                    state_path=state_path,
                    state=state,
                )
            atomic_write_text(
                draft_path,
                (
                    candidate_path.read_text(encoding="utf-8")
                    if candidate_path.is_file()
                    else ""
                ),
            )
            retry_purpose = prior_purpose
            if retry_purpose.endswith(":repair"):
                last_error = CandidateBuildError(
                    str(state.get("last_error", "the prior repair did not complete"))
                )

        validate_command = self.toolchain.candidate_source_validate_command(draft_path)
        prompt = (
            f"Implement candidate {action.run_id} in the staged Python file "
            f"{draft_path}. The frozen canonical path is {candidate_path}.\n"
            f"Operation: {action.op}; numeric parents: {action.parents or 'none'}.\n"
            "Read _candidate_brief.json first. The only authorized write is the "
            "staged Python file; Python publishes it after validation. Before "
            "finishing, run this exact source validation with the Bash tool and "
            f"fix the staged file until it exits 0: {validate_command}"
        )
        while admitted < MAX_IMPLEMENTATION_ATTEMPTS:
            if retry_purpose is not None:
                purpose = retry_purpose
                retry_purpose = None
            else:
                purpose = f"candidate_writer:{action.run_id}" + (
                    ":repair" if repair_needed else ""
                )
            attempt_prompt = prompt
            if purpose.endswith(":repair") and last_error is not None:
                attempt_prompt += (
                    "\nThe prior output failed the deterministic Python syntax/postcondition "
                    f"check. Repair it once using {diagnostic_path}. Error: {last_error}"
                )
            input_paths = [
                *context_paths,
                candidate_path,
                draft_path,
                diagnostic_path,
            ]
            admitted += 1
            state = {
                **state,
                "status": "started",
                "draft_revision": file_revision(draft_path),
                "attempts_admitted": admitted,
                "purpose": purpose,
                "last_error": str(last_error or ""),
            }
            atomic_write_json(state_path, state)
            try:
                self.models.edit(
                    AgentEditSpec(
                        purpose=purpose,
                        schema_version=1,
                        cwd=self.identity.repo_root,
                        system_prompt=CANDIDATE_WRITER_SYSTEM,
                        prompt=attempt_prompt,
                        tools=("Read", "Glob", "Grep", "Write", "Edit", "Bash"),
                        read_roots=tuple([*read_roots, diagnostic_path]),
                        write_paths=(draft_path,),
                        allowed_commands=frozenset({validate_command}),
                        input_paths=tuple(input_paths),
                        immutable_input_paths=tuple(
                            path for path in input_paths if path != draft_path
                        ),
                        max_turns=24,
                    ),
                    validate=lambda: self._validate_authored_python(draft_path),
                )
                state = {
                    **state,
                    "status": "publishing",
                    "expected_candidate_revision": file_revision(draft_path),
                }
                atomic_write_json(state_path, state)
                return self._publish_implementation(
                    action.run_id,
                    candidate_path=candidate_path,
                    draft_path=draft_path,
                    state_path=state_path,
                    state=state,
                )
            except SourceValidationRejected as exc:
                last_error = exc
                repair_needed = True
                diagnostic_path = self._write_contract_diagnostic(
                    action.run_id,
                    stage="implementation",
                    error=exc,
                )
                state = {
                    **state,
                    "status": "rejected",
                    "draft_revision": file_revision(draft_path),
                    "last_error_type": type(exc).__name__,
                    "last_error": str(exc),
                }
                atomic_write_json(state_path, state)
            except InferenceError as exc:
                diagnostic_path = self._write_contract_diagnostic(
                    action.run_id,
                    stage="implementation",
                    error=exc,
                )
                if is_retryable_upstream_failure(exc):
                    # Transient provider fault: the coordinator's upstream
                    # backoff owns the retry and may reopen this local window.
                    atomic_write_json(
                        state_path,
                        {
                            **state,
                            "status": "infrastructure_failed",
                            "last_error_type": type(exc).__name__,
                            "last_error": str(exc),
                        },
                    )
                    raise
                # Non-upstream inference failure (max turns, SDK error result,
                # request/contract rejection): the admitted attempt is consumed
                # and the loop continues like a rejected authoring attempt.
                last_error = exc
                repair_needed = True
                state = {
                    **state,
                    "status": "rejected",
                    "draft_revision": file_revision(draft_path),
                    "last_error_type": type(exc).__name__,
                    "last_error": str(exc),
                }
                atomic_write_json(state_path, state)
        if state.get("status") == "infrastructure_failed":
            raise InferenceError(
                "candidate writer transport retry limit reached: "
                + str(state.get("last_error", "unknown transport failure"))
            )
        raise CandidateBuildError(
            f"candidate writer failed for {action.run_id}: {last_error}"
        )

    def build_contract(self, action: RoundAction) -> None:
        if action.run_id is None:
            raise ValueError("cannot build a contract without run_id")
        candidate_dir = self.candidate_dir(action.run_id)
        candidate_path = candidate_dir / "train.py"
        configs_path = candidate_dir / "_warm_configs.json"
        space_path = candidate_dir / "_search_space.json"
        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        provided = self.is_provided_baseline(action.run_id)
        k = 1 if provided else self._tuner_integer("K", 5, minimum=1)
        values_state_path = candidate_dir / self.TUNING_VALUES_STATE
        values_progress = self._tuning_values_receipt(action)
        if values_progress is not None:
            self._forward_complete_tuning_values(
                action,
                receipt=values_progress,
                candidate_path=candidate_path,
                configs_path=configs_path,
                space_path=space_path,
            )
            return
        if values_state_path.exists():
            schema_ready = self._tuning_schema_receipt(action)
            schema_lint = (
                schema_ready["lint"]
                if schema_ready is not None
                else self._ensure_tuning_schema(
                    action,
                    candidate_path=candidate_path,
                    task_dir=task_dir,
                    provided=provided,
                )
            )
            self._propose_tuning_values(
                action,
                candidate_path=candidate_path,
                configs_path=configs_path,
                space_path=space_path,
                task_dir=task_dir,
                k=k,
                provided=provided,
                schema_lint=schema_lint,
            )
            return
        if self.contract_is_ready(action):
            return
        if provided and configs_path.is_file():
            atomic_write_json(
                configs_path,
                [self._provided_defaults(action.run_id, candidate_path)],
            )
        schema_lint: dict[str, Any] | None = None
        if candidate_path.is_file() and configs_path.is_file() and space_path.is_file():
            try:
                full_lint = self.toolchain.lint_contract(
                    candidate_path,
                    require_base_params=False,
                )
            except ValidationRejected:
                pass
            else:
                schema_lint = {
                    "ok": True,
                    "keys": full_lint.get("keys", []),
                    "kinds": {},
                    "float_log": {},
                    "make_model_defined": full_lint.get(
                        "make_model_defined", False
                    ),
                    "make_model_called": full_lint.get(
                        "make_model_called", False
                    ),
                    "errors": [],
                    "validated_as": "full_contract",
                }
                self._write_tuning_schema_receipt(
                    action,
                    candidate_path,
                    schema_lint,
                    source_candidate_revision=file_revision(candidate_path),
                )
                try:
                    self._recover_authored_contract(
                        action,
                        candidate_path=candidate_path,
                        configs_path=configs_path,
                        space_path=space_path,
                        k=k,
                    )
                except (ValidationRejected, ValueError, SyntaxError):
                    pass
                else:
                    return
        if schema_lint is None:
            schema_lint = self._ensure_tuning_schema(
                action,
                candidate_path=candidate_path,
                task_dir=task_dir,
                provided=provided,
            )
        if candidate_path.is_file() and configs_path.is_file() and space_path.is_file():
            try:
                self._recover_authored_contract(
                    action,
                    candidate_path=candidate_path,
                    configs_path=configs_path,
                    space_path=space_path,
                    k=k,
                )
                return
            except (ValidationRejected, ValueError, SyntaxError):
                pass
        self._propose_tuning_values(
            action,
            candidate_path=candidate_path,
            configs_path=configs_path,
            space_path=space_path,
            task_dir=task_dir,
            k=k,
            provided=provided,
            schema_lint=schema_lint,
        )

    def _ensure_tuning_schema(
        self,
        action: RoundAction,
        *,
        candidate_path: Path,
        task_dir: Path,
        provided: bool,
    ) -> dict[str, Any]:
        if action.run_id is None:  # pragma: no cover - checked by caller
            raise ValueError("cannot build a schema without run_id")
        ready = self._tuning_schema_receipt(action)
        if ready is not None:
            return ready["lint"]

        candidate_dir = candidate_path.parent
        draft_path = candidate_dir / self.TUNING_SCHEMA_DRAFT
        state_path = candidate_dir / self.TUNING_SCHEMA_STATE
        source_revision = file_revision(candidate_path)
        context_paths = self._candidate_authoring_context_paths(action)
        context_revision = paths_revision(context_paths)
        if state_path.exists():
            active_state = self._read_tuning_schema_state(
                state_path,
                run_id=action.run_id,
                context_revision=context_revision,
                source_revision=source_revision,
                candidate_path=candidate_path,
                draft_path=draft_path,
            )
            assert active_state is not None
            if active_state["status"] == "publishing":
                return self._publish_tuning_schema(
                    action,
                    candidate_path=candidate_path,
                    draft_path=draft_path,
                    state_path=state_path,
                    state=active_state,
                )
            if active_state["status"] == "completed":
                raise ArtifactError(
                    "completed tuning-schema state has no matching schema receipt"
                )

        existing_error: BaseException | None = None
        try:
            lint = self._validate_tuning_schema(candidate_path)
        except (ValidationRejected, ValueError, SyntaxError) as exc:
            existing_error = exc
            if provided:
                raise CandidateBuildError(
                    f"provided tuning schema failed for {action.run_id}: "
                    f"{exc}"
                ) from exc
        else:
            self._write_tuning_schema_receipt(
                action,
                candidate_path,
                lint,
                source_candidate_revision=file_revision(candidate_path),
            )
            return lint

        state = self._read_tuning_schema_state(
            state_path,
            run_id=action.run_id,
            context_revision=context_revision,
            source_revision=source_revision,
            candidate_path=candidate_path,
            draft_path=draft_path,
        )
        retry_purpose: str | None = None
        correction_needed = False
        if state is None:
            atomic_write_text(draft_path, candidate_path.read_text(encoding="utf-8"))
            state = {
                "schema_version": 1,
                "kind": "tuning_schema_authoring",
                "run_id": action.run_id,
                "status": "ready",
                "context_revision": context_revision,
                "source_candidate_revision": source_revision,
                "draft_revision": file_revision(draft_path),
                "attempts_admitted": 0,
            }
            atomic_write_json(state_path, state)
            last_error: BaseException | None = existing_error
        else:
            status = state["status"]
            if status == "publishing":
                return self._publish_tuning_schema(
                    action,
                    candidate_path=candidate_path,
                    draft_path=draft_path,
                    state_path=state_path,
                    state=state,
                )
            if status == "completed":
                raise ArtifactError(
                    "completed tuning-schema state has no matching schema receipt"
                )
            if status in {"validating", "validator_failed"}:
                try:
                    return self._validate_and_publish_tuning_schema(
                        action,
                        candidate_path=candidate_path,
                        draft_path=draft_path,
                        state_path=state_path,
                        state=state,
                    )
                except ValidationRejected as exc:
                    state = _read_json(state_path)
                    last_error = exc
                    correction_needed = True
            elif status == "infrastructure_failed":
                if (
                    int(state["attempts_admitted"]) >= MAX_TUNING_SCHEMA_REPAIRS + 1
                    and last_error_is_upstream_transport(state.get("last_error"))
                ):
                    state = {
                        **state,
                        "attempts_admitted": 0,
                        "status": "infrastructure_failed",
                    }
                    atomic_write_json(state_path, state)
                prior_index = max(int(state["attempts_admitted"]) - 1, 0)
                retry_purpose = state.get("purpose")
                if not isinstance(retry_purpose, str) or not retry_purpose:
                    retry_purpose = (
                        f"tuning_schema:{action.run_id}"
                        if prior_index == 0
                        else f"tuning_schema:{action.run_id}:repair:{prior_index}"
                    )
                atomic_write_text(
                    draft_path,
                    candidate_path.read_text(encoding="utf-8"),
                )
                last_error = CandidateBuildError(
                    str(state.get("last_error", existing_error or ""))
                )
            else:
                last_error = CandidateBuildError(
                    str(state.get("last_error", existing_error or ""))
                )
                correction_needed = status == "rejected"

        diagnostic_path = (
            self.identity.run_dir
            / ".orchestrator"
            / "candidate_contract_diagnostics"
            / f"{action.run_id}-schema.json"
        )
        if state["status"] in {"ready", "rejected"}:
            diagnostic = state.get("diagnostic")
            if isinstance(diagnostic, dict):
                self._write_contract_diagnostic(
                    action.run_id,
                    stage="schema",
                    payload=diagnostic,
                )
            else:
                self._write_contract_diagnostic(
                    action.run_id,
                    stage="schema",
                    error=last_error,
                )
        elif not diagnostic_path.is_file():
            raise ArtifactError("schema authoring state has no diagnostic artifact")

        read_roots = [task_dir, candidate_dir]
        fixed_input_paths = [candidate_path, draft_path, *context_paths]
        for parent in action.parents:
            read_roots.append(self.candidate_dir(parent) / "train.py")
        lint_command = self.toolchain.lint_schema_command(draft_path)
        base_prompt = (
            f"Prepare the code-side tuning schema for candidate {action.run_id} "
            f"in the staged Python file {draft_path}. The frozen source is "
            f"{candidate_path}.\nOperation: {action.op}; "
            f"parents: {action.parents or 'none'}.\n"
            "Read the candidate brief, task evaluation contract, and readonly "
            "prepare.py. Edit only the staged Python file; Python publishes it "
            "only after validation. The exact PARAM_SCHEMA literal grammar in "
            "the system instructions is a hard AST contract. Before finishing, "
            "run this exact schema lint with the Bash tool and fix the staged "
            f"file until it exits 0: {lint_command}"
        )
        admitted = int(state["attempts_admitted"])
        if state["status"] == "started":
            prior_index = admitted - 1
            prior_purpose = state.get("purpose")
            if not isinstance(prior_purpose, str) or not prior_purpose:
                prior_purpose = (
                    f"tuning_schema:{action.run_id}"
                    if prior_index == 0
                    else f"tuning_schema:{action.run_id}:repair:{prior_index}"
                )
            immutable_paths = tuple(
                [
                    *(path for path in fixed_input_paths if path != draft_path),
                    diagnostic_path,
                ]
            )
            if self._completed_edit_matches(
                purpose=prior_purpose,
                output_path=draft_path,
                immutable_input_paths=immutable_paths,
            ):
                state = {
                    **state,
                    "status": "validating",
                    "expected_candidate_revision": file_revision(draft_path),
                }
                atomic_write_json(state_path, state)
                try:
                    return self._validate_and_publish_tuning_schema(
                        action,
                        candidate_path=candidate_path,
                        draft_path=draft_path,
                        state_path=state_path,
                        state=state,
                    )
                except ValidationRejected as exc:
                    state = _read_json(state_path)
                    last_error = exc
                    admitted = int(state["attempts_admitted"])
                    correction_needed = True
            else:
                atomic_write_text(
                    draft_path,
                    candidate_path.read_text(encoding="utf-8"),
                )
                retry_purpose = prior_purpose
                if ":repair:" in retry_purpose:
                    last_error = CandidateBuildError(
                        str(state.get("last_error", "the prior repair did not complete"))
                    )

        while admitted < MAX_TUNING_SCHEMA_REPAIRS + 1:
            if retry_purpose is not None:
                purpose = retry_purpose
                retry_purpose = None
            elif correction_needed:
                purpose = f"tuning_schema:{action.run_id}:repair:{admitted}"
            else:
                purpose = f"tuning_schema:{action.run_id}"
            prompt = base_prompt + (
                "\n\nThe staged source failed its deterministic gate. "
                f"Read the complete Python-owned diagnostic at {diagnostic_path} "
                "and repair every reported schema error without changing the "
                "candidate strategy."
            )
            input_paths = [*fixed_input_paths, diagnostic_path]
            current_read_roots = [*read_roots, diagnostic_path]
            admitted += 1
            state = {
                **state,
                "status": "started",
                "draft_revision": file_revision(draft_path),
                "attempts_admitted": admitted,
                "purpose": purpose,
                "last_error": str(last_error or ""),
            }
            atomic_write_json(state_path, state)
            try:
                self.models.edit(
                    AgentEditSpec(
                        purpose=purpose,
                        schema_version=1,
                        cwd=self.identity.repo_root,
                        system_prompt=CONTRACT_BUILDER_SYSTEM,
                        prompt=prompt,
                        tools=("Read", "Glob", "Grep", "Write", "Edit", "Bash"),
                        read_roots=tuple(current_read_roots),
                        write_paths=(draft_path,),
                        allowed_commands=frozenset({lint_command}),
                        input_paths=tuple(input_paths),
                        immutable_input_paths=tuple(
                            path for path in input_paths if path != draft_path
                        ),
                        max_turns=24,
                    ),
                    validate=lambda: self._validate_authored_python(draft_path),
                )
            except SourceValidationRejected as exc:
                last_error = exc
                correction_needed = True
                diagnostic = self._contract_diagnostic_payload(
                    action.run_id,
                    stage="schema",
                    error=exc,
                )
                state = {
                    **state,
                    "status": "rejected",
                    "draft_revision": (
                        file_revision(draft_path) if draft_path.is_file() else None
                    ),
                    "last_error_type": type(exc).__name__,
                    "last_error": str(exc),
                    "diagnostic": diagnostic,
                }
                atomic_write_json(state_path, state)
                self._write_contract_diagnostic(
                    action.run_id,
                    stage="schema",
                    payload=diagnostic,
                )
                continue
            except InferenceError as exc:
                if is_retryable_upstream_failure(exc):
                    # Transient provider fault: the coordinator's upstream
                    # backoff owns the retry and may reopen this local window.
                    state = {
                        **state,
                        "status": "infrastructure_failed",
                        "last_error_type": type(exc).__name__,
                        "last_error": str(exc),
                    }
                    atomic_write_json(state_path, state)
                    raise
                # Non-upstream inference failure: consume the admitted attempt
                # and continue as a correction with a truthful diagnostic.
                last_error = exc
                correction_needed = True
                diagnostic = self._contract_diagnostic_payload(
                    action.run_id,
                    stage="schema",
                    error=exc,
                )
                state = {
                    **state,
                    "status": "rejected",
                    "draft_revision": (
                        file_revision(draft_path) if draft_path.is_file() else None
                    ),
                    "last_error_type": type(exc).__name__,
                    "last_error": str(exc),
                    "diagnostic": diagnostic,
                }
                atomic_write_json(state_path, state)
                self._write_contract_diagnostic(
                    action.run_id,
                    stage="schema",
                    payload=diagnostic,
                )
                continue

            state = {
                **state,
                "status": "validating",
                "expected_candidate_revision": file_revision(draft_path),
            }
            atomic_write_json(state_path, state)
            try:
                return self._validate_and_publish_tuning_schema(
                    action,
                    candidate_path=candidate_path,
                    draft_path=draft_path,
                    state_path=state_path,
                    state=state,
                )
            except ValidationRejected as exc:
                state = _read_json(state_path)
                last_error = exc
        if state.get("status") == "infrastructure_failed":
            raise InferenceError(
                "tuning-schema transport retry limit reached: "
                + str(state.get("last_error", "unknown transport failure"))
            )
        raise CandidateBuildError(
            f"tuning schema failed for {action.run_id}: {last_error}"
        )

    def _frozen_tuning_lineage(
        self,
        action: RoundAction,
        *,
        base_input_paths: list[Path],
        state_exists: bool,
    ) -> tuple[dict[str, Any], Path]:
        """Persist the semantic lineage once so admitted requests replay exactly."""
        if action.run_id is None:  # pragma: no cover - checked by caller
            raise ValueError("cannot freeze tuning lineage without run_id")
        path = self.candidate_dir(action.run_id) / self.TUNING_LINEAGE_EVIDENCE
        base_input_revision = paths_revision(base_input_paths)
        if path.exists():
            payload = _read_json(path)
            evidence = payload.get("evidence") if isinstance(payload, dict) else None
            if not (
                isinstance(payload, dict)
                and payload.get("schema_version") == 1
                and payload.get("kind") == "tuning_lineage_evidence"
                and payload.get("run_id") == action.run_id
                and payload.get("source_run_ids") == action.parents
                and payload.get("base_input_revision") == base_input_revision
                and isinstance(evidence, dict)
            ):
                raise ArtifactError("frozen tuning lineage is stale or malformed")
            return evidence, path
        if state_exists:
            raise ArtifactError("tuning-values state lost its frozen lineage evidence")
        evidence = self.toolchain.lineage_evidence(
            self.identity.run_dir,
            action.parents,
        )
        if not isinstance(evidence, dict):
            raise ArtifactError("lineage evidence must be an object")
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "kind": "tuning_lineage_evidence",
                "run_id": action.run_id,
                "source_run_ids": action.parents,
                "base_input_revision": base_input_revision,
                "evidence": evidence,
            },
        )
        return evidence, path

    def _propose_tuning_values(
        self,
        action: RoundAction,
        *,
        candidate_path: Path,
        configs_path: Path,
        space_path: Path,
        task_dir: Path,
        k: int,
        provided: bool,
        schema_lint: dict[str, Any],
    ) -> None:
        if action.run_id is None:  # pragma: no cover - checked by caller
            raise ValueError("cannot propose tuning values without run_id")
        candidate_dir = candidate_path.parent
        raw_schema_keys = schema_lint.get("keys")
        if (
            not isinstance(raw_schema_keys, list)
            or not raw_schema_keys
            or not all(isinstance(key, str) and key for key in raw_schema_keys)
            or len(set(raw_schema_keys)) != len(raw_schema_keys)
        ):
            raise ArtifactError("validated PARAM_SCHEMA receipt has invalid keys")
        schema_keys = set(raw_schema_keys)
        schema_receipt_path = candidate_dir / self.TUNING_SCHEMA_RECEIPT
        lineage_base_paths = [
            candidate_path,
            schema_receipt_path,
            *self._candidate_authoring_context_paths(action),
        ]
        values_state_path = candidate_dir / self.TUNING_VALUES_STATE
        lineage, lineage_path = self._frozen_tuning_lineage(
            action,
            base_input_paths=lineage_base_paths,
            state_exists=values_state_path.exists(),
        )
        fixed_input_paths = [*lineage_base_paths, lineage_path]
        values_input_revision = paths_revision(fixed_input_paths)
        values_state = self._read_tuning_values_state(
            values_state_path,
            run_id=action.run_id,
            k=k,
            input_revision=values_input_revision,
        )
        if values_state is None:
            values_state = {
                "schema_version": 1,
                "kind": "tuning_values_authoring",
                "run_id": action.run_id,
                "status": "ready",
                "k": k,
                "input_revision": values_input_revision,
                "attempts_admitted": 0,
                "transport_failures": 0,
            }
            atomic_write_json(values_state_path, values_state)
        elif values_state["status"] in {"draft_validated", "validated"}:
            configs_draft, space_draft = self._tuning_value_proposal_paths(
                candidate_dir,
                values_state.get("proposal_index"),
            )
            validated_space = self._validated_tuning_space_path(
                candidate_dir,
                int(values_state["proposal_index"]),
            )
            self._publish_tuning_value_drafts(
                state_path=values_state_path,
                state=values_state,
                configs_draft=configs_draft,
                space_draft=space_draft,
                validated_space=validated_space,
                configs_path=configs_path,
                space_path=space_path,
            )
            self._finalize_validated_tuning_values(
                action,
                state_path=values_state_path,
                candidate_path=candidate_path,
                configs_path=configs_path,
                space_path=space_path,
            )
            return
        elif values_state["status"] == "finalizing":
            self._finalize_validated_tuning_values(
                action,
                state_path=values_state_path,
                candidate_path=candidate_path,
                configs_path=configs_path,
                space_path=space_path,
            )
            return
        elif values_state["status"] == "completed":
            if self.contract_is_ready(action):
                return
            raise ArtifactError(
                "completed tuning-values state has no matching contract receipt"
            )
        elif values_state["status"] in {"validating", "validator_failed"}:
            configs_draft, space_draft = self._tuning_value_proposal_paths(
                candidate_dir,
                values_state.get("proposal_index"),
            )
            try:
                values_state = self._validate_tuning_value_proposal(
                    action,
                    candidate_path=candidate_path,
                    state_path=values_state_path,
                    state=values_state,
                    configs_draft=configs_draft,
                    space_draft=space_draft,
                )
            except ValidationRejected:
                values_state = _read_json(values_state_path)
            else:
                validated_space = self._validated_tuning_space_path(
                    candidate_dir,
                    int(values_state["proposal_index"]),
                )
                self._publish_tuning_value_drafts(
                    state_path=values_state_path,
                    state=values_state,
                    configs_draft=configs_draft,
                    space_draft=space_draft,
                    validated_space=validated_space,
                    configs_path=configs_path,
                    space_path=space_path,
                )
                self._finalize_validated_tuning_values(
                    action,
                    state_path=values_state_path,
                    candidate_path=candidate_path,
                    configs_path=configs_path,
                    space_path=space_path,
                )
                return
        provided_defaults = None
        if provided:
            provided_defaults = self._provided_defaults(
                action.run_id,
                candidate_path,
            )
        if provided_defaults is not None:
            if values_state["status"] != "ready":
                raise CandidateBuildError(
                    "deterministic provided tuning values were rejected"
                )
            proposal = self._provided_tuning_values(
                provided_defaults,
                schema_lint,
            )
            proposal_index = 1
            configs_draft, space_draft = self._tuning_value_proposal_paths(
                candidate_dir,
                proposal_index,
            )
            atomic_write_json(configs_draft, proposal.warm_configs)
            atomic_write_json(space_draft, proposal.search_space)
            values_state = {
                **values_state,
                "status": "validating",
                "proposal_index": proposal_index,
                "configs_draft_revision": file_revision(configs_draft),
                "space_draft_revision": file_revision(space_draft),
                "source": "provided_defaults",
            }
            atomic_write_json(values_state_path, values_state)
            try:
                values_state = self._validate_tuning_value_proposal(
                    action,
                    candidate_path=candidate_path,
                    state_path=values_state_path,
                    state=values_state,
                    configs_draft=configs_draft,
                    space_draft=space_draft,
                )
            except ValidationRejected as exc:
                raise CandidateBuildError(
                    f"provided tuning values failed validation: {exc}"
                ) from exc
            self._publish_tuning_value_drafts(
                state_path=values_state_path,
                state=values_state,
                configs_draft=configs_draft,
                space_draft=space_draft,
                validated_space=self._validated_tuning_space_path(
                    candidate_dir,
                    proposal_index,
                ),
                configs_path=configs_path,
                space_path=space_path,
            )
            self._finalize_validated_tuning_values(
                action,
                state_path=values_state_path,
                candidate_path=candidate_path,
                configs_path=configs_path,
                space_path=space_path,
            )
            return
        base_prompt = (
            f"Candidate: {candidate_path}\nK: {k}\nParents: "
            f"{action.parents or 'none'}\nProvided baseline: {provided}\n\n"
            "Validated PARAM_SCHEMA receipt:\n"
            + json.dumps(schema_lint, ensure_ascii=False)[:20_000]
            + "\n\nBounded lineage evidence:\n"
            + json.dumps(lineage, ensure_ascii=False)[:40_000]
            + "\n\nTask evaluation contract:\n"
            + _bounded_text(task_dir / "TASK.md", 30_000)
            + "\n\nFrozen candidate source:\n"
            + _bounded_text(candidate_path, 80_000)
        )
        last_error: BaseException | None = (
            CandidateBuildError(str(values_state.get("last_error", "")))
            if values_state.get("last_error")
            else None
        )
        diagnostic_path = (
            self.identity.run_dir
            / ".orchestrator"
            / "candidate_contract_diagnostics"
            / f"{action.run_id}-values.json"
        )
        admitted = int(values_state["attempts_admitted"])
        if (
            values_state["status"] == "infrastructure_failed"
            and int(values_state.get("transport_failures", 0))
            >= MAX_TUNING_VALUES_TRANSPORT_FAILURES
        ):
            if last_error_is_upstream_transport(values_state.get("last_error")):
                # Outer coordinator recovery: clear local transport exhaustion
                # so the next loop iteration issues a real provider call.
                values_state = {
                    **values_state,
                    "transport_failures": 0,
                    "status": "infrastructure_failed",
                }
                atomic_write_json(values_state_path, values_state)
            else:
                raise InferenceError(
                    "tuning-values transport retry limit reached: "
                    + str(values_state.get("last_error", "unknown transport failure"))
                )
        replay_pending = values_state["status"] in {
            "started",
            "request_failed",
            "infrastructure_failed",
        }
        while replay_pending or admitted < MAX_TUNING_VALUES_CORRECTIONS + 1:
            if replay_pending:
                purpose = values_state.get("purpose")
                if not isinstance(purpose, str) or not purpose:
                    raise ArtifactError("started tuning-values state lacks purpose")
                replay_pending = False
                request_diagnostic_revision = values_state.get(
                    "request_diagnostic_revision"
                )
                request_proposal_index = values_state.get(
                    "request_proposal_index"
                )
            else:
                purpose = (
                    f"tuning_values:{action.run_id}"
                    if admitted == 0
                    else f"tuning_values:{action.run_id}:correction:{admitted}"
                )
                request_diagnostic_revision = None
                request_proposal_index = None
                if values_state["status"] == "rejected":
                    diagnostic = values_state.get("diagnostic")
                    if not isinstance(diagnostic, dict):
                        raise ArtifactError(
                            "rejected tuning values lack a durable diagnostic"
                        )
                    self._write_contract_diagnostic(
                        action.run_id,
                        stage="values",
                        payload=diagnostic,
                    )
                    request_diagnostic_revision = file_revision(diagnostic_path)
                    request_proposal_index = values_state.get("proposal_index")
                admitted += 1
                values_state = {
                    **values_state,
                    "status": "started",
                    "attempts_admitted": admitted,
                    "purpose": purpose,
                    "request_diagnostic_revision": request_diagnostic_revision,
                    "request_proposal_index": request_proposal_index,
                    "last_error": str(last_error or ""),
                }
                atomic_write_json(values_state_path, values_state)
            prompt = base_prompt
            input_paths = list(fixed_input_paths)
            if request_diagnostic_revision is not None:
                if not (
                    isinstance(request_diagnostic_revision, str)
                    and diagnostic_path.is_file()
                    and file_revision(diagnostic_path)
                    == request_diagnostic_revision
                ):
                    raise ArtifactError(
                        "tuning-values request diagnostic changed after admission"
                    )
                input_paths.append(diagnostic_path)
                rejected_configs = "[not materialized]"
                rejected_space = "[not materialized]"
                if request_proposal_index is not None:
                    rejected_configs_path, rejected_space_path = (
                        self._tuning_value_proposal_paths(
                            candidate_dir,
                            request_proposal_index,
                        )
                    )
                    if not (
                        rejected_configs_path.is_file()
                        and rejected_space_path.is_file()
                        and values_state.get("configs_draft_revision")
                        == file_revision(rejected_configs_path)
                        and values_state.get("space_draft_revision")
                        == file_revision(rejected_space_path)
                    ):
                        raise ArtifactError(
                            "rejected tuning-value proposal is incomplete"
                        )
                    input_paths.extend(
                        [rejected_configs_path, rejected_space_path]
                    )
                    rejected_configs = _bounded_text(
                        rejected_configs_path, 40_000
                    )
                    rejected_space = _bounded_text(rejected_space_path, 30_000)
                prompt += (
                    "\n\nThe prior structured values failed deterministic validation. "
                    "Correct the values once using the complete Python-owned diagnostic "
                    "below; do not change PARAM_SCHEMA or code.\nDiagnostic:\n"
                    + _bounded_diagnostic_text(diagnostic_path)
                    + "\nRejected normalized warm configs:\n"
                    + rejected_configs
                    + "\nRejected normalized search space:\n"
                    + rejected_space
                )
            proposal_index = admitted
            configs_draft, space_draft = self._tuning_value_proposal_paths(
                candidate_dir,
                proposal_index,
            )
            try:
                proposal = self.models.infer(
                    purpose=purpose,
                    schema_version=TUNING_VALUES_SCHEMA_VERSION,
                    system_prompt=TUNING_VALUES_SYSTEM,
                    prompt=prompt,
                    schema=tuning_values_schema(k),
                    input_paths=input_paths,
                    parser=lambda value: TuningValues.from_response(
                        value,
                        k=k,
                        expected_keys=schema_keys,
                    ),
                    # This loop owns its own bounded correction: each attempt is
                    # durably admitted, writes a Python-owned diagnostic, and is
                    # replayed exactly on resume. A second correction inside the
                    # gateway would spend attempts the durable state never saw.
                    max_corrections=0,
                )
            except InferenceRequestError as exc:
                values_state = {
                    **values_state,
                    "status": "request_failed",
                    "last_error_type": type(exc).__name__,
                    "last_error": str(exc),
                }
                atomic_write_json(values_state_path, values_state)
                raise
            except InferenceContractError as exc:
                last_error = exc
                diagnostic = self._contract_diagnostic_payload(
                    action.run_id,
                    stage="values",
                    error=exc,
                )
                values_state = {
                    **values_state,
                    "status": "rejected",
                    "last_error_type": type(exc).__name__,
                    "last_error": str(exc),
                    "proposal_index": None,
                    "diagnostic": diagnostic,
                }
                atomic_write_json(values_state_path, values_state)
                self._write_contract_diagnostic(
                    action.run_id,
                    stage="values",
                    payload=diagnostic,
                )
                continue
            except InferenceError as exc:
                upstream = is_retryable_upstream_failure(exc)
                values_state = {
                    **values_state,
                    "status": "infrastructure_failed",
                    "transport_failures": int(
                        values_state.get("transport_failures", 0)
                    )
                    + (1 if upstream else 0),
                    "last_error_type": type(exc).__name__,
                    "last_error": str(exc),
                }
                atomic_write_json(values_state_path, values_state)
                if upstream:
                    # Transient provider fault: the local transport window and
                    # the coordinator's upstream backoff own this retry.
                    raise
                # Non-upstream inference failure: the candidate, not the run,
                # owns this crash. The admitted attempt produced no proposal,
                # so a re-entry replays it without a new admission.
                raise CandidateBuildError(
                    f"tuning values failed for {action.run_id}: {exc}"
                ) from exc

            configs = proposal.warm_configs
            if provided_defaults is not None:
                configs = [provided_defaults]
            atomic_write_json(configs_draft, configs)
            atomic_write_json(space_draft, proposal.search_space)
            values_state = {
                **values_state,
                "status": "validating",
                "proposal_index": proposal_index,
                "configs_draft_revision": file_revision(configs_draft),
                "space_draft_revision": file_revision(space_draft),
            }
            atomic_write_json(values_state_path, values_state)
            try:
                values_state = self._validate_tuning_value_proposal(
                    action,
                    candidate_path=candidate_path,
                    state_path=values_state_path,
                    state=values_state,
                    configs_draft=configs_draft,
                    space_draft=space_draft,
                )
            except ValidationRejected as exc:
                last_error = exc
                values_state = _read_json(values_state_path)
                continue
            self._publish_tuning_value_drafts(
                state_path=values_state_path,
                state=values_state,
                configs_draft=configs_draft,
                space_draft=space_draft,
                validated_space=self._validated_tuning_space_path(
                    candidate_dir,
                    proposal_index,
                ),
                configs_path=configs_path,
                space_path=space_path,
            )
            self._finalize_validated_tuning_values(
                action,
                state_path=values_state_path,
                candidate_path=candidate_path,
                configs_path=configs_path,
                space_path=space_path,
            )
            return
        raise CandidateBuildError(
            f"tuning values failed for {action.run_id}: {last_error}"
        )

    def evaluate(self, action: RoundAction) -> CandidateOutcome:
        if action.run_id is None:
            raise ValueError("cannot evaluate an action without run_id")
        run_id = action.run_id
        candidate_dir = self.candidate_dir(run_id)
        candidate_path = candidate_dir / "train.py"
        configs_path = candidate_dir / "_warm_configs.json"
        report_path = candidate_dir / "tune_report.json"
        provided = self.is_provided_baseline(run_id)
        k_eval = 1 if provided else self._tuner_integer("K_eval", 3, minimum=2)
        log_path = (
            self.identity.run_dir
            / ".orchestrator"
            / "workers"
            / f"{run_id}-warmstart.log"
        )

        for _ in range(9):
            terminal_failure = self._phase_a_terminal_failure(
                run_id, report_path
            )
            if terminal_failure is not None and not self._phase_a_retry_is_authorized(
                run_id,
                terminal_failure,
            ):
                outcome = self._handle_phase_a_terminal(
                    action,
                    terminal_failure,
                    report_path,
                )
                if outcome is not None:
                    return outcome
                continue
            completed = self._phase_a_completed_outcome(run_id, report_path)
            if completed is not None:
                return completed
            worker_input_revision = self._phase_a_worker_input_revision(run_id)
            worker_recovery = self._phase_a_worker_recovery(
                run_id,
                worker_input_revision,
            )
            if worker_recovery is not None:
                if worker_recovery["status"] in {
                    "recovery_started",
                    "exhausted",
                }:
                    raise ArtifactError(
                        "Phase A worker recovery is already consumed for "
                        f"candidate {run_id}"
                    )
                remaining = (
                    float(worker_recovery["retry_not_before_epoch"])
                    - time.time()
                )
                if remaining > 5.0:
                    raise ArtifactError(
                        f"Phase A worker recovery backoff is invalid for {run_id}"
                    )
                if remaining > 0:
                    time.sleep(remaining)
                self._consume_phase_a_worker_recovery(
                    run_id,
                    worker_input_revision,
                    worker_recovery,
                )
            result = self.toolchain.warmstart(
                candidate_path,
                configs_path,
                report_path,
                k_eval=k_eval,
                task_config=self.task_config,
                output_path=log_path,
            )
            if result.returncode == 0 and not result.timed_out:
                completed = self._phase_a_completed_outcome(run_id, report_path)
                if completed is None:
                    raise ValueError(
                        f"successful Phase A worker lacks a completed report: {report_path}"
                    )
                return completed
            if result.returncode == 3 and not result.timed_out:
                payload = parse_json_output(result.output)
                failure_category = (
                    payload.get("failure_category")
                    if isinstance(payload, dict)
                    else None
                )
                if not isinstance(payload, dict) or not isinstance(
                    failure_category, str
                ):
                    raise ToolFailure("warm-config evaluation failure", result)
                persisted = self._phase_a_terminal_failure(run_id, report_path)
                if persisted is None:
                    raise ArtifactError(
                        f"warmstart worker returned no durable terminal for {run_id}"
                    )
                if any(
                    payload.get(key) != persisted.get(key)
                    for key in (
                        "phase",
                        "status",
                        "crash_index",
                        "crash_params",
                        "objective_slot_consumed",
                        "failure_category",
                        "failure_ref",
                    )
                ):
                    raise ArtifactError(
                        f"worker stdout disagrees with its durable terminal for {run_id}"
                    )
                # Re-enter through the terminal-first path at the top of the
                # loop. It owns closure, debug admission, and retry authority.
                continue
            if result.returncode == 4 and not result.timed_out:
                return self._close_budget_exhausted(run_id, report_path)
            if result.timed_out:
                # The worker may have been killed after persisting an objective
                # intent/reservation but before returning its terminal payload.
                # One bounded, durable reinvocation runs the worker's
                # deterministic reconciler; a coordinator restart cannot grant
                # a fresh allowance for the same frozen inputs.
                recovery = self._record_phase_a_worker_timeout(
                    run_id,
                    worker_input_revision,
                    result,
                )
                if recovery["status"] == "retryable":
                    continue
            raise ToolFailure("warm-config evaluation process", result)

        return self._close_crash(run_id, report_path)

    def _phase_a_completed_outcome(
        self,
        run_id: str,
        report_path: Path,
    ) -> CandidateOutcome | None:
        if not report_path.is_file():
            return None
        report = _read_json(report_path)
        phase_a = report.get("phase_a") if isinstance(report, dict) else None
        if not isinstance(phase_a, dict) or phase_a.get("status") != "ok":
            return None
        score = phase_a.get("best_warm_score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError(
                f"successful Phase A lacks a finite best_warm_score: {report_path}"
            )
        self.toolchain.project_screening_report(
            self.identity.run_dir,
            run_id,
            report_path,
        )
        record = self.toolchain.record_score(
            self.identity.run_dir,
            run_id,
            float(score),
        )
        return CandidateOutcome(
            status=str(record.get("status")),
            best_score=float(score),
        )

    def _phase_a_worker_recovery_path(
        self,
        run_id: str,
        input_revision: str,
    ) -> Path:
        return (
            self.identity.run_dir
            / ".orchestrator"
            / "workers"
            / (
                f"{run_id}-warmstart-recovery-"
                f"{input_revision.removeprefix('sha256:')[:16]}.json"
            )
        )

    def _phase_a_worker_input_revision(self, run_id: str) -> str:
        """Bind recovery to score semantics while ignoring BASE_PARAMS publish."""
        return self._phase_a_retry_input_revision(run_id)

    def _phase_a_worker_recovery(
        self,
        run_id: str,
        input_revision: str,
    ) -> dict[str, Any] | None:
        if not _is_sha256(input_revision):
            raise ArtifactError(
                f"invalid Phase A worker input revision for {run_id}"
            )
        path = self._phase_a_worker_recovery_path(run_id, input_revision)
        if not path.is_file():
            return None
        try:
            receipt = _read_json(path)
        except ValueError as exc:
            raise ArtifactError(
                f"invalid Phase A worker recovery receipt: {path}"
            ) from exc
        last_timeout = receipt.get("last_timeout") if isinstance(receipt, dict) else None
        observed = receipt.get("timeouts_observed") if isinstance(receipt, dict) else None
        status = receipt.get("status") if isinstance(receipt, dict) else None
        retry_epoch = (
            receipt.get("retry_not_before_epoch")
            if isinstance(receipt, dict)
            else None
        )
        if not (
            isinstance(receipt, dict)
            and set(receipt)
            == {
                "schema_version",
                "kind",
                "run_id",
                "input_revision",
                "timeouts_observed",
                "status",
                "retry_not_before_epoch",
                "recovery_started_epoch",
                "last_timeout",
            }
            and receipt.get("schema_version") == 1
            and receipt.get("kind") == "phase_a_worker_recovery"
            and receipt.get("run_id") == run_id
            and receipt.get("input_revision") == input_revision
            and type(observed) is int
            and observed in {1, MAX_PHASE_A_WORKER_TIMEOUTS}
            and status
            in (
                {"retryable", "recovery_started"}
                if observed == 1
                else {"exhausted"}
            )
            and isinstance(retry_epoch, (int, float))
            and not isinstance(retry_epoch, bool)
            and math.isfinite(float(retry_epoch))
            and (
                receipt.get("recovery_started_epoch") is None
                if status == "retryable"
                else isinstance(
                    receipt.get("recovery_started_epoch"), (int, float)
                )
                and not isinstance(receipt.get("recovery_started_epoch"), bool)
                and math.isfinite(float(receipt["recovery_started_epoch"]))
            )
            and isinstance(last_timeout, dict)
            and set(last_timeout)
            == {"elapsed_seconds", "returncode", "interrupted"}
            and isinstance(last_timeout.get("elapsed_seconds"), (int, float))
            and not isinstance(last_timeout.get("elapsed_seconds"), bool)
            and math.isfinite(float(last_timeout["elapsed_seconds"]))
            and float(last_timeout["elapsed_seconds"]) >= 0.0
            and type(last_timeout.get("returncode")) is int
            and isinstance(last_timeout.get("interrupted"), bool)
        ):
            raise ArtifactError(
                f"malformed Phase A worker recovery receipt: {path}"
            )
        return receipt

    def _consume_phase_a_worker_recovery(
        self,
        run_id: str,
        input_revision: str,
        receipt: dict[str, Any],
    ) -> None:
        current = self._phase_a_worker_recovery(run_id, input_revision)
        if current != receipt or receipt.get("status") != "retryable":
            raise ArtifactError(
                f"Phase A worker recovery changed before consumption for {run_id}"
            )
        atomic_write_json(
            self._phase_a_worker_recovery_path(run_id, input_revision),
            {
                **receipt,
                "status": "recovery_started",
                "recovery_started_epoch": time.time(),
            },
        )

    def _record_phase_a_worker_timeout(
        self,
        run_id: str,
        input_revision: str,
        result: Any,
    ) -> dict[str, Any]:
        previous = self._phase_a_worker_recovery(run_id, input_revision)
        if previous is not None and previous.get("status") != "recovery_started":
            raise ArtifactError(
                f"Phase A timeout occurred without a consumed recovery for {run_id}"
            )
        observed = int(previous["timeouts_observed"]) + 1 if previous else 1
        if observed > MAX_PHASE_A_WORKER_TIMEOUTS:
            raise ArtifactError(
                f"Phase A worker recovery exceeded its durable cap for {run_id}"
            )
        elapsed = result.elapsed_seconds
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            raise ArtifactError(
                f"Phase A worker timeout has invalid elapsed time for {run_id}"
            )
        receipt = {
            "schema_version": 1,
            "kind": "phase_a_worker_recovery",
            "run_id": run_id,
            "input_revision": input_revision,
            "timeouts_observed": observed,
            "status": (
                "retryable"
                if observed < MAX_PHASE_A_WORKER_TIMEOUTS
                else "exhausted"
            ),
            "retry_not_before_epoch": (
                time.time() + PHASE_A_WORKER_RECOVERY_BACKOFF_SECONDS
            ),
            "recovery_started_epoch": (
                previous.get("recovery_started_epoch") if previous else None
            ),
            "last_timeout": {
                "elapsed_seconds": float(elapsed),
                "returncode": result.returncode,
                "interrupted": result.interrupted,
            },
        }
        atomic_write_json(
            self._phase_a_worker_recovery_path(run_id, input_revision),
            receipt,
        )
        return receipt

    def _phase_a_terminal_failure(
        self,
        run_id: str,
        report_path: Path,
    ) -> dict[str, Any] | None:
        if not report_path.is_file():
            return None
        report = _read_json(report_path)
        phase_a = report.get("phase_a") if isinstance(report, dict) else None
        if not isinstance(phase_a, dict):
            raise ArtifactError(f"malformed Phase A report: {report_path}")
        terminal = phase_a.get("terminal_failure")
        if terminal is None:
            return None
        if not isinstance(terminal, dict):
            raise ArtifactError(
                f"Phase A terminal failure must be an object: {report_path}"
            )
        if (
            terminal.get("phase") not in {"a", "preflight"}
            or terminal.get("status") != "crashed"
            or not isinstance(terminal.get("objective_slot_consumed"), bool)
            or not isinstance(terminal.get("failure_category"), str)
            or not isinstance(terminal.get("crash_params"), dict)
            or not isinstance(terminal.get("failure_receipt"), dict)
            or not isinstance(terminal.get("failure_ref"), dict)
            or not _is_failure_id(
                terminal.get("failure_ref", {}).get("failure_id")
            )
        ):
            raise ArtifactError(
                f"malformed Phase A terminal failure for candidate {run_id}"
            )
        failure_ref = terminal["failure_ref"]
        failure_id = failure_ref["failure_id"]
        if (
            set(failure_ref)
            != {"schema_version", "failure_id", "artifact", "sha256"}
            or failure_ref.get("schema_version") != 1
            or failure_ref.get("artifact")
            != f"_failures/{failure_id}.json"
            or not _is_sha256(failure_ref.get("sha256"))
        ):
            raise ArtifactError(
                f"malformed Phase A failure artifact reference for {run_id}"
            )
        verified_failure = self.toolchain.verify_failure_artifact(
            report_path,
            failure_id,
        )
        if verified_failure != {
            "failure_ref": failure_ref,
            "failure_receipt": terminal["failure_receipt"],
        }:
            raise ArtifactError(
                f"Phase A failure evidence does not match its immutable artifact for {run_id}"
            )
        terminal_revision = terminal.get("candidate_execution_revision")
        report_revision = phase_a.get("candidate_code_revision")
        if (
            not isinstance(terminal_revision, dict)
            or set(terminal_revision) != EXECUTION_REVISION_KEYS
            or not _is_sha256(terminal_revision.get("structure_sha256"))
            or not _is_sha256(terminal_revision.get("revision_sha256"))
            or terminal_revision != report_revision
        ):
            raise ArtifactError(
                f"Phase A terminal failure revision does not match its report for {run_id}"
            )
        if terminal["phase"] == "preflight":
            if (
                terminal["objective_slot_consumed"] is not False
                or terminal.get("objective_attempt_id") is not None
                or terminal.get("objective_reservation") is not None
            ):
                raise ArtifactError(
                    f"preflight terminal failure has objective accounting for {run_id}"
                )
        else:
            attempt_id = terminal.get("objective_attempt_id")
            reservation = terminal.get("objective_reservation")
            if not (
                terminal.get("objective_slot_consumed") is True
                and isinstance(attempt_id, str)
                and bool(attempt_id)
                and isinstance(reservation, dict)
                and reservation.get("schema_version") == 1
                and reservation.get("kind") == "score_attempt"
                and reservation.get("attempt_id") == attempt_id
                and reservation.get("run_id") == run_id
                and reservation.get("phase") == "phase_a"
                and reservation.get("method") == "warmstart"
                and reservation.get("params_sha256")
                == json_revision(terminal["crash_params"])
            ):
                raise ArtifactError(
                    f"Phase A terminal failure lacks an exact objective reservation for {run_id}"
                )
            candidate_path = self.candidate_dir(run_id) / "train.py"
            receipts = self.toolchain.objective_attempt_receipts(
                candidate_path,
                phase="phase_a",
                method="warmstart",
            )
            if sum(receipt == reservation for receipt in receipts) != 1:
                raise ArtifactError(
                    f"Phase A terminal reservation is absent from the durable attempt log for {run_id}"
                )
        return terminal

    def _phase_a_retry_input_revision(self, run_id: str) -> str:
        candidate_dir = self.candidate_dir(run_id)
        execution_revision = self.toolchain.candidate_execution_revision(
            candidate_dir / "train.py"
        )
        if not (
            isinstance(execution_revision, dict)
            and _is_sha256(execution_revision.get("structure_sha256"))
            and _is_sha256(execution_revision.get("revision_sha256"))
        ):
            raise ArtifactError(
                f"invalid candidate execution revision for Phase A retry {run_id}"
            )
        supporting_revision = paths_revision(
            (
                candidate_dir / "_warm_configs.json",
                candidate_dir / "_parameter_transfer.json",
                candidate_dir / self.CONTRACT_RECEIPT,
                candidate_dir / self.PREFLIGHT_RECEIPT,
                self.identity.run_dir / "framework_cfg.json",
            )
        )
        return json_revision(
            {
                "candidate_execution_revision": execution_revision,
                "supporting_revision": supporting_revision,
            }
        )

    def _phase_a_retry_is_authorized(
        self,
        run_id: str,
        terminal: dict[str, Any],
    ) -> bool:
        path = self.candidate_dir(run_id) / self.PHASE_A_RETRY_RECEIPT
        if not path.is_file():
            return False
        try:
            receipt = _read_json(path)
        except ValueError as exc:
            raise ArtifactError(f"invalid Phase A retry receipt: {path}") from exc
        expected_failure_id = (
            terminal.get("failure_ref", {}).get("failure_id")
            if isinstance(terminal.get("failure_ref"), dict)
            else None
        )
        if not (
            isinstance(receipt, dict)
            and receipt.get("schema_version") == 1
            and receipt.get("kind") == "phase_a_debug_retry"
            and receipt.get("status") == "authorized"
            and receipt.get("run_id") == run_id
            and isinstance(receipt.get("failure_id"), str)
            and bool(receipt.get("failure_id"))
            and _is_sha256(receipt.get("terminal_failure_revision"))
            and _is_sha256(receipt.get("retry_input_revision"))
        ):
            raise ArtifactError(
                f"malformed Phase A retry receipt for {run_id}"
            )
        if (
            receipt.get("failure_id") != expected_failure_id
            or receipt.get("terminal_failure_revision")
            != json_revision(terminal)
        ):
            # A successful retry can durably replace terminal A with terminal B
            # before the parent observes the worker exit.  A well-formed A
            # authorization is historical state, not corruption and never
            # authorization to replay B.
            return False
        if receipt.get("retry_input_revision") != self._phase_a_retry_input_revision(
            run_id
        ):
            raise ArtifactError(
                f"Phase A retry receipt is stale for candidate {run_id}"
            )
        return True

    def _authorize_phase_a_retry(
        self,
        run_id: str,
        terminal: dict[str, Any],
    ) -> None:
        failure_ref = terminal.get("failure_ref")
        failure_id = (
            failure_ref.get("failure_id")
            if isinstance(failure_ref, dict)
            else None
        )
        if not isinstance(failure_id, str) or not failure_id:
            raise ArtifactError(
                f"Phase A terminal failure lacks a failure identity for {run_id}"
            )
        atomic_write_json(
            self.candidate_dir(run_id) / self.PHASE_A_RETRY_RECEIPT,
            {
                "schema_version": 1,
                "kind": "phase_a_debug_retry",
                "status": "authorized",
                "run_id": run_id,
                "failure_id": failure_id,
                "terminal_failure_revision": json_revision(terminal),
                "retry_input_revision": self._phase_a_retry_input_revision(run_id),
            },
        )

    def _handle_phase_a_terminal(
        self,
        action: RoundAction,
        terminal: dict[str, Any],
        report_path: Path,
    ) -> CandidateOutcome | None:
        if action.run_id is None:  # pragma: no cover - checked by evaluate
            raise ValueError("cannot finalize a terminal failure without run_id")
        category = terminal["failure_category"]
        if category == "candidate_code_incompatibility":
            evidence = FailureEvidence.from_worker_payload(
                action.run_id, terminal
            )
            if not self._debug_once(
                action,
                evidence,
                report_path,
                terminal=terminal,
            ):
                return self._close_crash(action.run_id, report_path)
            return None
        if category in {
            "timeout_or_resource",
            "unknown_non_candidate_failure",
        }:
            # A preflight terminal never consumes an objective slot (enforced
            # by _phase_a_terminal_failure).  With earlier reservations this
            # closes the crash directly; without any, _close_crash raises
            # CandidateBuildError and the coordinator records an
            # evaluation-stage candidate close.  Neither blocks the run.
            return self._close_crash(action.run_id, report_path)
        if (
            terminal.get("objective_slot_consumed") is True
            and category
            in {
                "process_interruption",
                "process_interruption_or_orphaned_work",
            }
        ):
            return self._close_crash(action.run_id, report_path)
        raise ArtifactError(
            f"unsupported Phase A terminal failure category for {action.run_id}: "
            f"{category!r}"
        )

    def preflight(self, action: RoundAction) -> dict[str, Any]:
        if action.run_id is None:
            raise ValueError("cannot preflight an action without run_id")
        if self.preflight_is_ready(action):
            value = _read_json(
                self.candidate_dir(action.run_id) / self.PREFLIGHT_RECEIPT
            )
            if not isinstance(value, dict):  # pragma: no cover - readiness checked it
                raise ArtifactError("candidate preflight receipt is not an object")
            return value

        candidate_dir = self.candidate_dir(action.run_id)
        candidate_path = candidate_dir / "train.py"
        configs_path = candidate_dir / "_warm_configs.json"
        k_eval = (
            1
            if self.is_provided_baseline(action.run_id)
            else self._tuner_integer("K_eval", 3, minimum=2)
        )
        input_revision = paths_revision(self._preflight_input_paths(action.run_id))
        attempts_path = self.identity.run_dir / "evaluation_attempts.jsonl"
        attempts_before = paths_revision((attempts_path,))
        payload = None
        last_timeout: PreflightTimeout | None = None
        attempts = 0
        # A preflight timeout is side-effect-free and consumes no objective
        # slot, so it earns one bounded retry before the candidate is closed.
        for _attempt in range(self.PREFLIGHT_TIMEOUT_RETRIES + 1):
            attempts += 1
            try:
                payload = self.toolchain.candidate_preflight(
                    candidate_path,
                    configs_path,
                    k_eval=k_eval,
                    task_config=self.task_config,
                )
                last_timeout = None
                break
            except PreflightTimeout as exc:
                last_timeout = exc
        if last_timeout is not None:
            raise CandidateBuildError(
                f"candidate preflight timed out "
                f"{self.PREFLIGHT_TIMEOUT_RETRIES + 1} times: {last_timeout}"
            ) from last_timeout
        assert payload is not None
        if paths_revision(self._preflight_input_paths(action.run_id)) != input_revision:
            raise ArtifactError("candidate inputs changed during no-score preflight")
        if paths_revision((attempts_path,)) != attempts_before:
            raise ArtifactError("candidate preflight changed objective attempt accounting")
        if (
            not isinstance(payload, dict)
            or payload.get("status") not in {"ok", "not_declared"}
            or payload.get("objective_calls") != 0
        ):
            raise ArtifactError(f"invalid candidate preflight result: {payload!r}")
        receipt = {
            "schema_version": 1,
            "run_id": action.run_id,
            "input_revision": input_revision,
            "k_eval": k_eval,
            "status": payload["status"],
            "objective_calls": 0,
            "attempts": attempts,
            "result": payload,
        }
        atomic_write_json(candidate_dir / self.PREFLIGHT_RECEIPT, receipt)
        return receipt

    def record_build_failure(
        self,
        action: RoundAction,
        error: BaseException,
        *,
        stage: str,
    ) -> CandidateOutcome:
        """Close a candidate that failed before evaluation as a ledger crash.

        A build failure belongs to the candidate, not the run: the receipt
        records the failing stage and the ledger record is closed as a crash
        so the coordinator can resolve the action and continue the round.
        """
        if action.run_id is None:
            raise ValueError("cannot close an action without run_id")
        if stage not in {
            "implementation",
            "tuning_contract",
            "preflight",
            "evaluation",
        }:
            raise ValueError(f"invalid candidate build-failure stage: {stage}")
        receipt = (
            self.identity.run_dir
            / ".orchestrator"
            / "candidate_failures"
            / f"{action.run_id}.json"
        )
        atomic_write_json(
            receipt,
            {
                "schema_version": 1,
                "run_id": action.run_id,
                "stage": stage,
                "status": "crash",
                "error": f"{type(error).__name__}: {error}",
                "objective_calls": 0,
            },
        )
        record = self.toolchain.record_crash(self.identity.run_dir, action.run_id)
        return CandidateOutcome(status=str(record.get("status", "crash")))

    @staticmethod
    def _contract_diagnostic_payload(
        run_id: str,
        *,
        stage: str,
        error: BaseException,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": "candidate_contract_diagnostic",
            "run_id": run_id,
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
        }
        if isinstance(error, ToolFailure):
            try:
                validator_output = json.loads(error.result.output)
            except json.JSONDecodeError:
                validator_output = None
            if isinstance(validator_output, dict):
                payload["validator_output"] = validator_output
        return payload

    def _write_contract_diagnostic(
        self,
        run_id: str,
        *,
        stage: str,
        error: BaseException | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Path:
        if payload is None:
            if error is None:
                raise ValueError("contract diagnostic requires an error or payload")
            payload = self._contract_diagnostic_payload(
                run_id,
                stage=stage,
                error=error,
            )
        elif not (
            payload.get("schema_version") == 1
            and payload.get("kind") == "candidate_contract_diagnostic"
            and payload.get("run_id") == run_id
            and payload.get("stage") == stage
        ):
            raise ArtifactError("persisted candidate diagnostic has invalid identity")
        path = (
            self.identity.run_dir
            / ".orchestrator"
            / "candidate_contract_diagnostics"
            / f"{run_id}-{stage}.json"
        )
        atomic_write_json(path, payload)
        return path

    def _debug_once(
        self,
        action: RoundAction,
        evidence: FailureEvidence,
        report_path: Path,
        *,
        terminal: dict[str, Any] | None = None,
    ) -> bool:
        run_id = evidence.run_id
        candidate_dir = self.candidate_dir(run_id)
        candidate_path = candidate_dir / "train.py"
        configs_path = candidate_dir / "_warm_configs.json"
        space_path = candidate_dir / "_search_space.json"
        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        if terminal is None:
            # Direct unit-level callers predate durable worker terminals. The
            # active coordinator always passes the validated persisted object.
            terminal = {
                "phase": evidence.phase,
                "status": "crashed",
                "crash_index": evidence.crash_index,
                "crash_params": evidence.crash_params,
                "objective_slot_consumed": evidence.objective_slot_consumed,
                "failure_category": evidence.failure_category,
                "failure_receipt": evidence.failure_receipt,
                "failure_ref": evidence.failure_ref,
                "candidate_execution_revision": (
                    evidence.candidate_execution_revision or {}
                ),
            }
        terminal_revision = json_revision(terminal)
        repair_path = candidate_dir / self.PHASE_A_REPAIR_RECEIPT
        repair = self._matching_phase_a_repair(
            repair_path,
            run_id=run_id,
            terminal=terminal,
        )
        if repair is not None:
            try:
                self.debug_policy.reserve_analysis(
                    evidence,
                    reservation_id=terminal_revision,
                )
            except DebugAllowanceExhausted:
                return False
            return self._resume_phase_a_repair(
                action,
                evidence,
                terminal,
                repair,
                report_path,
            )

        expected_execution_revision = evidence.candidate_execution_revision
        if isinstance(expected_execution_revision, dict):
            current_execution_revision = self.toolchain.candidate_execution_revision(
                candidate_path
            )
            if current_execution_revision != expected_execution_revision:
                raise ArtifactError(
                    f"candidate {run_id} changed after its failure without a durable repair plan"
                )
        try:
            self.debug_policy.reserve_analysis(
                evidence,
                reservation_id=terminal_revision,
            )
        except DebugAllowanceExhausted:
            return False

        artifact_value = evidence.failure_ref.get("artifact")
        failure_artifact = None
        if artifact_value is not None:
            if not isinstance(artifact_value, str) or not artifact_value:
                raise ArtifactError(
                    f"failure artifact reference is malformed for candidate {run_id}"
                )
            candidate_root = candidate_dir.resolve()
            failure_artifact = (candidate_root / artifact_value).resolve()
            try:
                failure_artifact.relative_to(candidate_root)
            except ValueError as exc:
                raise ArtifactError(
                    f"failure artifact escapes candidate directory: {artifact_value!r}"
                ) from exc
            if not failure_artifact.is_file():
                raise ArtifactError(
                    f"failure artifact does not exist for candidate {run_id}: "
                    f"{artifact_value}"
                )
        prompt = (
            f"Candidate: {run_id}\nFailure phase: {evidence.phase}\n"
            f"Crash config index: {evidence.crash_index}\n"
            "Crash config:\n"
            + json.dumps(evidence.crash_params, ensure_ascii=False)
            + "\nFailure receipt:\n"
            + json.dumps(evidence.failure_receipt, ensure_ascii=False)
            + "\n\nTask contract:\n"
            + _bounded_text(task_dir / "TASK.md", 30_000)
            + "\n\nCandidate source:\n"
            + _bounded_text(candidate_path)
        )
        input_paths = [
            candidate_path,
            configs_path,
            space_path,
            report_path,
            task_dir / "TASK.md",
            task_dir / "task.toml",
        ]
        if failure_artifact is not None:
            input_paths.append(failure_artifact)
        try:
            decision = self.models.infer(
                purpose=f"debug:{run_id}:{evidence.fingerprint}",
                schema_version=1,
                system_prompt=DEBUG_SYSTEM,
                prompt=prompt,
                schema=DEBUG_SCHEMA,
                input_paths=input_paths,
                parser=parse_debug_response,
                # `reserve_analysis` above grants exactly one analyzer call per
                # candidate/failure fingerprint, and that reservation is already
                # durably written. A gateway correction would spend a second
                # backend call against a one-call allowance without recording
                # it, so a malformed diagnosis degrades to a crash instead.
                max_corrections=0,
            )
        except InferenceContractError:
            return False
        except InferenceError as exc:
            if is_retryable_upstream_failure(exc):
                # Transient provider fault: the coordinator's upstream backoff
                # owns this retry.
                raise
            # Non-upstream inference failure degrades the same way as a
            # malformed decision: the repair is skipped and the candidate
            # closes as a crash, so the run continues.
            return False
        decision_payload = {
            "verdict": decision.verdict.value,
            "rationale": decision.rationale,
            "corrected_config": decision.corrected_config,
            "repair_instructions": decision.repair_instructions,
        }
        repair = {
            "schema_version": 1,
            "kind": "phase_a_debug_repair",
            "status": "planned",
            "run_id": run_id,
            "failure_id": evidence.fingerprint,
            "terminal_failure_revision": terminal_revision,
            "analysis_reservation_id": terminal_revision,
            "candidate_revision_before": file_revision(candidate_path),
            "decision": decision_payload,
        }
        atomic_write_json(repair_path, repair)
        try:
            self.debug_policy.reserve_analysis(
                evidence,
                reservation_id=terminal_revision,
            )
        except DebugAllowanceExhausted:
            return False
        return self._resume_phase_a_repair(
            action,
            evidence,
            terminal,
            repair,
            report_path,
        )

    def _matching_phase_a_repair(
        self,
        path: Path,
        *,
        run_id: str,
        terminal: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            receipt = _read_json(path)
        except ValueError as exc:
            raise ArtifactError(f"invalid Phase A repair receipt: {path}") from exc
        if not (
            isinstance(receipt, dict)
            and receipt.get("schema_version") == 1
            and receipt.get("kind") == "phase_a_debug_repair"
            and receipt.get("status")
            in {"planned", "edit_completed", "rejected", "completed"}
            and receipt.get("run_id") == run_id
            and isinstance(receipt.get("failure_id"), str)
            and bool(receipt.get("failure_id"))
            and _is_sha256(receipt.get("terminal_failure_revision"))
            and _is_sha256(receipt.get("analysis_reservation_id"))
            and receipt.get("analysis_reservation_id")
            == receipt.get("terminal_failure_revision")
            and _is_sha256(receipt.get("candidate_revision_before"))
            and isinstance(receipt.get("decision"), dict)
        ):
            raise ArtifactError(f"malformed Phase A repair receipt: {path}")
        allowed_fields = {
            "schema_version",
            "kind",
            "status",
            "run_id",
            "failure_id",
            "terminal_failure_revision",
            "analysis_reservation_id",
            "candidate_revision_before",
            "decision",
        }
        if "candidate_revision_after" in receipt:
            allowed_fields.add("candidate_revision_after")
        if set(receipt) != allowed_fields:
            raise ArtifactError(
                f"Phase A repair receipt has unexpected fields: {path}"
            )
        if (
            receipt["failure_id"]
            != terminal.get("failure_ref", {}).get("failure_id")
            or receipt["terminal_failure_revision"] != json_revision(terminal)
        ):
            return None
        try:
            decision = DebugDecision.from_dict(receipt["decision"])
        except (TypeError, ValueError) as exc:
            raise ArtifactError(f"invalid decision in Phase A repair receipt: {path}") from exc
        after_revision = receipt.get("candidate_revision_after")
        if decision.verdict is DebugVerdict.CODE_INCOMPATIBLE:
            if receipt["status"] in {"edit_completed", "completed"} and not _is_sha256(
                after_revision
            ):
                raise ArtifactError(
                    f"completed Phase A code repair lacks its output revision: {path}"
                )
            if (
                receipt["status"] == "completed"
                and after_revision == receipt["candidate_revision_before"]
            ):
                raise ArtifactError(
                    f"completed Phase A code repair did not change the candidate: {path}"
                )
            if receipt["status"] == "planned" and after_revision is not None:
                raise ArtifactError(
                    f"planned Phase A code repair already claims an output: {path}"
                )
            if (
                receipt["status"] == "rejected"
                and after_revision is not None
                and not _is_sha256(after_revision)
            ):
                raise ArtifactError(
                    f"rejected Phase A code repair has an invalid output revision: {path}"
                )
        else:
            allowed_statuses = (
                {"planned", "rejected", "completed"}
                if decision.verdict is DebugVerdict.CONFIG_INVALID
                else {"planned", "rejected"}
            )
            if receipt["status"] not in allowed_statuses:
                raise ArtifactError(
                    f"Phase A repair verdict/status is impossible: {path}"
                )
            if after_revision is not None:
                raise ArtifactError(
                    f"non-code Phase A repair claims a candidate output revision: {path}"
                )
        return receipt

    def _write_phase_a_repair_status(
        self,
        run_id: str,
        repair: dict[str, Any],
        status: str,
    ) -> None:
        atomic_write_json(
            self.candidate_dir(run_id) / self.PHASE_A_REPAIR_RECEIPT,
            {**repair, "status": status},
        )

    def _resume_phase_a_repair(
        self,
        action: RoundAction,
        evidence: FailureEvidence,
        terminal: dict[str, Any],
        repair: dict[str, Any],
        report_path: Path,
    ) -> bool:
        run_id = evidence.run_id
        if repair["status"] == "rejected":
            return False
        decision = DebugDecision.from_dict(repair["decision"])
        if decision.verdict is DebugVerdict.ABANDON:
            self._write_phase_a_repair_status(run_id, repair, "rejected")
            return False

        candidate_dir = self.candidate_dir(run_id)
        candidate_path = candidate_dir / "train.py"
        configs_path = candidate_dir / "_warm_configs.json"
        space_path = candidate_dir / "_search_space.json"
        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name

        if decision.verdict is DebugVerdict.CONFIG_INVALID:
            expected_execution_revision = terminal.get(
                "candidate_execution_revision"
            )
            if (
                isinstance(expected_execution_revision, dict)
                and _is_sha256(
                    expected_execution_revision.get("structure_sha256")
                )
                and _is_sha256(
                    expected_execution_revision.get("revision_sha256")
                )
                and (
                self.toolchain.candidate_execution_revision(candidate_path)
                != expected_execution_revision
                )
            ):
                raise ArtifactError(
                    f"candidate {run_id} changed outside the durable config repair plan"
                )
            if evidence.crash_index == 0 and (
                action.parents or self.is_provided_baseline(run_id)
            ):
                self._write_phase_a_repair_status(run_id, repair, "rejected")
                return False
            configs = _read_json(configs_path)
            corrected = decision.corrected_config
            if (
                not isinstance(configs, list)
                or evidence.crash_index >= len(configs)
                or not isinstance(corrected, dict)
                or not isinstance(configs[evidence.crash_index], dict)
                or set(corrected) != set(configs[evidence.crash_index])
                or corrected == evidence.crash_params
            ):
                self._write_phase_a_repair_status(run_id, repair, "rejected")
                return False
            current = configs[evidence.crash_index]
            if current == evidence.crash_params:
                configs[evidence.crash_index] = corrected
                atomic_write_json(configs_path, configs)
            elif current != corrected:
                raise ArtifactError(
                    f"config changed outside the durable repair plan for {run_id}"
                )
            if evidence.crash_index == 0:
                corrected_path = candidate_dir / self.DEBUG_CORRECTED_BASE
                atomic_write_json(corrected_path, corrected)
                self.toolchain.apply_base_params(candidate_path, corrected_path)
        else:
            reservation_id = repair["terminal_failure_revision"]
            try:
                self.debug_policy.reserve_code_repair(
                    run_id,
                    reservation_id=reservation_id,
                )
            except DebugAllowanceExhausted:
                self._write_phase_a_repair_status(run_id, repair, "rejected")
                return False
            repair_prompt = (
                f"Repair {candidate_path} for this evidenced incompatibility.\n"
                f"Diagnosis: {decision.rationale}\n"
                f"Instructions: {decision.repair_instructions}\n"
                f"Failure receipt: {json.dumps(evidence.failure_receipt, ensure_ascii=False)}"
            )
            edit_spec = AgentEditSpec(
                purpose=f"debug_repair:{run_id}:{evidence.fingerprint}",
                schema_version=1,
                cwd=self.identity.repo_root,
                system_prompt=CODE_REPAIR_SYSTEM,
                prompt=repair_prompt,
                tools=("Read", "Grep", "Edit"),
                read_roots=(task_dir, candidate_dir),
                write_paths=(candidate_path,),
                input_paths=(
                    candidate_path,
                    candidate_dir / "prepare.py",
                    configs_path,
                    space_path,
                    report_path,
                    task_dir / "TASK.md",
                    task_dir / "task.toml",
                ),
                immutable_input_paths=(
                    candidate_dir / "prepare.py",
                    configs_path,
                    space_path,
                    report_path,
                    task_dir / "TASK.md",
                    task_dir / "task.toml",
                ),
                max_turns=12,
            )
            if repair["status"] == "edit_completed":
                if (
                    not isinstance(repair.get("candidate_revision_after"), str)
                    or file_revision(candidate_path)
                    != repair["candidate_revision_after"]
                ):
                    raise ArtifactError(
                        f"completed debug repair output changed for {run_id}"
                    )
            else:
                completed = getattr(
                    self.models, "completed_edit_matches", None
                )
                if callable(completed) and completed(edit_spec):
                    pass
                elif file_revision(candidate_path) == repair["candidate_revision_before"]:
                    try:
                        self.models.edit(
                            edit_spec,
                            validate=lambda: self._validate_authored_python(
                                candidate_path
                            ),
                        )
                    except SourceValidationRejected as exc:
                        # The repair reservation is consumed.  An invalid edit
                        # leaves the candidate source half-repaired, so close
                        # the candidate through the normal crash path instead
                        # of letting a rejected repair strand the run on
                        # resume without a completed edit receipt.
                        self._write_contract_diagnostic(
                            run_id,
                            stage="debug_repair",
                            error=exc,
                        )
                        self._write_phase_a_repair_status(
                            run_id, repair, "rejected"
                        )
                        return False
                    except InferenceError as exc:
                        if is_retryable_upstream_failure(exc):
                            # Transient provider fault: the coordinator's
                            # upstream backoff owns this retry, and the receipt
                            # stays `planned` so the resume re-enters here.
                            raise
                        # Max turns, SDK error, or a request/contract rejection
                        # leaves exactly the state an invalid edit does: a
                        # consumed reservation and possibly half-repaired
                        # source.  Degrade identically rather than parking the
                        # whole run — the candidate closes as a crash, which is
                        # what the debug analysis sibling already does for its
                        # own inference failures.
                        self._write_contract_diagnostic(
                            run_id,
                            stage="debug_repair",
                            error=exc,
                        )
                        self._write_phase_a_repair_status(
                            run_id, repair, "rejected"
                        )
                        return False
                else:
                    raise ArtifactError(
                        f"candidate {run_id} changed during repair without a completed edit receipt"
                    )
                repair = {
                    **repair,
                    "status": "edit_completed",
                    "candidate_revision_after": file_revision(candidate_path),
                }
                atomic_write_json(
                    candidate_dir / self.PHASE_A_REPAIR_RECEIPT,
                    repair,
                )
            contract = self.toolchain.lint_contract(candidate_path)
            repaired_structure = contract.get("candidate_structure_sha256")
            terminal_structure = terminal.get(
                "candidate_execution_revision", {}
            ).get("structure_sha256")
            if (
                not isinstance(repaired_structure, str)
                or not repaired_structure.startswith("sha256:")
                or repaired_structure == terminal_structure
            ):
                self._write_phase_a_repair_status(run_id, repair, "rejected")
                return False

        if action.parents:
            self.toolchain.build_inheritance(candidate_path, configs_path)
        self.toolchain.check_search_space(candidate_path, space_path, configs_path)
        try:
            self.preflight(action)
        except ValidationRejected:
            self._write_phase_a_repair_status(run_id, repair, "rejected")
            return False
        self._authorize_phase_a_retry(run_id, terminal)
        self._write_phase_a_repair_status(run_id, repair, "completed")
        return True

    def _close_budget_exhausted(self, run_id: str, report_path: Path) -> CandidateOutcome:
        attempts = self._objective_attempts(run_id)
        if attempts == 0:
            record = self.toolchain.resolve_unevaluated(
                self.identity.run_dir, self.identity.task_name, run_id
            )
            return CandidateOutcome(status=str(record.get("status", "unevaluated")))
        return self._close_crash(run_id, report_path)

    def _close_crash(self, run_id: str, report_path: Path) -> CandidateOutcome:
        if self._objective_attempts(run_id) == 0:
            raise CandidateBuildError(
                f"candidate {run_id} failed before any objective reservation; "
                "refusing to record an experimental crash"
            )
        if report_path.is_file():
            report = _read_json(report_path)
            phase_a = report.get("phase_a") if isinstance(report, dict) else None
            status = phase_a.get("status") if isinstance(phase_a, dict) else None
            if status == "ok":
                self.toolchain.project_screening_report(
                    self.identity.run_dir, run_id, report_path
                )
            elif status not in {"crashed", "preflight_failed", "budget_exhausted"}:
                raise ArtifactError(
                    f"cannot close candidate {run_id} from malformed Phase A "
                    f"status {status!r}: {report_path}"
                )
        record = self.toolchain.record_crash(self.identity.run_dir, run_id)
        return CandidateOutcome(status=str(record.get("status", "crash")))

    def _objective_attempts(self, run_id: str) -> int:
        status = self.toolchain.budget_status(self.identity.run_dir)
        rows = status.get("per_candidate") if isinstance(status, dict) else None
        if not isinstance(rows, list):
            raise ArtifactError("objective budget status lacks per_candidate rows")
        for row in rows:
            if not isinstance(row, dict) or row.get("run_id") != run_id:
                continue
            attempts = row.get("evals")
            if (
                not isinstance(attempts, int)
                or isinstance(attempts, bool)
                or attempts < 0
            ):
                raise ArtifactError(
                    f"objective budget status has invalid count for {run_id}"
                )
            return attempts
        return 0

    def _validate_authored_contract(
        self,
        candidate_path: Path,
        configs_path: Path,
        space_path: Path,
        k: int,
    ) -> dict[str, Any]:
        self._validate_python(candidate_path)
        lint = self.toolchain.lint_schema(candidate_path)
        configs = _read_json(configs_path)
        space = _read_json(space_path)
        if not isinstance(configs, list) or len(configs) != k:
            raise ValueError(f"_warm_configs.json must contain exactly {k} configs")
        if not all(isinstance(item, dict) for item in configs):
            raise ValueError("every warm config must be an object")
        if not isinstance(space, dict) or not space:
            raise ValueError("_search_space.json must be a non-empty object")
        return lint

    def _validate_tuning_schema(self, candidate_path: Path) -> dict[str, Any]:
        self._validate_python(candidate_path)
        return self.toolchain.lint_schema(candidate_path)

    def _completed_edit_matches(
        self,
        *,
        purpose: str,
        output_path: Path,
        immutable_input_paths: tuple[Path, ...],
    ) -> bool:
        journal = getattr(self.models, "journal", None)
        model = getattr(self.models, "model", None)
        if journal is None or not isinstance(model, str):
            return False
        return bool(
            journal.completed_output_matches(
                purpose=purpose,
                schema_version=1,
                model=model,
                output_path=output_path,
                immutable_input_paths=immutable_input_paths,
            )
        )

    @staticmethod
    def _read_implementation_state(
        state_path: Path,
        *,
        run_id: str,
        context_revision: str,
        source_revision: str,
        candidate_path: Path,
        draft_path: Path,
    ) -> dict[str, Any] | None:
        if not state_path.exists():
            return None
        state = _read_json(state_path)
        if not isinstance(state, dict):
            raise ArtifactError("implementation authoring state must be an object")
        if (
            state.get("schema_version") != 1
            or state.get("kind") != "candidate_implementation_authoring"
            or state.get("run_id") != run_id
        ):
            raise ArtifactError("implementation authoring state has invalid identity")
        if state.get("context_revision") != context_revision:
            raise ArtifactError("candidate authoring inputs changed after admission")
        attempts = state.get("attempts_admitted")
        status = state.get("status")
        if (
            not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or not 0 <= attempts <= MAX_IMPLEMENTATION_ATTEMPTS
            or status
            not in {
                "ready",
                "started",
                "rejected",
                "infrastructure_failed",
                "publishing",
                "completed",
            }
            or not draft_path.is_file()
        ):
            raise ArtifactError("implementation authoring state is malformed")
        if status in {"ready", "rejected"} and state.get(
            "draft_revision"
        ) != file_revision(draft_path):
            raise ArtifactError("implementation draft changed outside authoring")
        if status in {"publishing", "completed"}:
            expected = state.get("expected_candidate_revision")
            if not isinstance(expected, str) or file_revision(draft_path) != expected:
                raise ArtifactError("implementation publish intent is stale")
            if candidate_path.is_file() and file_revision(candidate_path) == expected:
                return state
        if state.get("source_candidate_revision") != source_revision:
            raise ArtifactError("canonical candidate changed during implementation authoring")
        return state

    def _publish_implementation(
        self,
        run_id: str,
        *,
        candidate_path: Path,
        draft_path: Path,
        state_path: Path,
        state: dict[str, Any],
    ) -> Path:
        if state.get("status") != "publishing":
            raise ArtifactError("implementation publish requires a durable intent")
        expected = state.get("expected_candidate_revision")
        if not isinstance(expected, str) or file_revision(draft_path) != expected:
            raise ArtifactError("implementation draft changed after validation")
        source_is_current = (
            paths_revision((candidate_path,))
            == state.get("source_candidate_revision")
        )
        candidate_is_published = (
            candidate_path.is_file() and file_revision(candidate_path) == expected
        )
        if not source_is_current and not candidate_is_published:
            raise ArtifactError("canonical candidate changed during implementation publish")
        self._validate_python(draft_path)
        if not candidate_is_published:
            atomic_write_text(
                candidate_path,
                draft_path.read_text(encoding="utf-8"),
            )
        self._validate_python(candidate_path)
        if file_revision(candidate_path) != expected:
            raise ArtifactError("published implementation has an unexpected revision")
        context_revision = state.get("context_revision")
        if not isinstance(context_revision, str):
            raise ArtifactError("implementation publish intent lacks context revision")
        self._write_implementation_receipt(
            run_id,
            candidate_path,
            "agent_sdk",
            input_revision=context_revision,
        )
        atomic_write_json(
            state_path,
            {
                **state,
                "status": "completed",
                "candidate_revision": expected,
            },
        )
        return candidate_path

    def _provided_defaults(
        self,
        run_id: str,
        candidate_path: Path,
    ) -> dict[str, Any]:
        try:
            return self.toolchain.provided_baseline_defaults(candidate_path)
        except ValidationRejected as exc:
            raise CandidateBuildError(
                f"provided baseline defaults failed for {run_id}: {exc}"
            ) from exc

    @staticmethod
    def _provided_tuning_values(
        defaults: dict[str, Any],
        schema_lint: dict[str, Any],
    ) -> TuningValues:
        keys = schema_lint.get("keys")
        kinds = schema_lint.get("kinds")
        float_log = schema_lint.get("float_log")
        if not (
            isinstance(keys, list)
            and all(isinstance(key, str) and key for key in keys)
            and isinstance(kinds, dict)
            and set(defaults) == set(keys) == set(kinds)
        ):
            raise CandidateBuildError(
                "provided defaults do not match the validated PARAM_SCHEMA"
            )
        # lint-schema is the authoritative reader of PARAM_SCHEMA log modes.
        # A lint receipt without per-key float log modes is stale; rejecting it
        # is safer than silently rendering log floats as linear.
        if not isinstance(float_log, dict) or not all(
            isinstance(float_log.get(key), bool)
            for key, kind in kinds.items()
            if kind == "float"
        ):
            raise CandidateBuildError(
                "validated PARAM_SCHEMA receipt lacks float log modes"
            )
        space: dict[str, list[Any]] = {}
        for key in keys:
            value = defaults[key]
            kind = kinds[key]
            _validate_json_primitive(value, where=f"DEFAULT_PARAMS.{key}")
            if kind == "int" and type(value) is int:
                space[key] = ["int", value, value]
            elif kind == "float" and _finite_number(value):
                space[key] = (
                    ["float", value, value, "log"]
                    if float_log.get(key)
                    else ["float", value, value]
                )
            elif kind == "categorical":
                space[key] = ["categorical", [value]]
            else:
                raise CandidateBuildError(
                    f"provided default {key!r} is incompatible with kind {kind!r}"
                )
        return TuningValues(warm_configs=[defaults], search_space=space)

    @staticmethod
    def _read_tuning_schema_state(
        state_path: Path,
        *,
        run_id: str,
        context_revision: str,
        source_revision: str,
        candidate_path: Path,
        draft_path: Path,
    ) -> dict[str, Any] | None:
        if not state_path.exists():
            return None
        state = _read_json(state_path)
        if not isinstance(state, dict):
            raise ArtifactError("tuning-schema authoring state must be an object")
        if (
            state.get("schema_version") != 1
            or state.get("kind") != "tuning_schema_authoring"
            or state.get("run_id") != run_id
        ):
            raise ArtifactError("tuning-schema authoring state has invalid identity")
        if state.get("context_revision") != context_revision:
            raise ArtifactError("tuning-schema inputs changed after authoring admission")
        if state.get("source_candidate_revision") != source_revision:
            status = state.get("status")
            expected = state.get("expected_candidate_revision")
            if not (
                status in {"publishing", "completed"}
                and isinstance(expected, str)
                and draft_path.is_file()
                and file_revision(draft_path) == expected
                and candidate_path.is_file()
                and file_revision(candidate_path) == expected
            ):
                raise ArtifactError("canonical candidate changed during schema authoring")
        attempts = state.get("attempts_admitted")
        if (
            not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or not 0 <= attempts <= MAX_TUNING_SCHEMA_REPAIRS + 1
            or state.get("status")
            not in {
                "ready",
                "started",
                "rejected",
                "infrastructure_failed",
                "validating",
                "validator_failed",
                "publishing",
                "completed",
            }
            or not draft_path.is_file()
        ):
            raise ArtifactError("tuning-schema authoring state is malformed")
        status = state.get("status")
        if status in {"ready", "rejected"} and state.get(
            "draft_revision"
        ) != file_revision(draft_path):
            raise ArtifactError("tuning-schema draft changed outside authoring")
        if status in {"validating", "validator_failed", "publishing"}:
            expected = state.get("expected_candidate_revision")
            if not isinstance(expected, str) or file_revision(draft_path) != expected:
                raise ArtifactError("tuning-schema validation intent is stale")
        if status == "publishing" and not (
            isinstance(state.get("lint"), dict)
            and state["lint"].get("ok") is True
        ):
            raise ArtifactError("tuning-schema publish intent lacks validated lint")
        return state

    @staticmethod
    def _read_tuning_values_state(
        state_path: Path,
        *,
        run_id: str,
        k: int,
        input_revision: str,
    ) -> dict[str, Any] | None:
        if not state_path.exists():
            return None
        state = _read_json(state_path)
        if not isinstance(state, dict):
            raise ArtifactError("tuning-values authoring state must be an object")
        if (
            state.get("schema_version") != 1
            or state.get("kind") != "tuning_values_authoring"
            or state.get("run_id") != run_id
        ):
            raise ArtifactError("tuning-values authoring state has invalid identity")
        if state.get("k") != k or state.get("input_revision") != input_revision:
            raise ArtifactError("tuning-values inputs changed after authoring admission")
        attempts = state.get("attempts_admitted")
        transport_failures = state.get("transport_failures", 0)
        if (
            not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or not 0 <= attempts <= MAX_TUNING_VALUES_CORRECTIONS + 1
            or not isinstance(transport_failures, int)
            or isinstance(transport_failures, bool)
            or not 0 <= transport_failures <= MAX_TUNING_VALUES_TRANSPORT_FAILURES
            or state.get("status")
            not in {
                "ready",
                "started",
                "rejected",
                "request_failed",
                "infrastructure_failed",
                "validating",
                "validator_failed",
                "draft_validated",
                "validated",
                "finalizing",
                "completed",
            }
        ):
            raise ArtifactError("tuning-values authoring state is malformed")
        status = state.get("status")
        proposal_index = state.get("proposal_index")
        proposal_required = status in {
            "validating",
            "validator_failed",
            "draft_validated",
            "validated",
            "finalizing",
            "completed",
        } or (status == "rejected" and proposal_index is not None)
        if proposal_required:
            if (
                not isinstance(proposal_index, int)
                or isinstance(proposal_index, bool)
                or not 1
                <= proposal_index
                <= MAX_TUNING_VALUES_CORRECTIONS + 1
            ):
                raise ArtifactError("tuning-values state has invalid proposal index")
            configs_draft = (
                state_path.parent
                / f"_warm_configs_proposal_{proposal_index}.json"
            )
            space_draft = (
                state_path.parent
                / f"_search_space_proposal_{proposal_index}.json"
            )
            if not (
                configs_draft.is_file()
                and space_draft.is_file()
                and state.get("configs_draft_revision")
                == file_revision(configs_draft)
                and state.get("space_draft_revision")
                == file_revision(space_draft)
            ):
                raise ArtifactError("tuning-value proposal changed outside authoring")
            if status in {
                "draft_validated",
                "validated",
                "finalizing",
                "completed",
            }:
                validated_space = (
                    state_path.parent
                    / f"_search_space_validated_{proposal_index}.json"
                )
                if not (
                    validated_space.is_file()
                    and state.get("validated_space_revision")
                    == file_revision(validated_space)
                ):
                    raise ArtifactError(
                        "validated tuning space changed outside authoring"
                    )
        return state

    @staticmethod
    def _tuning_value_proposal_paths(
        candidate_dir: Path,
        proposal_index: Any,
    ) -> tuple[Path, Path]:
        if (
            not isinstance(proposal_index, int)
            or isinstance(proposal_index, bool)
            or not 1 <= proposal_index <= MAX_TUNING_VALUES_CORRECTIONS + 1
        ):
            raise ArtifactError("tuning-values state has invalid proposal index")
        return (
            candidate_dir / f"_warm_configs_proposal_{proposal_index}.json",
            candidate_dir / f"_search_space_proposal_{proposal_index}.json",
        )

    @staticmethod
    def _validated_tuning_space_path(
        candidate_dir: Path,
        proposal_index: int,
    ) -> Path:
        return candidate_dir / f"_search_space_validated_{proposal_index}.json"

    def _validate_tuning_value_proposal(
        self,
        action: RoundAction,
        *,
        candidate_path: Path,
        state_path: Path,
        state: dict[str, Any],
        configs_draft: Path,
        space_draft: Path,
    ) -> dict[str, Any]:
        if action.run_id is None:
            raise ValueError("cannot validate tuning values without run_id")
        if state.get("status") not in {"validating", "validator_failed"}:
            raise ArtifactError("tuning-value validation lacks a durable intent")
        if not (
            configs_draft.is_file()
            and space_draft.is_file()
            and state.get("configs_draft_revision")
            == file_revision(configs_draft)
            and state.get("space_draft_revision") == file_revision(space_draft)
        ):
            raise ArtifactError("tuning-value proposal changed before validation")
        validated_space = self._validated_tuning_space_path(
            candidate_path.parent,
            int(state["proposal_index"]),
        )
        # The authoritative helper widens ranges in place. Give it a disposable
        # Python-owned projection so the model proposal remains immutable and a
        # killed validator can restart from the same bytes.
        atomic_write_text(
            validated_space,
            space_draft.read_text(encoding="utf-8"),
        )
        try:
            self.toolchain.check_search_space(
                candidate_path,
                validated_space,
                configs_draft,
            )
        except ValidationRejected as exc:
            diagnostic = self._contract_diagnostic_payload(
                action.run_id,
                stage="values",
                error=exc,
            )
            rejected = {
                **state,
                "status": "rejected",
                "last_error_type": type(exc).__name__,
                "last_error": str(exc),
                "diagnostic": diagnostic,
            }
            atomic_write_json(state_path, rejected)
            self._write_contract_diagnostic(
                action.run_id,
                stage="values",
                payload=diagnostic,
            )
            raise
        except ToolFailure as exc:
            diagnostic = self._contract_diagnostic_payload(
                action.run_id,
                stage="values",
                error=exc,
            )
            failed = {
                **state,
                "status": "validator_failed",
                "last_error_type": type(exc).__name__,
                "last_error": str(exc),
                "diagnostic": diagnostic,
            }
            atomic_write_json(state_path, failed)
            self._write_contract_diagnostic(
                action.run_id,
                stage="values",
                payload=diagnostic,
            )
            raise
        validated = {
            **state,
            "status": "draft_validated",
            "validated_space_revision": file_revision(validated_space),
        }
        atomic_write_json(state_path, validated)
        return validated

    @staticmethod
    def _publish_tuning_value_drafts(
        *,
        state_path: Path,
        state: dict[str, Any],
        configs_draft: Path,
        space_draft: Path,
        validated_space: Path,
        configs_path: Path,
        space_path: Path,
    ) -> None:
        status = state.get("status")
        if status == "validated":
            if not (
                configs_path.is_file()
                and space_path.is_file()
                and state.get("configs_revision") == file_revision(configs_path)
                and state.get("space_revision") == file_revision(space_path)
            ):
                raise ArtifactError("validated tuning-value artifacts changed")
            return
        if status != "draft_validated":
            raise ArtifactError("tuning-value publication requires validated drafts")
        if not (
            configs_draft.is_file()
            and space_draft.is_file()
            and validated_space.is_file()
            and state.get("configs_draft_revision")
            == file_revision(configs_draft)
            and state.get("space_draft_revision") == file_revision(space_draft)
            and state.get("validated_space_revision")
            == file_revision(validated_space)
        ):
            raise ArtifactError("validated tuning-value drafts changed")
        atomic_write_text(
            configs_path,
            configs_draft.read_text(encoding="utf-8"),
        )
        atomic_write_text(
            space_path,
            validated_space.read_text(encoding="utf-8"),
        )
        atomic_write_json(
            state_path,
            {
                **state,
                "status": "validated",
                "configs_revision": file_revision(configs_path),
                "space_revision": file_revision(space_path),
            },
        )

    def _validate_and_publish_tuning_schema(
        self,
        action: RoundAction,
        *,
        candidate_path: Path,
        draft_path: Path,
        state_path: Path,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        if action.run_id is None:
            raise ValueError("cannot validate a schema without run_id")
        if state.get("status") not in {"validating", "validator_failed"}:
            raise ArtifactError("tuning-schema validation lacks a durable intent")
        expected = state.get("expected_candidate_revision")
        if not isinstance(expected, str) or file_revision(draft_path) != expected:
            raise ArtifactError("tuning-schema draft changed before validation")
        if state.get("context_revision") != paths_revision(
            self._candidate_authoring_context_paths(action)
        ):
            raise ArtifactError("tuning-schema inputs changed before validation")
        try:
            lint = self._validate_tuning_schema(draft_path)
        except ValidationRejected as exc:
            diagnostic = self._contract_diagnostic_payload(
                action.run_id,
                stage="schema",
                error=exc,
            )
            rejected = {
                **state,
                "status": "rejected",
                "draft_revision": file_revision(draft_path),
                "last_error_type": type(exc).__name__,
                "last_error": str(exc),
                "diagnostic": diagnostic,
            }
            atomic_write_json(state_path, rejected)
            self._write_contract_diagnostic(
                action.run_id,
                stage="schema",
                payload=diagnostic,
            )
            raise
        except ToolFailure as exc:
            diagnostic = self._contract_diagnostic_payload(
                action.run_id,
                stage="schema",
                error=exc,
            )
            failed = {
                **state,
                "status": "validator_failed",
                "draft_revision": file_revision(draft_path),
                "last_error_type": type(exc).__name__,
                "last_error": str(exc),
                "diagnostic": diagnostic,
            }
            atomic_write_json(state_path, failed)
            self._write_contract_diagnostic(
                action.run_id,
                stage="schema",
                payload=diagnostic,
            )
            raise
        publishing = {
            **state,
            "status": "publishing",
            "expected_candidate_revision": expected,
            "lint": lint,
        }
        atomic_write_json(state_path, publishing)
        return self._publish_tuning_schema(
            action,
            candidate_path=candidate_path,
            draft_path=draft_path,
            state_path=state_path,
            state=publishing,
        )

    def _publish_tuning_schema(
        self,
        action: RoundAction,
        *,
        candidate_path: Path,
        draft_path: Path,
        state_path: Path,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        if action.run_id is None:
            raise ValueError("cannot publish a schema without run_id")
        run_id = action.run_id
        if state.get("status") != "publishing":
            raise ArtifactError("tuning-schema publish requires a durable intent")
        if state.get("context_revision") != paths_revision(
            self._candidate_authoring_context_paths(action)
        ):
            raise ArtifactError("tuning-schema inputs changed before publication")
        expected = state.get("expected_candidate_revision")
        if not isinstance(expected, str) or file_revision(draft_path) != expected:
            raise ArtifactError("tuning-schema draft changed after validation")
        lint = state.get("lint")
        if not isinstance(lint, dict) or lint.get("ok") is not True:
            raise ArtifactError("tuning-schema publish intent lacks validated lint")
        source_is_current = (
            file_revision(candidate_path)
            == state.get("source_candidate_revision")
        )
        candidate_is_published = file_revision(candidate_path) == expected
        if not source_is_current and not candidate_is_published:
            raise ArtifactError("canonical candidate changed during schema publish")
        self._validate_python(draft_path)
        if not candidate_is_published:
            atomic_write_text(
                candidate_path,
                draft_path.read_text(encoding="utf-8"),
            )
        self._validate_python(candidate_path)
        if file_revision(candidate_path) != expected:
            raise ArtifactError("published tuning schema has an unexpected revision")
        self._write_tuning_schema_receipt(
            action,
            candidate_path,
            lint,
            source_candidate_revision=str(state["source_candidate_revision"]),
        )
        atomic_write_json(
            state_path,
            {
                **state,
                "status": "completed",
                "draft_revision": file_revision(draft_path),
                "candidate_revision": file_revision(candidate_path),
            },
        )
        return lint

    def _forward_complete_tuning_values(
        self,
        action: RoundAction,
        *,
        receipt: dict[str, Any],
        candidate_path: Path,
        configs_path: Path,
        space_path: Path,
    ) -> None:
        if action.run_id is None:
            raise ValueError("cannot finalize tuning values without run_id")
        source_revision = str(receipt["source_candidate_revision"])
        expected_revision = str(receipt["expected_candidate_revision"])
        current_revision = file_revision(candidate_path)
        if current_revision == source_revision:
            contract_draft = candidate_path.parent / self.TUNING_CONTRACT_DRAFT
            if not (
                contract_draft.is_file()
                and file_revision(contract_draft) == expected_revision
            ):
                raise ArtifactError("tuning-contract publish intent lost its draft")
            atomic_write_text(
                candidate_path,
                contract_draft.read_text(encoding="utf-8"),
            )
        elif current_revision != expected_revision:
            raise ArtifactError("canonical candidate changed during contract publication")
        if file_revision(candidate_path) != expected_revision:
            raise ArtifactError("published tuning contract has an unexpected revision")
        values_revision = paths_revision((configs_path, space_path))
        if action.parents:
            self.toolchain.build_inheritance(candidate_path, configs_path)
            if paths_revision((configs_path, space_path)) != values_revision:
                raise ArtifactError(
                    "parameter inheritance changed after contract rendering"
                )
        self._write_tuning_values_receipt(
            action,
            status="applied",
            candidate_path=candidate_path,
            configs_path=configs_path,
            space_path=space_path,
            source_candidate_revision=source_revision,
            expected_candidate_revision=expected_revision,
        )
        self.toolchain.lint_contract(
            candidate_path,
            require_base_params=False,
        )
        self._write_contract_receipt(
            action,
            candidate_path,
            configs_path,
            space_path,
        )
        self._mark_tuning_values_completed(action)

    def _finalize_validated_tuning_values(
        self,
        action: RoundAction,
        *,
        state_path: Path,
        candidate_path: Path,
        configs_path: Path,
        space_path: Path,
    ) -> None:
        if action.run_id is None:
            raise ValueError("cannot finalize tuning values without run_id")
        state = _read_json(state_path)
        if not isinstance(state, dict):
            raise ArtifactError("tuning-values authoring state must be an object")
        status = state.get("status")
        if status == "validated":
            if not (
                state.get("configs_revision") == file_revision(configs_path)
                and state.get("space_revision") == file_revision(space_path)
            ):
                raise ArtifactError(
                    "canonical tuning values changed before finalization"
                )
            state = {
                **state,
                "status": "finalizing",
                "finalization_candidate_revision": file_revision(candidate_path),
            }
            atomic_write_json(state_path, state)
        elif status != "finalizing":
            raise ArtifactError("tuning values are not ready for finalization")

        if state.get("finalization_candidate_revision") != file_revision(
            candidate_path
        ):
            raise ArtifactError("canonical candidate changed during finalization")
        if not (
            state.get("configs_revision") == file_revision(configs_path)
            and state.get("space_revision") == file_revision(space_path)
        ):
            proposal_index = int(state["proposal_index"])
            configs_draft, _ = self._tuning_value_proposal_paths(
                candidate_path.parent,
                proposal_index,
            )
            validated_space = self._validated_tuning_space_path(
                candidate_path.parent,
                proposal_index,
            )
            if not (
                state.get("configs_draft_revision")
                == file_revision(configs_draft)
                and state.get("validated_space_revision")
                == file_revision(validated_space)
            ):
                raise ArtifactError("validated tuning-value recovery inputs changed")
            atomic_write_text(
                configs_path,
                configs_draft.read_text(encoding="utf-8"),
            )
            atomic_write_text(
                space_path,
                validated_space.read_text(encoding="utf-8"),
            )
            if not (
                state.get("configs_revision") == file_revision(configs_path)
                and state.get("space_revision") == file_revision(space_path)
            ):
                raise ArtifactError("tuning values did not restore for finalization")

        self._finalize_contract(
            action,
            candidate_path=candidate_path,
            configs_path=configs_path,
            space_path=space_path,
            raw_space_validated=True,
        )
        completed = {
            **state,
            "status": "completed",
            "contract_receipt_revision": paths_revision(
                (candidate_path.parent / self.CONTRACT_RECEIPT,)
            ),
        }
        atomic_write_json(state_path, completed)

    def _mark_tuning_values_completed(self, action: RoundAction) -> None:
        if action.run_id is None:
            return
        candidate_dir = self.candidate_dir(action.run_id)
        state_path = candidate_dir / self.TUNING_VALUES_STATE
        if not state_path.exists():
            return
        state = _read_json(state_path)
        if not (
            isinstance(state, dict)
            and state.get("schema_version") == 1
            and state.get("kind") == "tuning_values_authoring"
            and state.get("run_id") == action.run_id
            and state.get("status") in {"finalizing", "completed"}
        ):
            raise ArtifactError("tuning-values completion state is inconsistent")
        receipt_revision = paths_revision(
            (candidate_dir / self.CONTRACT_RECEIPT,)
        )
        if state.get("status") == "completed":
            if state.get("contract_receipt_revision") != receipt_revision:
                raise ArtifactError("completed tuning contract receipt changed")
            return
        atomic_write_json(
            state_path,
            {
                **state,
                "status": "completed",
                "contract_receipt_revision": receipt_revision,
            },
        )

    def _finalize_contract(
        self,
        action: RoundAction,
        *,
        candidate_path: Path,
        configs_path: Path,
        space_path: Path,
        raw_space_validated: bool = False,
    ) -> None:
        if not raw_space_validated:
            self.toolchain.check_search_space(
                candidate_path,
                space_path,
                configs_path,
            )
        if action.parents:
            self.toolchain.build_inheritance(candidate_path, configs_path)
            self.toolchain.check_search_space(
                candidate_path,
                space_path,
                configs_path,
            )

        contract_draft = candidate_path.parent / self.TUNING_CONTRACT_DRAFT
        atomic_write_text(
            contract_draft,
            candidate_path.read_text(encoding="utf-8"),
        )
        self.toolchain.apply_search_space(contract_draft, space_path)
        self.toolchain.lint_contract(
            contract_draft,
            require_base_params=False,
        )
        values_revision = paths_revision((configs_path, space_path))
        source_revision = file_revision(candidate_path)
        expected_revision = file_revision(contract_draft)
        self._write_tuning_values_receipt(
            action,
            status="applying",
            candidate_path=candidate_path,
            configs_path=configs_path,
            space_path=space_path,
            source_candidate_revision=source_revision,
            expected_candidate_revision=expected_revision,
        )
        atomic_write_text(
            candidate_path,
            contract_draft.read_text(encoding="utf-8"),
        )
        if file_revision(candidate_path) != expected_revision:
            raise ArtifactError("published tuning contract has an unexpected revision")
        if action.parents:
            # apply_search_space changes train.py, so refresh the exact lineage
            # binding against the final authored candidate revision.
            self.toolchain.build_inheritance(candidate_path, configs_path)
            if paths_revision((configs_path, space_path)) != values_revision:
                raise ArtifactError(
                    "parameter inheritance changed after contract rendering"
                )
        self._write_tuning_values_receipt(
            action,
            status="applied",
            candidate_path=candidate_path,
            configs_path=configs_path,
            space_path=space_path,
            source_candidate_revision=source_revision,
            expected_candidate_revision=expected_revision,
        )
        self.toolchain.lint_contract(
            candidate_path,
            require_base_params=False,
        )
        self._write_contract_receipt(
            action,
            candidate_path,
            configs_path,
            space_path,
        )

    def _recover_authored_contract(
        self,
        action: RoundAction,
        *,
        candidate_path: Path,
        configs_path: Path,
        space_path: Path,
        k: int,
    ) -> None:
        configs = _read_json(configs_path)
        if not isinstance(configs, list) or len(configs) != k:
            raise ValueError(f"_warm_configs.json must contain exactly {k} configs")
        candidate_dir = candidate_path.parent
        self._frozen_tuning_lineage(
            action,
            base_input_paths=[
                candidate_path,
                candidate_dir / self.TUNING_SCHEMA_RECEIPT,
                *self._candidate_authoring_context_paths(action),
            ],
            state_exists=(candidate_dir / self.TUNING_VALUES_STATE).exists(),
        )
        try:
            self.toolchain.lint_contract(
                candidate_path,
                require_base_params=False,
            )
        except ValidationRejected:
            self._validate_authored_contract(candidate_path, configs_path, space_path, k)
        self._finalize_contract(
            action,
            candidate_path=candidate_path,
            configs_path=configs_path,
            space_path=space_path,
        )

    def implementation_is_ready(self, action: RoundAction) -> bool:
        if action.run_id is None:
            return False
        candidate_dir = self.candidate_dir(action.run_id)
        candidate_path = candidate_dir / "train.py"
        receipt_path = candidate_dir / self.IMPLEMENTATION_RECEIPT
        try:
            receipt = _read_json(receipt_path)
        except ValueError:
            return False
        context_revision = paths_revision(
            self._candidate_authoring_context_paths(action)
        )
        if not (
            isinstance(receipt, dict)
            and receipt.get("schema_version") == 2
            and receipt.get("run_id") == action.run_id
            and receipt.get("input_revision") == context_revision
            and isinstance(receipt.get("candidate_revision"), str)
            and candidate_path.is_file()
        ):
            return False
        if self.is_provided_baseline(action.run_id) and not self.materialization_is_ready(
            action
        ):
            return False
        implementation_revision = receipt["candidate_revision"]
        if implementation_revision == file_revision(candidate_path):
            return True

        # A revision-bound successor publish may legitimately replace the
        # implementation before its own ready receipt is durable. Keep the
        # coordinator in the contract phase so that intent can forward-complete
        # instead of rerunning the candidate writer on the successor output.
        draft_path = candidate_dir / self.TUNING_SCHEMA_DRAFT
        state_path = candidate_dir / self.TUNING_SCHEMA_STATE
        try:
            state = _read_json(state_path)
        except ValueError:
            return False
        expected = state.get("expected_candidate_revision") if isinstance(state, dict) else None
        return bool(
            isinstance(state, dict)
            and state.get("schema_version") == 1
            and state.get("kind") == "tuning_schema_authoring"
            and state.get("run_id") == action.run_id
            and state.get("status") in {"publishing", "completed"}
            and state.get("context_revision") == context_revision
            and state.get("source_candidate_revision") == implementation_revision
            and isinstance(expected, str)
            and draft_path.is_file()
            and file_revision(draft_path) == expected
            and file_revision(candidate_path) == expected
        )

    def _tuning_schema_receipt(
        self, action: RoundAction
    ) -> dict[str, Any] | None:
        if action.run_id is None:
            return None
        candidate_dir = self.candidate_dir(action.run_id)
        candidate_path = candidate_dir / "train.py"
        try:
            receipt = _read_json(candidate_dir / self.TUNING_SCHEMA_RECEIPT)
        except ValueError:
            return None
        lint = receipt.get("lint") if isinstance(receipt, dict) else None
        if not (
            isinstance(receipt, dict)
            and receipt.get("schema_version") == 2
            and receipt.get("run_id") == action.run_id
            and candidate_path.is_file()
            and receipt.get("candidate_revision") == file_revision(candidate_path)
            and isinstance(receipt.get("source_candidate_revision"), str)
            and receipt.get("input_revision")
            == paths_revision(self._candidate_authoring_context_paths(action))
            and isinstance(lint, dict)
            and lint.get("ok") is True
        ):
            return None
        return receipt

    def tuning_schema_is_ready(self, action: RoundAction) -> bool:
        return self._tuning_schema_receipt(action) is not None

    def _tuning_values_receipt(
        self, action: RoundAction
    ) -> dict[str, Any] | None:
        if action.run_id is None:
            return None
        candidate_dir = self.candidate_dir(action.run_id)
        candidate_path = candidate_dir / "train.py"
        configs_path = candidate_dir / "_warm_configs.json"
        space_path = candidate_dir / "_search_space.json"
        schema_receipt_path = candidate_dir / self.TUNING_SCHEMA_RECEIPT
        lineage_path = candidate_dir / self.TUNING_LINEAGE_EVIDENCE
        transfer_path = candidate_dir / "_parameter_transfer.json"
        try:
            receipt = _read_json(candidate_dir / self.TUNING_VALUES_RECEIPT)
        except ValueError:
            return None
        if not isinstance(receipt, dict):
            return None
        status = receipt.get("status")
        source_revision = receipt.get("source_candidate_revision")
        expected_revision = receipt.get("expected_candidate_revision")
        if not (
            receipt.get("schema_version") == 2
            and receipt.get("kind") == "tuning_values_ready"
            and receipt.get("run_id") == action.run_id
            and receipt.get("input_revision")
            == paths_revision(self._candidate_authoring_context_paths(action))
            and status in {"applying", "applied"}
            and isinstance(source_revision, str)
            and isinstance(expected_revision, str)
            and candidate_path.is_file()
            and configs_path.is_file()
            and space_path.is_file()
            and schema_receipt_path.is_file()
            and lineage_path.is_file()
            and receipt.get("configs_revision") == file_revision(configs_path)
            and receipt.get("space_revision") == file_revision(space_path)
            and receipt.get("schema_receipt_revision")
            == paths_revision((schema_receipt_path,))
            and receipt.get("lineage_evidence_revision")
            == file_revision(lineage_path)
            and receipt.get("transfer_revision")
            == paths_revision((transfer_path,))
        ):
            return None
        candidate_revision = file_revision(candidate_path)
        if status == "applying":
            return (
                receipt
                if candidate_revision in {source_revision, expected_revision}
                else None
            )
        return receipt if candidate_revision == expected_revision else None

    def tuning_values_are_ready(self, action: RoundAction) -> bool:
        return self._tuning_values_receipt(action) is not None

    def contract_is_ready(self, action: RoundAction) -> bool:
        if action.run_id is None:
            return False
        candidate_dir = self.candidate_dir(action.run_id)
        values_receipt = self._tuning_values_receipt(action)
        paths = {
            "candidate_revision": candidate_dir / "train.py",
            "configs_revision": candidate_dir / "_warm_configs.json",
            "space_revision": candidate_dir / "_search_space.json",
        }
        try:
            receipt = _read_json(candidate_dir / self.CONTRACT_RECEIPT)
        except ValueError:
            return False
        return (
            isinstance(receipt, dict)
            and isinstance(values_receipt, dict)
            and values_receipt.get("status") == "applied"
            and receipt.get("schema_version") == 2
            and receipt.get("run_id") == action.run_id
            and receipt.get("input_revision")
            == paths_revision(self._candidate_authoring_context_paths(action))
            and receipt.get("values_receipt_revision")
            == paths_revision((candidate_dir / self.TUNING_VALUES_RECEIPT,))
            and all(
                path.is_file() and receipt.get(key) == file_revision(path)
                for key, path in paths.items()
            )
        )

    def preflight_is_ready(self, action: RoundAction) -> bool:
        if action.run_id is None:
            return False
        receipt_path = self.candidate_dir(action.run_id) / self.PREFLIGHT_RECEIPT
        try:
            receipt = _read_json(receipt_path)
        except ValueError:
            return False
        if not isinstance(receipt, dict):
            return False
        try:
            input_revision = paths_revision(
                self._preflight_input_paths(action.run_id)
            )
        except (OSError, ValueError):
            return False
        return (
            receipt.get("schema_version") == 1
            and receipt.get("run_id") == action.run_id
            and receipt.get("input_revision") == input_revision
            and receipt.get("objective_calls") == 0
            and receipt.get("status") in {"ok", "not_declared"}
        )

    def evaluation_has_started(self, action: RoundAction) -> bool:
        if action.run_id is None:
            return False
        report_path = self.candidate_dir(action.run_id) / "tune_report.json"
        if not report_path.exists():
            return False
        report = _read_json(report_path)
        if not isinstance(report, dict) or not isinstance(report.get("phase_a"), dict):
            raise ArtifactError(f"malformed Phase A report: {report_path}")
        return True

    def _write_implementation_receipt(
        self,
        run_id: str,
        candidate_path: Path,
        source: str,
        *,
        input_revision: str,
    ) -> None:
        if not isinstance(input_revision, str) or not input_revision:
            raise ValueError("implementation receipt requires an input revision")
        atomic_write_json(
            candidate_path.parent / self.IMPLEMENTATION_RECEIPT,
            {
                "schema_version": 2,
                "run_id": run_id,
                "source": source,
                "candidate_revision": file_revision(candidate_path),
                "input_revision": input_revision,
            },
        )

    def _write_tuning_schema_receipt(
        self,
        action: RoundAction,
        candidate_path: Path,
        lint: dict[str, Any],
        *,
        source_candidate_revision: str,
    ) -> None:
        if action.run_id is None:
            raise ValueError("cannot persist a schema receipt without run_id")
        if not isinstance(lint, dict) or lint.get("ok") is not True:
            raise ValueError("cannot persist an invalid tuning-schema receipt")
        context_revision = paths_revision(
            self._candidate_authoring_context_paths(action)
        )
        atomic_write_json(
            candidate_path.parent / self.TUNING_SCHEMA_RECEIPT,
            {
                "schema_version": 2,
                "run_id": action.run_id,
                "candidate_revision": file_revision(candidate_path),
                "source_candidate_revision": source_candidate_revision,
                "input_revision": context_revision,
                "lint": lint,
            },
        )

    def _write_tuning_values_receipt(
        self,
        action: RoundAction,
        *,
        status: str,
        candidate_path: Path,
        configs_path: Path,
        space_path: Path,
        source_candidate_revision: str,
        expected_candidate_revision: str,
    ) -> None:
        if action.run_id is None:
            raise ValueError("cannot persist tuning values without run_id")
        if status not in {"applying", "applied"}:
            raise ValueError(f"invalid tuning-values status: {status}")
        candidate_dir = candidate_path.parent
        schema_receipt_path = candidate_dir / self.TUNING_SCHEMA_RECEIPT
        lineage_path = candidate_dir / self.TUNING_LINEAGE_EVIDENCE
        if not all(
            path.is_file()
            for path in (
                candidate_path,
                configs_path,
                space_path,
                schema_receipt_path,
                lineage_path,
            )
        ):
            raise ArtifactError("cannot persist incomplete tuning-values progress")
        atomic_write_json(
            candidate_dir / self.TUNING_VALUES_RECEIPT,
            {
                "schema_version": 2,
                "kind": "tuning_values_ready",
                "run_id": action.run_id,
                "status": status,
                "input_revision": paths_revision(
                    self._candidate_authoring_context_paths(action)
                ),
                "source_candidate_revision": source_candidate_revision,
                "expected_candidate_revision": expected_candidate_revision,
                "configs_revision": file_revision(configs_path),
                "space_revision": file_revision(space_path),
                "schema_receipt_revision": paths_revision(
                    (schema_receipt_path,)
                ),
                "lineage_evidence_revision": file_revision(lineage_path),
                "transfer_revision": paths_revision(
                    (candidate_dir / "_parameter_transfer.json",)
                ),
            },
        )

    def _write_contract_receipt(
        self,
        action: RoundAction,
        candidate_path: Path,
        configs_path: Path,
        space_path: Path,
    ) -> None:
        if action.run_id is None:
            raise ValueError("cannot persist a contract without run_id")
        atomic_write_json(
            candidate_path.parent / self.CONTRACT_RECEIPT,
            {
                "schema_version": 2,
                "run_id": action.run_id,
                "input_revision": paths_revision(
                    self._candidate_authoring_context_paths(action)
                ),
                "candidate_revision": file_revision(candidate_path),
                "configs_revision": file_revision(configs_path),
                "space_revision": file_revision(space_path),
                "values_receipt_revision": paths_revision(
                    (candidate_path.parent / self.TUNING_VALUES_RECEIPT,)
                ),
            },
        )

    def _tuner_integer(self, key: str, default: int, *, minimum: int) -> int:
        cfg = _read_json(self.identity.run_dir / "framework_cfg.json")
        section = cfg.get("tuner", {}) if isinstance(cfg, dict) else {}
        value = section.get(key, default) if isinstance(section, dict) else default
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"framework_cfg tuner.{key} must be an integer >= {minimum}")
        return value

    def _provided_entrypoint_is_current(
        self,
        candidate_dir: Path,
        *,
        entrypoint: str,
        source: dict[str, Any],
    ) -> bool:
        seed = self.task_config.get("seed")
        if not isinstance(seed, dict):
            return False
        seed_entrypoint = seed.get("entrypoint", "train.py")
        provided = seed.get("provided")
        if (
            seed_entrypoint != entrypoint
            or not isinstance(provided, list)
            or not provided
            or any(not isinstance(item, str) for item in provided)
        ):
            return False
        repo_root = self.identity.repo_root.resolve()
        task_dir = (repo_root / "tasks" / self.identity.task_name).resolve()
        task_entrypoint = (task_dir / entrypoint).resolve()
        try:
            task_entrypoint.relative_to(task_dir)
            expected_path = task_entrypoint.relative_to(repo_root).as_posix()
        except ValueError:
            return False
        if not task_entrypoint.is_file():
            return False
        provided_paths = {
            candidate.resolve()
            for item in provided
            for candidate in (
                (Path(item),)
                if Path(item).is_absolute()
                else (task_dir / item, repo_root / item)
            )
        }
        if task_entrypoint not in provided_paths:
            return False
        expected_revision = file_revision(task_entrypoint)
        candidate_path = candidate_dir / entrypoint
        return (
            source.get("path") == expected_path
            and source.get("sha256") == expected_revision
            and candidate_path.is_file()
            and file_revision(candidate_path) == expected_revision
        )

    def _preflight_input_paths(self, run_id: str) -> tuple[Path, ...]:
        candidate_dir = self.candidate_dir(run_id)
        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        return (
            candidate_dir / "train.py",
            candidate_dir / "prepare.py",
            candidate_dir / "_warm_configs.json",
            candidate_dir / "_search_space.json",
            candidate_dir / "_parameter_transfer.json",
            candidate_dir / self.CONTRACT_RECEIPT,
            task_dir / "task.toml",
            self.identity.run_dir / "framework_cfg.json",
        )

    def _validate_authored_python(self, path: Path) -> Path:
        try:
            return self._validate_python(path)
        except (ValueError, SyntaxError) as exc:
            raise SourceValidationRejected(str(exc)) from exc

    def _validate_python(self, path: Path) -> Path:
        """Strict authored-source gate shared with the agent's Bash boundary.

        Delegates to tools/validate_candidate_source.py so the post-call gate
        and the editor's allow-listed command are one deterministic code path.
        """
        try:
            self.toolchain.validate_candidate_source(path)
        except ValidationRejected as exc:
            raise ValueError(_validation_error_messages(exc)) from exc
        return path
