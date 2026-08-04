"""Agentic background research behind a frozen, validated artifact boundary."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .artifacts import (
    atomic_write_json,
    atomic_write_text,
    file_revision,
    paths_revision,
)
from .llm import AgentEditSpec, InferenceError, ModelGateway
from .models import RunIdentity
from .process import ProcessResult
from .prompts import BACKGROUND_REGISTRY_SYSTEM, BACKGROUND_RETRIEVAL_SYSTEM
from .toolchain import ToolFailure, Toolchain, ValidationRejected
from .upstream import is_retryable_upstream_failure


MAX_BACKGROUND_REPAIR_ATTEMPTS = 5
MAX_BACKGROUND_REPAIR_DIAGNOSTIC_ERRORS = 256
# Retrieval gets its own admission because the bundled 48-turn research call
# died mid-repair: the 2026-08-04 autoresearch-baseline transcript needed ~30
# retrieval tool calls, so 40 turns covers searches plus import/validate
# iterations without also funding registry authoring.
BACKGROUND_RETRIEVAL_MAX_TURNS = 40
# Registry authoring with a frozen retrieval manifest finished in ~3 minutes in
# the observed 2026-08-04 run; 48 turns is ample for writing the registry and
# iterating on the allow-listed validators.
BACKGROUND_REGISTRY_MAX_TURNS = 48

BACKGROUND_STAGES = ("retrieval", "registry")


class BackgroundArtifactError(ValueError):
    """A frozen background artifact is inconsistent with deterministic inputs."""


class BackgroundBuilder:
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

    def ensure(self) -> None:
        run_dir = self.identity.run_dir
        induced = self._dimension_strategy() == "llm_induced"
        registry_artifacts = [run_dir / "background.md"]
        if induced:
            registry_artifacts.append(run_dir / "dimension_catalog.json")
        if self._has_provided_baseline():
            registry_artifacts.append(run_dir / "baseline_mechanisms.json")
        retrieval_manifest_path = run_dir / "background_retrieval.json"

        # Stage derivation is artifact-driven: a validated canonical manifest
        # is the retrieval stage's postcondition, so its presence places the
        # run in the registry stage regardless of what the state file records.
        initial_error: ValidationRejected | BackgroundArtifactError | None = None
        manifest_valid = False
        if retrieval_manifest_path.is_file():
            try:
                self.toolchain.validate_background_retrieval(run_dir)
                manifest_valid = True
            except ValidationRejected as exc:
                if self.identity.ledger_path.exists():
                    raise
                initial_error = exc
        elif self.identity.ledger_path.exists():
            raise ValueError(
                "frozen run is missing background artifacts: "
                "background_retrieval.json"
            )
        if manifest_valid:
            if all(path.is_file() for path in registry_artifacts):
                try:
                    self._validate_artifacts(induced=induced)
                    return
                except (ValidationRejected, BackgroundArtifactError) as exc:
                    if self.identity.ledger_path.exists():
                        raise
                    initial_error = exc
            elif self.identity.ledger_path.exists():
                missing = ", ".join(
                    path.name for path in registry_artifacts if not path.is_file()
                )
                raise ValueError(
                    f"frozen run is missing background artifacts: {missing}"
                )
        stage = "registry" if manifest_valid else "retrieval"

        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        docs_dir = self.identity.repo_root / "docs"
        contracts_dir = self.identity.repo_root / "contracts"
        fixed_input_paths = [
            task_dir / "TASK.md",
            task_dir / "task.toml",
            task_dir / "prepare.py",
            run_dir / "framework_cfg.json",
            docs_dir / "background-research.md",
            docs_dir / "dimension-induction.md",
            docs_dir / "agent-resources/background-researcher/background-template.md",
            docs_dir / "agent-resources/background-researcher/retrieval.md",
            docs_dir / "agent-resources/background-researcher/evidence-registry.md",
            contracts_dir / "semantic-dimensions-v1.json",
        ]
        seed_entrypoint = self._seed_entrypoint()
        if seed_entrypoint is not None:
            fixed_input_paths.append(seed_entrypoint)
        authoring_state_path = (
            run_dir / ".orchestrator" / "background_authoring.json"
        )
        input_revision = paths_revision(fixed_input_paths)
        state = self._read_authoring_state(
            authoring_state_path,
            input_revision=input_revision,
        )
        if state is None:
            fresh_counters = {
                "attempts_admitted": 0,
                "initial_admitted": False,
                "repairs_admitted": 0,
            }
            state = {
                "schema_version": 2,
                "kind": "background_authoring",
                "status": "rejected" if initial_error is not None else "ready",
                "stage": stage,
                "input_revision": input_revision,
                "stages": {
                    "retrieval": dict(fresh_counters),
                    "registry": dict(fresh_counters),
                },
                "purpose": None,
                **self._last_error_fields(initial_error),
            }
            if initial_error is not None:
                # Pre-existing invalid artifacts count as the stage's consumed
                # initial attempt; the first model call is a repair.
                state["stages"][stage] = {
                    "attempts_admitted": 1,
                    "initial_admitted": True,
                    "repairs_admitted": 0,
                }
            atomic_write_json(authoring_state_path, state)
        elif state["status"] == "completed":
            raise BackgroundArtifactError(
                "completed background authoring state has invalid artifacts"
            )
        else:
            state = self._reconcile_stage(
                state,
                stage,
                authoring_state_path=authoring_state_path,
            )
            stage_counters = state["stages"][stage]
            if initial_error is not None and not stage_counters["initial_admitted"]:
                state = {
                    **state,
                    "status": "rejected",
                    "stages": {
                        **state["stages"],
                        stage: {
                            **stage_counters,
                            "attempts_admitted": max(
                                1, int(stage_counters["attempts_admitted"])
                            ),
                            "initial_admitted": True,
                        },
                    },
                    **self._last_error_fields(initial_error),
                }
                atomic_write_json(authoring_state_path, state)
        last_error: BaseException | None = initial_error
        if last_error is None:
            last_error = self._resumed_error(state)
        if state["stage"] == "retrieval":
            state = self._run_stage(
                "retrieval",
                state=state,
                authoring_state_path=authoring_state_path,
                last_error=last_error,
                induced=induced,
                fixed_input_paths=fixed_input_paths,
            )
            last_error = None
        catalog_receipt = (
            None if induced else self.toolchain.background_catalog_receipt()
        )
        self._run_stage(
            "registry",
            state=state,
            authoring_state_path=authoring_state_path,
            last_error=last_error,
            induced=induced,
            fixed_input_paths=fixed_input_paths,
            catalog_receipt=catalog_receipt,
        )

    @staticmethod
    def _reconcile_stage(
        state: dict[str, Any],
        derived_stage: str,
        *,
        authoring_state_path: Path,
    ) -> dict[str, Any]:
        """Align the recorded stage with the artifact-derived one.

        The only legal divergence is a recorded retrieval stage beside a valid
        canonical manifest: the import side effect committed but the stage
        transition was not persisted, so forward-complete from the artifact.
        A registry-stage state without a valid manifest is contradictory —
        the registry agent never writes the manifest, so its loss is external.
        """
        recorded = state["stage"]
        if recorded == derived_stage:
            return state
        if derived_stage == "registry":
            state = {
                **state,
                "stage": "registry",
                "status": "ready",
                "purpose": None,
                **BackgroundBuilder._last_error_fields(None),
            }
            atomic_write_json(authoring_state_path, state)
            return state
        raise BackgroundArtifactError(
            "registry-stage background authoring state lacks a valid canonical "
            "retrieval manifest"
        )

    def _run_stage(
        self,
        stage: str,
        *,
        state: dict[str, Any],
        authoring_state_path: Path,
        last_error: BaseException | None,
        induced: bool,
        fixed_input_paths: list[Path],
        catalog_receipt: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        while (
            int(state["stages"][stage]["attempts_admitted"])
            < MAX_BACKGROUND_REPAIR_ATTEMPTS + 1
        ):
            counters = state["stages"][stage]
            resuming_started = state["status"] == "started"
            retry_purpose = state.get("purpose") if resuming_started else None
            if isinstance(retry_purpose, str) and retry_purpose:
                purpose = retry_purpose
            elif not counters["initial_admitted"]:
                purpose = f"background_{stage}"
                counters = {**counters, "initial_admitted": True}
            else:
                repairs_admitted = int(counters["repairs_admitted"])
                if repairs_admitted >= MAX_BACKGROUND_REPAIR_ATTEMPTS:
                    break
                next_repair = repairs_admitted + 1
                purpose = f"background_{stage}:repair:{next_repair}"
                counters = {**counters, "repairs_admitted": next_repair}
            diagnostic_path = (
                self._write_repair_diagnostic(last_error)
                if last_error is not None and not isinstance(last_error, InferenceError)
                else None
            )
            spec = self._stage_edit_spec(
                stage,
                purpose=purpose,
                last_error=last_error,
                diagnostic_path=diagnostic_path,
                induced=induced,
                fixed_input_paths=fixed_input_paths,
                catalog_receipt=catalog_receipt,
            )
            try:
                # A resumed "started" state re-drives an attempt whose
                # admission was already counted; only a fresh purpose consumes
                # a new slot.
                counters = {
                    **counters,
                    "attempts_admitted": int(counters["attempts_admitted"])
                    + (0 if resuming_started else 1),
                }
                state = {
                    **state,
                    "status": "started",
                    "stages": {**state["stages"], stage: counters},
                    "purpose": purpose,
                    **self._last_error_fields(last_error),
                }
                atomic_write_json(authoring_state_path, state)
                self.models.edit(
                    spec,
                    validate=lambda: self._finalize_stage(
                        stage,
                        induced=induced,
                        catalog_receipt=catalog_receipt,
                    ),
                )
                if stage == "retrieval":
                    # Persist the stage transition before registry authoring.
                    # A crash in this window is forward-completed from the
                    # canonical manifest by _reconcile_stage on the next run.
                    state = {
                        **state,
                        "stage": "registry",
                        "status": "ready",
                        "purpose": None,
                        **self._last_error_fields(None),
                    }
                    atomic_write_json(authoring_state_path, state)
                    return state
                state = {
                    **state,
                    "status": "completed",
                    **self._last_error_fields(None),
                }
                atomic_write_json(authoring_state_path, state)
                return state
            except (ValidationRejected, BackgroundArtifactError) as exc:
                last_error = exc
                state = {
                    **state,
                    "status": "rejected",
                    **self._last_error_fields(exc),
                }
                atomic_write_json(authoring_state_path, state)
            except InferenceError as exc:
                if is_retryable_upstream_failure(exc):
                    # Transient provider fault: the coordinator's upstream
                    # backoff owns the retry; do not consume the repair budget.
                    raise
                last_error = exc
                state = {
                    **state,
                    "status": "rejected",
                    **self._last_error_fields(exc),
                }
                atomic_write_json(authoring_state_path, state)
        raise ValueError(f"background {stage} artifacts remain invalid: {last_error}")

    def _stage_edit_spec(
        self,
        stage: str,
        *,
        purpose: str,
        last_error: BaseException | None,
        diagnostic_path: Path | None,
        induced: bool,
        fixed_input_paths: list[Path],
        catalog_receipt: dict[str, str] | None,
    ) -> AgentEditSpec:
        run_dir = self.identity.run_dir
        retrieval_manifest_path = run_dir / "background_retrieval.json"
        if stage == "retrieval":
            retrieval_draft_path = run_dir / "background_retrieval.draft.json"
            authored_paths = [retrieval_draft_path]
            derived_paths: tuple[Path, ...] = (retrieval_manifest_path,)
            tools: tuple[str, ...] = (
                "Read",
                "Glob",
                "Grep",
                "Write",
                "Edit",
                "Bash",
                "WebSearch",
                "WebFetch",
            )
            allowed_commands = frozenset(
                {
                    self.toolchain.background_retrieval_validate_command(run_dir),
                    self.toolchain.background_retrieval_import_command(
                        run_dir, retrieval_draft_path
                    ),
                }
            )
            system_prompt = BACKGROUND_RETRIEVAL_SYSTEM
            max_turns = BACKGROUND_RETRIEVAL_MAX_TURNS
        else:
            authored_paths = [run_dir / "background.md"]
            if induced:
                authored_paths.append(run_dir / "dimension_catalog.json")
            if self._has_provided_baseline():
                authored_paths.append(run_dir / "baseline_mechanisms.json")
            derived_paths = ()
            tools = ("Read", "Glob", "Grep", "Write", "Edit", "Bash")
            allowed_commands = self._registry_validation_commands(induced=induced)
            system_prompt = BACKGROUND_REGISTRY_SYSTEM
            max_turns = BACKGROUND_REGISTRY_MAX_TURNS
        mutable_paths = {*authored_paths, *derived_paths}
        input_paths = [
            *fixed_input_paths,
            *authored_paths,
            retrieval_manifest_path,
            *(() if diagnostic_path is None else (diagnostic_path,)),
        ]
        prompt = self._stage_prompt(
            stage,
            authored_paths=authored_paths,
            diagnostic_path=diagnostic_path,
            allowed_commands=allowed_commands,
            induced=induced,
            catalog_receipt=catalog_receipt,
        )
        return AgentEditSpec(
            purpose=purpose,
            schema_version=2,
            cwd=self.identity.repo_root,
            system_prompt=system_prompt,
            prompt=self._attempt_prompt(prompt, last_error),
            tools=tools,
            read_roots=(
                self.identity.repo_root / "tasks" / self.identity.task_name,
                self.identity.repo_root / "docs",
                self.identity.repo_root / "contracts",
                run_dir,
            ),
            write_paths=tuple(authored_paths),
            allowed_commands=allowed_commands,
            input_paths=tuple(input_paths),
            immutable_input_paths=tuple(
                path for path in input_paths if path not in mutable_paths
            ),
            derived_output_paths=derived_paths,
            max_turns=max_turns,
        )

    def _stage_prompt(
        self,
        stage: str,
        *,
        authored_paths: list[Path],
        diagnostic_path: Path | None,
        allowed_commands: frozenset[str],
        induced: bool,
        catalog_receipt: dict[str, str] | None,
    ) -> str:
        run_dir = self.identity.run_dir
        retrieval_manifest_path = run_dir / "background_retrieval.json"
        if stage == "retrieval":
            retrieval_draft_path = run_dir / "background_retrieval.draft.json"
            prompt = (
                f"Task: {self.identity.task_name}\nRun directory: {run_dir}\n"
                "Write only this authored output:\n- "
                + "\n- ".join(str(path) for path in authored_paths)
                + f"\nThe canonical retrieval manifest {retrieval_manifest_path} is "
                "Python-owned: do not write it. Record runtime WebSearch/WebFetch "
                f"work in the exact schema-1 draft {retrieval_draft_path}; retain "
                "exact bounded fetched text, not hashes or reconstructed receipts."
                "\nRead the repository background instructions and exact templates "
                "before writing."
            )
        else:
            prompt = (
                f"Task: {self.identity.task_name}\nRun directory: {run_dir}\n"
                f"Dimension strategy: {'llm_induced' if induced else 'catalog_subset'}\n"
                "Write only these authored outputs:\n- "
                + "\n- ".join(str(path) for path in authored_paths)
                + f"\nThe validated canonical retrieval manifest {retrieval_manifest_path} "
                "is frozen for this stage. Do not repeat searches or change its "
                "evidence.\n"
                + (
                    "Use this exact catalog receipt in background.md: "
                    + json.dumps(catalog_receipt, sort_keys=True)
                    if catalog_receipt is not None
                    else (
                        "Write the induced dimension catalog first and use any explicit "
                        "placeholder revision in background.md; Python will bind the exact "
                        "content-addressed catalog revision before validation."
                    )
                )
                + "\nRead the repository background instructions and exact templates before writing."
            )
        prompt += (
            f"\nRead the Python-owned repair diagnostic at {diagnostic_path} "
            "before editing; it is the authoritative validator feedback for this repair."
            if diagnostic_path is not None
            else ""
        )
        prompt += (
            "\nBefore finishing, run each of these exact validation commands "
            "with the Bash tool and iterate on the authored files until every "
            "command exits 0; no other shell command is permitted:\n"
            + "\n".join(f"- {command}" for command in sorted(allowed_commands))
            + (
                "\nThe canonical retrieval manifest may be produced only by "
                "running the import-external command on your draft; never "
                "write or edit the manifest file directly."
                if stage == "retrieval"
                else ""
            )
        )
        return prompt

    @staticmethod
    def _attempt_prompt(prompt: str, last_error: BaseException | None) -> str:
        if isinstance(last_error, InferenceError):
            # The prior attempt died inside the model call itself; no
            # validator diagnostic exists for it, so do not point at one.
            return prompt + (
                "\n\nThe prior bounded authoring attempt failed before "
                "validation with an inference error; the deterministic "
                "boundary did not reject the artifacts. Re-drive the same "
                "authoring task and finish within the turn budget.\n"
                + str(last_error)
            )
        if last_error is not None:
            return prompt + (
                "\n\nThe deterministic background boundary rejected the prior output. "
                "Repair every reported contract error. Preserve successful searches "
                "and exact retained evidence unless a reported error requires changing "
                "them. When the report is systemic, rebuild the affected registry and "
                "Markdown hierarchy from the exact template instead of patching only "
                "the listed examples.\n"
                + str(last_error)
            )
        return prompt

    def _registry_validation_commands(self, *, induced: bool) -> frozenset[str]:
        """Exact validator commands the bounded registry editor may run.

        Each string is byte-identical to the deterministic check Python re-runs
        after the call; the edit boundary matches Bash input by string
        equality. The frozen canonical manifest admits no import command in
        this stage.
        """
        run_dir = self.identity.run_dir
        commands = [
            self.toolchain.background_retrieval_validate_command(run_dir),
            self.toolchain.background_validate_command(
                run_dir,
                provided_baseline=self._has_provided_baseline(),
            ),
        ]
        if induced:
            commands.append(self.toolchain.dimension_catalog_validate_command(run_dir))
        return frozenset(commands)

    def _finalize_stage(
        self,
        stage: str,
        *,
        induced: bool,
        catalog_receipt: dict[str, str] | None,
    ) -> None:
        if stage == "retrieval":
            retrieval_draft_path = (
                self.identity.run_dir / "background_retrieval.draft.json"
            )
            if not retrieval_draft_path.is_file():
                raise BackgroundArtifactError(
                    f"background writer did not produce {retrieval_draft_path.name}"
                )
            self.toolchain.import_background_retrieval(
                self.identity.run_dir, retrieval_draft_path
            )
            self.toolchain.validate_background_retrieval(self.identity.run_dir)
            return
        if catalog_receipt is None:
            catalog_receipt = self.toolchain.background_catalog_receipt(
                self.identity.run_dir / "dimension_catalog.json"
            )
        self._bind_catalog_receipt(catalog_receipt)
        if self._has_provided_baseline():
            self._bind_baseline_entrypoint_receipt()
        self._validate_artifacts(induced=induced)

    @staticmethod
    def _last_error_fields(error: BaseException | None) -> dict[str, Any]:
        """Durable, typed form of the last boundary rejection.

        The validator output travels with the authoring state so a resumed
        repair can rebuild the original exception instead of degrading it to
        an untyped message.
        """
        if isinstance(error, ValidationRejected):
            return {
                "last_error": str(error),
                "last_error_kind": "validation_rejected",
                "last_error_label": error.label,
                "last_error_returncode": error.result.returncode,
                "last_error_output": error.result.output,
            }
        if isinstance(error, InferenceError):
            return {
                "last_error": str(error),
                "last_error_kind": "inference",
                "last_error_label": "",
                "last_error_returncode": 0,
                "last_error_output": "",
            }
        return {
            "last_error": str(error or ""),
            "last_error_kind": (
                "artifact_error"
                if isinstance(error, BackgroundArtifactError)
                else ""
            ),
            "last_error_label": "",
            "last_error_returncode": 0,
            "last_error_output": "",
        }

    @staticmethod
    def _resumed_error(state: dict[str, Any]) -> BaseException | None:
        """Rebuild the persisted rejection with its original type when possible."""
        message = state["last_error"].strip()
        if not message:
            return None
        if state.get("last_error_kind") == "validation_rejected":
            return ValidationRejected(
                state["last_error_label"],
                ProcessResult(
                    args=(),
                    returncode=state["last_error_returncode"],
                    output=state["last_error_output"],
                    elapsed_seconds=0.0,
                ),
            )
        if state.get("last_error_kind") == "inference":
            return InferenceError(message)
        return BackgroundArtifactError(message)

    @staticmethod
    def _valid_stage_counters(counters: Any) -> bool:
        if not isinstance(counters, dict):
            return False
        attempts = counters.get("attempts_admitted")
        repairs = counters.get("repairs_admitted")
        return (
            isinstance(attempts, int)
            and not isinstance(attempts, bool)
            and 0 <= attempts <= MAX_BACKGROUND_REPAIR_ATTEMPTS + 1
            and isinstance(repairs, int)
            and not isinstance(repairs, bool)
            and 0 <= repairs <= MAX_BACKGROUND_REPAIR_ATTEMPTS
            and isinstance(counters.get("initial_admitted"), bool)
        )

    @staticmethod
    def _read_authoring_state(
        path: Path,
        *,
        input_revision: str,
    ) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackgroundArtifactError(
                f"invalid background authoring state {path}: {exc}"
            ) from exc
        if isinstance(state, dict) and state.get("schema_version") == 1:
            raise BackgroundArtifactError(
                f"unsupported background authoring state {path}: schema_version 1 "
                "single-stage states predate the retrieval/registry split and "
                "cannot be resumed; start a fresh run tag"
            )
        stages = state.get("stages") if isinstance(state, dict) else None
        status = state.get("status") if isinstance(state, dict) else None
        purpose = state.get("purpose") if isinstance(state, dict) else None
        if not (
            isinstance(state, dict)
            and state.get("schema_version") == 2
            and state.get("kind") == "background_authoring"
            and state.get("input_revision") == input_revision
            and state.get("stage") in BACKGROUND_STAGES
            and isinstance(stages, dict)
            and all(
                BackgroundBuilder._valid_stage_counters(stages.get(name))
                for name in BACKGROUND_STAGES
            )
            and status in {"ready", "started", "rejected", "completed"}
            and (purpose is None or isinstance(purpose, str))
            and isinstance(state.get("last_error"), str)
            and state.get("last_error_kind", "")
            in {"", "artifact_error", "validation_rejected", "inference"}
            and isinstance(state.get("last_error_label", ""), str)
            and isinstance(state.get("last_error_output", ""), str)
            and isinstance(state.get("last_error_returncode", 0), int)
            and not isinstance(state.get("last_error_returncode", 0), bool)
        ):
            raise BackgroundArtifactError(
                f"malformed background authoring state: {path}"
            )
        if status == "started" and not purpose:
            raise BackgroundArtifactError(
                "started background authoring state lacks a purpose"
            )
        if status == "started" and not str(purpose).startswith(
            f"background_{state['stage']}"
        ):
            raise BackgroundArtifactError(
                "started background authoring purpose does not match its stage"
            )
        return state

    def _write_repair_diagnostic(self, error: BaseException) -> Path:
        """Persist bounded structured validation feedback for the next editor."""
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": "background_repair_diagnostic",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        if isinstance(error, ToolFailure):
            try:
                validator_output = json.loads(error.result.output)
            except json.JSONDecodeError:
                validator_output = None
            if isinstance(validator_output, dict):
                errors = validator_output.get("errors")
                if isinstance(errors, list) and all(
                    isinstance(item, str) for item in errors
                ):
                    payload["validator_errors"] = errors[
                        :MAX_BACKGROUND_REPAIR_DIAGNOSTIC_ERRORS
                    ]
                    payload["validator_errors_truncated"] = (
                        len(errors) > MAX_BACKGROUND_REPAIR_DIAGNOSTIC_ERRORS
                    )
                else:
                    payload["validator_output"] = validator_output
        path = self.identity.run_dir / ".orchestrator" / "background_repair_diagnostic.json"
        atomic_write_json(path, payload)
        return path

    def _bind_catalog_receipt(self, receipt: dict[str, str]) -> None:
        background_path = self.identity.run_dir / "background.md"
        try:
            text = background_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise BackgroundArtifactError(
                f"cannot read background artifact {background_path}: {exc}"
            ) from exc
        pattern = re.compile(
            r"(?P<prefix>^## Search space registry\s*\n```json\s*\n)"
            r"(?P<body>.*?)"
            r"(?P<suffix>^```\s*$)",
            flags=re.MULTILINE | re.DOTALL,
        )
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            raise BackgroundArtifactError(
                "background.md must contain exactly one fenced Search space registry"
            )
        match = matches[0]
        try:
            registry = json.loads(match.group("body"))
        except json.JSONDecodeError as exc:
            raise BackgroundArtifactError(
                f"background search-space registry is invalid JSON: {exc}"
            ) from exc
        catalog = registry.get("catalog") if isinstance(registry, dict) else None
        if not isinstance(catalog, dict):
            raise BackgroundArtifactError(
                "background search-space registry requires a catalog receipt"
            )
        if catalog.get("id") != receipt["id"]:
            raise BackgroundArtifactError(
                "background catalog id does not match the resolved catalog: "
                f"expected {receipt['id']!r}, got {catalog.get('id')!r}"
            )
        catalog["revision"] = receipt["revision"]
        rendered = json.dumps(registry, indent=2, ensure_ascii=False, allow_nan=False)
        updated = text[: match.start("body")] + rendered + "\n" + text[match.end("body") :]
        atomic_write_text(background_path, updated)

    def _bind_baseline_entrypoint_receipt(self) -> None:
        entrypoint = self._seed_entrypoint()
        if entrypoint is None:  # pragma: no cover - guarded by the caller
            return
        inventory_path = self.identity.run_dir / "baseline_mechanisms.json"
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackgroundArtifactError(
                f"invalid baseline mechanism inventory {inventory_path}: {exc}"
            ) from exc
        if not isinstance(inventory, dict):
            raise BackgroundArtifactError(
                "baseline mechanism inventory must be a JSON object"
            )
        existing = inventory.get("entrypoint")
        entrypoint_receipt = dict(existing) if isinstance(existing, dict) else {}
        entrypoint_receipt.update(
            {
                "path": entrypoint.relative_to(
                    self.identity.repo_root.resolve()
                ).as_posix(),
                "sha256": file_revision(entrypoint),
            }
        )
        inventory["entrypoint"] = entrypoint_receipt
        atomic_write_json(inventory_path, inventory)

    def _validate_artifacts(self, *, induced: bool) -> None:
        provided_baseline = self._has_provided_baseline()
        self.toolchain.validate_background(
            self.identity.run_dir,
            induced=induced,
            provided_baseline=provided_baseline,
        )
        if provided_baseline:
            self._validate_baseline_entrypoint_receipt()

    def _validate_baseline_entrypoint_receipt(self) -> None:
        entrypoint = self._seed_entrypoint()
        if entrypoint is None:  # pragma: no cover - guarded by the caller
            return
        inventory_path = self.identity.run_dir / "baseline_mechanisms.json"
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackgroundArtifactError(
                f"invalid baseline mechanism inventory {inventory_path}: {exc}"
            ) from exc
        receipt = inventory.get("entrypoint") if isinstance(inventory, dict) else None
        expected_path = entrypoint.relative_to(self.identity.repo_root.resolve()).as_posix()
        actual_revision = file_revision(entrypoint)
        if not isinstance(receipt, dict):
            raise BackgroundArtifactError(
                "baseline mechanism inventory requires an entrypoint receipt"
            )
        if receipt.get("path") != expected_path:
            raise BackgroundArtifactError(
                "baseline mechanism inventory entrypoint.path does not match "
                f"the task seed: expected {expected_path!r}"
            )
        if receipt.get("sha256") != actual_revision:
            raise BackgroundArtifactError(
                "baseline mechanism inventory entrypoint.sha256 does not match "
                f"the current task seed {expected_path}"
            )

    def _dimension_strategy(self) -> str:
        path = self.identity.run_dir / "framework_cfg.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        section = value.get("space_initialization", {})
        strategy = section.get("dimension_strategy", "catalog_subset") if isinstance(section, dict) else None
        if strategy not in {"catalog_subset", "llm_induced"}:
            raise ValueError(f"invalid dimension strategy in {path}: {strategy!r}")
        return strategy

    def _has_provided_baseline(self) -> bool:
        provided = self.task_config.get("seed", {}).get("provided")
        return isinstance(provided, list) and bool(provided)

    def _seed_entrypoint(self) -> Path | None:
        seed = self.task_config.get("seed")
        if not isinstance(seed, dict) or not self._has_provided_baseline():
            return None
        entrypoint = seed.get("entrypoint", "train.py")
        if not isinstance(entrypoint, str) or not entrypoint:
            raise ValueError("seed.entrypoint must be a non-empty string")
        task_dir = (
            self.identity.repo_root / "tasks" / self.identity.task_name
        ).resolve()
        resolved = (task_dir / entrypoint).resolve()
        try:
            resolved.relative_to(task_dir)
        except ValueError as exc:
            raise BackgroundArtifactError(
                f"seed.entrypoint escapes the task directory: {entrypoint!r}"
            ) from exc
        if not resolved.is_file():
            raise BackgroundArtifactError(
                f"provided seed entrypoint does not exist: {resolved}"
            )
        return resolved
