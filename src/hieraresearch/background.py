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
from .llm import AgentEditSpec, ModelGateway
from .models import RunIdentity
from .prompts import BACKGROUND_SYSTEM
from .toolchain import ToolFailure, Toolchain, ValidationRejected


MAX_BACKGROUND_REPAIR_ATTEMPTS = 5
MAX_BACKGROUND_REPAIR_DIAGNOSTIC_ERRORS = 256


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
        required = [run_dir / "background.md", run_dir / "background_retrieval.json"]
        if induced:
            required.append(run_dir / "dimension_catalog.json")
        if self._has_provided_baseline():
            required.append(run_dir / "baseline_mechanisms.json")
        initial_error: ValidationRejected | BackgroundArtifactError | None = None
        if all(path.is_file() for path in required):
            try:
                self._validate_artifacts(induced=induced)
                return
            except (ValidationRejected, BackgroundArtifactError) as exc:
                if self.identity.ledger_path.exists():
                    raise
                initial_error = exc
        elif self.identity.ledger_path.exists():
            missing = ", ".join(str(path.name) for path in required if not path.is_file())
            raise ValueError(f"frozen run is missing background artifacts: {missing}")

        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        docs_dir = self.identity.repo_root / "docs"
        contracts_dir = self.identity.repo_root / "contracts"
        background_path = run_dir / "background.md"
        retrieval_manifest_path = run_dir / "background_retrieval.json"
        catalog_receipt = (
            None if induced else self.toolchain.background_catalog_receipt()
        )
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
            inferred_initial_attempt = initial_error is not None
            state = {
                "schema_version": 1,
                "kind": "background_authoring",
                "status": "rejected" if inferred_initial_attempt else "ready",
                "input_revision": input_revision,
                "attempts_admitted": 1 if inferred_initial_attempt else 0,
                "initial_admitted": inferred_initial_attempt,
                "repairs_admitted": 0,
                "purpose": None,
                "last_error": str(initial_error or ""),
            }
            atomic_write_json(authoring_state_path, state)
        elif state["status"] == "completed":
            raise BackgroundArtifactError(
                "completed background authoring state has invalid artifacts"
            )
        elif initial_error is not None and not state["initial_admitted"]:
            state = {
                **state,
                "status": "rejected",
                "attempts_admitted": max(1, int(state["attempts_admitted"])),
                "initial_admitted": True,
                "last_error": str(initial_error),
            }
            atomic_write_json(authoring_state_path, state)
        last_error: BaseException | None = initial_error
        if last_error is None:
            last_error = self._resumed_error(state)
        while int(state["attempts_admitted"]) < MAX_BACKGROUND_REPAIR_ATTEMPTS + 1:
            resuming_started = state["status"] == "started"
            retry_purpose = state.get("purpose") if resuming_started else None
            if isinstance(retry_purpose, str) and retry_purpose:
                purpose = retry_purpose
            elif not state["initial_admitted"]:
                purpose = "background_research"
                state = {**state, "initial_admitted": True}
            else:
                repairs_admitted = int(state["repairs_admitted"])
                if repairs_admitted >= MAX_BACKGROUND_REPAIR_ATTEMPTS:
                    break
                next_repair = repairs_admitted + 1
                purpose = f"background_research:repair:{next_repair}"
                state = {**state, "repairs_admitted": next_repair}
            diagnostic_path = (
                self._write_repair_diagnostic(last_error)
                if last_error is not None
                else None
            )
            reuse_retrieval = self._has_valid_retrieval(retrieval_manifest_path)
            retrieval_draft_path = (
                None
                if reuse_retrieval
                else run_dir / "background_retrieval.draft.json"
            )
            authored_paths = [background_path]
            if retrieval_draft_path is not None:
                authored_paths.append(retrieval_draft_path)
            if induced:
                authored_paths.append(run_dir / "dimension_catalog.json")
            if self._has_provided_baseline():
                authored_paths.append(run_dir / "baseline_mechanisms.json")
            derived_paths = () if reuse_retrieval else (retrieval_manifest_path,)
            mutable_paths = {*authored_paths, *derived_paths}
            input_paths = [
                *fixed_input_paths,
                *authored_paths,
                retrieval_manifest_path,
                *(() if diagnostic_path is None else (diagnostic_path,)),
            ]
            prompt = (
                f"Task: {self.identity.task_name}\nRun directory: {run_dir}\n"
                f"Dimension strategy: {'llm_induced' if induced else 'catalog_subset'}\n"
                "Write only these authored outputs:\n- "
                + "\n- ".join(str(path) for path in authored_paths)
                + (
                    f"\nThe validated canonical retrieval manifest {retrieval_manifest_path} "
                    "is frozen for this repair. Do not repeat searches or change its evidence.\n"
                    if reuse_retrieval
                    else (
                        f"\nThe canonical retrieval manifest {retrieval_manifest_path} is "
                        "Python-owned: do not write it. Record runtime WebSearch/WebFetch "
                        f"work in the exact schema-1 draft {retrieval_draft_path}; retain "
                        "exact bounded fetched text, not hashes or reconstructed receipts.\n"
                    )
                )
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
                + (
                    f"\nRead the Python-owned repair diagnostic at {diagnostic_path} "
                    "before editing; it is the authoritative validator feedback for this repair."
                    if diagnostic_path is not None
                    else ""
                )
            )
            attempt_prompt = prompt
            if last_error is not None:
                attempt_prompt += (
                    "\n\nThe deterministic background boundary rejected the prior output. "
                    "Repair every reported contract error. Preserve successful searches "
                    "and exact retained evidence unless a reported error requires changing "
                    "them. When the report is systemic, rebuild the affected registry and "
                    "Markdown hierarchy from the exact template instead of patching only "
                    "the listed examples.\n"
                    + str(last_error)
                )
            try:
                state = {
                    **state,
                    "status": "started",
                    "attempts_admitted": int(state["attempts_admitted"]) + 1,
                    "purpose": purpose,
                    "last_error": str(last_error or ""),
                }
                atomic_write_json(authoring_state_path, state)
                self.models.edit(
                    AgentEditSpec(
                        purpose=purpose,
                        schema_version=2,
                        cwd=self.identity.repo_root,
                        system_prompt=BACKGROUND_SYSTEM,
                        prompt=attempt_prompt,
                        tools=(
                            "Read",
                            "Glob",
                            "Grep",
                            "Write",
                            "Edit",
                            *(
                                ()
                                if reuse_retrieval
                                else ("WebSearch", "WebFetch")
                            ),
                        ),
                        read_roots=(task_dir, docs_dir, contracts_dir, run_dir),
                        write_paths=tuple(authored_paths),
                        input_paths=tuple(input_paths),
                        immutable_input_paths=tuple(
                            path
                            for path in input_paths
                            if path not in mutable_paths
                        ),
                        derived_output_paths=derived_paths,
                        max_turns=48,
                    ),
                    validate=lambda: self._finalize_and_validate(
                        induced=induced,
                        retrieval_draft_path=retrieval_draft_path,
                        catalog_receipt=catalog_receipt,
                    ),
                )
                atomic_write_json(
                    authoring_state_path,
                    {
                        **state,
                        "status": "completed",
                        "last_error": "",
                    },
                )
                return
            except (ValidationRejected, BackgroundArtifactError) as exc:
                last_error = exc
                state = {
                    **state,
                    "status": "rejected",
                    "last_error": str(exc),
                }
                atomic_write_json(authoring_state_path, state)
        raise ValueError(f"background artifacts remain invalid: {last_error}")

    def _has_valid_retrieval(self, manifest_path: Path) -> bool:
        if not manifest_path.is_file():
            return False
        try:
            self.toolchain.validate_background_retrieval(self.identity.run_dir)
        except ValidationRejected:
            return False
        return True

    @staticmethod
    def _resumed_error(state: dict[str, Any]) -> BaseException | None:
        """Rebuild the persisted rejection from its durable message."""
        message = state["last_error"].strip()
        if not message:
            return None
        return BackgroundArtifactError(message)

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
        attempts = state.get("attempts_admitted") if isinstance(state, dict) else None
        repairs = state.get("repairs_admitted") if isinstance(state, dict) else None
        status = state.get("status") if isinstance(state, dict) else None
        purpose = state.get("purpose") if isinstance(state, dict) else None
        if not (
            isinstance(state, dict)
            and state.get("schema_version") == 1
            and state.get("kind") == "background_authoring"
            and state.get("input_revision") == input_revision
            and isinstance(attempts, int)
            and not isinstance(attempts, bool)
            and 0 <= attempts <= MAX_BACKGROUND_REPAIR_ATTEMPTS + 1
            and isinstance(repairs, int)
            and not isinstance(repairs, bool)
            and 0 <= repairs <= MAX_BACKGROUND_REPAIR_ATTEMPTS
            and isinstance(state.get("initial_admitted"), bool)
            and status in {"ready", "started", "rejected", "completed"}
            and (purpose is None or isinstance(purpose, str))
            and isinstance(state.get("last_error"), str)
            and state.get("last_error_kind", "")
            in {"", "artifact_error", "validation_rejected"}
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

    def _finalize_and_validate(
        self,
        *,
        induced: bool,
        retrieval_draft_path: Path | None,
        catalog_receipt: dict[str, str] | None,
    ) -> None:
        if retrieval_draft_path is not None:
            if not retrieval_draft_path.is_file():
                raise BackgroundArtifactError(
                    f"background writer did not produce {retrieval_draft_path.name}"
                )
            self.toolchain.import_background_retrieval(
                self.identity.run_dir, retrieval_draft_path
            )
        if catalog_receipt is None:
            catalog_receipt = self.toolchain.background_catalog_receipt(
                self.identity.run_dir / "dimension_catalog.json"
            )
        self._bind_catalog_receipt(catalog_receipt)
        if self._has_provided_baseline():
            self._bind_baseline_entrypoint_receipt()
        self._validate_artifacts(induced=induced)

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
