"""Agentic background research behind a frozen, validated artifact boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .llm import AgentEditSpec, ModelGateway
from .models import RunIdentity
from .prompts import BACKGROUND_SYSTEM
from .toolchain import ToolFailure, Toolchain


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
        if all(path.is_file() for path in required):
            try:
                self.toolchain.validate_background(
                    run_dir, induced=induced, provided_baseline=self._has_provided_baseline()
                )
                return
            except ToolFailure:
                if self.identity.ledger_path.exists():
                    raise
        elif self.identity.ledger_path.exists():
            missing = ", ".join(str(path.name) for path in required if not path.is_file())
            raise ValueError(f"frozen run is missing background artifacts: {missing}")

        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        docs_dir = self.identity.repo_root / "docs"
        contracts_dir = self.identity.repo_root / "contracts"
        output_paths = [
            run_dir / "background.md",
            run_dir / "background_retrieval.json",
        ]
        if induced:
            output_paths.append(run_dir / "dimension_catalog.json")
        if self._has_provided_baseline():
            output_paths.append(run_dir / "baseline_mechanisms.json")
        input_paths = [
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
            *output_paths,
        ]
        seed_entrypoint = self._seed_entrypoint()
        if seed_entrypoint is not None:
            input_paths.append(seed_entrypoint)
        prompt = (
            f"Task: {self.identity.task_name}\nRun directory: {run_dir}\n"
            f"Dimension strategy: {'llm_induced' if induced else 'catalog_subset'}\n"
            "Write only these outputs:\n- "
            + "\n- ".join(str(path) for path in output_paths)
            + "\nRead the repository background instructions and exact template before writing. "
            "The retrieval manifest must truthfully describe sources you actually inspected."
        )
        last_error: BaseException | None = None
        for attempt in range(2):
            attempt_prompt = prompt
            if last_error is not None:
                attempt_prompt += (
                    "\n\nThe deterministic background validator rejected the first output. "
                    "Repair only the reported contract error once.\n"
                    + str(last_error)
                )
            try:
                self.models.edit(
                    AgentEditSpec(
                        purpose="background_research" + (":repair" if attempt else ""),
                        schema_version=1,
                        cwd=self.identity.repo_root,
                        system_prompt=BACKGROUND_SYSTEM,
                        prompt=attempt_prompt,
                        tools=(
                            "Read",
                            "Glob",
                            "Grep",
                            "Write",
                            "Edit",
                            "WebSearch",
                            "WebFetch",
                        ),
                        read_roots=(task_dir, docs_dir, contracts_dir, run_dir),
                        write_paths=tuple(output_paths),
                        input_paths=tuple(input_paths),
                        immutable_input_paths=tuple(
                            path for path in input_paths if path not in output_paths
                        ),
                        max_turns=48,
                    ),
                    validate=lambda: self.toolchain.validate_background(
                        run_dir, induced=induced, provided_baseline=self._has_provided_baseline()
                    ),
                )
                return
            except ToolFailure as exc:
                last_error = exc
        raise ValueError(f"background artifacts remain invalid: {last_error}")

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
        return self.identity.repo_root / "tasks" / self.identity.task_name / entrypoint
