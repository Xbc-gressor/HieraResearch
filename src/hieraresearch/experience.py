"""Bounded belief refresh with deterministic validation and ledger application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .llm import ModelGateway
from .models import RunIdentity
from .prompts import EXPERIENCE_SYSTEM
from .schemas import EXPERIENCE_SCHEMA
from .toolchain import ToolFailure, Toolchain


def _object_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("experience response must be an object")
    return value


class ExperienceRefresh:
    def __init__(
        self,
        identity: RunIdentity,
        toolchain: Toolchain,
        models: ModelGateway,
    ):
        self.identity = identity
        self.toolchain = toolchain
        self.models = models

    def run(self, ledger_brief: dict[str, Any]) -> dict[str, Any]:
        run_dir = self.identity.run_dir
        views = self.toolchain.experience_views(run_dir)
        dag_revision = ledger_brief.get("dag_revision")
        output = (
            run_dir
            / ".orchestrator"
            / f"experience-proposal-dag-{dag_revision}.json"
        )
        prompt = (
            "Create the next complete experience snapshot from these bounded helper views. "
            "Do not infer omitted records or edges. Empty lists and an empty summary are valid.\n\n"
            + json.dumps(views, ensure_ascii=False)[:180_000]
        )
        input_paths = [
            run_dir / "ledger.json",
            run_dir / "background.md",
            self.identity.repo_root / "tasks" / self.identity.task_name / "TASK.md",
            self.identity.repo_root / "tasks" / self.identity.task_name / "task.toml",
        ]
        proposal = self.models.infer(
            purpose=f"experience_refresh:dag-{dag_revision}",
            schema_version=4,
            system_prompt=EXPERIENCE_SYSTEM,
            prompt=prompt,
            schema=EXPERIENCE_SCHEMA,
            input_paths=input_paths,
            parser=_object_response,
        )
        atomic_write_json(output, proposal)
        try:
            self.toolchain.validate_experience(run_dir, output)
        except ToolFailure as first_error:
            correction = self.models.infer(
                purpose=f"experience_refresh:dag-{dag_revision}:correction",
                schema_version=4,
                system_prompt=EXPERIENCE_SYSTEM,
                prompt=(
                    prompt
                    + "\n\nThe deterministic validator rejected the first snapshot. "
                    "Correct it once without expanding the evidence view.\nError:\n"
                    + str(first_error)
                    + "\nRejected snapshot:\n"
                    + json.dumps(proposal, ensure_ascii=False)
                ),
                schema=EXPERIENCE_SCHEMA,
                input_paths=[*input_paths, output],
                parser=_object_response,
            )
            atomic_write_json(output, correction)
            self.toolchain.validate_experience(run_dir, output)
        self.toolchain.store_experience(run_dir, output)
        return self.toolchain.apply_space_state(run_dir)
