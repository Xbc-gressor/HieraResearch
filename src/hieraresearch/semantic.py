"""One semantic admission: deterministic selection around bounded LLM judgments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .llm import ModelGateway
from .models import IdeaProposal, RunIdentity
from .prompts import IDEA_SYSTEM, SEMANTIC_PREDICTION_SYSTEM
from .schemas import IDEA_SCHEMA, prediction_schema
from .toolchain import ToolFailure, Toolchain


GAIN_POLICIES = {"gain", "gain_uncertainty", "gain_uncertainty_nocost"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _bounded_text(path: Path, limit: int = 60_000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated after {limit} characters]"


def _object_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("response must be an object")
    return value


class SemanticAdmission:
    def __init__(
        self,
        identity: RunIdentity,
        toolchain: Toolchain,
        models: ModelGateway,
    ):
        self.identity = identity
        self.toolchain = toolchain
        self.models = models

    def admit(
        self,
        *,
        run_id: str,
        op: str,
        parents: list[str],
        baseline_only: bool = False,
    ) -> dict[str, Any]:
        run_dir = self.identity.run_dir
        proposals = self.toolchain.semantic_propose(
            run_dir,
            run_id,
            op,
            parents,
            baseline_only=baseline_only,
        )
        policy = "coverage" if baseline_only else self._configured_policy()
        if policy == "coverage":
            point, policy_receipt, _ = self.toolchain.semantic_select(
                run_dir, proposals, policy="coverage"
            )
        elif policy in GAIN_POLICIES:
            point, policy_receipt = self._select_with_predictions(
                run_id=run_id,
                policy=policy,
                proposals=proposals,
            )
        else:
            raise ValueError(f"unsupported semantic policy: {policy!r}")

        if baseline_only:
            point_value = _read_json(point)
            point_id = point_value.get("point_id")
            idea = IdeaProposal(
                idea="Use the unchanged task-provided baseline implementation.",
                change=f"provided baseline at {point_id}",
                candidate_name_hint="provided_baseline",
                description="Task-provided baseline entrypoint",
            )
        else:
            idea = self._propose_idea(
                run_id=run_id,
                op=op,
                parents=parents,
                point=point,
                policy_receipt=policy_receipt,
            )
        return self.toolchain.admit_candidate(
            run_dir,
            self.identity.task_name,
            run_id,
            op,
            parents,
            point,
            policy_receipt,
            idea=idea.idea,
            change=idea.change,
            candidate_name_hint=idea.candidate_name_hint,
            description=idea.description,
        )

    def _configured_policy(self) -> str:
        path = self.identity.run_dir / "framework_cfg.json"
        value = _read_json(path)
        section = value.get("semantic_search", {})
        if not isinstance(section, dict):
            raise ValueError(f"{path}: semantic_search must be an object")
        policy = section.get("policy", "gain_uncertainty_nocost")
        if not isinstance(policy, str):
            raise ValueError(f"{path}: semantic_search.policy must be a string")
        return policy

    def _select_with_predictions(
        self,
        *,
        run_id: str,
        policy: str,
        proposals: Path,
    ) -> tuple[Path, Path]:
        run_dir = self.identity.run_dir
        gain_context = self.toolchain.semantic_gain_context(run_dir, proposals)
        include_cost = policy != "gain_uncertainty_nocost"
        prompt = (
            f"Policy: {policy}\n\n"
            "Valid proposals:\n"
            + _bounded_text(proposals)
            + "\n\nBounded experience conditioning:\n"
            + _bounded_text(gain_context)
            + "\n\nFrozen background excerpt:\n"
            + _bounded_text(run_dir / "background.md", 30_000)
        )
        input_paths = [
            proposals,
            gain_context,
            run_dir / "background.md",
            run_dir / "ledger.json",
            self.identity.repo_root / "tasks" / self.identity.task_name / "TASK.md",
            self.identity.repo_root / "tasks" / self.identity.task_name / "task.toml",
        ]
        predictions = self.models.infer(
            purpose=f"semantic_predictions:{run_id}",
            schema_version=3,
            system_prompt=SEMANTIC_PREDICTION_SYSTEM,
            prompt=prompt,
            schema=prediction_schema(include_cost=include_cost),
            input_paths=input_paths,
            parser=_object_response,
            max_tokens=12_000,
        )
        predictions_path = proposals.parent / "predictions.json"
        atomic_write_json(predictions_path, predictions)
        try:
            point, receipt, _ = self.toolchain.semantic_select(
                run_dir,
                proposals,
                predictions=predictions_path,
            )
            return point, receipt
        except ToolFailure as first_error:
            correction_prompt = (
                prompt
                + "\n\nYour first response was rejected by the deterministic selector. "
                "Correct it once; do not change the proposal set.\nError:\n"
                + str(first_error)
                + "\nRejected response:\n"
                + json.dumps(predictions, ensure_ascii=False)
            )
            corrected = self.models.infer(
                purpose=f"semantic_predictions:{run_id}:correction",
                schema_version=3,
                system_prompt=SEMANTIC_PREDICTION_SYSTEM,
                prompt=correction_prompt,
                schema=prediction_schema(include_cost=include_cost),
                input_paths=[*input_paths, predictions_path],
                parser=_object_response,
                max_tokens=12_000,
            )
            atomic_write_json(predictions_path, corrected)
            point, receipt, _ = self.toolchain.semantic_select(
                run_dir,
                proposals,
                predictions=predictions_path,
            )
            return point, receipt

    def _propose_idea(
        self,
        *,
        run_id: str,
        op: str,
        parents: list[str],
        point: Path,
        policy_receipt: Path,
    ) -> IdeaProposal:
        run_dir = self.identity.run_dir
        parent_records = [
            self.toolchain.ledger_record(run_dir, parent) for parent in parents
        ]
        prompt = (
            f"Task: {self.identity.task_name}\nOperation: {op}\nParents: {parents}\n\n"
            "Task contract:\n"
            + _bounded_text(
                self.identity.repo_root / "tasks" / self.identity.task_name / "TASK.md",
                30_000,
            )
            + "\n\nSelected semantic point:\n"
            + _bounded_text(point)
            + "\n\nSelection receipt:\n"
            + _bounded_text(policy_receipt, 20_000)
            + "\n\nParent records (bounded):\n"
            + json.dumps(parent_records, ensure_ascii=False)[:30_000]
        )
        return self.models.infer(
            purpose=f"candidate_idea:{run_id}",
            schema_version=1,
            system_prompt=IDEA_SYSTEM,
            prompt=prompt,
            schema=IDEA_SCHEMA,
            input_paths=[
                point,
                policy_receipt,
                run_dir / "background.md",
                run_dir / "ledger.json",
                self.identity.repo_root / "tasks" / self.identity.task_name / "TASK.md",
                self.identity.repo_root / "tasks" / self.identity.task_name / "task.toml",
            ],
            parser=IdeaProposal.from_dict,
            max_tokens=4096,
        )
