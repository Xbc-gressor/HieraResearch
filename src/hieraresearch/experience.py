"""Bounded belief refresh with deterministic validation and ledger application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import ArtifactError, atomic_write_json, file_revision, json_revision
from .llm import ModelGateway
from .models import RunIdentity
from .prompts import EXPERIENCE_SYSTEM
from .schemas import EXPERIENCE_SCHEMA, EXPERIENCE_SCHEMA_VERSION
from .toolchain import Toolchain, ValidationRejected


def _object_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("experience response must be an object")
    return value


_RECEIPT_SCHEMA_VERSION = 1
_RECEIPT_KIND = "experience_refresh"
_RECEIPT_STATUSES = {"prepared", "stored", "completed"}
_REFRESH_BASIS_FIELDS = {
    "dag_revision",
    "experience_dag_revision",
    "experience_dag_delta",
    "semantic_admission_blocked",
    "experience_refresh_required",
}


def _valid_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == len("sha256:") + 64
        and all(char in "0123456789abcdef" for char in value.removeprefix("sha256:"))
    )


def _brief_revision(ledger_brief: dict[str, Any]) -> str:
    return json_revision(
        {
            field: ledger_brief.get(field)
            for field in sorted(_REFRESH_BASIS_FIELDS)
        }
    )


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

    def _proposal_path(self, dag_revision: int) -> Path:
        return (
            self.identity.run_dir
            / ".orchestrator"
            / f"experience-proposal-dag-{dag_revision}.json"
        )

    def _receipt_path(self, dag_revision: int) -> Path:
        return (
            self.identity.run_dir
            / ".orchestrator"
            / f"experience-refresh-dag-{dag_revision}.json"
        )

    def _load_receipt(self, path: Path) -> dict[str, Any]:
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"invalid experience refresh receipt {path}: {exc}") from exc
        if not isinstance(receipt, dict):
            raise ArtifactError(f"experience refresh receipt must be an object: {path}")
        status = receipt.get("status")
        required = {
            "schema_version",
            "kind",
            "dag_revision",
            "brief_revision",
            "proposal_revision",
            "status",
        }
        if status in {"stored", "completed"}:
            required.add("stored_experience_revision")
        if status == "completed":
            required.update({"apply_result", "completion_mode"})
        if set(receipt) != required:
            raise ArtifactError(
                f"experience refresh receipt fields must be exactly {sorted(required)}: {path}"
            )
        dag_revision = receipt.get("dag_revision")
        if (
            receipt.get("schema_version") != _RECEIPT_SCHEMA_VERSION
            or receipt.get("kind") != _RECEIPT_KIND
            or status not in _RECEIPT_STATUSES
            or not isinstance(dag_revision, int)
            or isinstance(dag_revision, bool)
            or dag_revision < 0
            or path != self._receipt_path(dag_revision)
            or not _valid_revision(receipt.get("brief_revision"))
            or not _valid_revision(receipt.get("proposal_revision"))
            or (
                status in {"stored", "completed"}
                and not _valid_revision(receipt.get("stored_experience_revision"))
            )
            or (
                status == "completed"
                and not isinstance(receipt.get("apply_result"), dict)
            )
            or (
                status == "completed"
                and receipt.get("completion_mode") not in {"initial", "reconciled"}
            )
        ):
            raise ArtifactError(f"invalid experience refresh receipt: {path}")
        return receipt

    def _pending_receipt(self) -> dict[str, Any] | None:
        root = self.identity.run_dir / ".orchestrator"
        pending: list[dict[str, Any]] = []
        for path in sorted(root.glob("experience-refresh-dag-*.json")):
            receipt = self._load_receipt(path)
            if receipt["status"] != "completed":
                pending.append(receipt)
        if len(pending) > 1:
            raise ArtifactError("multiple incomplete experience refresh receipts")
        return pending[0] if pending else None

    def has_pending(self, ledger_brief: dict[str, Any]) -> bool:
        pending = self._pending_receipt()
        if pending is None:
            return False
        if pending["dag_revision"] != ledger_brief.get("dag_revision"):
            raise ArtifactError(
                "incomplete experience refresh does not match the current DAG revision"
            )
        return True

    def _load_proposal(self, receipt: dict[str, Any]) -> dict[str, Any]:
        output = self._proposal_path(receipt["dag_revision"])
        if not output.is_file() or file_revision(output) != receipt["proposal_revision"]:
            raise ArtifactError(f"experience proposal changed after validation: {output}")
        try:
            proposal = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"invalid experience proposal {output}: {exc}") from exc
        if not isinstance(proposal, dict):
            raise ArtifactError(f"experience proposal must be an object: {output}")
        return proposal

    @staticmethod
    def _expected_stored_experience(
        proposal: dict[str, Any], dag_revision: int
    ) -> dict[str, Any]:
        expected = dict(proposal)
        expected["dag_revision"] = dag_revision
        return expected

    def _stored_receipt(
        self,
        receipt: dict[str, Any],
        expected_experience: dict[str, Any],
    ) -> dict[str, Any]:
        stored = {
            **receipt,
            "status": "stored",
            "stored_experience_revision": json_revision(expected_experience),
        }
        atomic_write_json(self._receipt_path(receipt["dag_revision"]), stored)
        return stored

    def _resume(
        self,
        ledger_brief: dict[str, Any],
        receipt: dict[str, Any],
        *,
        recovery: bool,
    ) -> dict[str, Any]:
        if receipt["dag_revision"] != ledger_brief.get("dag_revision"):
            raise ArtifactError(
                "experience refresh receipt does not match the current DAG revision"
            )
        proposal = self._load_proposal(receipt)
        expected_experience = self._expected_stored_experience(
            proposal, receipt["dag_revision"]
        )
        expected_stored_revision = json_revision(expected_experience)
        status = receipt["status"]
        if status == "completed":
            if receipt["stored_experience_revision"] != expected_stored_revision:
                raise ArtifactError("completed experience refresh has a stale stored revision")
            return receipt["apply_result"]

        current_experience = self.toolchain.ledger_experience(self.identity.run_dir)
        if status == "prepared":
            if current_experience == expected_experience:
                receipt = self._stored_receipt(receipt, expected_experience)
            else:
                if _brief_revision(ledger_brief) != receipt["brief_revision"]:
                    raise ArtifactError(
                        "ledger brief changed after experience validation and before store"
                    )
                raw_cursor = ledger_brief.get("experience_dag_revision")
                cursor = 0 if raw_cursor is None else raw_cursor
                if (
                    not isinstance(cursor, int)
                    or isinstance(cursor, bool)
                    or cursor >= receipt["dag_revision"]
                ):
                    raise ArtifactError(
                        "stored experience contradicts the prepared refresh receipt"
                    )
                self.toolchain.store_experience(
                    self.identity.run_dir,
                    self._proposal_path(receipt["dag_revision"]),
                )
                receipt = self._stored_receipt(receipt, expected_experience)
        elif (
            receipt["stored_experience_revision"] != expected_stored_revision
            or current_experience != expected_experience
        ):
            raise ArtifactError("stored experience does not match its refresh receipt")

        result = self.toolchain.apply_space_state(self.identity.run_dir)
        if not isinstance(result, dict):
            raise ArtifactError("search-space state application must return an object")
        completed = {
            **receipt,
            "status": "completed",
            "apply_result": result,
            "completion_mode": "reconciled" if recovery else "initial",
        }
        atomic_write_json(self._receipt_path(receipt["dag_revision"]), completed)
        return result

    def run(self, ledger_brief: dict[str, Any]) -> dict[str, Any]:
        run_dir = self.identity.run_dir
        pending = self._pending_receipt()
        if pending is not None:
            return self._resume(ledger_brief, pending, recovery=True)
        views = self.toolchain.experience_views(run_dir)
        dag_revision = ledger_brief.get("dag_revision")
        if (
            not isinstance(dag_revision, int)
            or isinstance(dag_revision, bool)
            or dag_revision < 0
        ):
            raise ValueError("experience refresh requires a non-negative DAG revision")
        output = self._proposal_path(dag_revision)
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
            schema_version=EXPERIENCE_SCHEMA_VERSION,
            system_prompt=EXPERIENCE_SYSTEM,
            prompt=prompt,
            schema=EXPERIENCE_SCHEMA,
            input_paths=input_paths,
            parser=_object_response,
        )
        atomic_write_json(output, proposal)
        try:
            self.toolchain.validate_experience(run_dir, output)
        except ValidationRejected as first_error:
            correction = self.models.infer(
                purpose=f"experience_refresh:dag-{dag_revision}:correction",
                schema_version=EXPERIENCE_SCHEMA_VERSION,
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
        receipt = {
            "schema_version": _RECEIPT_SCHEMA_VERSION,
            "kind": _RECEIPT_KIND,
            "dag_revision": dag_revision,
            "brief_revision": _brief_revision(ledger_brief),
            "proposal_revision": file_revision(output),
            "status": "prepared",
        }
        atomic_write_json(self._receipt_path(dag_revision), receipt)
        self.toolchain.store_experience(run_dir, output)
        stored_proposal = self._load_proposal(receipt)
        expected_experience = self._expected_stored_experience(
            stored_proposal, dag_revision
        )
        receipt = self._stored_receipt(receipt, expected_experience)
        return self._resume(ledger_brief, receipt, recovery=False)
