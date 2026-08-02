"""Durable coordinator state and revision-bound inference receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import ActiveRound, CoordinatorState


class ArtifactError(RuntimeError):
    pass


class StaleInferenceError(ArtifactError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def json_revision(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def file_revision(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def paths_revision(paths: Iterable[Path]) -> str:
    """Hash file identity and bytes, including explicit absence.

    Callers pass only bounded contract inputs. Directories are rejected so a
    future file appearing beneath a broad root cannot evade the binding.
    """

    entries: list[dict[str, Any]] = []
    for raw_path in sorted((Path(path).resolve() for path in paths), key=str):
        if raw_path.is_dir():
            raise ArtifactError(f"revision input must be a file, not a directory: {raw_path}")
        if raw_path.exists():
            entries.append(
                {
                    "path": str(raw_path),
                    "sha256": file_revision(raw_path),
                    "size": raw_path.stat().st_size,
                }
            )
        else:
            entries.append({"path": str(raw_path), "missing": True})
    return json_revision(entries)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(Path(path), value.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_write_bytes(Path(path), payload)


class CoordinatorStore:
    def __init__(self, run_dir: Path):
        self.root = Path(run_dir) / ".orchestrator"
        self.path = self.root / "state.json"

    def load(self, task_name: str, tag: str) -> CoordinatorState:
        if not self.path.exists():
            return CoordinatorState(task_name=task_name, tag=tag)
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"invalid coordinator state {self.path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ArtifactError(f"coordinator state must be an object: {self.path}")
        state = CoordinatorState.from_dict(value)
        if state.task_name != task_name or state.tag != tag:
            raise ArtifactError(
                "coordinator state identity mismatch: "
                f"expected {task_name}/{tag}, got {state.task_name}/{state.tag}"
            )
        return state

    def save(self, state: CoordinatorState) -> None:
        atomic_write_json(self.path, state.to_dict())

    def complete_round(self, active: ActiveRound) -> Path:
        """Persist an immutable audit receipt before forgetting active state."""
        if not active.tuning_complete:
            raise ArtifactError("cannot complete a round before deep tuning returns")
        receipt = {
            "schema_version": 1,
            "kind": "completed_round",
            "round": active.to_dict(),
        }
        path = self.root / "rounds" / f"{active.round_id:06d}.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactError(f"invalid completed round receipt {path}: {exc}") from exc
            if existing != receipt:
                raise ArtifactError(f"completed round receipt changed: {path}")
            return path
        atomic_write_json(path, receipt)
        return path


@dataclass(frozen=True)
class InvocationReceipt:
    invocation_id: str
    purpose: str
    schema_version: int
    model: str
    input_revision: str
    response: Any
    path: Path


class InvocationJournal:
    """Append-only model call records below one run directory."""

    def __init__(self, run_dir: Path):
        self.root = Path(run_dir) / ".orchestrator" / "invocations"

    @staticmethod
    def _purpose_name(purpose: str) -> str:
        value = re.sub(r"[^a-z0-9_.-]+", "-", purpose.lower()).strip("-.")
        return value[:80] or "inference"

    def completed(
        self,
        *,
        purpose: str,
        schema_version: int,
        input_revision: str,
        model: str | None = None,
    ) -> InvocationReceipt | None:
        if not self.root.exists():
            return None
        for path in sorted(self.root.glob(f"{self._purpose_name(purpose)}-*"), reverse=True):
            receipt_path = path / "receipt.json"
            request_path = path / "request.json"
            response_path = path / "response.json"
            if not receipt_path.is_file():
                continue
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactError(f"invalid invocation receipt {receipt_path}: {exc}") from exc
            if not isinstance(receipt, dict):
                raise ArtifactError(f"invocation receipt must be an object: {receipt_path}")
            if receipt.get("purpose") != purpose or receipt.get("status") != "completed":
                continue
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
                response = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactError(f"invalid completed invocation {path}: {exc}") from exc
            if receipt.get("input_revision") != json_revision(request):
                raise ArtifactError(f"invocation request revision mismatch: {path}")
            if receipt.get("response_revision") != json_revision(response):
                raise ArtifactError(f"invocation response revision mismatch: {path}")
            if (
                receipt.get("schema_version") == schema_version
                and receipt.get("input_revision") == input_revision
                and (model is None or receipt.get("model") == model)
            ):
                return InvocationReceipt(
                    invocation_id=str(receipt["invocation_id"]),
                    purpose=purpose,
                    schema_version=schema_version,
                    model=str(receipt["model"]),
                    input_revision=input_revision,
                    response=response,
                    path=path,
                )
        return None

    def completed_output_matches(
        self,
        *,
        purpose: str,
        schema_version: int,
        model: str,
        output_path: Path,
        immutable_input_paths: Iterable[Path],
    ) -> bool:
        """Return whether a completed edit matches current inputs and output bytes."""
        output_path = Path(output_path).resolve()
        if not output_path.is_file() or not self.root.exists():
            return False
        expected_revision = file_revision(output_path)
        immutable_revision = paths_revision(immutable_input_paths)
        for path in sorted(
            self.root.glob(f"{self._purpose_name(purpose)}-*"), reverse=True
        ):
            try:
                receipt = json.loads(
                    (path / "receipt.json").read_text(encoding="utf-8")
                )
            except FileNotFoundError:
                continue
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactError(f"invalid invocation receipt {path}: {exc}") from exc
            if not isinstance(receipt, dict):
                raise ArtifactError(f"invocation receipt must be an object: {path}")
            if receipt.get("purpose") != purpose or receipt.get("status") != "completed":
                continue
            try:
                request = json.loads(
                    (path / "request.json").read_text(encoding="utf-8")
                )
                response = json.loads(
                    (path / "response.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactError(f"invalid completed invocation {path}: {exc}") from exc
            if receipt.get("input_revision") != json_revision(request):
                raise ArtifactError(f"invocation request revision mismatch: {path}")
            if receipt.get("response_revision") != json_revision(response):
                raise ArtifactError(f"invocation response revision mismatch: {path}")
            if not isinstance(request, dict) or request.get("kind") != "agent_edit":
                raise ArtifactError(f"completed edit request is malformed: {path}")
            write_paths = request.get("write_paths")
            if not isinstance(write_paths, list) or not all(
                isinstance(value, str) for value in write_paths
            ):
                raise ArtifactError(f"completed edit write paths are malformed: {path}")
            revisions = response.get("output_revisions") if isinstance(response, dict) else None
            if (
                receipt.get("schema_version") == schema_version
                and receipt.get("model") == model
                and request.get("immutable_input_revision") == immutable_revision
                and str(output_path) in write_paths
                and isinstance(revisions, dict)
                and revisions.get(str(output_path)) == expected_revision
            ):
                return True
        return False

    def begin(
        self,
        *,
        purpose: str,
        schema_version: int,
        model: str,
        input_revision: str,
        request: dict[str, Any],
    ) -> Path:
        invocation_id = str(uuid.uuid4())
        path = self.root / f"{self._purpose_name(purpose)}-{invocation_id}"
        path.mkdir(parents=True, exist_ok=False)
        atomic_write_json(path / "request.json", request)
        atomic_write_json(
            path / "receipt.json",
            {
                "invocation_id": invocation_id,
                "purpose": purpose,
                "schema_version": schema_version,
                "model": model,
                "input_revision": input_revision,
                "status": "running",
            },
        )
        return path

    def complete(
        self,
        path: Path,
        *,
        response: Any,
        metadata: dict[str, Any] | None = None,
    ) -> InvocationReceipt:
        receipt_path = path / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "running":
            raise ArtifactError(f"invocation is not running: {path}")
        atomic_write_json(path / "response.json", response)
        receipt["status"] = "completed"
        receipt["response_revision"] = json_revision(response)
        if metadata:
            receipt["metadata"] = metadata
        atomic_write_json(receipt_path, receipt)
        return InvocationReceipt(
            invocation_id=str(receipt["invocation_id"]),
            purpose=str(receipt["purpose"]),
            schema_version=int(receipt["schema_version"]),
            model=str(receipt["model"]),
            input_revision=str(receipt["input_revision"]),
            response=response,
            path=path,
        )

    def fail(self, path: Path, error: BaseException) -> None:
        receipt_path = path / "receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if receipt.get("status") == "completed":
            return
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(receipt_path, receipt)
