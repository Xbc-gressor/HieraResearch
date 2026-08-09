"""Append-only tuning failure artifacts and compact, frozen receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = 1
_FRAME_RE = re.compile(r'^\s*File "([^"]+)", line (\d+), in (.+)$')


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _exception_detail(traceback_text: str, error: BaseException) -> tuple[str, int | None]:
    lines = traceback_text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip():
            return lines[index].strip(), index + 1
    return f"{type(error).__name__}: {error}", None


def _selected_frames(traceback_text: str, candidate_path: Path) -> list[dict[str, Any]]:
    lines = traceback_text.splitlines()
    frames: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = _FRAME_RE.match(line)
        if not match:
            continue
        frame = {
            "path": match.group(1),
            "line": int(match.group(2)),
            "function": match.group(3).strip(),
            "traceback_line": index + 1,
        }
        if index + 1 < len(lines) and not _FRAME_RE.match(lines[index + 1]):
            next_line = lines[index + 1]
            code = next_line.strip()
            if code and next_line[:1].isspace():
                frame["code"] = code
        frames.append(frame)

    selected_indexes = set(range(max(0, len(frames) - 3), len(frames)))
    candidate = str(candidate_path)
    candidate_indexes = [
        index for index, frame in enumerate(frames)
        if frame["path"] == candidate or frame["path"].endswith(f"/{candidate_path.name}")
    ]
    selected_indexes.update(candidate_indexes[-2:])
    return [frames[index] for index in sorted(selected_indexes)]


def record_failure(
    *,
    report_path: Path,
    candidate_path: Path,
    phase: str,
    method: str,
    params: dict[str, Any],
    error: BaseException,
    traceback_text: str,
) -> dict[str, Any]:
    """Persist one failure once and return compact fields for a trial record.

    ``failure_id`` intentionally uses 16 hex characters for a short filename;
    the artifact keeps a full digest and loudly rejects a collision.
    """
    report_path = Path(report_path)
    candidate_path = Path(candidate_path)
    failure = {
        "candidate_path": str(candidate_path),
        "phase": phase,
        "method": method,
        "params": params,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback_text,
    }
    content_sha256 = _sha256(_canonical_json(failure))
    failure_id = f"fail-{content_sha256.removeprefix('sha256:')[:16]}"
    relative_artifact = f"_failures/{failure_id}.json"
    traceback_lines = traceback_text.splitlines()
    frames = _selected_frames(traceback_text, candidate_path)
    exception, exception_traceback_line = _exception_detail(traceback_text, error)
    retained_line_numbers = {frame["traceback_line"] for frame in frames}
    retained_line_numbers.update(
        frame["traceback_line"] + 1 for frame in frames if "code" in frame
    )
    if exception_traceback_line is not None:
        retained_line_numbers.add(exception_traceback_line)
    receipt = {
        "failure_id": failure_id,
        "artifact": relative_artifact,
        "exception": exception,
        "exception_traceback_line": exception_traceback_line,
        "frames": frames,
        "traceback_lines": len(traceback_lines),
        "retained_traceback_lines": len(retained_line_numbers),
        "omitted_traceback_lines": len(traceback_lines) - len(retained_line_numbers),
    }
    artifact_without_hash = {
        "schema_version": SCHEMA_VERSION,
        "failure_id": failure_id,
        "failure": failure,
        "receipt": receipt,
    }
    artifact_sha256 = _sha256(_canonical_json(artifact_without_hash))
    artifact = {**artifact_without_hash, "sha256": artifact_sha256}
    encoded = json.dumps(artifact, indent=2, ensure_ascii=False, default=str).encode("utf-8") + b"\n"

    target = report_path.parent / relative_artifact
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError:
        if target.read_bytes() != encoded:
            raise RuntimeError(f"immutable failure artifact differs: {target}") from None

    return {
        "error": f"{type(error).__name__}: {error}"[:300],
        "failure_ref": {
            "schema_version": SCHEMA_VERSION,
            "failure_id": failure_id,
            "artifact": relative_artifact,
            "sha256": artifact_sha256,
        },
        "failure_receipt": receipt,
    }


def _load_verified(report_path: Path, failure_id: str) -> dict[str, Any]:
    artifact_path = Path(report_path).parent / "_failures" / f"{failure_id}.json"
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("failure_id") != failure_id:
        raise ValueError(f"invalid failure artifact: {artifact_path}")
    expected = artifact.get("sha256")
    unhashed = {key: value for key, value in artifact.items() if key != "sha256"}
    actual = _sha256(_canonical_json(unhashed))
    if expected != actual:
        raise ValueError(f"failure artifact hash mismatch: {artifact_path}")
    return artifact


def render_failure(
    report_path: Path,
    failure_id: str,
    *,
    view: str = "receipt",
    line_range: tuple[int, int] | None = None,
) -> dict[str, Any] | str:
    """Read a frozen receipt or retrieve exact traceback content."""
    artifact = _load_verified(Path(report_path), failure_id)
    if view == "receipt":
        return artifact["receipt"]
    traceback_text = artifact["failure"]["traceback"]
    if view == "full":
        return traceback_text
    if view == "lines":
        if line_range is None:
            raise ValueError("line_range is required for lines view")
        start, end = line_range
        lines = traceback_text.splitlines(keepends=True)
        if start < 1 or end < start or end > len(lines):
            raise ValueError(f"line range must be within 1:{len(lines)}")
        return "".join(lines[start - 1:end])
    raise ValueError(f"unknown failure view: {view}")
