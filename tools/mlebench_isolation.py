"""Soft public-data isolation checks for MLE-bench evaluation.

This module deliberately does not claim mount-namespace security.  The
evaluator stages a public-only copy, audits paths, and records the mode as
``soft``.  ``mlebench_launch`` remains an optional stronger operator wrapper.
"""

from __future__ import annotations

import os
from pathlib import Path


MOUNT_PATH = Path("/mnt/mle-public")


class IsolationError(RuntimeError):
    pass


def _namespace_is_distinct() -> bool:
    """Return whether this process is in a mount namespace distinct from PID 1."""
    try:
        return os.stat("/proc/self/ns/mnt").st_ino != os.stat("/proc/1/ns/mnt").st_ino
    except OSError as exc:
        raise IsolationError(f"cannot inspect mount namespace: {exc}") from exc


def require_public_mount(staged_public: Path) -> Path:
    """Return the staged public tree and fail closed only on missing input."""
    staged_public = Path(staged_public).resolve()
    if not staged_public.is_dir():
        raise IsolationError(f"staged public directory is missing: {staged_public}")
    return staged_public


def audit_artifact_paths(artifact: Path, *, forbidden: tuple[str, ...] =
                         ("private", "answer", "answers", "solution", "gold")) -> None:
    """Reject candidate artifacts containing known private/gold path names."""
    artifact = Path(artifact).resolve()
    if not artifact.exists():
        raise IsolationError(f"candidate artifact is missing: {artifact}")
    lowered = {part.lower() for part in artifact.parts}
    hits = lowered.intersection(forbidden)
    if hits:
        raise IsolationError(f"candidate artifact references forbidden path parts: {sorted(hits)}")
    for path in artifact.rglob("*"):
        if path.name.lower() in forbidden:
            raise IsolationError(f"candidate artifact contains forbidden path: {path.relative_to(artifact)}")
