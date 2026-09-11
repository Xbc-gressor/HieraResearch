"""Fail-closed checks for the MLE-bench public-data mount.

The driver may stage data on the host, but a run is safe only when the
session is executing in a separate mount namespace with that staged tree
mounted at ``/mnt/mle-public``.  This module deliberately does not try to
create a namespace; the operator launch wrapper owns that boundary.
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
    """Verify the required namespace and public mount, returning its path."""
    staged_public = Path(staged_public).resolve()
    if not staged_public.is_dir():
        raise IsolationError(f"staged public directory is missing: {staged_public}")
    if os.environ.get("MLEBENCH_NAMESPACE_READY") != "1":
        raise IsolationError("MLEBENCH_NAMESPACE_READY=1 is required for MLE-bench runs")
    if not _namespace_is_distinct():
        raise IsolationError("driver is not running in a distinct mount namespace")
    if not MOUNT_PATH.is_dir():
        raise IsolationError(f"required public mount is missing: {MOUNT_PATH}")
    return MOUNT_PATH

