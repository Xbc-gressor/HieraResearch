"""Host-local resource leases for objective execution.

The experiment driver may run in several OS processes (different tags), but a
GPU objective is not a shareable background chore.  A task that declares
``[resources].accelerator = "cuda"`` holds this process-wide flock for the
entire warm-screening job, Phase-C search invocation, or hillclimb evaluation.
There is deliberately no timeout: callers wait for ownership, then run
serially.  CPU tasks do not acquire the lease.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time


CUDA_LOCK_PATH = Path("/tmp/shanhai-objective-cuda.lock")


@contextmanager
def task_resource_lease(task_toml: dict, *, owner: dict | None = None):
    resources = task_toml.get("resources", {})
    accelerator = resources.get("accelerator") if isinstance(resources, dict) else None
    if accelerator is None:
        yield
        return
    if accelerator != "cuda":
        raise ValueError(f"unsupported task accelerator: {accelerator!r}")

    CUDA_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CUDA_LOCK_PATH.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    **(owner or {}),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        try:
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "status": "released",
                        "released_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        **(owner or {}),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
