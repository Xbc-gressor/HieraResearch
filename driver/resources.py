"""Host-local, per-device resource leases for objective execution.

The experiment driver may run in several OS processes (different tags), and a
GPU objective must never share a physical GPU with another evaluation.  A task
that declares ``[resources].accelerator = "cuda"`` therefore leases whole
devices, one flock per device file, for the entire warm-screening job, Phase-C
search invocation, or hillclimb evaluation.  Different devices lease
concurrently; the same device serializes.  While the lease is held,
``CUDA_VISIBLE_DEVICES`` is pinned to the leased device(s) so child processes
land on the GPU the lease actually owns.

``[resources].devices = N`` (default 1) leases N distinct devices — e.g. the
DDP task, whose world size spans several GPUs.  N is capped at the visible
pool size, so a single-GPU box still runs (the task itself then fails or
regresses exactly as it would without pinning).

The device pool is the operator-set ``CUDA_VISIBLE_DEVICES`` when present,
else every device reported by ``nvidia-smi``.  There is deliberately no
timeout: callers wait for free devices, then run.  CPU tasks do not acquire
any lease.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


_LOCK_DIR_ENV = "SHANHAI_CUDA_LOCK_DIR"
_DEFAULT_LOCK_DIR = Path("/tmp")
_LOCK_NAME = "shanhai-objective-cuda-{dev}.lock"
_POLL_SECONDS = 1.0


def _lock_dir() -> Path:
    return Path(os.environ.get(_LOCK_DIR_ENV, str(_DEFAULT_LOCK_DIR)))


def _visible_devices() -> list[str]:
    """Device ids leasable on this host, as CUDA-visible tokens."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        devices = [dev.strip() for dev in env.split(",") if dev.strip()]
        if devices:
            return devices
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "cuda task but cannot enumerate devices: nvidia-smi failed"
        ) from exc
    devices = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if probe.returncode != 0 or not devices:
        raise RuntimeError(
            "cuda task but cannot enumerate devices: nvidia-smi reports none"
        )
    return devices


def _try_acquire(pool: list[str], want: int) -> list[tuple[str, object]]:
    """One non-blocking round: flock free device files until ``want`` held."""
    held: list[tuple[str, object]] = []
    for dev in pool:
        if len(held) >= want:
            break
        handle = (_lock_dir() / _LOCK_NAME.format(dev=dev)).open(
            "a+", encoding="utf-8"
        )
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        held.append((dev, handle))
    if len(held) < want:
        # Partial sets would starve multi-device contenders; close releases.
        for _, handle in held:
            handle.close()
        return []
    return held


def _write_record(handle, record: dict) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


@contextmanager
def task_resource_lease(task_toml: dict, *, owner: dict | None = None):
    resources = task_toml.get("resources", {})
    accelerator = resources.get("accelerator") if isinstance(resources, dict) else None
    if accelerator is None:
        yield
        return
    if accelerator != "cuda":
        raise ValueError(f"unsupported task accelerator: {accelerator!r}")
    devices = int(resources.get("devices", 1))
    if devices < 1:
        raise ValueError(f"[resources].devices must be >= 1, got {devices}")

    pool = _visible_devices()
    want = min(devices, len(pool))
    _lock_dir().mkdir(parents=True, exist_ok=True)
    while True:
        held = _try_acquire(pool, want)
        if held:
            break
        time.sleep(_POLL_SECONDS)

    leased = [dev for dev, _ in held]
    for dev, handle in held:
        _write_record(
            handle,
            {
                "pid": os.getpid(),
                "device": dev,
                "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **(owner or {}),
            },
        )
    previous_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(leased)
    try:
        yield
    finally:
        if previous_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cvd
        for dev, handle in held:
            _write_record(
                handle,
                {
                    "pid": os.getpid(),
                    "device": dev,
                    "status": "released",
                    "released_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    **(owner or {}),
                },
            )
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
