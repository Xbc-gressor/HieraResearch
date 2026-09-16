"""Host-local, per-device resource leases for objective execution.

The experiment driver may run in several OS processes (different tags), and a
GPU objective must never share a physical GPU with another evaluation.  A task
that declares ``[resources].accelerator = "cuda"`` therefore leases whole
devices, one flock per device file, for the entire warm-screening job, Phase-C
search invocation, or hillclimb evaluation.  Different devices lease
concurrently; the same device serializes.  The lease yields the child-process
environment (``result["env"]`` pins ``CUDA_VISIBLE_DEVICES`` to the leased
device(s)); the driver's own ``os.environ`` is never rewritten, so several
in-process channels can hold or await leases without polluting each other.

``[resources].devices = N`` (default 1) leases N distinct devices — e.g. the
DDP task, whose world size spans several GPUs.  N is capped at the visible
pool size, so a single-GPU box still runs (the task itself then fails or
regresses exactly as it would without pinning).

The device pool is the operator-set ``CUDA_VISIBLE_DEVICES`` when present,
else every device reported by ``nvidia-smi``.  Queueing for a free device is
bounded by ``wait_timeout`` (caller override, else ``[resources]
.lease_wait_timeout``, default 600 seconds; ``None`` = wait indefinitely); an
expired queue and an under-provisioned device both raise
``ResourceUnavailable``, which callers treat as transient.  CPU tasks do not
acquire any lease.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from tools.hardware import snapshot as hardware_snapshot


_LOCK_DIR_ENV = "SHANHAI_CUDA_LOCK_DIR"
_DEFAULT_LOCK_DIR = Path("/tmp")
_LOCK_NAME = "shanhai-objective-cuda-{dev}.lock"
_POLL_SECONDS = 1.0


class ResourceUnavailable(RuntimeError):
    """The host cannot currently grant the task's declared resources.

    Transient by construction: callers retry or defer.  It is never a
    candidate observation and must not be recorded as one.
    """


class ResourcePreflightError(ResourceUnavailable):
    """A leased device does not satisfy the task's declared resources."""


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


def _run_dir_for_owner(owner: dict | None) -> Path | None:
    """Resolve the enclosing run directory for any lease caller."""
    raw = (owner or {}).get("run_dir")
    if not raw:
        return None
    path = Path(str(raw)).resolve()
    if (path / "framework_cfg.json").is_file():
        return path
    for ancestor in (path, *path.parents):
        if (ancestor / "framework_cfg.json").is_file():
            return ancestor
    return None


def _append_lease_event(run_dir: Path | None, event: dict) -> None:
    """Append one lease observation without making telemetry fatal."""
    if run_dir is None:
        return
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "resource_leases.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    except OSError:
        # Resource accounting must never turn a successful evaluation into a
        # candidate failure. The lock itself remains authoritative.
        return


def check_gpu_memory(min_gib: float, *, device: str | None = None) -> dict:
    """Check free memory after a lease has been acquired.

    This is a preflight observation, not a hard guarantee against processes
    outside the lease; ``device`` selects the leased device explicitly.
    """
    if min_gib < 0:
        raise ValueError("min_gib must be >= 0")
    query = ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"]
    if device is not None:
        query[1:1] = ["-i", str(device)]
    try:
        probe = subprocess.run(query, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourcePreflightError("unable to query GPU memory") from exc
    if probe.returncode != 0 or not probe.stdout.strip():
        raise ResourcePreflightError("nvidia-smi returned no memory data")
    free_mib, total_mib = (float(part.strip()) for part in probe.stdout.splitlines()[0].split(",", 1))
    result = {"free_gib": free_mib / 1024.0, "total_gib": total_mib / 1024.0, "min_gib": float(min_gib)}
    if result["free_gib"] < min_gib:
        raise ResourcePreflightError(f"GPU free memory {result['free_gib']:.2f} GiB below {min_gib:.2f} GiB")
    return result


NO_LEASE = {"devices": [], "env": {}, "lease_wait_seconds": 0.0}


@contextmanager
def task_resource_lease(task_toml: dict, *, owner: dict | None = None,
                        wait_timeout: float | None = ...):
    resources = task_toml.get("resources", {})
    accelerator = resources.get("accelerator") if isinstance(resources, dict) else None
    if accelerator is None:
        yield dict(NO_LEASE)
        return
    if accelerator != "cuda":
        raise ValueError(f"unsupported task accelerator: {accelerator!r}")
    devices = int(resources.get("devices", 1))
    if devices < 1:
        raise ValueError(f"[resources].devices must be >= 1, got {devices}")

    pool = _visible_devices()
    want = min(devices, len(pool))
    _lock_dir().mkdir(parents=True, exist_ok=True)
    if wait_timeout is ...:
        wait_timeout = resources.get("lease_wait_timeout", 600)
        try:
            wait_timeout = float(wait_timeout)
        except (TypeError, ValueError):
            raise ValueError("[resources].lease_wait_timeout must be a number")
        if wait_timeout < 0:
            raise ValueError("[resources].lease_wait_timeout must be >= 0")
    started_wait = time.monotonic()
    while True:
        held = _try_acquire(pool, want)
        if held:
            break
        waited = time.monotonic() - started_wait
        if wait_timeout is not None and waited >= wait_timeout:
            raise ResourceUnavailable(
                f"timed out waiting {waited:.3f}s for {want} CUDA device lease"
            )
        remaining = _POLL_SECONDS if wait_timeout is None else wait_timeout - waited
        time.sleep(min(_POLL_SECONDS, max(0.01, remaining)))

    leased = [dev for dev, _ in held]
    run_dir = _run_dir_for_owner(owner)
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
    lease_event = {
        "schema_version": 1,
        "status": "acquired",
        "devices": leased,
        "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(owner or {}),
    }
    try:
        result = {
            "devices": leased,
            "env": {"CUDA_VISIBLE_DEVICES": ",".join(leased)},
            "lease_wait_seconds": time.monotonic() - started_wait,
        }
        minimum = resources.get("min_memory_gib")
        if minimum is not None:
            memory = [
                check_gpu_memory(float(minimum), device=device)
                for device in leased
            ]
            result["memory"] = memory[0] if len(memory) == 1 else {
                "devices": memory,
            }
        result["hardware"] = hardware_snapshot(devices=leased)
        lease_event.update(
            {
                "lease_wait_seconds": result["lease_wait_seconds"],
                "memory": result.get("memory"),
                "hardware": result["hardware"],
            }
        )
        _append_lease_event(run_dir, lease_event)
        yield result
    finally:
        _append_lease_event(
            run_dir,
            {
                "schema_version": 1,
                "status": "released",
                "devices": leased,
                "released_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **(owner or {}),
            },
        )
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
