"""Best-effort host hardware facts shared by run preflight and leases."""

from __future__ import annotations

import os
import platform
from pathlib import Path
import subprocess


def _memory_total_mb() -> float | None:
    """Read host RAM without adding a psutil dependency."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(float(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


def snapshot(*, devices: list[str] | None = None) -> dict:
    """Return stable host facts plus best-effort visible GPU facts.

    ``devices`` may contain CUDA-visible indices or UUIDs.  It is used by a
    lease to bind the snapshot to the devices actually held; without it the
    snapshot describes the operator-visible pool.
    """
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    proc = None
    try:
        proc = subprocess.run(
            query, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        pass

    requested = {str(value) for value in devices} if devices else None
    gpus: list[dict] = []
    if proc is not None and proc.returncode == 0:
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                continue
            index, uuid, name, driver, compute_cap = parts[:5]
            if requested is not None and index not in requested and uuid not in requested:
                continue
            try:
                total_mb = float(parts[5])
                free_mb = float(parts[6])
            except ValueError:
                continue
            gpus.append(
                {
                    "index": index,
                    "uuid": uuid,
                    "name": name,
                    "driver_version": driver,
                    "compute_capability": compute_cap,
                    "total_vram_mb": total_mb,
                    "free_vram_mb": free_mb,
                }
            )

    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_total_mb": _memory_total_mb(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpus,
    }
    if proc is not None and proc.returncode != 0:
        result["gpu_probe_error"] = (proc.stderr or "nvidia-smi failed").strip()[:500]
    return result
