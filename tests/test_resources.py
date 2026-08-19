"""Per-device cuda lease: undo/driver/resources.py.

Regression: the lease used to be one host-wide lock file, so two experiments
on two different GPUs serialized.  The lease is per device now; only a same-
device conflict waits.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver import resources  # noqa: E402

CUDA_TOML = {"resources": {"accelerator": "cuda"}}


@pytest.fixture
def pool_env(monkeypatch, tmp_path):
    monkeypatch.setenv(resources._LOCK_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(resources, "_POLL_SECONDS", 0.05)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return monkeypatch


def _lease_in_thread(devices_seen: list[str], done: threading.Event) -> None:
    with resources.task_resource_lease(CUDA_TOML):
        devices_seen.append(os.environ["CUDA_VISIBLE_DEVICES"])
        done.set()


def test_distinct_devices_lease_concurrently(pool_env, monkeypatch):
    monkeypatch.setattr(resources, "_visible_devices", lambda: ["0", "1"])
    with resources.task_resource_lease(CUDA_TOML):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        devices_seen, done = [], threading.Event()
        thread = threading.Thread(
            target=_lease_in_thread, args=(devices_seen, done)
        )
        thread.start()
        assert done.wait(timeout=5), "second device must lease while first is held"
        thread.join(timeout=5)
        assert devices_seen == ["1"]
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_same_device_serializes(pool_env, monkeypatch):
    monkeypatch.setattr(resources, "_visible_devices", lambda: ["0"])
    with resources.task_resource_lease(CUDA_TOML):
        devices_seen, done = [], threading.Event()
        thread = threading.Thread(
            target=_lease_in_thread, args=(devices_seen, done)
        )
        thread.start()
        assert not done.wait(timeout=0.5), "same device must wait for the lease"
    thread.join(timeout=5)
    assert done.is_set() and devices_seen == ["0"]


def test_multi_device_lease_pins_all(pool_env, monkeypatch):
    monkeypatch.setattr(resources, "_visible_devices", lambda: ["0", "1"])
    toml = {"resources": {"accelerator": "cuda", "devices": 2}}
    with resources.task_resource_lease(toml):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_cpu_task_leases_nothing(pool_env, monkeypatch):
    monkeypatch.setattr(
        resources,
        "_visible_devices",
        lambda: pytest.fail("CPU task must not enumerate devices"),
    )
    with resources.task_resource_lease({"resources": {}}):
        pass
