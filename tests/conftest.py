from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_gpu_evaluation_lease(tmp_path_factory):
    """Keep the unit suite away from the host's production GPU lease."""
    name = "HIERARESEARCH_GPU_LOCK_PATH"
    previous = os.environ.get(name)
    lock_path = tmp_path_factory.mktemp("gpu-evaluation-lease") / "lease.lock"
    os.environ[name] = str(lock_path)
    try:
        yield lock_path
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
