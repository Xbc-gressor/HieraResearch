import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RANK_RGPE = REPO / "tools/inner_benchmark/hebo_mace/rank_rgpe.py"

# Minimal search space
SEARCH_SPACE = {
    "lr": ["float", 1e-4, 1e-1, "log"],
    "hidden": ["int", 16, 128],
    "opt": ["categorical", ["adam", "sgd"]],
}

HISTORY = [
    {"params": {"lr": 0.01, "hidden": 32, "opt": "adam"}, "score": 0.5},
    {"params": {"lr": 0.005, "hidden": 64, "opt": "sgd"}, "score": 0.4},
    {"params": {"lr": 0.02, "hidden": 32, "opt": "adam"}, "score": 0.6},
]

POOL = [
    {"params": {"lr": 0.008, "hidden": 48, "opt": "adam"}},
    {"params": {"lr": 0.003, "hidden": 64, "opt": "sgd"}},
]

BASE_TASKS = [
    {
        "name": "donor_0",
        "shared_dims": ["lr", "hidden", "opt"],
        "points": [
            {"params": {"lr": 0.01, "hidden": 32, "opt": "adam"}, "y_std": 0.2},
            {"params": {"lr": 0.005, "hidden": 64, "opt": "sgd"}, "y_std": -0.5},
            {"params": {"lr": 0.05, "hidden": 16, "opt": "adam"}, "y_std": 1.1},
            {"params": {"lr": 0.001, "hidden": 128, "opt": "sgd"}, "y_std": -0.8},
            {"params": {"lr": 0.02, "hidden": 64, "opt": "adam"}, "y_std": 0.0},
        ],
    }
]


def run_payload(payload):
    proc = subprocess.Popen(
        [sys.executable, str(RANK_RGPE)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate(input=json.dumps(payload))
    assert proc.returncode == 0, f"Failed: {stderr}\nstdout: {stdout}"
    return json.loads(stdout)


def test_without_base_tasks():
    payload = {
        "search_space": SEARCH_SPACE,
        "history": HISTORY,
        "pool": [p["params"] for p in POOL],
        "seed": 42,
    }
    res = run_payload(payload)
    assert "values" in res
    assert len(res["values"]) == len(POOL)
    assert all(len(row) == 3 for row in res["values"])
    assert "rgpe" not in res
    print("test_without_base_tasks passed")


def test_with_base_tasks():
    payload = {
        "search_space": SEARCH_SPACE,
        "history": HISTORY,
        "pool": [p["params"] for p in POOL],
        "seed": 42,
        "base_tasks": BASE_TASKS,
        "rgpe_horizon": 24,
        "rgpe_bootstrap": 100,
    }
    res = run_payload(payload)
    assert "values" in res
    assert len(res["values"]) == len(POOL)
    assert all(len(row) == 3 for row in res["values"])
    assert "rgpe" in res
    assert "__target__" in res["rgpe"]["weights"]
    assert "donor_0" in res["rgpe"]["weights"]
    assert res["rgpe"]["n_target"] == 3
    assert res["rgpe"]["horizon"] == 24
    print("test_with_base_tasks passed, rgpe:", res["rgpe"])


if __name__ == "__main__":
    test_without_base_tasks()
    test_with_base_tasks()
