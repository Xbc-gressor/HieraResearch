from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.mle_resource_probe import run_resource_probe  # noqa: E402
from tools.tuners._common import _env_total_vram_mb  # noqa: E402


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray


class Model:
    def __init__(self):
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, x, y):
        self.fit_calls += 1
        assert len(x) == len(y)
        return self

    def predict(self, _x):
        self.predict_calls += 1
        raise AssertionError("resource probe must not score or predict")


def test_resource_probe_is_no_score_and_full_shape():
    dataset = Dataset(np.zeros((4, 2)), np.zeros(4))
    models = []

    def make_model(received, _params):
        assert received is dataset
        model = Model()
        models.append(model)
        return model

    result = run_resource_probe(make_model, {}, dataset)
    assert result["status"] == "ok"
    assert result["objective_calls"] == 0
    assert result["probe_rows"] == 4
    assert result["envelope_covers_worst_case"] is True
    assert models[0].fit_calls == 1
    assert models[0].predict_calls == 0


def test_all_mle_cuda_tasks_declare_both_probe_hooks():
    import tomllib

    for task in (
        "mle-spooky",
        "mle-cactus",
        "mle-denoising",
        "mle-insults",
        "mle-nomad2018",
        "mle-spaceship",
    ):
        config = tomllib.loads(
            (ROOT / "tasks" / task / "task.toml").read_text()
        )
        evaluation = config["evaluation"]
        assert evaluation["preflight_fn"] == "preflight_config"
        assert evaluation["resource_probe_fn"] == "resource_probe_config"


def test_lease_vram_is_authoritative_over_environment_snapshot(tmp_path):
    run_dir = tmp_path / "runs" / "task" / "tag"
    run_dir.mkdir(parents=True)
    (run_dir / "framework_cfg.json").write_text("{}")
    candidate = run_dir / "candidates" / "001" / "train.py"
    candidate.parent.mkdir(parents=True)
    (run_dir / "environment_preflight.json").write_text(
        '{"hardware": {"gpus": [{"total_vram_mb": 24576}]}}'
    )
    (run_dir / "resource_leases.jsonl").write_text(
        '{"status":"acquired","hardware":{"gpus":[{"total_vram_mb":12288}]}}\n'
    )
    assert _env_total_vram_mb(candidate) == 12288
