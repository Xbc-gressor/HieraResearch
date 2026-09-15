from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.mle_resource_probe import (  # noqa: E402
    run_resource_probe,
    run_smoke_probe,
)
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


def test_smoke_probe_subsamples_rows_and_caps_epochs():
    rows = 10_000
    dataset = Dataset(np.zeros((rows, 2)), np.zeros(rows))
    seen = {}

    def make_model(received, params):
        seen["dataset"] = received
        seen["params"] = params
        return Model()

    result = run_smoke_probe(make_model, {"epochs": 5, "lr": 0.1}, dataset)

    assert result["status"] == "ok"
    assert result["objective_calls"] == 0
    assert result["probe"] == "smoke"
    assert result["probe_rows"] == max(64, rows // 50)
    assert result["capped_epoch_keys"] == ["epochs"]
    assert seen["params"] == {"epochs": 1, "lr": 0.1}
    assert len(seen["dataset"].x_train) == result["probe_rows"]
    assert len(seen["dataset"].y_train) == result["probe_rows"]
    # The caller's full-shape split is never mutated.
    assert len(dataset.x_train) == rows


def test_smoke_probe_preserves_dataframe_splits():
    """insults/spaceship/nomad2018 score surfaces hand DataFrames to candidates."""
    import pandas as pd

    @dataclass
    class FrameDataset:
        name: str
        x_train: pd.DataFrame
        y_train: np.ndarray

    rows = 10_000
    dataset = FrameDataset(
        "frame",
        pd.DataFrame({"text": [f"row {i}" for i in range(rows)]}),
        np.zeros(rows),
    )
    seen = {}

    def make_model(received, _params):
        seen["x_train"] = received.x_train
        seen["y_train"] = received.y_train
        return Model()

    result = run_smoke_probe(make_model, {}, dataset)

    assert result["probe_rows"] == max(64, rows // 50)
    assert isinstance(seen["x_train"], pd.DataFrame)
    assert list(seen["x_train"].columns) == ["text"]
    assert len(seen["x_train"]) == result["probe_rows"]
    assert len(seen["x_train"].dropna()) == result["probe_rows"]
    assert isinstance(seen["y_train"], np.ndarray)
    assert len(seen["y_train"]) == result["probe_rows"]
    # The caller's full-shape split is never mutated.
    assert len(dataset.x_train) == rows


def test_smoke_probe_keeps_small_datasets_whole():
    dataset = Dataset(np.zeros((4, 2)), np.zeros(4))

    def make_model(received, _params):
        assert received is dataset
        return Model()

    result = run_smoke_probe(make_model, {}, dataset)

    assert result["status"] == "ok"
    assert result["probe_rows"] == 4
    assert result["capped_epoch_keys"] == []


def test_eval_loaders_resolve_prepare_tools_imports(tmp_path):
    """MLE prepare.py imports tools.mle_resource_probe; eval loaders must expose the repo root."""
    sys.path.insert(0, str(ROOT / "tools"))
    import preflight_env  # noqa: E402

    module = preflight_env._load_prepare(ROOT / "tasks" / "mle-spooky" / "prepare.py")
    assert callable(module.resource_probe_config)
    assert callable(module.preflight_config)

    candidate = tmp_path / "001"
    candidate.mkdir()
    (candidate / "prepare.py").write_text(
        (ROOT / "tasks" / "mle-spooky" / "prepare.py").read_text()
    )
    (candidate / "train.py").write_text(
        "BASE_PARAMS = {}\nSEARCH_SPACE = {}\nPARAM_SCHEMA = {}\n"
        "def make_model(dataset, params):\n    return None\n"
    )
    sys.path.insert(0, str(ROOT / "tools" / "tuners"))
    from _common import load_candidate_modules  # noqa: E402

    _, prepare_module = load_candidate_modules(
        candidate / "train.py",
        required_symbols=("make_model",),
    )
    assert callable(prepare_module.resource_probe_config)


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
