"""Fixed public-data evaluation and final export; never loads private answers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from tools.mle_resource_probe import run_resource_probe

TARGETS = ("formation_energy_ev_natom", "bandgap_energy_ev")


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    x_train: pd.DataFrame
    y_train: np.ndarray


def public_dir() -> Path:
    return Path(os.environ["MLEBENCH_PUBLIC_DATA"]).resolve()


@lru_cache(maxsize=1)
def _training_data() -> pd.DataFrame:
    frame = pd.read_csv(public_dir() / "train.csv")
    required = {"id", *TARGETS}
    if not required.issubset(frame.columns):
        raise ValueError("public train.csv is missing NOMAD target columns")
    if frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("public training IDs are invalid")
    if frame[list(TARGETS)].isna().any().any():
        raise ValueError("public training targets contain missing values")
    return frame


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.drop(columns=list(TARGETS), errors="ignore").copy()
    values = values.drop(columns=["id"], errors="ignore")
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("NOMAD tabular features must be numeric and non-null")
    return numeric


@lru_cache(maxsize=1)
def _split():
    frame = _training_data()
    train, valid = train_test_split(frame, test_size=0.2, random_state=42)
    dataset = DatasetSplit("nomad2018", _features(train), train[list(TARGETS)].to_numpy())
    return dataset, _features(valid), valid[list(TARGETS)].to_numpy()


def load_datasets():
    return [_split()[0]]


def _prediction(model, features: pd.DataFrame, rows: int) -> np.ndarray:
    values = np.asarray(model.predict(features), dtype=float)
    if values.shape != (rows, len(TARGETS)):
        raise ValueError("model must predict two NOMAD targets per row")
    if not np.isfinite(values).all():
        raise ValueError("predictions must be finite")
    return np.maximum(values, 0.0)


def evaluate_config(make_model, params: dict) -> float:
    dataset, features, labels = _split()
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    values = _prediction(model, features, len(features))
    scores = [mean_squared_log_error(labels[:, i], values[:, i]) ** 0.5
              for i in range(len(TARGETS))]
    return float(np.mean(scores))


def preflight_environment() -> dict:
    frame = _training_data()
    test = pd.read_csv(public_dir() / "test.csv")
    sample = pd.read_csv(public_dir() / "sample_submission.csv")
    if "id" not in test or any(target in test.columns for target in TARGETS):
        raise ValueError("public test must contain IDs without target columns")
    if list(sample.columns) != ["id", *TARGETS]:
        raise ValueError("unexpected NOMAD sample submission columns")
    if test.id.isna().any() or test.id.duplicated().any():
        raise ValueError("public test IDs are invalid")
    if not sample["id"].equals(test["id"]):
        raise ValueError("sample submission and test IDs differ")
    if set(frame.id) & set(test.id):
        raise ValueError("public training and test IDs overlap")
    _features(test)
    _split()
    return {"train_rows": len(frame), "test_rows": len(test), "gpu_required": False}


def preflight_config(make_model, params: dict) -> dict:
    """No-score full training-shape feasibility probe."""
    return run_resource_probe(make_model, params, _split()[0])


def resource_probe_config(make_model, params: dict) -> dict:
    """No-score resource envelope for tuner search-space clamping."""
    return run_resource_probe(make_model, params, _split()[0])


def export_submission(make_model, params: dict, output: Path) -> None:
    preflight_environment()
    frame = _training_data()
    test = pd.read_csv(public_dir() / "test.csv")
    dataset = DatasetSplit("nomad2018", _features(frame), frame[list(TARGETS)].to_numpy())
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    values = _prediction(model, _features(test), len(test))
    submission = pd.DataFrame(values, columns=TARGETS)
    submission.insert(0, "id", test.id.to_numpy())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(output)
