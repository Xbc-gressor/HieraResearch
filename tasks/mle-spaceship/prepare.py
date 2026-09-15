"""Fixed public-data evaluation and final export; never loads private answers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tools.mle_resource_probe import run_resource_probe, run_smoke_probe

FEATURES = (
    "PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name",
)
TARGET = "Transported"


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    x_train: pd.DataFrame
    y_train: np.ndarray


def public_dir() -> Path:
    return Path(os.environ["MLEBENCH_PUBLIC_DATA"]).resolve()


@lru_cache(maxsize=1)
def _training_data():
    frame = pd.read_csv(public_dir() / "train.csv")
    if list(frame.columns) != [*FEATURES, TARGET]:
        raise ValueError("public train.csv columns are invalid")
    if not frame.PassengerId.is_unique or frame[TARGET].isna().any():
        raise ValueError("public training IDs or labels are invalid")
    if frame[TARGET].dtype != bool:
        raise ValueError("public training labels must be boolean")
    return frame


@lru_cache(maxsize=1)
def _split():
    frame = _training_data()
    train, valid = train_test_split(
        frame, test_size=0.2, stratify=frame[TARGET], random_state=42,
    )
    dataset = DatasetSplit("spaceship", train[list(FEATURES)], train[TARGET].to_numpy())
    return dataset, valid[list(FEATURES)], valid[TARGET].to_numpy()


def load_datasets():
    return [_split()[0]]


def _predictions(model, features: pd.DataFrame) -> np.ndarray:
    values = pd.Series(np.asarray(model.predict(features)))
    if len(values) != len(features):
        raise ValueError("predict must return one label per row")
    if not values.isin([True, False, 0, 1]).all():
        raise ValueError("predictions must be boolean Transported labels")
    return values.astype(bool).to_numpy()


def evaluate_config(make_model, params: dict) -> float:
    """Score is 1 - validation accuracy (lower is better)."""
    dataset, features, labels = _split()
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    return float(1.0 - accuracy_score(labels, _predictions(model, features)))


def preflight_environment() -> dict:
    frame = _training_data()
    test = pd.read_csv(public_dir() / "test.csv")
    sample = pd.read_csv(public_dir() / "sample_submission.csv")
    if list(test.columns) != list(FEATURES):
        raise ValueError("expected the MLE-bench public unlabeled test.csv")
    if list(sample.columns) != ["PassengerId", TARGET]:
        raise ValueError("unexpected sample submission columns")
    if not test.PassengerId.is_unique or set(test.PassengerId) != set(sample.PassengerId):
        raise ValueError("test/sample IDs differ")
    if set(frame.PassengerId) & set(test.PassengerId):
        raise ValueError("public training and test IDs overlap")
    _split()
    return {"train_rows": len(frame), "test_rows": len(test), "gpu_required": False}


def preflight_config(make_model, params: dict) -> dict:
    """No-score seconds-scale smoke: construct and fit a subsample."""
    return run_smoke_probe(make_model, params, _split()[0])


def resource_probe_config(make_model, params: dict) -> dict:
    """No-score resource envelope for tuner search-space clamping."""
    return run_resource_probe(make_model, params, _split()[0])


def export_submission(make_model, params: dict, output: Path) -> None:
    """Refit on all public training rows using the already selected configuration."""
    preflight_environment()
    frame = _training_data()
    test = pd.read_csv(public_dir() / "test.csv")
    dataset = DatasetSplit("spaceship", frame[list(FEATURES)], frame[TARGET].to_numpy())
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    submission = pd.DataFrame({
        "PassengerId": test.PassengerId.to_numpy(),
        TARGET: _predictions(model, test[list(FEATURES)]),
    })
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(output)
