"""Fixed public-data evaluation and final export; never loads private answers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

CLASSES = ("EAP", "HPL", "MWS")


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray


def public_dir() -> Path:
    return Path(os.environ["MLEBENCH_PUBLIC_DATA"]).resolve()


@lru_cache(maxsize=1)
def _training_data():
    frame = pd.read_csv(public_dir() / "train.csv")
    if frame[["id", "text", "author"]].isna().any().any():
        raise ValueError("public train.csv contains missing values")
    if set(frame.author) != set(CLASSES) or not frame.id.is_unique:
        raise ValueError("public training classes or IDs are invalid")
    return frame


@lru_cache(maxsize=1)
def _split():
    frame = _training_data()
    train, valid = train_test_split(
        frame, test_size=0.2, stratify=frame.author, random_state=42,
    )
    dataset = DatasetSplit("spooky", train.text.to_numpy(), train.author.to_numpy())
    return dataset, valid.text.to_numpy(), valid.author.to_numpy()


def load_datasets():
    return [_split()[0]]


def _probabilities(model, texts):
    classes = list(model.classes_)
    if set(classes) != set(CLASSES) or len(classes) != len(CLASSES):
        raise ValueError("estimator must predict all three author classes")
    values = np.asarray(model.predict_proba(texts), dtype=float)
    if values.shape != (len(texts), len(CLASSES)):
        raise ValueError("invalid predict_proba shape")
    values = values[:, [classes.index(label) for label in CLASSES]]
    if (not np.isfinite(values).all() or (values < 0).any()
            or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("predict_proba must return finite normalized probabilities")
    return values


def evaluate_config(make_model, params: dict) -> float:
    dataset, texts, labels = _split()
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    return float(log_loss(labels, _probabilities(model, texts), labels=list(CLASSES)))


def preflight_environment() -> dict:
    frame = _training_data()
    test = pd.read_csv(public_dir() / "test.csv")
    sample = pd.read_csv(public_dir() / "sample_submission.csv")
    if "author" in test or not {"id", "text"}.issubset(test.columns):
        raise ValueError("expected the MLE-bench public unlabeled test.csv")
    if list(sample.columns) != ["id", *CLASSES]:
        raise ValueError("unexpected sample submission columns")
    if not test.id.is_unique or set(test.id) != set(sample.id):
        raise ValueError("test/sample IDs differ")
    if set(frame.id) & set(test.id):
        raise ValueError("public training and test IDs overlap")
    _split()
    return {"train_rows": len(frame), "test_rows": len(test), "gpu_required": False}


def export_submission(make_model, params: dict, output: Path) -> None:
    """Refit on all public training rows using the already selected configuration."""
    preflight_environment()
    frame = _training_data()
    test = pd.read_csv(public_dir() / "test.csv")
    dataset = DatasetSplit("spooky", frame.text.to_numpy(), frame.author.to_numpy())
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    submission = pd.DataFrame(_probabilities(model, test.text.to_numpy()), columns=CLASSES)
    submission.insert(0, "id", test.id.to_numpy())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(output)
