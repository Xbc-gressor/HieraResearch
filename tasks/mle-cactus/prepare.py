"""Fixed public-data evaluation and final export; never loads private answers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import io
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tools.mle_resource_probe import run_resource_probe, run_smoke_probe

TARGET = "has_cactus"
IMAGE_SIZE = (32, 32)


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
    if list(frame.columns) != ["id", TARGET]:
        raise ValueError("public train.csv columns are invalid")
    if not frame.id.is_unique or not set(frame[TARGET]) <= {0, 1}:
        raise ValueError("public training IDs or labels are invalid")
    return frame


@lru_cache(maxsize=None)
def _images(zip_name: str) -> dict[str, np.ndarray]:
    images = {}
    with zipfile.ZipFile(public_dir() / zip_name) as archive:
        for name in archive.namelist():
            image = Image.open(io.BytesIO(archive.read(name))).convert("RGB")
            if image.size != IMAGE_SIZE:
                raise ValueError(f"unexpected image size for {name}: {image.size}")
            images[name] = np.asarray(image, dtype=np.uint8)
    return images


def _stack(images: dict[str, np.ndarray], ids) -> np.ndarray:
    return np.stack([images[image_id] for image_id in ids])


@lru_cache(maxsize=1)
def _split():
    frame = _training_data()
    train, valid = train_test_split(
        frame, test_size=0.2, stratify=frame[TARGET], random_state=42,
    )
    images = _images("train.zip")
    dataset = DatasetSplit("cactus", _stack(images, train.id), train[TARGET].to_numpy())
    return dataset, _stack(images, valid.id), valid[TARGET].to_numpy()


def load_datasets():
    return [_split()[0]]


def _positive_scores(model, images: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if set(classes) != {0, 1} or len(classes) != 2:
        raise ValueError("estimator must predict both has_cactus classes")
    values = np.asarray(model.predict_proba(images), dtype=float)
    if values.shape != (len(images), 2):
        raise ValueError("invalid predict_proba shape")
    scores = values[:, classes.index(1)]
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("predict_proba must return finite probabilities")
    return scores


def evaluate_config(make_model, params: dict) -> float:
    """Score is 1 - validation ROC AUC (lower is better)."""
    dataset, images, labels = _split()
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    return float(1.0 - roc_auc_score(labels, _positive_scores(model, images)))


def preflight_environment() -> dict:
    frame = _training_data()
    sample = pd.read_csv(public_dir() / "sample_submission.csv")
    if list(sample.columns) != ["id", TARGET]:
        raise ValueError("unexpected sample submission columns")
    train_ids = set(_images("train.zip"))
    test_ids = set(_images("test.zip"))
    if set(frame.id) != train_ids:
        raise ValueError("train.csv IDs do not match train.zip entries")
    if not sample.id.is_unique or set(sample.id) != test_ids:
        raise ValueError("sample submission IDs do not match test.zip entries")
    if train_ids & test_ids:
        raise ValueError("public training and test images overlap")
    _split()
    return {"train_rows": len(frame), "test_rows": len(sample), "gpu_required": False}


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
    sample = pd.read_csv(public_dir() / "sample_submission.csv")
    train_images = _images("train.zip")
    test_images = _images("test.zip")
    dataset = DatasetSplit("cactus", _stack(train_images, frame.id), frame[TARGET].to_numpy())
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    scores = _positive_scores(model, _stack(test_images, sample.id))
    submission = pd.DataFrame({"id": sample.id.to_numpy(), TARGET: scores})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(output)
