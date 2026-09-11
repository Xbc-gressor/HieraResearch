"""Fixed public image-to-image evaluation and export; never loads answers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


IMAGE_SUFFIX = ".png"


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray


def public_dir() -> Path:
    return Path(os.environ["MLEBENCH_PUBLIC_DATA"]).resolve()


def _read_images(directory: Path, names: tuple[str, ...]) -> np.ndarray:
    values = []
    shape = None
    for name in names:
        with Image.open(directory / name) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        if shape is None:
            shape = array.shape
        if array.shape != shape:
            raise ValueError("all public document images must have one shape")
        values.append(array)
    if not values:
        raise ValueError(f"no PNG images found in {directory}")
    return np.stack(values)


@lru_cache(maxsize=1)
def _training_names() -> tuple[str, ...]:
    dirty = {path.name for path in (public_dir() / "train").glob(f"*{IMAGE_SUFFIX}")}
    clean = {path.name for path in (public_dir() / "train_cleaned").glob(f"*{IMAGE_SUFFIX}")}
    if not dirty or dirty != clean:
        raise ValueError("public train and train_cleaned image names must match")
    return tuple(sorted(dirty))


@lru_cache(maxsize=1)
def _split() -> tuple[DatasetSplit, np.ndarray, np.ndarray]:
    names = _training_names()
    train_names, valid_names = train_test_split(names, test_size=0.2, random_state=0)
    root = public_dir()
    train_dirty = _read_images(root / "train", tuple(train_names))
    train_clean = _read_images(root / "train_cleaned", tuple(train_names))
    valid_dirty = _read_images(root / "train", tuple(valid_names))
    valid_clean = _read_images(root / "train_cleaned", tuple(valid_names))
    return DatasetSplit("denoising", train_dirty, train_clean), valid_dirty, valid_clean


def load_datasets():
    return [_split()[0]]


def _predictions(model, images: np.ndarray, targets: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(model.predict(images), dtype=np.float32)
    if values.shape != images.shape:
        raise ValueError(f"predict returned {values.shape}, expected {images.shape}")
    if not np.isfinite(values).all():
        raise ValueError("predict must return finite pixel values")
    # Pixel values outside the image domain are never useful for this metric.
    return np.clip(values, 0.0, 1.0)


def evaluate_config(make_model, params: dict) -> float:
    dataset, valid_dirty, valid_clean = _split()
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    prediction = _predictions(model, valid_dirty)
    return float(np.sqrt(np.mean((prediction - valid_clean) ** 2)))


def preflight_environment() -> dict:
    root = public_dir()
    forbidden = ("answers.csv", "private", "solution.csv")
    leaked = [name for name in forbidden if (root / name).exists()]
    if leaked:
        raise ValueError(f"private answer artifacts are not allowed in public data: {leaked}")
    names = _training_names()
    test_names = tuple(sorted(path.name for path in (root / "test").glob(f"*{IMAGE_SUFFIX}")))
    if not test_names or set(names) & set(test_names):
        raise ValueError("public train and test image names must be nonempty and disjoint")
    sample_path = root / "sampleSubmission.csv"
    sample = pd.read_csv(sample_path)
    if list(sample.columns) != ["id", "value"] or sample.id.isna().any():
        raise ValueError("sampleSubmission.csv must contain id,value")
    # Decode every public image and verify that sample IDs describe test pixels.
    test_images = _read_images(root / "test", test_names)
    expected = {
        f"{Path(name).stem}_{row + 1}_{col + 1}"
        for name, image in zip(test_names, test_images)
        for row in range(image.shape[0])
        for col in range(image.shape[1])
    }
    if set(sample.id) != expected or sample.id.duplicated().any():
        raise ValueError("sampleSubmission.csv IDs do not match public test images")
    _split()
    return {
        "train_rows": len(names),
        "test_rows": len(test_names),
        "test_pixels": len(sample),
        "gpu_required": False,
    }


def _submission_values(model, names: tuple[str, ...]) -> dict[str, float]:
    images = _read_images(public_dir() / "test", names)
    values = _predictions(model, images)
    output = {}
    for name, image in zip(names, values):
        stem = Path(name).stem
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                output[f"{stem}_{row + 1}_{col + 1}"] = float(image[row, col])
    return output


def export_submission(make_model, params: dict, output: Path) -> None:
    preflight_environment()
    names = _training_names()
    root = public_dir()
    dataset = DatasetSplit(
        "denoising",
        _read_images(root / "train", names),
        _read_images(root / "train_cleaned", names),
    )
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    sample = pd.read_csv(root / "sampleSubmission.csv")
    predictions = _submission_values(model, tuple(sorted(path.name for path in (root / "test").glob("*.png"))))
    if set(sample.id) != set(predictions):
        raise ValueError("prediction IDs do not match sampleSubmission.csv")
    submission = pd.DataFrame({"id": sample.id, "value": sample.id.map(predictions)})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(output)
