"""Fixed public-data evaluation and final export; never loads private answers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
from py7zr.io import BytesIOFactory
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from tools.mle_resource_probe import run_resource_probe, run_smoke_probe

IMAGE_SIZE = 75
N_PIXELS = IMAGE_SIZE * IMAGE_SIZE
BANDS = ("band_1", "band_2")
PIXEL_COLUMNS = tuple(f"{band}_{i:04d}" for band in BANDS for i in range(N_PIXELS))
COLUMNS = PIXEL_COLUMNS + ("inc_angle",)
TRAIN_FIELDS = {"id", "band_1", "band_2", "inc_angle", "is_iceberg"}
TEST_FIELDS = TRAIN_FIELDS - {"is_iceberg"}
ARCHIVE_LIMIT_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    x_train: pd.DataFrame   # one image per row (index = id): band pixels in dB and inc_angle
    y_train: np.ndarray     # (n,) 1 = iceberg, 0 = ship


def public_dir() -> Path:
    return Path(os.environ["MLEBENCH_PUBLIC_DATA"]).resolve()


# --------------------------------------------------------------------------
# Published archives
# --------------------------------------------------------------------------

def read_archive(name: str) -> bytes:
    """Decompressed content of the single file inside ``<public>/<name>`` (a .7z archive)."""
    factory = BytesIOFactory(ARCHIVE_LIMIT_BYTES)
    with py7zr.SevenZipFile(public_dir() / name, "r") as archive:
        archive.extractall(factory=factory)
    if len(factory.products) != 1:
        raise ValueError(f"{name} must contain exactly one file, found {sorted(factory.products)}")
    stream = next(iter(factory.products.values()))
    stream.seek(0)
    return stream.read()


def train_records() -> list[dict]:
    """The published ``train.json`` as a list of records (parsed on every call).

    Each record has ``id``, ``band_1``, ``band_2`` (5625 floats in dB each,
    row-major 75x75), ``inc_angle`` (float, or the string ``"na"``) and
    ``is_iceberg``.
    """
    return json.loads(read_archive("train.json.7z"))


def test_records() -> list[dict]:
    """The published ``test.json``: the same fields as ``train.json`` without ``is_iceberg``."""
    return json.loads(read_archive("test.json.7z"))


@lru_cache(maxsize=1)
def sample_submission() -> pd.DataFrame:
    return pd.read_csv(BytesIO(read_archive("sample_submission.csv.7z")))


def _angle(value) -> float:
    if isinstance(value, str):
        if value.strip().lower() == "na":
            return np.nan
        return float(value)
    return float(value)


def _frame(records: list[dict]) -> pd.DataFrame:
    ids = pd.Index([str(record["id"]) for record in records], name="id")
    if not ids.is_unique:
        raise ValueError("duplicate image ids")
    pixels = np.asarray([record["band_1"] + record["band_2"] for record in records], dtype=np.float32)
    if pixels.shape != (len(records), 2 * N_PIXELS) or not np.isfinite(pixels).all():
        raise ValueError("every image must hold two bands of 5625 finite values")
    frame = pd.DataFrame(pixels, index=ids, columns=list(PIXEL_COLUMNS))
    frame["inc_angle"] = np.asarray([_angle(record["inc_angle"]) for record in records], dtype=np.float32)
    return frame


@lru_cache(maxsize=1)
def train_frame() -> pd.DataFrame:
    """All public training images in file order, indexed by ``id``.

    Columns ``band_1_0000..band_1_5624``, ``band_2_0000..band_2_5624`` (dB,
    row-major 75x75) and ``inc_angle`` (NaN where the file says ``"na"``).
    """
    return _frame(train_records())


@lru_cache(maxsize=1)
def train_labels() -> pd.Series:
    """``is_iceberg`` per training image, indexed by ``id`` in ``train_frame`` order."""
    records = train_records()
    return pd.Series([int(record["is_iceberg"]) for record in records],
                     index=pd.Index([str(record["id"]) for record in records], name="id"),
                     dtype=np.int8)


@lru_cache(maxsize=1)
def test_frame() -> pd.DataFrame:
    """All public test images in file order, with the same columns as ``train_frame``."""
    return _frame(test_records())


def bands(frame: pd.DataFrame) -> np.ndarray:
    """``(n, 2, 75, 75)`` float32 array of the pixel columns of any frame this module produces."""
    return frame[list(PIXEL_COLUMNS)].to_numpy(dtype=np.float32).reshape(len(frame), 2, IMAGE_SIZE, IMAGE_SIZE)


# --------------------------------------------------------------------------
# Fixed split, scoring and probes
# --------------------------------------------------------------------------

def _holdout_ids() -> tuple[list[str], list[str]]:
    labels = train_labels()
    fit_ids, holdout_ids = train_test_split(
        labels.index.to_numpy(), test_size=0.2, stratify=labels.to_numpy(), random_state=42
    )
    return [str(v) for v in fit_ids], [str(v) for v in holdout_ids]


@lru_cache(maxsize=1)
def _split():
    fit_ids, holdout_ids = _holdout_ids()
    frame, labels = train_frame(), train_labels()
    dataset = DatasetSplit("statoil-iceberg", frame.loc[fit_ids], labels.loc[fit_ids].to_numpy())
    return dataset, frame.loc[holdout_ids], labels.loc[holdout_ids].to_numpy()


def load_datasets():
    return [_split()[0]]


def _scores(model, x: pd.DataFrame) -> np.ndarray:
    """Iceberg probability per row from ``predict_proba``.

    Accepts a ``(n,)`` or ``(n, 1)`` array of positive-class probabilities or
    the ``(n, 2)`` layout of binary sklearn classifiers (the positive column is
    located through ``classes_``).
    """
    values = np.asarray(model.predict_proba(x), dtype=float)
    if values.ndim == 2 and values.shape[1] == 2:
        classes = [int(c) for c in getattr(model, "classes_", [0, 1])]
        values = values[:, classes.index(1)]
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.shape != (len(x),):
        raise ValueError("predict_proba must return one iceberg probability per image")
    return _probabilities(values)


def _probabilities(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("predictions contain non-finite values")
    if values.min() < -1e-6 or values.max() > 1 + 1e-6:
        raise ValueError("predictions must be probabilities in [0, 1]")
    return np.clip(values, 0.0, 1.0)


def _log_loss(labels: np.ndarray, values: np.ndarray) -> float:
    """Official metric: binary log loss (sklearn ``log_loss``)."""
    return float(log_loss(labels, values, labels=[0, 1]))


def evaluate_config(make_model, params: dict) -> float:
    dataset, x_holdout, y_holdout = _split()
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    return _log_loss(y_holdout, _scores(model, x_holdout))


def _test_ids() -> list[str]:
    """Test image ids in ``sample_submission.csv`` order."""
    return [str(v) for v in sample_submission()["id"]]


def preflight_environment() -> dict:
    train, test = train_records(), test_records()
    if not train or not test:
        raise ValueError("train.json or test.json is empty")
    if any(set(record) != TRAIN_FIELDS for record in train):
        raise ValueError("train.json fields differ from the published schema")
    if any(set(record) != TEST_FIELDS for record in test):
        raise ValueError("test.json fields differ from the published schema")
    if any(int(record["is_iceberg"]) not in (0, 1) for record in train):
        raise ValueError("is_iceberg must be 0 or 1")
    sample = sample_submission()
    if list(sample.columns) != ["id", "is_iceberg"]:
        raise ValueError("unexpected sample submission columns")
    test_ids = {str(record["id"]) for record in test}
    if not sample["id"].is_unique or {str(v) for v in sample["id"]} != test_ids:
        raise ValueError("sample submission ids do not match test.json")
    if test_ids & set(train_labels().index):
        raise ValueError("train and test ids overlap")
    test_frame()
    _split()
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "submission_rows": len(sample),
        "pixels_per_band": N_PIXELS,
        "train_missing_angles": int(train_frame()["inc_angle"].isna().sum()),
        "test_missing_angles": int(test_frame()["inc_angle"].isna().sum()),
        "train_icebergs": int(train_labels().sum()),
    }


def preflight_config(make_model, params: dict) -> dict:
    """No-score seconds-scale smoke: construct and fit a subsample."""
    return run_smoke_probe(make_model, params, _split()[0])


def resource_probe_config(make_model, params: dict) -> dict:
    """No-score resource envelope for tuner search-space clamping."""
    return run_resource_probe(make_model, params, _split()[0])


def export_submission(make_model, params: dict, output: Path) -> None:
    """Refit on all public training images using the already selected configuration."""
    preflight_environment()
    dataset = DatasetSplit("statoil-iceberg", train_frame(), train_labels().to_numpy())
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    test_ids = _test_ids()
    values = _scores(model, test_frame().loc[test_ids])
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": values})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(output)


def evaluate_protocol(artifact: Path, params: dict, *, stage: str = "protocol",
                      fidelity: str = "full") -> dict:
    """Score a candidate-produced ``protocol_outputs.npz`` in the evaluator process.

    The archive holds ``holdout_ids`` and ``test_ids`` (image ids as strings,
    in the task's holdout order and in sample-submission order) plus
    ``holdout_predictions`` and ``test_submission``, each ``(n,)`` iceberg
    probabilities.  This function owns the holdout labels and validates both
    prediction streams; it never imports candidate code.
    """
    artifact = Path(artifact)
    output = artifact / "protocol_outputs.npz" if artifact.is_dir() else artifact
    if not output.is_file():
        raise ValueError(f"protocol output is missing: {output}")
    values = np.load(output, allow_pickle=False)
    _, holdout_ids = _holdout_ids()
    test_ids = _test_ids()
    if [str(v) for v in values["holdout_ids"]] != holdout_ids:
        raise ValueError("protocol holdout IDs do not match the task split")
    if [str(v) for v in values["test_ids"]] != test_ids:
        raise ValueError("protocol test IDs do not match sample_submission.csv order")
    holdout_predictions = np.asarray(values["holdout_predictions"], dtype=float).reshape(-1)
    test_predictions = np.asarray(values["test_submission"], dtype=float).reshape(-1)
    if holdout_predictions.shape != (len(holdout_ids),):
        raise ValueError("invalid protocol holdout prediction shape")
    if test_predictions.shape != (len(test_ids),):
        raise ValueError("invalid protocol test prediction shape")
    holdout_predictions = _probabilities(holdout_predictions)
    test_predictions = _probabilities(test_predictions)
    score = _log_loss(train_labels().loc[holdout_ids].to_numpy(), holdout_predictions)
    return {
        "expected_holdout_ids": holdout_ids,
        "holdout_predictions": dict(zip(holdout_ids, holdout_predictions.tolist())),
        "expected_test_ids": test_ids,
        "test_submission": dict(zip(test_ids, test_predictions.tolist())),
        "score": score,
    }

