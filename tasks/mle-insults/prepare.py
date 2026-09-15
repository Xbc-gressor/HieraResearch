"""Fixed public-data evaluation and final export; never loads private answers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tools.mle_resource_probe import run_resource_probe, run_smoke_probe

TARGET = "Insult"
FEATURES = ("Date", "Comment")


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
    if list(frame.columns) != [TARGET, *FEATURES]:
        raise ValueError("public train.csv columns are invalid")
    if frame[TARGET].isna().any() or not set(frame[TARGET].unique()) <= {0, 1}:
        raise ValueError("public Insult labels must be binary and non-null")
    if frame["Comment"].isna().any():
        raise ValueError("public comments must be non-null")
    return frame


@lru_cache(maxsize=1)
def _split():
    frame = _training_data()
    train, valid = train_test_split(
        frame, test_size=0.2, stratify=frame[TARGET], random_state=42,
    )
    dataset = DatasetSplit("insults", train[list(FEATURES)].reset_index(drop=True),
                           train[TARGET].to_numpy())
    return dataset, valid[list(FEATURES)].reset_index(drop=True), valid[TARGET].to_numpy()


def load_datasets():
    return [_split()[0]]


def _positive_scores(model, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if set(classes) != {0, 1} or len(classes) != 2:
        raise ValueError("estimator must predict both Insult classes")
    values = np.asarray(model.predict_proba(features), dtype=float)
    if values.shape != (len(features), 2):
        raise ValueError("invalid predict_proba shape")
    scores = values[:, classes.index(1)]
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("predict_proba must return finite probabilities")
    return scores


def evaluate_config(make_model, params: dict) -> float:
    dataset, features, labels = _split()
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    return float(1.0 - roc_auc_score(labels, _positive_scores(model, features)))


def preflight_environment() -> dict:
    frame = _training_data()
    test = pd.read_csv(public_dir() / "test.csv")
    sample = pd.read_csv(public_dir() / "sample_submission_null.csv")
    if list(test.columns) != list(FEATURES):
        raise ValueError("expected public test columns Date, Comment")
    if list(sample.columns) != [TARGET, *FEATURES]:
        raise ValueError("unexpected sample submission columns")
    if test["Comment"].isna().any():
        raise ValueError("public test comments must be non-null")
    if set(frame["Comment"]) & set(test["Comment"]):
        raise ValueError("public training and test comments overlap")
    if len(sample) != len(test) or not sample["Comment"].equals(test["Comment"]):
        raise ValueError("sample submission and test rows differ")
    _split()
    return {"train_rows": len(frame), "test_rows": len(test), "gpu_required": False}


def preflight_config(make_model, params: dict) -> dict:
    """No-score seconds-scale smoke: construct and fit a subsample."""
    return run_smoke_probe(make_model, params, _split()[0])


def resource_probe_config(make_model, params: dict) -> dict:
    """No-score resource envelope for tuner search-space clamping."""
    return run_resource_probe(make_model, params, _split()[0])


def export_submission(make_model, params: dict, output: Path) -> None:
    preflight_environment()
    frame = _training_data()
    test = pd.read_csv(public_dir() / "test.csv")
    dataset = DatasetSplit("insults", frame[list(FEATURES)].reset_index(drop=True),
                           frame[TARGET].to_numpy())
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    submission = test.copy()
    submission.insert(0, TARGET, _positive_scores(model, test[list(FEATURES)]))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(output)
