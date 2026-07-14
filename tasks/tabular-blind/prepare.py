"""Fixed data loading and the single config->score evaluation for
`tabular-blind`.

BLIND task: three REAL tabular classification datasets are shipped as a
precomputed, anonymized array file (`data/datasets.npz`, named `d0`/`d1`/`d2`).
Their identities, feature meanings, and "which model/hyperparameters win" are
intentionally NOT disclosed here or in TASK.md — a candidate must discover what
works by *experiment* (try models, tune hyperparameters, read the held-out
score), not by looking anything up. The datasets were chosen because careful
hyperparameter tuning yields a large benefit that is only reached after a long
search, so better tuning/search is rewarded.

Do not modify during normal experiments. The held-out test split is never
exposed to candidates; only `x_train`/`y_train` are.

Scores follow the framework convention **lower is better**:
`neg_mean_balanced_accuracy = -mean(balanced_accuracy)` across the three
datasets. `evaluate_config` is the ONE evaluation surface; there is no separate
official run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

METRIC = "neg_mean_balanced_accuracy"  # negative mean balanced accuracy: lower is better


def _resolve_data_file() -> Path:
    """Locate datasets.npz robustly. The framework COPIES this prepare.py into each
    candidate dir (runs/.../candidates/<id>/), so a path relative to __file__ alone
    breaks. Look next to this file first, then walk up to the task's data dir."""
    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents):
        for c in (base / "data" / "datasets.npz",
                  base / "tasks" / "tabular-blind" / "data" / "datasets.npz"):
            if c.exists():
                return c
    raise FileNotFoundError(
        "datasets.npz not found (looked next to prepare.py and under tasks/tabular-blind/data)")


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray


_TEST_DATA: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_DATASETS_CACHE: list["DatasetSplit"] | None = None


def load_datasets() -> list[DatasetSplit]:
    """Fixed classification datasets, loaded from the precomputed file (cached).
    Only the training split is returned; the test split is held out internally."""
    global _DATASETS_CACHE
    if _DATASETS_CACHE is not None:
        return _DATASETS_CACHE
    blob = np.load(_resolve_data_file())
    n = int(blob["n_datasets"])
    _TEST_DATA.clear()
    out: list[DatasetSplit] = []
    for i in range(n):
        name = f"d{i}"
        _TEST_DATA[name] = (blob[f"d{i}_xte"], blob[f"d{i}_yte"])
        out.append(DatasetSplit(name=name, x_train=blob[f"d{i}_xtr"], y_train=blob[f"d{i}_ytr"]))
    _DATASETS_CACHE = out
    return _DATASETS_CACHE


def evaluate_config(make_model, params: dict) -> float:
    """The single `config -> score` evaluation (lower is better). Builds an
    estimator via `make_model(dataset, params)` per dataset, fits on the training
    split, scores on the fixed held-out test split, returns mean negative test
    accuracy. No separate official run — this value IS the candidate's score."""
    datasets = load_datasets()
    scores = []
    for dataset in datasets:
        estimator = make_model(dataset, params)
        estimator.fit(dataset.x_train, dataset.y_train)
        x_test, y_test = _TEST_DATA[dataset.name]
        scores.append(-float(balanced_accuracy_score(y_test, estimator.predict(x_test))))
    return float(np.mean(scores))
