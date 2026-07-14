"""Fixed data preparation and the single config→score evaluation for
tabular-model-search.

Do not modify this file during normal experiments. A candidate's tunable
construction lives in its own `train.py` (`make_model` + the tuner contract);
this file owns the data splits and the one scoring function.

Scores follow the framework convention that **lower is better**: this task
reports `neg_mean_test_accuracy = -mean(accuracy)`, so minimizing it is the same
as maximizing accuracy. `evaluate_config` is the **one** evaluation surface —
warm-start eval and Phase C tuning both call it; there is no separate official
run, so the score it returns IS the candidate's score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
TEST_SIZE = 0.30
METRIC = "neg_mean_test_accuracy"  # negative mean accuracy: lower is better


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    dataset_scores: dict[str, float]
    elapsed_seconds: float


_TEST_DATA: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_DATASETS_CACHE: list["DatasetSplit"] | None = None


def _make_dataset(
    *,
    name: str,
    n_samples: int,
    n_features: int,
    n_informative: int,
    n_redundant: int,
    n_classes: int,
    n_clusters_per_class: int,
    class_sep: float,
    flip_y: float,
    weights: list[float] | None,
    random_state: int,
) -> DatasetSplit:
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        weights=weights,
        class_sep=class_sep,
        flip_y=flip_y,
        shuffle=True,
        random_state=random_state,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    _TEST_DATA[name] = (x_test, y_test)
    return DatasetSplit(name=name, x_train=x_train, y_train=y_train)


def load_datasets() -> list[DatasetSplit]:
    """Return fixed noisy tabular classification datasets.

    Cached at module scope. Subsequent calls within the same process return
    the same list. `make_classification` is deterministic with fixed
    random_state, so caching is purely an optimization for repeated tuner
    evaluations.
    """
    global _DATASETS_CACHE
    if _DATASETS_CACHE is not None:
        return _DATASETS_CACHE
    _TEST_DATA.clear()
    _DATASETS_CACHE = [
        _make_dataset(
            name="noisy_binary",
            n_samples=2800,
            n_features=80,
            n_informative=12,
            n_redundant=18,
            n_classes=2,
            n_clusters_per_class=3,
            class_sep=0.65,
            flip_y=0.10,
            weights=[0.62, 0.38],
            random_state=11,
        ),
        _make_dataset(
            name="sparse_multiclass",
            n_samples=3600,
            n_features=140,
            n_informative=18,
            n_redundant=18,
            n_classes=4,
            n_clusters_per_class=2,
            class_sep=0.60,
            flip_y=0.08,
            weights=None,
            random_state=23,
        ),
        _make_dataset(
            name="high_dimensional",
            n_samples=1800,
            n_features=320,
            n_informative=22,
            n_redundant=45,
            n_classes=3,
            n_clusters_per_class=2,
            class_sep=0.55,
            flip_y=0.12,
            weights=[0.45, 0.35, 0.20],
            random_state=37,
        ),
    ]
    return _DATASETS_CACHE


def evaluate_config(make_model, params: dict) -> float:
    """The single `config → score` evaluation function (lower is better).

    The **one** evaluation surface (task.toml `[evaluation].score_fn`), called by
    the tuner scripts under tools/tuners/ for both warm-start eval and Phase C
    search. Builds an estimator via `make_model(dataset, params)` for each
    dataset, fits on the training split, scores on the fixed held-out test split,
    and returns the mean **negative** test accuracy across datasets. There is no
    separate official run, so this value IS the candidate's score; the test split
    is the optimization target, so the score is an optimistic estimate by
    construction.

    A different task would define its own `score_fn` (named in its task.toml)
    implementing whatever training/scoring loop fits its surface.
    """
    datasets = load_datasets()
    scores = []
    for dataset in datasets:
        estimator = make_model(dataset, params)
        estimator.fit(dataset.x_train, dataset.y_train)
        x_test, y_test = _TEST_DATA[dataset.name]
        scores.append(-float(accuracy_score(y_test, estimator.predict(x_test))))
    return float(np.mean(scores))


def format_scores(scores: dict[str, float]) -> str:
    return " ".join(f"{name}={score:.6f}" for name, score in scores.items())


def print_run_header(candidate_name: str, datasets: list[DatasetSplit]) -> None:
    """Print the fixed run header."""
    print(f"Datasets: {', '.join(dataset.name for dataset in datasets)}")
    print(f"Candidate: {candidate_name}")
    print(f"Metric: {METRIC}")
    print()


def print_dataset_score(dataset_name: str, score: float) -> None:
    """Print one dataset score in the fixed run format."""
    print(f"{dataset_name:16s} test={score:.6f}")


def print_summary(
    candidate_name: str,
    scores: dict[str, float],
    elapsed_seconds: float,
) -> EvaluationResult:
    """Print the fixed final summary and return its structured result."""
    mean_score = float(np.mean(list(scores.values())))

    print("---")
    print(f"metric:           {METRIC}")
    print(f"score:            {mean_score:.6f}")
    print(f"best_model:       {candidate_name}")
    print(f"dataset_scores:   {format_scores(scores)}")
    print(f"fit_seconds:      {elapsed_seconds:.1f}")
    print(f"num_datasets:     {len(scores)}")
    print("num_candidates:   1")

    return EvaluationResult(
        score=mean_score,
        dataset_scores=scores,
        elapsed_seconds=elapsed_seconds,
    )
