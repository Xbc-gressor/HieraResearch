"""Fixed data preparation and test scoring for tabular-model-search.

Do not modify this file during normal experiments. Edit `train.py` to change
the candidate implementation and training logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
TEST_SIZE = 0.30
METRIC = "mean_test_accuracy"


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
_USED_TEST_SCORES: set[str] = set()
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
    _USED_TEST_SCORES.clear()
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


def test_accuracy(estimator: object, dataset: DatasetSplit) -> float:
    """Score a fitted estimator on the fixed hidden test split.

    Each dataset may be scored once per process. Used by the main candidate
    `run_candidate()` to produce the official run score.
    """
    if dataset.name in _USED_TEST_SCORES:
        raise RuntimeError(f"test score already consumed for dataset: {dataset.name}")
    try:
        x_test, y_test = _TEST_DATA[dataset.name]
    except KeyError as exc:
        raise KeyError(f"unknown dataset split: {dataset.name}") from exc
    predictions = estimator.predict(x_test)
    _USED_TEST_SCORES.add(dataset.name)
    return float(accuracy_score(y_test, predictions))


def evaluate_config_for_tuning(make_model, params: dict) -> float:
    """Score one hyperparameter configuration across all task datasets.

    Called by tuner scripts under tools/tuners/. Builds an estimator via
    make_model for each dataset, fits on the training split, and scores on
    the test split via test_score_for_tuning (no one-shot lock). Returns
    the mean test accuracy across datasets — the task's optimization
    target during tuning.

    Task-specific evaluation logic lives here so tuner scripts stay
    generic. A different task (e.g., LLM pretraining) would define its own
    evaluate_config_for_tuning implementing whatever training/scoring loop
    fits its surface.
    """
    datasets = load_datasets()
    scores = []
    for dataset in datasets:
        estimator = make_model(dataset, params)
        estimator.fit(dataset.x_train, dataset.y_train)
        scores.append(test_score_for_tuning(estimator, dataset))
    return float(np.mean(scores))


def test_score_for_tuning(estimator: object, dataset: DatasetSplit) -> float:
    """Score on the test split WITHOUT consuming the one-shot lock.

    Intended for tuner scripts under tools/tuners/ that need to evaluate many
    configurations on the test set during hyperparameter search. This
    intentionally treats the test split as the optimization target rather
    than an unbiased benchmark; the final candidate score reported via
    `test_accuracy` (called once per dataset in run_candidate) is therefore
    a biased estimate after tuning.
    """
    try:
        x_test, y_test = _TEST_DATA[dataset.name]
    except KeyError as exc:
        raise KeyError(f"unknown dataset split: {dataset.name}") from exc
    return float(accuracy_score(y_test, estimator.predict(x_test)))


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
