"""Editable candidate implementation for tabular-model-search.

This whole file is one candidate. Autoresearch compares candidates by changing
this file across runs, not by enumerating many candidates inside one run.
"""

from __future__ import annotations

import time
import warnings

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning

from prepare import (
    DatasetSplit,
    RANDOM_SEED,
    load_datasets,
    print_dataset_score,
    print_run_header,
    print_summary,
    test_accuracy,
)


CANDIDATE_NAME = "extra_trees_baseline"


def build_model(dataset: DatasetSplit):
    """Build the model for one dataset.

    Future experiments should edit this function, add helper functions, or
    adjust the training logic below. The fixed data split and scoring stay in
    `prepare.py`.
    """
    return ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def train_model(dataset: DatasetSplit):
    """Train this candidate on one training split and return the fitted estimator."""
    estimator = build_model(dataset)
    estimator.fit(dataset.x_train, dataset.y_train)
    return estimator


def run_candidate() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    started = time.time()
    datasets = load_datasets()
    scores: dict[str, float] = {}

    print_run_header(CANDIDATE_NAME, datasets)
    for dataset in datasets:
        estimator = train_model(dataset)
        score = test_accuracy(estimator, dataset)
        scores[dataset.name] = score
        print_dataset_score(dataset.name, score)

    print_summary(CANDIDATE_NAME, scores, time.time() - started)


def main() -> int:
    run_candidate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
