"""Fixed public-data evaluation and final export; never loads private answers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit

from tools.mle_resource_probe import run_resource_probe, run_smoke_probe

ID = "qa_id"
INPUT_COLUMNS = (
    "question_title", "question_body", "question_user_name", "question_user_page",
    "answer", "answer_user_name", "answer_user_page", "url", "category", "host",
)
TARGETS = (
    "question_asker_intent_understanding", "question_body_critical",
    "question_conversational", "question_expect_short_answer",
    "question_fact_seeking", "question_has_commonly_accepted_answer",
    "question_interestingness_others", "question_interestingness_self",
    "question_multi_intent", "question_not_really_a_question",
    "question_opinion_seeking", "question_type_choice", "question_type_compare",
    "question_type_consequence", "question_type_definition", "question_type_entity",
    "question_type_instructions", "question_type_procedure",
    "question_type_reason_explanation", "question_type_spelling",
    "question_well_written", "answer_helpful", "answer_level_of_information",
    "answer_plausible", "answer_relevance", "answer_satisfaction",
    "answer_type_instructions", "answer_type_procedure",
    "answer_type_reason_explanation", "answer_well_written",
)


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    x_train: pd.DataFrame   # one question-answer pair per row (index = qa_id): the ten published input columns
    y_train: np.ndarray     # (n, 30) float targets in TARGETS order, each in [0, 1]


def public_dir() -> Path:
    return Path(os.environ["MLEBENCH_PUBLIC_DATA"]).resolve()


# --------------------------------------------------------------------------
# Published files
# --------------------------------------------------------------------------

def _indexed(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame[ID] = frame[ID].astype(np.int64)
    if not frame[ID].is_unique:
        raise ValueError("duplicate qa_id values")
    return frame.set_index(ID)


@lru_cache(maxsize=1)
def train_frame() -> pd.DataFrame:
    """All public training rows in file order, indexed by ``qa_id``.

    Columns are the ten input columns (``question_title``, ``question_body``,
    ``question_user_name``, ``question_user_page``, ``answer``,
    ``answer_user_name``, ``answer_user_page``, ``url``, ``category``,
    ``host``) followed by the 30 target columns.
    """
    frame = _indexed(pd.read_csv(public_dir() / "train.csv"))
    if list(frame.columns) != [*INPUT_COLUMNS, *TARGETS]:
        raise ValueError("public train.csv columns differ from the published schema")
    return frame


@lru_cache(maxsize=1)
def test_frame() -> pd.DataFrame:
    """All public test rows in file order, indexed by ``qa_id``; the ten input columns."""
    frame = _indexed(pd.read_csv(public_dir() / "test.csv"))
    if list(frame.columns) != list(INPUT_COLUMNS):
        raise ValueError("public test.csv columns differ from the published schema")
    return frame


@lru_cache(maxsize=1)
def sample_submission() -> pd.DataFrame:
    return pd.read_csv(public_dir() / "sample_submission.csv")


def inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """The ten input columns of any frame this module produces."""
    return frame[list(INPUT_COLUMNS)]


# --------------------------------------------------------------------------
# Fixed split, scoring and probes
# --------------------------------------------------------------------------

def _holdout_ids() -> tuple[list[int], list[int]]:
    """80/20 split of the public training rows, grouped by question (``url``), seed 42."""
    frame = train_frame()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    fit, holdout = next(splitter.split(frame, groups=frame["url"]))
    ids = frame.index.to_numpy()
    return [int(v) for v in ids[np.sort(fit)]], [int(v) for v in ids[np.sort(holdout)]]


@lru_cache(maxsize=1)
def _split():
    fit_ids, holdout_ids = _holdout_ids()
    frame = train_frame()
    dataset = DatasetSplit(
        "google-quest",
        inputs(frame.loc[fit_ids]),
        frame.loc[fit_ids, list(TARGETS)].to_numpy(dtype=float),
    )
    return dataset, inputs(frame.loc[holdout_ids]), frame.loc[holdout_ids, list(TARGETS)].to_numpy(dtype=float)


def load_datasets():
    return [_split()[0]]


def _predictions(values, n_rows: int, name: str = "predictions") -> np.ndarray:
    """``(n, 30)`` finite float array in TARGETS order (a DataFrame is reordered by column name)."""
    if isinstance(values, pd.DataFrame):
        if set(TARGETS).issubset(values.columns):
            values = values[list(TARGETS)]
        values = values.to_numpy()
    values = np.asarray(values, dtype=float)
    if values.shape != (n_rows, len(TARGETS)):
        raise ValueError(f"{name} must have shape ({n_rows}, {len(TARGETS)}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contain non-finite values")
    return values


def mean_spearman(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Official metric: mean over the 30 target columns of Spearman's rank correlation.

    A constant prediction column has no defined correlation; it counts as 0
    here (official grading returns an undefined overall score in that case).
    """
    correlations = []
    for column in range(len(TARGETS)):
        value = spearmanr(predictions[:, column], labels[:, column]).correlation
        correlations.append(0.0 if np.isnan(value) else float(value))
    return float(np.mean(correlations))


def _score(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Lower-is-better surface: ``1 - mean column-wise Spearman``."""
    return 1.0 - mean_spearman(labels, predictions)


def evaluate_config(make_model, params: dict) -> float:
    dataset, x_holdout, y_holdout = _split()
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    return _score(y_holdout, _predictions(model.predict(x_holdout), len(x_holdout)))


def _test_ids() -> list[int]:
    """Test ``qa_id`` values in ``sample_submission.csv`` order."""
    return [int(v) for v in sample_submission()[ID]]


def preflight_environment() -> dict:
    train, test, sample = train_frame(), test_frame(), sample_submission()
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("public train.csv or test.csv contains missing values")
    targets = train[list(TARGETS)].to_numpy(dtype=float)
    if targets.min() < 0 or targets.max() > 1:
        raise ValueError("training targets must lie in [0, 1]")
    if list(sample.columns) != [ID, *TARGETS]:
        raise ValueError("unexpected sample submission columns")
    if not sample[ID].is_unique or set(_test_ids()) != set(test.index):
        raise ValueError("sample submission ids do not match test.csv")
    if set(train.index) & set(test.index):
        raise ValueError("train and test qa_id values overlap")
    _, holdout_ids = _holdout_ids()
    holdout_targets = train.loc[holdout_ids, list(TARGETS)]
    constant = [column for column in TARGETS if holdout_targets[column].nunique() < 2]
    if constant:
        raise ValueError(f"holdout targets are constant: {constant}")
    _split()
    return {
        "train_rows": len(train),
        "train_questions": int(train["url"].nunique()),
        "holdout_rows": len(holdout_ids),
        "test_rows": len(test),
        "submission_rows": len(sample),
        "targets": len(TARGETS),
    }


def preflight_config(make_model, params: dict) -> dict:
    """No-score seconds-scale smoke: construct and fit a subsample."""
    return run_smoke_probe(make_model, params, _split()[0])


def resource_probe_config(make_model, params: dict) -> dict:
    """No-score resource envelope for tuner search-space clamping."""
    return run_resource_probe(make_model, params, _split()[0])


def export_submission(make_model, params: dict, output: Path) -> None:
    """Refit on all public training rows using the already selected configuration."""
    preflight_environment()
    frame = train_frame()
    dataset = DatasetSplit("google-quest", inputs(frame), frame[list(TARGETS)].to_numpy(dtype=float))
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    test_ids = _test_ids()
    values = _predictions(model.predict(inputs(test_frame().loc[test_ids])), len(test_ids))
    submission = pd.DataFrame(values, columns=list(TARGETS))
    submission.insert(0, ID, test_ids)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(output)


def evaluate_protocol(artifact: Path, params: dict, *, stage: str = "protocol",
                      fidelity: str = "full") -> dict:
    """Score a candidate-produced ``protocol_outputs.npz`` in the evaluator process.

    The archive holds ``holdout_ids`` and ``test_ids`` (integer ``qa_id``
    values in the task's holdout order and in sample-submission order) plus
    ``holdout_predictions`` and ``test_submission``, each ``(n, 30)`` in
    TARGETS order.  This function owns the holdout labels and validates both
    prediction streams; it never imports candidate code.
    """
    artifact = Path(artifact)
    output = artifact / "protocol_outputs.npz" if artifact.is_dir() else artifact
    if not output.is_file():
        raise ValueError(f"protocol output is missing: {output}")
    values = np.load(output, allow_pickle=False)
    _, holdout_ids = _holdout_ids()
    test_ids = _test_ids()
    if [int(v) for v in values["holdout_ids"]] != holdout_ids:
        raise ValueError("protocol holdout IDs do not match the task split")
    if [int(v) for v in values["test_ids"]] != test_ids:
        raise ValueError("protocol test IDs do not match sample_submission.csv order")
    holdout_predictions = _predictions(values["holdout_predictions"], len(holdout_ids), "protocol holdout predictions")
    test_predictions = _predictions(values["test_submission"], len(test_ids), "protocol test predictions")
    labels = train_frame().loc[holdout_ids, list(TARGETS)].to_numpy(dtype=float)
    return {
        "expected_holdout_ids": holdout_ids,
        "holdout_predictions": dict(zip(holdout_ids, holdout_predictions.tolist())),
        "expected_test_ids": test_ids,
        "test_submission": dict(zip(test_ids, test_predictions.tolist())),
        "score": _score(labels, holdout_predictions),
    }
