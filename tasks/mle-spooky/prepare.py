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

from tools.mle_resource_probe import run_resource_probe, run_smoke_probe

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


def _protocol_array(values, name: str) -> np.ndarray:
    """One member of the protocol archive, with a legible dtype remedy.

    ``np.load`` opens an ``.npz`` lazily, so an object-dtype array that
    ``allow_pickle=False`` refuses fails at member access rather than at open
    time.  Saving string ids straight from pandas is the usual cause.
    """
    try:
        return np.asarray(values[name])
    except ValueError as exc:
        raise ValueError(
            f"protocol {name} could not be read ({exc}); save ids as a string "
            "array, e.g. np.array(list(ids)), not an object array"
        ) from exc


def evaluate_protocol(artifact: Path, params: dict, *, stage: str = "protocol",
                      fidelity: str = "full") -> dict:
    """Score a candidate-produced protocol artifact in the evaluator process.

    The candidate only writes ``torch_outputs.npz``; this function owns the
    public split labels and independently validates both prediction streams.
    It never imports candidate code or opens files under the candidate source
    tree.
    """
    artifact = Path(artifact)
    output = artifact / "torch_outputs.npz" if artifact.is_dir() else artifact
    if not output.is_file():
        raise ValueError(f"protocol output is missing: {output}")
    values = np.load(output, allow_pickle=False)
    dataset, holdout_texts, holdout_labels = _split()
    frame = _training_data()
    _, valid_frame = train_test_split(
        frame, test_size=0.2, stratify=frame.author, random_state=42,
    )
    test = pd.read_csv(public_dir() / "test.csv")
    expected_holdout = _protocol_array(values, "holdout_ids")
    canonical_holdout = valid_frame.id.to_numpy()
    if expected_holdout.tolist() != canonical_holdout.tolist():
        raise ValueError("protocol holdout IDs do not match task split")
    if len(expected_holdout) != len(holdout_labels) or len(set(expected_holdout)) != len(expected_holdout):
        raise ValueError("protocol holdout IDs have the wrong cardinality")
    holdout_ids = list(expected_holdout.tolist())
    test_ids = list(_protocol_array(values, "test_ids").tolist())
    if test_ids != list(test.id.to_numpy()):
        raise ValueError("protocol test IDs do not match public test.csv")
    holdout_predictions = _protocol_array(values, "holdout_predictions").astype(float)
    test_predictions = _protocol_array(values, "test_submission").astype(float)
    if holdout_predictions.shape != (len(holdout_ids), len(CLASSES)):
        raise ValueError("invalid protocol holdout prediction shape")
    if test_predictions.shape != (len(test_ids), len(CLASSES)):
        raise ValueError("invalid protocol test prediction shape")
    if not np.isfinite(holdout_predictions).all() or not np.isfinite(test_predictions).all():
        raise ValueError("protocol predictions contain non-finite values")
    if not np.allclose(holdout_predictions.sum(axis=1), 1, atol=1e-6) or not np.allclose(test_predictions.sum(axis=1), 1, atol=1e-6):
        raise ValueError("protocol probabilities must sum to one")
    # Summing to one does not make every value a probability: a row like
    # (-0.2, 0.6, 0.6) sums to one and is rejected by official grading.
    for stream, probabilities in (("holdout", holdout_predictions),
                                  ("test", test_predictions)):
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError(
                f"protocol {stream} probabilities must lie in [0, 1]"
            )
    # Canonicalize harmless float32 summation drift before scoring and
    # persisting the evaluator output manifest.
    holdout_predictions = holdout_predictions / holdout_predictions.sum(axis=1, keepdims=True)
    test_predictions = test_predictions / test_predictions.sum(axis=1, keepdims=True)
    score = float(log_loss(holdout_labels, holdout_predictions, labels=list(CLASSES)))
    return {
        "expected_holdout_ids": holdout_ids,
        "holdout_predictions": dict(zip(holdout_ids, holdout_predictions.tolist())),
        "expected_test_ids": test_ids,
        "test_submission": dict(zip(test_ids, test_predictions.tolist())),
        "score": score,
    }


def evaluate_official(artifact: Path, params: dict, *, stage: str = "official",
                      fidelity: str = "full") -> dict:
    """Validate the finalized artifact through the task protocol."""
    return evaluate_protocol(artifact, params, stage=stage, fidelity=fidelity)
