"""Fixed public-data evaluation and final export; never loads private answers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import wave

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tools.mle_resource_probe import run_resource_probe, run_smoke_probe

N_SPECIES = 19
SPECIES = tuple(range(N_SPECIES))
SAMPLE_RATE = 16000
TRAIN_FOLD, TEST_FOLD = 0, 1
SPECTROGRAM_KINDS = (
    "spectrograms",
    "filtered_spectrograms",
    "supervised_segmentation",
    "segmentation_examples",
)


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    x_train: pd.DataFrame   # one recording per row (index = rec_id), waveform samples as columns
    y_train: np.ndarray     # (n, 19) species presence indicators


def public_dir() -> Path:
    return Path(os.environ["MLEBENCH_PUBLIC_DATA"]).resolve()


def _essential(name: str) -> Path:
    return public_dir() / "essential_data" / name


def _supplemental(name: str) -> Path:
    return public_dir() / "supplemental_data" / name


# --------------------------------------------------------------------------
# Recording table and labels
# --------------------------------------------------------------------------

def _read_labels() -> dict[int, tuple[int, ...] | None]:
    """``rec_id,[labels]`` rows: a species tuple, empty for silent clips, None for ``?``."""
    labels: dict[int, tuple[int, ...] | None] = {}
    for line in _essential("rec_labels_test_hidden.txt").read_text().splitlines()[1:]:
        if not line.strip():
            continue
        rec_id, *rest = line.split(",")
        rest = [value.strip() for value in rest if value.strip()]
        labels[int(rec_id)] = None if rest == ["?"] else tuple(sorted(int(v) for v in rest))
    return labels


@lru_cache(maxsize=1)
def recordings() -> pd.DataFrame:
    """Every public recording, indexed by ``rec_id``.

    Columns: ``filename`` (stem shared by the wav and every spectrogram),
    ``fold`` (0 = labelled training recording, 1 = unlabelled test recording)
    and ``labels`` (tuple of present species ids; ``None`` for test rows).
    """
    folds = pd.read_csv(_essential("CVfolds_2.txt"))
    names = pd.read_csv(_essential("rec_id2filename.txt"))
    frame = folds.merge(names, on="rec_id", how="outer")
    if frame.isna().any().any() or not frame.rec_id.is_unique or not set(frame.fold) <= {TRAIN_FOLD, TEST_FOLD}:
        raise ValueError("CVfolds_2.txt and rec_id2filename.txt disagree or are malformed")
    labels = _read_labels()
    if set(labels) != set(frame.rec_id):
        raise ValueError("rec_labels_test_hidden.txt does not cover the recording list")
    frame["labels"] = [labels[rec_id] for rec_id in frame.rec_id]
    is_test = frame.fold == TEST_FOLD
    if any(label is not None for label in frame.labels[is_test]) or any(
        label is None for label in frame.labels[~is_test]
    ):
        raise ValueError("label visibility does not match the train/test folds")
    for label in frame.labels[~is_test]:
        if not set(label) <= set(SPECIES):
            raise ValueError(f"species id out of range: {label}")
    return frame.set_index("rec_id").sort_index()


def _train_ids() -> np.ndarray:
    frame = recordings()
    return frame.index[frame.fold == TRAIN_FOLD].to_numpy()


def _label_matrix(rec_ids) -> np.ndarray:
    labels = recordings().labels
    matrix = np.zeros((len(rec_ids), N_SPECIES), dtype=np.int8)
    for row, rec_id in enumerate(rec_ids):
        matrix[row, list(labels.loc[int(rec_id)])] = 1
    return matrix


# --------------------------------------------------------------------------
# Per-recording loaders (train and test alike), addressed by rec_id
# --------------------------------------------------------------------------

def filename_of(rec_id: int) -> str:
    return str(recordings().loc[int(rec_id), "filename"])


def wav_path(rec_id: int) -> Path:
    return _essential("src_wavs") / f"{filename_of(rec_id)}.wav"


def load_waveform(rec_id: int) -> np.ndarray:
    """Mono 16 kHz samples of one recording as float32 in [-1, 1]."""
    with wave.open(str(wav_path(rec_id)), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError(f"unexpected wav format for rec_id {rec_id}")
        frames = stream.readframes(stream.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def spectrogram_path(rec_id: int, kind: str = "spectrograms") -> Path:
    """BMP of one recording under ``supplemental_data/<kind>``.

    ``segmentation_examples`` exists only for a few training recordings; check
    ``.is_file()`` before loading that kind.
    """
    if kind not in SPECTROGRAM_KINDS:
        raise ValueError(f"kind must be one of {SPECTROGRAM_KINDS}")
    return _supplemental(kind) / f"{filename_of(rec_id)}.bmp"


def load_spectrogram(rec_id: int, kind: str = "spectrograms") -> np.ndarray:
    """Pixel array of the published BMP (uint8, time on x, frequency on y).

    ``spectrograms`` and ``filtered_spectrograms`` are grayscale ``(256, 1246)``;
    ``supervised_segmentation`` and ``segmentation_examples`` are RGB
    ``(256, 1246, 3)``.
    """
    with Image.open(spectrogram_path(rec_id, kind)) as image:
        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")
        return np.asarray(image)


def _read_numeric_rows(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines()[1:]:
        if line.strip():
            rows.append([float(value) for value in line.split(",") if value.strip()])
    return rows


def _segment_table(name: str, value_columns: list[str] | None) -> pd.DataFrame:
    frame = pd.DataFrame(_read_numeric_rows(_supplemental(name)))
    if frame.empty or frame.shape[1] < 3:
        raise ValueError(f"malformed {name}")
    values = value_columns or [f"f{i}" for i in range(frame.shape[1] - 2)]
    if len(values) != frame.shape[1] - 2:
        raise ValueError(f"unexpected column count in {name}")
    frame.columns = ["rec_id", "segment_id", *values]
    return frame.astype({"rec_id": int, "segment_id": int})


@lru_cache(maxsize=1)
def segment_features() -> pd.DataFrame:
    """Baseline segmentation: one row per segment, ``rec_id, segment_id, f0..f37``.

    Recordings in which the baseline segmenter found nothing have no rows.
    """
    return _segment_table("segment_features.txt", None)


@lru_cache(maxsize=1)
def segment_rectangles() -> pd.DataFrame:
    """Bounding box of every baseline segment in spectrogram pixel coordinates.

    Columns ``rec_id, segment_id, x_min, x_max, y_min, y_max``.
    """
    return _segment_table("segment_rectangles.txt", ["x_min", "x_max", "y_min", "y_max"])


@lru_cache(maxsize=1)
def histogram_of_segments() -> pd.DataFrame:
    """Baseline 100-bin codebook histogram per recording, indexed by ``rec_id``.

    Every public recording has a row; one without baseline segments is all zeros.
    """
    frame = pd.DataFrame(_read_numeric_rows(_supplemental("histogram_of_segments.txt")))
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("malformed histogram_of_segments.txt")
    frame.columns = ["rec_id", *[f"bin_{i}" for i in range(frame.shape[1] - 1)]]
    return frame.astype({"rec_id": int}).set_index("rec_id")


@lru_cache(maxsize=1)
def species_list() -> pd.DataFrame:
    """The published ``species_list.txt`` table (class id, code, species name)."""
    return pd.read_csv(_essential("species_list.txt"))


# --------------------------------------------------------------------------
# Fixed split, scoring and probes
# --------------------------------------------------------------------------

def _waveforms(rec_ids) -> pd.DataFrame:
    rec_ids = np.asarray(rec_ids, dtype=int)
    values = np.stack([load_waveform(rec_id) for rec_id in rec_ids])
    return pd.DataFrame(values, index=pd.Index(rec_ids, name="rec_id"))


def _holdout_ids() -> tuple[np.ndarray, np.ndarray]:
    train_ids = _train_ids()
    strata = np.minimum(_label_matrix(train_ids).sum(axis=1), 2)
    return train_test_split(train_ids, test_size=0.2, stratify=strata, random_state=42)


@lru_cache(maxsize=1)
def _split():
    fit_ids, holdout_ids = _holdout_ids()
    dataset = DatasetSplit("mlsp-birds", _waveforms(fit_ids), _label_matrix(fit_ids))
    return dataset, _waveforms(holdout_ids), _label_matrix(holdout_ids)


def load_datasets():
    return [_split()[0]]


def _scores(model, x: pd.DataFrame) -> np.ndarray:
    """(n, 19) species-presence probabilities from the estimator's ``predict_proba``.

    Accepts a single ``(n, 19)`` array or the per-output list that sklearn's
    multi-output estimators return (one ``(n, k)`` block per species, with the
    positive column located through ``classes_``).
    """
    raw = model.predict_proba(x)
    if isinstance(raw, (list, tuple)):
        if len(raw) != N_SPECIES:
            raise ValueError("predict_proba must cover all 19 species")
        classes = getattr(model, "classes_", None)
        per_output = isinstance(classes, (list, tuple)) and len(classes) == N_SPECIES
        columns = []
        for k, block in enumerate(raw):
            block = np.asarray(block, dtype=float)
            if block.ndim != 2 or block.shape[0] != len(x):
                raise ValueError("invalid per-species predict_proba shape")
            labels = [int(v) for v in classes[k]] if per_output else list(range(block.shape[1]))
            columns.append(block[:, labels.index(1)] if 1 in labels else np.zeros(len(x)))
        values = np.column_stack(columns)
    else:
        values = np.asarray(raw, dtype=float)
    if values.shape != (len(x), N_SPECIES):
        raise ValueError("predict_proba must return one probability per species and recording")
    return _probabilities(values)


def _probabilities(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("predictions contain non-finite values")
    if values.min() < -1e-6 or values.max() > 1 + 1e-6:
        raise ValueError("predictions must be probabilities in [0, 1]")
    return np.clip(values, 0.0, 1.0)


def _one_minus_auc(labels: np.ndarray, values: np.ndarray) -> float:
    """Official metric: one ROC AUC over every (recording, species) pair."""
    return float(1.0 - roc_auc_score(labels.ravel(), values.ravel()))


def evaluate_config(make_model, params: dict) -> float:
    dataset, x_holdout, y_holdout = _split()
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    return _one_minus_auc(y_holdout, _scores(model, x_holdout))


def _test_ids() -> np.ndarray:
    """Test recordings in ``sample_submission.csv`` order."""
    sample = pd.read_csv(public_dir() / "sample_submission.csv")
    return pd.unique(sample.Id.to_numpy() // 100)


def preflight_environment() -> dict:
    frame = recordings()
    sample = pd.read_csv(public_dir() / "sample_submission.csv")
    if list(sample.columns) != ["Id", "Probability"]:
        raise ValueError("unexpected sample submission columns")
    test_ids = frame.index[frame.fold == TEST_FOLD]
    expected = {int(rec_id) * 100 + species for rec_id in test_ids for species in SPECIES}
    if not sample.Id.is_unique or set(sample.Id) != expected:
        raise ValueError("sample submission Ids do not match the test recordings")
    lengths = {len(load_waveform(rec_id)) for rec_id in frame.index}
    if len(lengths) != 1:
        raise ValueError(f"recordings differ in length: {sorted(lengths)}")
    for rec_id in frame.index:
        for kind in SPECTROGRAM_KINDS[:3]:
            with Image.open(spectrogram_path(rec_id, kind)) as image:
                image.verify()
    examples = sum(spectrogram_path(rec_id, "segmentation_examples").is_file() for rec_id in frame.index)
    for table in (segment_features(), segment_rectangles()):
        if not set(table.rec_id) <= set(frame.index):
            raise ValueError("segment tables reference unknown recordings")
    if not set(histogram_of_segments().index) <= set(frame.index):
        raise ValueError("histogram_of_segments.txt references unknown recordings")
    species_list()
    _split()
    return {
        "train_rows": int((frame.fold == TRAIN_FOLD).sum()),
        "test_rows": len(test_ids),
        "submission_rows": len(sample),
        "samples_per_recording": lengths.pop(),
        "segmentation_examples": examples,
    }


def preflight_config(make_model, params: dict) -> dict:
    """No-score seconds-scale smoke: construct and fit a subsample."""
    return run_smoke_probe(make_model, params, _split()[0])


def resource_probe_config(make_model, params: dict) -> dict:
    """No-score resource envelope for tuner search-space clamping."""
    return run_resource_probe(make_model, params, _split()[0])


def _submission_frame(test_ids: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    sample = pd.read_csv(public_dir() / "sample_submission.csv")
    lookup = {
        int(rec_id) * 100 + species: float(values[row, species])
        for row, rec_id in enumerate(test_ids)
        for species in SPECIES
    }
    probability = sample.Id.map(lookup)
    if probability.isna().any():
        raise ValueError("predictions do not cover every sample submission Id")
    return pd.DataFrame({"Id": sample.Id.to_numpy(), "Probability": probability.to_numpy()})


def export_submission(make_model, params: dict, output: Path) -> None:
    """Refit on all public training recordings using the already selected configuration."""
    preflight_environment()
    train_ids = _train_ids()
    dataset = DatasetSplit("mlsp-birds", _waveforms(train_ids), _label_matrix(train_ids))
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    test_ids = _test_ids()
    submission = _submission_frame(test_ids, _scores(model, _waveforms(test_ids)))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(output)


def evaluate_protocol(artifact: Path, params: dict, *, stage: str = "protocol",
                      fidelity: str = "full") -> dict:
    """Score a candidate-produced ``protocol_outputs.npz`` in the evaluator process.

    The archive holds ``holdout_ids`` and ``test_ids`` (integer rec_ids in the
    task's holdout order and in sample-submission order) plus
    ``holdout_predictions`` and ``test_submission``, each ``(n, 19)`` species
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
    if np.asarray(values["holdout_ids"]).astype(int).tolist() != holdout_ids.tolist():
        raise ValueError("protocol holdout IDs do not match the task split")
    if np.asarray(values["test_ids"]).astype(int).tolist() != test_ids.tolist():
        raise ValueError("protocol test IDs do not match sample_submission.csv order")
    holdout_predictions = np.asarray(values["holdout_predictions"], dtype=float)
    test_predictions = np.asarray(values["test_submission"], dtype=float)
    if holdout_predictions.shape != (len(holdout_ids), N_SPECIES):
        raise ValueError("invalid protocol holdout prediction shape")
    if test_predictions.shape != (len(test_ids), N_SPECIES):
        raise ValueError("invalid protocol test prediction shape")
    holdout_predictions = _probabilities(holdout_predictions)
    test_predictions = _probabilities(test_predictions)
    score = _one_minus_auc(_label_matrix(holdout_ids), holdout_predictions)
    holdout_keys = [int(v) for v in holdout_ids]
    test_keys = [int(v) for v in test_ids]
    return {
        "expected_holdout_ids": holdout_keys,
        "holdout_predictions": dict(zip(holdout_keys, holdout_predictions.tolist())),
        "expected_test_ids": test_keys,
        "test_submission": dict(zip(test_keys, test_predictions.tolist())),
        "score": score,
    }
