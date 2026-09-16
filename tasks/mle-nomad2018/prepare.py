"""Fixed public-data evaluation and final export; never loads private answers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from tools.mle_resource_probe import run_resource_probe, run_smoke_probe

TARGETS = ("formation_energy_ev_natom", "bandgap_energy_ev")
GEOMETRY_KEY = "geometry_key"
SPLITS = ("train", "test")


@dataclass(frozen=True)
class Geometry:
    """One material's unit cell, as published in its ``geometry.xyz`` file."""

    key: str
    lattice: np.ndarray          # (3, 3) lattice vectors, angstroms
    positions: np.ndarray        # (n_atoms, 3) cartesian coordinates, angstroms
    elements: tuple[str, ...]    # per-atom element symbol


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    x_train: pd.DataFrame
    y_train: np.ndarray


def public_dir() -> Path:
    return Path(os.environ["MLEBENCH_PUBLIC_DATA"]).resolve()


def _geometry_path(key: str) -> Path:
    split, _, identifier = str(key).partition("/")
    if split not in SPLITS or not identifier.isdigit():
        raise ValueError(f"geometry key must be '<{'|'.join(SPLITS)}>/<id>': {key!r}")
    return public_dir() / split / identifier / "geometry.xyz"


@lru_cache(maxsize=None)
def load_geometry(key: str) -> Geometry:
    """Atomic structure for one row, addressed by its ``x_train`` index label.

    The published tabular columns summarize the cell; the per-atom positions
    live in these files and are the material's full description.
    """
    path = _geometry_path(key)
    lattice: list[list[float]] = []
    positions: list[list[float]] = []
    elements: list[str] = []
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "lattice_vector":
                lattice.append([float(value) for value in fields[1:4]])
            elif fields[0] == "atom":
                positions.append([float(value) for value in fields[1:4]])
                elements.append(fields[4])
    if len(lattice) != 3 or not positions:
        raise ValueError(f"malformed geometry file: {path}")
    return Geometry(
        str(key),
        np.asarray(lattice, dtype=float),
        np.asarray(positions, dtype=float),
        tuple(elements),
    )


def load_geometries(keys) -> list[Geometry]:
    """Geometries for an index or key sequence, in the order given."""
    return [load_geometry(str(key)) for key in keys]


@lru_cache(maxsize=1)
def _training_data() -> pd.DataFrame:
    frame = pd.read_csv(public_dir() / "train.csv")
    required = {"id", *TARGETS}
    if not required.issubset(frame.columns):
        raise ValueError("public train.csv is missing NOMAD target columns")
    if frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("public training IDs are invalid")
    if frame[list(TARGETS)].isna().any().any():
        raise ValueError("public training targets contain missing values")
    return frame


def _features(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Numeric tabular columns, indexed by each row's geometry key.

    The frame stays fully numeric so ordinary sklearn pipelines accept it;
    the index carries the identity a candidate needs to reach that row's
    ``geometry.xyz`` through ``load_geometry``.
    """
    if "id" not in frame.columns:
        raise ValueError("NOMAD rows must carry their submission id")
    values = frame.drop(columns=list(TARGETS), errors="ignore").copy()
    keys = values.pop("id")
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("NOMAD tabular features must be numeric and non-null")
    numeric.index = pd.Index(
        [f"{split}/{int(key)}" for key in keys], name=GEOMETRY_KEY
    )
    return numeric


@lru_cache(maxsize=1)
def _split():
    frame = _training_data()
    train, valid = train_test_split(frame, test_size=0.2, random_state=42)
    dataset = DatasetSplit(
        "nomad2018", _features(train, "train"), train[list(TARGETS)].to_numpy()
    )
    return dataset, _features(valid, "train"), valid[list(TARGETS)].to_numpy()


def load_datasets():
    return [_split()[0]]


def _prediction(model, features: pd.DataFrame, rows: int) -> np.ndarray:
    values = np.asarray(model.predict(features), dtype=float)
    if values.shape != (rows, len(TARGETS)):
        raise ValueError("model must predict two NOMAD targets per row")
    if not np.isfinite(values).all():
        raise ValueError("predictions must be finite")
    # RMSLE is undefined below -1 and both targets are physically non-negative.
    return np.maximum(values, 0.0)


def evaluate_config(make_model, params: dict) -> float:
    dataset, features, labels = _split()
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    values = _prediction(model, features, len(features))
    scores = [mean_squared_log_error(labels[:, i], values[:, i]) ** 0.5
              for i in range(len(TARGETS))]
    return float(np.mean(scores))


def preflight_environment() -> dict:
    frame = _training_data()
    test = pd.read_csv(public_dir() / "test.csv")
    sample = pd.read_csv(public_dir() / "sample_submission.csv")
    if "id" not in test or any(target in test.columns for target in TARGETS):
        raise ValueError("public test must contain IDs without target columns")
    if list(sample.columns) != ["id", *TARGETS]:
        raise ValueError("unexpected NOMAD sample submission columns")
    if test.id.isna().any() or test.id.duplicated().any():
        raise ValueError("public test IDs are invalid")
    if not sample["id"].equals(test["id"]):
        raise ValueError("sample submission and test IDs differ")
    # The official preparation renumbers each split from 1, so an id repeats
    # across train and test.  It is only a submission key, and the repetition
    # is expected rather than a sign of leakage.
    train_features = _features(frame, "train")
    test_features = _features(test, "test")
    # Every row must be able to reach its own structure, and every structure
    # must parse; a candidate discovering otherwise mid-run would just crash.
    geometries = load_geometries(train_features.index) + load_geometries(
        test_features.index
    )
    atoms = sum(len(geometry.elements) for geometry in geometries)
    _split()
    return {
        "train_rows": len(frame),
        "test_rows": len(test),
        "geometry_files": len(geometries),
        "geometry_atoms": atoms,
        "gpu_required": False,
    }


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
    dataset = DatasetSplit(
        "nomad2018", _features(frame, "train"), frame[list(TARGETS)].to_numpy()
    )
    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)
    values = _prediction(model, _features(test, "test"), len(test))
    submission = pd.DataFrame(values, columns=TARGETS)
    submission.insert(0, "id", test.id.to_numpy())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(output)
