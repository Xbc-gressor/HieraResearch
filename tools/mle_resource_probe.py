"""Evaluator-side no-score probes for sklearn-style MLE tasks."""

from __future__ import annotations

import dataclasses
import gc
import math
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

_EPOCH_KEYS = ("epochs", "n_epochs", "max_epochs", "num_train_epochs")
_SMOKE_ROW_FRACTION = 0.02
_SMOKE_MIN_ROWS = 64


def _cuda_peak_reset(torch: Any) -> None:
    """Reset allocator peak stats after establishing the CUDA context.

    ``reset_peak_memory_stats`` does not lazily initialize the device, so
    calling it before any CUDA allocation fails with ``Invalid device
    argument`` (observed on sklearn candidates that never touch CUDA).
    """
    if torch is None or not torch.cuda.is_available():
        return
    torch.cuda.init()
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()


def run_resource_probe(
    make_model: Callable[[Any, dict], Any],
    params: dict,
    dataset: Any,
) -> dict:
    """Fit the candidate on the full training shape without scoring it.

    MLE tasks expose heterogeneous sklearn-style datasets, so the fixed
    evaluator supplies the dataset while this helper owns only telemetry. A
    full training-shape fit is the conservative envelope for these tasks;
    validation and test data are never touched.
    """
    try:
        import torch
    except ImportError:
        torch = None

    _cuda_peak_reset(torch)

    model = make_model(dataset, params)
    model.fit(dataset.x_train, dataset.y_train)

    peak_mb = None
    total_mb = None
    device_count = 0
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        device_count = torch.cuda.device_count()
        peaks = [
            torch.cuda.max_memory_allocated(device) / 1024 / 1024
            for device in range(device_count)
        ]
        observed_peak = max(peaks, default=0.0)
        peak_mb = round(observed_peak, 1) if observed_peak > 0 else None
        totals = [
            torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
            for device in range(device_count)
        ]
        total_mb = round(min(totals, default=0.0), 1) or None

    rows = len(dataset.x_train)
    del model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "ok",
        "objective_calls": 0,
        "peak_vram_mb": peak_mb,
        "total_vram_mb": total_mb,
        "probe_rows": rows,
        "probe_device_count": device_count,
        "envelope_covers_worst_case": True,
    }


def _subsample(values: Any, index: np.ndarray) -> Any:
    """Positional row selection that preserves the container type.

    Some tasks hand pd.DataFrame (or Series) splits to candidates on the
    score surface; silently downgrading them to ndarrays here would break
    the smoke's contract with the surface it is predicting.  Image tasks
    hand one array per sample and those arrays need not share a shape, so a
    sequence split is selected element-wise instead of through ``np.asarray``,
    which cannot represent an inhomogeneous stack.
    """
    iloc = getattr(values, "iloc", None)
    if iloc is not None:
        return iloc[index]
    if isinstance(values, (list, tuple)):
        selected = [values[int(position)] for position in index]
        return tuple(selected) if isinstance(values, tuple) else selected
    return np.asarray(values)[index]


def _smoke_split(dataset: Any, take: int) -> Any:
    """Same dataset object with a deterministic row subsample of the split."""
    index = np.sort(
        np.random.default_rng(0).choice(len(dataset.x_train), size=take, replace=False)
    )
    if dataclasses.is_dataclass(dataset) and not isinstance(dataset, type):
        return dataclasses.replace(
            dataset,
            x_train=_subsample(dataset.x_train, index),
            y_train=_subsample(dataset.y_train, index),
        )
    clone = SimpleNamespace(**dict(vars(dataset)))
    clone.x_train = _subsample(dataset.x_train, index)
    clone.y_train = _subsample(dataset.y_train, index)
    return clone


def run_smoke_probe(
    make_model: Callable[[Any, dict], Any],
    params: dict,
    dataset: Any,
) -> dict:
    """Fit the candidate on a small subsample for at most one epoch.

    The smoke answers one question in seconds: does this candidate's code
    construct and train at all? Envelope feasibility (full-shape time and
    memory) belongs to the per-point runtime limit on the score surface and
    to the resource probe, not here, so no VRAM telemetry is reported.
    """
    smoke_params = dict(params)
    capped = [
        key
        for key in _EPOCH_KEYS
        if key in smoke_params
        and isinstance(smoke_params[key], (int, float))
        and not isinstance(smoke_params[key], bool)
    ]
    for key in capped:
        smoke_params[key] = 1

    rows = len(dataset.x_train)
    take = min(rows, max(_SMOKE_MIN_ROWS, math.ceil(rows * _SMOKE_ROW_FRACTION)))
    if take < rows:
        dataset = _smoke_split(dataset, take)

    model = make_model(dataset, smoke_params)
    model.fit(dataset.x_train, dataset.y_train)
    del model
    return {
        "status": "ok",
        "objective_calls": 0,
        "probe": "smoke",
        "probe_rows": take,
        "capped_epoch_keys": capped,
    }
