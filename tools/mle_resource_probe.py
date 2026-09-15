"""Evaluator-side no-score resource probes for sklearn-style MLE tasks."""

from __future__ import annotations

import gc
from typing import Any, Callable


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

    if torch is not None and torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

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
