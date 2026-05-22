"""Shared helpers for hyperparameter tuner scripts under tools/tuners/.

Loaded by warmstart_eval.py, grid_search.py, bo_search.py, cmaes_search.py.

Contract assumed of the candidate's train.py:
- BASE_PARAMS: dict[str, Any]
- SEARCH_SPACE: dict[str, tuple]
    - ("float", low, high) or ("float", low, high, "log")
    - ("int", low, high)
    - ("categorical", [opt1, opt2, ...])
- make_model(...): task-defined factory taking a params dict; signature
  is whatever the task's `evaluate_config_for_tuning` knows how to call.

Contract assumed of the candidate's prepare.py:
- evaluate_config_for_tuning(make_model, params: dict) -> float
  This is the **only** evaluation surface tuner scripts use. The task
  owns the fit/eval loop, the data iteration, the score aggregation, and
  the direction (higher- or lower-is-better, set by task.toml). Tuner
  scripts treat it as a black-box scalar oracle.

The test set is the optimization target during hyperparameter search;
the final value in results.tsv is therefore biased — see program.md and
the prepare.py docstring of test_score_for_tuning for why.

Tuner scripts persist trial-level history to tune_report.json
incrementally — see `append_trial` and `read_prior_trials`. Concurrency is
not an issue because tuner-orchestrator blocks on each subprocess; only
one writer touches the file at any time.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_SYMBOLS = ("BASE_PARAMS", "SEARCH_SPACE", "make_model")


def load_candidate_modules(candidate_path: Path) -> tuple[Any, Any]:
    """Load the candidate's train.py and prepare.py modules.

    Returns (train_module, prepare_module). Adds the candidate dir to
    sys.path so that the candidate's own `from prepare import ...` resolves
    to its own copy.
    """
    candidate_dir = candidate_path.parent
    sys.path.insert(0, str(candidate_dir))

    prepare_path = candidate_dir / "prepare.py"
    if not prepare_path.is_file():
        raise FileNotFoundError(f"candidate missing prepare.py: {prepare_path}")

    prepare_spec = importlib.util.spec_from_file_location("prepare", prepare_path)
    prepare_module = importlib.util.module_from_spec(prepare_spec)
    sys.modules["prepare"] = prepare_module
    prepare_spec.loader.exec_module(prepare_module)

    if not hasattr(prepare_module, "evaluate_config_for_tuning"):
        raise RuntimeError(
            f"candidate's prepare.py missing evaluate_config_for_tuning: "
            f"{prepare_path}"
        )

    train_spec = importlib.util.spec_from_file_location(
        "candidate_train", candidate_path
    )
    train_module = importlib.util.module_from_spec(train_spec)
    train_spec.loader.exec_module(train_module)

    for symbol in REQUIRED_SYMBOLS:
        if not hasattr(train_module, symbol):
            raise RuntimeError(
                f"candidate missing required symbol {symbol!r}: {candidate_path}"
            )

    return train_module, prepare_module


def cast_params_to_search_space(params: dict, search_space: dict) -> dict:
    """Coerce numpy/raw values in params to native Python types matching
    their SEARCH_SPACE entry. Ensures clean JSON output.
    """
    out = {}
    for key, raw in params.items():
        if key not in search_space:
            out[key] = _to_native(raw)
            continue
        kind = search_space[key][0]
        if kind == "int":
            out[key] = int(raw)
        elif kind == "float":
            out[key] = float(raw)
        else:
            out[key] = _to_native(raw)
    return out


def _to_native(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def search_space_for_json(search_space: dict) -> dict:
    """Convert SEARCH_SPACE tuples into JSON-serializable lists for output."""
    return {k: list(v) if isinstance(v, tuple) else v for k, v in search_space.items()}


def write_json(payload: dict) -> None:
    """Write the tuner result payload as a single JSON line on stdout."""
    print(json.dumps(payload, default=_to_native))


# ---------- tune_report.json helpers ----------


def read_tune_report(report_path: Path) -> dict:
    """Read tune_report.json. Return empty dict if missing."""
    if not report_path.exists():
        return {}
    with open(report_path) as f:
        return json.load(f)


def write_tune_report(report_path: Path, report: dict) -> None:
    """Atomically rewrite tune_report.json."""
    tmp = report_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2, default=_to_native)
    tmp.replace(report_path)


def append_warmstart_trial(report_path: Path, trial: dict) -> None:
    """Append one warm-start trial to phase_a.warm_start_configs."""
    report = read_tune_report(report_path)
    phase_a = report.setdefault(
        "phase_a", {"warm_start_configs": [], "base_score": None}
    )
    phase_a.setdefault("warm_start_configs", []).append(trial)
    write_tune_report(report_path, report)


def append_trial(report_path: Path, method: str, trial: dict) -> None:
    """Append one trial to phase_c.stages[method].trials. Single-writer-safe
    because tuner-orchestrator blocks on the subprocess.
    """
    report = read_tune_report(report_path)
    phase_c = report.setdefault("phase_c", {"stages": []})
    stages = phase_c.setdefault("stages", [])
    stage = next((s for s in stages if s.get("method") == method), None)
    if stage is None:
        stage = {"method": method, "trials": []}
        stages.append(stage)
    stage.setdefault("trials", []).append(trial)
    write_tune_report(report_path, report)


def read_prior_trials(report_path: Path) -> list[dict]:
    """Collect all (params, score) trials seen so far for this candidate.

    Flattens phase_a.base (the candidate-writer BASE_PARAMS evaluation)
    + phase_a.warm_start_configs + every phase_c.stages[*].trials. The
    base trial is included as the first prior so downstream tuners never
    lose sight of the candidate's starting point: BO injects it as a TPE
    completed trial; CMA-ES considers it when picking the best-so-far x0;
    grid's PatienceMonitor seeds start_best from it.
    """
    report = read_tune_report(report_path)
    trials: list[dict] = []
    phase_a = report.get("phase_a", {})
    base_params = phase_a.get("base_params")
    base_score = phase_a.get("base_score")
    if base_params is not None and base_score is not None:
        trials.append({"params": base_params, "score": base_score})
    trials.extend(phase_a.get("warm_start_configs", []))
    for stage in report.get("phase_c", {}).get("stages", []):
        trials.extend(stage.get("trials", []))
    return trials


# ---------- early-stopping ----------


class PatienceMonitor:
    """Pure-patience early stopping. No min_delta — strictly-better resets
    the counter. Direction-aware so it works for any metric (higher- or
    lower-is-better).

    Usage:
        monitor = PatienceMonitor(patience=10, lower_is_better=False,
                                  start_best=prior_best)
        for trial in search:
            score = evaluate(trial)
            if monitor.update(score):
                # patience exceeded; stop the search
                break
    """

    def __init__(
        self,
        patience: int,
        lower_is_better: bool = False,
        start_best: float | None = None,
    ):
        self.patience = patience
        self.lower_is_better = lower_is_better
        if start_best is not None:
            self.best = float(start_best)
        else:
            self.best = float("inf") if lower_is_better else float("-inf")
        self.since = 0

    def update(self, score: float) -> bool:
        """Update with one new score. Return True iff patience is exceeded
        (caller should stop)."""
        improved = (
            score < self.best if self.lower_is_better else score > self.best
        )
        if improved:
            self.best = score
            self.since = 0
        else:
            self.since += 1
        return self.since >= self.patience


def prior_best_score(
    prior_trials: list[dict], lower_is_better: bool = False
) -> float | None:
    """Return the best score among prior_trials, respecting direction.
    Returns None if the list is empty or no scores are numeric.
    """
    scores = [
        t["score"]
        for t in prior_trials
        if isinstance(t.get("score"), (int, float))
    ]
    if not scores:
        return None
    return min(scores) if lower_is_better else max(scores)
