"""Shared helpers for hyperparameter tuner scripts under tools/tuners/.

Loaded by warmstart_eval.py, grid_search.py, bo_search.py, cmaes_search.py.

Contract assumed of the candidate's train.py:
- BASE_PARAMS: dict[str, Any]
- SEARCH_SPACE: dict[str, tuple]
    - ("float", low, high) or ("float", low, high, "log")
    - ("int", low, high)
    - ("categorical", [opt1, opt2, ...])
- make_model(...): task-defined factory taking a params dict; signature
  is whatever the task's `score_fn` knows how to call.

Contract assumed of the candidate's prepare.py:
- score_fn(make_model, params: dict) -> float
  The **one** evaluation surface (task.toml `[evaluation].score_fn`). The task
  owns the fit/eval loop, the data iteration, and the score aggregation. Scores
  are **always lower-is-better** — the task's eval fn must conform (a
  higher-is-better metric returns its negation/complement). Tuner scripts treat
  it as a black-box scalar oracle to minimize.

There is a **single global `config → score` function** (no separate official
run): warm-start eval and Phase C tuning both call it, so the score it returns
IS the candidate's score recorded in ledger.json. The test set is the
optimization target, so that score is an optimistic estimate by construction.

Tuner scripts persist trial-level history to tune_report.json
incrementally — see `append_trial` and `read_prior_trials`. Concurrency is
not an issue because tuner-orchestrator blocks on each subprocess; only
one writer touches the file at any time.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validate_tasks import ROOT, parse_task_toml  # noqa: E402


REQUIRED_SYMBOLS = ("BASE_PARAMS", "SEARCH_SPACE", "make_model")
DEFAULT_SCORE_FN = "evaluate_config"


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

    train_spec = importlib.util.spec_from_file_location(
        "candidate_train", candidate_path
    )
    train_module = importlib.util.module_from_spec(train_spec)
    sys.modules["candidate_train"] = train_module
    train_spec.loader.exec_module(train_module)

    for symbol in REQUIRED_SYMBOLS:
        if not hasattr(train_module, symbol):
            raise RuntimeError(
                f"candidate missing required symbol {symbol!r}: {candidate_path}"
            )

    return train_module, prepare_module


def _infer_task_name(path: Path) -> str | None:
    """Recover the task name from a candidate path runs/<task>/<tag>/..."""
    parts = path.resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "runs" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def resolve_score_fn(prepare_module: Any, candidate_path: Path):
    """Return the task's single `config → score` evaluation callable.

    The function name comes from the task's `[evaluation].score_fn` in
    task.toml, defaulting to `evaluate_config`. This is the ONE evaluation
    surface — warm-start eval and Phase C tuning both call it; there is no
    separate official run. Signature: `score_fn(make_model, params) -> float`
    (lower-is-better). Tuner scripts stay task-agnostic: they call whatever the
    task declares rather than a hard-coded name.
    """
    fn_name = DEFAULT_SCORE_FN
    task_name = _infer_task_name(Path(candidate_path))
    if task_name:
        task_toml = ROOT / "tasks" / task_name / "task.toml"
        if task_toml.exists():
            try:
                configured = parse_task_toml(task_toml).get("evaluation", {})
            except ValueError:
                configured = {}
            name = configured.get("score_fn") if isinstance(configured, dict) else None
            if isinstance(name, str) and name:
                fn_name = name
    if not hasattr(prepare_module, fn_name):
        raise RuntimeError(
            f"prepare.py missing score fn {fn_name!r} "
            f"(task.toml [evaluation].score_fn)"
        )
    return getattr(prepare_module, fn_name)


def load_run_cfg(ref_path: Any, section: str) -> dict:
    """Per-run framework-hyperparameter overrides from `<run_dir>/framework_cfg.json`.

    `ref_path` is any path inside the run (a candidate train.py or tune_report);
    the function walks up ancestors to the run dir holding `framework_cfg.json`.
    Shape: `{"got": {...}, "tuner": {...}}`. Returns the requested section ({} if
    absent). Lets a Phase-3 OFAT trial override framework meta-params per run
    without code edits, so a headless run honors them. Pure stdlib.
    """
    p = Path(ref_path).resolve()
    for anc in p.parents:
        cfg = anc / "framework_cfg.json"
        if cfg.is_file():
            try:
                return dict(json.loads(cfg.read_text()).get(section, {}))
            except (ValueError, OSError):
                return {}
    return {}


def read_runtime_limit(ref_path: Any) -> float | None:
    """Top-level `per_runtime_limit` (seconds) from `<run_dir>/framework_cfg.json`
    (walking up from ref_path). Returns a positive float, else None (no limit)."""
    p = Path(ref_path).resolve()
    for anc in p.parents:
        cfg = anc / "framework_cfg.json"
        if cfg.is_file():
            try:
                v = json.loads(cfg.read_text()).get("per_runtime_limit")
            except (ValueError, OSError):
                return None
            try:
                v = float(v)
                return v if v > 0 else None
            except (TypeError, ValueError):
                return None
    return None


def timed_eval(evaluate, make_model, params: dict, candidate_path: Any) -> float:
    """Run ONE config eval, enforcing `per_runtime_limit` (framework_cfg.json).

    No limit -> call `evaluate(make_model, params)` in-process (fast path, no
    overhead). With a limit -> run the eval in a fresh subprocess in its own
    process group (`_eval_one.py`), hard-killing the whole tree on timeout and
    scoring it `+inf` (a too-slow config is treated as a crash). Scores are
    lower-is-better, so +inf is the worst possible — the tuner discards it.
    """
    limit = read_runtime_limit(candidate_path)
    if limit is None:
        return float(evaluate(make_model, params))
    eval_one = str(Path(__file__).resolve().parent / "_eval_one.py")
    posix = os.name == "posix"
    kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    if posix:
        kwargs["start_new_session"] = True       # own process group, for a clean group-kill
    elif os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        [sys.executable, eval_one, str(candidate_path), json.dumps(params)], **kwargs)
    try:
        out, _ = proc.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        try:
            if posix:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return float("inf")          # exceeded per_runtime_limit -> crash
    for line in out.splitlines():
        if line.startswith("RESULT:"):
            try:
                return float(line[len("RESULT:"):])
            except ValueError:
                return float("inf")
    return float("inf")              # child errored / printed no result -> crash


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


def set_stage_meta(report_path: Path, method: str, **meta: Any) -> None:
    """Merge stage-level summary fields (status, elapsed_seconds, early_stopped)
    into phase_c.stages[method] so a downstream summarizer can read them from
    the report rather than the script's stdout. Creates the stage if missing
    (e.g. a method that rejected before running any trial). Single-writer-safe
    because tuner-orchestrator blocks on the subprocess.
    """
    report = read_tune_report(report_path)
    phase_c = report.setdefault("phase_c", {"stages": []})
    stages = phase_c.setdefault("stages", [])
    stage = next((s for s in stages if s.get("method") == method), None)
    if stage is None:
        stage = {"method": method, "trials": []}
        stages.append(stage)
    stage.update(meta)
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
    trials.extend(phase_a.get("warm_start_configs", []))
    for stage in report.get("phase_c", {}).get("stages", []):
        trials.extend(stage.get("trials", []))
    return trials


def read_deferred_configs(report_path: Path) -> list[dict]:
    """Warm configs the extractor PROPOSED but did NOT evaluate at step 0+1
    (the K − K_eval deferred ones). They carry `params` only (no score) and are
    evaluated by the deep-tuner FIRST if this candidate is selected — BO enqueues
    them as initial trials, grid evaluates them before its sweep. Returns the list
    of param dicts (empty when none)."""
    phase_a = read_tune_report(report_path).get("phase_a", {})
    return [c["params"] for c in phase_a.get("deferred_configs", []) if c.get("params")]


# ---------- early-stopping ----------


class PatienceMonitor:
    """Pure-patience early stopping. No min_delta — strictly-better resets the
    counter. Score is always lower-is-better (the task's eval fn conforms).

    Usage:
        monitor = PatienceMonitor(patience=10, start_best=prior_best)
        for trial in search:
            score = evaluate(trial)
            if monitor.update(score):
                # patience exceeded; stop the search
                break
    """

    def __init__(self, patience: int, start_best: float | None = None):
        self.patience = patience
        self.best = float(start_best) if start_best is not None else float("inf")
        self.since = 0

    def update(self, score: float) -> bool:
        """Update with one new score. Return True iff patience is exceeded
        (caller should stop)."""
        if score < self.best:
            self.best = score
            self.since = 0
        else:
            self.since += 1
        return self.since >= self.patience


def prior_best_score(prior_trials: list[dict]) -> float | None:
    """Return the best (minimum) score among prior_trials, or None if the list
    is empty or no scores are numeric."""
    scores = [
        t["score"]
        for t in prior_trials
        if isinstance(t.get("score"), (int, float))
    ]
    return min(scores) if scores else None
