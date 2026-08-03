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
IS the candidate's score recorded in ledger.json. An optional task-owned
preflight exercises feasibility without calling that score surface or consuming
its strict run-level budget.

Tuner scripts persist trial-level history to tune_report.json
incrementally — see `append_trial` and `read_prior_trials`. Phase C also holds
a candidate-local nonblocking file lock, so an accidental second orchestrator
cannot race report writes or objective reservations for the same candidate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
import sys

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - Phase C currently runs on POSIX hosts.
    fcntl = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation_budget import (  # noqa: E402
    EvaluationBudgetExhausted,
    reserve_evaluation,
)
from run_cfg import find_framework_cfg, read_framework_cfg  # noqa: E402
from validate_tasks import ROOT, parse_task_toml  # noqa: E402


REQUIRED_SYMBOLS = ("BASE_PARAMS", "SEARCH_SPACE", "make_model")
DEFAULT_SCORE_FN = "evaluate_config"
DEFAULT_PREFLIGHT_LIMIT = 180.0
DEEP_TUNE_INVOCATION_STARTED_AT = "invocation_started_at_epoch_seconds"
PHASE_C_LOCK_FILENAME = ".phase_c.lock"


class DeepTuneStageAdmissionError(RuntimeError):
    """The requested Phase-C method is not the next legal stage."""


class DeepTuneTimeExhausted(RuntimeError):
    """The candidate-level cumulative Phase-C wall-clock cap is exhausted."""

    def __init__(self, message: str, *, attempt_reserved: bool = False):
        super().__init__(message)
        self.attempt_reserved = attempt_reserved


class _PhaseCLock:
    """Small raw-fd owner; unlike ``open()``, GC never emits ResourceWarning."""

    def __init__(self, path: Path):
        self.path = path
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)

    def close(self) -> None:
        fd = getattr(self, "fd", None)
        if fd is None:
            return
        self.fd = None
        try:
            os.close(fd)
        except OSError:
            pass

    def __del__(self):
        self.close()


def is_finite_score(value: Any) -> bool:
    """Whether ``value`` is a real, finite tuner score.

    JSON's Python implementation accepts Infinity/NaN as floats, so a numeric
    type check alone is insufficient at every report/cache boundary.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _load_source_module(
    name: str,
    path: Path,
    *,
    source_bytes: bytes | None = None,
) -> Any:
    """Execute the bytes currently on disk, never an mtime-based ``.pyc``.

    Candidate files are rewritten repeatedly during one run.  Python's normal
    source loader may accept a stale bytecode cache when an edit keeps the same
    byte length and lands within the filesystem timestamp granularity.  That
    would let the AST gates inspect one contract while the evaluator executes
    another.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        code = compile(
            path.read_bytes() if source_bytes is None else source_bytes,
            str(path),
            "exec",
        )
        exec(code, module.__dict__)
    except BaseException:
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
        raise
    return module


def _canonical_contract_value(value: Any, *, symbol: str) -> str:
    """Type- and order-sensitive representation for literal contract values.

    Mapping order affects seeded grid/CMA encodings, so equal key/value sets in
    a different insertion order are still a runtime contract divergence.
    """
    try:
        return json.dumps(
            value,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"runtime {symbol} is not a JSON-stable literal mapping: {exc}"
        ) from None


def load_candidate_modules(
    candidate_path: Path,
    *,
    required_symbols: tuple[str, ...] = REQUIRED_SYMBOLS,
    expected_execution_revision: dict | None = None,
) -> tuple[Any, Any]:
    """Load the candidate's train.py and prepare.py modules.

    Returns (train_module, prepare_module). Adds the candidate dir to
    sys.path so that the candidate's own `from prepare import ...` resolves
    to its own copy.
    """
    candidate_path = Path(candidate_path)
    candidate_source = candidate_path.read_bytes()
    expected_contract = None
    execution_revision = None
    if required_symbols == REQUIRED_SYMBOLS:
        # This is the non-optional execution boundary. The CLI linters remain
        # useful diagnostics, but no tuner may import a duplicate or relationally
        # inconsistent contract even if an agent skipped those commands.
        from tune_tools import (
            _candidate_execution_revision,
            _read_literal_mapping,
            lint_contract,
        )

        contract = lint_contract(candidate_path)
        if not contract["ok"]:
            raise RuntimeError(
                "candidate tuning contract is invalid: "
                + json.dumps(contract["errors"], ensure_ascii=False)
            )
        expected_contract = {
            symbol: _read_literal_mapping(candidate_path, symbol)
            for symbol in ("PARAM_SCHEMA", "SEARCH_SPACE", "BASE_PARAMS")
        }
        if candidate_path.read_bytes() != candidate_source:
            raise RuntimeError(
                "candidate changed while its tuner contract was being validated"
            )
        execution_revision = _candidate_execution_revision(candidate_path)
        if (
            expected_execution_revision is not None
            and execution_revision != expected_execution_revision
        ):
            raise RuntimeError(
                "candidate execution revision changed before module import"
            )

    candidate_dir = candidate_path.parent
    sys.path.insert(0, str(candidate_dir))

    prepare_path = candidate_dir / "prepare.py"
    if not prepare_path.is_file():
        raise FileNotFoundError(f"candidate missing prepare.py: {prepare_path}")
    prepare_source = prepare_path.read_bytes()

    prepare_module = _load_source_module(
        "prepare",
        prepare_path,
        source_bytes=prepare_source,
    )
    train_module = _load_source_module(
        "candidate_train",
        candidate_path,
        source_bytes=candidate_source,
    )

    if (
        candidate_path.read_bytes() != candidate_source
        or prepare_path.read_bytes() != prepare_source
    ):
        raise RuntimeError(
            "candidate or evaluator changed while modules were being loaded"
        )

    for symbol in required_symbols:
        if not hasattr(train_module, symbol):
            raise RuntimeError(
                f"candidate missing required symbol {symbol!r}: {candidate_path}"
            )

    if expected_contract is not None:
        # Module-level code may not diverge from the AST literals the linters
        # read, or the tuner would search one space while scoring another.
        # Order matters: it feeds the seeded grid/CMA encodings.
        for symbol, expected in expected_contract.items():
            runtime = getattr(train_module, symbol, None)
            if not isinstance(runtime, dict):
                raise RuntimeError(
                    f"runtime {symbol} must remain a mapping after module import"
                )
            if _canonical_contract_value(
                runtime, symbol=symbol
            ) != _canonical_contract_value(expected, symbol=symbol):
                raise RuntimeError(
                    f"runtime {symbol} differs from its AST literal declaration; "
                    "module-level mutation is forbidden"
                )
        current_revision = _candidate_execution_revision(candidate_path)
        if current_revision != execution_revision:
            raise RuntimeError(
                "candidate execution revision changed while modules were loaded"
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
            configured = parse_task_toml(task_toml).get("evaluation", {})
            if not isinstance(configured, dict):
                raise RuntimeError("task.toml [evaluation] must be a table")
            name = configured.get("score_fn") if isinstance(configured, dict) else None
            if name is not None and (not isinstance(name, str) or not name):
                raise RuntimeError(
                    "task.toml evaluation.score_fn must be a non-empty string"
                )
            if name is not None:
                fn_name = name
    if not hasattr(prepare_module, fn_name):
        raise RuntimeError(
            f"prepare.py missing score fn {fn_name!r} "
            f"(task.toml [evaluation].score_fn)"
        )
    return getattr(prepare_module, fn_name)


def _configured_probe_name(candidate_path: Path, key: str) -> str | None:
    """Return a declared no-score probe symbol without importing candidate code."""
    task_name = _infer_task_name(Path(candidate_path))
    if not task_name:
        return None
    task_toml = ROOT / "tasks" / task_name / "task.toml"
    if not task_toml.exists():
        return None
    configured = parse_task_toml(task_toml).get("evaluation", {})
    if not isinstance(configured, dict):
        raise RuntimeError("task.toml [evaluation] must be a table")
    name = configured.get(key)
    if name is None:
        return None
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"task.toml evaluation.{key} must be a non-empty string")
    return name


def _configured_preflight_name(candidate_path: Path) -> str | None:
    """Return the declared candidate preflight symbol without importing code."""
    return _configured_probe_name(Path(candidate_path), "preflight_fn")


def _configured_resource_probe_name(candidate_path: Path) -> str | None:
    """Return the declared worst-case resource-probe symbol, if the task has one."""
    return _configured_probe_name(Path(candidate_path), "resource_probe_fn")


def _resolve_probe_fn(prepare_module: Any, candidate_path: Path, key: str):
    name = _configured_probe_name(Path(candidate_path), key)
    if name is None:
        return None
    if not hasattr(prepare_module, name):
        raise RuntimeError(
            f"prepare.py missing {key} {name!r} "
            f"(task.toml [evaluation].{key})"
        )
    return getattr(prepare_module, name)


def resolve_preflight_fn(prepare_module: Any, candidate_path: Path):
    """Return the optional task-owned no-score preflight callable.

    The hook is declared as ``evaluation.preflight_fn``.  It must accept
    ``(make_model, params)`` and must not call the task's score function or
    validation metric.  Tasks without the declaration keep the historical
    direct-evaluation behavior.
    """
    return _resolve_probe_fn(prepare_module, candidate_path, "preflight_fn")


def resolve_resource_probe_fn(prepare_module: Any, candidate_path: Path):
    """Return the optional task-owned worst-case resource probe.

    Declared as ``evaluation.resource_probe_fn``, with the same no-score
    contract as ``preflight_fn``.  It exists because a correctness preflight
    measures whichever training shape ``run()`` starts with, which is not an
    upper bound when a candidate ramps that shape mid-run.  Tasks without the
    declaration leave the space clamp trusting ``preflight_fn`` as before.
    """
    return _resolve_probe_fn(prepare_module, candidate_path, "resource_probe_fn")


def load_run_cfg(ref_path: Any, section: str) -> dict:
    """Per-run framework-hyperparameter overrides from `<run_dir>/framework_cfg.json`.

    `ref_path` is any path inside the run (a candidate train.py or tune_report);
    the function walks up ancestors to the run dir holding `framework_cfg.json`.
    Shape: `{"got": {...}, "tuner": {...}}`. Returns the requested section ({} if
    absent). Lets a Phase-3 OFAT trial override framework meta-params per run
    without code edits, so a headless run honors them. Pure stdlib.
    A cfg file that exists but cannot be parsed raises RunConfigError instead of
    silently reverting to defaults.
    """
    cfg = find_framework_cfg(ref_path)
    if cfg is None:
        return {}
    return dict(read_framework_cfg(cfg).get(section, {}))


def _acquire_phase_c_lock(ref_path: Any):
    """Hold one non-blocking candidate-local Phase-C writer lock."""
    if fcntl is None:
        raise DeepTuneStageAdmissionError(
            "Phase-C tuning requires POSIX flock support"
        )
    lock_path = Path(ref_path).resolve().parent / PHASE_C_LOCK_FILENAME
    try:
        handle = _PhaseCLock(lock_path)
    except OSError as exc:
        raise DeepTuneStageAdmissionError(
            f"cannot open Phase-C lock {lock_path}: {exc}"
        ) from None
    try:
        fcntl.flock(handle.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise DeepTuneStageAdmissionError(
            f"another Phase-C tuner already holds {lock_path}"
        ) from exc
    except OSError as exc:
        handle.close()
        raise DeepTuneStageAdmissionError(
            f"cannot acquire Phase-C lock {lock_path}: {exc}"
        ) from None
    return handle


def deep_tune_time_budget(
    ref_path: Any,
    report_path: Path,
    method: str,
) -> dict:
    """Admit and begin one deterministic Phase-C stage invocation.

    This is the first boundary every search script calls, before importing an
    optimizer or candidate, probing preflight, or constructing a study/grid. It
    validates the deterministic method/fallback chain, recovers wall time from
    an interrupted ``running`` invocation, and persists the new invocation's
    epoch start before any expensive work begins.

    Normal invocations use ``time.monotonic`` for precise accounting. The epoch
    start is solely a crash-recovery journal: if the process disappears before
    terminal metadata is written, the next invocation charges the abandoned
    interval before granting any remaining time.
    """
    lock_handle = _acquire_phase_c_lock(ref_path)
    try:
        budget = _deep_tune_time_budget_locked(ref_path, report_path, method)
    except BaseException:
        lock_handle.close()
        raise
    # The live file handle is intentionally retained in the invocation-local
    # budget object. Its lifetime is the tuner process/function lifetime, so the
    # OS releases the lock even after SIGKILL while concurrent tuners fail closed.
    budget["_phase_c_lock_handle"] = lock_handle
    return budget


def _deep_tune_time_budget_locked(
    ref_path: Any,
    report_path: Path,
    method: str,
) -> dict:
    """Locked implementation for :func:`deep_tune_time_budget`."""
    started_monotonic = time.monotonic()
    started_epoch = time.time()
    # Phase C has no wall-clock limit. The budget is trial-denominated:
    # patience, n_trials, the per-candidate objective cap, and the run share.
    # A seconds cap sized below the sampler's startup regime silently degraded
    # every stage to random fallback draws (run 0730-ds-ex100-1: 3600 s at
    # ~450 s/eval never reached TPE's 10-trial startup, in every stage). The
    # legacy `tuner.deep_tune_time_limit_seconds` key still parses but is
    # ignored. Elapsed accounting below is kept for receipts and crash
    # recovery, not for enforcement.
    limit = math.inf
    report = read_tune_report(report_path)
    phase_c = report.get("phase_c")
    if phase_c is None:
        phase_c = {"stages": []}
        report["phase_c"] = phase_c
    elif not isinstance(phase_c, dict):
        raise DeepTuneStageAdmissionError("phase_c must be an object")
    stages = phase_c.get("stages")
    if stages is None and "stages" not in phase_c:
        stages = []
        phase_c["stages"] = stages
    if not isinstance(stages, list) or not all(
        isinstance(stage, dict) for stage in stages
    ):
        raise DeepTuneStageAdmissionError(
            "phase_c.stages must be a list of objects"
        )

    # AST-only: reject a wrong/repeated fallback before importing candidate code
    # or admitting any preflight/objective work.
    from tune_tools import (
        _TERMINAL_STAGE_STATUSES,
        _read_search_space,
        has_applied_close,
        has_validated_applied_close,
        select_method,
        validate_candidate_execution_revision,
        validate_phase_a_candidate_state,
    )

    try:
        bouts = stages_by_bout(stages)
        validate_phase_a_candidate_state(
            report,
            Path(ref_path),
            require_warm_base_applied=not has_applied_close(report),
        )
        candidate_execution_revision = validate_candidate_execution_revision(
            report,
            Path(ref_path),
        )
        search_space = _read_search_space(Path(ref_path))
    except (SystemExit, ValueError) as exc:
        raise DeepTuneStageAdmissionError(str(exc)) from None
    selected = select_method(len(search_space))
    method_chain = [selected["method"], *selected["fallback"]]
    if method not in method_chain:
        raise DeepTuneStageAdmissionError(
            f"method {method!r} is outside deterministic chain {method_chain!r}"
        )
    position = method_chain.index(method)

    current = bouts[-1] if bouts else []
    methods = [stage.get("method") for stage in current]
    if not all(isinstance(value, str) for value in methods):
        raise DeepTuneStageAdmissionError(
            f"phase_c stage methods must be strings: {methods!r}"
        )
    if len(methods) != len(set(methods)):
        raise DeepTuneStageAdmissionError(
            f"phase_c bout contains duplicate method stages: {methods!r}"
        )
    expected_prefix = method_chain[:position]
    actual_prefix = methods[:position]
    if actual_prefix != expected_prefix or any(
        current[index].get("status") != "rejected"
        for index in range(min(position, len(current)))
    ):
        raise DeepTuneStageAdmissionError(
            f"method {method!r} requires rejected prefix {expected_prefix!r}; "
            f"found methods/statuses "
            f"{[(stage.get('method'), stage.get('status')) for stage in current]!r}"
        )

    current_terminal = bool(current) and all(
        item.get("status") in _TERMINAL_STAGE_STATUSES for item in current
    )
    stage = None
    bout_index = len(bouts) - 1 if bouts else 0
    if current_terminal and method == method_chain[0]:
        # The previous bout's chain closed out (finalizable or exhausted). A
        # validated applied close lets a NEW bout restart the chain — the new
        # stage reuses the method under the next bout_index.
        if not has_validated_applied_close(report):
            raise DeepTuneStageAdmissionError(
                "the previous bout must be finalized "
                "(tools/finalize_tuning.py) before a new bout starts"
            )
        bout_index = len(bouts)
        stage = {"method": method, "trials": [], "bout_index": bout_index}
        stages.append(stage)
    else:
        stage = next(
            (item for item in current if item.get("method") == method),
            None,
        )
        if stage is None:
            if len(current) != position:
                raise DeepTuneStageAdmissionError(
                    f"method {method!r} is not the next Phase-C stage of its bout"
                )
            stage = {"method": method, "trials": []}
            if bout_index > 0:
                stage["bout_index"] = bout_index
            stages.append(stage)
        else:
            if current.index(stage) != position or len(current) != position + 1:
                raise DeepTuneStageAdmissionError(
                    f"method {method!r} is not the active final Phase-C stage "
                    "of its bout"
                )
            if stage.get("status") != "running":
                raise DeepTuneStageAdmissionError(
                    f"terminal Phase-C method {method!r} cannot be rerun "
                    f"within its bout (status={stage.get('status')!r})"
                )

    # Recover every abandoned invocation conservatively before calculating the
    # candidate-level total. A legal report has at most the active method here,
    # but stale timestamps must never disappear even in a malformed report.
    for item in stages:
        invocation_started = item.pop(DEEP_TUNE_INVOCATION_STARTED_AT, None)
        if (
            isinstance(invocation_started, (int, float))
            and not isinstance(invocation_started, bool)
            and math.isfinite(float(invocation_started))
        ):
            prior = item.get("elapsed_seconds", 0.0)
            prior = (
                float(prior)
                if isinstance(prior, (int, float))
                and not isinstance(prior, bool)
                and math.isfinite(float(prior))
                and float(prior) >= 0
                else 0.0
            )
            item["elapsed_seconds"] = prior + max(
                0.0, started_epoch - float(invocation_started)
            )
            recovered = item.get("recovered_interrupted_invocations", 0)
            recovered = (
                int(recovered)
                if isinstance(recovered, int) and not isinstance(recovered, bool)
                else 0
            )
            item["recovered_interrupted_invocations"] = recovered + 1

    total_used = 0.0
    stage_used = 0.0
    for item in stages:
        elapsed = item.get("elapsed_seconds")
        if (
            isinstance(elapsed, (int, float))
            and not isinstance(elapsed, bool)
            and math.isfinite(float(elapsed))
            and float(elapsed) > 0
        ):
            total_used += float(elapsed)
            if item.get("method") == method:
                stage_used += float(elapsed)
    remaining = max(0.0, limit - total_used)
    # Persist the journal even at zero remaining. The caller normally closes it
    # immediately as time_exhausted, but a kill in that tiny window must leave a
    # resumable active stage rather than an unstatused, permanently inadmissible
    # record.
    stage.update(
        {
            "status": "running",
            DEEP_TUNE_INVOCATION_STARTED_AT: started_epoch,
        }
    )
    write_tune_report(report_path, report)
    return {
        # Receipts serialize this as null: no wall-clock limit exists anymore.
        # Internal arithmetic uses remaining_seconds (always +inf).
        "limit_seconds": None,
        "used_seconds": total_used,
        "stage_used_seconds": stage_used,
        "remaining_seconds": remaining,
        "started_monotonic": started_monotonic,
        "started_epoch": started_epoch,
        "candidate_execution_revision": candidate_execution_revision,
        "bout_index": bout_index,
    }


def deep_tune_stage_elapsed(time_budget: dict) -> float:
    """Exact cumulative elapsed seconds for the current method stage."""
    return float(time_budget["stage_used_seconds"]) + max(
        0.0, time.monotonic() - float(time_budget["started_monotonic"])
    )


def deep_tune_time_remaining(time_budget: dict) -> float:
    """Remaining Phase-C seconds: always +inf since the wall clock was removed.

    Kept so callers (optuna ``timeout=``, ``phase_time_limit_seconds=``
    suppliers) keep working unchanged; +inf maps to "no limit" downstream.
    """
    invocation_elapsed = max(
        0.0, time.monotonic() - float(time_budget["started_monotonic"])
    )
    return max(0.0, float(time_budget["remaining_seconds"]) - invocation_elapsed)


def ensure_deep_tune_time_remaining(time_budget: dict) -> None:
    """No-op retained for call-site compatibility: no wall cap exists."""
    if deep_tune_time_remaining(time_budget) <= 0:
        raise DeepTuneTimeExhausted(
            "candidate deep-tune wall-clock allocation exhausted"
        )


def read_runtime_limit(ref_path: Any) -> float | None:
    """Top-level `per_runtime_limit` (seconds) from `<run_dir>/framework_cfg.json`
    (walking up from ref_path). Returns a positive float, else None (no limit).
    A cfg file that exists but cannot be parsed raises RunConfigError: silently
    dropping the limit would let an evaluation run unbounded."""
    cfg = find_framework_cfg(ref_path)
    if cfg is None:
        return None
    v = read_framework_cfg(cfg).get("per_runtime_limit")
    try:
        v = float(v)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def read_preflight_limit(ref_path: Any) -> float:
    """Return the bounded no-score preflight timeout for one config."""
    cfg = find_framework_cfg(ref_path)
    if cfg is None:
        return DEFAULT_PREFLIGHT_LIMIT
    data = read_framework_cfg(cfg)
    value = data.get("preflight_runtime_limit")
    if value is not None:
        try:
            value = float(value)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    runtime = data.get("per_runtime_limit")
    try:
        runtime = float(runtime)
        if runtime > 0:
            return min(DEFAULT_PREFLIGHT_LIMIT, runtime)
    except (TypeError, ValueError):
        pass
    return DEFAULT_PREFLIGHT_LIMIT


def _communicate_with_limit(
    command: list[str],
    *,
    limit: float,
    label: str,
) -> tuple[str, str, int]:
    posix = os.name == "posix"
    kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    if posix:
        kwargs["start_new_session"] = True
    elif os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(command, **kwargs)
    try:
        out, err = proc.communicate(timeout=limit)
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
        raise TimeoutError(f"{label}={limit:g}s") from None
    return out, err, int(proc.returncode)


def _resolve_phase_time_limit(
    value: float | Callable[[], float] | None,
) -> float | None:
    """Resolve a live Phase-C remaining-time supplier into a strict limit."""
    if callable(value):
        value = value()
    if value is None:
        return None
    try:
        limit = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "phase_time_limit_seconds must be a finite positive number"
        ) from None
    if limit == math.inf:
        # No Phase-C wall clock exists (removed); only per_runtime_limit binds.
        return None
    if not math.isfinite(limit) or limit <= 0:
        raise DeepTuneTimeExhausted(
            "candidate deep-tune wall-clock allocation exhausted"
        )
    return limit


def timed_eval(
    evaluate,
    make_model,
    params: dict,
    candidate_path: Any,
    *,
    phase: str = "unknown",
    method: str = "unknown",
    phase_time_limit_seconds: float | Callable[[], float] | None = None,
    expected_execution_revision: dict | None = None,
) -> float:
    """Run ONE config eval under task and Phase-C hard wall-clock limits.

    Outside Phase C, no task limit keeps the historical in-process fast path.
    Whenever either ``per_runtime_limit`` or a Phase-C remaining allocation is
    supplied, run in a fresh subprocess and hard-kill its process group at the
    smaller limit. The objective reservation happens first, so a subprocess
    timeout remains an admitted/charged evaluation attempt.

    Timeouts, child-process errors, missing results, and non-finite scores raise
    so callers record an auditable failed trial instead of caching ``+inf`` as
    if it were a successful score.
    """
    # Fail before reservation when the caller already knows no Phase-C time
    # remains. A live supplier is resolved again after atomic reservation so
    # lock contention/I/O cannot grant the child a stale, overly large timeout.
    phase_limit = _resolve_phase_time_limit(phase_time_limit_seconds)
    reserve_evaluation(
        candidate_path,
        params=params,
        phase=phase,
        method=method,
    )
    try:
        phase_limit = _resolve_phase_time_limit(phase_time_limit_seconds)
    except DeepTuneTimeExhausted as exc:
        raise DeepTuneTimeExhausted(
            str(exc),
            attempt_reserved=True,
        ) from None
    runtime_limit = read_runtime_limit(candidate_path)
    if runtime_limit is None and phase_limit is None:
        score = float(evaluate(make_model, params))
        if not is_finite_score(score):
            raise ValueError(f"evaluation returned non-finite score: {score!r}")
        return score
    phase_binds = (
        phase_limit is not None
        and (runtime_limit is None or phase_limit <= runtime_limit)
    )
    if runtime_limit is None:
        limit = float(phase_limit)
    elif phase_limit is None:
        limit = runtime_limit
    else:
        limit = min(runtime_limit, phase_limit)
    eval_one = str(Path(__file__).resolve().parent / "_eval_one.py")
    try:
        out, err, returncode = _communicate_with_limit(
            [
                sys.executable,
                eval_one,
                str(candidate_path),
                json.dumps(params),
                json.dumps(expected_execution_revision),
            ],
            limit=limit,
            label=(
                "evaluation exceeded Phase-C remaining wall time"
                if phase_binds
                else "evaluation exceeded per_runtime_limit"
            ),
        )
    except TimeoutError as exc:
        if phase_binds:
            raise DeepTuneTimeExhausted(
                str(exc),
                attempt_reserved=True,
            ) from None
        raise
    for line in out.splitlines():
        if line.startswith("RESULT:"):
            try:
                score = float(line[len("RESULT:"):])
            except ValueError:
                raise ValueError(
                    f"evaluation subprocess printed invalid result: {line!r}"
                ) from None
            if not is_finite_score(score):
                raise ValueError(f"evaluation returned non-finite score: {score!r}")
            return score

    detail = err.strip()
    if len(detail) > 4000:
        detail = "...[stderr truncated]...\n" + detail[-4000:]
    message = (
        f"evaluation subprocess exited with code {returncode} "
        "without a RESULT line"
    )
    if detail:
        message += f"\nchild stderr:\n{detail}"
    raise RuntimeError(message)


def timed_preflight(
    params: dict,
    candidate_path: Any,
    *,
    phase_time_limit_seconds: float | Callable[[], float] | None = None,
    expected_execution_revision: dict | None = None,
    probe_mode: str = "preflight",
) -> dict | None:
    """Run one task-owned no-score probe in an isolated subprocess.

    ``probe_mode`` selects which task hook runs: ``"preflight"`` (the
    correctness check, ``evaluation.preflight_fn``) or ``"resource"`` (the
    worst-case memory envelope, ``evaluation.resource_probe_fn``).

    Returns ``None`` when the task declares no hook for the requested mode, so
    callers should choose the mode from the task's declarations rather than
    treating the absence as a failure.  Crucially, neither mode ever reserves an
    objective evaluation slot.
    """
    if probe_mode not in {"preflight", "resource"}:
        raise ValueError(f"unknown probe_mode {probe_mode!r}")
    configured = (
        _configured_resource_probe_name(Path(candidate_path))
        if probe_mode == "resource"
        else _configured_preflight_name(Path(candidate_path))
    )
    if configured is None:
        return None
    preflight_one = str(Path(__file__).resolve().parent / "_preflight_one.py")
    phase_limit = _resolve_phase_time_limit(phase_time_limit_seconds)
    configured_limit = read_preflight_limit(candidate_path)
    phase_binds = (
        phase_limit is not None and phase_limit <= configured_limit
    )
    limit = (
        configured_limit
        if phase_limit is None
        else min(configured_limit, phase_limit)
    )
    try:
        out, err, returncode = _communicate_with_limit(
            [
                sys.executable,
                preflight_one,
                str(candidate_path),
                json.dumps(params),
                json.dumps(expected_execution_revision),
                probe_mode,
            ],
            limit=limit,
            label=(
                "preflight exceeded Phase-C remaining wall time"
                if phase_binds
                else "preflight exceeded preflight_runtime_limit"
            ),
        )
    except TimeoutError as exc:
        if phase_binds:
            raise DeepTuneTimeExhausted(str(exc)) from None
        raise
    for line in out.splitlines():
        if not line.startswith("PREFLIGHT:"):
            continue
        payload = line[len("PREFLIGHT:"):]
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            raise ValueError(f"preflight subprocess printed invalid result: {line!r}") from None
        return value if isinstance(value, dict) else {"result": value}

    detail = err.strip()
    if len(detail) > 4000:
        detail = "...[stderr truncated]...\n" + detail[-4000:]
    message = (
        f"preflight subprocess exited with code {returncode} "
        "without a PREFLIGHT line"
    )
    if detail:
        message += f"\nchild stderr:\n{detail}"
    raise RuntimeError(message)


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


def stage_bout_index(stage: dict) -> int:
    """0-based bout a Phase-C stage belongs to (legacy unstamped stages: 0)."""
    value = stage.get("bout_index", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def stages_by_bout(stages: list) -> list[list[dict]]:
    """Group ordered stages into bouts. Bout indices must start at 0, be
    contiguous, and never regress along the list."""
    bouts: list[list[dict]] = []
    for index, stage in enumerate(stages):
        bout = stage_bout_index(stage)
        if bout < len(bouts) - 1:
            raise ValueError(
                f"phase_c.stages[{index}] bout_index {bout} regresses below an "
                "earlier bout"
            )
        if bout > len(bouts):
            raise ValueError(
                f"phase_c.stages[{index}] bout_index {bout} skips a bout"
            )
        if bout == len(bouts):
            bouts.append([])
        bouts[-1].append(stage)
    return bouts


def append_preflight_attempt(
    report_path: Path,
    *,
    source: str,
    params: dict,
    status: str,
    result: dict | None = None,
    failure: dict | None = None,
) -> None:
    """Append one no-score feasibility attempt to the candidate report."""
    report = read_tune_report(report_path)
    preflight = report.setdefault("preflight", {"attempts": [], "invocations": 0})
    row = {
        "params": params,
        "source": source,
        "status": status,
    }
    if result is not None:
        row["result"] = result
    if failure is not None:
        row.update(failure)
    preflight.setdefault("attempts", []).append(row)
    preflight["invocations"] = len(preflight["attempts"])
    if status == "failed":
        preflight["status"] = "failed"
    elif preflight.get("status") != "failed":
        preflight["status"] = "ok"
    write_tune_report(report_path, report)


def append_trial(report_path: Path, method: str, trial: dict) -> None:
    """Append one trial to the LAST phase_c.stages[method] (the active stage of
    that method's current bout; the same method recurs across bouts).
    Single-writer-safe because tuner-orchestrator blocks on the subprocess.
    """
    report = read_tune_report(report_path)
    phase_c = report.setdefault("phase_c", {"stages": []})
    stages = phase_c.setdefault("stages", [])
    stage = next((s for s in reversed(stages) if s.get("method") == method), None)
    if stage is None:
        stage = {"method": method, "trials": []}
        stages.append(stage)
    stage.setdefault("trials", []).append(trial)
    write_tune_report(report_path, report)


def set_stage_meta(
    report_path: Path,
    method: str,
    *,
    bout_index: int | None = None,
    **meta: Any,
) -> None:
    """Merge stage-level summary fields (status, elapsed_seconds, early_stopped)
    into the LAST phase_c.stages[method] (the active stage of that method's
    current bout; the same method recurs across bouts) so a downstream
    summarizer can read them from the report rather than the script's stdout.
    Creates the stage if missing (e.g. a method that rejected before running
    any trial), stamping ``bout_index`` on it when a positive bout is given.
    Single-writer-safe because tuner-orchestrator blocks on the subprocess.
    """
    report = read_tune_report(report_path)
    phase_c = report.setdefault("phase_c", {"stages": []})
    stages = phase_c.setdefault("stages", [])
    stage = next((s for s in reversed(stages) if s.get("method") == method), None)
    if stage is None:
        stage = {"method": method, "trials": []}
        if isinstance(bout_index, int) and not isinstance(bout_index, bool) and bout_index > 0:
            stage["bout_index"] = bout_index
        stages.append(stage)
    status = meta.get("status")
    if status is not None and status != "running":
        invocation_started = stage.pop(DEEP_TUNE_INVOCATION_STARTED_AT, None)
        if "elapsed_seconds" not in meta and (
            isinstance(invocation_started, (int, float))
            and not isinstance(invocation_started, bool)
            and math.isfinite(float(invocation_started))
        ):
            prior = stage.get("elapsed_seconds", 0.0)
            prior = (
                float(prior)
                if isinstance(prior, (int, float))
                and not isinstance(prior, bool)
                and math.isfinite(float(prior))
                and float(prior) >= 0
                else 0.0
            )
            meta["elapsed_seconds"] = prior + max(
                0.0, time.time() - float(invocation_started)
            )
    stage.update(meta)
    write_tune_report(report_path, report)


def read_prior_trials(report_path: Path) -> list[dict]:
    """Collect all (params, score) trials seen so far for this candidate.

    Flattens phase_a.base (the candidate-writer BASE_PARAMS evaluation)
    + phase_a.warm_start_configs + every phase_c.stages[*].trials. The
    base trial is included as the first prior so downstream tuners never
    lose sight of the candidate's starting point: BO injects it as a TPE
    completed trial; CMA-ES considers it when picking the best-so-far x0.
    Patience seeding (best AND streak, including scoreless trials) goes
    through `prior_patience_state` instead.
    """
    report = read_tune_report(report_path)
    trials: list[dict] = []
    phase_a = report.get("phase_a", {})
    trials.extend(phase_a.get("warm_start_configs", []))
    for stage in report.get("phase_c", {}).get("stages", []):
        trials.extend(stage.get("trials", []))
    return [
        trial
        for trial in trials
        if isinstance(trial.get("params"), dict)
        and is_finite_score(trial.get("score"))
    ]


def read_prior_infeasible_trials(report_path: Path) -> list[dict]:
    """Persisted trials that ended config-infeasible, for re-injection into a
    fresh study as constrained-infeasible points (same traversal as
    read_prior_trials, which keeps only scored trials and would lose them on
    every tuner restart).

    Included: preflight_rejected trials (infeasible by definition) and failed
    trials explicitly classified config_infeasible at record time. Legacy
    failed trials without the flag are NOT treated as infeasible: an
    unclassified crash is not evidence about the parameter region.
    """
    report = read_tune_report(report_path)
    trials: list[dict] = []
    for stage in report.get("phase_c", {}).get("stages", []):
        trials.extend(stage.get("trials", []))
    return [
        trial
        for trial in trials
        if isinstance(trial.get("params"), dict)
        and not is_finite_score(trial.get("score"))
        and (
            trial.get("status") == "preflight_rejected"
            or (trial.get("status") == "failed" and trial.get("config_infeasible"))
        )
    ]


def read_deferred_configs(report_path: Path) -> list[dict]:
    """Warm configs the extractor PROPOSED but did NOT evaluate at step 0+1
    (the K − K_eval deferred ones). They carry `params` only (no score) and are
    evaluated by the deep-tuner FIRST if this candidate is selected — BO enqueues
    them as initial trials, grid evaluates them before its sweep. Returns the list
    of param dicts (empty when none)."""
    phase_a = read_tune_report(report_path).get("phase_a", {})
    return [c["params"] for c in phase_a.get("deferred_configs", []) if c.get("params")]


def read_attempted_configs(report_path: Path) -> list[dict]:
    """Every config already evaluated or rejected by preflight.

    Deferred configs are intentionally excluded until a tuner actually attempts
    them. Deterministic/resumed methods use this view to avoid spending another
    objective slot on an identical point.
    """
    report = read_tune_report(report_path)
    configs: list[dict] = []
    for trial in report.get("phase_a", {}).get("warm_start_configs", []):
        if isinstance(trial, dict) and isinstance(trial.get("params"), dict):
            configs.append(trial["params"])
    for stage in report.get("phase_c", {}).get("stages", []):
        if not isinstance(stage, dict):
            continue
        for trial in stage.get("trials", []):
            if isinstance(trial, dict) and isinstance(trial.get("params"), dict):
                configs.append(trial["params"])
    return configs


def params_identity(params: dict) -> str:
    """Stable JSON identity for one fully materialized parameter mapping."""
    return json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_to_native,
    )


def attempted_config_identities(
    report_path: Path,
    search_space: dict,
) -> set[str]:
    """Canonical identities of historical configs compatible with this space."""
    identities: set[str] = set()
    for params in read_attempted_configs(report_path):
        try:
            normalized = cast_params_to_search_space(dict(params), search_space)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if set(normalized) == set(search_space):
            identities.add(params_identity(normalized))
    return identities


def deduplicate_configs(
    configs: list[dict],
    *,
    seen: set[str] | None = None,
) -> tuple[list[dict], int, set[str]]:
    """Keep first occurrences in order and return ``(unique, skipped, seen)``."""
    identities = set() if seen is None else set(seen)
    unique: list[dict] = []
    skipped = 0
    for params in configs:
        identity = params_identity(params)
        if identity in identities:
            skipped += 1
            continue
        identities.add(identity)
        unique.append(params)
    return unique, skipped, identities


# ---------- early-stopping ----------


class PatienceMonitor:
    """Pure-patience early stopping. No min_delta — strictly-better resets the
    counter. Score is always lower-is-better (the task's eval fn conforms).

    Every trial counts: a trial that produced no score (crash, timeout,
    preflight-rejected) is a non-improvement and increments the counter via
    `update_failed`, so failure streaks cannot dilute or sidestep patience.
    `start_since` seeds the counter from `prior_patience_state` so a resumed
    or restarted search continues the persisted Phase-C streak instead of
    getting a fresh patience window. Warm screening contributes `start_best`
    but never `start_since`.

    Usage:
        best, streak = prior_patience_state(report_path)
        monitor = PatienceMonitor(patience=10, start_best=best, start_since=streak)
        for trial in search:
            score = evaluate(trial)
            if monitor.update(score) if score is not None else monitor.update_failed():
                # patience exceeded; stop the search
                break
    """

    def __init__(
        self,
        patience: int,
        start_best: float | None = None,
        start_since: int = 0,
    ):
        self.patience = patience
        self.best = float(start_best) if start_best is not None else float("inf")
        self.since = int(start_since)

    def update(self, score: float) -> bool:
        """Update with one new score. Return True iff patience is exceeded
        (caller should stop)."""
        if score < self.best:
            self.best = score
            self.since = 0
        else:
            self.since += 1
        return self.since >= self.patience

    def update_failed(self) -> bool:
        """Update with one trial that produced no score (failed or
        preflight-rejected). It burned wall-clock — and usually an objective
        slot — without yielding an improvement, so it counts toward patience.
        Never moves `best`. Return True iff patience is exceeded."""
        self.since += 1
        return self.since >= self.patience


def prior_best_score(prior_trials: list[dict]) -> float | None:
    """Return the best (minimum) score among prior_trials, or None if the list
    is empty or no scores are numeric."""
    scores = [
        t["score"]
        for t in prior_trials
        if is_finite_score(t.get("score"))
    ]
    return min(scores) if scores else None


def prior_patience_state(
    report_path: Path,
    bout_index: int | None = None,
) -> tuple[float | None, int]:
    """Replay persisted trials into (best_score, patience streak) for
    PatienceMonitor seeding. The improvement bar is GLOBAL (warm screening
    plus every bout's trials); the patience streak is scoped to one bout
    (default: the current/max bout), so a continuation bout starts a fresh
    window while still having to beat the run's best to reset.

    Only Phase-C rows of the scoped bout increment the streak. Warm screening
    is a deliberate spread over distinct numeric regimes, not a stalled
    optimizer. The inherited config-0 fidelity control is not an incumbent, so
    it cannot set ``best`` or reset patience; a Phase-C duplicate of its exact
    params counts as a spent non-improving trial.
    """
    report = read_tune_report(report_path)
    phase_a = report.get("phase_a", {})
    stages = report.get("phase_c", {}).get("stages", [])
    if bout_index is None:
        bout_index = max((stage_bout_index(s) for s in stages), default=0)

    best: float | None = None
    inherited_param_ids: set[str] = set()
    for trial in phase_a.get("warm_start_configs", []):
        if (
            trial.get("role") == "inherited_control"
            and isinstance(trial.get("params"), dict)
        ):
            inherited_param_ids.add(params_identity(trial["params"]))
            continue
        score = trial.get("score")
        if is_finite_score(score) and (best is None or float(score) < best):
            best = float(score)

    def absorbs(trial: dict) -> bool:
        """Whether the trial improves the running best (reset) or not."""
        nonlocal best
        is_inherited_duplicate = (
            isinstance(trial.get("params"), dict)
            and params_identity(trial["params"]) in inherited_param_ids
        )
        score = trial.get("score")
        if (
            not is_inherited_duplicate
            and is_finite_score(score)
            and (best is None or float(score) < best)
        ):
            best = float(score)
            return True
        return False

    # Earlier bouts only move the global bar; they never seed the streak.
    for stage in stages:
        if stage_bout_index(stage) >= bout_index:
            continue
        for trial in stage.get("trials", []):
            absorbs(trial)

    streak = 0
    for stage in stages:
        if stage_bout_index(stage) != bout_index:
            continue
        for trial in stage.get("trials", []):
            if absorbs(trial):
                streak = 0
            else:
                streak += 1
    return best, streak


# ---------- search-space feasibility clamp ----------

SPACE_CLAMP_HEADROOM = 0.85
SPACE_CLAMP_MAX_PROBES = 40
SPACE_CLAMP_MAX_PULL_ROUNDS = 6
SPACE_CLAMP_SCHEMA_VERSION = 2
# 3: feasibility additionally requires the probe's own envelope receipt (a
# half-context probe no longer certifies a full-context run). Bumping this
# intentionally invalidates clamps cached under the weaker oracle.
SPACE_CLAMP_ALGORITHM_VERSION = 3


class _ClampProbeBudgetExceeded(Exception):
    """Exploration probes exhausted; fall back to the known-feasible base box."""


def is_config_infeasible_error(exc: BaseException) -> bool:
    """True when an evaluation failure is a property of THIS config and may
    feed sampler constraints: evaluation timeouts (the config cannot finish
    within the contract's runtime limit) and out-of-memory errors. Anything
    else — infrastructure flakes, candidate-wide bugs, malformed subprocess
    results — is not a parameter-region signal and must not teach the sampler
    a false feasibility boundary."""
    if isinstance(exc, TimeoutError):
        return True
    return "out of memory" in str(exc).lower()


def _env_total_vram_mb(ref_path: Any) -> float | None:
    """Total VRAM (MB) recorded by the run's environment preflight, if any."""
    cfg = find_framework_cfg(ref_path)
    if cfg is None:
        return None
    env_path = cfg.parent / "environment_preflight.json"
    try:
        data = json.loads(env_path.read_text())
        value = float(data["hook_result"]["total_vram_mb"])
        return value if value > 0 else None
    except (OSError, KeyError, TypeError, ValueError):
        return None


def params_within_search_space(params: dict, search_space: dict) -> bool:
    """True iff every space key is present in params and within bounds/options."""
    for key, entry in search_space.items():
        if key not in params:
            return False
        value = params[key]
        kind = entry[0]
        if kind in ("int", "float"):
            try:
                v = float(value)
            except (TypeError, ValueError):
                return False
            if not (float(entry[1]) <= v <= float(entry[2])):
                return False
        elif kind == "categorical":
            if value not in entry[1]:
                return False
    return True


def split_configs_by_space(configs: list[dict], search_space: dict) -> tuple[list[dict], list[dict]]:
    """Partition param dicts into (inside, outside) the search space.

    Used for deferred warm configs after a space clamp: outside configs are
    skipped (never attempted, no budget, no patience effect — the clamp marked
    that region infeasible) and must be accounted for via the tuners'
    deferred_skipped_outside_space receipts.
    """
    inside = [p for p in configs if params_within_search_space(p, search_space)]
    outside = [p for p in configs if not params_within_search_space(p, search_space)]
    return inside, outside


def clamp_search_space_to_preflight(
    search_space: dict,
    base_params: dict | None,
    candidate_path: Any,
    report_path: Path,
    *,
    headroom: float = SPACE_CLAMP_HEADROOM,
    admission_check: Callable[[], None] | None = None,
    phase_time_limit_seconds: float | Callable[[], float] | None = None,
    expected_execution_revision: dict | None = None,
) -> dict:
    """Shrink numeric SEARCH_SPACE upper bounds until the box's upper corner is
    preflight-feasible within the VRAM headroom.

    Which dims drive memory (and how) is task-specific and unknowable to the
    tuner, so instead of a memory model the task's own no-score preflight is
    used as the ground-truth oracle: probe the all-upper corner once (a sane
    proposed space costs exactly one probe), and only when it fails attribute
    per dim (solo-max probes) and binary-search each failing dim's largest
    feasible bound. Probes never consume objective budget; every probe is
    recorded as a preflight attempt with source 'space_clamp'.

    A probe counts as feasible only when it passes, reports peak_vram_mb within
    `headroom` of the environment's total VRAM, AND does not report
    `envelope_covers_worst_case: false` — a bare pass is not enough, because a
    full training run peaks higher than a single step, and a candidate that
    ramps its training shape mid-run peaks higher than its own first step (the
    observed pass-preflight-then-OOM pattern). Without peak telemetry the
    predicate degrades to pass/fail only.

    Probes prefer the task's `evaluation.resource_probe_fn`, which measures the
    worst-case training shape, and fall back to `preflight_fn` when the task
    declares none.

    No-op (returns search_space unchanged) when base params, task preflight,
    or VRAM telemetry are unavailable, or when the base point itself exceeds
    the headroom — the candidate then lives at the memory edge by design and
    crashes are handled by the normal failure path. The outcome is cached in
    tune_report under 'search_space_clamp' keyed by an input hash, so tuner
    resumes and later stages do not re-probe.
    """
    if not base_params or _configured_preflight_name(Path(candidate_path)) is None:
        return search_space
    total_vram = _env_total_vram_mb(candidate_path)
    if total_vram is None:
        return search_space
    # Prefer the worst-case envelope oracle; `probe()` falls back to the
    # correctness preflight when the task declares no resource probe.
    probe_mode = (
        "resource"
        if _configured_resource_probe_name(Path(candidate_path)) is not None
        else "preflight"
    )

    clampable = [
        key
        for key, entry in search_space.items()
        if entry[0] in ("int", "float")
        and key in base_params
        and float(entry[2]) > float(base_params[key])
    ]
    if not clampable:
        return search_space

    input_hash = hashlib.sha256(
        json.dumps(
            {
                "algorithm_version": SPACE_CLAMP_ALGORITHM_VERSION,
                "space": search_space,
                "base": base_params,
                "total_vram": total_vram,
                "headroom": headroom,
                "max_probes": SPACE_CLAMP_MAX_PROBES,
                "max_pull_rounds": SPACE_CLAMP_MAX_PULL_ROUNDS,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    cached = read_tune_report(report_path).get("search_space_clamp")
    cache_outcome = cached.get("outcome") if isinstance(cached, dict) else None
    cache_corner_feasible = (
        cached.get("corner_feasible") if isinstance(cached, dict) else None
    )
    cache_has_valid_outcome = (
        cache_outcome
        in {"already_feasible", "clamped_feasible", "collapsed_to_base"}
        and cache_corner_feasible is True
    ) or (
        cache_outcome == "base_infeasible_noop"
        and cache_corner_feasible is False
        and cached.get("clamped_search_space")
        == {k: list(v) for k, v in search_space.items()}
    )
    if (
        isinstance(cached, dict)
        and cached.get("schema_version") == SPACE_CLAMP_SCHEMA_VERSION
        and cached.get("algorithm_version") == SPACE_CLAMP_ALGORITHM_VERSION
        and cached.get("input_sha256") == input_hash
        and cache_has_valid_outcome
        and isinstance(cached.get("clamped_search_space"), dict)
        and set(cached["clamped_search_space"]) == set(search_space)
    ):
        return {k: tuple(v) for k, v in cached["clamped_search_space"].items()}

    probes: list[dict] = []

    def probe(
        params: dict, label: str, *, essential: bool = False
    ) -> tuple[bool, float | None]:
        """One preflight probe. Returns (feasible, peak_vram_mb or None).

        Exploration probes are capped at SPACE_CLAMP_MAX_PROBES so a patholog-
        ical space cannot turn the clamp into unbounded wall-clock work; the
        cap never applies to the collapse path's essential confirmation probes.
        Every probe records its elapsed seconds in the receipt.
        """
        if not essential and len(probes) >= SPACE_CLAMP_MAX_PROBES:
            raise _ClampProbeBudgetExceeded
        if admission_check is not None:
            admission_check()
        started = time.time()
        elapsed = lambda: round(time.time() - started, 1)  # noqa: E731
        try:
            preflight_kwargs = {}
            if phase_time_limit_seconds is not None:
                preflight_kwargs["phase_time_limit_seconds"] = (
                    phase_time_limit_seconds
                )
            if expected_execution_revision is not None:
                preflight_kwargs["expected_execution_revision"] = (
                    expected_execution_revision
                )
            result = timed_preflight(
                params,
                candidate_path,
                probe_mode=probe_mode,
                **preflight_kwargs,
            )
        except DeepTuneTimeExhausted:
            raise
        except Exception as exc:
            append_preflight_attempt(
                report_path,
                source="space_clamp",
                params=params,
                status="failed",
                failure={"error": f"{type(exc).__name__}: {exc}"[:500]},
            )
            probes.append(
                {"label": label, "feasible": False, "elapsed_seconds": elapsed()}
            )
            if admission_check is not None:
                admission_check()
            return False, None
        append_preflight_attempt(
            report_path,
            source="space_clamp",
            params=params,
            status="ok",
            result=result or {"status": "ok"},
        )
        peak = (result or {}).get("peak_vram_mb")
        # A probe whose own telemetry says it did not reach the worst-case
        # training shape cannot certify feasibility, however low its peak: run
        # 0802-sonnet-ex125-1/007 passed at 44.7 GB on a half-context first step
        # and then OOMed at 74.9 GB once its curriculum reached full context.
        # `None` (task reports no envelope) keeps the historical pass/fail
        # behavior rather than blocking every task that lacks the field.
        covers_worst_case = (result or {}).get("envelope_covers_worst_case")
        feasible = (
            covers_worst_case is not False
            and (peak is None or float(peak) <= headroom * total_vram)
        )
        probes.append({
            "label": label,
            "feasible": feasible,
            "peak_vram_mb": peak,
            "envelope_covers_worst_case": covers_worst_case,
            "probe_seq_len": (result or {}).get("probe_seq_len"),
            "elapsed_seconds": elapsed(),
        })
        if admission_check is not None:
            admission_check()
        return feasible, (float(peak) if peak is not None else None)

    def at(dim_values: dict) -> dict:
        point = dict(base_params)
        point.update(dim_values)
        return point

    upper = {key: search_space[key][2] for key in clampable}
    clamped = {k: tuple(v) for k, v in search_space.items()}

    def collapse(dim_keys) -> None:
        """Pull the given dims' upper bounds all the way to the base value."""
        for key in dim_keys:
            entry = clamped[key]
            base_v = float(base_params[key])
            new_hi = max(base_v, float(entry[1]))  # never below the lower bound
            clamped[key] = tuple(
                (int(new_hi) if entry[0] == "int" else new_hi) if i == 2 else v
                for i, v in enumerate(entry)
            )

    def corner_point() -> dict:
        return at({key: clamped[key][2] for key in clampable})

    outcome = "unclamped"
    corner_feasible = False
    base_ok: bool | None = None
    try:
        corner_feasible, _ = probe(at(upper), "corner")
        if corner_feasible:
            outcome = "already_feasible"  # the common sane case: one probe
        else:
            base_ok, base_peak = probe(at({}), "base", essential=True)
            if not base_ok:
                # The incumbent itself exceeds the headroom: no room to clamp
                # into. Explicit unsuccessful outcome; the ORIGINAL space is
                # returned and crashes are handled by the normal failure path.
                outcome = "base_infeasible_noop"
            else:
                solo_peaks: dict[str, float | None] = {}
                failing = []
                for key in clampable:
                    ok, peak = probe(at({key: upper[key]}), f"solo_max:{key}")
                    solo_peaks[key] = peak
                    if not ok:
                        failing.append(key)
                for key in failing:
                    entry = search_space[key]
                    kind, log = entry[0], len(entry) >= 4 and entry[3] == "log"
                    lo_ok, hi_bad = float(base_params[key]), float(entry[2])
                    for _ in range(4):
                        if kind == "int":
                            if hi_bad - lo_ok <= 1:
                                break
                            mid = (lo_ok + hi_bad) // 2
                        elif log:
                            mid = math.sqrt(lo_ok * hi_bad)
                        else:
                            mid = (lo_ok + hi_bad) / 2
                        ok, _ = probe(at({key: mid}), f"bisect:{key}")
                        if ok:
                            lo_ok = mid
                        else:
                            hi_bad = mid
                    new_hi = int(lo_ok) if kind == "int" else lo_ok
                    if new_hi != entry[2]:
                        clamped[key] = tuple(
                            new_hi if i == 2 else v for i, v in enumerate(entry)
                        )
                corner_feasible, _ = probe(corner_point(), "corner_after_clamp")
                # Residual joint interaction (e.g. batch x depth): solo probes
                # cannot attribute it, but they DID measure each dim's marginal
                # peak. Pull the measured memory contributors (failed solos, or
                # solo peak clearly above base) toward base jointly;
                # memory-neutral dims (LRs) stay put.
                contributors = [
                    key
                    for key in clampable
                    if key in failing
                    or (
                        solo_peaks.get(key) is not None
                        and base_peak is not None
                        and solo_peaks[key] - base_peak > 0.005 * total_vram
                    )
                ]
                for _ in range(SPACE_CLAMP_MAX_PULL_ROUNDS):
                    if corner_feasible or not contributors:
                        break
                    for key in contributors:
                        entry = clamped[key]
                        pulled = float(base_params[key]) + 0.8 * (
                            float(entry[2]) - float(base_params[key])
                        )
                        new_hi = (
                            max(int(pulled), int(base_params[key]))
                            if entry[0] == "int"
                            else pulled
                        )
                        clamped[key] = tuple(
                            new_hi if i == 2 else v for i, v in enumerate(entry)
                        )
                    corner_feasible, _ = probe(corner_point(), "corner_after_clamp")
                if corner_feasible:
                    outcome = "clamped_feasible"
                else:
                    # Never return a box whose corner is known infeasible:
                    # collapse contributors to base, and if even that corner
                    # fails, collapse every clampable dim — that corner IS the
                    # base point, which passed its probe above.
                    collapse(contributors)
                    corner_feasible, _ = probe(
                        corner_point(), "corner_collapsed", essential=True
                    )
                    if not corner_feasible:
                        collapse(clampable)
                        corner_feasible = True
                    outcome = "collapsed_to_base"
    except _ClampProbeBudgetExceeded:
        # Probe governance: stop exploring and fall back to the box whose
        # corner is the base point — feasible by direct evidence, when the
        # base probe ran and passed.
        if base_ok:
            collapse(clampable)
            corner_feasible = True
            outcome = "collapsed_to_base"

    report = read_tune_report(report_path)
    report["search_space_clamp"] = {
        "schema_version": SPACE_CLAMP_SCHEMA_VERSION,
        "algorithm_version": SPACE_CLAMP_ALGORITHM_VERSION,
        "input_sha256": input_hash,
        "outcome": outcome,
        "total_vram_mb": total_vram,
        "headroom": headroom,
        "corner_feasible": corner_feasible,
        "probes": probes,
        "original_search_space": {k: list(v) for k, v in search_space.items()},
        "clamped_search_space": {k: list(v) for k, v in clamped.items()},
    }
    write_tune_report(report_path, report)
    return clamped
