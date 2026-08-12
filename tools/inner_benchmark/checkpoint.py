"""Frozen-checkpoint data contract for the inner-tuner benchmark (PLAN §七).

``checkpoint.json`` (schema_version=2, benchmark-owned; intentionally
incompatible with the old benchmark remnant format) is written once by the
freezing tool and treated as authoritative by the runner: history scores are
already re-measured on the local machine at freeze time, so crash rows carry
``score: null`` with ``status: "crash"``.

Layout on disk::

    <checkpoint_dir>/
        checkpoint.json
        <candidate_relpath>/   # frozen candidate dir (train.py + prepare.py),
                               # or a direct train.py path

Schema (required unless noted)::

    {
      "schema_version": 2,
      "checkpoint_id": "string",
      "regime": "first" | "continuation" | "deep",
      "stratum": "first" | "cont_improved" | "cont_not_improved" | "deep",
      "source": {"...": "opaque provenance dict"},        # default {}
      "candidate_relpath": "candidate",
      "task": {"score_fn": "evaluate_config",
               "preflight_fn": "preflight_config",
               "per_runtime_limit": 900,                   # number or null
               "project": "tasks/<task>"},                 # optional; see below
      "incumbent": {"params": {...}, "score": 1.23},
      "incumbent_is_inherited_control": false,
      "history": [{"params": {...}, "score": 1.23 | null,
                   "status": "ok" | "crash",
                   "origin": "phase_a" | "bout_0" | ...,   # optional
                   "role": "inherited_control" | null}],   # optional
      "deferred_configs": [{"params": {...}}],             # default []
      "extra": {}                                          # default {}
    }

regime/stratum consistency is validated: first<->first,
continuation<->cont_improved|cont_not_improved, deep<->deep.

``deferred_configs`` is reserved for the Current arm (PLAN §6.0 / §七); other
arms must not read it.

``task.project`` (repo-root-relative uv project path, e.g.
``tasks/autoresearch-baseline``) selects the interpreter that runs the
evaluation/preflight subprocesses: cells execute in the repo root env, while
the objective subprocess runs under the task project (production's
``uv --project tasks/<task> run python ...`` split, driver/loops/common.py).
Absent (test fixtures only) -> the subprocess uses sys.executable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

SCHEMA_VERSION = 2
CHECKPOINT_FILENAME = "checkpoint.json"

REGIMES = ("first", "continuation", "deep")
STRATA = ("first", "cont_improved", "cont_not_improved", "deep")
_REGIME_STRATA = {
    "first": ("first",),
    "continuation": ("cont_improved", "cont_not_improved"),
    "deep": ("deep",),
}


@dataclass(frozen=True)
class TaskSpec:
    """Task evaluation wiring: fn names resolved in the candidate's prepare.py.

    ``project`` is the repo-root-relative uv project path the evaluation
    subprocess runs under (None -> sys.executable; see module docstring).
    """

    score_fn: str
    preflight_fn: str
    per_runtime_limit: float | None
    project: str | None = None


@dataclass(frozen=True)
class Incumbent:
    """Checkpoint incumbent: an executed config with a finite re-measured score."""

    params: dict
    score: float


@dataclass(frozen=True)
class HistoryRow:
    """One executed (re-measured) history trial. Crash rows carry score=None."""

    params: dict
    score: float | None
    status: str  # "ok" | "crash"
    origin: str | None = None  # "phase_a" | "bout_0" | ...
    role: str | None = None  # "inherited_control" | None


@dataclass(frozen=True)
class Checkpoint:
    """Parsed, validated frozen checkpoint. ``candidate_path`` is the resolved
    train.py path (checkpoint_dir / candidate_relpath, dir resolved to its
    train.py)."""

    checkpoint_id: str
    regime: str
    stratum: str
    source: dict
    checkpoint_dir: Path
    candidate_relpath: str
    candidate_path: Path
    task: TaskSpec
    incumbent: Incumbent
    incumbent_is_inherited_control: bool
    history: tuple[HistoryRow, ...] = ()
    deferred_configs: tuple[dict, ...] = ()  # params dicts; Current arm only
    extra: dict = field(default_factory=dict)

    def finite_unique_history(self, contract) -> list[tuple[dict, float]]:
        """Finite, unique, executed (params, score) pairs — the WARMUP-countable
        observations (PLAN §5.1).

        The incumbent is an executed finite config and counts (listed first);
        history rows follow in recorded order. Dedupe by
        ``contract.params_identity`` (production cast + identity) keeping the
        first occurrence; crash rows are excluded.
        """
        seen: set[str] = set()
        out: list[tuple[dict, float]] = []
        entries = [(self.incumbent.params, self.incumbent.score)]
        entries.extend((row.params, row.score) for row in self.history)
        for params, score in entries:
            if score is None or not math.isfinite(score):
                continue
            identity = contract.params_identity(params)
            if identity in seen:
                continue
            seen.add(identity)
            out.append((dict(params), float(score)))
        return out


def load_checkpoint(checkpoint_dir) -> Checkpoint:
    """Read and validate ``<checkpoint_dir>/checkpoint.json`` (schema_version=2).

    Raises FileNotFoundError for a missing checkpoint.json or candidate, and
    ValueError for any schema/validity violation (wrong version, inconsistent
    regime/stratum, non-finite incumbent score, malformed history rows, ...).
    """
    directory = Path(checkpoint_dir)
    path = directory / CHECKPOINT_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level value must be an object")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema_version {version!r}; expected "
            f"{SCHEMA_VERSION} (benchmark-owned, intentionally incompatible "
            "with the old remnant format)"
        )

    checkpoint_id = _require(data, "checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError(f"{path}: checkpoint_id must be a non-empty string")

    regime = _require(data, "regime")
    stratum = _require(data, "stratum")
    if regime not in REGIMES:
        raise ValueError(f"{path}: regime must be one of {REGIMES}, got {regime!r}")
    if stratum not in STRATA:
        raise ValueError(f"{path}: stratum must be one of {STRATA}, got {stratum!r}")
    if stratum not in _REGIME_STRATA[regime]:
        raise ValueError(
            f"{path}: stratum {stratum!r} inconsistent with regime {regime!r} "
            f"(expected one of {_REGIME_STRATA[regime]})"
        )

    source = data.get("source", {})
    if not isinstance(source, dict):
        raise ValueError(f"{path}: source must be an object")

    candidate_relpath = _require(data, "candidate_relpath")
    if not isinstance(candidate_relpath, str) or not candidate_relpath:
        raise ValueError(f"{path}: candidate_relpath must be a non-empty string")
    raw = directory / candidate_relpath
    candidate_path = raw / "train.py" if raw.is_dir() else raw
    if not candidate_path.exists():
        raise FileNotFoundError(f"{path}: candidate not found: {candidate_path}")

    task = _task_spec(path, _require(data, "task"))
    incumbent = _incumbent(path, _require(data, "incumbent"))

    inherited = _require(data, "incumbent_is_inherited_control")
    if not isinstance(inherited, bool):
        raise ValueError(
            f"{path}: incumbent_is_inherited_control must be a boolean"
        )

    history_raw = _require(data, "history")
    if not isinstance(history_raw, list):
        raise ValueError(f"{path}: history must be a list")
    history = tuple(_history_row(path, index, row) for index, row in enumerate(history_raw))

    deferred_raw = data.get("deferred_configs", [])
    if not isinstance(deferred_raw, list):
        raise ValueError(f"{path}: deferred_configs must be a list")
    deferred = []
    for index, entry in enumerate(deferred_raw):
        if not isinstance(entry, dict) or not isinstance(entry.get("params"), dict):
            raise ValueError(
                f"{path}: deferred_configs[{index}] must be an object with a params dict"
            )
        deferred.append(dict(entry["params"]))

    extra = data.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError(f"{path}: extra must be an object")

    return Checkpoint(
        checkpoint_id=checkpoint_id,
        regime=regime,
        stratum=stratum,
        source=source,
        checkpoint_dir=directory,
        candidate_relpath=candidate_relpath,
        candidate_path=candidate_path,
        task=task,
        incumbent=incumbent,
        incumbent_is_inherited_control=inherited,
        history=history,
        deferred_configs=tuple(deferred),
        extra=extra,
    )


def _require(data: dict, key: str):
    if key not in data:
        raise ValueError(f"checkpoint missing required key {key!r}")
    return data[key]


def _finite_number(value, ctx: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{ctx} must be a finite number, got {value!r}")
    return float(value)


def _task_spec(path: Path, raw) -> TaskSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: task must be an object")
    score_fn = raw.get("score_fn")
    preflight_fn = raw.get("preflight_fn")
    if not isinstance(score_fn, str) or not score_fn:
        raise ValueError(f"{path}: task.score_fn must be a non-empty string")
    if not isinstance(preflight_fn, str) or not preflight_fn:
        raise ValueError(f"{path}: task.preflight_fn must be a non-empty string")
    limit = raw.get("per_runtime_limit")
    if limit is not None:
        limit = _finite_number(limit, f"{path}: task.per_runtime_limit")
    project = raw.get("project")
    if project is not None and (not isinstance(project, str) or not project):
        raise ValueError(f"{path}: task.project must be a non-empty string or null")
    return TaskSpec(
        score_fn=score_fn,
        preflight_fn=preflight_fn,
        per_runtime_limit=limit,
        project=project,
    )


def _incumbent(path: Path, raw) -> Incumbent:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: incumbent must be an object")
    params = raw.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{path}: incumbent.params must be an object")
    score = _finite_number(raw.get("score"), f"{path}: incumbent.score")
    return Incumbent(params=dict(params), score=score)


def _history_row(path: Path, index: int, raw) -> HistoryRow:
    ctx = f"{path}: history[{index}]"
    if not isinstance(raw, dict):
        raise ValueError(f"{ctx} must be an object")
    params = raw.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{ctx}.params must be an object")
    status = raw.get("status")
    if status not in ("ok", "crash"):
        raise ValueError(f"{ctx}.status must be 'ok' or 'crash', got {status!r}")
    score = raw.get("score")
    if status == "ok":
        score = _finite_number(score, f"{ctx}.score")
    elif score is not None:
        raise ValueError(f"{ctx}: crash rows carry score null, got {score!r}")
    origin = raw.get("origin")
    role = raw.get("role")
    if origin is not None and not isinstance(origin, str):
        raise ValueError(f"{ctx}.origin must be a string or null")
    if role is not None and not isinstance(role, str):
        raise ValueError(f"{ctx}.role must be a string or null")
    return HistoryRow(
        params=dict(params), score=score, status=status, origin=origin, role=role
    )
