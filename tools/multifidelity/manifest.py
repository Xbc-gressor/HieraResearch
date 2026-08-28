"""Frozen manifests, digests, and mechanical checks for the multifidelity
benchmark harness.

One job = one frozen candidate + BASE_PARAMS + requested train seconds on one
explicit GPU. The job id binds every semantic field of the request, including
the GPU UUID (wall-clock budgets make the physical device part of the
observation) and (purpose, repeat_index), so calibration/adjudication repeats
of the same request get independent ids and are never skipped by job-level
resume.

Pool manifests freeze qualification tiers and candidates before any judge or
fidelity evaluation; once written they are immutable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
FIDELITY_SEMANTICS = "independent_compressed_schedule"
FULL_TRAIN_SECONDS = 300
MATRIX_FIDELITIES = (30, 60, 120, 300)
PURPOSES = ("matrix", "calibration", "adjudication")
EVALUATION_PATHS = ("adapter", "official")
TERMINAL_STATUSES = ("ok", "crash", "timeout", "budget_contract_failure")
QUALITY_TIERS = ("competitive", "borderline", "severe")
COMPETITIVE_FACTOR = 1.10
BORDERLINE_FACTOR = 1.25
POOL_SIZE = 6

REQUEST_FILENAME = "request.json"
PROVISIONAL_FILENAME = "result_provisional.json"
RESULT_FILENAME = "result.json"
STDOUT_FILENAME = "stdout.log"
STDERR_FILENAME = "stderr.log"

# Every field here is semantic: changing any of them yields a NEW job, and an
# old result must never be reused for it.
JOB_ID_FIELDS = (
    "experiment_id",
    "pool_id",
    "candidate_id",
    "candidate_execution_revision",
    "params_digest",
    "requested_train_seconds",
    "fidelity_semantics",
    "evaluation_path",
    "purpose",
    "repeat_index",
    "seed",
    "gpu_uuid",
    "task_artifact_digest",
)

# Summary keys the harness parses from a candidate's fixed final summary block.
SUMMARY_FLOAT_KEYS = (
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_params_M",
)
SUMMARY_INT_KEYS = ("num_steps", "depth")


class ManifestError(RuntimeError):
    """A manifest, request, or result violates the frozen harness contract."""


# ---------------------------------------------------------------------------
# canonical json / digests / atomic io
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def atomic_write_json(path: Path, doc: Any) -> None:
    """fsync + os.replace so a killed process never leaves a torn artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


# ---------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------


def params_digest(params: dict) -> str:
    if not isinstance(params, dict):
        raise ManifestError("params must be a mapping")
    return digest(params)


def candidate_fallback_id(
    source_prefix: Any,
    point_id: str,
    op: str,
    parents: list,
    candidate_execution_revision: dict,
    params_digest_value: str,
) -> str:
    """Canonical candidate id when no stable semantic candidate id exists."""
    return "cand-" + digest(
        {
            "source_prefix": source_prefix,
            "point_id": point_id,
            "op": op,
            "parents": [str(p) for p in parents],
            "candidate_execution_revision": candidate_execution_revision,
            "params_digest": params_digest_value,
        }
    )[len("sha256:"):][:20]


def job_id_for(request: dict) -> str:
    """Job id over the request's semantic fields only.

    gpu_id is deliberately absent (display/index fact); gpu_uuid is the
    binding. Any missing semantic field is an error, never a default.
    """
    material = {}
    for field in JOB_ID_FIELDS:
        if field not in request:
            raise ManifestError(f"job request missing semantic field {field!r}")
        material[field] = request[field]
    if material["fidelity_semantics"] != FIDELITY_SEMANTICS:
        raise ManifestError(
            f"unknown fidelity_semantics {material['fidelity_semantics']!r}"
        )
    if material["purpose"] not in PURPOSES:
        raise ManifestError(f"unknown purpose {material['purpose']!r}")
    if material["evaluation_path"] not in EVALUATION_PATHS:
        raise ManifestError(
            f"unknown evaluation_path {material['evaluation_path']!r}"
        )
    if not isinstance(material["repeat_index"], int) or isinstance(
        material["repeat_index"], bool
    ):
        raise ManifestError("repeat_index must be an integer")
    return "job-" + digest(material)[len("sha256:"):][:24]


def build_request(
    *,
    experiment_id: str,
    pool_id: str,
    candidate_id: str,
    candidate_path: str,
    candidate_execution_revision: dict,
    params: dict,
    requested_train_seconds: int,
    purpose: str,
    repeat_index: int,
    seed: int,
    gpu_id: int,
    gpu_uuid: str,
    task_artifact_digest: str,
    evaluation_path: str = "adapter",
) -> dict:
    request = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "pool_id": pool_id,
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_path),
        "candidate_execution_revision": candidate_execution_revision,
        "params": params,
        "params_digest": params_digest(params),
        "requested_train_seconds": int(requested_train_seconds),
        "fidelity_semantics": FIDELITY_SEMANTICS,
        "evaluation_path": evaluation_path,
        "purpose": purpose,
        "repeat_index": int(repeat_index),
        "seed": int(seed),
        "gpu_id": int(gpu_id),
        "gpu_uuid": gpu_uuid,
        "task_artifact_digest": task_artifact_digest,
    }
    request["job_id"] = job_id_for(request)
    return request


def validate_request(request: dict) -> None:
    """A stored request must recompute to its own job id and params digest."""
    if request.get("job_id") != job_id_for(request):
        raise ManifestError("request job_id does not recompute from its fields")
    if request.get("params_digest") != params_digest(request.get("params")):
        raise ManifestError("request params_digest does not match params")


# ---------------------------------------------------------------------------
# task artifact binding
# ---------------------------------------------------------------------------


def task_artifact_binding(
    task_dir: Path,
    runner_commit: str,
    cache_dir: Path | None = None,
) -> dict:
    """Bind the fixed evaluation surface: task-root prepare.py + task.toml,
    tokenizer artifacts, a name+size manifest of the data shards, and the
    runner commit. Shard bytes are not hashed (tens of GB); the pinned name
    and size manifest is the mechanical identity used here."""
    task_dir = Path(task_dir)
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "autoresearch"
    components: dict[str, Any] = {
        "prepare_py": sha256_file(task_dir / "prepare.py"),
        "task_toml": sha256_file(task_dir / "task.toml"),
        "runner_commit": runner_commit,
    }
    tokenizer_dir = cache_dir / "tokenizer"
    components["tokenizer_pkl"] = sha256_file(tokenizer_dir / "tokenizer.pkl")
    components["token_bytes_pt"] = sha256_file(tokenizer_dir / "token_bytes.pt")
    data_dir = cache_dir / "data"
    shards = sorted(
        p.name for p in data_dir.glob("*.parquet") if not p.name.endswith(".tmp")
    )
    components["data_manifest"] = digest(
        [[name, (data_dir / name).stat().st_size] for name in shards]
    )
    return {
        "components": components,
        "task_artifact_digest": digest(components),
    }


# ---------------------------------------------------------------------------
# freeze checks
# ---------------------------------------------------------------------------


def reads_env_train_budget_seconds(train_path: Path) -> bool:
    """Whether the pinned train.py reads `.train_budget_seconds` anywhere.

    A candidate that hardcodes its schedule to 300s runs its low-fidelity jobs
    as truncation rather than a compressed schedule. It still runs; the
    analyzer annotates the semantic difference per pool.
    """
    tree = ast.parse(Path(train_path).read_text(errors="replace"))
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == "train_budget_seconds"
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(tree)
    )


def parse_stdout_summary(text: str) -> dict:
    """Parse the candidate's fixed final summary block from full stdout.

    Returns whichever known keys parse; last occurrence wins. Missing keys are
    simply absent — the caller decides whether that is a contract failure.
    """
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        key = key.strip()
        raw = raw.strip()
        if key in SUMMARY_FLOAT_KEYS:
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                out[key] = value
        elif key in SUMMARY_INT_KEYS:
            try:
                out[key] = int(raw)
            except ValueError:
                continue
    return out


def summary_has_training_seconds(text: str) -> bool:
    return "training_seconds" in parse_stdout_summary(text)


def budget_tolerance_seconds(requested_train_seconds: float) -> float:
    """Frozen candidate-independent tolerance: max(2 s, 15% of the request)."""
    return max(2.0, 0.15 * float(requested_train_seconds))


def budget_compliant(requested_train_seconds: float, completed: Any) -> bool:
    if not is_finite_number(completed):
        return False
    return abs(float(completed) - float(requested_train_seconds)) <= (
        budget_tolerance_seconds(requested_train_seconds)
    )


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


def is_terminal_result(result: Any) -> bool:
    return (
        isinstance(result, dict)
        and result.get("status") in TERMINAL_STATUSES
    )


def validate_result_against_request(result: dict, request: dict) -> list[str]:
    """Whether a stored result is THE terminal observation for this request.

    Used by job-level resume: any mismatch rejects the result rather than
    silently reusing it for a different execution.
    """
    errors = []
    if not is_terminal_result(result):
        errors.append(f"result status {result.get('status')!r} is not terminal")
    for field in (
        "job_id",
        "params_digest",
        "requested_train_seconds",
        "task_artifact_digest",
        "candidate_execution_revision",
        "evaluation_path",
        "purpose",
        "repeat_index",
    ):
        if result.get(field) != request.get(field):
            errors.append(
                f"result {field} {result.get(field)!r} != request "
                f"{request.get(field)!r}"
            )
    if result.get("status") == "ok" and not is_finite_number(
        result.get("score")
    ):
        errors.append(f"ok result has non-finite score {result.get('score')!r}")
    return errors


# ---------------------------------------------------------------------------
# quality tiers
# ---------------------------------------------------------------------------


def quality_tier(qualification_score: Any, anchor_score: float) -> str:
    """Frozen tiering: competitive <= 1.10*anchor < borderline <= 1.25*anchor;
    crash/non-finite or worse is severe. Lower is better throughout."""
    if not is_finite_number(anchor_score):
        raise ManifestError("anchor score must be finite")
    if not is_finite_number(qualification_score):
        return "severe"
    q = float(qualification_score)
    if q <= COMPETITIVE_FACTOR * anchor_score:
        return "competitive"
    if q <= BORDERLINE_FACTOR * anchor_score:
        return "borderline"
    return "severe"


# ---------------------------------------------------------------------------
# manifest shape checks (light: only what downstream tools read)
# ---------------------------------------------------------------------------


def validate_experiment(doc: dict) -> None:
    for field in ("experiment_id", "task", "fidelities", "hardware"):
        if field not in doc:
            raise ManifestError(f"experiment manifest missing {field!r}")
    task = doc["task"]
    if task.get("fidelity_semantics") != FIDELITY_SEMANTICS:
        raise ManifestError("experiment task.fidelity_semantics mismatch")
    if task.get("full_train_seconds") != FULL_TRAIN_SECONDS:
        raise ManifestError("experiment task.full_train_seconds must be 300")
    if not task.get("task_artifact_digest"):
        raise ManifestError("experiment task.task_artifact_digest missing")
    devices = doc["hardware"].get("devices") or []
    seen_uuids = set()
    for device in devices:
        if "gpu_id" not in device or not device.get("gpu_uuid"):
            raise ManifestError("hardware.devices entries need gpu_id+gpu_uuid")
        if device["gpu_uuid"] in seen_uuids:
            raise ManifestError(f"duplicate gpu_uuid {device['gpu_uuid']!r}")
        seen_uuids.add(device["gpu_uuid"])
    if not devices:
        raise ManifestError("experiment hardware.devices is empty")


def validate_pool_manifest(doc: dict) -> None:
    for field in ("pool_id", "cohort", "anchor", "candidates"):
        if field not in doc:
            raise ManifestError(f"pool manifest missing {field!r}")
    if doc["cohort"] not in ("natural", "competitive"):
        raise ManifestError(f"unknown cohort {doc['cohort']!r}")
    if not is_finite_number(doc["anchor"].get("qualification_score")):
        raise ManifestError("pool anchor.qualification_score must be finite")
    seen = set()
    for candidate in doc["candidates"]:
        for field in (
            "candidate_id",
            "coverage_rank",
            "candidate_path",
            "candidate_execution_revision",
            "params",
            "params_digest",
            "quality",
            "freeze_checks",
        ):
            if field not in candidate:
                raise ManifestError(
                    f"pool candidate missing {field!r} "
                    f"({candidate.get('candidate_id')})"
                )
        if candidate["quality"].get("tier") not in QUALITY_TIERS:
            raise ManifestError(
                f"unknown quality tier for {candidate['candidate_id']}"
            )
        if candidate["candidate_id"] in seen:
            raise ManifestError(
                f"duplicate candidate_id {candidate['candidate_id']}"
            )
        seen.add(candidate["candidate_id"])


def freeze_immutable(path: Path, doc: dict) -> bool:
    """Write a frozen artifact. Identical re-write is a no-op (returns False);
    any divergence from an existing file is an error, never an overwrite."""
    path = Path(path)
    if path.exists():
        existing = load_json(path)
        if canonical_json(existing) == canonical_json(doc):
            return False
        raise ManifestError(
            f"{path} is frozen and differs from the new content; refusing to "
            "overwrite"
        )
    atomic_write_json(path, doc)
    return True
