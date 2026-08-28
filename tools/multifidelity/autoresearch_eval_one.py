"""Run ONE multifidelity job in a fresh subprocess (autoresearch-baseline).

Usage (invoked by run_matrix.py with stdout/stderr already redirected to
<job_dir>/stdout.log and <job_dir>/stderr.log):

    python autoresearch_eval_one.py <job_dir>

Reads <job_dir>/request.json and writes <job_dir>/result_provisional.json via
fsync + os.replace. The parent finalizes result.json with parent-side facts
(elapsed_accelerator_seconds, GPU assignment, log digests).

The adapter path changes exactly one field of the fixed evaluation surface:
`env.train_budget_seconds` (asserted to be 300 before the override). The
candidate keeps its own budget-normalized schedule and the unchanged full
evaluator. The official path calls the task's `evaluate_config` verbatim and
only accepts a 300-second request (used by the full-fidelity equivalence
gate). No checkpoints are read or written on either path.

Statuses written here: ok / crash / budget_contract_failure. Timeout is a
parent-side classification (the parent kills the process group).
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "tuners"))

import manifest  # noqa: E402


def _fail(job_dir: Path, request: dict, error: str, elapsed: float) -> None:
    _write_provisional(
        job_dir, request, status="crash", score=None, summary={},
        error=error, elapsed=elapsed,
    )


def _write_provisional(
    job_dir: Path,
    request: dict,
    *,
    status: str,
    score: float | None,
    summary: dict,
    error: str | None,
    elapsed: float,
) -> None:
    result = {
        "schema_version": manifest.SCHEMA_VERSION,
        "job_id": request["job_id"],
        "status": status,
        "score": score,
        "metric": "val_bpb",
        "requested_train_seconds": request["requested_train_seconds"],
        "completed_train_seconds": summary.get("training_seconds"),
        "num_steps": summary.get("num_steps"),
        "child_elapsed_seconds": elapsed,
        "candidate_execution_revision": request["candidate_execution_revision"],
        "params_digest": request["params_digest"],
        "task_artifact_digest": request["task_artifact_digest"],
        "evaluation_path": request["evaluation_path"],
        "purpose": request["purpose"],
        "repeat_index": request["repeat_index"],
        "seed": request["seed"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "summary": summary,
        "error": error,
    }
    manifest.atomic_write_json(
        job_dir / manifest.PROVISIONAL_FILENAME, result
    )


def main() -> int:
    job_dir = Path(sys.argv[1])
    request = manifest.load_json(job_dir / manifest.REQUEST_FILENAME)
    manifest.validate_request(request)
    started = time.monotonic()

    candidate_path = Path(request["candidate_path"])
    prepare_path = candidate_path.parent / "prepare.py"
    error = None
    score = None
    try:
        candidate_bytes = candidate_path.read_bytes()
        prepare_bytes = prepare_path.read_bytes()

        from _common import load_candidate_modules, resolve_score_fn

        train_module, prepare_module = load_candidate_modules(
            candidate_path,
            expected_execution_revision=request["candidate_execution_revision"],
        )

        requested = request["requested_train_seconds"]
        if request["evaluation_path"] == "official":
            if requested != manifest.FULL_TRAIN_SECONDS:
                raise RuntimeError(
                    "official evaluation path only accepts "
                    f"{manifest.FULL_TRAIN_SECONDS}s requests, got {requested}"
                )
            evaluate = resolve_score_fn(prepare_module, candidate_path)
            score = float(evaluate(train_module.make_model, request["params"]))
        else:
            env = prepare_module.PretrainEnv()
            if int(env.train_budget_seconds) != manifest.FULL_TRAIN_SECONDS:
                raise RuntimeError(
                    "evaluation surface default train_budget_seconds is "
                    f"{env.train_budget_seconds!r}, expected "
                    f"{manifest.FULL_TRAIN_SECONDS}"
                )
            if int(env.seed) != int(request["seed"]):
                raise RuntimeError(
                    f"request seed {request['seed']} != fixed env seed "
                    f"{env.seed}"
                )
            env.train_budget_seconds = requested
            trainer = train_module.make_model(env, request["params"])
            score = float(trainer.run())
        if not manifest.is_finite_number(score):
            raise ValueError(f"trainer returned non-finite score: {score!r}")

        if (
            candidate_path.read_bytes() != candidate_bytes
            or prepare_path.read_bytes() != prepare_bytes
        ):
            raise RuntimeError(
                "candidate or evaluator bytes changed during the run"
            )
    except BaseException as exc:  # classified as crash, never a fake score
        traceback.print_exc(file=sys.stderr)
        error = f"{type(exc).__name__}: {exc}"
        score = None

    elapsed = time.monotonic() - started
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_path = job_dir / manifest.STDOUT_FILENAME
    try:
        stdout_text = stdout_path.read_text(errors="replace")
    except OSError:
        stdout_text = ""
    summary = manifest.parse_stdout_summary(stdout_text)

    if error is not None:
        _fail(job_dir, request, error, elapsed)
        return 0
    if not manifest.budget_compliant(
        request["requested_train_seconds"], summary.get("training_seconds")
    ):
        _write_provisional(
            job_dir, request, status="budget_contract_failure", score=None,
            summary=summary,
            error=(
                "summary training_seconds "
                f"{summary.get('training_seconds')!r} outside tolerance "
                f"±{manifest.budget_tolerance_seconds(request['requested_train_seconds']):g}s "
                f"of {request['requested_train_seconds']}s"
            ),
            elapsed=elapsed,
        )
        return 0
    _write_provisional(
        job_dir, request, status="ok", score=score, summary=summary,
        error=None, elapsed=elapsed,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        # Last resort: request unreadable or provisional unwritable. Print the
        # failure and exit nonzero; the parent records a crash result.
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
