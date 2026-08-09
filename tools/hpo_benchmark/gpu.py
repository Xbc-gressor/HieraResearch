"""Manual multi-server GPU runner for the ten-arm HPO benchmark.

Each server first calibrates a frozen checkpoint on that machine, then runs
independent (checkpoint, arm, seed) cells against the immutable calibration.
There is deliberately no global scheduler in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TUNER_DIR = ROOT / "tools" / "tuners"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TUNER_DIR))

from _common import _communicate_with_limit, is_finite_score, timed_preflight  # noqa: E402

from tools.apply_base_params import apply as apply_base_params  # noqa: E402
from tools.hpo_benchmark.core import (  # noqa: E402
    BenchmarkContext,
    BenchmarkRunner,
    Observation,
    SearchSpace,
    load_arm,
)
from tools.hpo_benchmark.providers import ClaudeCLIProposalProvider  # noqa: E402


ARM_SPECS = {
    "current": "tools.hpo_benchmark.arms.current:create_arm",
    "llm_hillclimb": "tools.hpo_benchmark.arms.hillclimb:create_arm",
    "llm_active_set": "tools.hpo_benchmark.arms.active_set:create_arm",
    "pure_smac": "tools.hpo_benchmark.arms.pure_smac:create_arm",
    "llm_pool_smac_rank": "tools.hpo_benchmark.arms.smac_rank:create_arm",
    "llm_pool_gp_rank": "tools.hpo_benchmark.arms.gp_rank:create_arm",
    "llm_pool_tpe_rank": "tools.hpo_benchmark.arms.tpe_rank:create_arm",
    "llm_pool_hebo_rank": "tools.hpo_benchmark.arms.hebo_rank:create_arm",
    "local_trust_region": "tools.hpo_benchmark.arms.local_trust_region:create_arm",
    "spsa": "tools.hpo_benchmark.arms.spsa:create_arm",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
        )


def _last_json_object(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _frozen_search_space(
    candidate_dir: Path, report: Mapping[str, Any], regime: str
) -> Mapping[str, Sequence[Any]]:
    if regime == "first":
        summary = _last_json_object(candidate_dir / "_phase_c_bo.log")
        if summary is not None and isinstance(summary.get("search_space"), Mapping):
            return summary["search_space"]
    clamp = report.get("search_space_clamp")
    if isinstance(clamp, Mapping) and isinstance(
        clamp.get("clamped_search_space"), Mapping
    ):
        clamped = clamp["clamped_search_space"]
        if regime == "continuation":
            # A later bout can overwrite search_space_clamp in the report. If
            # that box excludes an actually admitted first-bout configuration,
            # it cannot be the frozen continuation space; fall back to Phase A.
            candidate_space = SearchSpace.from_legacy(clamped)
            phase_c = report.get("phase_c")
            stages = phase_c.get("stages") if isinstance(phase_c, Mapping) else None
            if isinstance(stages, list) and stages and isinstance(stages[0], Mapping):
                admitted_params = []
                for row in stages[0].get("trials", []):
                    if not isinstance(row, Mapping) or row.get("status") == "preflight_rejected":
                        continue
                    if len(admitted_params) >= 10:
                        break
                    params = row.get("params")
                    if isinstance(params, Mapping):
                        admitted_params.append(params)
                if len(admitted_params) == 10 and all(
                    candidate_space.project(params) == params
                    for params in admitted_params
                ):
                    return clamped
            else:
                return clamped
        else:
            return clamped
    phase_a = report.get("phase_a")
    if isinstance(phase_a, Mapping) and isinstance(phase_a.get("search_space"), Mapping):
        return phase_a["search_space"]
    raise ValueError("tune_report.json contains no usable frozen search space")


def _source_checkpoint(
    candidate_dir: Path, regime: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    report = _read_json(candidate_dir / "tune_report.json")
    if not isinstance(report, Mapping):
        raise ValueError("tune_report.json must contain an object")
    space = dict(_frozen_search_space(candidate_dir, report, regime))
    phase_a = report.get("phase_a")
    if not isinstance(phase_a, Mapping):
        raise ValueError("tune_report.json has no phase_a object")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(phase_a.get("warm_start_configs", [])):
        if not isinstance(row, Mapping) or not is_finite_score(row.get("score")):
            continue
        rows.append(
            {
                "source_id": f"phase-a-{index:03d}",
                "params": dict(row["params"]),
                "source_score": float(row["score"]),
                "origin": "phase_a",
                "eligible_incumbent": row.get("role") != "inherited_control",
                "metadata": {
                    key: row[key]
                    for key in ("proposed_index", "role", "params_sha256")
                    if key in row
                },
            }
        )

    if regime == "continuation":
        phase_c = report.get("phase_c")
        stages = phase_c.get("stages") if isinstance(phase_c, Mapping) else None
        if not isinstance(stages, list) or not stages or not isinstance(stages[0], Mapping):
            raise ValueError("continuation checkpoint requires a first phase_c stage")
        admitted = 0
        for index, row in enumerate(stages[0].get("trials", [])):
            if not isinstance(row, Mapping):
                continue
            if row.get("status") == "preflight_rejected":
                continue
            if admitted >= 10:
                break
            admitted += 1
            if not isinstance(row.get("params"), Mapping):
                raise ValueError(f"phase_c trial {index} has no params object")
            rows.append(
                {
                    "source_id": f"phase-c-{index:03d}",
                    "params": dict(row["params"]),
                    "source_score": (
                        float(row["score"])
                        if is_finite_score(row.get("score"))
                        else None
                    ),
                    "origin": "first_bout",
                    "eligible_incumbent": True,
                    "metadata": {"source_status": row.get("status", "ok")},
                }
            )
        if admitted != 10:
            raise ValueError(
                f"continuation checkpoint has {admitted} admitted first-bout trials, expected 10"
            )

    deferred = [
        dict(row["params"])
        for row in phase_a.get("deferred_configs", [])
        if isinstance(row, Mapping) and isinstance(row.get("params"), Mapping)
    ]
    return space, rows, deferred


def _remote_incumbent(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get("eligible_incumbent") and is_finite_score(row.get("source_score"))
    ]
    if not eligible:
        raise ValueError("source checkpoint has no eligible finite incumbent")
    return min(eligible, key=lambda row: float(row["source_score"]))


def _hardware_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        metadata["gpus"] = [
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        ]
    else:
        metadata["nvidia_smi_error"] = completed.stderr.strip()[:500]
    return metadata


def _software_metadata() -> dict[str, Any]:
    versions = {}
    for package in ("torch", "numpy", "optuna", "smac", "gpytorch", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _command_version(command: str) -> str | None:
    completed = subprocess.run(
        [command, "--version"], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


class CandidateObjective:
    """Task-owned preflight plus one fresh subprocess per score attempt."""

    def __init__(self, candidate_path: Path, *, evaluation_timeout: float):
        self.candidate_path = Path(candidate_path)
        self.evaluation_timeout = float(evaluation_timeout)

    def preflight(self, params: Mapping[str, Any]) -> str | None:
        try:
            timed_preflight(dict(params), self.candidate_path)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return None

    def evaluate(self, params: Mapping[str, Any]) -> float:
        eval_one = str(TUNER_DIR / "_eval_one.py")
        out, err, returncode = _communicate_with_limit(
            [
                sys.executable,
                eval_one,
                str(self.candidate_path),
                json.dumps(dict(params), ensure_ascii=False, allow_nan=False),
            ],
            limit=self.evaluation_timeout,
            label="benchmark evaluation exceeded evaluation_timeout",
        )
        for line in out.splitlines():
            if line.startswith("RESULT:"):
                score = float(line[len("RESULT:") :])
                if not is_finite_score(score):
                    raise ValueError(f"evaluation returned non-finite score: {score!r}")
                return score
        detail = err.strip()
        if len(detail) > 4000:
            detail = "...[stderr truncated]...\n" + detail[-4000:]
        raise RuntimeError(
            f"evaluation subprocess exited with code {returncode} without a RESULT line"
            + (f"\nchild stderr:\n{detail}" if detail else "")
        )


def _copy_candidate(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in (
        "train.py",
        "prepare.py",
        "_candidate_brief.json",
        "_parameter_transfer.json",
        "_search_space.json",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


def prepare_checkpoint(args: argparse.Namespace) -> int:
    source = args.source_candidate.resolve()
    output = args.output.resolve()
    checkpoint_path = output / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = _read_json(checkpoint_path)
        if checkpoint.get("checkpoint_id") != args.checkpoint_id:
            raise RuntimeError(
                "existing checkpoint_id does not match the requested checkpoint"
            )
        if checkpoint.get("regime") != args.regime:
            raise RuntimeError(
                "existing checkpoint regime does not match the requested regime"
            )
        candidate_dir = output / "candidate"
        for name, field in (
            ("train.py", "frozen_train_sha256"),
            ("prepare.py", "frozen_prepare_sha256"),
        ):
            if not (candidate_dir / name).is_file() or _sha256(
                candidate_dir / name
            ) != checkpoint.get(field):
                raise RuntimeError(
                    f"existing checkpoint candidate {name} changed after calibration"
                )
        print(f"checkpoint already complete: {checkpoint_path}")
        return 0
    missing = [
        name
        for name in ("train.py", "prepare.py", "tune_report.json")
        if not (source / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"source candidate is missing required files: {', '.join(missing)}"
        )

    space_mapping, source_rows, deferred = _source_checkpoint(source, args.regime)
    space = SearchSpace.from_legacy(space_mapping)
    remote_incumbent = _remote_incumbent(source_rows)
    expected = {
        "checkpoint_id": args.checkpoint_id,
        "regime": args.regime,
        "source_train_sha256": _sha256(source / "train.py"),
        "source_prepare_sha256": _sha256(source / "prepare.py"),
        "space": space_mapping,
        "source_rows": source_rows,
        "deferred_configs": deferred,
        "evaluation_timeout": args.evaluation_timeout,
    }
    candidate_dir = output / "candidate"
    spec_path = output / "checkpoint_spec.json"
    if output.exists():
        if not spec_path.is_file() or not (candidate_dir / "train.py").is_file():
            raise FileExistsError(
                f"checkpoint output exists without resumable artifacts: {output}"
            )
        spec = _read_json(spec_path)
        for key, value in expected.items():
            if spec.get(key) != value:
                raise RuntimeError(
                    f"cannot resume checkpoint: {key} differs from checkpoint_spec.json"
                )
        if _sha256(candidate_dir / "train.py") != spec["frozen_train_sha256"]:
            raise RuntimeError("cannot resume checkpoint: frozen train.py changed")
        if _sha256(candidate_dir / "prepare.py") != spec["frozen_prepare_sha256"]:
            raise RuntimeError("cannot resume checkpoint: frozen prepare.py changed")
        print(f"resuming calibration: {output}")
    else:
        output.mkdir(parents=True)
        _copy_candidate(source, candidate_dir)
        apply_receipt = apply_base_params(
            candidate_dir / "train.py", dict(remote_incumbent["params"])
        )
        spec = {
            "schema_version": 1,
            "checkpoint_id": args.checkpoint_id,
            "regime": args.regime,
            "source_candidate": str(source),
            "source_train_sha256": _sha256(source / "train.py"),
            "source_prepare_sha256": _sha256(source / "prepare.py"),
            "frozen_train_sha256": _sha256(candidate_dir / "train.py"),
            "frozen_prepare_sha256": _sha256(candidate_dir / "prepare.py"),
            "base_params_restore": apply_receipt,
            "space": space_mapping,
            "source_rows": source_rows,
            "deferred_configs": deferred,
            "hardware": _hardware_metadata(),
            "evaluation_timeout": args.evaluation_timeout,
        }
        _write_json(spec_path, spec)

    objective = CandidateObjective(
        candidate_dir / "train.py", evaluation_timeout=args.evaluation_timeout
    )
    events = output / "calibration.jsonl"
    cached: dict[str, dict[str, Any]] = {}
    if events.is_file():
        for line in events.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            cached[value["observation_id"]] = value
    observations: list[dict[str, Any]] = []
    for row in source_rows:
        if row["source_id"] in cached:
            observation = cached[row["source_id"]]
            observations.append(observation)
            print(
                f"{row['source_id']}: cached {observation['status']} "
                f"score={observation['score']}"
            )
            continue
        params = space.project(row["params"])
        rejection = objective.preflight(params)
        if rejection is not None:
            observation = {
                "observation_id": row["source_id"],
                "params": params,
                "score": None,
                "status": "rejected",
                "origin": row["origin"],
                "eligible_incumbent": False,
                "consumes_budget": False,
                "failure": rejection,
                "metadata": {**row["metadata"], "source_score": row["source_score"]},
            }
        else:
            try:
                score = objective.evaluate(params)
                status, failure = "ok", None
            except Exception as exc:
                score, status = "+inf", "crash"
                failure = f"{type(exc).__name__}: {exc}"
            observation = {
                "observation_id": row["source_id"],
                "params": params,
                "score": score,
                "status": status,
                "origin": row["origin"],
                "eligible_incumbent": bool(row["eligible_incumbent"]),
                "consumes_budget": True,
                "failure": failure,
                "metadata": {**row["metadata"], "source_score": row["source_score"]},
            }
        observations.append(observation)
        _append_jsonl(events, observation)
        print(
            f"{row['source_id']}: {observation['status']} "
            f"score={observation['score']}"
        )

    finite_eligible = [
        row
        for row in observations
        if row["status"] == "ok"
        and row["eligible_incumbent"]
        and is_finite_score(row["score"])
    ]
    if not finite_eligible:
        raise RuntimeError("local calibration produced no finite eligible incumbent")
    local_incumbent = min(finite_eligible, key=lambda row: float(row["score"]))
    checkpoint = {
        **spec,
        "observations": observations,
        "local_incumbent_id": local_incumbent["observation_id"],
        "local_incumbent_score": local_incumbent["score"],
    }
    _write_json(checkpoint_path, checkpoint)
    print(
        f"checkpoint ready: {checkpoint_path}\n"
        f"local incumbent: {local_incumbent['observation_id']} "
        f"score={local_incumbent['score']}"
    )
    return 0


def _load_context(
    checkpoint_dir: Path,
    *,
    budget: int,
    seed: int,
    cell_metadata: Mapping[str, Any] | None = None,
) -> tuple[BenchmarkContext, dict[str, Any]]:
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    data = _read_json(checkpoint_path)
    space = SearchSpace.from_legacy(data["space"])
    observations = tuple(
        Observation(
            observation_id=row["observation_id"],
            params=row["params"],
            score=(math.inf if row["score"] == "+inf" else row["score"]),
            status=row["status"],
            origin=row["origin"],
            eligible_incumbent=row["eligible_incumbent"],
            consumes_budget=row["consumes_budget"],
            failure=row.get("failure"),
            metadata=row.get("metadata", {}),
        )
        for row in data["observations"]
    )
    context = BenchmarkContext(
        checkpoint_id=data["checkpoint_id"],
        regime=data["regime"],
        space=space,
        observations=observations,
        budget=budget,
        seed=seed,
        metadata={
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_hardware": data["hardware"],
            "cell_hardware": _hardware_metadata(),
            "source_train_sha256": data["source_train_sha256"],
            "frozen_train_sha256": data["frozen_train_sha256"],
            "source_prepare_sha256": data["source_prepare_sha256"],
            "frozen_prepare_sha256": data["frozen_prepare_sha256"],
            "software_versions": _software_metadata(),
            **dict(cell_metadata or {}),
        },
    )
    return context, data


def run_cell(args: argparse.Namespace) -> int:
    checkpoint_dir = args.checkpoint.resolve()
    context, checkpoint = _load_context(
        checkpoint_dir,
        budget=args.budget,
        seed=args.seed,
        cell_metadata={
            "proposal_model": args.model,
            "claude_cli": args.claude_cli,
            "claude_cli_version": _command_version(args.claude_cli),
            "gpu_runner_sha256": _sha256(Path(__file__)),
            "benchmark_core_sha256": _sha256(ROOT / "tools" / "hpo_benchmark" / "core.py"),
        },
    )
    provider = ClaudeCLIProposalProvider(
        model=args.model,
        cwd=ROOT,
        cli_path=args.claude_cli,
        max_budget_usd=args.max_llm_call_usd,
    )
    factory = load_arm(ARM_SPECS[args.arm])
    policy = factory(
        provider=provider,
        deferred_configs=(
            checkpoint["deferred_configs"] if context.regime == "first" else ()
        ),
    )
    candidate_path = checkpoint_dir / "candidate" / "train.py"
    if _sha256(candidate_path) != checkpoint["frozen_train_sha256"]:
        raise RuntimeError("frozen checkpoint candidate changed after calibration")
    prepare_path = checkpoint_dir / "candidate" / "prepare.py"
    if _sha256(prepare_path) != checkpoint["frozen_prepare_sha256"]:
        raise RuntimeError("frozen checkpoint evaluator changed after calibration")
    objective = CandidateObjective(
        candidate_path, evaluation_timeout=args.evaluation_timeout
    )
    result = BenchmarkRunner(
        context, objective, args.output.resolve()
    ).run(policy)
    print(
        f"{result['checkpoint_id']} {result['arm']} seed={result['seed']}: "
        f"improvement={result['improvement']:.8f} "
        f"({result['initial_incumbent_score']:.8f} -> "
        f"{result['final_incumbent_score']:.8f})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-checkpoint", help="copy and calibrate one frozen checkpoint"
    )
    prepare.add_argument("--source-candidate", type=Path, required=True)
    prepare.add_argument("--regime", choices=("first", "continuation"), required=True)
    prepare.add_argument("--checkpoint-id", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--evaluation-timeout", type=float, default=900.0)
    prepare.set_defaults(func=prepare_checkpoint)

    cell = subparsers.add_parser("run-cell", help="run one arm and seed")
    cell.add_argument("--checkpoint", type=Path, required=True)
    cell.add_argument("--arm", choices=tuple(ARM_SPECS), required=True)
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument("--budget", type=int, default=10)
    cell.add_argument("--output", type=Path, required=True)
    cell.add_argument("--model", default="sonnet")
    cell.add_argument("--claude-cli", default="claude")
    cell.add_argument("--max-llm-call-usd", type=float)
    cell.add_argument("--evaluation-timeout", type=float, default=900.0)
    cell.set_defaults(func=run_cell)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
