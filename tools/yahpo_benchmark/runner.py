from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from driver.events import EventsLog

from .aggregate import aggregate
from .io import read_json, write_json
from .llm import SDKLLMProvider
from .objective import YahpoObjective
from .optimizers import HEBOOnly, HEBOMaceLLMPool, LLAMBOModernBatched
from .suite import TaskSpec, task_from_dict
from .types import Observation


OPTIMIZERS = ("hebo_only", "hebo_mace_llm_pool", "llambo_modern_batched")
PROTOCOL_ID = "yahpo-llambo-pilot-v1"


def prepare_task(
    spec: TaskSpec,
    *,
    data_path: Path,
    seed: int,
    output_path: Path,
    initial_count: int = 5,
) -> dict[str, Any]:
    objective = YahpoObjective(spec, data_path=data_path, seed=seed)
    configs = objective.sample_initial(initial_count)
    observations = [objective.evaluate(config) for config in configs]
    prepared = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "seed": seed,
        "initial_count": initial_count,
        "objective": objective.manifest(),
        "initial_observations": [_observation_dict(item) for item in observations],
    }
    write_json(output_path, prepared)
    return prepared


def run_cell(
    *,
    prepared_path: Path,
    data_path: Path,
    optimizer_name: str,
    model: str,
    bo_trials: int,
    stage: str,
    output_dir: Path,
    session_root: Path,
    provider=None,
) -> dict[str, Any]:
    prepared = read_json(prepared_path)
    spec = task_from_dict(prepared["objective"]["task"])
    seed = int(prepared["seed"])
    initial_count = int(prepared["initial_count"])
    objective = YahpoObjective(spec, data_path=data_path, seed=seed)
    observations = [
        Observation(
            params=row["params"],
            raw_target=float(row["raw_target"]),
            score=float(row["score"]),
        )
        for row in prepared["initial_observations"]
    ]
    if provider is None and optimizer_name != "hebo_only":
        provider = SDKLLMProvider(model=model, session_root=session_root)
    optimizer = _make_optimizer(
        optimizer_name,
        objective=objective,
        observations=observations,
        seed=seed,
        initial_count=initial_count,
        provider=provider,
    )
    events = EventsLog(output_dir)
    trials = _initial_trials(objective, observations)
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "task": {"key": spec.key, **spec.as_dict()},
        "seed": seed,
        "model": model if optimizer_name != "hebo_only" else None,
        "optimizer": optimizer_name,
        "initial_count": initial_count,
        "bo_trials": bo_trials,
        "status": "running",
        "error": None,
        "trials": trials,
        "usage": optimizer.usage.as_dict(),
    }
    _refresh_terminal(result)
    write_json(output_dir / "result.json", result)
    for step in range(bo_trials):
        remaining = bo_trials - step
        try:
            params = optimizer.ask(remaining)
        except Exception as exc:
            result["status"] = "arm_error"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["usage"] = optimizer.usage.as_dict()
            result["terminal_score"] = float("inf")
            result["terminal_normalized_regret"] = float("inf")
            events.emit(
                "yahpo_arm_error",
                optimizer=optimizer_name,
                trial_no=initial_count + step + 1,
                error=result["error"],
            )
            write_json(output_dir / "result.json", result)
            return result
        # Objective/adapter failures are benchmark failures, not optimizer
        # robustness outcomes. Let them abort the smoke/matrix visibly.
        observation = objective.evaluate(params)
        optimizer.tell(observation)
        incumbent = min(float(row["score"]) for row in trials)
        incumbent = min(incumbent, observation.score)
        row = {
            "trial_no": initial_count + step + 1,
            "source": optimizer_name,
            **_observation_dict(observation),
            "incumbent_score": incumbent,
            "normalized_regret": objective.normalized_regret(incumbent),
            "ask_info": optimizer.last_ask_info,
        }
        trials.append(row)
        result["usage"] = optimizer.usage.as_dict()
        _refresh_terminal(result)
        events.emit(
            "yahpo_trial",
            optimizer=optimizer_name,
            trial_no=row["trial_no"],
            score=observation.score,
            incumbent_score=incumbent,
        )
        write_json(output_dir / "result.json", result)
    result["status"] = "complete"
    _refresh_terminal(result)
    write_json(output_dir / "result.json", result)
    events.emit(
        "yahpo_run_complete",
        optimizer=optimizer_name,
        trials=len(trials),
        total_cost_usd=result["usage"]["total_cost_usd"],
    )
    return result


def run_matrix(
    *,
    tasks: tuple[TaskSpec, ...],
    data_path: Path,
    output_dir: Path,
    model: str,
    stage: str,
    bo_trials: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_paths = {}
    for spec in tasks:
        path = output_dir / "prepared" / spec.key / "task.json"
        prepare_task(spec, data_path=data_path, seed=seed, output_path=path)
        prepared_paths[spec.key] = path
    write_json(
        output_dir / "matrix.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "stage": stage,
            "tasks": [task.as_dict() for task in tasks],
            "optimizers": list(OPTIMIZERS),
            "seed": seed,
            "initial_count": 5,
            "bo_trials": bo_trials,
            "model": model,
        },
    )

    jobs = []
    for spec in tasks:
        for optimizer_name in OPTIMIZERS:
            jobs.append((spec, optimizer_name))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for spec, optimizer_name in jobs:
            opaque = uuid.uuid4().hex[:12]
            futures.append(
                executor.submit(
                    run_cell,
                    prepared_path=prepared_paths[spec.key],
                    data_path=data_path,
                    optimizer_name=optimizer_name,
                    model=model,
                    bo_trials=bo_trials,
                    stage=stage,
                    output_dir=output_dir / "cells" / spec.key / optimizer_name,
                    session_root=output_dir / "_llm_sessions" / opaque,
                )
            )
        for future in as_completed(futures):
            future.result()
    return aggregate(output_dir)


def _make_optimizer(
    name: str,
    *,
    objective: YahpoObjective,
    observations: list[Observation],
    seed: int,
    initial_count: int,
    provider,
):
    common = {
        "space": objective.space,
        "task_card": objective.task_card(),
        "observations": observations,
        "seed": seed,
        "initial_count": initial_count,
    }
    if name == "hebo_only":
        return HEBOOnly(**common)
    if name == "hebo_mace_llm_pool":
        return HEBOMaceLLMPool(provider=provider, **common)
    if name == "llambo_modern_batched":
        return LLAMBOModernBatched(
            provider=provider,
            minimize_raw=objective.minimize_raw,
            **common,
        )
    raise ValueError(f"unknown optimizer: {name}")


def _initial_trials(
    objective: YahpoObjective, observations: list[Observation]
) -> list[dict[str, Any]]:
    rows = []
    incumbent = float("inf")
    for trial_no, observation in enumerate(observations, start=1):
        incumbent = min(incumbent, observation.score)
        rows.append(
            {
                "trial_no": trial_no,
                "source": "shared_random_initial_design",
                **_observation_dict(observation),
                "incumbent_score": incumbent,
                "normalized_regret": objective.normalized_regret(incumbent),
                "ask_info": None,
            }
        )
    return rows


def _observation_dict(observation: Observation) -> dict[str, Any]:
    return {
        "params": observation.params,
        "raw_target": observation.raw_target,
        "score": observation.score,
    }


def _refresh_terminal(result: dict[str, Any]) -> None:
    if not result["trials"]:
        result["terminal_score"] = float("inf")
        result["terminal_normalized_regret"] = float("inf")
        return
    last = result["trials"][-1]
    result["terminal_score"] = float(last["incumbent_score"])
    result["terminal_normalized_regret"] = float(last["normalized_regret"])
