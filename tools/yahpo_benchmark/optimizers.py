from __future__ import annotations

import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .llm import LLMCallFailed, LLMProvider
from .types import Observation, Usage


REPO_ROOT = Path(__file__).resolve().parents[2]
HEBO_SUGGEST = REPO_ROOT / "tools" / "inner_benchmark" / "hebo_mace" / "suggest.py"
HEBO_RANK = REPO_ROOT / "tools" / "inner_benchmark" / "hebo_mace" / "rank.py"
MAX_LLM_ATTEMPTS = 3


class OptimizerError(RuntimeError):
    pass


def _call_json_script(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(path)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise OptimizerError(
            f"{path.name} returned invalid JSON: {proc.stdout[-500:]!r}; "
            f"stderr={proc.stderr[-500:]!r}"
        ) from exc
    if proc.returncode != 0 or "error" in result:
        raise OptimizerError(
            f"{path.name} failed: {result.get('error') or proc.stderr[-500:]}"
        )
    return result


def official_hebo_suggest(payload: dict[str, Any]) -> dict[str, Any]:
    return _call_json_script(HEBO_SUGGEST, payload)


def official_mace_rank(payload: dict[str, Any]) -> list[list[float]]:
    return _call_json_script(HEBO_RANK, payload)["values"]


class BaseOptimizer:
    name = "base"

    def __init__(
        self,
        *,
        space,
        task_card: dict[str, Any],
        observations: list[Observation],
        seed: int,
        initial_count: int = 5,
    ):
        self.space = space
        self.task_card = task_card
        self.observations = list(observations)
        self.seed = int(seed)
        self.initial_count = int(initial_count)
        self.usage = Usage()
        self.last_ask_info: dict[str, Any] = {}

    @property
    def bo_step(self) -> int:
        return len(self.observations) - self.initial_count

    def tell(self, observation: Observation) -> None:
        self.observations.append(observation)

    def history_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "configuration": observation.params,
                "objective_value": observation.raw_target,
            }
            for observation in self.observations
        ]

    def hebo_history(self) -> list[dict[str, Any]]:
        return [
            {
                "params": self.space.encode_for_hebo(observation.params),
                "score": observation.score,
            }
            for observation in self.observations
        ]

    def seen_keys(self) -> set[str]:
        return {self.space.key(observation.params) for observation in self.observations}


class HEBOOnly(BaseOptimizer):
    name = "hebo_only"

    def __init__(self, *, suggest_fn: Callable | None = None, **kwargs):
        super().__init__(**kwargs)
        self._suggest_fn = suggest_fn or official_hebo_suggest

    def ask(self, remaining_budget: int) -> dict[str, Any]:
        del remaining_budget
        step_seed = self.seed + 1009 * (self.bo_step + 1)
        result = self._suggest_fn(
            {
                "search_space": self.space.hebo_contract(),
                "history": self.hebo_history(),
                "seed": step_seed,
                "scramble_seed": self.seed,
                "quasi_index": 0,
                "rand_sample": self.initial_count,
            }
        )
        candidate = self.space.decode_from_hebo(result["suggestion"])
        if self.space.key(candidate) in self.seen_keys():
            raise OptimizerError("HEBO suggestion decodes to an observed configuration")
        self.last_ask_info = {
            "mode": result.get("mode"),
            "front_size": result.get("front_size"),
        }
        return candidate


class HEBOMaceLLMPool(BaseOptimizer):
    name = "hebo_mace_llm_pool"

    def __init__(
        self,
        *,
        provider: LLMProvider,
        pool_size: int = 5,
        rank_fn: Callable | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.provider = provider
        self.pool_size = int(pool_size)
        self._rank_fn = rank_fn or official_mace_rank

    def ask(self, remaining_budget: int) -> dict[str, Any]:
        pool = _collect_candidates(
            provider=self.provider,
            usage=self.usage,
            strategy="semantic_pool",
            count=self.pool_size,
            task_card=self.task_card,
            history=self.history_payload(),
            remaining_budget=remaining_budget,
            space=self.space,
            seen=self.seen_keys(),
        )
        step_seed = self.seed + 1009 * (self.bo_step + 1)
        values = self._rank_fn(
            {
                "search_space": self.space.hebo_contract(),
                "history": self.hebo_history(),
                "pool": [self.space.encode_for_hebo(item) for item in pool],
                "seed": step_seed,
            }
        )
        if len(values) != len(pool) or any(len(row) != 3 for row in values):
            raise OptimizerError("HEBO MACE ranker returned an invalid value matrix")
        front = _first_pareto_front(values)
        chosen_index = random.Random(step_seed).choice(front)
        self.last_ask_info = {
            "pool": pool,
            "acquisition_values": values,
            "pareto_front": front,
            "chosen_index": chosen_index,
        }
        return pool[chosen_index]


class LLAMBOModernBatched(BaseOptimizer):
    name = "llambo_modern_batched"

    def __init__(
        self,
        *,
        provider: LLMProvider,
        minimize_raw: bool,
        candidate_count: int = 20,
        prediction_count: int = 10,
        alpha: float = -0.1,
        epsilon: float = 1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.provider = provider
        self.minimize_raw = bool(minimize_raw)
        self.candidate_count = int(candidate_count)
        self.prediction_count = int(prediction_count)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)

    def ask(self, remaining_budget: int) -> dict[str, Any]:
        scores = [observation.score for observation in self.observations]
        best, worst = min(scores), max(scores)
        desired_score = best - self.alpha * (worst - best)
        desired_raw = desired_score if self.minimize_raw else -desired_score
        candidates = _collect_candidates(
            provider=self.provider,
            usage=self.usage,
            strategy="llambo_target_conditioned",
            count=self.candidate_count,
            task_card=self.task_card,
            history=self.history_payload(),
            remaining_budget=remaining_budget,
            space=self.space,
            seen=self.seen_keys(),
            desired_raw_target=desired_raw,
        )
        raw_predictions = _score_candidates(
            provider=self.provider,
            usage=self.usage,
            task_card=self.task_card,
            history=self.history_payload(),
            candidates=candidates,
            prediction_count=self.prediction_count,
        )
        score_predictions = [
            row if self.minimize_raw else [-value for value in row]
            for row in raw_predictions
        ]
        eis = []
        means = []
        standard_deviations = []
        for row in score_predictions:
            mean = statistics.fmean(row)
            std = statistics.pstdev(row)
            means.append(mean)
            standard_deviations.append(std)
            eis.append(_expected_improvement(best, mean, std, self.epsilon))
        chosen_index = max(range(len(eis)), key=eis.__getitem__)
        self.last_ask_info = {
            "desired_raw_target": desired_raw,
            "candidates": candidates,
            "prediction_means_score": means,
            "prediction_stddevs_score": standard_deviations,
            "expected_improvements": eis,
            "chosen_index": chosen_index,
        }
        return candidates[chosen_index]


def _collect_candidates(
    *,
    provider: LLMProvider,
    usage: Usage,
    strategy: str,
    count: int,
    task_card: dict[str, Any],
    history: list[dict[str, Any]],
    remaining_budget: int,
    space,
    seen: set[str],
    desired_raw_target: float | None = None,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    accepted_keys: set[str] = set()
    correction: list[str] = []
    for _ in range(MAX_LLM_ATTEMPTS):
        needed = count - len(accepted)
        payload = {
            "strategy": strategy,
            "task_card": task_card,
            "observations": history,
            "remaining_objective_budget": remaining_budget,
            "requested_count": needed,
            "already_accepted": accepted,
            "correction": correction,
        }
        if desired_raw_target is not None:
            payload["desired_raw_objective_value"] = desired_raw_target
        try:
            response = provider.call("sampler", payload)
        except LLMCallFailed as exc:
            usage.add(exc.usage)
            correction = [str(exc)]
            continue
        usage.add(response.usage)
        configs = response.receipt.get("configs")
        if not isinstance(configs, list) or len(configs) != needed:
            correction = [
                f"expected exactly {needed} configs, got "
                f"{len(configs) if isinstance(configs, list) else type(configs).__name__}"
            ]
            continue
        correction = []
        for index, raw in enumerate(configs):
            if not isinstance(raw, dict):
                correction.append(f"config {index} is not an object")
                continue
            try:
                candidate = space.canonicalize(raw)
                key = space.key(candidate)
            except Exception as exc:
                correction.append(f"config {index} invalid: {exc}")
                continue
            if key in seen:
                correction.append(f"config {index} duplicates an observation")
                continue
            if key in accepted_keys:
                correction.append(f"config {index} duplicates the current pool")
                continue
            accepted.append(candidate)
            accepted_keys.add(key)
        if len(accepted) == count:
            return accepted
        correction.append(f"provide {count - len(accepted)} replacement configs")
    raise OptimizerError(
        f"candidate sampler failed to produce {count} valid unique configs after "
        f"{MAX_LLM_ATTEMPTS} attempts"
    )


def _score_candidates(
    *,
    provider: LLMProvider,
    usage: Usage,
    task_card: dict[str, Any],
    history: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    prediction_count: int,
) -> list[list[float]]:
    correction: list[str] = []
    for _ in range(MAX_LLM_ATTEMPTS):
        payload = {
            "task_card": task_card,
            "observations": history,
            "candidates": candidates,
            "predictions_per_candidate": prediction_count,
            "correction": correction,
        }
        try:
            response = provider.call("scorer", payload)
        except LLMCallFailed as exc:
            usage.add(exc.usage)
            correction = [str(exc)]
            continue
        usage.add(response.usage)
        predictions = response.receipt.get("predictions")
        problem = _prediction_problem(
            predictions,
            candidate_count=len(candidates),
            prediction_count=prediction_count,
        )
        if problem is None:
            return [[float(value) for value in row] for row in predictions]
        correction = [problem]
    raise OptimizerError(
        "LLAMBO scorer failed to produce a finite rectangular prediction matrix "
        f"after {MAX_LLM_ATTEMPTS} attempts"
    )


def _prediction_problem(
    predictions: Any, *, candidate_count: int, prediction_count: int
) -> str | None:
    if not isinstance(predictions, list) or len(predictions) != candidate_count:
        return f"expected {candidate_count} prediction rows"
    for index, row in enumerate(predictions):
        if not isinstance(row, list) or len(row) != prediction_count:
            return f"prediction row {index} must contain {prediction_count} values"
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in row
        ):
            return f"prediction row {index} contains a non-finite/non-numeric value"
    return None


def _expected_improvement(best: float, mean: float, std: float, epsilon: float) -> float:
    improvement = best - mean
    if std <= epsilon:
        return max(improvement, 0.0)
    z = improvement / std
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return improvement * cdf + std * pdf


def _first_pareto_front(values: list[list[float]]) -> list[int]:
    front = []
    for i, point in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue
            if all(b >= a for a, b in zip(point, other)) and any(
                b > a for a, b in zip(point, other)
            ):
                dominated = True
                break
        if not dominated:
            front.append(i)
    if not front:
        raise OptimizerError("MACE returned an empty Pareto front")
    return front
