from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

from .space import SearchSpace, _plain
from .suite import TaskSpec
from .types import Observation


class YahpoObjective:
    def __init__(self, spec: TaskSpec, *, data_path: Path, seed: int):
        self.spec = spec
        self.data_path = Path(data_path).resolve()
        self._validate_data_version()

        from yahpo_gym.local_config import local_config

        # Avoid writing YAHPO's process-global ~/.config/yahpo_gym file. Every
        # benchmark worker in this process uses the same explicitly supplied path.
        local_config._config = {"data_path": str(self.data_path)}
        module = "lcbench" if spec.scenario == "lcbench" else "rbv2"
        importlib.import_module(f"yahpo_gym.benchmarks.{module}")
        from yahpo_gym import benchmark_set

        self.benchmark = benchmark_set.BenchmarkSet(
            spec.scenario,
            instance=spec.instance,
            noisy=False,
            multithread=False,
        )
        opt_space = self.benchmark.get_opt_space(drop_fidelity_params=True, seed=seed)
        self.space = SearchSpace(
            opt_space,
            instance_name=self.benchmark.config.instance_names,
            instance=spec.instance,
        )
        self.fidelity = self._max_fidelity()
        target_index = self.benchmark.config.y_names.index(spec.target)
        self.minimize_raw = bool(
            self.benchmark.config.config["y_minimize"][target_index]
        )
        self.y_best, self.y_worst = self._target_range()

    def sample_initial(self, count: int) -> list[dict[str, Any]]:
        configs: list[dict[str, Any]] = []
        seen: set[str] = set()
        while len(configs) < count:
            candidate = self.space.sample(1)[0]
            key = self.space.key(candidate)
            if key in seen:
                continue
            seen.add(key)
            configs.append(candidate)
        return configs

    def evaluate(self, params: dict[str, Any]) -> Observation:
        canonical = self.space.canonicalize(params)
        full = {
            **canonical,
            **self.fidelity,
            self.benchmark.config.instance_names: self.space._instance_value(),
        }
        result = self.benchmark.objective_function(full, logging=False)[0]
        raw = float(result[self.spec.target])
        if not math.isfinite(raw):
            raise RuntimeError(f"YAHPO returned non-finite {self.spec.target}: {raw}")
        score = raw if self.minimize_raw else -raw
        return Observation(params=canonical, raw_target=raw, score=score)

    def normalized_regret(self, score: float) -> float:
        return (score - self.y_best) / (self.y_worst - self.y_best)

    def task_card(self) -> dict[str, Any]:
        return {
            "learner_family": self.spec.learner_family,
            "dataset": self.spec.public_metadata(),
            "objective": {
                "name": self.spec.target,
                "direction": "minimize" if self.minimize_raw else "maximize",
            },
            "hyperparameters": self.space.task_dimensions(),
            "conditional_clauses": self.space.conditions(),
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "task": self.spec.as_dict(),
            "target_direction": "minimize" if self.minimize_raw else "maximize",
            "fidelity": self.fidelity,
            "y_best": self.y_best,
            "y_worst": self.y_worst,
            "dimension_count": self.space.dimension_count,
            "task_card": self.task_card(),
        }

    def raw_from_score(self, score: float) -> float:
        return score if self.minimize_raw else -score

    def score_from_raw(self, raw: float) -> float:
        return raw if self.minimize_raw else -raw

    def _max_fidelity(self) -> dict[str, Any]:
        values = {}
        for hp in self.benchmark.get_fidelity_space().get_hyperparameters():
            if hasattr(hp, "upper"):
                value = hp.upper
            elif hasattr(hp, "choices"):
                value = hp.choices[-1]
            elif hasattr(hp, "sequence"):
                value = hp.sequence[-1]
            else:
                value = hp.default_value
            values[hp.name] = _plain(value)
        return values

    def _target_range(self) -> tuple[float, float]:
        stats = self.benchmark.target_stats.copy()
        rows = stats[
            (stats["scenario"].astype(str) == self.spec.scenario)
            & (stats["instance"].astype(str) == self.spec.instance)
            & (stats["metric"].astype(str) == self.spec.target)
        ]
        values = {
            str(row["statistic"]): float(row["value"])
            for _, row in rows.iterrows()
        }
        if set(values) != {"min", "max"}:
            raise RuntimeError(
                f"missing target statistics for {self.spec.key}/{self.spec.target}"
            )
        if self.minimize_raw:
            y_best, y_worst = values["min"], values["max"]
        else:
            y_best, y_worst = -values["max"], -values["min"]
        if not y_worst > y_best:
            raise RuntimeError(f"invalid target range: {y_best}, {y_worst}")
        return y_best, y_worst

    def _validate_data_version(self) -> None:
        version_file = self.data_path / "VERSION"
        if not version_file.is_file():
            raise FileNotFoundError(f"YAHPO data VERSION missing: {version_file}")
        version = version_file.read_text(encoding="utf-8").strip()
        if version != "VERSION:1.0.2":
            raise ValueError(f"expected YAHPO data VERSION:1.0.2, got {version!r}")
