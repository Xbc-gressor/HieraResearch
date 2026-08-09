from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools.hpo_benchmark.arms.pure_smac import PureSMACArm, _RealSMACBackend
from tools.hpo_benchmark.core import (
    BenchmarkContext,
    BenchmarkRunner,
    FunctionObjective,
    Observation,
    SearchSpace,
    load_arm,
)


class FakeSMACBackend:
    def __init__(self, context):
        self.context = context
        self.proposals = iter([{"x": 3.0}, {"x": 2.0}, {"x": 1.0}])
        self.outcomes = []

    def ask(self):
        return next(self.proposals)

    def tell(self, outcome):
        self.outcomes.append(outcome)

    def snapshot(self):
        return {
            "backend": "fake_smac",
            "history_injected": len(self.context.observations),
            "told": len(self.outcomes),
        }


class PureSMACArmTests(unittest.TestCase):
    def test_real_adapter_disables_sobol_and_uses_constants_for_fixed_dimensions(self):
        calls = {"hyperparameters": [], "initial_design": [], "facade": []}

        class ConfigurationSpace:
            def __init__(self, seed):
                self.seed = seed

            def add(self, hyperparameter):
                calls["hyperparameters"].append(hyperparameter)

        class Configuration(dict):
            def __init__(self, configspace, values):
                super().__init__(values)

        class Scenario:
            def __init__(self, configspace, **kwargs):
                self.configspace = configspace
                self.__dict__.update(kwargs)

        class TrialInfo:
            def __init__(self, config, seed):
                self.config = config
                self.seed = seed

        class TrialValue:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class StatusType:
            SUCCESS = "success"
            CRASHED = "crashed"

        class HyperparameterOptimizationFacade:
            @staticmethod
            def get_initial_design(scenario, *, n_configs):
                calls["initial_design"].append(n_configs)
                return ("initial-design", n_configs)

            def __init__(self, scenario, target, **kwargs):
                calls["facade"].append(kwargs)

            def tell(self, info, value):
                pass

        configspace_module = types.ModuleType("ConfigSpace")
        configspace_module.Categorical = lambda name, choices: ("categorical", name, choices)
        configspace_module.Configuration = Configuration
        configspace_module.ConfigurationSpace = ConfigurationSpace
        configspace_module.Constant = lambda name, value: ("constant", name, value)
        configspace_module.Float = lambda name, bounds, **kwargs: ("float", name, bounds)
        configspace_module.Integer = lambda name, bounds, **kwargs: ("integer", name, bounds)
        smac_module = types.ModuleType("smac")
        smac_module.HyperparameterOptimizationFacade = HyperparameterOptimizationFacade
        smac_module.Scenario = Scenario
        dataclasses_module = types.ModuleType("smac.runhistory.dataclasses")
        dataclasses_module.TrialInfo = TrialInfo
        dataclasses_module.TrialValue = TrialValue
        enumerations_module = types.ModuleType("smac.runhistory.enumerations")
        enumerations_module.StatusType = StatusType

        context = BenchmarkContext(
            checkpoint_id="smac-real-adapter",
            regime="first",
            space=SearchSpace.from_legacy(
                {
                    "x": ("float", -5.0, 5.0),
                    "fixed": ("int", 2, 2),
                    "mode": ("categorical", ["only"]),
                }
            ),
            observations=(
                Observation(
                    observation_id="warm",
                    params={"x": 4.0, "fixed": 2, "mode": "only"},
                    score=16.0,
                    status="ok",
                    origin="phase_a",
                    consumes_budget=False,
                ),
            ),
            budget=10,
            seed=11,
        )
        with mock.patch.dict(
            sys.modules,
            {
                "ConfigSpace": configspace_module,
                "smac": smac_module,
                "smac.runhistory.dataclasses": dataclasses_module,
                "smac.runhistory.enumerations": enumerations_module,
            },
        ):
            backend = _RealSMACBackend(context)
            backend._temporary_directory.cleanup()

        self.assertEqual(calls["initial_design"], [0])
        self.assertEqual(
            calls["facade"][0]["initial_design"], ("initial-design", 0)
        )
        self.assertIn(("constant", "fixed", 2), calls["hyperparameters"])
        self.assertIn(("constant", "mode", "only"), calls["hyperparameters"])

    def test_smac_owns_every_post_checkpoint_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = BenchmarkContext(
                checkpoint_id="smac-toy",
                regime="first",
                space=SearchSpace.from_legacy({"x": ("float", -5.0, 5.0)}),
                observations=(
                    Observation(
                        observation_id="warm",
                        params={"x": 4.0},
                        score=16.0,
                        status="ok",
                        origin="phase_a",
                        consumes_budget=False,
                    ),
                ),
                budget=3,
                seed=11,
            )
            backends = []

            def factory(received):
                backend = FakeSMACBackend(received)
                backends.append(backend)
                return backend

            result = BenchmarkRunner(
                context,
                FunctionObjective(lambda params: params["x"] ** 2),
                Path(tmp) / "run",
            ).run(PureSMACArm(backend_factory=factory))

            self.assertEqual(result["final_incumbent_score"], 1.0)
            self.assertEqual(result["policy_snapshot"]["history_injected"], 1)
            self.assertEqual(len(backends[0].outcomes), 3)
            events = [
                json.loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
            ]
            proposals = [event for event in events if event["kind"] == "proposal_batch"]
            self.assertEqual(
                [event["proposals"][0]["origin"] for event in proposals],
                ["pure_smac"] * 3,
            )
            self.assertTrue(
                all(
                    event["proposals"][0]["metadata"]["acquisition"]
                    == "expected_improvement"
                    for event in proposals
                )
            )

    def test_factory_is_loadable_without_a_registry(self):
        factory = load_arm("tools.hpo_benchmark.arms.pure_smac:create_arm")
        self.assertTrue(callable(factory))


if __name__ == "__main__":
    unittest.main()
