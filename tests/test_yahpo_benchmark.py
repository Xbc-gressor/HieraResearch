from __future__ import annotations

from pathlib import Path

import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
)

from tools.yahpo_benchmark.io import write_json
from tools.yahpo_benchmark.llm import LLMResponse
from tools.yahpo_benchmark.optimizers import (
    HEBOMaceLLMPool,
    LLAMBOModernBatched,
)
from tools.yahpo_benchmark.space import SearchSpace
from tools.yahpo_benchmark.suite import TaskSpec
from tools.yahpo_benchmark.types import Observation, Usage


class ScriptedProvider:
    def __init__(self, receipts):
        self.receipts = list(receipts)
        self.calls = []

    def call(self, kind, payload):
        self.calls.append((kind, payload))
        receipt = self.receipts.pop(0)
        return LLMResponse(
            receipt=receipt,
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_cost_usd": 0.01,
                "status": "ok",
            },
        )


def numeric_space(seed=0):
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters(
        [UniformFloatHyperparameter("x", 0.0, 1.0), Constant("instance", "1")]
    )
    return SearchSpace(cs, instance_name="instance", instance="1")


def initial_observations():
    return [
        Observation(params={"x": value}, raw_target=value, score=value)
        for value in (0.0, 0.1, 0.2, 0.3, 0.4)
    ]


def test_conditional_space_round_trip_uses_rectangular_hebo_encoding():
    cs = ConfigurationSpace(seed=0)
    parent = CategoricalHyperparameter("parent", ["a", "b"])
    child = UniformFloatHyperparameter("child", 0.0, 1.0, default_value=0.25)
    cs.add_hyperparameters([parent, child, Constant("instance", "7")])
    cs.add_condition(EqualsCondition(child, parent, "b"))
    space = SearchSpace(cs, instance_name="instance", instance="7")

    inactive = space.canonicalize({"parent": "a"})
    assert inactive == {"parent": "a"}
    assert space.encode_for_hebo(inactive) == {"child": 0.25, "parent": "a"}
    assert space.decode_from_hebo({"child": 0.8, "parent": "a"}) == {
        "parent": "a"
    }
    assert space.decode_from_hebo({"child": 0.8, "parent": "b"}) == {
        "child": 0.8,
        "parent": "b",
    }


def test_hebo_mace_pool_executes_unique_pareto_winner():
    provider = ScriptedProvider(
        [
            {"configs": [{"x": 0.0}] * 5},
            {"configs": [{"x": value} for value in (0.5, 0.6, 0.7, 0.8, 0.9)]},
        ]
    )
    optimizer = HEBOMaceLLMPool(
        space=numeric_space(),
        task_card={},
        observations=initial_observations(),
        seed=3,
        provider=provider,
        rank_fn=lambda payload: [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ],
    )

    assert optimizer.ask(20) == {"x": 0.7}
    assert optimizer.last_ask_info["pareto_front"] == [2]
    assert optimizer.usage.calls == 2


def test_llambo_batched_uses_raw_predictions_and_deterministic_ei():
    candidates = [{"x": 0.5 + index * 0.02} for index in range(20)]
    predictions = [[0.45] * 10 for _ in candidates]
    predictions[7] = [0.95] * 10
    provider = ScriptedProvider(
        [{"configs": candidates}, {"predictions": predictions}]
    )
    maximize_history = [
        Observation(params={"x": value}, raw_target=value, score=-value)
        for value in (0.0, 0.1, 0.2, 0.3, 0.4)
    ]
    optimizer = LLAMBOModernBatched(
        space=numeric_space(),
        task_card={"objective": {"direction": "maximize"}},
        observations=maximize_history,
        seed=0,
        provider=provider,
        minimize_raw=False,
    )

    assert optimizer.ask(20) == candidates[7]
    sampler_payload = provider.calls[0][1]
    assert sampler_payload["desired_raw_objective_value"] == pytest.approx(0.36)
    assert optimizer.usage.calls == 2


def test_runner_counts_shared_five_inside_each_optimizer_budget(tmp_path, monkeypatch):
    from tools.yahpo_benchmark import runner

    spec = TaskSpec("fake", "1", "metric", "learner", 10, 1, 0, 2)
    initial = initial_observations()
    prepared_path = tmp_path / "prepared.json"
    write_json(
        prepared_path,
        {
            "seed": 0,
            "initial_count": 5,
            "objective": {"task": spec.as_dict()},
            "initial_observations": [
                {"params": row.params, "raw_target": row.raw_target, "score": row.score}
                for row in initial
            ],
        },
    )

    class FakeObjective:
        def __init__(self, spec, **kwargs):
            self.spec = spec
            self.space = numeric_space()
            self.minimize_raw = True

        def task_card(self):
            return {}

        def evaluate(self, params):
            return Observation(params=params, raw_target=params["x"], score=params["x"])

        def normalized_regret(self, score):
            return score

    class FakeOptimizer:
        def __init__(self, observations):
            self.observations = list(observations)
            self.usage = Usage()
            self.last_ask_info = {}

        def ask(self, remaining):
            return {"x": 0.5 + 0.1 * (3 - remaining)}

        def tell(self, observation):
            self.observations.append(observation)

    monkeypatch.setattr(runner, "YahpoObjective", FakeObjective)
    monkeypatch.setattr(
        runner,
        "_make_optimizer",
        lambda name, observations, **kwargs: FakeOptimizer(observations),
    )

    starts = []
    for name in runner.OPTIMIZERS:
        result = runner.run_cell(
            prepared_path=prepared_path,
            data_path=tmp_path,
            optimizer_name=name,
            model="fake",
            bo_trials=2,
            stage="test",
            output_dir=tmp_path / name,
            session_root=tmp_path / "sessions" / name,
            provider=object(),
        )
        assert result["status"] == "complete"
        assert len(result["trials"]) == 7
        starts.append([row["params"] for row in result["trials"][:5]])
    assert starts[0] == starts[1] == starts[2]
