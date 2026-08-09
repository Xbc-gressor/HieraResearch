from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.hpo_benchmark.arms.active_set import LLMActiveSetArm
from tools.hpo_benchmark.core import (
    BenchmarkContext,
    BenchmarkRunner,
    FunctionObjective,
    Observation,
    PolicyContractError,
    SearchSpace,
    load_arm,
)
from tools.hpo_benchmark.providers import ReplayProposalProvider


def move(name: str, direction: int, step: float) -> dict:
    return {"name": name, "direction": direction, "step": step}


def response(*moves: dict) -> dict:
    return {
        "active_dimensions": list(moves),
        "reason": "probe the selected local interaction",
    }


class ActiveSetArmTests(unittest.TestCase):
    def test_runner_builds_one_atomic_pair_from_a_shared_incumbent(self):
        with tempfile.TemporaryDirectory() as tmp:
            space = SearchSpace.from_legacy(
                {
                    "x": ("float", -10.0, 10.0),
                    "y": ("int", 0, 10),
                    "mode": ("categorical", ["a", "b"]),
                }
            )
            initial = Observation(
                observation_id="warm",
                params={"x": 0.0, "y": 5, "mode": "a"},
                score=8.0,
                status="ok",
                origin="warm",
                consumes_budget=False,
            )
            context = BenchmarkContext(
                checkpoint_id="active-set-pair",
                regime="first",
                space=space,
                observations=(initial,),
                budget=2,
                seed=0,
            )
            provider = ReplayProposalProvider(
                [response(move("x", 1, 0.10), move("y", -1, 0.20))]
            )
            factory = load_arm(
                "tools.hpo_benchmark.arms.active_set:create_arm"
            )
            result = BenchmarkRunner(
                context,
                FunctionObjective(
                    lambda params: (params["x"] - 2.0) ** 2
                    + (params["y"] - 3) ** 2
                ),
                Path(tmp) / "run",
            ).run(factory(provider=provider))

            self.assertEqual(result["evaluations_consumed"], 2)
            self.assertEqual(result["final_incumbent_score"], 0.0)
            self.assertEqual(result["policy_snapshot"]["completed_rounds"], 1)
            events = [
                json.loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
            ]
            batch = next(event for event in events if event["kind"] == "proposal_batch")
            self.assertTrue(batch["atomic"])
            self.assertEqual(len(batch["proposals"]), 2)
            plus, minus = batch["proposals"]
            self.assertEqual(plus["params"], {"x": 2.0, "y": 3, "mode": "a"})
            self.assertEqual(minus["params"], {"x": -2.0, "y": 7, "mode": "a"})
            self.assertEqual(
                [plus["metadata"]["group_role"], minus["metadata"]["group_role"]],
                ["plus", "minus"],
            )
            self.assertEqual(
                plus["metadata"]["center_observation_id"],
                minus["metadata"]["center_observation_id"],
            )

    def test_log_dimension_moves_symmetrically_in_log_coordinates(self):
        space = SearchSpace.from_legacy({"lr": ("float", 0.001, 0.1, "log")})
        initial = Observation(
            observation_id="warm",
            params={"lr": 0.01},
            score=1.0,
            status="ok",
            origin="warm",
            consumes_budget=False,
        )
        context = BenchmarkContext(
            checkpoint_id="active-set-log",
            regime="continuation",
            space=space,
            observations=(initial,),
            budget=2,
            seed=0,
        )
        arm = LLMActiveSetArm(
            ReplayProposalProvider([response(move("lr", 1, 0.10))])
        )
        arm.initialize(context)

        batch = arm.ask(2)

        plus = batch.proposals[0].params["lr"]
        minus = batch.proposals[1].params["lr"]
        self.assertGreater(plus, 0.01)
        self.assertLess(minus, 0.01)
        self.assertTrue(math.isclose(plus * minus, 0.01**2, rel_tol=1e-12))

    def test_near_boundary_shrinks_to_an_actually_symmetric_step(self):
        space = SearchSpace.from_legacy({"x": ("float", 0.0, 1.0)})
        initial = Observation(
            observation_id="warm",
            params={"x": 0.05},
            score=1.0,
            status="ok",
            origin="warm",
            consumes_budget=False,
        )
        context = BenchmarkContext(
            checkpoint_id="active-set-near-boundary",
            regime="first",
            space=space,
            observations=(initial,),
            budget=2,
            seed=0,
        )
        arm = LLMActiveSetArm(
            ReplayProposalProvider([response(move("x", 1, 0.10))])
        )
        arm.initialize(context)

        batch = arm.ask(2)

        plus = batch.proposals[0]
        minus = batch.proposals[1]
        self.assertAlmostEqual(plus.params["x"], 0.10)
        self.assertAlmostEqual(minus.params["x"], 0.0)
        self.assertAlmostEqual(
            plus.params["x"] - 0.05,
            0.05 - minus.params["x"],
        )
        self.assertEqual(plus.metadata["active_dimensions"][0]["step"], 0.10)
        self.assertAlmostEqual(plus.metadata["effective_steps"]["x"], 0.05)

    def test_projection_collapse_retries_with_a_fresh_active_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            space = SearchSpace.from_legacy(
                {"x": ("float", 0.0, 1.0), "y": ("float", 0.0, 1.0)}
            )
            initial = Observation(
                observation_id="warm",
                params={"x": 1.0, "y": 0.5},
                score=1.25,
                status="ok",
                origin="warm",
                consumes_budget=False,
            )
            context = BenchmarkContext(
                checkpoint_id="active-set-retry",
                regime="first",
                space=space,
                observations=(initial,),
                budget=2,
                seed=0,
            )
            provider = ReplayProposalProvider(
                [
                    response(move("x", 1, 0.10)),
                    response(move("y", 1, 0.10)),
                ]
            )
            result = BenchmarkRunner(
                context,
                FunctionObjective(lambda params: params["x"] ** 2 + params["y"] ** 2),
                Path(tmp) / "run",
            ).run(LLMActiveSetArm(provider))

            self.assertEqual(result["evaluations_consumed"], 2)
            self.assertEqual(result["proposal_batches"], 2)
            self.assertEqual(result["policy_snapshot"]["rejected_batches"], 1)
            self.assertEqual(len(provider.calls), 2)
            second_prompt = json.loads(provider.calls[1]["prompt"].split("\n\n", 1)[1])
            self.assertEqual(len(second_prompt["recent_rejections"]), 2)

    def test_rejects_odd_budget_and_non_numeric_selection(self):
        categorical = SearchSpace.from_legacy(
            {
                "x": ("float", 0.0, 1.0),
                "mode": ("categorical", ["a", "b"]),
            }
        )
        initial = Observation(
            observation_id="warm",
            params={"x": 0.5, "mode": "a"},
            score=1.0,
            status="ok",
            origin="warm",
            consumes_budget=False,
        )
        odd = BenchmarkContext(
            checkpoint_id="active-set-odd",
            regime="first",
            space=categorical,
            observations=(initial,),
            budget=3,
            seed=0,
        )
        with self.assertRaisesRegex(PolicyContractError, "even evaluation budget"):
            LLMActiveSetArm(ReplayProposalProvider([])).initialize(odd)

        even = BenchmarkContext(
            checkpoint_id="active-set-categorical",
            regime="first",
            space=categorical,
            observations=(initial,),
            budget=2,
            seed=0,
        )
        arm = LLMActiveSetArm(
            ReplayProposalProvider([response(move("mode", 1, 0.10))])
        )
        arm.initialize(even)
        with self.assertRaisesRegex(PolicyContractError, "non-numeric or fixed"):
            arm.ask(2)


if __name__ == "__main__":
    unittest.main()
