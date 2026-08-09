from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.hpo_benchmark.arms.spsa import SPSAArm
from tools.hpo_benchmark.core import (
    BenchmarkContext,
    BenchmarkRunner,
    EvaluationOutcome,
    FunctionObjective,
    Observation,
    PolicyContractError,
    SearchSpace,
    load_arm,
)


def outcome(
    observation_id: str, params: dict, score: float, incumbent_before: str
) -> EvaluationOutcome:
    observation = Observation(
        observation_id=observation_id,
        params=params,
        score=score,
        status="ok",
        origin="spsa",
    )
    return EvaluationOutcome(
        observation=observation,
        incumbent_before_id=incumbent_before,
        incumbent_after_id=observation_id,
    )


class SPSAArmTests(unittest.TestCase):
    def test_pair_updates_center_and_holds_noncontinuous_dimensions_fixed(self):
        space = SearchSpace.from_legacy(
            {
                "x": ("float", 0.0, 1.0),
                "depth": ("int", 1, 5),
                "mode": ("categorical", ["a", "b"]),
            }
        )
        initial = Observation(
            observation_id="warm",
            params={"x": 0.75, "depth": 3, "mode": "a"},
            score=0.75**2,
            status="ok",
            origin="warm",
            consumes_budget=False,
        )
        context = BenchmarkContext(
            checkpoint_id="spsa-step",
            regime="first",
            space=space,
            observations=(initial,),
            budget=4,
            seed=2,
        )
        arm = SPSAArm()
        arm.initialize(context)

        batch = arm.ask(4)

        self.assertTrue(batch.atomic)
        self.assertEqual(len(batch.proposals), 2)
        self.assertEqual(
            sorted(proposal.params["x"] for proposal in batch.proposals),
            [0.65, 0.85],
        )
        self.assertTrue(
            all(proposal.params["depth"] == 3 for proposal in batch.proposals)
        )
        self.assertTrue(
            all(proposal.params["mode"] == "a" for proposal in batch.proposals)
        )

        plus, minus = batch.proposals
        arm.tell(
            (
                outcome("plus", dict(plus.params), plus.params["x"] ** 2, "warm"),
                outcome("minus", dict(minus.params), minus.params["x"] ** 2, "warm"),
            )
        )

        self.assertAlmostEqual(arm.snapshot()["center_params"]["x"], 0.675)
        self.assertEqual(arm.snapshot()["center_params"]["depth"], 3)
        self.assertEqual(arm.snapshot()["center_params"]["mode"], "a")

    def test_runner_consumes_atomic_pairs_and_records_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            space = SearchSpace.from_legacy({"x": ("float", -1.0, 1.0)})
            initial = Observation(
                observation_id="warm",
                params={"x": 0.8},
                score=0.64,
                status="ok",
                origin="warm",
                consumes_budget=False,
            )
            context = BenchmarkContext(
                checkpoint_id="spsa-runner",
                regime="continuation",
                space=space,
                observations=(initial,),
                budget=10,
                seed=3,
            )
            factory = load_arm("tools.hpo_benchmark.arms.spsa:create_arm")
            result = BenchmarkRunner(
                context,
                FunctionObjective(lambda params: params["x"] ** 2),
                Path(tmp) / "run",
            ).run(factory())

            self.assertEqual(result["evaluations_consumed"], 10)
            self.assertEqual(result["proposal_batches"], 5)
            self.assertEqual(result["policy_snapshot"]["completed_rounds"], 5)
            self.assertLess(result["final_incumbent_score"], 0.64)

            events = [
                json.loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
            ]
            proposals = [
                event for event in events if event["kind"] == "proposal_batch"
            ]
            self.assertTrue(all(proposal["atomic"] for proposal in proposals))
            self.assertEqual(proposals[0]["metadata"]["a_k"], 0.05)
            self.assertEqual(proposals[0]["metadata"]["c_k"], 0.1)

    def test_preflight_rejection_retries_with_a_smaller_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            space = SearchSpace.from_legacy({"x": ("float", 0.0, 1.0)})
            initial = Observation(
                observation_id="warm",
                params={"x": 0.5},
                score=0.25,
                status="ok",
                origin="warm",
                consumes_budget=False,
            )
            context = BenchmarkContext(
                checkpoint_id="spsa-preflight-retry",
                regime="first",
                space=space,
                observations=(initial,),
                budget=2,
                seed=0,
            )
            result = BenchmarkRunner(
                context,
                FunctionObjective(
                    lambda params: params["x"] ** 2,
                    preflight_fn=lambda params: (
                        "outside local feasible region"
                        if abs(params["x"] - 0.5) > 0.075
                        else None
                    ),
                ),
                Path(tmp) / "run",
            ).run(SPSAArm())

            self.assertEqual(result["evaluations_consumed"], 2)
            self.assertEqual(result["proposal_batches"], 2)
            self.assertEqual(result["policy_snapshot"]["rejected_batches"], 1)
            events = [
                json.loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
            ]
            proposals = [
                event for event in events if event["kind"] == "proposal_batch"
            ]
            self.assertEqual(
                [proposal["metadata"]["c_k"] for proposal in proposals],
                [0.1, 0.05],
            )
            self.assertEqual(
                [proposal["metadata"]["retry"] for proposal in proposals],
                [0, 1],
            )

    def test_requires_even_budget_and_a_continuous_dimension(self):
        initial_float = Observation(
            observation_id="warm",
            params={"x": 0.5},
            score=0.25,
            status="ok",
            origin="warm",
            consumes_budget=False,
        )
        odd = BenchmarkContext(
            checkpoint_id="odd",
            regime="first",
            space=SearchSpace.from_legacy({"x": ("float", 0.0, 1.0)}),
            observations=(initial_float,),
            budget=3,
            seed=0,
        )
        with self.assertRaisesRegex(PolicyContractError, "even evaluation budget"):
            SPSAArm().initialize(odd)

        initial_int = Observation(
            observation_id="warm",
            params={"x": 1},
            score=1.0,
            status="ok",
            origin="warm",
            consumes_budget=False,
        )
        integers = BenchmarkContext(
            checkpoint_id="integers",
            regime="first",
            space=SearchSpace.from_legacy({"x": ("int", 0, 2)}),
            observations=(initial_int,),
            budget=2,
            seed=0,
        )
        with self.assertRaisesRegex(PolicyContractError, "continuous dimension"):
            SPSAArm().initialize(integers)


if __name__ == "__main__":
    unittest.main()
