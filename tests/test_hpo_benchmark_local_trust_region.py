from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.hpo_benchmark.arms.local_trust_region import LocalTrustRegionArm
from tools.hpo_benchmark.core import (
    BenchmarkContext,
    BenchmarkRunner,
    FunctionObjective,
    Observation,
    SearchSpace,
    load_arm,
)


class LocalTrustRegionArmTests(unittest.TestCase):
    def test_runner_adapts_radius_after_improvement_and_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            space = SearchSpace.from_legacy({"x": ("float", 0.0, 1.0)})
            initial = Observation(
                observation_id="warm",
                params={"x": 0.8},
                score=1.0,
                status="ok",
                origin="warm",
                consumes_budget=False,
            )
            context = BenchmarkContext(
                checkpoint_id="local-trust-region-toy",
                regime="first",
                space=space,
                observations=(initial,),
                budget=2,
                seed=4,
            )
            scores = iter((0.5, 2.0))
            result = BenchmarkRunner(
                context,
                FunctionObjective(lambda _params: next(scores)),
                Path(tmp) / "run",
            ).run(LocalTrustRegionArm())

            self.assertEqual(result["evaluations_consumed"], 2)
            self.assertEqual(result["final_incumbent_score"], 0.5)
            self.assertAlmostEqual(result["policy_snapshot"]["radius"], 0.1875)
            self.assertEqual(result["policy_snapshot"]["improvement_count"], 1)

            events = [
                json.loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
            ]
            proposals = [
                event for event in events if event["kind"] == "proposal_batch"
            ]
            self.assertEqual(
                [proposal["metadata"]["radius"] for proposal in proposals],
                [0.25, 0.375],
            )
            first = proposals[0]["proposals"][0]
            self.assertLessEqual(abs(first["params"]["x"] - 0.8), 0.25)
            self.assertEqual(first["metadata"]["sampling"], "independent_uniform_box")

    def test_full_budget_run_is_reproducible_for_a_seed(self):
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
                checkpoint_id="local-reproducibility",
                regime="continuation",
                space=space,
                observations=(initial,),
                budget=10,
                seed=11,
            )

            results = []
            proposals = []
            for run_name in ("first", "second"):
                output = Path(tmp) / run_name
                results.append(
                    BenchmarkRunner(
                        context,
                        FunctionObjective(lambda params: params["x"] ** 2),
                        output,
                    ).run(LocalTrustRegionArm())
                )
                events = [
                    json.loads(line)
                    for line in (output / "events.jsonl").read_text().splitlines()
                ]
                proposals.append(
                    [
                        event["proposals"][0]["params"]
                        for event in events
                        if event["kind"] == "proposal_batch"
                    ]
                )

            self.assertEqual(results[0]["evaluations_consumed"], 10)
            self.assertEqual(
                results[0]["best_by_evaluation"],
                results[1]["best_by_evaluation"],
            )
            self.assertEqual(proposals[0], proposals[1])
            self.assertLess(results[0]["final_incumbent_score"], 0.64)

    def test_radius_one_replaces_a_categorical_dimension(self):
        space = SearchSpace.from_legacy(
            {"mode": ("categorical", ["a", "b", "c"])}
        )
        initial = Observation(
            observation_id="warm",
            params={"mode": "a"},
            score=1.0,
            status="ok",
            origin="warm",
            consumes_budget=False,
        )
        context = BenchmarkContext(
            checkpoint_id="categorical-local",
            regime="continuation",
            space=space,
            observations=(initial,),
            budget=1,
            seed=7,
        )
        factory = load_arm(
            "tools.hpo_benchmark.arms.local_trust_region:create_arm"
        )
        arm = factory(initial_radius=1.0)
        arm.initialize(context)

        batch = arm.ask(1)

        self.assertNotEqual(batch.proposals[0].params["mode"], "a")
        self.assertEqual(
            batch.proposals[0].metadata["categorical_replace_probability"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
