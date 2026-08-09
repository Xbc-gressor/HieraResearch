from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.hpo_benchmark.arms.gp_rank import (
    GPEIRanker,
    LLMPoolGPRankArm,
    NormalizedOneHotEncoder,
)
from tools.hpo_benchmark.arms.pool_rank import PoolRanking
from tools.hpo_benchmark.arms.smac_rank import LLMPoolSMACRankArm
from tools.hpo_benchmark.core import (
    BenchmarkContext,
    BenchmarkRunner,
    FunctionObjective,
    Observation,
    SearchSpace,
    load_arm,
)
from tools.hpo_benchmark.providers import ReplayProposalProvider


def replay_pool(values) -> dict:
    return {
        "proposals": [
            {"params": {"x": float(value)}, "reason": f"try {value}"}
            for value in values
        ]
    }


class FixedRanker:
    name = "fixed_ei"

    def rank(self, space, observations, pool, *, seed):
        del space, observations, seed
        order = (3, 0, 1, 2, 4, 5, 6, 7)
        return PoolRanking(
            order=order,
            member_scores=tuple(
                {
                    "expected_improvement": 10.0 if index == 3 else 0.0,
                    "acquisition_rank": order.index(index) + 1,
                }
                for index in range(len(pool))
            ),
        )


class SMACAndGPRankArmTests(unittest.TestCase):
    def history(self) -> tuple[Observation, ...]:
        return tuple(
            Observation(
                observation_id=f"warm-{index}",
                params={"x": float(index)},
                score=float((index - 3) ** 2),
                status="ok",
                origin="warm",
                consumes_budget=False,
            )
            for index in range(8)
        )

    def context(self) -> BenchmarkContext:
        return BenchmarkContext(
            checkpoint_id="smac-gp-rank-toy",
            regime="continuation",
            space=SearchSpace.from_legacy({"x": ("float", 0.0, 20.0)}),
            observations=self.history(),
            budget=1,
            seed=13,
        )

    def test_both_arms_execute_only_the_ranked_pool_winner(self):
        for arm_type in (LLMPoolSMACRankArm, LLMPoolGPRankArm):
            with self.subTest(arm=arm_type.__name__), tempfile.TemporaryDirectory() as tmp:
                provider = ReplayProposalProvider([replay_pool(range(8, 16))])
                result = BenchmarkRunner(
                    self.context(),
                    FunctionObjective(lambda params: params["x"] ** 2),
                    Path(tmp) / "run",
                ).run(arm_type(provider, ranker=FixedRanker()))

                self.assertEqual(result["evaluations_consumed"], 1)
                self.assertEqual(result["policy_snapshot"]["ranker_calls"], 1)
                events = [
                    json.loads(line)
                    for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
                ]
                batch = next(
                    event for event in events if event["kind"] == "proposal_batch"
                )
                self.assertEqual(batch["proposals"][0]["params"], {"x": 11.0})
                evidence = batch["proposals"][0]["metadata"]
                self.assertEqual(len(evidence["pool"]), 8)
                self.assertEqual(evidence["selected_pool_index"], 3)
                self.assertEqual(
                    evidence["pool"][3]["ranker_scores"]["acquisition_rank"],
                    1,
                )

    def test_gp_uses_normalized_numeric_one_hot_and_matern_ei(self):
        space = SearchSpace.from_legacy(
            {
                "x": ("float", 0.0, 10.0),
                "scale": ("float", 1e-2, 1e2, "log"),
                "kind": ("categorical", ["a", "b"]),
            }
        )
        encoder = NormalizedOneHotEncoder(space)
        encoded = encoder.encode({"x": 5.0, "scale": 1.0, "kind": "b"})
        self.assertAlmostEqual(encoded[0], 0.5)
        self.assertAlmostEqual(encoded[1], 0.5)
        self.assertEqual(encoded[2:], [0.0, 1.0])

        observations = tuple(
            Observation(
                observation_id=f"gp-{index}",
                params={
                    "x": float(index),
                    "scale": 10.0 ** ((index - 4) / 2),
                    "kind": "a" if index % 2 == 0 else "b",
                },
                score=float((index - 3) ** 2 + index % 2),
                status="ok",
                origin="history",
                consumes_budget=False,
            )
            for index in range(8)
        )
        pool = tuple(
            {
                "x": index + 0.25,
                "scale": 10.0 ** ((index - 4) / 2),
                "kind": "b" if index % 2 == 0 else "a",
            }
            for index in range(8)
        )

        ranking = GPEIRanker().rank(space, observations, pool, seed=5)

        self.assertEqual(set(ranking.order), set(range(8)))
        self.assertEqual(
            ranking.member_scores[ranking.order[0]]["acquisition_rank"], 1
        )
        self.assertTrue(
            all(
                member["kernel"] == "matern_5_2"
                and member["categorical_encoding"] == "one_hot"
                and math.isfinite(member["expected_improvement"])
                and member["expected_improvement"] >= 0.0
                and member["predicted_std"] >= 0.0
                for member in ranking.member_scores
            )
        )

    def test_factories_load_without_optional_smac_dependencies(self):
        for spec in (
            "tools.hpo_benchmark.arms.smac_rank:create_arm",
            "tools.hpo_benchmark.arms.gp_rank:create_arm",
        ):
            self.assertTrue(callable(load_arm(spec)))


if __name__ == "__main__":
    unittest.main()
