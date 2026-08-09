from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.hpo_benchmark.arms.hebo_rank import (
    LLMPoolHEBORankArm,
    nondominated_fronts,
)
from tools.hpo_benchmark.arms.pool_rank import PoolRanking
from tools.hpo_benchmark.arms.tpe_rank import (
    LLMPoolTPERankArm,
    TPEDensityRatioRanker,
)
from tools.hpo_benchmark.core import (
    BenchmarkContext,
    BenchmarkRunner,
    FunctionObjective,
    Observation,
    SearchSpace,
    load_arm,
)
from tools.hpo_benchmark.providers import ReplayProposalProvider


def history(count: int) -> tuple[Observation, ...]:
    return tuple(
        Observation(
            observation_id=f"warm-{index}",
            params={"x": float(index)},
            score=float(index**2),
            status="ok",
            origin="warm",
            consumes_budget=False,
        )
        for index in range(count)
    )


def replay_pool(values: list[float]) -> dict:
    return {
        "proposals": [
            {"params": {"x": value}, "reason": f"try {value}"}
            for value in values
        ]
    }


class FixedRanker:
    name = "fixed_test_ranker"

    def __init__(self, order: tuple[int, ...]):
        self.order = order
        self.calls = 0

    def rank(self, space, observations, pool, *, seed):
        self.calls += 1
        return PoolRanking(
            order=self.order,
            member_scores=tuple({"fixed_score": index} for index in range(len(pool))),
        )


class PoolRankArmTests(unittest.TestCase):
    def setUp(self):
        self.space = SearchSpace.from_legacy({"x": ("float", 0.0, 10.0)})

    def context(self, count: int, *, budget: int = 1) -> BenchmarkContext:
        return BenchmarkContext(
            checkpoint_id="pool-rank-toy",
            regime="continuation",
            space=self.space,
            observations=history(count),
            budget=budget,
            seed=17,
        )

    def test_tpe_and_hebo_share_prompt_and_llm_order_before_threshold(self):
        output = replay_pool([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
        tpe_provider = ReplayProposalProvider([output])
        hebo_provider = ReplayProposalProvider([output])
        context = self.context(1)
        tpe = LLMPoolTPERankArm(tpe_provider)
        hebo = LLMPoolHEBORankArm(hebo_provider)

        tpe.initialize(context)
        hebo.initialize(context)
        tpe_batch = tpe.ask(1)
        hebo_batch = hebo.ask(1)

        self.assertEqual(tpe_batch.proposals[0].params, {"x": 0.5})
        self.assertEqual(hebo_batch.proposals[0].params, {"x": 0.5})
        self.assertEqual(tpe_provider.calls[0]["prompt"], hebo_provider.calls[0]["prompt"])
        self.assertEqual(tpe_batch.metadata["selection_mode"], "llm_order")
        self.assertIsNone(tpe_batch.proposals[0].metadata["pool"][0]["ranker_scores"])

    def test_explicit_tpe_density_ratio_prefers_the_good_region(self):
        ranker = TPEDensityRatioRanker()
        pool = [{"x": value} for value in (0.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5)]

        ranking = ranker.rank(self.space, history(8), pool, seed=0)

        self.assertEqual(ranking.order[0], 0)
        self.assertGreater(
            ranking.member_scores[0]["log_density_ratio"],
            ranking.member_scores[1]["log_density_ratio"],
        )

    def test_ranker_turns_on_after_the_eighth_effective_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = ReplayProposalProvider(
                [
                    replay_pool([7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2]),
                    replay_pool([8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0]),
                ]
            )
            ranker = FixedRanker((2, 0, 1, 3, 4, 5, 6, 7))
            result = BenchmarkRunner(
                self.context(7, budget=2),
                FunctionObjective(lambda params: params["x"] ** 2),
                Path(tmp) / "threshold",
            ).run(LLMPoolHEBORankArm(provider, ranker=ranker))

            self.assertEqual(result["evaluations_consumed"], 2)
            self.assertEqual(result["policy_snapshot"]["provider_calls"], 2)
            self.assertEqual(result["policy_snapshot"]["ranker_calls"], 1)
            self.assertEqual(ranker.calls, 1)
            events = [
                json.loads(line)
                for line in (Path(tmp) / "threshold" / "events.jsonl").read_text().splitlines()
            ]
            batches = [event for event in events if event["kind"] == "proposal_batch"]
            self.assertEqual(
                [event["metadata"]["selection_mode"] for event in batches],
                ["llm_order", "fixed_test_ranker"],
            )
            self.assertEqual(batches[1]["proposals"][0]["params"], {"x": 8.5})

    def test_ranked_pool_is_recorded_and_reused_after_preflight_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = replay_pool([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
            provider = ReplayProposalProvider([output])
            ranker = FixedRanker((3, 1, 0, 2, 4, 5, 6, 7))
            arm = LLMPoolHEBORankArm(provider, ranker=ranker)
            result = BenchmarkRunner(
                self.context(8),
                FunctionObjective(
                    lambda params: params["x"] ** 2,
                    preflight_fn=lambda params: (
                        "synthetic rejection" if params["x"] == 3.5 else None
                    ),
                ),
                Path(tmp) / "run",
            ).run(arm)

            self.assertEqual(result["evaluations_consumed"], 1)
            self.assertEqual(result["rejection_count"], 1)
            self.assertEqual(result["proposal_batches"], 2)
            self.assertEqual(result["policy_snapshot"]["provider_calls"], 1)
            self.assertEqual(result["policy_snapshot"]["ranker_calls"], 1)
            self.assertEqual(ranker.calls, 1)

            events = [
                json.loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
            ]
            batches = [event for event in events if event["kind"] == "proposal_batch"]
            self.assertEqual(
                [event["proposals"][0]["params"] for event in batches],
                [{"x": 3.5}, {"x": 1.5}],
            )
            self.assertEqual(batches[0]["metadata"]["pool_id"], batches[1]["metadata"]["pool_id"])
            pool = batches[0]["proposals"][0]["metadata"]["pool"]
            self.assertEqual(len(pool), 8)
            self.assertEqual(pool[3]["ranker_scores"], {"fixed_score": 3})
            self.assertEqual(batches[1]["metadata"]["selected_rank"], 2)

    def test_hebo_pareto_fronts_and_factories_are_deterministic(self):
        self.assertEqual(
            nondominated_fronts(
                ((0.0, 2.0, 2.0), (1.0, 1.0, 1.0), (2.0, 0.0, 2.0), (3.0, 3.0, 3.0))
            ),
            (0, 0, 0, 1),
        )
        self.assertTrue(
            callable(load_arm("tools.hpo_benchmark.arms.tpe_rank:create_arm"))
        )
        self.assertTrue(
            callable(load_arm("tools.hpo_benchmark.arms.hebo_rank:create_arm"))
        )


if __name__ == "__main__":
    unittest.main()
