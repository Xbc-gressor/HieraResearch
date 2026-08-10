from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.hpo_benchmark.arms.hillclimb import LLMHillclimbArm
from tools.hpo_benchmark.core import (
    BenchmarkContext,
    BenchmarkRunner,
    FunctionObjective,
    Observation,
    SearchSpace,
)
from tools.hpo_benchmark.providers import ReplayProposalProvider
from tools.hpo_benchmark.summary import FocusedSummaryBuilder


class HillclimbArmTests(unittest.TestCase):
    def test_fresh_calls_follow_the_improving_incumbent(self):
        with tempfile.TemporaryDirectory() as tmp:
            space = SearchSpace.from_legacy(
                {"x": ("float", -5.0, 5.0), "y": ("int", -5, 5)}
            )
            initial = Observation(
                observation_id="warm",
                params={"x": 4.0, "y": -3},
                score=25.0,
                status="ok",
                origin="warm",
                consumes_budget=False,
            )
            context = BenchmarkContext(
                checkpoint_id="hillclimb-toy",
                regime="first",
                space=space,
                observations=(initial,),
                budget=3,
                seed=0,
            )
            provider = ReplayProposalProvider(
                [
                    {"changes": {"x": 2.0}, "reason": "reduce x"},
                    {"changes": {"y": -1}, "reason": "reduce y"},
                    {"changes": {"x": 0.0}, "reason": "finish x"},
                ]
            )
            result = BenchmarkRunner(
                context,
                FunctionObjective(lambda params: params["x"] ** 2 + params["y"] ** 2),
                Path(tmp) / "run",
            ).run(LLMHillclimbArm(provider))

            self.assertEqual(result["final_incumbent_score"], 1.0)
            self.assertEqual(result["first_improvement_evaluation"], 1)
            self.assertEqual(len(provider.calls), 3)
            prompts = [json.loads(call["prompt"].split("\n\n", 1)[1]) for call in provider.calls]
            self.assertEqual([prompt["remaining_budget"] for prompt in prompts], [3, 2, 1])
            self.assertEqual(prompts[1]["incumbent"]["params"], {"x": 2.0, "y": -3})
            self.assertEqual(prompts[2]["incumbent"]["params"], {"x": 2.0, "y": -1})

            events = [
                json.loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
            ]
            proposal = next(event for event in events if event["kind"] == "proposal_batch")
            metadata = proposal["proposals"][0]["metadata"]
            self.assertEqual(metadata["model"], "replay")
            self.assertIn("raw_output", metadata)

    def test_focused_summary_caps_history(self):
        space = SearchSpace.from_legacy({"x": ("float", 0.0, 20.0)})
        observations = tuple(
            Observation(
                observation_id=f"obs-{index}",
                params={"x": float(index)},
                score=float(20 - index),
                status="ok",
                origin="history",
                consumes_budget=False,
            )
            for index in range(12)
        )
        context = BenchmarkContext(
            checkpoint_id="summary-toy",
            regime="continuation",
            space=space,
            observations=observations,
            budget=10,
            seed=0,
        )
        summary = FocusedSummaryBuilder().build(
            context, observations, remaining_budget=10
        )
        self.assertLessEqual(len(summary["focused_history"]), 8)
        self.assertEqual(summary["incumbent"]["observation_id"], "obs-11")
        self.assertEqual(summary["search_space"][0]["name"], "x")

    def test_focused_summary_drops_partial_recent_rows(self):
        space = SearchSpace.from_legacy(
            {"x": ("float", 0.0, 10.0), "y": ("int", 0, 10)}
        )
        full = Observation(
            observation_id="full",
            params={"x": 1.0, "y": 1},
            score=2.0,
            status="ok",
            origin="history",
            consumes_budget=False,
        )
        partial = Observation(
            observation_id="partial-recent",
            params={"x": 2.0},
            score=3.0,
            status="ok",
            origin="old-space",
            consumes_budget=False,
        )
        context = BenchmarkContext(
            checkpoint_id="heterogeneous-history",
            regime="continuation",
            space=space,
            observations=(full, partial),
            budget=10,
            seed=0,
        )

        summary = FocusedSummaryBuilder().build(
            context, context.observations, remaining_budget=10
        )

        self.assertEqual(summary["incumbent"]["observation_id"], "full")
        self.assertNotIn(
            "partial-recent",
            {row["observation_id"] for row in summary["focused_history"]},
        )

    def test_focused_summary_includes_projectable_preflight_rejections(self):
        space = SearchSpace.from_legacy({"x": ("float", 0.0, 10.0)})
        initial = Observation(
            observation_id="warm",
            params={"x": 5.0},
            score=1.0,
            status="ok",
            origin="history",
            consumes_budget=False,
        )
        rejected = Observation(
            observation_id="preflight-rejected",
            params={"x": 7.0},
            score=None,
            status="rejected",
            origin="llm_hillclimb",
            eligible_incumbent=False,
            consumes_budget=False,
            failure="synthetic resource rejection",
        )
        context = BenchmarkContext(
            checkpoint_id="rejection-summary",
            regime="first",
            space=space,
            observations=(initial,),
            budget=10,
            seed=0,
        )

        summary = FocusedSummaryBuilder().build(
            context, (initial, rejected), remaining_budget=10
        )

        row = next(
            item
            for item in summary["focused_history"]
            if item["observation_id"] == "preflight-rejected"
        )
        self.assertEqual(row["status"], "rejected")
        self.assertEqual(row["failure"], "synthetic resource rejection")

    def test_hillclimb_uses_full_space_incumbent_when_global_best_is_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            space = SearchSpace.from_legacy(
                {"x": ("float", 0.0, 10.0), "y": ("int", 0, 10)}
            )
            partial_best = Observation(
                observation_id="partial-best",
                params={"x": 0.0},
                score=0.0,
                status="ok",
                origin="old-space",
                consumes_budget=False,
            )
            full = Observation(
                observation_id="full",
                params={"x": 4.0, "y": 4},
                score=32.0,
                status="ok",
                origin="history",
                consumes_budget=False,
            )
            context = BenchmarkContext(
                checkpoint_id="partial-incumbent",
                regime="continuation",
                space=space,
                observations=(partial_best, full),
                budget=1,
                seed=0,
            )
            provider = ReplayProposalProvider(
                [{"changes": {"x": 3.0}, "reason": "reduce x"}]
            )

            result = BenchmarkRunner(
                context,
                FunctionObjective(
                    lambda params: params["x"] ** 2 + params["y"] ** 2
                ),
                Path(tmp) / "run",
            ).run(LLMHillclimbArm(provider))

            self.assertEqual(result["evaluations_consumed"], 1)
            prompt = json.loads(provider.calls[0]["prompt"].split("\n\n", 1)[1])
            self.assertEqual(prompt["incumbent"]["observation_id"], "full")
            events = [
                json.loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
            ]
            proposal = next(
                event for event in events if event["kind"] == "proposal_batch"
            )
            self.assertEqual(
                proposal["proposals"][0]["params"],
                {"x": 3.0, "y": 4},
            )


    def test_provider_echoing_summary_keys_is_repaired_by_corrective_ask(self):
        with tempfile.TemporaryDirectory() as tmp:
            space = SearchSpace.from_legacy({"x": ("float", -5.0, 5.0)})
            initial = Observation(
                observation_id="warm",
                params={"x": 4.0},
                score=16.0,
                status="ok",
                origin="warm",
                consumes_budget=False,
            )
            context = BenchmarkContext(
                checkpoint_id="hillclimb-repair",
                regime="first",
                space=space,
                observations=(initial,),
                budget=1,
                seed=0,
            )
            provider = ReplayProposalProvider(
                [
                    {
                        "changes": {
                            "checkpoint_id": "hillclimb-repair",
                            "focused_history": [],
                            "incumbent": {"params": {"x": 4.0}},
                            "regime": "first",
                            "remaining_budget": 1,
                            "search_space": [],
                        },
                        "reason": "echoed the summary",
                    },
                    {"changes": {"x": 2.0}, "reason": "reduce x"},
                ]
            )
            result = BenchmarkRunner(
                context,
                FunctionObjective(lambda params: params["x"] ** 2),
                Path(tmp) / "run",
            ).run(LLMHillclimbArm(provider))

            self.assertEqual(result["evaluations_consumed"], 1)
            self.assertEqual(len(provider.calls), 2)
            self.assertIn("not search-space dimensions", provider.calls[1]["prompt"])
            events = [
                json.loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
            ]
            proposal = next(
                event for event in events if event["kind"] == "proposal_batch"
            )["proposals"][0]
            self.assertEqual(proposal["params"], {"x": 2.0})
            self.assertFalse(proposal["metadata"]["degraded"])

    def test_broken_provider_degrades_to_a_random_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            space = SearchSpace.from_legacy({"x": ("float", -5.0, 5.0)})
            initial = Observation(
                observation_id="warm",
                params={"x": 4.0},
                score=16.0,
                status="ok",
                origin="warm",
                consumes_budget=False,
            )
            context = BenchmarkContext(
                checkpoint_id="hillclimb-degraded",
                regime="first",
                space=space,
                observations=(initial,),
                budget=1,
                seed=0,
            )
            provider = ReplayProposalProvider(
                [{"changes": {"not_a_dim": 1.0}, "reason": "bad"}] * 4
            )
            result = BenchmarkRunner(
                context,
                FunctionObjective(lambda params: params["x"] ** 2),
                Path(tmp) / "run",
            ).run(LLMHillclimbArm(provider))

            self.assertEqual(result["evaluations_consumed"], 1)
            self.assertEqual(len(provider.calls), 4)
            self.assertEqual(result["policy_snapshot"]["degraded_calls"], 1)
            events = [
                json.loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
            ]
            proposal = next(
                event for event in events if event["kind"] == "proposal_batch"
            )["proposals"][0]
            self.assertEqual(set(proposal["params"]), {"x"})
            self.assertNotEqual(proposal["params"], {"x": 4.0})
            self.assertTrue(proposal["metadata"]["degraded"])
            self.assertIn("degraded fallback", proposal["metadata"]["reason"])


if __name__ == "__main__":
    unittest.main()
