from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.hpo_benchmark.arms.current import CurrentArm
from tools.hpo_benchmark.core import (
    BenchmarkContext,
    BenchmarkRunner,
    FunctionObjective,
    Observation,
    PolicyContractError,
    SearchSpace,
)
from tools.hpo_benchmark.providers import ReplayProposalProvider


def checkpoint(*, regime: str, budget: int) -> BenchmarkContext:
    return BenchmarkContext(
        checkpoint_id=f"current-{regime}",
        regime=regime,
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
        budget=budget,
        seed=7,
    )


class FakeBackend:
    def __init__(self, context, proposals):
        self.context = context
        self.proposals = iter(proposals)
        self.outcomes = []

    def ask(self):
        return next(self.proposals)

    def tell(self, outcome):
        self.outcomes.append(outcome)

    def snapshot(self):
        return {"backend": "fake_tpe", "told": len(self.outcomes)}


class CurrentArmTests(unittest.TestCase):
    def test_first_bout_runs_deferred_before_tpe_and_primes_tpe_with_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            created = []

            def factory(context):
                backend = FakeBackend(context, [{"x": 2.0}, {"x": 1.0}])
                created.append(backend)
                return backend

            result = BenchmarkRunner(
                checkpoint(regime="first", budget=3),
                FunctionObjective(lambda params: params["x"] ** 2),
                Path(tmp) / "run",
            ).run(
                CurrentArm(
                    deferred_configs=[{"x": 3.0}],
                    tpe_backend_factory=factory,
                )
            )

            self.assertEqual(result["final_incumbent_score"], 1.0)
            self.assertEqual(result["policy_snapshot"]["queue_evaluated"], 1)
            self.assertEqual(len(created), 1)
            self.assertEqual(
                [observation.params for observation in created[0].context.observations],
                [{"x": 4.0}, {"x": 3.0}],
            )
            events = [
                json.loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
            ]
            origins = [
                event["proposals"][0]["origin"]
                for event in events
                if event["kind"] == "proposal_batch"
            ]
            self.assertEqual(
                origins, ["current_deferred", "current_tpe", "current_tpe"]
            )
            tpe_metadata = next(
                event["proposals"][0]["metadata"]
                for event in events
                if event["kind"] == "proposal_batch"
                and event["proposals"][0]["origin"] == "current_tpe"
            )
            self.assertEqual(tpe_metadata["n_startup_trials"], 8)
            self.assertTrue(tpe_metadata["multivariate"])
            self.assertTrue(tpe_metadata["group"])

    def test_continuation_calls_fresh_provider_once_then_enters_tpe(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = ReplayProposalProvider(
                [
                    {
                        "proposals": [
                            {"params": {"x": 3.0}, "reason": "near incumbent"},
                            {"params": {"x": 2.0}, "reason": "larger step"},
                            {"params": {"x": 1.0}, "reason": "probe basin"},
                        ]
                    }
                ]
            )
            created = []

            def factory(context):
                backend = FakeBackend(context, [{"x": 0.0}])
                created.append(backend)
                return backend

            result = BenchmarkRunner(
                checkpoint(regime="continuation", budget=4),
                FunctionObjective(lambda params: params["x"] ** 2),
                Path(tmp) / "run",
            ).run(
                CurrentArm(provider=provider, tpe_backend_factory=factory)
            )

            self.assertEqual(result["final_incumbent_score"], 0.0)
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(result["policy_snapshot"]["provider_calls"], 1)
            self.assertEqual(result["policy_snapshot"]["queue_evaluated"], 3)
            self.assertEqual(len(created[0].context.observations), 4)
            prompt = json.loads(provider.calls[0]["prompt"].split("\n\n", 1)[1])
            self.assertEqual(prompt["regime"], "continuation")
            self.assertEqual(prompt["remaining_budget"], 4)

    def test_broken_rewarm_skips_to_tpe(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = ReplayProposalProvider([{"proposals": "garbage"}] * 4)
            created = []

            def factory(context):
                backend = FakeBackend(context, [{"x": 0.0}, {"x": -1.0}])
                created.append(backend)
                return backend

            result = BenchmarkRunner(
                checkpoint(regime="continuation", budget=2),
                FunctionObjective(lambda params: params["x"] ** 2),
                Path(tmp) / "run",
            ).run(CurrentArm(provider=provider, tpe_backend_factory=factory))

            self.assertEqual(result["evaluations_consumed"], 2)
            self.assertEqual(len(provider.calls), 4)
            self.assertIn("must be a list", provider.calls[1]["prompt"])
            self.assertTrue(result["policy_snapshot"]["rewarm_degraded"])
            self.assertEqual(result["policy_snapshot"]["queue_evaluated"], 0)
            self.assertEqual(result["final_incumbent_score"], 0.0)
            events = [
                json.loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text().splitlines()
            ]
            origins = [
                event["proposals"][0]["origin"]
                for event in events
                if event["kind"] == "proposal_batch"
            ]
            self.assertEqual(origins, ["current_tpe", "current_tpe"])

    def test_continuation_requires_rewarm_provider(self):
        arm = CurrentArm()
        with self.assertRaisesRegex(
            PolicyContractError, "require.*rewarm proposal provider"
        ):
            arm.initialize(checkpoint(regime="continuation", budget=1))


if __name__ == "__main__":
    unittest.main()
