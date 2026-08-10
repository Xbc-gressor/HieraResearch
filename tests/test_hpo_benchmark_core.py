from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.hpo_benchmark.core import (
    BenchmarkContext,
    BenchmarkRunner,
    ConfigInfeasibleError,
    EvaluationOutcome,
    FunctionObjective,
    Observation,
    PolicyContractError,
    Proposal,
    ProposalBatch,
    SearchSpace,
    load_arm,
)


class ScriptedArm:
    name = "scripted"

    def __init__(self, batches: list[ProposalBatch]):
        self.batches = list(batches)
        self.outcomes: list[EvaluationOutcome] = []

    def initialize(self, context: BenchmarkContext) -> None:
        self.context = context

    def ask(self, remaining_budget: int) -> ProposalBatch:
        return self.batches.pop(0)

    def tell(self, outcomes: tuple[EvaluationOutcome, ...]) -> None:
        self.outcomes.extend(outcomes)

    def snapshot(self) -> dict:
        return {"outcomes": len(self.outcomes)}


def observation(
    observation_id: str,
    x: float,
    score: float,
    *,
    eligible_incumbent: bool = True,
) -> Observation:
    return Observation(
        observation_id=observation_id,
        params={"x": x},
        score=score,
        status="ok",
        origin="warm",
        eligible_incumbent=eligible_incumbent,
        consumes_budget=False,
    )


class SearchSpaceTests(unittest.TestCase):
    def test_legacy_space_projects_numeric_values_and_rejects_bad_categories(self):
        space = SearchSpace.from_legacy(
            {
                "rate": ("float", 0.001, 0.1, "log"),
                "depth": ("int", 2, 8),
                "mode": ("categorical", ["a", "b"]),
            }
        )
        self.assertEqual(
            space.project({"rate": 1.0, "depth": 6.7, "mode": "b"}),
            {"rate": 0.1, "depth": 7, "mode": "b"},
        )
        with self.assertRaisesRegex(ValueError, "expects one of"):
            space.project({"rate": 0.01, "depth": 4, "mode": "c"})


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.space = SearchSpace.from_legacy({"x": ("float", -5.0, 5.0)})

    def tearDown(self):
        self.tmp.cleanup()

    def _context(self, history: tuple[Observation, ...], *, budget: int = 2):
        return BenchmarkContext(
            checkpoint_id="toy",
            regime="first",
            space=self.space,
            observations=history,
            budget=budget,
            seed=3,
        )

    def test_lower_score_wins_and_ineligible_control_cannot_be_incumbent(self):
        history = (
            observation("inherited-control", 0.0, -100.0, eligible_incumbent=False),
            observation("warm", 4.0, 16.0),
        )
        arm = ScriptedArm(
            [
                ProposalBatch((Proposal({"x": 3.0}, "scripted"),)),
                ProposalBatch((Proposal({"x": 2.0}, "scripted"),)),
            ]
        )
        result = BenchmarkRunner(
            self._context(history),
            FunctionObjective(lambda params: params["x"] ** 2),
            Path(self.tmp.name) / "run",
        ).run(arm)

        self.assertEqual(result["initial_incumbent_score"], 16.0)
        self.assertEqual(result["final_incumbent_score"], 4.0)
        self.assertEqual(result["evaluations_consumed"], 2)
        self.assertEqual(result["improvement_at"], {"2": 12.0})

    def test_preflight_rejection_does_not_consume_but_crash_does(self):
        history = (observation("warm", 4.0, 16.0),)
        arm = ScriptedArm(
            [
                ProposalBatch(
                    (
                        Proposal({"x": -1.0}, "scripted"),
                        Proposal({"x": 3.0}, "scripted"),
                    )
                ),
                ProposalBatch((Proposal({"x": 2.0}, "scripted"),)),
            ]
        )

        def evaluate(params):
            if params["x"] == 3.0:
                raise ConfigInfeasibleError("synthetic config crash")
            return params["x"] ** 2

        result = BenchmarkRunner(
            self._context(history),
            FunctionObjective(
                evaluate,
                preflight_fn=lambda params: "negative x rejected" if params["x"] < 0 else None,
            ),
            Path(self.tmp.name) / "run",
        ).run(arm)

        self.assertEqual(result["evaluations_consumed"], 2)
        self.assertEqual(result["final_incumbent_score"], 4.0)
        self.assertEqual(result["crash_count"], 1)
        self.assertEqual(result["rejection_count"], 1)
        statuses = [outcome.observation.status for outcome in arm.outcomes]
        self.assertEqual(statuses, ["rejected", "crash", "ok"])
        self.assertTrue(math.isinf(arm.outcomes[1].observation.score))
        events = [
            json.loads(line)
            for line in (Path(self.tmp.name) / "run" / "events.jsonl").read_text().splitlines()
        ]
        crash = next(
            event for event in events
            if event["kind"] == "evaluation" and event["observation"]["status"] == "crash"
        )
        self.assertEqual(crash["observation"]["score"], "+inf")

    def test_unknown_objective_errors_propagate_and_leave_failure_receipt(self):
        history = (observation("warm", 4.0, 16.0),)

        def fail_preflight(params):
            raise RuntimeError("unknown preflight failure")

        def fail_evaluation(params):
            raise RuntimeError("unknown evaluation failure")

        cases = (
            (
                "preflight",
                FunctionObjective(lambda params: 1.0, preflight_fn=fail_preflight),
            ),
            ("evaluation", FunctionObjective(fail_evaluation)),
        )
        for name, objective in cases:
            with self.subTest(name=name):
                output = Path(self.tmp.name) / name
                arm = ScriptedArm(
                    [ProposalBatch((Proposal({"x": 3.0}, "scripted"),))]
                )
                with self.assertRaisesRegex(RuntimeError, f"unknown {name} failure"):
                    BenchmarkRunner(
                        self._context(history, budget=1), objective, output
                    ).run(arm)

                failure = json.loads(
                    (output / "failure.json").read_text(encoding="utf-8")
                )
                self.assertEqual(failure["error_type"], "RuntimeError")
                self.assertEqual(failure["error"], f"unknown {name} failure")
                self.assertFalse((output / "result.json").exists())

    def test_atomic_batch_is_not_partially_evaluated(self):
        history = (observation("warm", 4.0, 16.0),)
        arm = ScriptedArm(
            [
                ProposalBatch(
                    (
                        Proposal({"x": 3.0}, "scripted"),
                        Proposal({"unknown": 2.0}, "scripted"),
                    ),
                    atomic=True,
                )
            ]
        )
        calls = []
        with self.assertRaisesRegex(PolicyContractError, "no budget-consuming progress"):
            BenchmarkRunner(
                self._context(history),
                FunctionObjective(lambda params: calls.append(params) or 1.0),
                Path(self.tmp.name) / "atomic",
            ).run(arm)
        self.assertEqual(calls, [])
        self.assertEqual(
            [outcome.observation.status for outcome in arm.outcomes],
            ["rejected", "rejected"],
        )

    def test_preflight_only_batch_is_reported_then_arm_can_retry(self):
        history = (observation("warm", 4.0, 16.0),)
        arm = ScriptedArm(
            [
                ProposalBatch((Proposal({"x": -1.0}, "scripted"),)),
                ProposalBatch((Proposal({"x": 3.0}, "scripted"),)),
            ]
        )
        result = BenchmarkRunner(
            self._context(history, budget=1),
            FunctionObjective(
                lambda params: params["x"] ** 2,
                preflight_fn=lambda params: "negative x rejected" if params["x"] < 0 else None,
            ),
            Path(self.tmp.name) / "retry",
        ).run(arm)

        self.assertEqual(result["evaluations_consumed"], 1)
        self.assertEqual(result["proposal_batches"], 2)
        self.assertEqual(
            [outcome.observation.status for outcome in arm.outcomes],
            ["rejected", "ok"],
        )

    def test_projection_duplicate_is_reported_then_arm_can_retry(self):
        space = SearchSpace.from_legacy({"y": ("int", 0, 5)})
        initial = Observation(
            observation_id="warm",
            params={"y": 5},
            score=25.0,
            status="ok",
            origin="warm",
            consumes_budget=False,
        )
        context = BenchmarkContext(
            checkpoint_id="integer-projection",
            regime="first",
            space=space,
            observations=(initial,),
            budget=1,
            seed=3,
        )
        arm = ScriptedArm(
            [
                ProposalBatch((Proposal({"y": 5.4}, "scripted"),)),
                ProposalBatch((Proposal({"y": 4}, "scripted"),)),
            ]
        )
        result = BenchmarkRunner(
            context,
            FunctionObjective(lambda params: params["y"] ** 2),
            Path(self.tmp.name) / "projection-retry",
        ).run(arm)

        self.assertEqual(result["evaluations_consumed"], 1)
        self.assertEqual(result["proposal_batches"], 2)
        self.assertEqual(result["final_incumbent_score"], 16.0)
        self.assertEqual(
            [outcome.observation.status for outcome in arm.outcomes],
            ["rejected", "ok"],
        )
        self.assertEqual(arm.outcomes[0].observation.params, {"y": 5})
        self.assertEqual(
            arm.outcomes[0].observation.failure,
            "duplicate configuration",
        )

    def test_arm_factory_loads_without_central_registry(self):
        factory = load_arm("tools.hpo_benchmark.arms.hillclimb:create_arm")
        self.assertTrue(callable(factory))


if __name__ == "__main__":
    unittest.main()
