"""CPU-only smoke demonstrating dynamic arm loading and shared artifacts.

Run:
  python -m tools.hpo_benchmark.demo --output /tmp/hpo-core-demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import (
    BenchmarkContext,
    BenchmarkRunner,
    FunctionObjective,
    Observation,
    SearchSpace,
    load_arm,
)
from .providers import ReplayProposalProvider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm",
        default="tools.hpo_benchmark.arms.hillclimb:create_arm",
        help="module:function arm factory",
    )
    args = parser.parse_args()

    space = SearchSpace.from_legacy(
        {"x": ("float", -5.0, 5.0), "y": ("int", -5, 5)}
    )
    initial = Observation(
        observation_id="warm-000",
        params={"x": 4.0, "y": -3},
        score=25.0,
        status="ok",
        origin="warm",
        consumes_budget=False,
    )
    context = BenchmarkContext(
        checkpoint_id="cpu-demo",
        regime="first",
        space=space,
        observations=(initial,),
        budget=3,
        seed=0,
    )
    provider = ReplayProposalProvider(
        [
            {"changes": {"x": 2.0}, "reason": "move x toward zero"},
            {"changes": {"y": -1}, "reason": "move y toward zero"},
            {"changes": {"x": 0.0}, "reason": "finish x coordinate"},
        ]
    )
    arm = load_arm(args.arm)(provider=provider)
    result = BenchmarkRunner(
        context,
        FunctionObjective(lambda params: float(params["x"]) ** 2 + int(params["y"]) ** 2),
        args.output,
    ).run(arm)
    print(
        f"{result['arm']}: {result['initial_incumbent_score']} -> "
        f"{result['final_incumbent_score']} in {result['evaluations_consumed']} evaluations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
