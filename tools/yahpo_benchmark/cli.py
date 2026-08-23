from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import aggregate
from .io import read_json
from .runner import PROTOCOL_ID, run_matrix
from .suite import PILOT_TASKS, SMOKE_TASKS


DEFAULT_MODEL = "grok/grok-4.6"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, workers in (("smoke", 4), ("pilot", 8)):
        cmd = sub.add_parser(name)
        cmd.add_argument("--data-path", type=Path, required=True)
        cmd.add_argument("--output-dir", type=Path, required=True)
        cmd.add_argument("--model", default=DEFAULT_MODEL)
        cmd.add_argument("--seed", type=int, default=0)
        cmd.add_argument("--workers", type=int, default=workers)
        if name == "pilot":
            cmd.add_argument("--smoke-summary", type=Path, required=True)
            cmd.add_argument("--hard-cap-usd", type=float, default=125.0)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--input-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "aggregate":
        summary = aggregate(args.input_dir)
    elif args.command == "smoke":
        summary = run_matrix(
            tasks=SMOKE_TASKS,
            data_path=args.data_path,
            output_dir=args.output_dir,
            model=args.model,
            stage="smoke",
            bo_trials=2,
            seed=args.seed,
            workers=args.workers,
        )
    else:
        smoke = read_json(args.smoke_summary)
        if (
            smoke.get("protocol_id") != PROTOCOL_ID
            or smoke.get("stage") != "smoke"
            or smoke.get("initial_count") != 5
            or smoke.get("bo_trials") != 2
        ):
            raise ValueError("summary is not a compatible smoke result")
        projection = float(smoke["cost_projection"]["stage1_projected_usd"])
        if smoke.get("model") != args.model:
            raise ValueError(
                f"smoke model {smoke.get('model')!r} does not match {args.model!r}"
            )
        incomplete = {
            name: data
            for name, data in smoke.get("by_optimizer", {}).items()
            if data.get("runs") != 2 or data.get("complete") != 2
        }
        if smoke.get("result_count") != 6 or incomplete:
            raise RuntimeError(f"smoke protocol did not complete: {incomplete}")
        if projection > args.hard_cap_usd:
            raise RuntimeError(
                f"projected Stage 1 cost ${projection:.2f} exceeds "
                f"hard cap ${args.hard_cap_usd:.2f}"
            )
        summary = run_matrix(
            tasks=PILOT_TASKS,
            data_path=args.data_path,
            output_dir=args.output_dir,
            model=args.model,
            stage="pilot",
            bo_trials=20,
            seed=args.seed,
            workers=args.workers,
        )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
