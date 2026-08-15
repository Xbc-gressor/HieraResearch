"""Local trust region arm (PLAN §6.2).

Surrogate-free local-search baseline: tests the value of locality + adaptive
radius with NO model. NOT a TuRBO reproduction — no GP, no Thompson sampling,
no success/failure counters.

Protocol (all arithmetic in the codec's normalized z space):

- radius ``r`` starts at 0.25 per cell (cold; nothing inherited across bouts),
  clamped to [0.025, 1.0];
- anchor = the current incumbent at each proposal;
- every varying numeric dimension samples independently from
  ``[max(0, z-r), min(1, z+r)]``; every varying categorical dimension flips to
  another legal option with probability ``min(r, 1)`` (single-option
  categoricals never participate; degenerate numerics cannot move by
  construction — decode maps every z to the one legal value);
- duplicates against ALL executed history (not just the anchor) are dropped
  and resampled in-arm, bounded at 32 consecutive drops -> ArmError;
- strict improvement grows ``r <- min(1.0, 1.5r)``; anything else (including
  crash) shrinks ``r <- max(0.025, 0.5r)``. A preflight rejection is not an
  evaluation: the arm simply proposes again with the radius untouched (the
  runner's 5-consecutive-reject tripwire bounds that path).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tuners"))

import arm_api  # noqa: E402
import tune_tools  # noqa: E402

R0 = 0.25
R_MIN = 0.025
R_MAX = 1.0
GROW = 1.5
SHRINK = 0.5
MAX_CONSECUTIVE_DUPLICATES = 32

_EXECUTED = ("ok", "crash")


class LocalTrustRegion:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "local_tr"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        contract = ctx.contract
        codec = ctx.codec
        varying_numeric = [
            index
            for index, dim in enumerate(codec.numeric_dimensions)
            if not dim.is_degenerate
        ]
        varying_categoricals = [
            dim for dim in codec.categorical_dimensions if not dim.is_degenerate
        ]
        if not varying_numeric and not varying_categoricals:
            raise arm_api.Unsupported(
                "local_tr: checkpoint has no varying dimension to move"
            )
        np_rng = ctx.np_rng
        internal_duplicates = 0
        internal_resamples = 0

        def executed_configs() -> list[dict]:
            return [
                trial.config
                for trial in ctx.state.trials
                if trial.status in _EXECUTED
            ]

        # Production FIRST-bout resume injects the persisted radius; the
        # benchmark cell contract stays cold-start R0 when extras omit it.
        extras = getattr(ctx, "extras", None) or {}
        try:
            radius = float(extras.get("tr_radius", R0))
        except (TypeError, ValueError):
            radius = R0
        radius = min(R_MAX, max(R_MIN, radius))
        try:
            while True:
                anchor_z, anchor_cats = codec.encode(ctx.state.incumbent_config)
                consecutive_duplicates = 0
                while True:
                    z = anchor_z.copy()
                    for index in varying_numeric:
                        lo = max(0.0, anchor_z[index] - radius)
                        hi = min(1.0, anchor_z[index] + radius)
                        z[index] = np_rng.uniform(lo, hi)
                    cats = dict(anchor_cats)
                    for dim in varying_categoricals:
                        if np_rng.random() < min(radius, 1.0):
                            others = [
                                option
                                for option in dim.options
                                # Production categorical membership is
                                # type-strict (1 and 1.0 are distinct
                                # options); builtin != is not.
                                if not tune_tools._categorical_value_equal(
                                    option, cats[dim.name]
                                )
                            ]
                            cats[dim.name] = others[int(np_rng.integers(len(others)))]
                    params = codec.decode(z, cats)
                    if not contract.is_duplicate(params, executed_configs()):
                        break
                    internal_duplicates += 1
                    internal_resamples += 1
                    consecutive_duplicates += 1
                    if consecutive_duplicates >= MAX_CONSECUTIVE_DUPLICATES:
                        raise arm_api.ArmError(
                            f"local_tr: {MAX_CONSECUTIVE_DUPLICATES} consecutive "
                            "in-arm duplicate draws"
                        )
                incumbent_before = ctx.state.incumbent_score
                feedback = yield arm_api.Proposal(
                    params=params,
                    source="local_tr",
                    arm_state={"tr_radius": radius},
                )
                if feedback.kind != "outcome":
                    continue  # preflight rejection: not evaluated, radius untouched
                if feedback.status == "ok" and feedback.score < incumbent_before:
                    radius = min(R_MAX, GROW * radius)
                else:
                    radius = max(R_MIN, SHRINK * radius)
        finally:
            ctx.emit(
                {
                    "internal_duplicate_count": internal_duplicates,
                    "internal_resample_count": internal_resamples,
                }
            )


ARM = LocalTrustRegion()
