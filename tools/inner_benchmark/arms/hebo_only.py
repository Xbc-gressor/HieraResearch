"""Official HEBO-only control arm (PLAN-inner-arms-mixup-alt §2, ``hebo_only``).

No LLM. Every step mirrors one official ``HEBO.suggest`` call through the
``hebo_mace/suggest.py`` subprocess (or the ``hebo_suggest_fn`` test seam):

- input: the shared ``live_hebo_history(ctx)`` — identical finiteness /
  dedupe / order / incumbent semantics as mixup and alt;
- warmup: the official ``rand_sample = 1 + num_paras`` quasi_sample branch
  IS this arm's cold start, presented as-is — no extra WARMUP gate, no LLM
  fallback. Cells run long enough (E2: 24 evals) for the surrogate phase to
  have real room after warmup;
- surrogate: ``initial_suggest_extra`` defaults to empty, i.e. official
  ``initial_suggest=best_x``;
- sources: ``hebo_quasi`` (warmup) / ``hebo_suggest`` (surrogate). Official
  quasi points are NOT ``ranker_fallback`` and never enter that count.

Seeding (arm_api checklist #5): the per-step seed is the shared pure
function ``step_seed(ctx.seed, step_index)`` — the same rule mixup and
alt's BO steps use, so a paired (checkpoint, seed) mixup cell sees
bit-identical surrogate randomness at every step index; the Sobol sequence
is located by the trajectory-level ``scramble_seed + quasi_index``, consumed
immediately on each suggestion (a duplicate/task-preflight reject never
replays a point).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arm_api  # noqa: E402
from arms import hebo_common  # noqa: E402


class HeboOnly:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "hebo_only"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        suggest_fn = ctx.extras.get("hebo_suggest_fn") or hebo_common.subprocess_suggest_fn
        scramble_seed = hebo_common.trajectory_scramble_seed(ctx.seed)
        quasi_index = 0
        step_index = 0
        while True:
            history = hebo_common.live_hebo_history(ctx)
            result = hebo_common.call_suggest(
                suggest_fn,
                ctx,
                history=history,
                seed=hebo_common.step_seed(ctx.seed, step_index),
                scramble_seed=scramble_seed,
                quasi_index=quasi_index,
                initial_suggest_extra=[],
            )
            quasi_index += result["quasi_consumed"]
            mode = result["mode"]
            arm_state = {
                "hebo_mode": mode,
                "quasi_index": quasi_index,  # trajectory position AFTER this point
                "step_index": step_index,
            }
            if mode == "surrogate":
                arm_state["front_size"] = result.get("front_size")
            yield arm_api.Proposal(
                params=result["suggestion"],
                source="hebo_quasi" if mode == "quasi" else "hebo_suggest",
                arm_state=arm_state,
            )
            step_index += 1


ARM = HeboOnly()
