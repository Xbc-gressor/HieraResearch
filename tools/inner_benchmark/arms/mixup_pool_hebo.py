"""LLM-pool-as-initial_suggest fusion arm (PLAN-inner-arms-mixup-alt §3,
``mixup_pool_hebo``).

The shadow-replay recommendation (ANALYSIS-shadow-lhs-pool.md): official
HEBO whose EvolutionOpt initial population is seeded with the LLM pool,
instead of the veto structure (MACE only ranks inside the pool).

- Input every step: the shared ``live_hebo_history(ctx)`` — literally the
  same rows ``hebo_only`` fits on.
- While live history < official ``rand_sample = 1 + num_paras`` the arm does
  NOT create an LLM session and does NOT ask for a pool: it executes the
  same official quasi_sample point ``hebo_only`` would (source
  ``hebo_quasi`` — official warmup, never ``ranker_fallback``). Under a
  paired (checkpoint, seed) the two arms' quasi prefixes are identical, and
  the per-step suggest seed is the same pure function
  ``step_seed(ctx.seed, step_index)`` derived BEFORE ``ask_pool`` — so once
  the surrogate phase starts, the ONLY structural difference between the
  arms is ``initial_suggest_extra=pool`` (the causal channel for "does the
  LLM prior help").
- At/above the threshold the PoolDriver session is created (first message
  rendered from the LIVE trials/incumbent, so the proposer sees this cell's
  own quasi rows), the filtered pool becomes ``initial_suggest_extra``, and
  the officially selected point is executed (source ``mixup_hebo``).

Prompting: ``MIXUP_PROTOCOL`` replaces the POOL_PROTOCOL slot — the pool
members are SEEDS for the evolutionary acquisition search; the executed
config is usually a descendant, so the proposer should offer the patterns /
directions it wants covered. Outcome feedback is unchanged
(``report_outcome`` shows the executed CAST params and score).

Surrogate-step arm_state: the pool persistence triple + ``front_size``
(final-generation size after dedupe/uniqueness) + ``nearest_seed_z_dist`` /
``hit_seed`` (codec-z geometry of the executed point vs the seeds). These
three are trajectory observables only.

Registered mechanistic-drift risk (to be carried into SELECTION-inner-v2.md):
telling the LLM its configs are seeds makes gaming NSGA-II itself a rational
strategy. Within one bout 5/100 population share bounds the impact; if this
arm ever runs across bouts in production, the outcome-feedback loop lets the
drift accumulate.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arm_api  # noqa: E402
from arms import hebo_common  # noqa: E402
from arms.pool import PoolDriver, pool_persistence_state  # noqa: E402

MIXUP_PROTOCOL = (
    "LLM pool protocol (seed variant): each step you generate exactly POOL=5 "
    "complete, mutually distinct candidate configs in one call, ranked by "
    "your own judgment (order[0] is the config you most want investigated). "
    "Your five configs are used as SEEDS of an evolutionary acquisition "
    "search (a GP-surrogate multi-objective NSGA-II over the whole search "
    "space): the config actually executed is usually a DESCENDANT of your "
    "seeds, not a literal pool member. So propose the patterns and "
    "directions you most want the search to cover — diverse, well-separated "
    "seeds give the evolution meaningful material, while five near-identical "
    "configs collapse its search to one spot. After each step the "
    "authoritative outcome of the actually-executed config is appended (the "
    "CAST params that ran, and its score); unexecuted pool members are not "
    "outcome evidence. Every step asks for a fresh pool."
)


class MixupPoolHebo:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "mixup_pool_hebo"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        suggest_fn = ctx.extras.get("hebo_suggest_fn") or hebo_common.subprocess_suggest_fn
        scramble_seed = hebo_common.trajectory_scramble_seed(ctx.seed)
        rand_sample = hebo_common.official_rand_sample(ctx.contract)
        quasi_index = 0
        step_index = 0
        driver = None
        try:
            while True:
                history = hebo_common.live_hebo_history(ctx)
                if len(history) < rand_sample:
                    # Official warmup: no LLM session, no pool — the same
                    # quasi point hebo_only executes at this step index.
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
                    yield arm_api.Proposal(
                        params=result["suggestion"],
                        source="hebo_quasi",
                        arm_state={
                            "hebo_mode": result["mode"],
                            "quasi_index": quasi_index,
                            "step_index": step_index,
                        },
                    )
                else:
                    if driver is None:
                        # First surrogate step: create the session rendering
                        # the LIVE state (checkpoint history + this cell's
                        # quasi rows) and the current incumbent.
                        driver = PoolDriver(
                            ctx,
                            protocol=MIXUP_PROTOCOL,
                            first_trials=ctx.state.trials,
                            first_live_incumbent=ctx.state.best_so_far(),
                        )
                    # F1: the step seed is fixed BEFORE the LLM round-trip.
                    seed = hebo_common.step_seed(ctx.seed, step_index)
                    result = driver.ask_pool()
                    pool = result["pool"]
                    suggestion = hebo_common.call_suggest(
                        suggest_fn,
                        ctx,
                        history=history,
                        seed=seed,
                        scramble_seed=scramble_seed,
                        quasi_index=quasi_index,
                        initial_suggest_extra=pool,
                    )
                    quasi_index += suggestion["quasi_consumed"]
                    arm_state = {
                        "hebo_mode": suggestion["mode"],
                        "quasi_index": quasi_index,
                        "step_index": step_index,
                        "pool_size": len(pool),
                        "pool_attempts": result["attempts"],
                        **pool_persistence_state(result),
                        **hebo_common.seed_geometry(ctx, suggestion["suggestion"], pool),
                    }
                    if "front_size" in suggestion:
                        arm_state["front_size"] = suggestion["front_size"]
                    # Capture before yield: the runner advances the incumbent
                    # before the feedback returns.
                    incumbent_before = ctx.state.incumbent_score
                    feedback = yield arm_api.Proposal(
                        params=suggestion["suggestion"],
                        source="mixup_hebo",
                        rationale=result["rationale"],
                        arm_state=arm_state,
                    )
                    driver.report_outcome(
                        suggestion["suggestion"], feedback,
                        incumbent_before=incumbent_before,
                    )
                step_index += 1
        finally:
            if driver is not None:
                ctx.emit(driver.totals())


ARM = MixupPoolHebo()
