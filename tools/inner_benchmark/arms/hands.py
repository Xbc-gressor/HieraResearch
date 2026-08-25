"""Evolved-front ∪ literal-LLM-pool fusion arm (DESIGN-inner-arm-hands,
``hands``).

The methodological correction of the mixup arm's weaknesses
(DESIGN-inner-arm-hands §1.1): the LLM pool is NOT seeded into the
evolutionary population. Every surrogate step runs the OFFICIAL HEBO
pipeline with ``initial_suggest=best_x`` only — byte-identical GP fit and
EvolutionOpt randomness to ``hebo_only`` at the same live history and step
seed — and the official hebo.py:182 pick is then replaced by the union
selector: the filtered pool competes with the final generation on the SAME
fitted MACE acquisition, and the executed point is a uniform draw from the
first Pareto front of (final generation ∪ pool)
(``hebo_mace/suggest.py`` union mode). A pool member wins only when it is
nondominated by every evolved candidate on [-lcb, logEI, logPI].

- Input every step: the shared ``live_hebo_history(ctx)`` — literally the
  same rows ``hebo_only`` fits on.
- While live history < official ``rand_sample = 1 + num_paras`` the arm does
  NOT create an LLM session and does NOT ask for a pool: it executes the
  same official quasi_sample point ``hebo_only`` would (source
  ``hands_quasi`` — official warmup, never ``ranker_fallback``). Under a
  paired (checkpoint, seed) the two arms' quasi prefixes are identical, and
  the per-step suggest seed is the same pure function
  ``step_seed(ctx.seed, step_index)`` derived BEFORE ``ask_pool`` — so once
  the surrogate phase starts, the ONLY structural difference from
  ``hebo_only`` is the union selector (the causal channel for "do literal
  LLM candidates add value when scored by the surrogate").
- At/above the threshold the PoolDriver session is created (first message
  rendered from the LIVE trials/incumbent) and ``ask_pool()`` supplies the
  literal candidates; the executed point's provenance splits the source:
  ``hands_pool`` (a literal pool member won) / ``hands_front`` (an evolved
  candidate won).

Prompting (DESIGN §4): the DEFAULT ``POOL_PROTOCOL`` — the proposer is never
told that a GP/MACE surrogate scores its configs or that evolved candidates
compete with them, keeping the LLM behavior surface like-for-like with the
veto arm ``pool_hebo_mace``. There is no mixup-style "your configs are
seeds" disclosure, so the registered NSGA-II-gaming incentive does not
exist here. Evolved-side outcomes reach the session tagged ``[hebo_probe]``
(the alt arm's external-outcome channel); pool-member outcomes use the
ordinary ``report_outcome``.

Surrogate-step arm_state: the pool persistence triple + ``front_size``
(final-generation size) + ``union_front_size`` / ``pool_survivors`` /
``pool_survivor_indices`` (the viability statistics of DESIGN §6) +
``chosen_from`` (+ ``chosen_pool_index`` on pool wins, ``nearest_pool_z_dist``
on front wins — trajectory observables only).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arm_api  # noqa: E402
from arms import hebo_common  # noqa: E402
from arms.pool import PoolDriver, pool_persistence_state  # noqa: E402


class Hands:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "hands"

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
                        source="hands_quasi",
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
                        initial_suggest_extra=[],
                        pool=pool,
                    )
                    quasi_index += suggestion["quasi_consumed"]
                    chosen_from = suggestion["chosen_from"]
                    arm_state = {
                        "hebo_mode": suggestion["mode"],
                        "quasi_index": quasi_index,
                        "step_index": step_index,
                        "pool_size": len(pool),
                        "pool_attempts": result["attempts"],
                        **pool_persistence_state(result),
                        "front_size": suggestion["front_size"],
                        "union_front_size": suggestion["union_front_size"],
                        "pool_survivors": len(suggestion["pool_survivor_indices"]),
                        "pool_survivor_indices": suggestion["pool_survivor_indices"],
                        "chosen_from": chosen_from,
                    }
                    if chosen_from == "pool":
                        arm_state["chosen_pool_index"] = suggestion["chosen_pool_index"]
                    else:
                        arm_state["nearest_pool_z_dist"] = min(
                            hebo_common.z_distance(ctx, suggestion["suggestion"], member)
                            for member in pool
                        )
                    # Capture before yield: the runner advances the incumbent
                    # before the feedback returns.
                    incumbent_before = ctx.state.incumbent_score
                    feedback = yield arm_api.Proposal(
                        params=suggestion["suggestion"],
                        source="hands_pool" if chosen_from == "pool" else "hands_front",
                        rationale=result["rationale"],
                        arm_state=arm_state,
                    )
                    if chosen_from == "pool":
                        driver.report_outcome(
                            suggestion["suggestion"], feedback,
                            incumbent_before=incumbent_before,
                        )
                    elif feedback.kind == "outcome":
                        driver.report_external_outcome(
                            feedback.executed_params or suggestion["suggestion"],
                            status=feedback.status,
                            score=feedback.score,
                            incumbent_before=incumbent_before,
                            tag="[hebo_probe]",
                        )
                    else:
                        driver.report_external_rejection(
                            suggestion["suggestion"],
                            stage=feedback.stage,
                            reason=feedback.reason,
                            tag="[hebo_probe]",
                        )
                step_index += 1
        finally:
            if driver is not None:
                ctx.emit(driver.totals())


ARM = Hands()
