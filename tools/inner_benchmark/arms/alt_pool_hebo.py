"""50/50 alternating arm (PLAN-inner-arms-mixup-alt §4, ``alt_pool_hebo``).

Within any even-budget cell, even steps are BO and odd steps are LLM: BO0 ->
LLM1 -> ... -> BO22 -> LLM23 at B=24 (exactly 12 complete pairs); BO first
so every BO point is followed by an LLM step that can consume it — the
"global probe -> precise LLM follow-up -> surrogate absorbs the follow-up"
loop.

- BO step: the shared ``live_hebo_history(ctx)`` through the
  ``hebo_mace/suggest.py`` mirror (``initial_suggest_extra`` empty = official
  best_x only), so the surrogate fit automatically includes every
  axis-aligned row the LLM steps produced. Sources: ``alt_hebo_quasi``
  (official warmup branch) / ``alt_hebo`` (surrogate). Per-step seed: the
  same pure ``step_seed(ctx.seed, step_index)`` rule as hebo_only/mixup.
- LLM step: ``ask_pool()`` then the proposer's rank-1 config is executed
  VERBATIM (source ``alt_llm``) — per-axis precise control (line searches,
  single-axis clones) is this arm's raison d'etre vs the veto/fusion arms.
  Fallback (rank-1 filtered as an executed-history duplicate): execute the
  filtered ``pool[0]`` and push a correction note into the session; the
  driver's ``internal_duplicate_count`` already counted the filtered rank-1.
  A preflight rejection consumes no evaluation phase: the same BO/LLM side
  retries, preserving exactly B/2 complete pairs for an even objective
  budget. Rejected BO suggestions still consume their Sobol position and
  proposal RNG seed, so the retry cannot replay the same point.

Information flow (both channels must work or the alternation is pointless):

- LLM -> BO: the live ``finite_unique_history`` — no code needed.
- BO -> LLM: every executed BO outcome and preflight rejection enters the LLM
  session as a TAGGED message (``[hebo_quasi]`` / ``[hebo_probe]``) via
  ``PoolDriver.report_external_outcome``, riding the next ask. The session
  is created only after BO0 has executed (PoolDriver's zero-LLM-cost
  precheck still runs at arm start): the first message then renders the
  LIVE history (BO0 row included) and the current incumbent through the
  ``first_message_blocks`` trials/incumbent overrides, and BO0's tagged
  message rides the first ask like every later BO step's. One session for
  the whole bout.

Mechanical invariants (acceptance):
(i) LLM steps execute rank-1 verbatim (precise control undiluted);
(ii) BO surrogate fits include every LLM row (local axis-aligned data is
globally exploited);
(iii) the LLM reads tagged BO outcomes — the regression test proves the
tagged outcome reaches the next ask.

``pool_follows_probe`` is a descriptive diagnostic only: LLM-step arm_state
carries the z distance/direction from the last executed BO point to the
executed rank-1, plus the pre-registered E1 covariate ``z_dist_to_incumbent``.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arm_api  # noqa: E402
from arms import hebo_common  # noqa: E402
from arms.pool import PoolDriver, pool_persistence_state  # noqa: E402

ALT_PROTOCOL = (
    "LLM pool protocol (alternating variant): you propose on ALTERNATING "
    "rounds — every other step is executed by an external Bayesian "
    "optimizer, not by you. On YOUR turn you generate exactly POOL=5 "
    "complete, mutually distinct configs in one call, ranked by your own "
    "judgment, and your rank-1 config (order[0]) is executed VERBATIM — "
    "that is your precise-control channel: single-axis clones and line "
    "searches are yours to drive, so make rank-1 serve a coherent "
    "multi-step direction that continues across the intervening BO rounds. "
    "(If rank-1 duplicates an already-executed config it is filtered and "
    "your next-ranked unique config runs instead, with a correction note.) "
    "Unexecuted pool members are not outcome evidence.\n"
    "The rounds in between are executed by the optimizer: a [hebo_quasi] "
    "Sobol space-filling point during warmup, a [hebo_probe] point chosen "
    "by a GP-MACE surrogate afterwards. Each such outcome is reported to "
    "you with its tag before your next turn. How to read those rows: a "
    "crash or preflight failure is negative FEASIBILITY evidence for that "
    "region; a finite score is a single noisy observation — a "
    "worse-than-incumbent probe does not kill the whole region; an "
    "improvement is a lead worth following up locally, not something you "
    "must copy. A [hebo_probe] row is just one external observation chosen "
    "by the surrogate's acquisition trade-off — weigh it together with its "
    "distance from your direction, the noise scale, and the existing "
    "history. A [hebo_quasi] row is space-filling warmup and carries no "
    "surrogate judgment about the region's value. Every executed row "
    "(yours included) is absorbed by the surrogate."
)


class AltPoolHebo:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "alt_pool_hebo"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        # Zero-LLM-cost gates at arm start (the session itself is deferred
        # until after BO0 — see module docstring).
        PoolDriver.precheck(ctx)
        suggest_fn = ctx.extras.get("hebo_suggest_fn") or hebo_common.subprocess_suggest_fn
        scramble_seed = hebo_common.trajectory_scramble_seed(ctx.seed)
        quasi_index = 0
        # step_index advances only after an objective evaluation.  This keeps
        # an even budget at exactly BO, LLM, ... even when preflight rejects a
        # proposal without consuming budget.  proposal_index advances on
        # every attempted proposal so a rejected HEBO point is retried with
        # fresh non-Sobol randomness (quasi_index independently advances as
        # soon as a Sobol point is suggested).
        step_index = 0
        proposal_index = 0
        driver = None
        last_bo = None  # params of the last EXECUTED BO point (probe-follow diagnostic)
        pending_bo_rejections = []  # retained before the post-BO0 session exists

        def ensure_driver():
            nonlocal driver
            if driver is None:
                driver = PoolDriver(
                    ctx,
                    protocol=ALT_PROTOCOL,
                    first_trials=ctx.state.trials,
                    first_live_incumbent=ctx.state.best_so_far(),
                )

        try:
            while True:
                if step_index % 2 == 0:
                    # --- BO step -------------------------------------------
                    history = hebo_common.live_hebo_history(ctx)
                    result = hebo_common.call_suggest(
                        suggest_fn,
                        ctx,
                        history=history,
                        seed=hebo_common.step_seed(ctx.seed, proposal_index),
                        scramble_seed=scramble_seed,
                        quasi_index=quasi_index,
                        initial_suggest_extra=[],
                    )
                    quasi_index += result["quasi_consumed"]
                    mode = result["mode"]
                    arm_state = {
                        "hebo_mode": mode,
                        "quasi_index": quasi_index,
                        "step_index": step_index,
                        "proposal_index": proposal_index,
                    }
                    if "front_size" in result:
                        arm_state["front_size"] = result["front_size"]
                    incumbent_before = ctx.state.incumbent_score
                    feedback = yield arm_api.Proposal(
                        params=result["suggestion"],
                        source="alt_hebo_quasi" if mode == "quasi" else "alt_hebo",
                        arm_state=arm_state,
                    )
                    proposal_index += 1
                    if feedback.kind == "outcome":
                        executed = feedback.executed_params or result["suggestion"]
                        ensure_driver()
                        for rejected in pending_bo_rejections:
                            driver.report_external_rejection(**rejected)
                        pending_bo_rejections.clear()
                        driver.report_external_outcome(
                            executed,
                            status=feedback.status,
                            score=feedback.score,
                            incumbent_before=incumbent_before,
                            tag="[hebo_quasi]" if mode == "quasi" else "[hebo_probe]",
                        )
                        last_bo = executed
                        step_index += 1
                    else:
                        rejected = {
                            "params": result["suggestion"],
                            "stage": feedback.stage,
                            "reason": feedback.reason,
                            "tag": "[hebo_quasi]" if mode == "quasi" else "[hebo_probe]",
                        }
                        if driver is None:
                            # BO0 has not executed yet, so preserve the
                            # delayed-session contract and replay this message
                            # immediately after the first successful BO.
                            pending_bo_rejections.append(rejected)
                        else:
                            driver.report_external_rejection(**rejected)
                else:
                    # --- LLM step ------------------------------------------
                    ensure_driver()
                    result = driver.ask_pool()
                    if not result["pool_duplicate_mask"][0]:
                        chosen = result["pool_ranked"][0]
                        literal_rank1 = True
                    else:
                        # F2 fallback: rank-1 was filtered as an executed-
                        # history duplicate — execute the filtered pool[0]
                        # and tell the session.
                        chosen = result["pool"][0]
                        literal_rank1 = False
                        driver.push_correction(
                            "your rank-1 config duplicated executed history "
                            "and was filtered out; your next-ranked unique "
                            "config was executed instead. Do not propose "
                            "already-executed configs."
                        )
                    arm_state = {
                        "step_index": step_index,
                        "proposal_index": proposal_index,
                        "literal_rank1": literal_rank1,
                        "pool_size": len(result["pool"]),
                        "pool_attempts": result["attempts"],
                        # Pre-registered E1 covariate: executed rank-1's z
                        # distance to the current incumbent.
                        "z_dist_to_incumbent": hebo_common.z_distance(
                            ctx, chosen, ctx.state.incumbent_config
                        ),
                        **pool_persistence_state(result),
                    }
                    if last_bo is not None:
                        delta = hebo_common.z_delta(ctx, chosen, last_bo)
                        arm_state["probe_z_delta"] = delta
                        arm_state["probe_z_dist"] = float(
                            sum(d * d for d in delta)
                        ) ** 0.5
                    incumbent_before = ctx.state.incumbent_score
                    feedback = yield arm_api.Proposal(
                        params=chosen,
                        source="alt_llm",
                        rationale=result["rationale"],
                        arm_state=arm_state,
                    )
                    proposal_index += 1
                    driver.report_outcome(chosen, feedback, incumbent_before=incumbent_before)
                    if feedback.kind == "outcome":
                        step_index += 1
        finally:
            if driver is not None:
                ctx.emit(driver.totals())


ARM = AltPoolHebo()
