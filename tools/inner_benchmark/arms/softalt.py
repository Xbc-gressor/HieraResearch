"""Soft alternation between LLM rank-1 and LLM-pool + HEBO-MACE.

Every step starts with the same POOL=5 LLM proposer.  Even evaluation steps
use the exact live-history selector from ``pool_hebo_mace``; odd evaluation
steps execute the proposer's highest-ranked non-duplicate config directly.
The arm therefore gives the LLM an undiluted decision every other evaluation
without removing HEBO's ability to rerank a diverse LLM-generated pool on the
interleaved steps.  Preflight rejections do not advance the phase.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arm_api  # noqa: E402
from arms import pool_hebo_mace  # noqa: E402
from arms.pool import PoolDriver, pool_persistence_state  # noqa: E402


SOFTALT_PROTOCOL = (
    "LLM pool protocol (soft-alternating variant): generate exactly POOL=5 "
    "complete, mutually distinct configs in one call, ranked by your own "
    "judgment. Successful evaluation rounds alternate, starting with a "
    "[hebo_rerank] round and then an [llm_direct] round. On [hebo_rerank], "
    "the official HEBO MACE acquisition chooses one member of your pool; "
    "make the pool diverse enough to give that selector meaningful choices. "
    "On [llm_direct], your rank-1 non-duplicate config is executed directly; "
    "use that turn for the precise move you judge best. A preflight rejection "
    "consumes no evaluation and retries the same kind of round. Unexecuted "
    "pool members are not outcome evidence, and every executed outcome is "
    "absorbed into the HEBO history."
)


class SoftAlt:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "softalt"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        driver = PoolDriver(ctx, protocol=SOFTALT_PROTOCOL)
        rank_fn = ctx.extras.get("hebo_rank_fn") or pool_hebo_mace._subprocess_rank_fn
        step_index = 0
        ranker_fallback_count = 0
        try:
            while True:
                result = driver.ask_pool()
                pool = result["pool"]
                rerank_step = step_index % 2 == 0
                arm_state = {
                    "step_index": step_index,
                    "selection": "hebo_rerank" if rerank_step else "llm_direct",
                    "pool_size": len(pool),
                    "pool_attempts": result["attempts"],
                    **pool_persistence_state(result),
                }

                if rerank_step:
                    chosen_index, selection_state = pool_hebo_mace.select_pool(
                        rank_fn,
                        ctx,
                        ctx.state.finite_unique_history(),
                        pool,
                    )
                    arm_state.update(selection_state)
                    if arm_state["ranker_fallback"]:
                        ranker_fallback_count += 1
                    source = "softalt_hebo"
                else:
                    chosen_index = 0
                    literal_rank1 = not result["pool_duplicate_mask"][0]
                    arm_state["literal_rank1"] = literal_rank1
                    if not literal_rank1:
                        driver.push_correction(
                            "your rank-1 config duplicated executed history "
                            "and was filtered out; your next-ranked unique "
                            "config was executed instead. Do not propose "
                            "already-executed configs."
                        )
                    source = "softalt_llm"

                chosen = pool[chosen_index]
                arm_state["chosen_index"] = int(chosen_index)
                incumbent_before = ctx.state.incumbent_score
                feedback = yield arm_api.Proposal(
                    params=chosen,
                    source=source,
                    rationale=result["rationale"],
                    arm_state=arm_state,
                )
                driver.report_outcome(
                    chosen, feedback, incumbent_before=incumbent_before
                )
                if feedback.kind == "outcome":
                    step_index += 1
        finally:
            ctx.emit(
                {**driver.totals(), "ranker_fallback_count": ranker_fallback_count}
            )


ARM = SoftAlt()
