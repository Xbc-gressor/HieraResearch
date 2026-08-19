"""Bernoulli-sampled alternation between LLM rank-1 and LLM-pool + HEBO-MACE.

Same POOL=5 LLM proposer as ``softalt``, but the per-round type is not a
strict alternation: each evaluation round is [hebo_rerank] with probability

    p(t) = clip((max(t - SHIFT, 0)**ALPHA - 1) / SCALE, 0, 1)

where t = ``len(finite_unique_history)`` at decision time (frozen checkpoint
history + this cell's executed evaluations), and [llm_direct] otherwise.
Early rounds are almost pure LLM direct moves; HEBO's share grows as the
surrogate gains history (ALPHA=0.5, SHIFT=7, SCALE=4 gives p(8)=0,
p(16)=0.5, p(24)~0.78). The coin is flipped ONCE per step BEFORE the pool is
generated — ask_pool retries do not re-draw — and the round type is announced
to the proposer. Preflight rejections do not advance t (the history length
is unchanged).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arm_api  # noqa: E402
from arms import pool_hebo_mace  # noqa: E402
from arms.pool import PoolDriver, pool_persistence_state  # noqa: E402


ALPHA = 0.5
SHIFT = 7.0
SCALE = 4.0


def hebo_probability(t: int) -> float:
    """P([hebo_rerank]) at history count t; 0 for t <= SHIFT + 1."""
    return min(max(((max(t - SHIFT, 0.0) ** ALPHA) - 1.0) / SCALE, 0.0), 1.0)


def bernsalt_protocol() -> str:
    # Built at run() time so monkeypatched constants render correctly.
    return (
        "LLM pool protocol (Bernoulli-alternating variant): generate exactly "
        "POOL=5 complete, mutually distinct configs in one call, ranked by "
        "your own judgment. Every evaluation round is independently one of "
        "two types, drawn before the ask and announced with it: "
        f"[hebo_rerank] with probability p(t) = clip(((t - {SHIFT:g})"
        f"**{ALPHA:g} - 1) / {SCALE:g}, 0, 1), where t is the number of "
        "executed evaluations in history, and [llm_direct] otherwise. On "
        "[hebo_rerank], the official HEBO MACE acquisition chooses one member "
        "of your pool; make the pool diverse enough to give that selector "
        "meaningful choices. On [llm_direct], your rank-1 non-duplicate "
        "config is executed directly; use that turn for the precise move you "
        "judge best. A preflight rejection consumes no evaluation and "
        "retries the same kind of round. Unexecuted pool members are not "
        "outcome evidence, and every executed outcome is absorbed into the "
        "HEBO history."
    )


_ROUND_ANNOUNCE = {
    True: (
        "[hebo_rerank] this round: the official HEBO MACE acquisition "
        "chooses one member of your pool; make the pool diverse enough to "
        "give that selector meaningful choices."
    ),
    False: (
        "[llm_direct] this round: your rank-1 non-duplicate config is "
        "executed directly; use it for the precise move you judge best."
    ),
}


class Bernsalt:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "bernsalt"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        driver = PoolDriver(ctx, protocol=bernsalt_protocol())
        rank_fn = ctx.extras.get("hebo_rank_fn") or pool_hebo_mace._subprocess_rank_fn
        step_index = 0
        ranker_fallback_count = 0
        try:
            while True:
                t = len(ctx.state.finite_unique_history())
                p_hebo = hebo_probability(t)
                u = float(ctx.np_rng.random())
                rerank_step = u < p_hebo
                driver.announce(_ROUND_ANNOUNCE[rerank_step])
                result = driver.ask_pool()
                pool = result["pool"]
                arm_state = {
                    "step_index": step_index,
                    "t": int(t),
                    "p_hebo": float(p_hebo),
                    "u": u,
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
                    source = "bernsalt_hebo"
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
                    source = "bernsalt_llm"

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


ARM = Bernsalt()
