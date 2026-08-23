"""LLM pool + self-rank arm (PLAN §6.4, ``llm_pool_self_rank``).

Each step the proposer generates POOL=5 configs and ranks them in the same
call (arms/pool.py driver); this arm executes the proposer's rank-1 config
verbatim — that IS the arm's selection strategy, not a fallback, and is never
recorded as one. No second LLM judge, no numerical ranker fit. This arm is
the direct control for the numerical pool-rankers in INITIAL/DEEP cells.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arm_api  # noqa: E402
from arms.pool import PoolDriver, pool_persistence_state  # noqa: E402


class LlmPoolSelfRank:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "llm_pool_self_rank"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        driver = PoolDriver(ctx)
        try:
            while True:
                result = driver.ask_pool()
                chosen = result["pool"][0]  # proposer's rank 1 after filtering
                # Capture before yield: the runner advances the incumbent
                # before the feedback returns; the verdict needs the score
                # the proposal had to beat.
                incumbent_before = ctx.state.incumbent_score
                feedback = yield arm_api.Proposal(
                    params=chosen,
                    source="pool_self_rank1",
                    rationale=result["rationale"],
                    arm_state={
                        "pool_size": len(result["pool"]),
                        "pool_attempts": result["attempts"],
                        **pool_persistence_state(result),
                    },
                )
                driver.report_outcome(chosen, feedback, incumbent_before=incumbent_before)
        finally:
            ctx.emit(driver.totals())


ARM = LlmPoolSelfRank()
