"""LLM pool + HEBO MACE ranker arm with an LLM Pareto break-tie
(PLAN-inner-arm-hebo-mace-llmtie, ``pool_hebo_mace_llmtie``).

Step-for-step identical to ``pool_hebo_mace`` — same PoolDriver proposer,
same official HEBO MACE ranker, same nondominated sort (maximization),
same WARMUP gate. The ONLY difference: when the first Pareto front has
more than one member, the executed config is chosen by a fresh, throwaway
LLM judge session (role ``bench-tiebreak-judge``) instead of a
``ctx.np_rng`` uniform draw:

- One session per tie event, asked, then discarded — memoryless across
  ties, the same i.i.d. structure as the rng draw it replaces.
- The judge sees the live standard view (``first_message_blocks`` with
  ``trials`` / ``live_incumbent`` at tie time) plus a ``front`` payload:
  per tied member a display index, the PRODUCTION-CAST params (what would
  actually run — the runner's int cast truncates toward zero, arm_api
  checklist #2), and the member's MACE acquisition 3-vector
  ``[-lcb, log EI, log PI]`` (larger-is-better).
- Display order is a per-tie reproducible permutation
  (``ctx.np_rng.permutation``): front order is pool order is proposer rank
  order (arms/pool.py), so showing it verbatim would leak the proposer's
  ranking through the position channel. The judge's ``choice`` is a
  DISPLAY index, mapped back to the real pool index here; the mapping is
  persisted as ``displayed_front_pool_indices``.
- Proposer rank numbers/rationale and dominated pool members are never
  shown.

An invalid/out-of-range ``choice`` is re-asked with a ``correction``
block on the same session; 3 consecutive failures raise ``ArmError`` —
fail-fast, no rng fallback (same stance as the ranker).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arm_api  # noqa: E402
import llm  # noqa: E402
from arms import pool_hebo_mace  # noqa: E402

ROLE = "bench-tiebreak-judge"
MAX_CONSECUTIVE_FAILED_ATTEMPTS = 3

TIEBREAK_PROTOCOL = (
    "Pareto break-tie protocol: a numerical multi-objective ranker (HEBO "
    "MACE) scored this step's candidate pool, and the candidates listed "
    "under `front` form its first Pareto front — mutually nondominated "
    "across the three acquisition components, so the ranker itself cannot "
    "choose between them. Pick exactly ONE to execute next by returning "
    "its display index as `choice`."
)


def _format_front(ctx, values, pool, displayed) -> str:
    """The ``front`` extras block: one entry per tied member, in DISPLAY
    order, with production-cast params and the MACE acquisition vector."""
    lines = []
    for display_index, pool_index in enumerate(displayed):
        cast = ctx.contract.cast(pool[pool_index])
        params = json.dumps(cast, separators=(",", ":"), ensure_ascii=False, default=str)
        vector = ", ".join(f"{float(v):.6g}" for v in values[pool_index])
        lines.append(
            f"[{display_index}] params: {params}\n"
            f"    acquisition [-lcb, log_EI, log_PI] (larger is better): [{vector}]"
        )
    return "\n".join(lines)


def _llm_break_tie(ctx, front, values, pool) -> tuple[int, dict]:
    """One fresh judge session for this tie; returns (pool_index, state)."""
    factory = ctx.extras["session_factory"]  # guaranteed by PoolDriver.precheck
    # Display order: per-tie reproducible permutation of the front (module
    # docstring — the verbatim front order would leak proposer rank).
    displayed = [front[int(i)] for i in ctx.np_rng.permutation(len(front))]
    session = factory(
        ROLE,
        first_extras=llm.first_message_blocks(
            ctx.checkpoint,
            ctx.contract,
            protocol=TIEBREAK_PROTOCOL,
            budget_remaining=ctx.state.budget_remaining,
            trials=ctx.state.trials,
            live_incumbent=ctx.state.best_so_far(),
        ),
    )
    front_text = _format_front(ctx, values, pool, displayed)
    correction = None
    failures = 0
    while True:
        extra = {"front": front_text}
        if correction is not None:
            extra["correction"] = correction
        try:
            receipt = session.ask(extra=extra)
        except Exception as exc:  # InvocationFailed etc.: one failed attempt
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILED_ATTEMPTS:
                raise arm_api.ArmError(
                    f"tiebreak judge: {MAX_CONSECUTIVE_FAILED_ATTEMPTS} "
                    f"consecutive failed attempts (last: {exc})"
                )
            correction = (
                f"your last answer failed ({exc}). Submit the receipt again "
                "with a valid choice."
            )
            continue
        choice = receipt.get("choice") if isinstance(receipt, dict) else None
        if (
            isinstance(choice, bool)
            or not isinstance(choice, int)
            or not 0 <= choice < len(displayed)
        ):
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILED_ATTEMPTS:
                raise arm_api.ArmError(
                    f"tiebreak judge: {MAX_CONSECUTIVE_FAILED_ATTEMPTS} "
                    f"consecutive invalid choices (last: {choice!r})"
                )
            correction = (
                f"choice must be an integer display index in "
                f"0..{len(displayed) - 1}, got {choice!r}. Submit again."
            )
            continue
        totals = session.totals()
        state = {
            "tiebreak": {
                "front_size": len(front),
                "judge_choice": int(choice),
                "displayed_front_pool_indices": [int(i) for i in displayed],
            },
            # AGGREGATE_ARM_STATE_KEYS on THIS event's arm_state; metrics
            # sums them across events (no collision with the driver emit,
            # which lands on cell_end).
            "llm_calls": totals["llm_calls"],
            "llm_input_tokens": totals["llm_input_tokens"],
            "llm_output_tokens": totals["llm_output_tokens"],
        }
        return displayed[choice], state


class PoolHeboMaceLlmTie(pool_hebo_mace.PoolHeboMace):
    """pool_hebo_mace with the rng Pareto break-tie replaced by the judge."""

    name = "pool_hebo_mace_llmtie"

    @staticmethod
    def select_pool(rank_fn, ctx, history, pool) -> tuple[int, dict]:
        if len(history) < arm_api.WARMUP:
            return 0, {"ranker_fallback": True}
        values = pool_hebo_mace._rank(rank_fn, ctx, history, pool)
        front = pool_hebo_mace._first_pareto_front(values)
        state = {
            "ranker_fallback": False,
            "acquisition_values": values.tolist(),
            "pareto_front": [int(index) for index in front],
        }
        if len(front) == 1:
            return front[0], state
        chosen_index, tiebreak_state = _llm_break_tie(ctx, front, values, pool)
        state.update(tiebreak_state)
        return chosen_index, state


ARM = PoolHeboMaceLlmTie()
