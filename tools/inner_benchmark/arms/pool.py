"""Pool-proposer driver shared by the four pool-rank arms (PLAN §6.4).

One bout-scoped ``bench-pool-proposer`` session per cell; each step asks for
POOL=5 unique configs + the proposer's self-ranking in ONE call. This module
owns the arm-side obligations the bench roles deliberately do NOT enforce
(arm_api author checklist #10):

- receipt semantics: exactly POOL config dicts, each contract-shaped
  (exact key set, castable, in-bounds), mutually distinct, and ``order`` a
  permutation of 0..POOL-1;
- internal dedupe: pool members duplicating ANY executed history row
  (crashes and the incumbent included) are dropped before ranking and
  counted under ``internal_duplicate_count`` — the runner never sees them;
- retries: an unparseable/failed invocation, a semantically invalid pool, or
  an all-duplicate pool after filtering is a failed attempt (the proposer is
  re-asked with a ``correction`` block; no budget is consumed); 3 consecutive
  failed attempts end the cell as ``arm_error``.

The selection variant (self-rank / GP-EI / TPE / HEBO) lives in the arm, not
here: ``ask_pool`` returns the filtered pool in PROPOSER rank order and the
arm picks one config from it.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tuners"))

import arm_api  # noqa: E402
import llm  # noqa: E402
import tune_tools  # noqa: E402

ROLE = "bench-pool-proposer"
MAX_CONSECUTIVE_FAILED_ATTEMPTS = 3

POOL_PROTOCOL = (
    "LLM pool protocol (PLAN §6.4): each step you generate exactly POOL=5 "
    "complete, mutually distinct candidate configs in one call, ranked by "
    "your own judgment (order[0] is the config you most want executed). A "
    "deterministic selector then executes exactly ONE config from the pool "
    "and appends its authoritative outcome. Unexecuted pool members are not "
    "outcome evidence; every step asks for a fresh pool."
)

_EXECUTED = ("ok", "crash")


class PoolDriver:
    """Per-cell proposer session driver. One instance per cell (in run())."""

    def __init__(self, ctx) -> None:
        if not ctx.contract.varying_dimensions:
            # Uniform across all four pool arms: nothing can move, so every
            # pool would all-duplicate into ArmError — mark the cell
            # unsupported instead, before any session exists (zero LLM cost).
            raise arm_api.Unsupported("pool arm: no varying dimensions")
        factory = ctx.extras.get("session_factory")
        if factory is None:
            raise arm_api.ArmError(
                "pool arm requires ctx.extras['session_factory'] "
                "(llm.make_bout_session_factory); the cell wiring provides it"
            )
        self._ctx = ctx
        self._contract = ctx.contract
        self._session = factory(
            ROLE,
            first_extras=llm.first_message_blocks(
                ctx.checkpoint,
                ctx.contract,
                protocol=POOL_PROTOCOL,
                budget_remaining=ctx.budget,
            ),
        )
        self.internal_duplicate_count = 0
        self._pending_messages: list[str] = []
        self._consecutive_failures = 0

    def report_outcome(self, params: dict, feedback, *, incumbent_before: float) -> None:
        """Append the authoritative result of the EXECUTED config; it rides on
        the next ask (outcome blocks are per-evaluation appends, PLAN §四).

        ``incumbent_before`` must be captured by the arm BEFORE yielding its
        proposal: by the time the feedback arrives, the runner has already
        advanced ``ctx.state.incumbent_score`` — reading it here would
        compare the new score against itself and misreport every strict
        improvement as "did NOT improve". Shows the CAST params that actually
        ran (arm_api checklist #2), not the raw receipt values."""
        shown = (
            feedback.executed_params
            if feedback.executed_params is not None
            else params
        )
        if feedback.kind == "outcome":
            message = llm.outcome_message(
                shown,
                status=feedback.status,
                score=feedback.score,
                incumbent_score=incumbent_before,
                budget_remaining=self._ctx.state.budget_remaining,
            )
        else:
            message = (
                f"config: {_compact(shown)}\n"
                f"result: REJECTED by the {feedback.stage} preflight "
                f"({feedback.reason}); no budget was consumed and no score "
                "exists. Do not propose it again."
            )
        self._pending_messages.append(message)

    def ask_pool(self) -> dict:
        """Return {"pool": filtered configs in proposer rank order,
        "pool_ranked": pre-filter configs in proposer rank order,
        "pool_duplicate_mask": per-ranked-member True when dropped as an
        executed-history duplicate, "order": the proposer's raw permutation,
        "rationale": str|None, "attempts": asks used}. Raises ArmError after
        3 consecutive failed attempts.

        ``pool_ranked`` / ``order`` / ``pool_duplicate_mask`` exist so arms
        can persist the complete pool and original order (PLAN §6.4) — rank
        scores (``pool_eis`` etc.) are position indexes into the ranked pool
        and uninterpretable without it.
        """
        attempts = 0
        while True:
            attempts += 1
            extra: dict = {}
            if self._pending_messages:
                extra[llm.OUTCOME_KEY] = "\n\n".join(self._pending_messages)
            try:
                receipt = self._session.ask(extra=extra)
            except Exception as exc:  # InvocationFailed: schema retries exhausted
                self._fail(f"proposer invocation failed: {exc}")
                continue
            # Clear only after a successful ask: a failed invocation never
            # entered the transcript, so the pending outcomes must survive
            # it and ride the retry.
            self._pending_messages = []
            problems = self._validate_receipt(receipt)
            if problems:
                self._pending_messages.append(
                    "correction: your last pool was rejected — "
                    + "; ".join(problems)
                    + ". Generate a fresh pool of exactly 5 configs fixing "
                    "exactly these problems."
                )
                self._fail("; ".join(problems))
                continue
            configs = receipt["configs"]
            ranked = [configs[index] for index in receipt["order"]]
            pool = []
            duplicate_mask = []
            for config in ranked:
                if self._is_executed_duplicate(config):
                    self.internal_duplicate_count += 1
                    duplicate_mask.append(True)
                    continue
                duplicate_mask.append(False)
                pool.append(config)
            if not pool:
                self._pending_messages.append(
                    "correction: every pool member duplicated executed "
                    "history and was filtered out. Generate a fresh pool of "
                    "configs you have NOT seen evaluated."
                )
                self._fail("all-duplicate pool after filtering")
                continue
            self._consecutive_failures = 0
            return {
                "pool": pool,
                "pool_ranked": ranked,
                "pool_duplicate_mask": duplicate_mask,
                "order": list(receipt["order"]),
                "rationale": receipt.get("rationale"),
                "attempts": attempts,
            }

    def _fail(self, reason: str) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILED_ATTEMPTS:
            raise arm_api.ArmError(
                f"pool proposer: {MAX_CONSECUTIVE_FAILED_ATTEMPTS} consecutive "
                f"failed attempts (last: {reason})"
            )

    def _validate_receipt(self, receipt) -> list[str]:
        """Semantic checks the bench role deliberately does not enforce."""
        if not isinstance(receipt, dict):
            return [f"receipt is not an object: {type(receipt).__name__}"]
        configs = receipt.get("configs")
        order = receipt.get("order")
        if not isinstance(configs, list) or len(configs) != arm_api.POOL:
            return [f"configs must be a list of exactly {arm_api.POOL} dicts"]
        problems: list[str] = []
        names = [dim.name for dim in self._contract.dimensions]
        identities: set[str] = set()
        for index, config in enumerate(configs):
            if not isinstance(config, dict):
                problems.append(f"configs[{index}] is not an object")
                continue
            if sorted(config) != sorted(names):
                problems.append(
                    f"configs[{index}] keys {sorted(config)} != declared "
                    f"parameters {sorted(names)}"
                )
                continue
            try:
                cast = self._contract.cast(config)
            except (TypeError, ValueError, ArithmeticError) as exc:
                problems.append(f"configs[{index}] does not cast: {exc}")
                continue
            violations = tune_tools._bounds_violations(
                cast, self._contract.search_space
            )
            if violations:
                problems.append(f"configs[{index}] is out of space: {violations}")
                continue
            identity = self._contract.params_identity(cast)
            if identity in identities:
                problems.append(f"configs[{index}] duplicates another pool member")
                continue
            identities.add(identity)
        if problems:
            return problems
        if (
            not isinstance(order, list)
            or len(order) != arm_api.POOL
            # Type check must precede sorted(): mixed-type items raise
            # TypeError there and would escape the correction channel.
            or any(isinstance(item, bool) or not isinstance(item, int) for item in order)
            or sorted(order) != list(range(arm_api.POOL))
        ):
            problems.append(
                f"order must be a permutation of 0..{arm_api.POOL - 1}, got {order!r}"
            )
        return problems

    def _is_executed_duplicate(self, config: dict) -> bool:
        return self._contract.is_duplicate(
            config,
            [
                trial.config
                for trial in self._ctx.state.trials
                if trial.status in _EXECUTED
            ],
        )

    def totals(self) -> dict:
        """Aggregate keys for ctx.emit (arm_api.AGGREGATE_ARM_STATE_KEYS)."""
        return {
            **self._session.totals(),
            "internal_duplicate_count": self.internal_duplicate_count,
        }


def pool_persistence_state(result: dict) -> dict:
    """arm_state keys persisting the complete pool (PLAN §6.4): the pre-
    filter pool in proposer rank order, the proposer's raw permutation over
    its declaration order (``pool_configs[k] == raw_configs[pool_order[k]]``),
    and which ranked members were dropped as executed-history duplicates.
    Rank scores (``pool_eis`` etc.) index into the filtered pool — i.e.
    ``pool_configs`` with the masked members dropped, in order.
    """
    return {
        "pool_configs": result["pool_ranked"],
        "pool_order": result["order"],
        "pool_duplicate_mask": result["pool_duplicate_mask"],
    }


def _compact(params: dict) -> str:
    return json.dumps(params, separators=(",", ":"), ensure_ascii=False, default=str)


__all__ = ["POOL_PROTOCOL", "PoolDriver", "pool_persistence_state"]
