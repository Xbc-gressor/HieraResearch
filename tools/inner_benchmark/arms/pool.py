"""Pool-proposer driver shared by the pool-rank arms (PLAN §6.4) and the
mixup/alt HEBO arms (PLAN-inner-arms-mixup-alt §3/§4).

One bout-scoped ``bench-pool-proposer`` session per cell; each step asks for
the driver's configured number of unique configs (POOL=5 by default) + the
proposer's self-ranking in ONE call. This module owns the arm-side obligations
the bench roles deliberately do NOT enforce (arm_api author checklist #10):

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

def pool_protocol(pool_size: int) -> str:
    count_word = "five" if pool_size == 5 else str(pool_size)
    return (
        "LLM pool protocol (PLAN §6.4): each step you generate exactly "
        f"POOL={pool_size} complete, mutually distinct candidate configs in "
        "one call, ranked by your own judgment (order[0] is the config you "
        "most want executed). A deterministic selector then executes exactly "
        "ONE config from the pool and appends its authoritative outcome. "
        "Unexecuted pool members are not outcome evidence; every step asks "
        "for a fresh pool. If recent evidence shows the incumbent region is "
        "converged, build the pool to cover genuinely different regions "
        "rather than near-duplicates of the incumbent — "
        f"{count_word} near-identical configs waste the selector's choice."
    )


POOL_PROTOCOL = pool_protocol(arm_api.POOL)

_EXECUTED = ("ok", "crash")

# report_external_outcome tags (alt arm's BO steps, PLAN §4): each BO outcome
# enters the LLM session with an explicit provenance label.
EXTERNAL_OUTCOME_NOTES = {
    "[hebo_probe]": "executed by the BO surrogate, not from your pool",
    "[hebo_quasi]": "Sobol warmup point, not selected by the surrogate",
}


class PoolDriver:
    """Per-cell proposer session driver. One instance per cell (in run()).

    Keyword increments over the base pool protocol (all default to the
    original behavior; PLAN §3/§4 arms use them):

    - ``protocol``: replaces the POOL_PROTOCOL text slot of the first
      message (mixup's seed semantics / alt's alternation semantics);
    - ``first_trials`` / ``first_live_incumbent``: overrides forwarded to
      ``llm.first_message_blocks`` so a session created mid-cell renders
      the LIVE history (including this cell's own quasi/BO rows) and the
      current incumbent instead of the frozen checkpoint snapshot;
    - the budget block reports the live ``state.budget_remaining`` at
      construction (equal to ``ctx.budget`` when the driver is built at
      cell start, as the existing pool arms do).
    """

    def __init__(
        self,
        ctx,
        *,
        protocol: str | None = None,
        pool_size: int = arm_api.POOL,
        first_trials=None,
        first_live_incumbent=None,
    ) -> None:
        self.precheck(ctx)
        factory = ctx.extras["session_factory"]
        self._ctx = ctx
        self._contract = ctx.contract
        self._pool_size = pool_size
        self._session = factory(
            ROLE,
            first_extras=llm.first_message_blocks(
                ctx.checkpoint,
                ctx.contract,
                protocol=(
                    protocol if protocol is not None else pool_protocol(pool_size)
                ),
                budget_remaining=ctx.state.budget_remaining,
                trials=first_trials,
                live_incumbent=first_live_incumbent,
            ),
        )
        self.internal_duplicate_count = 0
        self._pending_messages: list[str] = []
        self._consecutive_failures = 0

    @staticmethod
    def precheck(ctx) -> None:
        """The pool arms' zero-LLM-cost gates, before any session exists.

        Uniform across the pool arms: nothing can move, so every pool would
        all-duplicate into ArmError — mark the cell unsupported instead.
        Arms that defer session creation (alt) call this at arm start.
        """
        if not ctx.contract.varying_dimensions:
            raise arm_api.Unsupported("pool arm: no varying dimensions")
        if ctx.extras.get("session_factory") is None:
            raise arm_api.ArmError(
                "pool arm requires ctx.extras['session_factory'] "
                "(llm.make_bout_session_factory); the cell wiring provides it"
            )

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

    def report_external_outcome(
        self, params: dict, *, status: str, score, incumbent_before: float, tag: str
    ) -> None:
        """Append the outcome of a config executed OUTSIDE the pool (alt
        arm's BO steps), with an explicit provenance tag; rides on the next
        ask exactly like report_outcome (PLAN-inner-arms-mixup-alt §4:
        the BO->LLM information channel).

        ``tag`` is ``[hebo_probe]`` (surrogate-selected point) or
        ``[hebo_quasi]`` (Sobol warmup point); the parenthetical explains
        the provenance so the proposer never mistakes a probe for its own
        executed pool member. ``params`` are the CAST params that ran."""
        try:
            note = EXTERNAL_OUTCOME_NOTES[tag]
        except KeyError:
            raise ValueError(f"unknown external outcome tag {tag!r}") from None
        message = llm.outcome_message(
            params,
            status=status,
            score=score,
            incumbent_score=incumbent_before,
            budget_remaining=self._ctx.state.budget_remaining,
        )
        self._pending_messages.append(f"{tag} ({note})\n{message}")

    def report_external_rejection(
        self, params: dict, *, stage: str, reason: str, tag: str
    ) -> None:
        """Append a tagged preflight rejection for an external proposal.

        Unlike an objective outcome, a rejection has no score and consumes no
        budget.  Keeping it on the same tagged channel is important for alt:
        the proposer must be able to distinguish a rejected Sobol warmup point
        from a rejected surrogate probe.
        """
        try:
            note = EXTERNAL_OUTCOME_NOTES[tag]
        except KeyError:
            raise ValueError(f"unknown external outcome tag {tag!r}") from None
        self._pending_messages.append(
            f"{tag} ({note})\n"
            f"config: {_compact(params)}\n"
            f"result: REJECTED by the {stage} preflight ({reason}); no budget "
            "was consumed and no score exists. Treat this as feasibility "
            "evidence and do not propose it again."
        )

    def push_correction(self, message: str) -> None:
        """Queue an arm-originated correction note on the next ask (e.g.
        alt's rank-1-was-a-duplicate fallback), same channel as the
        driver's own receipt corrections."""
        self._pending_messages.append(f"correction: {message}")

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
                    + f". Generate a fresh pool of exactly {self._pool_size} "
                    "configs fixing "
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
        if not isinstance(configs, list) or len(configs) != self._pool_size:
            return [f"configs must be a list of exactly {self._pool_size} dicts"]
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
            or len(order) != self._pool_size
            # Type check must precede sorted(): mixed-type items raise
            # TypeError there and would escape the correction channel.
            or any(isinstance(item, bool) or not isinstance(item, int) for item in order)
            or sorted(order) != list(range(self._pool_size))
        ):
            problems.append(
                f"order must be a permutation of 0..{self._pool_size - 1}, got {order!r}"
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


__all__ = [
    "POOL_PROTOCOL",
    "PoolDriver",
    "pool_persistence_state",
    "pool_protocol",
]
