"""Arm protocol for the inner-tuner benchmark (PLAN §5.1, §六).

Every one of the 9 arms runs under the same generator-based protocol machine
(``runner.run_cell``). The generator shape handles paired proposals (active-set
x+/x-, SPSA ±) and internal queues naturally: the arm holds its own control
flow between ``yield`` points.

Fixed constants (PLAN §一):

- ``B = 10`` — objective evaluations per cell (run_cell takes ``budget=B`` by
  default; tests may pass smaller budgets).
- ``POOL = 5`` — pool size for the pool-rank arms.
- ``WARMUP = 8`` — finite/unique/executed history count below which numerical
  rankers fall back to the LLM's own rank.

Arm shape
---------

An arm is an object (or any callable returning one) with:

- ``name: str`` — recorded in manifest/result/events.
- ``run(ctx) -> Generator[Proposal, Feedback, None]`` — the protocol body.
  Must be a generator function (or return a real generator).
- ``active_dimensions(contract) -> int | None`` — OPTIONAL, duck-typed. The
  number of search-space dimensions this arm actually moves (e.g. SPSA's
  continuous-dimension count), recorded in the manifest. Absent or returning
  None records null.
- ``calibration_report(ctx) -> dict | None`` — OPTIONAL, duck-typed. Called
  once at cell start (before the first proposal) for arm-specific calibration
  values (e.g. SPSA ``s`` / ``a_0``), merged into the manifest ``extra`` slot.

A hook that raises ends the cell as ``arm_error`` (contained by the runner:
the manifest is still written with null/{} fallbacks for the fields that
never came back; the failure does not propagate).

Protocol
--------

- The arm yields a ``Proposal``; the runner sends back exactly one ``Feedback``
  per proposal (``generator.send``).
- Arms are expected to propose until the runner stops asking. When the budget
  is exhausted the runner CLOSES the generator without sending the final
  feedback — the pending ``yield`` raises ``GeneratorExit``. Use
  ``try/finally`` for cleanup; do not rely on receiving the last feedback.
- A generator that RETURNS while budget remains is recorded as ``arm_error``
  ("arm exhausted early").
- Raise ``Unsupported`` when this arm's protocol cannot run on the checkpoint
  (e.g. SPSA with zero continuous dimensions; active-set with an empty
  feasible set at any poll start). The cell ends ``unsupported`` whether raised
  before the first proposal or mid-cell; partial history is kept.
- Raise ``ArmError`` for the arm's own protocol failures (duplicate-resample
  exhaustion, pool parse failures, ...). The cell ends ``arm_error``. Note the
  5-consecutive-preflight-reject tripwire is the RUNNER's trigger, not the
  arm's — arms just keep receiving ``preflight_rejected`` feedback.
- Any other exception escaping the generator ends the cell ``arm_error`` with
  the traceback recorded in result.json; it does not propagate.

Runner guarantees (what arms can rely on)
-----------------------------------------

- Deterministic preflight runs on every proposal before anything else: params
  must cast cleanly via the contract (``schema_invalid``), be in-bounds per
  SEARCH_SPACE (``out_of_space``), and not exact-duplicate any executed
  history row (``duplicate``). Rejections consume NO budget and come back as
  ``Feedback.preflight_rejected(stage="deterministic", reason=...)``.
- Otherwise the task preflight runs; rejections come back as
  ``Feedback.preflight_rejected(stage="task", reason=...)`` and consume NO
  budget either. A task-rejected config is NOT added to the duplicate set (the
  rejection may be transient), so re-proposing it costs another full preflight
  run — it is the arm's job not to. Both kinds of rejection are appended to
  ``ctx.state.trials`` with status ``preflight_rejected`` and score None, so an
  arm can introspect what bounced without keeping its own book.
- Otherwise the objective starts: 1 budget is consumed, the outcome comes back
  as ``Feedback.outcome(status="ok"|"crash", score=float|None, detail=...)``.
  A crash consumes budget and never improves the incumbent.
- 5 CONSECUTIVE preflight rejections (deterministic and task counted together)
  terminate the cell as ``arm_error``. Any objective start resets the counter.
- The search space cannot be legally exhausted: measured over the full real
  corpus (333 readable candidate contracts), 331 have at least one continuous
  dimension and the 2 fully discrete ones have 1.1e7 / 1.5e8 configurations —
  none is anywhere near B=10. So "every legal config is already a duplicate"
  is not a state arms need to handle, and there is no terminal status for it.

Metric aggregation key contract
-------------------------------

The runner sums numeric values reported under exactly these keys in event
``arm_state`` (from ``Proposal.arm_state`` and/or ``ctx.emit``) into
result.json (default 0): ``llm_calls``, ``llm_input_tokens``,
``llm_output_tokens``, ``ranker_fallback_count``, ``internal_duplicate_count``,
``internal_resample_count`` (``AGGREGATE_ARM_STATE_KEYS``). Arms that make LLM
calls or hit a ranker fallback MUST report them under these exact keys.

``internal_duplicate_count`` / ``internal_resample_count`` are the arm's OWN
duplicate handling: a candidate the arm generated, recognized as a duplicate of
executed history, and dropped or resampled BEFORE yielding it. The runner
cannot see those — ``counts.duplicates`` only records proposals that reached
preflight — so an arm that dedupes well and an arm that never generates
duplicates are otherwise indistinguishable in §九.

Non-numeric values are ignored. On a key collision between
``Proposal.arm_state`` and ``ctx.emit`` values for the same event, the emit
value WINS (shadowing, not summation). Values emitted after the arm's last
proposal are flushed onto the ``cell_end`` event, so a final tally in a
``finally`` block is still aggregated.

State access rule
-----------------

``ctx.state`` is the live, runner-owned CellState. Arms may READ it freely
(incumbent, trials, budget_remaining, finite_unique_history) but must NOT
mutate it: the only way an arm changes state is through its own proposals
coming back as outcomes. (The one sanctioned arm-owned slot is
``ctx.state.arm_state`` semantics belong to the runner; arms expose their
per-event state via ``Proposal.arm_state`` / ``ctx.emit`` instead.)

Author checklist (things that bite)
-----------------------------------

1. ``Proposal.params`` and ``arm_state`` values must be JSON-native. A
   ``set`` / ``Path`` / custom object is written AFTER the objective already
   ran; the runner contains the failure, but the cell ends ``arm_error`` with
   its budget spent. ``np.float64`` / ``np.int64`` / ``ndarray`` / ``np.bool_``
   survive (the artifacts layer narrows them), but ``bool(...)`` /
   ``float(...)`` / ``.tolist()`` at the boundary is the honest habit.
2. What you propose is not necessarily what runs. Everything downstream uses
   the CAST config, and the production int cast truncates toward zero (3.7 runs
   as 3). ``Feedback.executed_params`` tells you what actually ran; going
   through ``ctx.codec`` (nearest, half-up) avoids the surprise entirely.
3. Duplicate detection covers every executed config INCLUDING crashed ones and
   the checkpoint incumbent. Proposing the incumbent buys a ``duplicate``
   rejection, not an evaluation.
4. Two WARMUP counters exist and can differ:
   ``ctx.checkpoint.finite_unique_history(ctx.contract)`` is the frozen count,
   ``ctx.state.finite_unique_history()`` the live one (it grows with your own
   outcomes). Say in your arm which one gates your fallback.
5. Third-party optimizers must be seeded from ``ctx.seed`` (or an RNG derived
   from ``ctx.rng``). The runner does NOT isolate or reset global ``random`` /
   ``np.random``, so an unseeded library call makes the cell irreproducible
   with no visible trace.
6. There is no clean early exit. Returning while budget remains is
   ``arm_error``. If your protocol genuinely cannot continue, raise
   ``Unsupported`` (protocol does not apply) or ``ArmError`` (it broke) with a
   reason — never silently keep proposing duplicates.
7. Your ``finally`` runs on the SUCCESS path (budget exhaustion closes the
   generator). A raising cleanup no longer relabels the cell, but it is
   recorded as ``cleanup_traceback`` in result.json — keep cleanup total.
8. Paired arms (active-set x+/x-, SPSA ±) get no runner support for a half
   pair. The §6.1 feasible set only guarantees DETERMINISTIC preflight; the
   TASK preflight can still reject x- after x+ spent its budget, with no
   refund. Decide the rule (skip the update / treat as crash / re-poll) and
   document it in the arm.
9. Whether one arm object is reused across cells is UNDEFINED, and
   ``calibration_report(ctx)`` runs before ``run(ctx)`` with no channel between
   them. Do not stash per-cell state on ``self`` — it would leak across cells.
10. The bench role receipts are schema-checked, NOT semantically checked (the
    roles carry no postconditions, deliberately). A pool receipt with 4 configs
    instead of POOL, a repeated config, or an ``order`` that is not a
    permutation all validate. Enforcing pool cardinality / uniqueness /
    permutation is the ARM's job; the role's corrective_attempts give you a
    retry channel when you reject one.
11. The search space cannot be legally exhausted on real checkpoints (see the
    protocol section) — do not build an exhaustion path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generator

if TYPE_CHECKING:  # annotations only; keep import time stdlib-light
    import random

    import numpy as np

    import checkpoint as _checkpoint
    import codec as _codec
    import space as _space
    import state as _state

B = 10
POOL = 5
WARMUP = 8

AGGREGATE_ARM_STATE_KEYS = (
    "llm_calls",
    "llm_input_tokens",
    "llm_output_tokens",
    "ranker_fallback_count",
    "internal_duplicate_count",
    "internal_resample_count",
)


@dataclass(frozen=True)
class Proposal:
    """One arm proposal handed to the runner.

    ``arm_state`` is a snapshot merged into the event's arm-state field (e.g.
    TR radius, SPSA pair id, pool rank scores). Values must be
    JSON-serializable. Numeric values under AGGREGATE_ARM_STATE_KEYS are summed
    into result.json by the runner.
    """

    params: dict
    source: str  # e.g. "tpe", "rewarm", "deferred", "anchor_probe", "pool_rank1"
    rationale: str | None = None
    arm_state: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Feedback:
    """The runner's answer to exactly one Proposal.

    kind="outcome": the objective ran (budget consumed) — ``status`` is
    "ok"|"crash", ``score`` a finite float or None (crash), ``detail`` the
    error tail on crash, ``executed_params`` the CAST params that actually ran
    (production int cast truncates toward zero, so a proposed 3.7 runs as 3 —
    compare against your proposal if that matters to you).

    kind="preflight_rejected": the proposal was never executed (no budget
    consumed) — ``stage`` is "deterministic"|"task", ``reason`` a human/log
    string that starts with the machine reason for deterministic rejects
    ("schema_invalid"/"out_of_space"/"duplicate").
    """

    kind: str  # "outcome" | "preflight_rejected"
    status: str | None = None  # outcome only: "ok" | "crash"
    score: float | None = None  # outcome only: finite float or None (crash)
    detail: str | None = None  # outcome only: error tail on crash
    stage: str | None = None  # preflight_rejected only: "deterministic" | "task"
    reason: str | None = None  # preflight_rejected only
    executed_params: dict | None = None  # outcome only: the CAST params that ran

    def __post_init__(self) -> None:
        if self.kind not in ("outcome", "preflight_rejected"):
            raise ValueError(f"unknown feedback kind {self.kind!r}")

    @classmethod
    def outcome(
        cls,
        *,
        status: str,
        score: float | None,
        detail: str | None = None,
        executed_params: dict | None = None,
    ) -> "Feedback":
        return cls(
            kind="outcome",
            status=status,
            score=score,
            detail=detail,
            executed_params=executed_params,
        )

    @classmethod
    def preflight_rejected(cls, *, stage: str, reason: str) -> "Feedback":
        return cls(kind="preflight_rejected", stage=stage, reason=reason)


class Unsupported(Exception):
    """The arm protocol cannot run on this checkpoint (reason in the message).

    Cell ends "unsupported" whether raised before the first proposal or
    mid-cell."""


class ArmError(Exception):
    """The arm malfunctioned (reason in the message). Cell ends "arm_error".

    Raised by arms for their own protocol failures; the runner also records
    "arm_error" itself for runner-detected violations (early return, non-Proposal
    yields, 5 consecutive preflight rejections)."""


@dataclass
class CellContext:
    """Everything the runner hands to the arm at cell start.

    - ``contract``: space.CandidateContract for the frozen candidate.
    - ``codec``: codec.Codec built on the contract (normalized z-space).
    - ``checkpoint``: checkpoint.Checkpoint (authoritative frozen state).
    - ``state``: live state.CellState — READ-ONLY for arms (see module docstring).
    - ``rng`` / ``np_rng``: the cell's seeded random.Random / numpy Generator;
      ALL arm stochasticity must derive from these (PLAN §5.3).
    - ``seed``: the cell's integer seed. Pass it to any third-party optimizer
      that seeds itself (optuna TPESampler(seed=...), HEBO, sklearn
      random_state) — the runner does NOT isolate or reset global
      random/np.random, so an unseeded library call silently makes the cell
      irreproducible.
    - ``budget``: the INITIAL budget (live remaining budget is on state).
    - ``extras``: arm-specific wiring (e.g. deferred_configs for Current only,
      LLM session factory for LLM arms); {} for arms that must not see them.
    - ``emit(values: dict)``: convenience to merge arm-specific log extras into
      the NEXT proposal's event arm_state. Only meaningful while the arm holds
      control (between receiving a feedback and yielding the next proposal).
    """

    contract: "_space.CandidateContract"
    codec: "_codec.Codec"
    state: "_state.CellState"
    checkpoint: "_checkpoint.Checkpoint"
    rng: "random.Random"
    np_rng: "np.random.Generator"
    budget: int
    seed: int = 0
    extras: dict = field(default_factory=dict)
    emit: Callable[[dict], None] = lambda values: None


__all__ = [
    "AGGREGATE_ARM_STATE_KEYS",
    "ArmError",
    "B",
    "CellContext",
    "Feedback",
    "POOL",
    "Proposal",
    "Unsupported",
    "WARMUP",
]
