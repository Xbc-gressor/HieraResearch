"""Frozen mechanical resource contract for scheduler v3.2 (design §2).

`B = 10` objective evaluations per tuning bout and `MAX_BOUTS_PER_CANDIDATE
= 4` are the new policy invariant, replacing the legacy per-candidate
attempt cap. Two rules follow and are enforced here rather than at each
call site:

* a bout is admitted at full `B` or not at all — no truncation to fit a
  candidate cap or the tail of the global budget;
* a candidate that has completed `MAX_BOUTS_PER_CANDIDATE` bouts is
  permanently ineligible.

`eligible()` is the single deterministic predicate the policy, the rollout
simulator, and the real execution layer all call. They cannot disagree
about which actions exist, which is what makes a rollout's action set
mean the same thing as the driver's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


BOUT_TRIALS = 10
B_FIRST = 8
MAX_BOUTS_PER_CANDIDATE = 4

#: First bout index of the DEEP regime (0 completed bouts = FIRST, 1 =
#: CONTINUE, >=2 = DEEP) — mirrors the frozen inner policy
#: deferred-random8-hebo10-spsa10-v1 (design §2). A candidate without a
#: non-degenerate continuous dimension has no DEEP action.
DEEP_MIN_BOUTS = 2

#: Objective calls one generation round reserves per admitted candidate.
#: Mirrors ``got_select.cmd_decide``'s ``k_eval`` floor: a non-fresh
#: candidate spends one slot on its inherited fidelity control and needs at
#: least one more for a selectable row.
DEFAULT_K_EVAL = 2


@dataclass(frozen=True)
class ResourceContract:
    """The bout-level resource contract in force for one run."""

    bout_trials: int = BOUT_TRIALS
    max_bouts: int = MAX_BOUTS_PER_CANDIDATE
    k_eval: int = DEFAULT_K_EVAL
    first_bout_trials: int = B_FIRST

    def __post_init__(self) -> None:
        for name in ("bout_trials", "max_bouts", "k_eval", "first_bout_trials"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def bout_cost(self, bouts_used: int) -> int:
        """Objective evaluations one complete bout charges (design §2:
        ``B_FIRST=8`` for a first bout, ``B=10`` for CONTINUE/DEEP)."""
        return self.first_bout_trials if bouts_used == 0 else self.bout_trials

    def lifetime_cost(self) -> int:
        """Objective evaluations one candidate spends across its full
        bout contract (design §2: ``8+10+10+10=38`` under the frozen
        regime-conditioned policy)."""
        if self.max_bouts <= 0:
            return 0
        return self.bout_cost(0) + self.bout_trials * (self.max_bouts - 1)

    def bout_admissible(self, remaining_budget: int, bouts_used: int) -> bool:
        """Whether the remaining global budget funds one complete bout."""
        return remaining_budget >= self.bout_cost(bouts_used)

    def generation_admits(self, remaining_budget: int) -> int:
        """How many candidates the next generation round could admit.

        The same floor division ``got_select`` applies to its action list.
        Judging DEFER feasibility means asking whether this is positive —
        a mechanical question about budget, not a prediction about what the
        generator will actually produce.
        """
        return max(0, remaining_budget // self.k_eval)


@dataclass(frozen=True)
class CandidateView:
    """Scheduler-visible mechanical facts about one candidate.

    Deliberately narrow: these are the fields the eligibility predicate and
    the score transition consume. Diagnostics that are recorded but not
    conditioned on live in the snapshot, not here.
    """

    run_id: str
    best_score: float
    bouts_used: int
    previous_gain: float | None = None
    has_unresolved_descendant: bool = False
    crashed: bool = False
    #: FIRST bouts consume step-0+1 configs that were proposed but never
    #: evaluated. They occupy trial slots inside B; they never add cost
    #: beyond it. Carried so a simulated FIRST bout costs what a real one
    #: costs.
    deferred_warm_backlog: int = 0
    #: DEEP bouts are two-sided SPSA, which needs a non-degenerate
    #: continuous dimension to perturb. Without one the candidate has no
    #: DEEP action (design §2.1) — it is done after its CONTINUE bout.
    has_movable_continuous: bool = True

    @property
    def is_first(self) -> bool:
        return self.bouts_used == 0


def ineligibility_reason(
    candidate: CandidateView,
    remaining_budget: int,
    contract: ResourceContract,
) -> str | None:
    """Why this candidate cannot receive a bout now, or None if it can.

    Ordered from permanent to transient so a receipt records the most
    durable cause.
    """
    if candidate.crashed:
        return "crashed"
    if candidate.bouts_used >= contract.max_bouts:
        return f"bout cap reached ({contract.max_bouts})"
    if candidate.has_unresolved_descendant:
        return "unresolved primary descendant"
    if (
        candidate.bouts_used >= DEEP_MIN_BOUTS
        and not candidate.has_movable_continuous
    ):
        return "no movable continuous dimension for a DEEP bout"
    cost = contract.bout_cost(candidate.bouts_used)
    if remaining_budget < cost:
        return (
            f"remaining budget {remaining_budget} cannot admit a full "
            f"{cost}-trial bout"
        )
    return None


def eligible(
    candidate: CandidateView,
    remaining_budget: int,
    contract: ResourceContract,
) -> bool:
    return ineligibility_reason(candidate, remaining_budget, contract) is None


def eligible_candidates(
    candidates: Iterable[CandidateView],
    remaining_budget: int,
    contract: ResourceContract,
) -> list[CandidateView]:
    return [
        candidate
        for candidate in candidates
        if eligible(candidate, remaining_budget, contract)
    ]


def defer_available(remaining_budget: int, contract: ResourceContract) -> bool:
    """Whether DEFER is a strategy action (design §1).

    DEFER means "start no bout now; take another generation round first,
    then decide with more information". It stops being an action once the
    remaining budget could not admit any candidate next round — at that
    point deferring buys no future option, so the state is terminal rather
    than DEFER-only.
    """
    return contract.generation_admits(remaining_budget) > 0
