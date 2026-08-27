"""Mechanical resource contract shared by current and comparison schedulers.

An exact ``bout_cost_schedule`` carries the current INITIAL/DEEP policy. The
unscheduled B_FIRST/B defaults below encode the historical v3.2 comparison.

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
#: Mirrors ``got_select.cmd_decide``'s per-candidate screening reservation.
#: Three preserves the production experiment setting: a non-fresh candidate
#: scores its inherited control and two alternative rows.
DEFAULT_K_EVAL = 3
MIN_GENERATION_K_EVAL = 2


@dataclass(frozen=True)
class ResourceContract:
    """The bout-level resource contract in force for one run."""

    bout_trials: int = BOUT_TRIALS
    max_bouts: int = MAX_BOUTS_PER_CANDIDATE
    k_eval: int = DEFAULT_K_EVAL
    first_bout_trials: int = B_FIRST
    # Optional exact per-bout schedule for policies whose resource shape is
    # not FIRST + repeated-LATER.  v3.2 leaves this unset; the deterministic
    # anchor/challenger tournament supplies the active inner policy's first
    # three bout costs here.
    bout_cost_schedule: tuple[int, ...] | None = None
    # Candidate-aware first-bout override (transfer-scheduler design §5.2):
    # a ``global_donor`` candidate's first bout is the TRANSFERRED segment
    # priced here instead of the schedule's INITIAL cost. None — every
    # pre-transfer contract — keeps ``bout_cost(bouts_used)`` as the whole
    # story for every candidate.
    transferred_first_bout_trials: int | None = None
    # Some inner policies switch to a numeric-only kernel before the generic
    # DEEP boundary. None means no additional policy-specific requirement.
    numeric_required_from_bout_index: int | None = None

    def __post_init__(self) -> None:
        for name in ("bout_trials", "max_bouts", "k_eval", "first_bout_trials"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.bout_cost_schedule is not None:
            if len(self.bout_cost_schedule) != self.max_bouts:
                raise ValueError(
                    "bout_cost_schedule must contain exactly max_bouts entries"
                )
            for value in self.bout_cost_schedule:
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(
                        "bout_cost_schedule entries must be positive integers"
                    )
        if self.transferred_first_bout_trials is not None:
            if self.bout_cost_schedule is None:
                raise ValueError(
                    "transferred_first_bout_trials requires bout_cost_schedule"
                )
            value = self.transferred_first_bout_trials
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    "transferred_first_bout_trials must be a positive integer"
                )
        required_from = self.numeric_required_from_bout_index
        if required_from is not None and (
            not isinstance(required_from, int)
            or isinstance(required_from, bool)
            or required_from < 0
            or required_from >= self.max_bouts
        ):
            raise ValueError(
                "numeric_required_from_bout_index must be a valid bout index"
            )

    def bout_cost(self, bouts_used: int) -> int:
        """Objective evaluations one complete bout charges (design §2:
        ``B_FIRST=8`` for a first bout, ``B=10`` for CONTINUE/DEEP)."""
        if self.bout_cost_schedule is not None:
            if bouts_used < 0 or bouts_used >= len(self.bout_cost_schedule):
                raise ValueError(f"bout index outside contract: {bouts_used}")
            return self.bout_cost_schedule[bouts_used]
        return self.first_bout_trials if bouts_used == 0 else self.bout_trials

    def bout_cost_for(self, candidate: "CandidateView") -> int:
        """Objective evaluations THIS candidate's next complete bout charges.

        The single candidate-aware cost entry point (transfer-scheduler
        design §5.2): a ``global_donor`` candidate's first bout is the
        TRANSFERRED segment priced by ``transferred_first_bout_trials``.
        Every other case — any later bout, an ordinary candidate, or a
        contract that never set the field — is exactly
        ``bout_cost(candidate.bouts_used)``, so callers never write their own
        mode branch.
        """
        if (
            self.transferred_first_bout_trials is not None
            and candidate.bouts_used == 0
            and candidate.initialization_mode == "global_donor"
        ):
            return self.transferred_first_bout_trials
        return self.bout_cost(candidate.bouts_used)

    def lifetime_cost(self) -> int:
        """Objective evaluations one candidate spends across its full
        bout contract (design §2: ``8+10+10+10=38`` under the frozen
        regime-conditioned policy)."""
        if self.bout_cost_schedule is not None:
            return sum(self.bout_cost_schedule)
        return self.bout_cost(0) + self.bout_trials * (self.max_bouts - 1)

    def bout_admissible(self, remaining_budget: int, bouts_used: int) -> bool:
        """Whether the remaining global budget funds one complete bout."""
        return remaining_budget >= self.bout_cost(bouts_used)

    def screening_reservation(
        self,
        remaining_budget: int,
        *,
        allow_terminal_degrade: bool = False,
    ) -> int:
        """Objective calls reserved for each candidate in the next generation.

        ``k_eval`` remains the normal screening fidelity.  Once no later
        scheduler decision depends on the new candidate, a final generation
        may use the two-row minimum instead of stranding the global budget.
        """
        if (
            allow_terminal_degrade
            and MIN_GENERATION_K_EVAL <= remaining_budget < self.k_eval
        ):
            return remaining_budget
        return self.k_eval

    def generation_admits(
        self,
        remaining_budget: int,
        *,
        allow_terminal_degrade: bool = False,
    ) -> int:
        """How many candidates the next generation round could admit.

        Normal scheduling uses full ``k_eval`` fidelity. Terminal best-effort
        mode may admit one two-row screen when that is all the budget left.
        """
        reservation = self.screening_reservation(
            remaining_budget,
            allow_terminal_degrade=allow_terminal_degrade,
        )
        return max(0, remaining_budget // reservation)


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
    #: Initialization segments consume step-0+1 configs that were proposed
    #: evaluated. They occupy trial slots inside B; they never add cost
    #: beyond it. Carried so a simulated initialization segment costs what a
    #: real one costs.
    deferred_warm_backlog: int = 0
    #: Historical SPSA DEEP bouts need a non-degenerate
    #: continuous dimension to perturb. Without one the candidate has no
    #: DEEP action (design §2.1).
    has_movable_continuous: bool = True
    #: TuRBO can move float and integer dimensions, but not a categorical-only
    #: space. Used only when the active resource contract requests it.
    has_movable_numeric: bool = True
    #: The candidate's stamped Phase-A initialization mode (design §3.4).
    #: ``ordinary`` — including every pre-transfer report — keeps the
    #: schedule's first-bout cost; ``global_donor`` prices the first bout as
    #: the TRANSFERRED segment (see ``ResourceContract.bout_cost_for``).
    initialization_mode: str = "ordinary"
    #: Bound global-donor snapshot id, when Phase A ran one.
    donor_snapshot_id: str | None = None
    #: The donor row was processed by warm evaluation (observation status
    #: other than "not_evaluated").
    donor_evaluated: bool = False
    #: ``global_donor_observation.status == "finite"`` — the TRANSFERRED
    #: challenger-eligibility fact (design §6.1).
    donor_finite: bool = False

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
    numeric_from = contract.numeric_required_from_bout_index
    if (
        numeric_from is not None
        and candidate.bouts_used >= numeric_from
        and not candidate.has_movable_numeric
    ):
        return "no varying numeric dimension for this bout"
    if (
        candidate.bouts_used >= DEEP_MIN_BOUTS
        and not candidate.has_movable_continuous
    ):
        return "no movable continuous dimension for a DEEP bout"
    cost = contract.bout_cost_for(candidate)
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
