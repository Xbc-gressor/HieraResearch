"""Full-remaining-budget paired rollout (design §7).

Every root action is simulated under a frozen reference policy until the
state is terminal, and scored by the raw global improvement it reaches:

    Q(S, a) = E[ g(S) - g_terminal | a_0 = a, a_{t>0} ~ pi_ref ]

**Why the world model may be this coarse.** The rollout is not trying to
predict which candidate arrives in round 7. It estimates a *difference*,
Q(TUNE(i)) - Q(DEFER). Most future randomness is shared between the two
branches, so pairing cancels it and leaves the part the root action
actually causes.

**Why pairing must key on event identity.** Sharing a seed and drawing in
call order is not enough, and this is a design contract rather than an
implementation detail. TUNE consumes a tuning draw first; DEFER consumes an
arrival draw first. Under seed-order pairing the two branches read the same
random stream at different offsets, so scenario `m` means a different
future in each branch and the variance reduction silently becomes
variance *injection*. Randomness is therefore addressed by what the draw is
*for*:

    arrival at future round k           -> (scenario, "arrival", k)
    candidate c's prospective bout b    -> (scenario, "tune", c, b)

Two branches that reach the same future event get the same draw; branches
diverge only where the root action genuinely changed which events exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

from .contract import CandidateView
from .evidence import (
    ArrivalModel,
    TuningModel,
    VALID,
    admitted_prefix,
    bout_class,
)
from .state import SchedulerState


def crn_draw(scenario_id: int, *event_key) -> float:
    """A uniform draw addressed by scenario and future-event identity.

    Deterministic and stateless: the same event in any branch, in any
    process, in a replay months later, yields the same number. That is what
    makes the pairing reproducible rather than merely seeded.
    """
    key = "|".join([str(scenario_id), *(str(part) for part in event_key)])
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return struct.unpack("<Q", digest)[0] / float(1 << 64)


@dataclass(frozen=True)
class ReferencePolicy:
    """The frozen `pi_ref` used inside rollouts (never for the real choice).

    Greedy on immediate expected global improvement with a bout-count
    tie-break, and it tunes whenever any eligible candidate exists. It is
    deliberately simple and versioned: rollout values are only comparable
    across decisions if the continuation policy is fixed, and every receipt
    records which version produced them.

    `-r2` marks the round structure this policy is evaluated under: one
    `pi_ref` choice per driver round, each round followed by a generation
    episode. Values produced before that are not comparable with these.
    """

    version: str = "greedy-headroom-r2"

    def choose(self, state: SchedulerState) -> tuple[str, str | None]:
        eligible = state.eligible()
        if not eligible:
            return ("DEFER", None) if state.defer_available() else ("STOP", None)
        # Prefer the candidate closest to the global best (smallest
        # headroom): its gain has the shortest distance to travel before it
        # moves the run's raw best. Fewer bouts breaks ties — an untouched
        # candidate has more of its gain distribution ahead of it.
        best = min(
            eligible,
            key=lambda candidate: (
                state.headroom(candidate),
                candidate.bouts_used,
                candidate.run_id,
            ),
        )
        return ("TUNE", best.run_id)


@dataclass(frozen=True)
class RolloutConfig:
    """Experiment/implementation configuration (§7, §13: not frozen design)."""

    scenarios: int = 64
    tie_band: float = 1e-9
    reference_policy: ReferencePolicy = ReferencePolicy()
    #: Hard bound on simulated steps. Every step either charges a bout or
    #: admits an arrival, so a full-budget rollout terminates on its own;
    #: this only stops a pathological zero-cost episode from spinning.
    max_steps: int = 4096


class _ScenarioClock:
    """Per-scenario counters for future-event identity.

    Future generation rounds are numbered from the root, so both branches
    call the arrival at "future round k" by the same name. Bouts are
    identified by `(candidate, prospective bout index)` — that index is a
    property of the candidate's own history, so a candidate reached by two
    different paths still keys its next bout identically.
    """

    def __init__(self, scenario_id: int):
        self.scenario_id = scenario_id
        self.future_round = 0

    def arrival(self) -> float:
        draw = crn_draw(self.scenario_id, "arrival", self.future_round)
        self.future_round += 1
        return draw

    def tune(self, candidate: CandidateView) -> float:
        return crn_draw(
            self.scenario_id, "tune", candidate.run_id, candidate.bouts_used
        )


def _bout_cost(outcome, cap: int) -> int:
    """What a resampled bout charges the simulated budget.

    The observed cost is authoritative, capped at the bout's regime cost
    (``B_FIRST`` for a first bout, ``B`` otherwise) because a bout can never
    admit more. Zero is a real observation, not a
    missing one: a bout whose whole method chain was rejected reserved no
    objective slot, so charging it a full `B` would make the simulated
    budget drain faster than the real one and shorten every rollout that
    resampled it. It still burns one of the candidate's `MAX_BOUTS`, so a
    free bout cannot be repeated without end.
    """
    observed = int(outcome.cost)
    return max(0, min(observed, cap))


def _next_round_generation(
    state: SchedulerState,
    scenario_id: int,
    counter: list[int],
    clock: _ScenarioClock,
    arrival: ArrivalModel,
) -> SchedulerState:
    """Admit the next round's generation episode, whatever step 3 just did.

    The driver's round is `generation -> at most one bout` and step 2 is
    not conditioned on step 3 (`experiment.py`), so a round boundary always
    offers a generation episode. An episode the budget cannot admit at all
    contributes nothing and is not terminal by itself: the run can still
    spend what is left on bouts.
    """
    if not arrival.supported():
        return state
    episode = arrival.sample(clock.arrival())
    # Charge admission at the contract's k_eval — the same predicate
    # defer_available divides by — not the episode's recorded cost, so the
    # simulated DEFER admits at least one arrival exactly when the action
    # set said DEFER was available.
    prefix = admitted_prefix(
        episode, state.remaining_budget, per_candidate_cost=state.contract.k_eval
    )
    if not prefix:
        return state
    # Every gap in one episode is anchored to the SAME pre-episode global
    # best (equation (7)), even after an earlier admitted candidate in this
    # episode already improved it. The gaps were observed relative to one
    # anchor; re-anchoring mid-episode would compound them into a
    # fictitious sequence.
    anchor = state.global_best
    arrivals = []
    for gap, cost in prefix:
        counter[0] += 1
        arrivals.append(
            (
                f"sim-{scenario_id}-{counter[0]}",
                None if gap is None else anchor + gap,
                cost,
            )
        )
    return state.admit_arrivals(arrivals)


def _simulate(
    state: SchedulerState,
    root_action: tuple[str, str | None],
    scenario_id: int,
    tuning: TuningModel,
    arrival: ArrivalModel,
    config: RolloutConfig,
) -> float:
    """One scenario: apply the root action, then follow pi_ref to terminal.

    The simulated round mirrors the driver's, which is what makes `Q` a
    counterfactual about the real system. In `experiment.py` a round runs
    step 2 (generate + warm evaluate, **unconditionally**) and then step 3
    (at most one tuning bout, which is where the scheduler is asked). The
    scheduler therefore chooses *after* this round's candidates already
    exist, and the next generation episode arrives whether step 3 tuned or
    not.

    So arrival is not something DEFER purchases. Charging it only to the
    DEFER branch — and letting the TUNE branch chain bouts with no
    intervening generation — misstates both actions: it credits DEFER with
    candidates TUNE would also have received, and it lets TUNE postpone
    generation in a way the driver cannot. What actually separates the two
    is budget: a bout spends `B` evaluations that would otherwise fund
    later rounds and later bouts, and DEFER keeps that `B` plus the option
    to spend it on a candidate chosen with one more round of information.
    """
    clock = _ScenarioClock(scenario_id)
    initial_best = state.global_best
    action = root_action
    counter = [0]

    for _ in range(config.max_steps):
        kind, target = action
        if kind == "STOP":
            break
        if kind == "TUNE":
            candidate = next(
                (c for c in state.candidates if c.run_id == target), None
            )
            if candidate is None:
                break
            klass = bout_class(candidate.bouts_used)
            if not tuning.supported(klass):
                break
            outcome = tuning.sample(klass, clock.tune(candidate))
            # `Z` decides whether the bout's `D` counts at all: a failed or
            # scientifically invalid bout still consumed its evaluations and
            # still burns one of the candidate's bouts, but it moves no
            # score. Applying its sampled `D` anyway would let failures
            # improve the simulated run.
            gain = outcome.gain if outcome.status == VALID else 0.0
            state = state.apply_bout(
                candidate.run_id,
                gain,
                cost=_bout_cost(
                    outcome, state.contract.bout_cost(candidate.bouts_used)
                ),
            )
        elif kind != "DEFER":
            break
        # DEFER itself changes nothing: it declines this round's bout. Its
        # value shows up as the budget it did not spend.

        before = state
        state = _next_round_generation(state, scenario_id, counter, clock, arrival)
        if kind == "DEFER" and state is before:
            # A DEFER round that admitted no arrival changed nothing at
            # all. Admission is charged at the contract's k_eval, the same
            # predicate `defer_available` uses, so with a supported arrival
            # model this cannot happen; it remains reachable only when
            # arrival is unsupported. Without the backstop the scenario
            # would spin to `max_steps` and report the same terminal best —
            # a silent stall rather than an answer.
            break
        if state.terminal():
            break
        action = config.reference_policy.choose(state)

    return initial_best - state.global_best


@dataclass(frozen=True)
class ActionValue:
    """Monte-Carlo value of one root action, with its paired sample."""

    action: str
    run_id: str | None
    q_hat: float
    samples: tuple[float, ...]

    @property
    def key(self) -> tuple[str, str | None]:
        return (self.action, self.run_id)


def evaluate_actions(
    state: SchedulerState,
    tuning: TuningModel,
    arrival: ArrivalModel,
    config: RolloutConfig | None = None,
) -> list[ActionValue]:
    """Estimate Q for every root action over shared paired scenarios."""
    config = config or RolloutConfig()
    values = []
    for action, run_id in state.actions():
        samples = tuple(
            _simulate(
                state,
                (action, run_id),
                scenario_id,
                tuning,
                arrival,
                config,
            )
            for scenario_id in range(config.scenarios)
        )
        values.append(
            ActionValue(
                action=action,
                run_id=run_id,
                q_hat=sum(samples) / len(samples) if samples else 0.0,
                samples=samples,
            )
        )
    return values


def paired_difference(
    champion: ActionValue,
    runner_up: ActionValue,
) -> dict:
    """Paired mean difference and its standard error.

    Because scenario `m` is the same future in both branches, the paired
    per-scenario differences — not the two independent Q estimates — carry
    the comparison. The SE below is over those differences, which is the
    quantity that tells the policy whether a margin is resolvable at all.
    """
    pairs = list(zip(champion.samples, runner_up.samples))
    if not pairs:
        return {"mean": 0.0, "se": None, "n": 0}
    diffs = [a - b for a, b in pairs]
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return {"mean": mean, "se": None, "n": n}
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return {"mean": mean, "se": (variance / n) ** 0.5, "n": n}
