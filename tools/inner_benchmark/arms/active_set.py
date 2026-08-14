"""LLM active-set arm (PLAN §6.1): LLM-guided coordinate polling.

The mathematical parent is derivative-free pattern / direct search (Torczon
1997; Audet & Dennis 2006 MADS); the LLM only replaces the "which dimension
does this poll move" selector. This is NOT a MADS reproduction — no
direction-density condition, no convergence claim.

Protocol (all pair arithmetic in the codec's normalized z space):

- Each poll freezes ``anchor = current incumbent`` and computes the feasible
  (parameter, step) joint set over ``STEPS = {0.05, 0.10, 0.20, 0.40}``:
  - continuous: raw ``z-s >= 0`` and ``z+s <= 1`` (no projection remedy —
    an out-of-range step is simply infeasible);
  - integer: both inverse transforms (decode's own z-projection + clamping
    applies) land on legal values DIFFERENT from the anchor's — the PLAN text
    deliberately gives ints a value-based rule, so a boundary-clamped side is
    fine as long as it moves;
  - neither constructed side (x+ / x-) may be an exact duplicate of ANY
    executed history row (checkpoint history, incumbent, own outcomes —
    crashes included); a duplicate on either side kills the whole combo.
  Degenerate dimensions are excluded by these rules alone (a degenerate
  continuous encodes to z=0 so ``z-s < 0``; a degenerate int cannot move), no
  special-casing. Categorical parameters are frozen in this arm.
- The FILTERED feasible set (not the raw bucket) is offered to the LLM under
  ``llm.FEASIBLE_SET_KEY``; the LLM picks exactly ONE (parameter, step
  magnitude) — never the sign. Receipts are only schema-checked by the bench
  role, so membership in the feasible set is validated here (arm_api author
  checklist #10): an out-of-set (parameter, step) — or a failed invocation —
  counts one failed attempt and is re-asked with a ``correction`` block;
  3 consecutive failed attempts end the cell as ``arm_error``.
- x+ and x- are built from the SAME anchor, all other parameters untouched,
  and each costs 1 evaluation. Both outcomes are reported back only after
  BOTH sides complete (``llm.OUTCOME_KEY``, one ``llm.outcome_message`` per
  side scored against the anchor's score — the poll's reference — plus the
  post-poll incumbent); even when x+ already improved, x- still runs from
  the original anchor. The runner's state picks the new incumbent as the best
  of {anchor, x+, x-}.
- An empty feasible set at ANY poll start (including the first) raises
  ``Unsupported`` — never a silent degradation to single-sided probes or
  another search.

Half-pair rule (PLAN leaves this open; arm_api checklist #8): the feasible
set guarantees only the DETERMINISTIC preflight. A task-preflight rejection
of one side simply means that side has no score — the other side is still
evaluated against the unchanged anchor, both results (outcome or rejection)
are fed back to the LLM, and the poll completes normally; a poll whose BOTH
sides are rejected consumed 0 budget and moves straight to the next poll
(the runner's 5-consecutive-reject tripwire is the backstop). No re-poll, no
crash substitution.

``internal_duplicate_count`` counts (parameter, step) combos dropped during
feasible-set construction because a constructed side duplicated executed
history — the runner never sees those, so without this count the arm's
duplicate exposure would be invisible to §九.

One bout-scoped ``bench-active-set`` session per cell (PLAN §四); LLM usage
is emitted from ``session.totals()`` in ``finally``. The arm itself is
deterministic (no RNG): all stochasticity lives in the LLM.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tuners"))

import arm_api  # noqa: E402
import llm  # noqa: E402

ROLE = "bench-active-set"
STEPS = (0.05, 0.10, 0.20, 0.40)
MAX_CONSECUTIVE_FAILED_ATTEMPTS = 3

ACTIVE_SET_PROTOCOL = (
    "LLM active-set protocol (PLAN §6.1, LLM-guided coordinate polling): "
    "each poll freezes the current incumbent as the anchor and offers you "
    "the feasible (parameter, step) joint set. You pick exactly ONE pair; "
    "the controller then evaluates BOTH sides — x+ = anchor+step and x- = "
    "anchor-step on that parameter in normalized [0,1] space, every other "
    "parameter frozen at the anchor — so each selection costs 2 objective "
    "evaluations. You never pick the sign. Both outcomes are reported back "
    "only after BOTH sides complete; even if x+ already improved, x- still "
    "runs from the same anchor. The new incumbent is the best of {anchor, "
    "x+, x-} and anchors the next poll. Categorical parameters are frozen "
    "in this arm. The poll loop repeats until the budget is exhausted. "
    "Prefer axes not yet probed or with real (above-noise) signal; do not "
    "spend a poll refining an axis whose past effects were within the "
    "noise band (~0.005)."
)

_EXECUTED = ("ok", "crash")


class ActiveSet:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "active_set"

    def active_dimensions(self, contract) -> int:
        """Varying numeric dimensions (categoricals are frozen in this arm)."""
        return len([dim for dim in contract.numeric_dimensions
                    if not dim.is_degenerate])

    def run(self, ctx):
        contract = ctx.contract
        codec = ctx.codec
        session = None
        internal_duplicates = 0
        pending: list[str] = []  # messages riding on the next ask's OUTCOME_KEY
        poll_index = 0
        try:
            while True:
                anchor_config = dict(ctx.state.incumbent_config)
                anchor_score = float(ctx.state.incumbent_score)
                anchor_z, anchor_cats = codec.encode(anchor_config)
                feasible, pruned = _feasible_steps(
                    ctx, anchor_config, anchor_z, anchor_cats
                )
                internal_duplicates += pruned
                if not feasible:
                    raise arm_api.Unsupported(
                        "active_set: empty feasible (parameter, step) set at "
                        f"poll {poll_index + 1} start"
                    )
                if session is None:
                    session = _open_session(ctx)
                poll_index += 1
                (parameter, step), choice_rationale = _ask_choice(
                    session,
                    _feasible_set_text(feasible, contract, anchor_config),
                    feasible,
                    pending,
                )
                index = _numeric_index(codec, parameter)
                anchor_identity = contract.params_identity(anchor_config)
                sides: dict = {}
                for side, sign in (("x+", 1.0), ("x-", -1.0)):
                    z = anchor_z.copy()
                    z[index] = anchor_z[index] + sign * step
                    params = codec.decode(z, anchor_cats)
                    feedback = yield arm_api.Proposal(
                        params=params,
                        source="active_set",
                        rationale=(
                            f"poll {poll_index}: {parameter} step {step:g} {side}"
                            + (
                                f" — {choice_rationale}"
                                if isinstance(choice_rationale, str)
                                and choice_rationale
                                else ""
                            )
                        ),
                        arm_state={
                            "poll_index": poll_index,
                            "anchor": anchor_identity,
                            "parameter": parameter,
                            "step": float(step),
                            "side": side,
                        },
                    )
                    # decode() output already went through the production cast,
                    # so params IS what ran on the ok path (checklist #2).
                    sides[side] = (params, feedback)
                pending.append(
                    _poll_outcome_message(
                        poll_index, parameter, step, anchor_score, sides, ctx
                    )
                )
        finally:
            totals = {"internal_duplicate_count": internal_duplicates}
            if session is not None:
                totals = {**session.totals(), **totals}
            ctx.emit(totals)


ARM = ActiveSet()


# --- controller helpers --------------------------------------------------------


def _open_session(ctx):
    factory = ctx.extras.get("session_factory")
    if factory is None:
        raise arm_api.ArmError(
            "active_set requires ctx.extras['session_factory'] "
            "(llm.make_bout_session_factory); the cell wiring provides it"
        )
    return factory(
        ROLE,
        first_extras=llm.first_message_blocks(
            ctx.checkpoint,
            ctx.contract,
            protocol=ACTIVE_SET_PROTOCOL,
            budget_remaining=ctx.budget,
        ),
    )


def _numeric_index(codec, name: str) -> int:
    for index, dim in enumerate(codec.numeric_dimensions):
        if dim.name == name:
            return index
    raise arm_api.ArmError(f"active_set: {name!r} is not a numeric dimension")


def _pair(codec, anchor_z, anchor_cats, index: int, step: float):
    """(x+, x-) from the same anchor; one z coordinate moved, decode projects."""
    z_plus = anchor_z.copy()
    z_plus[index] = anchor_z[index] + step
    z_minus = anchor_z.copy()
    z_minus[index] = anchor_z[index] - step
    return (
        codec.decode(z_plus, anchor_cats),
        codec.decode(z_minus, anchor_cats),
    )


def _feasible_steps(ctx, anchor_config, anchor_z, anchor_cats):
    """The feasible (parameter, step) joint set at this anchor (PLAN §6.1.2).

    -> ({parameter: [feasible steps]}, duplicate_prune_count). One count per
    combo dropped because a constructed side duplicated executed history.
    """
    contract = ctx.contract
    codec = ctx.codec
    executed = [
        trial.config for trial in ctx.state.trials if trial.status in _EXECUTED
    ]
    feasible: dict[str, list[float]] = {}
    pruned = 0
    for index, dim in enumerate(codec.numeric_dimensions):
        z = float(anchor_z[index])
        steps = []
        for step in STEPS:
            if dim.kind == "float" and (z - step < 0.0 or z + step > 1.0):
                continue  # continuous: no projection remedy
            x_plus, x_minus = _pair(codec, anchor_z, anchor_cats, index, step)
            if dim.kind == "int" and (
                x_plus[dim.name] == anchor_config[dim.name]
                or x_minus[dim.name] == anchor_config[dim.name]
            ):
                continue  # integer: both sides must MOVE off the anchor value
            if contract.is_duplicate(x_plus, executed) or contract.is_duplicate(
                x_minus, executed
            ):
                pruned += 1
                continue
            steps.append(step)
        if steps:
            feasible[dim.name] = steps
    return feasible, pruned


def _feasible_set_text(feasible: dict, contract, anchor_config: dict) -> str:
    """One line per feasible parameter listing its feasible step values."""
    lines = [
        "Feasible (parameter, step) pairs for this poll; anchor = current "
        "incumbent. step is a magnitude in normalized [0,1] space: the "
        "controller evaluates BOTH anchor+step and anchor-step on the chosen "
        "parameter from this same anchor; every other parameter stays frozen.",
    ]
    for dim in contract.numeric_dimensions:
        if dim.name not in feasible:
            continue
        desc = f"{dim.kind}{' log-scale' if dim.log else ''} in [{dim.lo}, {dim.hi}]"
        steps = ", ".join(f"{step:g}" for step in feasible[dim.name])
        lines.append(
            f"- {dim.name} ({desc}; anchor value "
            f"{json.dumps(anchor_config[dim.name])}): feasible steps {steps}"
        )
    lines.append(
        "Pick exactly ONE pair from this list: receipt {\"parameter\": <name>, "
        "\"step\": <one of its listed steps>}."
    )
    return "\n".join(lines)


def _validate_choice(receipt, feasible: dict):
    """Semantic receipt check the bench role deliberately does not enforce.

    -> ((parameter, canonical_step), None) or (None, problem).
    """
    if not isinstance(receipt, dict):
        return None, f"receipt is not an object: {type(receipt).__name__}"
    parameter = receipt.get("parameter")
    step = receipt.get("step")
    if not isinstance(parameter, str) or parameter not in feasible:
        return None, (
            f"parameter {parameter!r} is not in the feasible set "
            f"({sorted(feasible)})"
        )
    if isinstance(step, bool) or not isinstance(step, (int, float)):
        return None, f"step {step!r} is not a number"
    for candidate in feasible[parameter]:
        if math.isclose(float(step), candidate, rel_tol=1e-9, abs_tol=1e-12):
            return (parameter, candidate), None
    return None, (
        f"step {step!r} is not feasible for parameter {parameter!r} "
        f"(feasible: {feasible[parameter]})"
    )


def _ask_choice(session, feasible_text: str, feasible: dict, pending: list):
    """Ask until the LLM picks one feasible (parameter, step); ArmError after
    3 consecutive failed attempts (failed invocations count too)."""
    failures = 0
    while True:
        extra = {llm.FEASIBLE_SET_KEY: feasible_text}
        if pending:
            extra[llm.OUTCOME_KEY] = "\n\n".join(pending)
        try:
            receipt = session.ask(extra=extra)
        except Exception as exc:  # InvocationFailed: schema retries exhausted
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILED_ATTEMPTS:
                raise arm_api.ArmError(
                    f"active_set: {MAX_CONSECUTIVE_FAILED_ATTEMPTS} consecutive "
                    f"failed selection attempts (last: invocation failed: {exc})"
                )
            pending.append(
                f"correction: the selection invocation failed ({exc}). Submit "
                "a receipt with 'parameter' and 'step' from the feasible set."
            )
            continue
        if pending:  # delivered on a successful ask; only then drop them
            pending.clear()
        choice, problem = _validate_choice(receipt, feasible)
        if problem is not None:
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILED_ATTEMPTS:
                raise arm_api.ArmError(
                    f"active_set: {MAX_CONSECUTIVE_FAILED_ATTEMPTS} consecutive "
                    f"failed selection attempts (last: {problem})"
                )
            pending.append(
                f"correction: {problem}. Pick exactly one (parameter, step) "
                "pair listed in the feasible_set block."
            )
            continue
        # The LLM's own reasoning rides in the proposal rationale (the poll
        # tag keeps the structured side/parameter attribution), matching the
        # sibling arms' receipt-rationale passthrough.
        return choice, receipt.get("rationale")


def _poll_outcome_message(
    poll_index: int, parameter: str, step: float, anchor_score: float, sides: dict, ctx
) -> str:
    """Both sides' authoritative results, appended after the pair completes
    (PLAN §6.1.5). Each side is scored against the ANCHOR's score — the
    poll's reference — with the post-poll incumbent named at the end."""
    blocks = [
        f"poll {poll_index} outcome — parameter={parameter}, step={step:g}, "
        f"anchor score={anchor_score}:",
    ]
    for side in ("x+", "x-"):
        params, feedback = sides[side]
        if feedback.kind == "outcome":
            body = llm.outcome_message(
                params,
                status=feedback.status,
                score=feedback.score,
                incumbent_score=anchor_score,
                budget_remaining=ctx.state.budget_remaining,
            )
        else:
            body = (
                f"config: {_compact(params)}\n"
                f"result: REJECTED by the {feedback.stage} preflight "
                f"({feedback.reason}); no budget was consumed and no score "
                "exists. This side of the poll simply has no measurement."
            )
        blocks.append(f"--- {side} ---\n{body}")
    blocks.append(
        "incumbent after this poll: score "
        f"{ctx.state.incumbent_score} (the best of {{anchor, x+, x-}} anchors "
        "the next poll)"
    )
    return "\n".join(blocks)


def _compact(params: dict) -> str:
    return json.dumps(params, separators=(",", ":"), ensure_ascii=False, default=str)
