"""LLM hillclimb arm (PLAN §6.0) — production hillclimb on the parameter-only
editing surface.

母本 driver/loops/hillclimb.py: one working copy, edit → evaluate →
keep/revert, a persistent editor session, revert-to-best before every fresh
idea. The benchmark narrows the editing surface from the whole train.py to the
checkpoint-declared SEARCH_SPACE parameter values — ONE parameter value change
per invocation (the only narrowing, §6.0) — and the runner owns evaluation,
budget and state:

- working copy: a copy of the frozen train.py in a per-cell tempdir (the only
  file the editor may Read/Edit); removed in ``finally`` (cleanup is total).
- revert-to-best, parameter version: before EVERY invocation the working
  copy's BASE_PARAMS is rewritten to the CURRENT incumbent via production
  ``apply_base_params.apply``. If the previous edit left the file unparseable
  for the sync (apply raises SystemExit/SyntaxError/ValueError — broken
  syntax, deleted/duplicated/non-literal BASE_PARAMS, added keys), the
  pristine frozen source is restored first and the sync retried. Structural
  edits are otherwise left in place — they are inert: evaluation always runs
  the FROZEN candidate with the yielded params, never the working copy.
- session: one bout-scoped ``bench-hillclimb-editor`` session from
  ``ctx.extras['session_factory']``, persistent across the bout (production
  chaining semantics); the first message carries the §四 blocks via
  ``llm.first_message_blocks``.
- per-invocation extra: ``working_copy`` (absolute path), ``outcome`` (the
  previous evaluation's authoritative result — score + KEEP/DISCARD, CRASH, a
  preflight rejection, or the correction for the previous invalid edit;
  absent on the first invocation), and from the second invocation on
  ``history``: the structured BOUT history — this cell's executed trials only
  (≤ B lines; the checkpoint history already rides in the first message),
  ``llm.format_history`` against the bout's starting incumbent. Explicit
  deviation per §6.0: production never pushes scores; analysis describes this
  arm as "hillclimb protocol + full outcome history".
- the production step-0 baseline evaluation is replaced by the checkpoint
  incumbent (no budget spent). A crash scores +inf and is discarded, reported
  back as a failure — no production diagnosis/repair loop.

Edit-loop failure rules (arm-owned; the bench role carries no postconditions):

- INVALID EDIT, nothing yielded: ``edited=false`` (editor declined), an
  unreadable BASE_PARAMS (file left syntactically broken or non-literal), an
  invocation failure, a read-back config identical to the incumbent (an
  exact duplicate the runner would reject — counted under
  ``internal_duplicate_count``, arm_api metric contract), or a read-back
  config changing MORE THAN ONE declared parameter value (the §6.0
  one-change narrowing, enforced at read-back via the production cast +
  type-strict categorical comparison). The failure rides on the next
  invocation's ``outcome`` as a correction. 3 CONSECUTIVE invalid edits ->
  ArmError; yielding any proposal resets the streak.
- anything else read back from BASE_PARAMS is yielded verbatim — that is,
  any read-back changing exactly one declared parameter: out-of-space,
  schema-invalid, and duplicate-vs-history configs are the runner's
  deterministic preflight's job (PLAN §5.1, the natural channel), and the
  rejection is reported back as the next outcome. The runner's
  5-consecutive-reject tripwire bounds an editor stuck on illegal configs.
- a checkpoint with no varying dimension is ``Unsupported`` up front (no legal
  single-parameter change exists).

Per-proposal ``arm_state`` carries ``editor_attempts`` (asks used since the
previous yield, the successful one included). Counts emitted in ``finally``:
``session.totals()`` (llm_calls / llm_input_tokens / llm_output_tokens) and
``internal_duplicate_count`` under arm_api.AGGREGATE_ARM_STATE_KEYS; the
invalid-edit tally rides along as a non-aggregated event field. This arm uses
no RNG — the editor session is the only stochastic source.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tuners"))

import apply_base_params  # noqa: E402
import arm_api  # noqa: E402
import llm  # noqa: E402
import tune_tools  # noqa: E402

ROLE = "bench-hillclimb-editor"
MAX_CONSECUTIVE_INVALID_EDITS = 3

HILLCLIMB_PROTOCOL = (
    "LLM hillclimb protocol (PLAN §6.0): production hillclimb narrowed to "
    "SEARCH_SPACE parameter values. Each invocation you change the value of "
    "exactly ONE tunable parameter in the working copy — never two "
    "parameters, never structural edits. The runner evaluates every edited "
    "config through its own objective path; you never run anything yourself. "
    "Before every invocation the working copy is synced to the current "
    "incumbent (revert-to-best), so what you Read IS the best-so-far config. "
    "After each evaluation you receive the authoritative outcome: the score "
    "and KEEP (strictly improved the then-current incumbent — the edit "
    "becomes the new base) or DISCARD (no improvement, or a crash — "
    "reverted). A crash scores +inf, the worst outcome, and is dropped. "
    "Scores are always lower-is-better. You keep editing one change at a "
    "time until the evaluation budget is exhausted."
)


class LlmHillclimb:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "llm_hillclimb"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        factory = ctx.extras.get("session_factory")
        if factory is None:
            raise arm_api.ArmError(
                "llm_hillclimb requires ctx.extras['session_factory'] "
                "(llm.make_bout_session_factory); the cell wiring provides it"
            )
        contract = ctx.contract
        if not contract.varying_dimensions:
            raise arm_api.Unsupported(
                "llm_hillclimb: checkpoint has no varying dimension to edit"
            )
        session = factory(
            ROLE,
            first_extras=llm.first_message_blocks(
                ctx.checkpoint,
                contract,
                protocol=HILLCLIMB_PROTOCOL,
                budget_remaining=ctx.budget,
            ),
        )
        frozen_source = ctx.checkpoint.candidate_path.read_text(encoding="utf-8")
        workdir = Path(tempfile.mkdtemp(prefix="ib_llm_hillclimb_"))
        working_copy = workdir / "train.py"
        working_copy.write_text(frozen_source, encoding="utf-8")
        internal_duplicates = 0
        invalid_edits = 0
        try:
            consecutive_invalid = 0
            attempts = 0  # asks since the previous yield, this one included
            first_ask = True
            pending_message = None
            initial_trial_count = len(ctx.state.trials)
            while True:
                # Revert-to-best, parameter version: sync BASE_PARAMS to the
                # current incumbent before every invocation.
                incumbent = dict(ctx.state.incumbent_config)
                incumbent_identity = contract.params_identity(incumbent)
                try:
                    apply_base_params.apply(working_copy, incumbent)
                except (SystemExit, SyntaxError, ValueError):
                    # The previous edit left the file unparseable for the
                    # sync; restore the pristine frozen source and retry.
                    working_copy.write_text(frozen_source, encoding="utf-8")
                    try:
                        apply_base_params.apply(working_copy, incumbent)
                    except (SystemExit, SyntaxError, ValueError) as exc:
                        # Pristine source + frozen incumbent not syncing is a
                        # broken checkpoint, not an edit problem — and a
                        # SystemExit must never escape run_cell
                        # (BaseException slips past the runner's containment).
                        raise arm_api.ArmError(
                            "llm_hillclimb: cannot sync the pristine working "
                            f"copy to the frozen incumbent: {exc}"
                        ) from exc

                extra = {llm.WORKING_COPY_KEY: str(working_copy)}
                if pending_message is not None:
                    extra[llm.OUTCOME_KEY] = pending_message
                if not first_ask:
                    # The checkpoint history rides in first_extras; this adds
                    # the bout-local history from the second invocation on.
                    extra["history"] = llm.format_history(
                        ctx.state.trials[initial_trial_count:],
                        incumbent_score_at_each_eval=ctx.checkpoint.incumbent.score,
                    )
                attempts += 1
                try:
                    receipt = session.ask(extra=extra)
                except Exception as exc:  # InvocationFailed: schema retries exhausted
                    consecutive_invalid += 1
                    invalid_edits += 1
                    correction = (
                        "correction: your previous invocation failed before "
                        f"completing ({exc}). Change the value of exactly ONE "
                        "tunable parameter in the working copy now."
                    )
                    # A failed ask never entered the transcript: keep any
                    # undelivered outcome riding alongside the correction.
                    pending_message = (
                        f"{pending_message}\n\n{correction}"
                        if pending_message is not None
                        else correction
                    )
                    _check_streak(consecutive_invalid, f"invocation failed: {exc}")
                    continue
                # Only a successful ask consumes the pending outcome and the
                # first-ask marker: BoutSession re-merges first_extras after a
                # failed invocation, and consuming first_ask early would
                # shadow the checkpoint history with an empty bout history.
                pending_message = None
                first_ask = False

                if not isinstance(receipt, dict) or receipt.get("edited") is not True:
                    consecutive_invalid += 1
                    invalid_edits += 1
                    pending_message = (
                        "correction: you reported edited=false — no parameter "
                        "change was made this invocation. Every invocation "
                        "must change the value of exactly ONE tunable "
                        "parameter in the working copy. Read it and make one "
                        "edit now."
                    )
                    _check_streak(consecutive_invalid, "edited=false")
                    continue

                summary = receipt.get("summary")
                try:
                    new_params = tune_tools._read_literal_mapping(
                        working_copy, "BASE_PARAMS"
                    )
                except (SyntaxError, ValueError) as exc:
                    consecutive_invalid += 1
                    invalid_edits += 1
                    pending_message = (
                        "failure: your last edit left the working copy's "
                        f"BASE_PARAMS unreadable ({exc}); the file has been "
                        "restored and re-synced to the incumbent. Change "
                        "exactly ONE parameter VALUE — never restructure the "
                        "file."
                    )
                    _check_streak(
                        consecutive_invalid, f"unreadable BASE_PARAMS: {exc}"
                    )
                    continue

                try:
                    same_as_incumbent = (
                        contract.params_identity(new_params) == incumbent_identity
                    )
                except (TypeError, ValueError, ArithmeticError):
                    # Uncastable: the deterministic preflight's schema_invalid
                    # is the natural channel, not an invalid edit.
                    same_as_incumbent = False
                if same_as_incumbent:
                    consecutive_invalid += 1
                    invalid_edits += 1
                    internal_duplicates += 1
                    pending_message = (
                        "correction: your edit left the config identical to "
                        "the incumbent — an exact duplicate, which is "
                        "preflight-rejected and never evaluated. Change "
                        "exactly ONE tunable parameter to a value that "
                        "differs from the incumbent config."
                    )
                    _check_streak(
                        consecutive_invalid, "edit identical to the incumbent"
                    )
                    continue

                changed = _changed_dimensions(contract, new_params, incumbent)
                if changed is not None and len(changed) > 1:
                    # The §6.0 one-change narrowing, enforced: anything the
                    # runner would evaluate must differ from the incumbent in
                    # exactly one declared parameter, else the bout stops
                    # being hillclimb and arm attribution is lost.
                    consecutive_invalid += 1
                    invalid_edits += 1
                    pending_message = (
                        "correction: your edit changed "
                        f"{len(changed)} parameters ({', '.join(changed)}) — "
                        "exactly ONE parameter value may change per "
                        "invocation. The working copy has been restored and "
                        "re-synced to the incumbent; change exactly one "
                        "parameter VALUE now."
                    )
                    _check_streak(
                        consecutive_invalid,
                        f"multi-parameter edit: {', '.join(changed)}",
                    )
                    continue

                consecutive_invalid = 0
                incumbent_before = ctx.state.incumbent_score
                feedback = yield arm_api.Proposal(
                    params=new_params,
                    source="hillclimb_edit",
                    rationale=summary if isinstance(summary, str) else None,
                    arm_state={"editor_attempts": attempts},
                )
                attempts = 0
                pending_message = _outcome_block(
                    feedback,
                    params=new_params,
                    incumbent_before=incumbent_before,
                    budget_remaining=ctx.state.budget_remaining,
                )
        finally:
            ctx.emit(
                {
                    **session.totals(),
                    "internal_duplicate_count": internal_duplicates,
                    "invalid_edit_count": invalid_edits,
                }
            )
            shutil.rmtree(workdir, ignore_errors=True)


def _changed_dimensions(contract, new_params: dict, incumbent: dict) -> list[str] | None:
    """Declared dimensions whose value differs between the read-back config
    and the incumbent. Returns None when the read-back is not exactly
    contract-shaped or uncastable — malformed key sets and junk values are
    the runner's deterministic preflight's job, not an invalid edit.

    Production cast first (int truncation toward zero), then per-dimension
    comparison: type-strict for categoricals (production
    ``_categorical_value_equal``), exact for numerics (both sides post-cast).
    """
    if set(new_params) != set(contract.search_space):
        return None
    try:
        cast_new = contract.cast(new_params)
        cast_incumbent = contract.cast(incumbent)
    except (KeyError, TypeError, ValueError, ArithmeticError, OverflowError):
        return None
    changed = []
    for dim in contract.dimensions:
        new_value = cast_new[dim.name]
        old_value = cast_incumbent[dim.name]
        if dim.kind == "categorical":
            if not tune_tools._categorical_value_equal(new_value, old_value):
                changed.append(dim.name)
        elif new_value != old_value:
            changed.append(dim.name)
    return changed


def _check_streak(consecutive_invalid: int, reason: str) -> None:
    if consecutive_invalid >= MAX_CONSECUTIVE_INVALID_EDITS:
        raise arm_api.ArmError(
            f"llm_hillclimb: {MAX_CONSECUTIVE_INVALID_EDITS} consecutive "
            f"invalid edits (last: {reason})"
        )


def _outcome_block(
    feedback, *, params: dict, incumbent_before: float, budget_remaining: int
) -> str:
    """The next invocation's ``outcome`` block: the previous proposal's
    authoritative result as KEEP / DISCARD / CRASH / preflight rejection
    (production results.tsv vocabulary; §6.0 "score、keep/discard")."""
    if feedback.kind != "outcome":
        return "\n".join([
            f"config: {_compact(params)}",
            f"result: REJECTED by the {feedback.stage} preflight "
            f"({feedback.reason}); no budget was consumed and no score "
            "exists. Do not propose it again.",
            f"remaining budget: {budget_remaining} objective evaluations",
        ])
    executed = (
        feedback.executed_params if feedback.executed_params is not None else params
    )
    if feedback.status == "crash":
        return "\n".join([
            f"config: {_compact(executed)}",
            "result: CRASH (score = +inf, the worst outcome; the budget was "
            "still consumed)",
            "verdict: DISCARD — the crashed config was dropped and the "
            "working copy has been re-synced to the incumbent. Do not retry "
            "it.",
            f"remaining budget: {budget_remaining} objective evaluations",
        ])
    if feedback.score < incumbent_before:
        verdict = (
            "KEEP — strictly improved the then-current incumbent "
            f"({incumbent_before} -> {feedback.score}); this config is the "
            "new incumbent the working copy is synced to."
        )
    else:
        verdict = (
            "DISCARD — did not improve the then-current incumbent "
            f"(score {feedback.score}, incumbent {incumbent_before}); the "
            "working copy has been re-synced to the incumbent."
        )
    return "\n".join([
        f"config: {_compact(executed)}",
        f"result: score = {feedback.score}",
        f"verdict: {verdict}",
        f"remaining budget: {budget_remaining} objective evaluations",
    ])


def _compact(params: dict) -> str:
    return json.dumps(params, separators=(",", ":"), ensure_ascii=False, default=str)


ARM = LlmHillclimb()
