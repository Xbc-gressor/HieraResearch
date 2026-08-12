"""Per-cell metrics for the inner-tuner benchmark (PLAN §九).

Pure functions over the cell's executed outcomes — no I/O, no runner imports.

Scores are always lower-is-better; the relative improvement of a best score
against the cell's initial incumbent score is

    100 * (score_start - score_best) / max(abs(score_start), 1e-12)

so positive values mean improvement. Crash outcomes never improve best-so-far
(they are +inf conceptually and carry score=None in state).

Metric definitions (all evaluation indices are 1-based over objective
evaluations only):

- ``relative_improvement_at[k]`` for k in (2, 4, 6, 8, 10): best-so-far
  relative improvement after k evaluations. A cell that stopped before k
  evaluations reports the improvement it had reached when it stopped — its
  best-so-far curve is CONSTANT past that point (no further evaluation can
  lower it), so this is the true value at k, not an imputation. None only
  when the cell ran zero evaluations. In result.json the int keys stringify
  ("2", "4", ...) — analysis code (Task 12) must parse them back to int.
- ``auc``: mean of the best-so-far relative improvement over the FULL horizon
  k = 1..budget, with the curve padded at its last value past the cell's
  actual evaluation count (same constant-past-termination argument as above).
  Normalizing by ``budget`` rather than by the evaluation count is what makes
  early termination cost something: a cell that improved once and then died at
  evaluation 3 must not score like a cell that held the same improvement for
  all 10. None when the cell had zero evaluations (there is no curve).
- ``beat_initial_incumbent``: the final best finite score is strictly below
  the initial incumbent score.
- ``first_improvement_eval``: 1-based index of the first evaluation that
  strictly improved the RUNNING incumbent (PLAN §5.2 improvement definition:
  a new finite score strictly below the current incumbent); None when no
  evaluation improved.
- ``final_relative_improvement``: relative improvement of the final best
  score (0.0 when nothing improved / nothing was evaluated).
- ``counts``: crashes (objective evaluations with status "crash") are derived
  from ``outcomes``; invalid (deterministic schema_invalid/out_of_space
  rejects), duplicates (deterministic duplicate rejects) and
  task_preflight_rejected are counted by the runner and passed in via
  ``counts``.
- Aggregation hooks: numeric values under AGGREGATE_ARM_STATE_KEYS
  (``llm_calls``, ``llm_input_tokens``, ``llm_output_tokens``,
  ``ranker_fallback_count``) are summed over every event's ``arm_state``
  (passed in via ``arm_states``); default 0, non-numeric values ignored.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_api import AGGREGATE_ARM_STATE_KEYS  # noqa: E402

IMPROVEMENT_AT = (2, 4, 6, 8, 10)

COUNT_KEYS = ("crashes", "invalid", "duplicates", "task_preflight_rejected")


def relative_improvement(score_start: float, score_best: float) -> float:
    """PLAN §九 relative improvement (percent; higher is better)."""
    return 100.0 * (score_start - score_best) / max(abs(score_start), 1e-12)


def compute_metrics(
    *,
    initial_incumbent_score: float,
    outcomes,
    budget: int,
    counts: dict | None = None,
    arm_states=(),
) -> dict:
    """Compute the per-cell metric dict merged into result.json.

    ``outcomes``: this cell's executed evaluations in order; each item needs
    ``.status`` ("ok"|"crash") and ``.score`` (float|None) — state.Trial
    qualifies. ``counts``: runner-maintained reject counts (keys "invalid",
    "duplicates", "task_preflight_rejected"; missing keys default to 0).
    ``arm_states``: every written event's arm_state dict (aggregation hooks).
    """
    score_start = float(initial_incumbent_score)
    running_best = score_start
    best_so_far: list[float] = []  # running best after each evaluation
    first_improvement_eval: int | None = None
    crashes = 0
    for index, outcome in enumerate(outcomes, start=1):
        if outcome.status == "crash":
            crashes += 1
        elif (
            outcome.status == "ok"
            and outcome.score is not None
            and math.isfinite(outcome.score)
            and outcome.score < running_best
        ):
            running_best = float(outcome.score)
            if first_improvement_eval is None:
                first_improvement_eval = index
        best_so_far.append(running_best)

    n_evaluations = len(best_so_far)
    improvements = [relative_improvement(score_start, best) for best in best_so_far]
    counts = counts or {}

    # One truncation convention for every trajectory metric: a terminated
    # cell's best-so-far curve is constant past its last evaluation, so pad it
    # to the full horizon with its last value and read every k off the padded
    # curve.
    horizon = max(int(budget), n_evaluations)
    padded = (
        improvements + [improvements[-1]] * (horizon - n_evaluations)
        if improvements
        else []
    )

    result = {
        "budget": budget,
        "evaluations": n_evaluations,
        "initial_incumbent_score": score_start,
        "final_best_score": running_best,
        "final_relative_improvement": improvements[-1] if improvements else 0.0,
        "relative_improvement_at": {
            k: padded[k - 1] if padded and k <= horizon else None
            for k in IMPROVEMENT_AT
        },
        "auc": (sum(padded) / horizon) if padded else None,
        "beat_initial_incumbent": running_best < score_start,
        "first_improvement_eval": first_improvement_eval,
        "counts": {
            "crashes": crashes,
            "invalid": int(counts.get("invalid", 0)),
            "duplicates": int(counts.get("duplicates", 0)),
            "task_preflight_rejected": int(counts.get("task_preflight_rejected", 0)),
        },
    }
    for key in AGGREGATE_ARM_STATE_KEYS:
        result[key] = sum(
            arm_state[key]
            for arm_state in arm_states
            if isinstance(arm_state.get(key), (int, float))
        )
    return result
