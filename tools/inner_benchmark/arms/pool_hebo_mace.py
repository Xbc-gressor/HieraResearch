"""LLM pool + HEBO MACE ranker arm (PLAN §6.4, ``pool_hebo_mace``).

Each step the proposer generates POOL=5 configs in proposer rank order
(arms/pool.py driver); this arm ranks the duplicate-filtered pool with the
OFFICIAL HEBO MACE acquisition (pinned commit; MACE itself is never
reimplemented in this repo — the ranker is installed by the root
``uv sync`` and invoked in a fresh subprocess with that environment) and
executes ONE config:

- WARMUP gate (§6.4 step 4): when the LIVE count
  ``ctx.state.finite_unique_history()`` (checkpoint history + this cell's own
  outcomes, growing over the cell) is below ``WARMUP=8``, the proposer's
  rank-1 config is executed with ``ranker_fallback=true`` — an engineering
  fallback used by the initial observations of a FIRST-bout deployment and
  expected never to trigger on continuation/deep checkpoints.
- Otherwise the pool is scored by HEBO MACE through the rank_fn seam
  (``ctx.extras['hebo_rank_fn']`` when injected — the test seam; otherwise
  the subprocess above). Seam signature:
  ``rank_fn(*, search_space, history, pool, seed) -> list[list[float]]``
  with one acquisition vector per pool member, LARGER-IS-BETTER per
  component (rank.py negates HEBO's minimize-convention columns; see
  hebo_mace/rank.py). A nondominated sort (maximization) is done IN THIS
  ARM — pure numpy over POOL=5 points — and the executed config is the
  unique first-Pareto-front member, or a ``ctx.np_rng``-uniform choice among
  a tied front (PLAN: "按该 cell 的固定 RNG seed 在 front 内均匀选择 1 个").

ranker failure (nonzero subprocess exit, ``{"error": ...}`` payload,
malformed values, or a raising seam) raises ``ArmError`` —
fail-fast per PLAN §6.4, never an invented "approximate HEBO".
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

import arm_api  # noqa: E402
from arms.pool import PoolDriver, pool_persistence_state  # noqa: E402

HEBO_PROJECT_DIR = Path(__file__).resolve().parent.parent / "hebo_mace"


class PoolHeboMace:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "pool_hebo_mace"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        driver = PoolDriver(ctx)
        rank_fn = ctx.extras.get("hebo_rank_fn") or _subprocess_rank_fn
        ranker_fallback_count = 0
        try:
            while True:
                result = driver.ask_pool()
                pool = result["pool"]
                history = ctx.state.finite_unique_history()
                arm_state = {
                    "pool_size": len(pool),
                    "pool_attempts": result["attempts"],
                    **pool_persistence_state(result),
                }
                if len(history) < arm_api.WARMUP:
                    ranker_fallback_count += 1
                    chosen_index = 0
                    arm_state["ranker_fallback"] = True
                else:
                    values = _rank(rank_fn, ctx, history, pool)
                    front = _first_pareto_front(values)
                    if len(front) == 1:
                        chosen_index = front[0]
                    else:
                        chosen_index = front[int(ctx.np_rng.integers(len(front)))]
                    arm_state.update(
                        {
                            "ranker_fallback": False,
                            "acquisition_values": values.tolist(),
                            "pareto_front": [int(index) for index in front],
                        }
                    )
                chosen = pool[chosen_index]
                arm_state["chosen_index"] = int(chosen_index)
                # Capture before yield: the runner advances the incumbent
                # before the feedback returns; the verdict needs the score
                # the proposal had to beat.
                incumbent_before = ctx.state.incumbent_score
                feedback = yield arm_api.Proposal(
                    params=chosen,
                    source="pool_hebo_mace",
                    rationale=result["rationale"],
                    arm_state=arm_state,
                )
                driver.report_outcome(chosen, feedback, incumbent_before=incumbent_before)
        finally:
            ctx.emit(
                {**driver.totals(), "ranker_fallback_count": ranker_fallback_count}
            )


def _subprocess_rank_fn(*, search_space, history, pool, seed):
    """Default rank_fn: root-env rank.py over JSON stdin/stdout."""
    payload = json.dumps(
        {"search_space": search_space, "history": history, "pool": pool, "seed": seed}
    )
    proc = subprocess.run(
        [sys.executable, str(HEBO_PROJECT_DIR / "rank.py")],
        input=payload,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise arm_api.ArmError(
            "hebo mace ranker subprocess failed "
            f"(exit {proc.returncode}): "
            f"stderr: {proc.stderr.strip()[-400:]!r} "
            f"stdout: {proc.stdout.strip()[-200:]!r}"
        )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise arm_api.ArmError(
            f"hebo mace ranker returned non-JSON stdout: {proc.stdout[:200]!r}"
        ) from exc
    if not isinstance(result, dict) or "values" not in result:
        detail = result.get("error") if isinstance(result, dict) else proc.stdout[:200]
        raise arm_api.ArmError(f"hebo mace ranker returned no values: {detail!r}")
    return result["values"]


def _rank(rank_fn, ctx, history, pool) -> np.ndarray:
    """Call the rank_fn seam and validate its output into an (n_pool, k) array."""
    fittable = [{"params": params, "score": score} for params, score in history]
    try:
        values = rank_fn(
            search_space=ctx.contract.search_space,
            history=fittable,
            pool=pool,
            seed=ctx.seed,
        )
    except arm_api.ArmError:
        raise
    except Exception as exc:
        raise arm_api.ArmError(f"hebo mace rank_fn failed: {exc}") from exc
    return _validate_values(values, len(pool))


def _validate_values(values, n_pool: int) -> np.ndarray:
    """Seam contract: one non-empty finite-float vector per pool member, all
    the same width. Anything else is a ranker protocol failure -> ArmError."""
    if not isinstance(values, (list, tuple)) or len(values) != n_pool:
        raise arm_api.ArmError(
            f"hebo mace ranker returned {len(values) if isinstance(values, (list, tuple)) else type(values).__name__} "
            f"acquisition vectors for a pool of {n_pool}"
        )
    rows = []
    width = None
    for row in values:
        if not isinstance(row, (list, tuple)) or not row:
            raise arm_api.ArmError(f"hebo mace acquisition row is not a non-empty vector: {row!r}")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise arm_api.ArmError("hebo mace acquisition vectors have inconsistent widths")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise arm_api.ArmError(f"hebo mace acquisition value is not a finite number: {value!r}")
        rows.append([float(value) for value in row])
    return np.array(rows, dtype=float)


def _first_pareto_front(values: np.ndarray) -> list[int]:
    """Indices of the nondominated (first Pareto front) points, MAXIMIZATION.

    i dominates j iff i is >= j on every objective and > on at least one.
    POOL=5, so the O(n^2) pairwise check is the whole implementation — no
    pymoo dependency in this env.
    """
    n = values.shape[0]
    front = []
    for i in range(n):
        dominated = any(
            i != j
            and np.all(values[j] >= values[i])
            and np.any(values[j] > values[i])
            for j in range(n)
        )
        if not dominated:
            front.append(i)
    return front


ARM = PoolHeboMace()
