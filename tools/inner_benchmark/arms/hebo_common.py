"""Shared official-HEBO suggest plumbing for the three HEBO-family arms
(PLAN-inner-arms-mixup-alt §1): ``hebo_only`` / ``mixup_pool_hebo`` /
``alt_pool_hebo``.

- ``live_hebo_history(ctx)`` — the SINGLE input surface of all three arms:
  ``ctx.state.finite_unique_history()`` rows as suggest-payload dicts. The
  runner already inserts the checkpoint incumbent into the live state when
  the freezer did not list it as a history row, so this helper does NOT add
  it again; finiteness, dedupe, order, and incumbent semantics are literally
  identical across the three arms.
- ``official_rand_sample(contract)`` — the official warmup threshold
  ``1 + num_paras`` (hebo.py:57), needed arm-side because ``mixup``/``alt``
  must know whether a step is warmup BEFORE deciding to involve the LLM.
- ``step_seed(cell_seed, step_index)`` — every step's suggest seed as a PURE
  function of (cell seed, step index), derived before any LLM round-trip.
  This is what makes mixup-vs-hebo_only a same-(checkpoint, seed) paired
  contrast whose ONLY structural difference is ``initial_suggest_extra``:
  both arms hand the surrogate phase identical GP-fit / ``space.sample(100)``
  / ``np.random.choice`` randomness at every step index (same rule covers
  alt's BO steps).
- ``trajectory_scramble_seed(cell_seed)`` — the trajectory-level Sobol
  scramble seed, derived once per cell and fixed for the whole trajectory.
- ``call_suggest`` — the seam wrapper: ``ctx.extras['hebo_suggest_fn']``
  injected by tests, otherwise the ``hebo_mace/suggest.py`` subprocess
  (same shape as pool_hebo_mace's ``hebo_rank_fn`` seam). Any failure ->
  ``ArmError`` (fail-fast, never an invented fallback).
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

import arm_api  # noqa: E402

HEBO_PROJECT_DIR = Path(__file__).resolve().parent.parent / "hebo_mace"

_SUGGEST_MODES = ("quasi", "surrogate")


def live_hebo_history(ctx) -> list[dict]:
    """Live finite-unique history as suggest-payload rows (module docstring)."""
    return [
        {"params": params, "score": float(score)}
        for params, score in ctx.state.finite_unique_history()
    ]


def official_rand_sample(contract) -> int:
    """Official HEBO warmup threshold: 1 + num_paras (hebo.py:57)."""
    return 1 + len(contract.search_space)


def step_seed(cell_seed: int, step_index: int) -> int:
    """Per-step suggest seed — pure function of (cell seed, step index)."""
    return (cell_seed * 1_000_003 + step_index) % (2**31 - 1)


def trajectory_scramble_seed(cell_seed: int) -> int:
    """Fixed Sobol scramble seed for one cell trajectory."""
    return (cell_seed * 65_537 + 1) % (2**31 - 1)


def call_suggest(
    suggest_fn,
    ctx,
    *,
    history: list,
    seed: int,
    scramble_seed: int,
    quasi_index: int,
    initial_suggest_extra: list,
) -> dict:
    """Invoke the suggest seam and validate its answer into the arm contract.

    Returns the seam's dict (``suggestion`` / ``mode`` / ``quasi_consumed``,
    optional ``front_size``). A raising seam or a malformed answer is an
    ``ArmError`` — same fail-fast stance as pool_hebo_mace's rank_fn.
    """
    try:
        result = suggest_fn(
            search_space=ctx.contract.search_space,
            history=history,
            seed=seed,
            scramble_seed=scramble_seed,
            quasi_index=quasi_index,
            initial_suggest_extra=list(initial_suggest_extra),
        )
    except arm_api.ArmError:
        raise
    except Exception as exc:
        raise arm_api.ArmError(f"hebo suggest_fn failed: {exc}") from exc
    if not isinstance(result, dict):
        raise arm_api.ArmError(f"hebo suggest returned non-dict: {type(result).__name__}")
    if result.get("mode") not in _SUGGEST_MODES:
        raise arm_api.ArmError(f"hebo suggest returned unknown mode: {result.get('mode')!r}")
    if not isinstance(result.get("suggestion"), dict):
        raise arm_api.ArmError(
            f"hebo suggest returned no suggestion dict: {result.get('suggestion')!r}"
        )
    consumed = result.get("quasi_consumed")
    if isinstance(consumed, bool) or not isinstance(consumed, int) or consumed < 0:
        raise arm_api.ArmError(f"hebo suggest returned bad quasi_consumed: {consumed!r}")
    if result["mode"] == "quasi" and consumed < 1:
        raise arm_api.ArmError("hebo suggest quasi mode must consume >= 1 Sobol point")
    return result


def subprocess_suggest_fn(
    *, search_space, history, seed, scramble_seed, quasi_index, initial_suggest_extra
):
    """Default suggest_fn: root-env hebo_mace/suggest.py over JSON stdin/stdout."""
    payload = json.dumps(
        {
            "search_space": search_space,
            "history": history,
            "seed": seed,
            "scramble_seed": scramble_seed,
            "quasi_index": quasi_index,
            "initial_suggest_extra": initial_suggest_extra,
        }
    )
    proc = subprocess.run(
        [sys.executable, str(HEBO_PROJECT_DIR / "suggest.py")],
        input=payload,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise arm_api.ArmError(
            "hebo suggest subprocess failed "
            f"(exit {proc.returncode}): "
            f"stderr: {proc.stderr.strip()[-400:]!r} "
            f"stdout: {proc.stdout.strip()[-200:]!r}"
        )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise arm_api.ArmError(
            f"hebo suggest returned non-JSON stdout: {proc.stdout[:200]!r}"
        ) from exc
    if not isinstance(result, dict) or "suggestion" not in result:
        detail = result.get("error") if isinstance(result, dict) else proc.stdout[:200]
        raise arm_api.ArmError(f"hebo suggest returned no suggestion: {detail!r}")
    return result


def z_delta(ctx, params: dict, reference: dict) -> list[float]:
    """Signed per-numeric-dimension codec-z delta (params - reference).

    Descriptive trajectory diagnostics only (nearest-seed / probe-follow /
    incumbent-distance covariates); categorical dimensions are not part of z.
    """
    z, _ = ctx.codec.encode(params)
    z_ref, _ = ctx.codec.encode(reference)
    return [float(d) for d in (z - z_ref).tolist()]


def z_distance(ctx, params: dict, reference: dict) -> float:
    """L2 norm of ``z_delta``."""
    return float(np.linalg.norm(z_delta(ctx, params, reference)))


def seed_geometry(ctx, suggestion: dict, pool: list) -> dict:
    """mixup's seed-influence diagnostics: z distance from the suggested
    point to its nearest pool seed, and whether the suggestion literally IS
    a seed (expected to be rare — the executed point is usually a descendant).
    """
    nearest = min(z_distance(ctx, suggestion, member) for member in pool) if pool else None
    identities = {ctx.contract.params_identity(member) for member in pool}
    return {
        "nearest_seed_z_dist": nearest,
        "hit_seed": ctx.contract.params_identity(suggestion) in identities,
    }


__all__ = [
    "HEBO_PROJECT_DIR",
    "call_suggest",
    "live_hebo_history",
    "official_rand_sample",
    "seed_geometry",
    "step_seed",
    "subprocess_suggest_fn",
    "trajectory_scramble_seed",
    "z_delta",
    "z_distance",
]
