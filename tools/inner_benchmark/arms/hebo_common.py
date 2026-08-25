"""Shared official-HEBO suggest plumbing for the HEBO-family arms
(PLAN-inner-arms-mixup-alt §1; union mode: DESIGN-inner-arm-hands §3):
``hebo_only`` / ``mixup_pool_hebo`` / ``alt_pool_hebo`` / ``hands``.

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

  Union mode (``hands``, DESIGN §3): the optional ``pool`` keyword carries
  the filtered LLM pool and is MUTUALLY EXCLUSIVE with a non-empty
  ``initial_suggest_extra`` (checked here, before the seam). It is passed to
  the seam ONLY when non-empty, so seams written for the pre-union arms
  (no ``pool`` parameter) keep working unchanged. In union mode the answer
  must carry the provenance fields (``chosen_from`` /
  ``chosen_pool_index`` / ``union_front_size`` /
  ``pool_survivor_indices``); a pool-provenance suggestion is additionally
  checked to be LITERALLY the named pool member.
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
    pool: list | None = None,
) -> dict:
    """Invoke the suggest seam and validate its answer into the arm contract.

    Returns the seam's dict (``suggestion`` / ``mode`` / ``quasi_consumed``,
    optional ``front_size``; union mode adds ``chosen_from`` /
    ``chosen_pool_index`` / ``union_front_size`` / ``pool_survivor_indices``).
    A raising seam or a malformed answer is an ``ArmError`` — same fail-fast
    stance as pool_hebo_mace's rank_fn.
    """
    pool = list(pool or [])
    if pool and initial_suggest_extra:
        raise arm_api.ArmError(
            "hebo suggest: pool (union mode) and initial_suggest_extra are "
            "mutually exclusive"
        )
    try:
        kwargs = dict(
            search_space=ctx.contract.search_space,
            history=history,
            seed=seed,
            scramble_seed=scramble_seed,
            quasi_index=quasi_index,
            initial_suggest_extra=list(initial_suggest_extra),
        )
        # Only union-mode callers hand the seam a pool; pre-union seams may
        # legitimately lack the parameter.
        if pool:
            kwargs["pool"] = pool
        result = suggest_fn(**kwargs)
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
    if pool and result["mode"] == "surrogate":
        _validate_union_provenance(ctx, result, pool)
    return result


def _validate_union_provenance(ctx, result: dict, pool: list) -> None:
    """Union-mode answer contract (DESIGN-inner-arm-hands §3.1)."""
    chosen_from = result.get("chosen_from")
    if chosen_from not in ("pool", "front"):
        raise arm_api.ArmError(f"hebo suggest returned bad chosen_from: {chosen_from!r}")
    union_front_size = result.get("union_front_size")
    if isinstance(union_front_size, bool) or not isinstance(union_front_size, int) \
            or union_front_size < 1:
        raise arm_api.ArmError(
            f"hebo suggest returned bad union_front_size: {union_front_size!r}"
        )
    survivors = result.get("pool_survivor_indices")
    if not isinstance(survivors, list) or any(
        isinstance(index, bool) or not isinstance(index, int)
        or not 0 <= index < len(pool)
        for index in survivors
    ):
        raise arm_api.ArmError(
            f"hebo suggest returned bad pool_survivor_indices: {survivors!r}"
        )
    chosen_pool_index = result.get("chosen_pool_index")
    if chosen_from == "pool":
        if isinstance(chosen_pool_index, bool) or not isinstance(chosen_pool_index, int) \
                or not 0 <= chosen_pool_index < len(pool):
            raise arm_api.ArmError(
                f"hebo suggest pool choice needs a valid chosen_pool_index, "
                f"got: {chosen_pool_index!r}"
            )
        if chosen_pool_index not in survivors:
            raise arm_api.ArmError(
                "hebo suggest chose a pool member outside the union front"
            )
        # Pool provenance means the executed config is LITERALLY that pool
        # member — the hands feedback channel depends on it.
        if ctx.contract.params_identity(result["suggestion"]) != ctx.contract.params_identity(
            pool[chosen_pool_index]
        ):
            raise arm_api.ArmError(
                "hebo suggest pool choice does not match the named pool member"
            )
    elif chosen_pool_index is not None:
        raise arm_api.ArmError(
            f"hebo suggest front choice must have chosen_pool_index null, "
            f"got: {chosen_pool_index!r}"
        )


def subprocess_suggest_fn(
    *, search_space, history, seed, scramble_seed, quasi_index,
    initial_suggest_extra, pool=None,
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
            "pool": list(pool or []),
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
