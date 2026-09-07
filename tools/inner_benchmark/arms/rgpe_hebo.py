"""RGPE history-transfer arm (DESIGN-rgpe-history-transfer-v1 §1.3,
``rgpe_hebo``): ``hebo_only`` with donor trajectories as base tasks.

Identical protocol to ``hebo_only`` — same live history surface, same
per-step / trajectory seeding, same quasi/surrogate sources — but every step
goes through ``hebo_mace/suggest_rgpe.py`` with three extra payload keys:

- ``base_tasks``: ``ctx.extras["base_tasks"]`` (``donor_history.build_base_tasks``
  output; empty -> the seam degenerates to plain HEBO);
- ``rgpe_horizon``: ``ctx.extras["rgpe_horizon"]`` or, by default, the
  candidate's planned trial total in this cell = initial finite-unique
  history + budget (eq. 9's H);
- ``base_cache_key``: checkpoint id + seed, so the in-process seam can keep
  the base GPs fitted across steps of one cell.

The seam is ``ctx.extras["hebo_suggest_fn"]`` when injected (tests, offline
harness running in-process), else the ``suggest_rgpe.py`` subprocess. The
extra keys are bound with ``functools.partial`` so the shared
``hebo_common.call_suggest`` wrapper stays untouched.

Per step, ``arm_state["rgpe"]`` records the seam's weight / dropped-base
report when present (observability only; not an aggregate key).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arm_api  # noqa: E402
from arms import hebo_common  # noqa: E402

SUGGEST_RGPE_PY = hebo_common.HEBO_PROJECT_DIR / "suggest_rgpe.py"


def subprocess_suggest_fn(**payload):
    """Default seam: root-env hebo_mace/suggest_rgpe.py over JSON stdin/stdout."""
    proc = subprocess.run(
        [sys.executable, str(SUGGEST_RGPE_PY)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise arm_api.ArmError(
            "rgpe suggest subprocess failed "
            f"(exit {proc.returncode}): "
            f"stderr: {proc.stderr.strip()[-400:]!r} "
            f"stdout: {proc.stdout.strip()[-200:]!r}"
        )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise arm_api.ArmError(
            f"rgpe suggest returned non-JSON stdout: {proc.stdout[:200]!r}"
        ) from exc
    if not isinstance(result, dict) or "suggestion" not in result:
        detail = result.get("error") if isinstance(result, dict) else proc.stdout[:200]
        raise arm_api.ArmError(f"rgpe suggest returned no suggestion: {detail!r}")
    return result


class RgpeHebo:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "rgpe_hebo"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        base_tasks = list(ctx.extras.get("base_tasks") or [])
        horizon = ctx.extras.get("rgpe_horizon")
        if horizon is None:
            horizon = len(ctx.checkpoint.finite_unique_history(ctx.contract)) + ctx.budget
        raw_fn = ctx.extras.get("hebo_suggest_fn") or subprocess_suggest_fn
        suggest_fn = functools.partial(
            raw_fn,
            base_tasks=base_tasks,
            rgpe_horizon=int(horizon),
            base_cache_key=f"{ctx.checkpoint.checkpoint_id}:{ctx.seed}",
        )
        scramble_seed = hebo_common.trajectory_scramble_seed(ctx.seed)
        quasi_index = 0
        step_index = 0
        while True:
            history = hebo_common.live_hebo_history(ctx)
            result = hebo_common.call_suggest(
                suggest_fn,
                ctx,
                history=history,
                seed=hebo_common.step_seed(ctx.seed, step_index),
                scramble_seed=scramble_seed,
                quasi_index=quasi_index,
                initial_suggest_extra=[],
            )
            quasi_index += result["quasi_consumed"]
            mode = result["mode"]
            arm_state = {
                "hebo_mode": mode,
                "quasi_index": quasi_index,
                "step_index": step_index,
                "n_base_tasks": len(base_tasks),
            }
            if mode == "surrogate":
                arm_state["front_size"] = result.get("front_size")
                if isinstance(result.get("rgpe"), dict):
                    arm_state["rgpe"] = result["rgpe"]
            yield arm_api.Proposal(
                params=result["suggestion"],
                source="hebo_quasi" if mode == "quasi" else "hebo_suggest",
                arm_state=arm_state,
            )
            step_index += 1


ARM = RgpeHebo()
