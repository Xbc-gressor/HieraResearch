"""Benchmark arms (PLAN §六) — lazy registry.

Importing an arm module pays for that arm's optimizer deps (optuna / sklearn /
the isolated HEBO env), so the registry never imports arms itself: a cell
driver loads exactly the one arm it runs. Every arm module defines a
module-level ``ARM`` singleton (arm_api protocol; per-cell state lives in the
``run(ctx)`` generator's locals, never on the object).
"""

from __future__ import annotations

import importlib

ARM_MODULES = {
    "current": "arms.current",
    "random_search": "arms.random_search",
    "llm_hillclimb": "arms.llm_hillclimb",
    "active_set": "arms.active_set",
    "local_tr": "arms.local_tr",
    "spsa": "arms.spsa",
    "llm_pool_self_rank": "arms.llm_pool_self_rank",
    "pool_gp_ei": "arms.pool_gp_ei",
    "pool_tpe": "arms.pool_tpe",
    "pool_hebo_mace": "arms.pool_hebo_mace",
    "pool3_hebo_mace": "arms.pool3_hebo_mace",
    "pool7_hebo_mace": "arms.pool7_hebo_mace",
    "hebo_only": "arms.hebo_only",
    "mixup_pool_hebo": "arms.mixup_pool_hebo",
    "alt_pool_hebo": "arms.alt_pool_hebo",
    "softalt": "arms.softalt",
    "bernsalt": "arms.bernsalt",
    "turbo": "arms.turbo",
}


def load_arm(name: str):
    """Import the named arm module and return its ARM singleton."""
    if name not in ARM_MODULES:
        raise KeyError(f"unknown arm {name!r}; registered: {sorted(ARM_MODULES)}")
    module = importlib.import_module(ARM_MODULES[name])
    return module.ARM


__all__ = ["ARM_MODULES", "load_arm"]
