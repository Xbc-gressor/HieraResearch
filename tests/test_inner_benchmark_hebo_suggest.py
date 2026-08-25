"""Fidelity tests for hebo_mace/suggest.py (PLAN-inner-arms-mixup-alt §5).

The stateless subprocess mirror is compared against the OFFICIAL
``hebo.optimizers.hebo.HEBO`` at the pinned commit, same seeds:

- multi-step warmup: one long-lived official HEBO instance running
  suggest/observe sequentially vs fresh subprocess invocations carrying the
  same ``scramble_seed`` and increasing ``quasi_index`` — every warmup point
  must match one-for-one (a single-step test cannot prove the subprocess
  preserves the official sequential Sobol state);
- single-step surrogate: official ``HEBO.suggest`` vs the subprocess, with
  and without ``initial_suggest_extra`` (the extra variant extends the
  official call through an EvolutionOpt.optimize wrapper, so every other
  official line still runs verbatim).

These need the pinned HEBO environment (repository-root uv env); they skip
elsewhere. Runtime is tens of seconds (real GP fits + EvolutionOpt).
"""

from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

hebo = pytest.importorskip("hebo", reason="needs the pinned HEBO environment")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUGGEST_PY = ROOT / "tools" / "inner_benchmark" / "hebo_mace" / "suggest.py"

_spec = importlib.util.spec_from_file_location("hebo_suggest_mirror", SUGGEST_PY)
suggest_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suggest_mod)

SPACE = {
    "depth": ["int", 1, 8],
    "lr": ["float", 0.0001, 0.1, "log"],
    "mode": ["categorical", ["fast", "slow"]],
}
# num_paras = 3 -> official rand_sample = 1 + 3 = 4.

HISTORY_ROW = {
    "params": {"depth": 4, "lr": 0.001, "mode": "fast"},
    "score": 100.0,
}
SURROGATE_HISTORY = [
    {"params": {"depth": 4, "lr": 0.001, "mode": "fast"}, "score": 100.0},
    {"params": {"depth": 1, "lr": 0.0002, "mode": "slow"}, "score": 95.0},
    {"params": {"depth": 2, "lr": 0.0003, "mode": "slow"}, "score": 96.0},
    {"params": {"depth": 3, "lr": 0.0004, "mode": "fast"}, "score": 97.0},
    {"params": {"depth": 5, "lr": 0.0005, "mode": "slow"}, "score": 92.0},
]


@pytest.fixture(autouse=True)
def _restore_global_rng():
    """The official-side driver seeds the global RNGs exactly like the
    subprocess does; leave no residue for other tests."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    yield
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)


def _seed_all(seed: int) -> None:
    """The harness's seed injection point (rank.py:119-122 / suggest.py)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _official_instance(history, scramble_seed: int):
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO

    space = DesignSpace().parse(suggest_mod._hebo_space_config(SPACE))
    opt = HEBO(space, scramble_seed=scramble_seed)
    if history:
        df = pd.DataFrame([row["params"] for row in history])
        y = np.array([[float(row["score"])] for row in history], dtype=float)
        opt.observe_new_data(df, y)
    return opt


def _run_subprocess(payload: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, str(SUGGEST_PY)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"subprocess failed: {proc.stderr[-1000:]}"
    return json.loads(proc.stdout)


def _official_params(df_row) -> dict:
    return suggest_mod._row_to_params(df_row, SPACE)


def test_multi_step_warmup_matches_official_sequential_sobol() -> None:
    scramble_seed = 12345
    history = [dict(HISTORY_ROW)]
    # One long-lived official instance: its Sobol engine advances draw by draw.
    opt = _official_instance(history, scramble_seed)
    for quasi_index in range(3):  # rows 1..3: still below rand_sample = 4
        step_seed = 100 + quasi_index
        _seed_all(step_seed)  # mirrors the per-step harness injection
        official_point = opt.suggest(n_suggestions=1)
        sub = _run_subprocess(
            {
                "search_space": SPACE,
                "history": history,
                "seed": step_seed,
                "scramble_seed": scramble_seed,
                "quasi_index": quasi_index,
            }
        )
        assert sub["mode"] == "quasi"
        assert sub["quasi_consumed"] == 1
        assert sub["suggestion"] == _official_params(official_point.iloc[0])
        # Both sides observe the same point before the next step.
        score = float(50 + quasi_index)
        opt.observe_new_data(official_point, np.array([[score]]))
        history.append({"params": sub["suggestion"], "score": score})


def test_single_step_surrogate_matches_official() -> None:
    scramble_seed = 999
    seed = 7
    _seed_all(seed)
    opt = _official_instance(SURROGATE_HISTORY, scramble_seed)
    official_rec = opt.suggest(n_suggestions=1)

    sub = _run_subprocess(
        {
            "search_space": SPACE,
            "history": SURROGATE_HISTORY,
            "seed": seed,
            "scramble_seed": scramble_seed,
            "quasi_index": 0,
        }
    )
    assert sub["mode"] == "surrogate"
    assert sub["quasi_consumed"] == 0
    assert sub["front_size"] >= 1
    assert sub["suggestion"] == _official_params(official_rec.iloc[0])


def test_surrogate_with_initial_suggest_extra_matches_extended_official() -> None:
    """The pool-as-initial_suggest variant: the ONLY change against official
    is the initial_suggest content (concat(extra, best_x)); every other line
    of official suggest runs verbatim via an EvolutionOpt.optimize wrapper."""
    from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt

    scramble_seed = 999
    seed = 7
    extra = [
        {"depth": 7, "lr": 0.005, "mode": "fast"},
        {"depth": 2, "lr": 0.02, "mode": "slow"},
    ]
    original = EvolutionOpt.optimize

    def extended(self, initial_suggest=None, **kwargs):
        return original(
            self,
            initial_suggest=pd.concat(
                [pd.DataFrame(extra), initial_suggest], ignore_index=True
            ),
            **kwargs,
        )

    _seed_all(seed)
    opt = _official_instance(SURROGATE_HISTORY, scramble_seed)
    with mock.patch.object(EvolutionOpt, "optimize", extended):
        official_rec = opt.suggest(n_suggestions=1)

    sub = _run_subprocess(
        {
            "search_space": SPACE,
            "history": SURROGATE_HISTORY,
            "seed": seed,
            "scramble_seed": scramble_seed,
            "quasi_index": 0,
            "initial_suggest_extra": extra,
        }
    )
    assert sub["mode"] == "surrogate"
    assert sub["suggestion"] == _official_params(official_rec.iloc[0])


# --- union mode (hands, DESIGN-inner-arm-hands §3) ---------------------------

UNION_POOL = [
    {"depth": 7, "lr": 0.005, "mode": "fast"},
    {"depth": 2, "lr": 0.02, "mode": "slow"},
    {"depth": 6, "lr": 0.0003, "mode": "fast"},
]


def _union_payload(pool, seed=7, scramble_seed=999):
    return {
        "search_space": SPACE,
        "history": SURROGATE_HISTORY,
        "seed": seed,
        "scramble_seed": scramble_seed,
        "quasi_index": 0,
        "pool": pool,
    }


def test_union_mode_provenance_contract_and_determinism() -> None:
    sub = _run_subprocess(_union_payload(UNION_POOL))
    assert sub["mode"] == "surrogate"
    assert sub["quasi_consumed"] == 0
    assert sub["front_size"] >= 1
    assert sub["chosen_from"] in ("pool", "front")
    assert sub["union_front_size"] >= 1
    survivors = sub["pool_survivor_indices"]
    assert all(index in range(len(UNION_POOL)) for index in survivors)
    if sub["chosen_from"] == "pool":
        index = sub["chosen_pool_index"]
        assert index in range(len(UNION_POOL)) and index in survivors
        # Pool provenance is a LITERAL pool member.
        assert sub["suggestion"] == UNION_POOL[index]
    else:
        assert sub["chosen_pool_index"] is None
        assert sorted(sub["suggestion"]) == sorted(SPACE)
    # Same payload -> bit-identical answer (fit + evolution + pick).
    assert _run_subprocess(_union_payload(UNION_POOL)) == sub


def test_union_mode_final_generation_matches_poolless_call() -> None:
    """The union pipeline runs the official steps 1-4 byte-identical with
    initial_suggest=best_x only: the final-generation size (and thus the
    rec frame) equals the poolless call at the same seed."""
    union = _run_subprocess(_union_payload(UNION_POOL))
    poolless = _run_subprocess(
        {
            "search_space": SPACE,
            "history": SURROGATE_HISTORY,
            "seed": 7,
            "scramble_seed": 999,
            "quasi_index": 0,
        }
    )
    assert union["front_size"] == poolless["front_size"]


def test_union_mode_drops_history_duplicates_from_pool() -> None:
    # The first pool member literally duplicates a history row: it can never
    # be chosen, never survive into the front, and the pool shrinks to the
    # two real candidates.
    dup_pool = [dict(SURROGATE_HISTORY[0]["params"]), *UNION_POOL[1:]]
    sub = _run_subprocess(_union_payload(dup_pool))
    assert 0 not in sub["pool_survivor_indices"]
    if sub["chosen_from"] == "pool":
        assert sub["chosen_pool_index"] != 0
        assert sub["suggestion"] == dup_pool[sub["chosen_pool_index"]]


def test_union_mode_rejects_initial_suggest_extra_combo() -> None:
    payload = _union_payload(UNION_POOL)
    payload["initial_suggest_extra"] = [dict(UNION_POOL[0])]
    proc = subprocess.run(
        [sys.executable, str(SUGGEST_PY)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "mutually exclusive" in proc.stdout
