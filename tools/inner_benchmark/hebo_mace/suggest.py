"""Official-HEBO suggest mirror for the inner-tuner benchmark
(PLAN-inner-arms-mixup-alt §1).

Runs in a fresh subprocess of the repository-root uv environment; invoked by
the three HEBO-family arms (``hebo_only`` / ``mixup_pool_hebo`` /
``alt_pool_hebo``) as::

    <root .venv python> tools/inner_benchmark/hebo_mace/suggest.py

stdin (one JSON object)::

    {
      "search_space": {"<name>": ["float", lo, hi] | ["float", lo, hi, "log"]
                                | ["int", lo, hi] | ["categorical", [opt, ...]],
                       ...},              // raw contract SEARCH_SPACE, declaration order
      "history":      [{"params": {...}, "score": <finite float>}, ...],
      "seed":         <int>,   // per-step seed: all non-Sobol stochasticity
                               // (GP fit, EvolutionOpt, np.random.choice)
      "scramble_seed": <int>,  // trajectory-level Sobol scramble seed — derived
                               // once from the cell seed, fixed for the whole cell
      "quasi_index":  <int>,   // official warmup Sobol points already consumed
                               // by this trajectory (0-based position)
      "initial_suggest_extra": [{"params"...}, ...]  // optional, default [];
                               // prepended to best_x as EvolutionOpt's initial_suggest
    }

stdout (one JSON object)::

    {"suggestion": {"<name>": value, ...},   // one config, raw param space
     "mode": "quasi" | "surrogate",
     "quasi_consumed": <int>,  // Sobol points this call drew from the
                              // trajectory sequence (1 for quasi mode; the
                              // surrogate branch's uniqueness top-up count,
                              // usually 0)
     "front_size": <int>}      // surrogate mode only: final-generation size
                              // after drop_duplicates + history-uniqueness
    // or
    {"error": "<what failed>"} // nonzero exit code as well

This mirrors ``hebo.optimizers.hebo.HEBO.suggest`` at the pinned commit
(n_suggestions=1, fix_input=None) EXACTLY, with one structural difference:
the process is stateless, so the official instance's sequential Sobol state
is reproduced by position — a fresh SobolEngine with the trajectory's fixed
``scramble_seed``, drawn to ``quasi_index + n`` and returning the last n
points. torch's scrambled SobolEngine draws are position-determined by the
constructor seed (its scramble uses a dedicated Generator, never the global
RNG), so this equals one long-lived official instance drawing point by point.

Step-by-step correspondence with official ``suggest`` (hebo.py:119-194):

1. ``len(history) < rand_sample`` (official default ``1 + num_paras``,
   hebo.py:57) -> warmup: one ``quasi_sample`` point (hebo.py:63-75
   verbatim), ``mode="quasi"``;
2. otherwise the surrogate branch: objective transform + ``gp`` fit
   (bit-identical to hebo_mace/rank.py:137-151, itself copied from
   hebo.py:127-147), ``py_best`` at the argmin observation, HEBO's kappa
   schedule with n_suggestions=1;
3. ``initial_suggest = concat(extra, [best_x])`` into
   ``EvolutionOpt(pop=100, iters=100, sobol_init=True)`` (hebo.py:165-166);
   MACE has num_obj=3, so ``es=None`` resolves to nsga2 exactly like the
   official constructor default;
4. final-generation ``drop_duplicates`` -> ``check_unique`` against history
   (hebo.py:196-197) -> uniqueness top-up from the SAME trajectory Sobol
   sequence (hebo.py:169-180, cnt>3 tolerance) -> ``np.random.choice`` of 1
   (hebo.py:182). The ``n_suggestions > 2`` directed-override block
   (hebo.py:187-192) is dead code at n_suggestions=1 and consumes no RNG, so
   it is not reproduced.

All non-Sobol stochasticity (torch GP init + pSGLD fitting + MACE's noise
perturbation, ``space.sample(100)`` initial population, pymoo NSGA-II,
``np.random.choice``) is seeded from the payload ``seed`` before any HEBO
call (same injection point as rank.py:119-122). Sobol randomness is located
ONLY by the trajectory-level ``scramble_seed + quasi_index`` pair — a fresh
subprocess must never restart the sequence at its first point, so callers
consume and increment ``quasi_index`` by ``quasi_consumed`` immediately when
a suggestion returns, even if the point is later duplicate/task-preflight
rejected.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys


def _hebo_space_config(search_space: dict) -> list[dict]:
    """Raw contract SEARCH_SPACE -> HEBO DesignSpace.parse record list.

    Declaration order is preserved. float+log -> 'pow' (HEBO's log-scale
    numeric type, log10 transform); float -> 'num'; int -> 'int';
    categorical -> 'cat' with the declared options verbatim.
    (Kept bit-identical to hebo_mace/rank.py.)
    """
    records = []
    for name, entry in search_space.items():
        kind = entry[0]
        if kind == "float":
            ptype = "pow" if len(entry) == 4 else "num"
            records.append(
                {"name": name, "type": ptype, "lb": float(entry[1]), "ub": float(entry[2])}
            )
        elif kind == "int":
            records.append(
                {"name": name, "type": "int", "lb": int(entry[1]), "ub": int(entry[2])}
            )
        elif kind == "categorical":
            records.append({"name": name, "type": "cat", "categories": list(entry[1])})
        else:
            raise ValueError(f"dimension {name!r}: unknown kind {kind!r}")
    return records


def _model_config(space) -> dict:
    """HEBO.model_config for model_name='gp' at the pinned commit."""
    cfg = {
        "lr": 0.01,
        "num_epochs": 100,
        "verbose": False,
        "noise_lb": 8e-4,
        "pred_likeli": False,
    }
    if space.num_categorical > 0:
        cfg["num_uniqs"] = [
            len(space.paras[name].categories) for name in space.enum_names
        ]
    return cfg


def compute(payload: dict) -> dict:
    import random

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.preprocessing import power_transform
    from torch.quasirandom import SobolEngine

    from hebo.acquisitions.acq import MACE
    from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt
    from hebo.design_space.design_space import DesignSpace
    from hebo.models.model_factory import get_model

    seed = int(payload["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # hebo.optimizers.hebo forces single-threaded torch at import; the
    # surrogate path here imports the same libraries without that module, so
    # set it explicitly to keep GP-fit numerics on the official footing.
    torch.set_num_threads(min(1, torch.get_num_threads()))

    space = DesignSpace().parse(_hebo_space_config(payload["search_space"]))
    history = payload["history"]
    if not history:
        raise ValueError("history is empty")
    scramble_seed = int(payload["scramble_seed"])
    quasi_index = int(payload["quasi_index"])
    if quasi_index < 0:
        raise ValueError(f"quasi_index must be >= 0, got {quasi_index}")
    extra = payload.get("initial_suggest_extra") or []

    # Official default (hebo.py:57): rand_sample = 1 + num_paras.
    rand_sample = 1 + space.num_paras

    def quasi_sample(start: int, n: int) -> "pd.DataFrame":
        """n points of the trajectory Sobol sequence from ``start`` on —
        hebo.py:63-75 (quasi_sample, fix_input=None) verbatim, with the
        long-lived engine's sequential state reproduced by position: a fresh
        engine with the trajectory scramble seed drawn to start+n equals the
        official instance's point at that position (module docstring).
        """
        eng = SobolEngine(space.num_paras, scramble=True, seed=scramble_seed)
        samp = eng.draw(start + n)[start:]
        samp = samp * (space.opt_ub - space.opt_lb) + space.opt_lb
        x = samp[:, : space.num_numeric]
        xe = samp[:, space.num_numeric :]
        for i, n_name in enumerate(space.numeric_names):
            if space.paras[n_name].is_discrete_after_transform:
                x[:, i] = x[:, i].round()
        return space.inverse_transform(x, xe)

    def check_unique(X: "pd.DataFrame", rec: "pd.DataFrame") -> list:
        """hebo.py:196-197 verbatim, with the history frame as self.X."""
        return (
            (~pd.concat([X, rec], axis=0).duplicated().tail(rec.shape[0]).values).tolist()
        )

    df_X = pd.DataFrame([row["params"] for row in history])

    # --- hebo.py:122-124: warmup branch ------------------------------------
    if df_X.shape[0] < rand_sample:
        sample = quasi_sample(quasi_index, 1)
        return {
            "suggestion": _row_to_params(sample.iloc[0], payload["search_space"]),
            "mode": "quasi",
            "quasi_consumed": 1,
        }

    # --- hebo.py:126-147: objective transform + surrogate fit ---------------
    # Bit-identical to hebo_mace/rank.py (same official lines).
    y_raw = np.array([[float(row["score"])] for row in history], dtype=float)
    X, Xe = space.transform(df_X)
    try:
        if y_raw.min() <= 0:
            y = torch.FloatTensor(power_transform(y_raw / y_raw.std(), method="yeo-johnson"))
        else:
            y = torch.FloatTensor(power_transform(y_raw / y_raw.std(), method="box-cox"))
            if y.std() < 0.5:
                y = torch.FloatTensor(power_transform(y_raw / y_raw.std(), method="yeo-johnson"))
        if y.std() < 0.5:
            raise RuntimeError("Power transformation failed")
        model = get_model("gp", space.num_numeric, space.num_categorical, 1, **_model_config(space))
        model.fit(X, Xe, y)
    except Exception:
        y = torch.FloatTensor(y_raw).clone()
        model = get_model("gp", space.num_numeric, space.num_categorical, 1, **_model_config(space))
        model.fit(X, Xe, y)

    # --- hebo.py:149-160: best_x / py_best / kappa --------------------------
    best_id = int(np.argmin(y_raw.reshape(-1)))  # get_best_id(fix_input=None)
    best_x = df_X.iloc[[best_id]]
    py_best, _ = model.predict(*space.transform(best_x))
    py_best = float(py_best.detach().numpy().squeeze())

    n_obs = df_X.shape[0]
    iter_ = max(1, n_obs // 1)  # n_suggestions = 1
    upsi, delta = 0.5, 0.01
    n_dim = df_X.shape[1]
    kappa = math.sqrt(
        upsi * 2 * ((2.0 + n_dim / 2.0) * math.log(iter_) + math.log(3 * math.pi**2 / (3 * delta)))
    )

    # --- hebo.py:162-167: MACE + EvolutionOpt -------------------------------
    acq = MACE(model, best_y=py_best, kappa=kappa)
    initial_suggest = pd.concat([pd.DataFrame(extra), best_x], ignore_index=True)
    opt = EvolutionOpt(space, acq, pop=100, iters=100, verbose=False, es=None)
    rec = opt.optimize(initial_suggest=initial_suggest, fix_input=None).drop_duplicates()
    rec = rec[check_unique(df_X, rec)]
    front_size = int(rec.shape[0])

    # --- hebo.py:169-180: uniqueness top-up from the Sobol sequence ---------
    quasi_consumed = 0
    cnt = 0
    while rec.shape[0] < 1:  # n_suggestions = 1
        rand_rec = quasi_sample(quasi_index + quasi_consumed, 1)
        quasi_consumed += 1
        rand_rec = rand_rec[check_unique(df_X, rand_rec)]
        rec = pd.concat([rec, rand_rec], axis=0, ignore_index=True)
        cnt += 1
        if cnt > 3:
            # sometimes the design space is so small that duplicated sampling
            # is unavoidable (official comment)
            break
    if rec.shape[0] < 1:
        rand_rec = quasi_sample(quasi_index + quasi_consumed, 1)
        quasi_consumed += 1
        rec = pd.concat([rec, rand_rec], axis=0, ignore_index=True)

    # --- hebo.py:182: uniform pick of 1 from the final generation -----------
    # The n_suggestions > 2 directed overrides (hebo.py:187-192) never fire at
    # n_suggestions=1; their mu/sig predictions consume no RNG, so omitting
    # them leaves the stream bit-identical.
    select_id = np.random.choice(rec.shape[0], 1, replace=False).tolist()
    rec_selected = rec.iloc[select_id].copy()
    return {
        "suggestion": _row_to_params(rec_selected.iloc[0], payload["search_space"]),
        "mode": "surrogate",
        "quasi_consumed": quasi_consumed,
        "front_size": front_size,
    }


def _row_to_params(row, search_space: dict) -> dict:
    """One inverse_transform output row -> JSON-native params, in SEARCH_SPACE
    declaration order, cast per declared kind (int dims arrive rounded from
    HEBO's own inverse_transform; categorical dims arrive as labels).
    """
    params = {}
    for name, entry in search_space.items():
        value = row[name]
        if hasattr(value, "item"):
            value = value.item()
        kind = entry[0]
        if kind == "int":
            params[name] = int(value)
        elif kind == "float":
            params[name] = float(value)
        else:
            params[name] = value
    return params


def main() -> int:
    payload = json.loads(sys.stdin.read())
    try:
        # HEBO/gpytorch print fitting chatter (e.g. "jitter = ...") to stdout;
        # keep stdout reserved for the single JSON answer.
        with contextlib.redirect_stdout(sys.stderr):
            result = compute(payload)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
