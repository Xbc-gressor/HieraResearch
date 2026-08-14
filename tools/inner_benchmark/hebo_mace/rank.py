"""Official-HEBO MACE ranker for the inner-tuner benchmark (PLAN §6.4).

Runs in a fresh subprocess of the repository-root uv environment; invoked by
``arms/pool_hebo_mace.py`` as::

    <root .venv python> tools/inner_benchmark/hebo_mace/rank.py

stdin (one JSON object)::

    {
      "search_space": {"<name>": ["float", lo, hi] | ["float", lo, hi, "log"]
                                | ["int", lo, hi] | ["categorical", [opt, ...]],
                       ...},              // raw contract SEARCH_SPACE, declaration order
      "history":      [{"params": {...}, "score": <finite float>}, ...],
      "pool":         [{"params"...}, ...], // POOL=5 candidate configs
      "seed":         <int>                // the cell seed (runner-pinned)
    }

stdout (one JSON object)::

    {"values": [[v0, v1, v2], ...]}   // one 3-vector per pool member, same order
    // or
    {"error": "<what failed>"}        // nonzero exit code as well

SIGN CONVENTION — every returned component is LARGER-IS-BETTER. Official
``hebo.acquisitions.acq.MACE.eval`` returns columns ``[lcb, -log EI, -log PI]``
in a MINIMIZE convention (its docstring: "minimize (-1 * EI, -1 * PI, lcb)";
HEBO feeds it to pymoo NSGA-II, which minimizes). We negate once at the
boundary, so the arm's nondominated sort maximizes ``[-lcb, log EI, log PI]``.
The middle objectives are log-domain (MACE's internal approximation branch
included); only dominance matters downstream.

What this script does mirrors ``hebo.optimizers.hebo.HEBO.suggest`` at the
pinned commit EXACTLY up to acquisition construction, then evaluates the
official MACE on the externally given pool instead of running EvolutionOpt:

- HEBO's own ``DesignSpace`` encoding (num / pow / int / cat per dimension);
- HEBO's own objective transform (box-cox on y/std when min(y) > 0, else
  yeo-johnson, with the std < 0.5 retry, and the raw-y fallback when the
  power transform or model fit fails);
- HEBO's default ``gp`` (gpytorch) surrogate with HEBO's own model_config
  (lr=0.01, num_epochs=100, noise_lb=8e-4, pred_likeli=False, num_uniqs per
  categorical dimension);
- ``best_y = py_best`` — the surrogate's predicted mean at the argmin
  observation (HEBO's "LCB < py_best"), not raw min(y);
- HEBO's kappa schedule with n_suggestions=1 (iter = n_observations).

Ranking external candidates with ``MACE.eval`` changes NO algorithm
semantics: evaluation of the acquisition at given points is exactly what
HEBO's own acquisition optimizer does internally. Nothing of MACE is
reimplemented here.

All stochasticity (torch GP init + pSGLD fitting + MACE's noise perturbation,
numpy, stdlib random) is seeded from the cell seed before any HEBO call.
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


def compute(payload: dict) -> list:
    import random

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.preprocessing import power_transform

    from hebo.acquisitions.acq import MACE
    from hebo.design_space.design_space import DesignSpace
    from hebo.models.model_factory import get_model

    seed = int(payload["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    space = DesignSpace().parse(_hebo_space_config(payload["search_space"]))
    history = payload["history"]
    pool = payload["pool"]
    if not history:
        raise ValueError("history is empty; the arm's WARMUP gate must prevent this")
    if not pool:
        raise ValueError("pool is empty")

    df_X = pd.DataFrame([row["params"] for row in history])
    y_raw = np.array([[float(row["score"])] for row in history], dtype=float)
    X, Xe = space.transform(df_X)

    # --- HEBO.suggest (pinned commit): objective transform + surrogate fit ---
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

    # --- HEBO.suggest: best_y and kappa for the acquisition -----------------
    best_id = int(np.argmin(y_raw.reshape(-1)))
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

    # --- official MACE evaluated on the external pool ------------------------
    acq = MACE(model, best_y=py_best, kappa=kappa)
    xp, xep = space.transform(pd.DataFrame(pool))
    out = acq.eval(xp, xep)  # minimize-convention columns [lcb, -log EI, -log PI]
    return (-out).detach().numpy().tolist()  # maximize convention (module docstring)


def main() -> int:
    payload = json.loads(sys.stdin.read())
    try:
        # HEBO/gpytorch print fitting chatter (e.g. "jitter = ...") to stdout;
        # keep stdout reserved for the single JSON answer.
        with contextlib.redirect_stdout(sys.stderr):
            values = compute(payload)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps({"values": values}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
