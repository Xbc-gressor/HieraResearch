"""RGPE-ensemble variant of the official-HEBO MACE ranker (DESIGN-rgpe-history-transfer-v2 §1.2).

Sibling of ``rank.py``: DesignSpace / target GP / py_best / kappa / MACE.eval
are identical. The ONE structural change is the surrogate MACE sees: an
RGPE(mean) ensemble of the target GP and one base GP per donor trajectory,
wrapped unconditionally by ``_SaneGP``.

Payload = ``rank.py``'s payload plus::

    "base_tasks": [{"name": str,
                    "shared_dims": [<recipient dim name>, ...],
                    "points": [{"params": {<shared dim>: value}, "y_std": float}, ...]},
                   ...],                 // donor_history.build_base_tasks output
    "rgpe_horizon": <int>,               // H of eq. 9: total planned target trials
    "rgpe_bootstrap": <int>,             // optional, default 1000 (S of Alg. 1)
    "base_cache_key": <str>              // optional: in-process base-GP cache key

history contains TARGET observations only.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys

TARGET_KEY = "__target__"
DEFAULT_BOOTSTRAP = 1000

_BASE_CACHE: dict = {}


def _hebo_space_config(search_space: dict) -> list[dict]:
    """Raw contract SEARCH_SPACE -> HEBO DesignSpace.parse records."""
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


class _BaseTask:
    """One donor trajectory as a fitted HEBO GP over the recipient's shared dims."""

    def __init__(self, name: str, shared_dims: list, space, model) -> None:
        self.name = name
        self.shared_dims = list(shared_dims)
        self.space = space
        self.model = model

    def predict_mean(self, df_X):
        xc, xe = self.space.transform(df_X[self.shared_dims])
        mu, _ = self.model.predict(xc, xe)
        return mu.detach().reshape(-1)


def _fit_base_tasks(base_tasks: list, search_space: dict, scramble_seed: int,
                    cache_key) -> list:
    import numpy as np
    import pandas as pd
    import torch

    from hebo.design_space.design_space import DesignSpace
    from hebo.models.model_factory import get_model

    key = None
    if cache_key is not None:
        key = (str(cache_key), int(scramble_seed),
               tuple((t["name"], len(t["points"])) for t in base_tasks))
        if key in _BASE_CACHE:
            return _BASE_CACHE[key]

    torch.manual_seed(int(scramble_seed))
    np.random.seed(int(scramble_seed) % (2**31 - 1))
    fitted = []
    for task in base_tasks:
        shared = [name for name in search_space if name in set(task["shared_dims"])]
        if not shared or not task["points"]:
            continue
        sub_space = DesignSpace().parse(
            _hebo_space_config({name: search_space[name] for name in shared})
        )
        df = pd.DataFrame([point["params"] for point in task["points"]])[shared]
        y = torch.FloatTensor([[float(point["y_std"])] for point in task["points"]])
        xc, xe = sub_space.transform(df)
        model = get_model(
            "gp", sub_space.num_numeric, sub_space.num_categorical, 1,
            **_model_config(sub_space),
        )
        model.fit(xc, xe, y)
        fitted.append(_BaseTask(task["name"], shared, sub_space, model))
    if key is not None:
        _BASE_CACHE[key] = fitted
    return fitted


def _gp_loo_means(model):
    """Closed-form leave-one-out posterior means of a fitted HEBO GP."""
    import gpytorch
    import torch

    gp = model.gp
    y = model.y.reshape(-1)
    try:
        gp.train()
        model.lik.train()
        with torch.no_grad(), gpytorch.settings.debug(False):
            prior = gp(model.Xc, model.Xe)
            K = prior.covariance_matrix.clone()
            m = prior.mean.reshape(-1)
        noise = gp.likelihood.noise.detach().reshape(-1)[0]
        K = K + torch.eye(K.shape[0]) * (noise + 1e-6)
        K_inv = torch.linalg.inv(K.double())
        alpha = K_inv @ (y - m).double()
        loo = y.double() - alpha / torch.diagonal(K_inv)
        return loo.float().detach().numpy()
    finally:
        gp.eval()
        model.lik.eval()


def _rgpe_weights(preds, y, *, n_target: int, horizon: int, bootstrap: int, rng):
    """Algorithm 1 + eq. 9 dilution. preds: (M+1, n) array, base models first, target LAST."""
    import numpy as np

    n_models, n = preds.shape
    if n_target < 3 or n < 2:
        return np.full(n_models, 1.0 / n_models), []
    idx = rng.integers(0, n, size=(bootstrap, n))
    y_b = y[idx]
    truth = y_b[:, :, None] < y_b[:, None, :]
    losses = np.empty((n_models, bootstrap), dtype=float)
    for i in range(n_models):
        p = preds[i][idx]
        losses[i] = ((p[:, :, None] < p[:, None, :]) != truth).sum(axis=(1, 2))

    target = n_models - 1
    dropped = []
    frac = min(1.0, max(0.0, n_target / float(horizon))) if horizon > 0 else 1.0
    for i in range(target):
        p_better = float(np.mean(losses[i] < losses[target]))
        p_drop = min(1.0, max(0.0, 1.0 - (1.0 - frac) * p_better))
        if rng.random() < p_drop:
            losses[i] = np.inf
            dropped.append(i)

    weights = np.zeros(n_models)
    minima = losses.min(axis=0)
    is_min = losses == minima[None, :]
    counts = is_min.sum(axis=0).astype(float)
    weights = (is_min / counts[None, :]).sum(axis=1) / bootstrap
    return weights, dropped


class _Ensemble:
    """RGPE(mean): weighted mean of target + base GPs, target variance/noise."""

    def __init__(self, target_model, space, base_tasks: list, weights) -> None:
        self.target = target_model
        self.space = space
        self.base_tasks = base_tasks
        self.weights = [float(w) for w in weights]

    def predict(self, xc, xe):
        mu_t, var_t = self.target.predict(xc, xe)
        mu = mu_t.reshape(-1) * self.weights[-1]
        active = [(w, task) for w, task in zip(self.weights[:-1], self.base_tasks) if w > 0]
        if active:
            df = self.space.inverse_transform(xc, xe)
            for w, task in active:
                mu = mu + w * task.predict_mean(df)
        return mu.reshape(-1, 1), var_t

    @property
    def noise(self):
        return self.target.noise


class _DummyConstantGP:
    """Fallback surrogate when target observations are too sparse for HEBO GP fit."""

    def __init__(self, y_val: float, noise: float = 1e-4) -> None:
        import torch
        self.y_val = float(y_val)
        self.noise = torch.tensor(float(noise))
        self.y = torch.tensor([[float(y_val)]])

    def predict(self, xc, xe):
        import torch
        n = len(xc) if xc is not None and len(xc) > 0 else (len(xe) if xe is not None else 1)
        mu = torch.full((n, 1), self.y_val, dtype=torch.float32)
        var = torch.full((n, 1), 1.0, dtype=torch.float32)
        return mu, var


class _SaneGP:
    """Bounds a surrogate fitted on very few target observations."""

    def __init__(self, model, y) -> None:
        self.model = model
        lo, hi = float(y.min()), float(y.max())
        span = max(hi - lo, 1e-6)
        self.lo, self.hi = lo - span, hi + span
        self.var_floor = (0.01 * span) ** 2
        self.var_ceil = span ** 2

    def predict(self, xc, xe):
        import torch

        mu, var = self.model.predict(xc, xe)
        mu = torch.nan_to_num(mu, nan=self.hi, posinf=self.hi, neginf=self.lo)
        var = torch.nan_to_num(
            var, nan=self.var_floor, posinf=self.var_ceil, neginf=self.var_floor
        )
        return mu.clamp(self.lo, self.hi), var.clamp(self.var_floor, self.var_ceil)

    @property
    def noise(self):
        return self.model.noise


def compute(payload: dict) -> tuple[list, dict | None]:
    import random

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.preprocessing import power_transform

    from hebo.acquisitions.acq import MACE
    from hebo.design_space.design_space import DesignSpace
    from hebo.models.model_factory import get_model

    seed = int(payload["seed"])
    torch.set_num_threads(min(1, torch.get_num_threads()))

    space = DesignSpace().parse(_hebo_space_config(payload["search_space"]))
    history = payload["history"]
    pool = payload["pool"]
    if not history:
        raise ValueError("history is empty")
    if not pool:
        raise ValueError("pool is empty")

    base_task_specs = payload.get("base_tasks") or []
    scramble_seed = int(payload.get("scramble_seed", seed))
    base_tasks = _fit_base_tasks(
        base_task_specs,
        payload["search_space"],
        scramble_seed,
        payload.get("base_cache_key"),
    )

    # Seed target-side stochasticity AFTER base tasks fit
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df_X = pd.DataFrame([row["params"] for row in history])
    y_raw = np.array([[float(row["score"])] for row in history], dtype=float)
    X, Xe = space.transform(df_X)

    # --- HEBO objective transform + surrogate fit ---
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
        try:
            model = get_model("gp", space.num_numeric, space.num_categorical, 1, **_model_config(space))
            model.fit(X, Xe, y)
        except Exception:
            model = _DummyConstantGP(float(y_raw[0, 0]))

    rgpe_info = None
    surrogate = model
    if base_tasks:
        n_target = df_X.shape[0]
        horizon = int(payload.get("rgpe_horizon") or 0)
        bootstrap = int(payload.get("rgpe_bootstrap") or DEFAULT_BOOTSTRAP)
        if isinstance(model, _DummyConstantGP):
            target_loo = y_raw.reshape(-1)
        else:
            target_loo = _gp_loo_means(model)
        preds = np.vstack(
            [task.predict_mean(df_X).numpy() for task in base_tasks]
            + [target_loo]
        )
        weights, dropped = _rgpe_weights(
            preds,
            y_raw.reshape(-1),
            n_target=n_target,
            horizon=horizon,
            bootstrap=bootstrap,
            rng=np.random.default_rng(seed),
        )
        surrogate = _Ensemble(model, space, base_tasks, weights)
        rgpe_info = {
            "weights": {
                **{task.name: float(w) for task, w in zip(base_tasks, weights[:-1])},
                TARGET_KEY: float(weights[-1]),
            },
            "dropped": [base_tasks[i].name for i in dropped],
            "n_target": int(n_target),
            "horizon": horizon,
        }

    # _SaneGP unconditionally wraps surrogate
    surrogate = _SaneGP(surrogate, y)

    # --- best_x / py_best / kappa ---
    best_id = int(np.argmin(y_raw.reshape(-1)))
    best_x = df_X.iloc[[best_id]]
    py_best, _ = surrogate.predict(*space.transform(best_x))
    py_best = float(py_best.detach().numpy().squeeze())

    n_obs = df_X.shape[0]
    iter_ = max(1, n_obs // 1)
    upsi, delta = 0.5, 0.01
    n_dim = df_X.shape[1]
    kappa = math.sqrt(
        upsi * 2 * ((2.0 + n_dim / 2.0) * math.log(iter_) + math.log(3 * math.pi**2 / (3 * delta)))
    )

    # --- official MACE evaluated on external pool ---
    acq = MACE(surrogate, best_y=py_best, kappa=kappa)
    xp, xep = space.transform(pd.DataFrame(pool))
    out = acq.eval(xp, xep)
    values = (-out).detach().numpy().tolist()
    return values, rgpe_info


def main() -> int:
    payload = json.loads(sys.stdin.read())
    try:
        with contextlib.redirect_stdout(sys.stderr):
            values, rgpe_info = compute(payload)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    resp = {"values": values}
    if rgpe_info is not None:
        resp["rgpe"] = rgpe_info
    print(json.dumps(resp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
