"""RGPE-ensemble variant of the official-HEBO suggest mirror
(DESIGN-rgpe-history-transfer-v1 §1.2; Feurer, Letham, Hutter & Bakshy,
"Practical Transfer Learning for Bayesian Optimization", arXiv:1802.02219).

Sibling of ``suggest.py``: the DesignSpace / Sobol warmup / objective
transform / target GP / kappa / MACE / EvolutionOpt / uniqueness top-up
sections are copied from it unchanged (that layer's convention — keep them in
sync by hand). The ONE structural change is the surrogate MACE sees: an
RGPE(mean) ensemble of the target GP and one base GP per donor trajectory.

Payload = ``suggest.py``'s payload (without ``pool``) plus::

    "base_tasks": [{"name": str,
                    "shared_dims": [<recipient dim name>, ...],
                    "points": [{"params": {<shared dim>: value}, "y_std": float}, ...]},
                   ...],                 // donor_history.build_base_tasks output
    "rgpe_horizon": <int>,               // H of eq. 9: total planned target trials
    "rgpe_bootstrap": <int>,             // optional, default 1000 (S of Alg. 1)
    "rgpe_warmup": <int>,                // optional: override the warmup length
                                         // (control arms: RGPE's 3-point start
                                         // without base tasks)
    "base_cache_key": <str>              // optional: in-process base-GP cache key

``history`` / ``n_obs`` / ``kappa`` / ``best_x`` are TARGET observations only —
donor points never masquerade as target observations.

Ensemble (RGPE(mean), paper §4 + ablation): ``mu = sum_i w_i mu_i(x_shared_i)``,
``var = var_target``, ``noise = noise_target``. Weights follow Algorithm 1:
pairwise ranking loss (eq. 3) on the target observations, the target model's
own predictions taken leave-one-out (eq. 4) in closed form from its fitted GP
(Rasmussen & Williams eq. 5.12), S bootstrap replicates, weight = fraction of
replicates in which the model has the lowest loss (ties broken uniformly).
Weight dilution (eq. 9): base i is dropped with probability
``clip(1 - (1 - n_t/H) * P(l_i < l_t), 0, 1)`` before the argmin. Uniform
weights while ``n_t < 3``.

Warmup: with at least one base task the surrogate branch starts at 3 target
observations instead of the official ``1 + num_paras``; without base tasks
this file behaves exactly like ``suggest.py`` (no ``rgpe`` result key).

Base GPs are fitted under ``scramble_seed`` (trajectory-constant) and cached
per ``base_cache_key`` in-process; the global RNGs are re-seeded with the
step ``seed`` afterwards, so the target-side random stream is identical
whether or not the cache hit. Result adds ``"rgpe": {"weights": {name: w,
"__target__": w_t}, "dropped": [names], "n_target": n_t, "horizon": H}`` on
surrogate steps that used base tasks.

Usable in-process (``compute(payload)`` / ``suggest``) or as a subprocess over
JSON stdin/stdout like ``suggest.py``.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys

TARGET_KEY = "__target__"
RGPE_WARMUP = 3
DEFAULT_BOOTSTRAP = 1000

_BASE_CACHE: dict = {}


def _hebo_space_config(search_space: dict) -> list[dict]:
    """Raw contract SEARCH_SPACE -> HEBO DesignSpace.parse records (suggest.py)."""
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
    """HEBO.model_config for model_name='gp' at the pinned commit (suggest.py)."""
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
        self.space = space  # DesignSpace over shared dims only, recipient bounds
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
    """Closed-form leave-one-out posterior means of a fitted HEBO GP at its own
    training inputs (Rasmussen & Williams eq. 5.12), in the model's scaled-y
    space (ranking losses are scale-invariant)."""
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
    """Algorithm 1 + eq. 9 dilution. ``preds``: (M+1, n) array, base models
    first, target LAST. Returns (weights (M+1,), dropped base indices)."""
    import numpy as np

    n_models, n = preds.shape
    if n_target < RGPE_WARMUP or n < 2:
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
    """RGPE(mean): weighted mean of target + base GPs, target variance/noise.
    Duck-types the two members MACE reads: ``predict`` and ``noise``."""

    def __init__(self, target_model, space, base_tasks: list, weights) -> None:
        self.target = target_model
        self.space = space
        self.base_tasks = base_tasks
        self.weights = [float(w) for w in weights]  # base weights..., target last

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


class _SaneGP:
    """Bounds a surrogate fitted on very few target observations.

    Starting the surrogate at ``RGPE_WARMUP`` observations exposes a HEBO GP
    failure mode that the official ``1 + num_paras`` warmup never reaches: the
    fit degenerates into a spike at a training point (measured on tabular
    r3-gb-sklearn at n=3: mean -352 where the observed y span is ~2.7) with the
    variance pinned at ``noise_lb``. MACE's standardized improvement
    ``(py_best - mu) / sigma`` then diverges and gpytorch rejects the value as
    outside ``Normal(0, 1)``'s support, losing the whole cell.

    Clamping the mean into the observed band and flooring the variance keeps
    the acquisition finite without touching the ranking-weight logic, which
    measurement shows is behaving correctly (it had already assigned the
    degenerate target GP a weight of 0.018).
    """

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
        # The ill-conditioned few-observation fit also returns outright NaN on a
        # small fraction of calls (measured: 1 of 99), and clamp propagates it.
        # A NaN mean becomes the worst end of the band so the minimizing
        # acquisition never chases it.
        mu = torch.nan_to_num(mu, nan=self.hi, posinf=self.hi, neginf=self.lo)
        var = torch.nan_to_num(
            var, nan=self.var_floor, posinf=self.var_ceil, neginf=self.var_floor
        )
        return mu.clamp(self.lo, self.hi), var.clamp(self.var_floor, self.var_ceil)

    @property
    def noise(self):
        return self.model.noise


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

    base_tasks = _fit_base_tasks(
        payload.get("base_tasks") or [],
        payload["search_space"],
        scramble_seed,
        payload.get("base_cache_key"),
    )
    # Target-side stream: seeded AFTER the (possibly cached) base fits so it is
    # identical to suggest.py's whether or not the cache hit.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if payload.get("rgpe_warmup"):
        rand_sample = int(payload["rgpe_warmup"])
    else:
        rand_sample = RGPE_WARMUP if base_tasks else 1 + space.num_paras

    def quasi_sample(start: int, n: int) -> "pd.DataFrame":
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
        return (
            (~pd.concat([X, rec], axis=0).duplicated().tail(rec.shape[0]).values).tolist()
        )

    df_X = pd.DataFrame([row["params"] for row in history])

    # --- warmup branch (hebo.py:122-124) ------------------------------------
    if df_X.shape[0] < rand_sample:
        sample = quasi_sample(quasi_index, 1)
        return {
            "suggestion": _row_to_params(sample.iloc[0], payload["search_space"]),
            "mode": "quasi",
            "quasi_consumed": 1,
        }

    # --- objective transform + target GP (hebo.py:126-147) ------------------
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

    # --- RGPE weights + ensemble --------------------------------------------
    rgpe_info = None
    surrogate = model
    if base_tasks:
        n_target = df_X.shape[0]
        horizon = int(payload.get("rgpe_horizon") or 0)
        bootstrap = int(payload.get("rgpe_bootstrap") or DEFAULT_BOOTSTRAP)
        preds = np.vstack(
            [task.predict_mean(df_X).numpy() for task in base_tasks]
            + [_gp_loo_means(model)]
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

    surrogate = _SaneGP(surrogate, y)

    # --- best_x / py_best / kappa (hebo.py:149-160) --------------------------
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

    # --- MACE + EvolutionOpt (hebo.py:162-167) ------------------------------
    acq = MACE(surrogate, best_y=py_best, kappa=kappa)
    initial_suggest = pd.concat([pd.DataFrame(extra), best_x], ignore_index=True)
    opt = EvolutionOpt(space, acq, pop=100, iters=100, verbose=False, es=None)
    rec = opt.optimize(initial_suggest=initial_suggest, fix_input=None).drop_duplicates()
    rec = rec[check_unique(df_X, rec)]
    front_size = int(rec.shape[0])

    # --- uniqueness top-up (hebo.py:169-180) --------------------------------
    quasi_consumed = 0
    cnt = 0
    while rec.shape[0] < 1:
        rand_rec = quasi_sample(quasi_index + quasi_consumed, 1)
        quasi_consumed += 1
        rand_rec = rand_rec[check_unique(df_X, rand_rec)]
        rec = pd.concat([rec, rand_rec], axis=0, ignore_index=True)
        cnt += 1
        if cnt > 3:
            break
    if rec.shape[0] < 1:
        rand_rec = quasi_sample(quasi_index + quasi_consumed, 1)
        quasi_consumed += 1
        rec = pd.concat([rec, rand_rec], axis=0, ignore_index=True)

    # --- uniform pick of 1 (hebo.py:182) ------------------------------------
    select_id = np.random.choice(rec.shape[0], 1, replace=False).tolist()
    rec_selected = rec.iloc[select_id].copy()
    result = {
        "suggestion": _row_to_params(rec_selected.iloc[0], payload["search_space"]),
        "mode": "surrogate",
        "quasi_consumed": quasi_consumed,
        "front_size": front_size,
    }
    if rgpe_info is not None:
        result["rgpe"] = rgpe_info
    return result


suggest = compute


def _row_to_params(row, search_space: dict) -> dict:
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
        with contextlib.redirect_stdout(sys.stderr):
            result = compute(payload)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
