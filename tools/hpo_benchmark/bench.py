"""HPO-benchmark runner (Phase 1).

For each (candidate × optimizer × seed): inject the candidate's warm configs as
priors, run the optimizer for a FIXED trial budget at patience=infinity, and log
the ordered post-warm trial scores + the warm-best anchor. patience outcomes are
derived OFFLINE from these curves by report.py (no per-patience reruns).

Optimizers here (batch 1, no new deps — all via optuna):
  random · tpe · tpe+ (multivariate) · cmaes
Batch 2 (GP-BO / SMAC / agent) plug into the same registry + output schema.

Run (per the task env, ABS path):
  uv --directory tasks/hard-interactions run python <ROOT>/tools/hpo_benchmark/bench.py \
      --candidates L1-svm,M2-stack --optimizers random,tpe --n-trials 8 --seeds 1
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import optuna  # noqa: E402
from optuna.distributions import (  # noqa: E402
    FloatDistribution, IntDistribution, CategoricalDistribution,
)
from _common import load_candidate_modules, resolve_score_fn  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

BENCH = ROOT / "runs" / "hard-interactions" / "hpo-bench"
CANDDIR = BENCH / "candidates"
RESDIR = BENCH / "results"
STUDYDIR = BENCH / "_studies"  # per-cell optuna sqlite journals (intra-cell resume)


def _dist(spec):
    kind = spec[0]
    if kind == "int":
        return IntDistribution(int(spec[1]), int(spec[2]))
    if kind == "float":
        return FloatDistribution(float(spec[1]), float(spec[2]),
                                 log=(len(spec) > 3 and spec[3] == "log"))
    if kind == "categorical":
        return CategoricalDistribution(list(spec[1]))
    raise ValueError(f"bad spec {spec}")


def _suggest(trial, space):
    p = {}
    for k, spec in space.items():
        kind = spec[0]
        if kind == "int":
            p[k] = trial.suggest_int(k, int(spec[1]), int(spec[2]))
        elif kind == "float":
            p[k] = trial.suggest_float(k, float(spec[1]), float(spec[2]),
                                       log=(len(spec) > 3 and spec[3] == "log"))
        else:
            p[k] = trial.suggest_categorical(k, list(spec[1]))
    return p


def _cast_warm(params, space):
    out = {}
    for k, spec in space.items():
        if k not in params:
            return None  # warm config missing a key → skip as a prior
        v = params[k]
        out[k] = int(v) if spec[0] == "int" else (float(v) if spec[0] == "float" else v)
    return out


def make_sampler(name, seed):
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "tpe+":
        return optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True,
                                          n_startup_trials=10)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    if name == "cmaes+":
        return optuna.samplers.CmaEsSampler(seed=seed, restart_strategy="ipop",
                                            popsize=8, lr_adapt=True)
    if name == "gp":
        return optuna.samplers.GPSampler(seed=seed)
    raise ValueError(f"unknown optimizer {name}")


def load_candidate(label):
    cdir = CANDDIR / label
    train_mod, prep_mod = load_candidate_modules(cdir / "train.py")
    evaluate = resolve_score_fn(prep_mod, cdir / "train.py")
    space = train_mod.SEARCH_SPACE
    make_model = train_mod.make_model
    rep = json.loads((cdir / "tune_report.json").read_text())
    warm = [(c["params"], c["score"]) for c in rep["phase_a"]["warm_start_configs"]
            if isinstance(c.get("score"), (int, float))]
    warm_best = min((s for _, s in warm), default=None)
    return make_model, evaluate, space, warm, warm_best


OPTUNA_OPTS = {"random", "tpe", "tpe+", "cmaes", "cmaes+", "gp"}


def _run_optuna(label, opt, n_trials, seed):
    make_model, evaluate, space, warm, warm_best = load_candidate(label)
    dists = {k: _dist(v) for k, v in space.items()}
    n_warm = sum(1 for p, _ in warm if _cast_warm(p, space) is not None)
    STUDYDIR.mkdir(parents=True, exist_ok=True)
    sname = f"{label}__{opt}__s{seed}"
    # sqlite-persisted study → a reaped cell resumes mid-run (intra-cell checkpoint).
    study = optuna.create_study(
        direction="minimize", sampler=make_sampler(opt, seed),
        study_name=sname, storage=f"sqlite:///{STUDYDIR / (sname + '.db')}",
        load_if_exists=True)
    if len(study.trials) == 0:  # brand new → seed warm priors once
        for params, score in warm:
            cp = _cast_warm(params, space)
            if cp is not None:
                study.add_trial(optuna.trial.create_trial(
                    params=cp, distributions=dists, value=float(score)))

    def objective(trial):
        return float(evaluate(make_model, _suggest(trial, space)))

    done = max(0, len(study.trials) - n_warm)
    remaining = max(0, n_trials - done)
    t0 = time.time()
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    post = [t.value for t in study.trials[n_warm:] if t.value is not None]
    return space, warm_best, post, time.time() - t0


def _run_smac(label, n_trials, seed):
    """SMAC (random-forest-surrogate BO) via ask/tell; warm configs seeded into
    the runhistory so they prime the RF surrogate (parity with the BO priors)."""
    import tempfile
    from ConfigSpace import ConfigurationSpace, Configuration, Float, Integer, Categorical
    from smac import HyperparameterOptimizationFacade as HPO, Scenario
    from smac.runhistory.dataclasses import TrialInfo, TrialValue

    make_model, evaluate, space, warm, warm_best = load_candidate(label)
    cs = ConfigurationSpace(seed=seed)
    for k, spec in space.items():
        kind = spec[0]
        if kind == "int":
            cs.add(Integer(k, (int(spec[1]), int(spec[2]))))
        elif kind == "float":
            cs.add(Float(k, (float(spec[1]), float(spec[2])),
                         log=(len(spec) > 3 and spec[3] == "log")))
        else:
            cs.add(Categorical(k, list(spec[1])))

    post = []
    with tempfile.TemporaryDirectory() as tmp:
        scenario = Scenario(cs, n_trials=len(warm) + n_trials, deterministic=True,
                            output_directory=Path(tmp), seed=seed)
        smac = HPO(scenario, lambda config, seed=0: 0.0, overwrite=True,
                   logging_level=False)
        for params, score in warm:
            cp = _cast_warm(params, space)
            if cp is None:
                continue
            try:
                smac.tell(TrialInfo(Configuration(cs, values=cp), seed=seed),
                          TrialValue(cost=float(score)))
            except Exception:
                pass  # a warm config outside CS bounds → skip as a prior
        t0 = time.time()
        for _ in range(n_trials):
            info = smac.ask()
            cost = float(evaluate(make_model, dict(info.config)))
            smac.tell(info, TrialValue(cost=cost))
            post.append(cost)
        elapsed = time.time() - t0
    return space, warm_best, post, elapsed


def run_one(label, opt, n_trials, seed):
    if opt in OPTUNA_OPTS:
        space, warm_best, post, elapsed = _run_optuna(label, opt, n_trials, seed)
    elif opt == "smac":
        space, warm_best, post, elapsed = _run_smac(label, n_trials, seed)
    else:
        raise ValueError(f"unknown optimizer {opt}")
    if not post:
        raise RuntimeError(f"no completed trials for {opt} (optimizer unavailable?)")
    return {
        "candidate": label, "optimizer": opt, "seed": seed,
        "n_dims": len(space), "warm_best": warm_best,
        "trial_values": post, "n_trials": len(post),
        "final_best": min([warm_best] + post) if warm_best is not None else min(post, default=None),
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="L1-svm,L2-rf,M1-gbdt,M2-stack,M3-mlp,M4-xgb,H1-bigstack,H2-deepmlp")
    ap.add_argument("--optimizers", default="random,tpe,tpe+,cmaes")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (0..seeds-1)")
    args = ap.parse_args()
    RESDIR.mkdir(parents=True, exist_ok=True)
    cands = args.candidates.split(",")
    opts = args.optimizers.split(",")
    for label in cands:
        for opt in opts:
            for seed in range(args.seeds):
                tag = f"{label}__{opt}__s{seed}"
                out = RESDIR / f"{tag}.json"
                if out.exists():
                    prev = json.loads(out.read_text())
                    if "error" not in prev and prev.get("n_trials", 0) >= args.n_trials:
                        print(f"{tag}: (cached n={prev.get('n_trials')}, skip)")
                        continue
                try:
                    res = run_one(label, opt, args.n_trials, seed)
                except Exception as exc:
                    res = {"candidate": label, "optimizer": opt, "seed": seed,
                           "error": f"{type(exc).__name__}: {exc}"[:300]}
                (RESDIR / f"{tag}.json").write_text(json.dumps(res, indent=2))
                imp = ("" if "error" in res else
                       (f"warm={res['warm_best']:.4f}→final={res['final_best']:.4f}"
                        + (" IMPROVED" if res['final_best'] < res['warm_best'] - 1e-9 else "")))
                print(f"{tag}: {res.get('error', imp)} [{res.get('elapsed_sec','?')}s]")


if __name__ == "__main__":
    main()
