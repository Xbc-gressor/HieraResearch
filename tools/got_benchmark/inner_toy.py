"""Two-level toy: outer architecture search (as before) + a synthetic INNER HP
landscape per candidate, so the inner tuner (warm-start K + patience search) is
actually exercised. Lets K / patience / n_trials / top_percentile / n_min be tuned
offline together with the outer params.

Candidate score = base(genome) + bowl(hp), where
  base(genome)  = Σ(genome[d]-target[d])²  — the architecture quality (same
                  separable+complementary structure as the outer toy: crossover
                  combining complementary lineages lowers base).
  bowl(hp)      = bowl_scale · mean((hp - hp_opt)²)  — an H-dim HP penalty the
                  inner tuner minimizes. hp_opt is per-toy fixed.
Fully-tuned score → base (bowl→0); un-/under-tuned → base + residual bowl.
HEADROOM: with H=6 and only K warm random samples, warm-best-of-K leaves bowl
residual that more inner trials close → deep-tune is genuinely useful (and the
headroom is controllable via H / bowl_scale).
"""
from __future__ import annotations
import random

H = 6              # HP dimensionality of the inner bowl
BOWL_SCALE = 1.2   # weight of the HP penalty vs the architecture base


def make_inner_toy(seed, D, dir_dims, p_crash=0.05, fresh_noise=0.02,
                   improve_noise=0.04, H=H, bowl_scale=BOWL_SCALE):
    rng = random.Random(seed)
    target = [rng.uniform(0, 1) for _ in range(D)]
    hp_opt = [rng.uniform(0.2, 0.8) for _ in range(H)]
    dir_ids = [f"tf-{i+1:02d}" for i in range(len(dir_dims))]
    dirs = dict(zip(dir_ids, dir_dims))

    def base(gn):
        return sum((gn[d] - target[d]) ** 2 for d in range(D))

    def bowl(hp):
        return bowl_scale * sum((hp[i] - hp_opt[i]) ** 2 for i in range(H)) / H

    # ---- outer ops (same shape as the outer toy) ----
    def make_fresh(dir_id):
        owned = set(dirs[dir_id])
        return [(target[d] + rng.gauss(0, fresh_noise)) if d in owned
                else rng.uniform(0, 1) for d in range(D)]

    def make_improve(gn):
        return [min(1.0, max(0.0, gn[d] + rng.gauss(0, improve_noise))) for d in range(D)]

    def make_crossover(a, b):
        return [a[d] if rng.random() < 0.5 else b[d] for d in range(D)]

    ops = {"fresh": make_fresh, "improve": make_improve, "crossover": make_crossover}

    # ---- inner: warm-start best-of-K, and patience-bounded deep-tune ----
    def warm_eval(gn, K):
        """Step-0+1 score: base + best-of-K random HP samples (or crash).
        Returns (score, crashed, n_evals_used) — each HP sample is one validation."""
        if rng.random() < p_crash:
            return (float("inf"), True, 1)  # crashed on the first config attempt
        b = base(gn)
        best_bowl = min(bowl([rng.random() for _ in range(H)]) for _ in range(K))
        return (b + best_bowl, False, K)

    def deep_tune(gn, warm_score, n_trials, patience):
        """Continue searching HP from the warm best; random proposals with
        patience early-stop. Returns (improved_score, n_evals_used)."""
        b = base(gn)
        best, since, used = warm_score, 0, 0
        for _ in range(n_trials):
            used += 1
            s = b + bowl([rng.random() for _ in range(H)])
            if s < best - 1e-12:
                best, since = s, 0
            else:
                since += 1
                if since >= patience:
                    break
        return best, used

    return dict(ops=ops, dir_priority=dir_ids, warm_eval=warm_eval,
                deep_tune=deep_tune, base=base, bowl=bowl, hp_opt=hp_opt)


TOY_SPECS = {
    "base": dict(D=8, dir_dims=[[0, 1], [2, 3], [4, 5], [6, 7]], p_crash=0.05),
    "manydir": dict(D=12, dir_dims=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], p_crash=0.05),
    "rugged": dict(D=10, dir_dims=[[0, 1, 2], [3, 4], [5, 6], [7, 8, 9]], p_crash=0.12, improve_noise=0.08),
}


def _headroom_check():
    """For a fixed good genome: how warm-best-of-K and deep-tune close the bowl."""
    import statistics as st
    print(f"headroom check (H={H}, bowl_scale={BOWL_SCALE}) — bowl residual by K, then deep-tune")
    for toy in TOY_SPECS:
        spec = TOY_SPECS[toy]
        t = make_inner_toy(seed=0, **spec)
        gn = t["ops"]["fresh"]("tf-01")  # a single fresh (only 1 lineage right → base not great)
        b = t["base"](gn)
        # warm best-of-K bowl residual (avg over a few rng draws via fresh re-seeds)
        warmK = {}
        for K in (1, 3, 5, 8):
            vals = []
            for s in range(8):
                tt = make_inner_toy(seed=s, **spec)
                g2 = tt["ops"]["fresh"]("tf-01")
                ws, _, _ = tt["warm_eval"](g2, K)
                vals.append(ws - tt["base"](g2))  # bowl residual
            warmK[K] = st.mean(vals)
        # deep-tune from K=5 warm, patience 20, n_trials 40
        dt = []
        for s in range(8):
            tt = make_inner_toy(seed=s, **spec)
            g2 = tt["ops"]["fresh"]("tf-01")
            ws, _, _ = tt["warm_eval"](g2, 5)
            fs, _ = tt["deep_tune"](g2, ws, n_trials=40, patience=20)
            dt.append(fs - tt["base"](g2))
        print(f"  {toy:8} bowl残差: " + " ".join(f"K{k}={warmK[k]:.4f}" for k in (1, 3, 5, 8))
              + f"  →深调(K5,p20,40)={st.mean(dt):.4f}")


if __name__ == "__main__":
    _headroom_check()
