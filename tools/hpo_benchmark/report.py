"""HPO-benchmark report (Phase 1).

Reads runs/hard-interactions/hpo-bench/results/*.json (one per candidate ×
optimizer × seed, each holding the patience=inf trial curve) and DERIVES every
patience outcome offline, then tabulates:
  - per (candidate, optimizer): best achievable final + the patience that reaches
    it + whether it beats warm + trials/cost
  - per optimizer: aggregate hit-rate + mean improvement
  - per dimensionality bucket: which optimizer wins (input to rewriting select-method)

Run:  python3 tools/hpo_benchmark/report.py
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESDIR = ROOT / "runs" / "hard-interactions" / "hpo-bench" / "results"
PATIENCES = [6, 12, 20, 40, 10**9]  # 10**9 == infinity
EPS = 1e-9


def derive(warm_best, trial_values, patience):
    """Replay early-stop with `patience` over the post-warm curve, anchored at
    warm_best. Returns (final_best, trials_used)."""
    best = warm_best
    since = 0
    used = 0
    for v in trial_values:
        used += 1
        if v < best - EPS:
            best = v
            since = 0
        else:
            since += 1
            if since >= patience:
                break
    return best, used


def pname(p):
    return "inf" if p >= 10**9 else str(p)


def load():
    rows = []
    for f in sorted(RESDIR.glob("*.json")):
        d = json.loads(f.read_text())
        if "error" in d or d.get("warm_best") is None:
            continue
        rows.append(d)
    return rows


def main():
    rows = load()
    if not rows:
        print("no results yet at", RESDIR)
        return
    # average over seeds: group by (candidate, optimizer, patience)
    by_co = defaultdict(list)  # (cand,opt) -> list of row
    for r in rows:
        by_co[(r["candidate"], r["optimizer"])].append(r)

    print(f"=== per (candidate × optimizer): final by patience (avg over {1} seed(s)) ===")
    print(f"{'candidate':<12} {'n_dims':>6} {'optimizer':<8} {'warm':>9} " +
          " ".join(f"p={pname(p):>4}" for p in PATIENCES) + "  best_imp")
    # collect for aggregates
    opt_hits = defaultdict(lambda: [0, 0, 0.0])  # opt -> [hits, total, sum_imp]
    bucket_best = defaultdict(lambda: defaultdict(list))  # dimbucket -> opt -> [imp]

    def bucket(n):
        return "low(≤3)" if n <= 3 else ("mid(4-12)" if n <= 12 else "high(≥13)")

    for (cand, opt), rs in sorted(by_co.items()):
        n_dims = rs[0]["n_dims"]
        warm = rs[0]["warm_best"]
        finals = {}
        for p in PATIENCES:
            vals = [derive(r["warm_best"], r["trial_values"], p)[0] for r in rs]
            finals[p] = sum(vals) / len(vals)
        best_final = min(finals.values())
        imp = warm - best_final  # positive = improvement (lower is better)
        opt_hits[opt][1] += 1
        opt_hits[opt][2] += imp
        if imp > 1e-4:
            opt_hits[opt][0] += 1
        bucket_best[bucket(n_dims)][opt].append(imp)
        print(f"{cand:<12} {n_dims:>6} {opt:<8} {warm:>9.4f} " +
              " ".join(f"{finals[p]:>6.4f}" for p in PATIENCES) +
              f"  {'+'+format(imp,'.4f') if imp>1e-4 else '  ~0   '}")

    print("\n=== per optimizer: hit-rate + mean improvement (over candidates) ===")
    print(f"{'optimizer':<8} {'hit/total':>10} {'mean_imp':>9}")
    for opt, (h, t, s) in sorted(opt_hits.items()):
        print(f"{opt:<8} {f'{h}/{t}':>10} {s/t:>9.4f}")

    print("\n=== per dimensionality bucket: best optimizer (→ rewrite select-method) ===")
    for buck in ["low(≤3)", "mid(4-12)", "high(≥13)"]:
        if buck not in bucket_best:
            continue
        ranked = sorted(((opt, sum(v)/len(v)) for opt, v in bucket_best[buck].items()),
                        key=lambda x: -x[1])
        print(f"  {buck:<10} " + " > ".join(f"{o}({i:+.4f})" for o, i in ranked))


if __name__ == "__main__":
    main()
