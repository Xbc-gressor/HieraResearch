"""Equal-validation-budget group sweeps on the two-level toy (outer+inner).

Every cfg gets the SAME total validation budget (max_evals); ranked by final@budget
(per-toy min-max normalized, averaged). Keys route automatically: K/patience/
top_percentile/n_trials → inner kwargs; everything else → the got cfg.

  python3 tools/got_benchmark/sweep_inner.py --group G2 --max-evals 200
  python3 tools/got_benchmark/sweep_inner.py --group G5K --set B=3,alpha=0.7
"""
from __future__ import annotations
import argparse
import itertools
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "got_benchmark"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import bench_inner as bi  # noqa: E402
from got_select import DEFAULT_CFG  # noqa: E402

INNER_KEYS = {"K", "patience", "top_percentile", "n_trials"}
GROUPS = {
    "G2": {"B": [1, 2, 3, 4], "C": [1.0, 1.5, 2.5], "alpha": [0.3, 0.5, 0.7]},
    "G1": {"c_pucb": [0.1, 0.2, 0.4, 0.8, 1.5], "c_leaf": [0.1, 0.2, 0.4, 0.8, 1.5]},
    "G1tau": {"tau": [0.1, 0.3, 0.6]},
    "G3": {"gamma": [0.3, 0.6, 0.8, 0.95]},
    "G4": {"n_seed": [2, 3, 5, 8], "S": [5, 10, 20, 40]},
    "G5K": {"K": [2, 3, 5, 8]},
    "G5pat": {"patience": [6, 12, 20, 40]},
    "G5topP": {"top_percentile": [70, 80, 90]},
}
BASE_INNER = dict(K=5, patience=20, top_percentile=80, n_trials=40)


def sweep(base_cfg, base_inner, grid, max_evals, seeds, metric="final"):
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    toys = list(bi.it.TOY_SPECS)
    rows = []
    for combo in combos:
        cfg = dict(base_cfg); inner = dict(base_inner)
        for k, v in zip(keys, combo):
            (inner if k in INNER_KEYS else cfg)[k] = v
        rows.append((dict(zip(keys, combo)),
                     bi.run_cfg_inner(cfg, max_evals=max_evals, seeds=seeds, **inner)))
    norm = {}
    for t in toys:
        vals = [pt[t][metric] for _, pt in rows]
        lo, hi = min(vals), max(vals); rng = (hi - lo) or 1.0
        norm[t] = [(v - lo) / rng for v in vals]
    scored = [(mean(norm[t][i] for t in toys), combo,
               {t: round(pt[t][metric], 4) for t in toys})
              for i, (combo, pt) in enumerate(rows)]
    scored.sort(key=lambda x: x[0])
    return scored


def _parse(s):
    out = {}
    for kv in [x for x in s.split(",") if x]:
        k, v = kv.split("=")
        out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(GROUPS))
    ap.add_argument("--max-evals", type=int, default=200)
    ap.add_argument("--set", default="", help="fix prior winners (routed to cfg/inner by key)")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    base_cfg = dict(DEFAULT_CFG); base_inner = dict(BASE_INNER)
    for k, v in _parse(args.set).items():
        (base_inner if k in INNER_KEYS else base_cfg)[k] = v
    scored = sweep(base_cfg, base_inner, GROUPS[args.group], args.max_evals,
                   tuple(range(args.seeds)), metric="final")
    print(f"=== {args.group} | final@{args.max_evals} (lower norm=better) | "
          f"set: {args.set or 'defaults'} | {len(scored)} combos ===")
    for agg, combo, pt in scored[:12]:
        print(f"  norm={agg:.3f}  {combo}  per-toy={pt}")
    print(f"  BEST: {scored[0][1]}  (worst: {scored[-1][1]})")


if __name__ == "__main__":
    main()
