"""Group sweeps for the outer-S-GoT benchmark (block coordinate descent).

Each group = a small joint grid over its coupled params. Runs every combo across
all toys × seeds, normalizes the chosen metric per-toy (min-max → 0=best, since
lower is better), averages across toys, ranks. Thread prior-group winners into the
base via --set so groups are tuned sequentially.

  python3 tools/got_benchmark/sweep.py --group G2 --metric auc
  python3 tools/got_benchmark/sweep.py --group G1 --set B=3,C=1.0,alpha=0.5
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

import bench  # noqa: E402
from got_select import DEFAULT_CFG  # noqa: E402

GROUPS = {
    "G2": {"B": [1, 2, 3, 4], "C": [1.0, 1.5, 2.5], "alpha": [0.3, 0.5, 0.7]},
    "G1": {"c_pucb": [0.1, 0.2, 0.4, 0.8, 1.5], "c_leaf": [0.1, 0.2, 0.4, 0.8, 1.5]},
    "G1tau": {"tau": [0.1, 0.3, 0.6]},
    "G3": {"gamma": [0.3, 0.6, 0.8, 0.95]},
    "G4": {"n_seed": [2, 3, 5, 8], "S": [5, 10, 20, 40]},
}


def sweep(base, grid, seeds=(0, 1, 2), metric="auc"):
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    toys = list(bench.toysmod.TOY_SPECS)
    rows = []
    for combo in combos:
        cfg = dict(base)
        cfg.update(dict(zip(keys, combo)))
        rows.append((dict(zip(keys, combo)), bench.run_cfg(cfg, toys, seeds)))
    # per-toy min-max normalize (lower=better → 0=best); average across toys.
    norm = {}
    for t in toys:
        vals = [pt[t][metric] for _, pt in rows]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        norm[t] = [(v - lo) / rng for v in vals]
    scored = [(mean(norm[t][i] for t in toys), combo,
               {t: round(pt[t][metric], 4) for t in toys})
              for i, (combo, pt) in enumerate(rows)]
    scored.sort(key=lambda x: x[0])
    return scored


def _parse_set(s):
    out = {}
    for kv in [x for x in s.split(",") if x]:
        k, v = kv.split("=")
        out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(GROUPS))
    ap.add_argument("--metric", default="auc")
    ap.add_argument("--set", default="", help="fix prior winners, e.g. B=3,C=1.0,alpha=0.5")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    base = dict(DEFAULT_CFG)
    base.update(_parse_set(args.set))
    scored = sweep(base, GROUPS[args.group], seeds=tuple(range(args.seeds)), metric=args.metric)
    print(f"=== {args.group} | metric={args.metric} (lower norm=better) | "
          f"base override: {args.set or 'defaults'} | {len(scored)} combos ===")
    for agg, combo, pertoy in scored[:10]:
        print(f"  norm={agg:.3f}  {combo}  per-toy={pertoy}")
    print(f"  BEST: {scored[0][1]}  (worst: {scored[-1][1]})")


if __name__ == "__main__":
    main()
