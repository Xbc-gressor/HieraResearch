"""Full equal-budget re-tune (block coordinate descent) on the two-level toy,
emitting markdown data tables + the final cfg. Each group: sweep at fixed
max_evals, pick winner (final@budget, per-toy normalized), fix into base, next.

  python3 tools/got_benchmark/retune.py --max-evals 200 --seeds 8 > /tmp/retune.md
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "got_benchmark"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import sweep_inner as si  # noqa: E402
from got_select import DEFAULT_CFG  # noqa: E402

SEQ = ["G2", "G1", "G1tau", "G3", "G4", "G5K", "G5pat", "G5topP", "G1"]  # last G1 = refback
TITLES = {"G2": "G2 形状 B/C/alpha", "G1": "G1 探索 c_pucb/c_leaf", "G1tau": "tau",
          "G3": "G3 gamma", "G4": "G4 多样性 n_seed/S", "G5K": "G5 内层 K",
          "G5pat": "G5 内层 patience", "G5topP": "G5 内层 top_percentile"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-evals", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    base_cfg = dict(DEFAULT_CFG); base_inner = dict(si.BASE_INNER)
    toys = list(si.bi.it.TOY_SPECS)

    print(f"# 等预算重调数据（max_evals={args.max_evals}, {len(seeds)} seeds, 3 toy）\n")
    print(f"> 两层 toy，final@{args.max_evals}（越低越好），norm=跨 toy min-max 归一化均值。块坐标下降，最后一组 G1=回扫。\n")
    for i, grp in enumerate(SEQ):
        tag = TITLES.get(grp, grp) + (" (回扫)" if grp == "G1" and i > 0 else "")
        scored = si.sweep(base_cfg, base_inner, si.GROUPS[grp], args.max_evals, seeds, metric="final")
        keys = list(si.GROUPS[grp])
        print(f"\n## {tag}")
        print("| norm | " + " | ".join(keys) + " | " + " | ".join(toys) + " |")
        print("|" + "---|" * (len(keys) + len(toys) + 1))
        for agg, combo, pt in scored:
            print(f"| {agg:.3f} | " + " | ".join(str(combo[k]) for k in keys)
                  + " | " + " | ".join(f"{pt[t]:.4f}" for t in toys) + " |")
        winner = scored[0][1]
        for k, v in winner.items():
            (base_inner if k in si.INNER_KEYS else base_cfg)[k] = v
        print(f"\n**赢家: {winner}**")

    print("\n## 最终调后配置（等预算）")
    print(f"- **外层 cfg**: {base_cfg}")
    print(f"- **内层**: {base_inner}（n_min 由 top_percentile 派生）")


if __name__ == "__main__":
    main()
