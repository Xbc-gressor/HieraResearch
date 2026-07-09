"""Dump the full outer-benchmark sweep data as a markdown appendix.

Reproduces the block-coordinate-descent sequence + S detail + best-of-K + the
DEFAULT-vs-TUNED comparison, printing full ranked tables (all combos). Output is
appended to dev_plan/got-benchmark-report.md.

  python3 tools/got_benchmark/report_data.py > /tmp/got_appendix.md
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "got_benchmark"))
sys.path.insert(0, str(ROOT / "tools"))

import bench  # noqa: E402
import sweep as swp  # noqa: E402
from got_select import DEFAULT_CFG  # noqa: E402

OLD = dict(n_seed=3, S=10, B=2, C=1.5, alpha=0.5, gamma=0.6, c_pucb=0.4, c_leaf=0.4, tau=0.3)


def table(title, base, grid, note=""):
    scored = swp.sweep(base, grid, seeds=(0, 1, 2), metric="auc")
    keys = list(grid)
    print(f"\n### {title}")
    if note:
        print(f"_{note}_\n")
    print("| norm(AUC) | " + " | ".join(keys) + " | base | manydir | rugged |")
    print("|" + "---|" * (len(keys) + 4))
    for agg, combo, pt in scored:
        cells = " | ".join(str(combo[k]) for k in keys)
        print(f"| {agg:.3f} | {cells} | {pt['base']:.4f} | {pt['manydir']:.4f} | {pt['rugged']:.4f} |")


def main():
    print("## 附录 A：详细扫描数据")
    print("\n> 离线 toy 块坐标下降，3 toy × 3 seed，metric = AUC（best 曲线均值，越低越好），"
          "norm = 跨 toy min-max 归一化后均值（0=该组最佳）。`run_sgot` n_gens=60。复现：`tools/got_benchmark/sweep.py`。")

    table("A1. G2 形状（base = 旧默认）", dict(OLD),
          {"B": [1, 2, 3, 4], "C": [1.0, 1.5, 2.5], "alpha": [0.3, 0.5, 0.7]})

    b2 = dict(OLD); b2.update(B=4, C=1.5, alpha=0.7)
    table("A2. G1 探索 c_pucb×c_leaf（base + B=4,C=1.5,alpha=0.7）", b2,
          {"c_pucb": [0.1, 0.2, 0.4, 0.8, 1.5], "c_leaf": [0.1, 0.2, 0.4, 0.8, 1.5]})

    b2b = dict(b2); b2b.update(c_pucb=0.8, c_leaf=0.2)
    table("A3. tau（base + c_pucb=0.8,c_leaf=0.2）", b2b, {"tau": [0.1, 0.3, 0.6]})

    b3 = dict(b2b); b3.update(tau=0.3)
    table("A4. G3 gamma（base + tau=0.3）", b3, {"gamma": [0.3, 0.6, 0.8, 0.95]})

    b4 = dict(b3); b4.update(gamma=0.6)
    table("A5. G4 n_seed×S（base + gamma=0.6）", b4,
          {"n_seed": [2, 3, 5, 8], "S": [5, 10, 20, 40]})

    b5 = dict(b4); b5.update(n_seed=5, S=5)
    table("A6. 回扫 G1（base = 全部其它赢家）", b5,
          {"c_pucb": [0.1, 0.2, 0.4, 0.8, 1.5], "c_leaf": [0.1, 0.2, 0.4, 0.8, 1.5]})

    # A7. S detail at the final-ish base (5 seeds), raw AUC per toy
    print("\n### A7. S 在锁定 base 细扫（n_seed=5,B=4,alpha=0.7,c_pucb=0.8；5 seed；原始 AUC）")
    print("| S | base | manydir | rugged | 3toy均值 |")
    print("|---|---|---|---|---|")
    sbase = dict(DEFAULT_CFG)
    for S in [5, 10, 20, 40]:
        cfg = dict(sbase); cfg["S"] = S
        pt = bench.run_cfg(cfg, None, tuple(range(5)))
        aucs = [pt[t]["auc"] for t in pt]
        print(f"| {S} | {pt['base']['auc']:.4f} | {pt['manydir']['auc']:.4f} | "
              f"{pt['rugged']['auc']:.4f} | {mean(aucs):.4f} |")

    # A8. best-of-K (inner)
    print("\n### A8. 内层 K：best-of-K（plan_hpo warm 数据，8 候选）")
    cbase = ROOT / "dev_plan" / "plan_hpo" / "data" / "candidates"
    rows = []
    if cbase.is_dir():
        print("| 候选 | K=1 | K=2 | K=3 | K=4 | K=5 |")
        print("|---|---|---|---|---|---|")
        for c in sorted(os.listdir(cbase)):
            p = cbase / c / "tune_report.json"
            if not p.exists():
                continue
            sc = [w.get("score") for w in json.loads(p.read_text()).get("phase_a", {}).get("warm_start_configs", [])
                  if isinstance(w.get("score"), (int, float))]
            if len(sc) < 5:
                continue
            bok = [min(sc[:k]) for k in range(1, 6)]
            rows.append(bok)
            print(f"| {c} | " + " | ".join(f"{b:.4f}" for b in bok) + " |")
        if rows:
            mb = [mean(r[k] for r in rows) for k in range(5)]
            print(f"| **均值** | " + " | ".join(f"**{mb[k]:.4f}**" for k in range(5)) + " |")
            print(f"\n_相对 K=5 还差：" + "，".join(f"K{k+1}={mb[k]-mb[4]:+.4f}" for k in range(4))
                  + " → best-of-K 在 K=3 即 plateau。_")

    # A9. DEFAULT vs TUNED (5 seeds), raw per toy
    print("\n### A9. 旧默认 vs 调后默认（5 seed，原始 per-toy）")
    print("| cfg | toy | best@20 | best@40 | final | auc |")
    print("|---|---|---|---|---|---|")
    for name, cfg in [("旧默认", dict(OLD)), ("调后", dict(DEFAULT_CFG))]:
        pt = bench.run_cfg(cfg, None, tuple(range(5)))
        for t, m in pt.items():
            print(f"| {name} | {t} | {m['best@20']:.4f} | {m['best@40']:.4f} | "
                  f"{m['final']:.4f} | {m['auc']:.4f} |")


if __name__ == "__main__":
    main()
