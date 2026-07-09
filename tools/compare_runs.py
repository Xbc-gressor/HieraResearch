#!/usr/bin/env python3
"""Compare several autoresearch runs side by side — the Phase-3 OFAT harness.

Each run dir holds a `ledger.json` and (for an OFAT trial) a `framework_cfg.json`
naming the framework-hyperparameter override under test. This tool reads each
run, labels it by its `framework_cfg.json` (or tag), and tabulates the
**efficiency** metrics that discriminate framework configs on a plateauing task:
running-best after the first K candidates, candidates-to-reach-best, final best,
ideas, and #HPO. Writes `compare.json` + `compare.csv` and prints a table.

Reuses `summarize_run.summarize` so the metric definitions stay in one place.

Usage:
    python tools/compare_runs.py runs/<task>/<tagA> runs/<task>/<tagB> ... [--out-dir DIR]
    python tools/compare_runs.py --glob 'runs/tabular-model-search/p3-*'
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_run import load_records, summarize  # noqa: E402


def _cfg_label(run_dir: Path) -> str:
    """One-line label of the run's framework_cfg.json override (or 'default')."""
    p = run_dir / "framework_cfg.json"
    if not p.is_file():
        return "default"
    try:
        cfg = json.loads(p.read_text())
    except (ValueError, OSError):
        return "?"
    parts = []
    for section in ("got", "tuner"):
        for k, v in (cfg.get(section) or {}).items():
            parts.append(f"{k}={v}")
    return ",".join(parts) or "default"


KS = (5, 10, 15, 20, 25)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", type=Path, help="run dirs (runs/<task>/<tag>)")
    ap.add_argument("--glob", default=None, help="glob for run dirs, e.g. 'runs/<task>/p3-*'")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    dirs = list(args.run_dirs)
    if args.glob:
        dirs += [Path(p) for p in sorted(_glob.glob(args.glob))]
    dirs = [d for d in dirs if (d / "ledger.json").is_file()]
    if not dirs:
        sys.exit("no run dirs with a ledger.json given")

    rows = []
    for d in dirs:
        data, records = load_records(d)
        s = summarize(data, records)
        eff = s["efficiency"]
        rows.append({
            "tag": data.get("tag") or d.name,
            "config": _cfg_label(d),
            "n_ideas": s["n_ideas"], "n_hpo": s["n_hpo"], "n_crashes": s["n_crashes"],
            "best": (s["best"] or {}).get("final_best_score"),
            "candidates_to_best": s["candidates_to_best"],
            **{f"best@{k}": eff.get(f"best_after_{k}_candidates") for k in KS},
        })

    # print table
    def f(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    cols = ["tag", "config", "n_ideas", "n_hpo", "candidates_to_best", "best"] + [f"best@{k}" for k in KS]
    widths = {c: max(len(c), *(len(f(r.get(c))) for r in rows)) for c in cols}
    print("\n=== OFAT run comparison (efficiency: lower best@K = reached a better score with fewer candidates) ===")
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(f(r.get(c)).ljust(widths[c]) for c in cols))
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "compare.json").write_text(json.dumps(rows, indent=2))
    with open(args.out_dir / "compare.csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c)) for c in cols) + "\n")
    print(f"wrote: {args.out_dir/'compare.json'} , {args.out_dir/'compare.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
