#!/usr/bin/env python3
"""Summarize one autoresearch run and render its idea graph.

Reads `runs/<task>/<tag>/ledger.json` and produces, into the run dir (or
`--out-dir`):

- `summary.json` + a printed text report: counts by op / status, how many ideas
  were generated, how many were deep-tuned (HPO), which deep-tunes improved, the
  best candidate, the total `evaluate_config` budget consumed (Σ trials), op
  success rates, and a per-candidate table.
- `idea_graph.dot` — a Graphviz DOT of the development DAG (lineage edges; nodes
  styled by status; HPO'd nodes double-circled; the best node gold). Always
  written (pure stdlib).
- `idea_graph.png` — a chronological idea-search plot (final_best_score vs run_id,
  lineage edges, running-best frontier, HPO'd points ringed, best starred).
  Rendered only if matplotlib is importable; otherwise skipped with a note.

The summary is pure stdlib (no env needed). The PNG needs matplotlib + numpy, so
run via the task env for it: `uv --directory tasks/<task> run python
tools/summarize_run.py --run-dir runs/<task>/<tag>`.

Usage:
    python tools/summarize_run.py --run-dir runs/<task>/<tag> [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


CRASH_SENTINEL = float("inf")


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def load_records(run_dir: Path) -> tuple[dict, list[dict]]:
    ledger = run_dir / "ledger.json"
    if not ledger.is_file():
        sys.exit(f"no ledger.json in {run_dir}")
    data = json.loads(ledger.read_text())
    return data, data.get("records", [])


def _numeric_parents(rec: dict) -> list[str]:
    """Return numeric DAG parents (P1 source_run_ids contains only these)."""
    return [s for s in (rec.get("source_run_ids") or []) if str(s).isdigit()]


def _improved(rec: dict) -> Optional[bool]:
    """For a deep-tuned record: did Phase C beat the frozen warm best?"""
    if not rec.get("tune"):
        return None
    bw, fb = rec.get("best_warm_score"), rec.get("final_best_score")
    if _is_num(bw) and _is_num(fb):
        return fb < bw - 1e-12
    return None


def summarize(data: dict, records: list[dict], threshold: Optional[float] = None) -> dict:
    scored = [r for r in records if _is_num(r.get("final_best_score")) and r.get("status") != "crash"]
    tuned = [r for r in records if r.get("tune")]
    crashes = [r for r in records if r.get("status") == "crash"]
    best = min(scored, key=lambda r: r["final_best_score"]) if scored else None

    # Eval budget counts every score_fn call, including failures. Keep the
    # finite-score total separately so the warm/Phase-C breakdown remains an
    # observation breakdown instead of misclassifying failed warm retries.
    def attempted_evals(r) -> int:
        t = r.get("trials_attempted")
        if not _is_num(t):
            t = r.get("trials_completed")
        if _is_num(t):
            return int(t)
        k = r.get("warm_start_K")
        return int(k) if _is_num(k) else 0

    def completed_evals(r) -> int:
        t = r.get("trials_completed")
        if _is_num(t):
            return int(t)
        k = r.get("warm_start_K")
        return int(k) if _is_num(k) else 0

    total_evals = sum(attempted_evals(r) for r in records)
    total_completed = sum(completed_evals(r) for r in records)
    warm_evals = sum(int(r["warm_start_K"]) for r in records if _is_num(r.get("warm_start_K")))
    phase_c_evals = total_completed - warm_evals
    failed_evals = total_evals - total_completed

    op_counts = Counter(r.get("op") for r in records)
    status_counts = Counter(r.get("status") for r in records)
    method_counts = Counter(r.get("phase_c_method") for r in tuned if r.get("phase_c_method"))

    # which op produced keeps (was an improvement-at-its-time)
    op_keep = Counter(r.get("op") for r in records if r.get("status") == "keep")

    tuned_detail = []
    improved = 0
    for r in tuned:
        imp = _improved(r)
        improved += 1 if imp else 0
        tuned_detail.append({
            "run_id": r.get("run_id"), "candidate_name": r.get("candidate_name"),
            "method": r.get("phase_c_method"), "n_dims": r.get("n_dims"),
            "best_warm_score": r.get("best_warm_score"),
            "final_best_score": r.get("final_best_score"), "improved": imp,
        })

    # efficiency curve: running-best final_best_score after the first K candidates
    # (run_id order) — the cross-config metric for Phase-3 OFAT (search reaches a
    # noisy-task plateau, so "best at a small budget" discriminates, "best at 200"
    # does not). candidates_to_best = first candidate index reaching the run's best.
    ordered = sorted(scored, key=lambda r: int(r["run_id"]))

    def best_after(k: int):
        vals = [r["final_best_score"] for r in ordered[:k]]
        return min(vals) if vals else None

    eff = {f"best_after_{k}_candidates": best_after(k) for k in (5, 10, 15, 20, 25)}
    cands_to_best = None
    if best is not None:
        bsf = float("inf")
        for i, r in enumerate(ordered, 1):
            bsf = min(bsf, r["final_best_score"])
            if best is not None and abs(bsf - best["final_best_score"]) < 1e-12:
                cands_to_best = i
                break

    cands_to_thr = None
    if threshold is not None:
        bsf = float("inf")
        for i, r in enumerate(ordered, 1):
            bsf = min(bsf, r["final_best_score"])
            if bsf <= threshold:
                cands_to_thr = i
                break

    return {
        "task": data.get("task"), "tag": data.get("tag"), "metric": data.get("metric"),
        "n_ideas": len(records),
        "efficiency": eff,
        "candidates_to_best": cands_to_best,
        "threshold": threshold,
        "candidates_to_threshold": cands_to_thr,
        "op_counts": dict(op_counts),
        "status_counts": dict(status_counts),
        "n_crashes": len(crashes),
        "n_hpo": len(tuned),
        "hpo_methods": dict(method_counts),
        "n_hpo_improved": improved,
        "op_keep_counts": dict(op_keep),
        "eval_budget_total": total_evals,
        "eval_budget_completed": total_completed,
        "eval_budget_failed": failed_evals,
        "eval_budget_warm": warm_evals,
        "eval_budget_phase_c": phase_c_evals,
        "best": None if best is None else {
            "run_id": best.get("run_id"), "candidate_name": best.get("candidate_name"),
            "final_best_score": best.get("final_best_score"),
            "op": best.get("op"), "tuned": bool(best.get("tune")),
            "point_id": (best.get("semantic_point") or {}).get("point_id"),
            "semantic_policy": ((best.get("policy_receipt") or {}).get("policy") or {}).get("name"),
        },
        "tuned_detail": tuned_detail,
        "records": [{
            "run_id": r.get("run_id"), "op": r.get("op"),
            "source_run_ids": r.get("source_run_ids"),
            "point_id": (r.get("semantic_point") or {}).get("point_id"),
            "semantic_policy": ((r.get("policy_receipt") or {}).get("policy") or {}).get("name"),
            "candidate_name": r.get("candidate_name"),
            "final_best_score": r.get("final_best_score"),
            "status": r.get("status"), "tune": bool(r.get("tune")),
            "n_dims": r.get("n_dims"), "phase_c_method": r.get("phase_c_method"),
        } for r in records],
    }


def print_report(s: dict) -> None:
    p = print
    p(f"\n=== autoresearch run summary: {s['task']} / {s['tag']} ===")
    p(f"metric: {s['metric']} (lower is better)")
    p(f"\nideas generated:       {s['n_ideas']}")
    p(f"  by op:               {s['op_counts']}")
    p(f"  by status:           {s['status_counts']}")
    p(f"  crashes:             {s['n_crashes']}")
    p(f"  keeps by op:         {s['op_keep_counts']}")
    p(f"\ndeep-tuning (HPO):     {s['n_hpo']} candidates tuned")
    p(f"  methods:             {s['hpo_methods']}")
    p(f"  improved over warm:  {s['n_hpo_improved']} / {s['n_hpo']}")
    p(f"\nevaluate_config budget: {s['eval_budget_total']} total"
      f"  (warm {s['eval_budget_warm']} + Phase-C {s['eval_budget_phase_c']})")
    if s["best"]:
        b = s["best"]
        p(f"\nbest candidate:        {b['run_id']} {b['candidate_name']}"
          f"  score={b['final_best_score']:.4f}  op={b['op']}  tuned={b['tuned']}")
    p(f"\nefficiency (running-best after first K candidates):")
    for k in (5, 10, 15, 20, 25):
        v = s["efficiency"].get(f"best_after_{k}_candidates")
        p(f"  after {k:>2} candidates: {v:.4f}" if _is_num(v) else f"  after {k:>2} candidates: —")
    p(f"  candidates to reach run best: {s['candidates_to_best']}")
    p("\nHPO'd candidates:")
    for t in s["tuned_detail"]:
        bw = f"{t['best_warm_score']:.4f}" if _is_num(t["best_warm_score"]) else "—"
        fb = f"{t['final_best_score']:.4f}" if _is_num(t["final_best_score"]) else "—"
        p(f"  {t['run_id']} {str(t['candidate_name'])[:34]:34s} {t['method']:5s}"
          f" {t['n_dims']:>3}d  warm={bw} final={fb} improved={t['improved']}")
    p("\nper-candidate:")
    p(f"  {'id':>3} {'op':>9} {'parents':>10} {'score':>9} {'stat':>7} {'hpo':>4}  name")
    for r in s["records"]:
        sc = f"{r['final_best_score']:.4f}" if _is_num(r["final_best_score"]) else "—"
        par = ",".join(_numeric_parents(r)) or "·"
        p(f"  {str(r['run_id']):>3} {str(r['op']):>9} {par:>10} {sc:>9}"
          f" {str(r['status']):>7} {'★' if r['tune'] else ' ':>4}  {r['candidate_name']}")
    p("")


_STATUS_FILL = {"keep": "#bfe3b6", "discard": "#e8e8e8", "crash": "#f3b6b6", "pending": "#fff2b6"}


def write_dot(records: list[dict], path: Path) -> None:
    lines = ["digraph idea_dag {", '  rankdir=LR;', '  node [style="filled,setlinewidth(1)", fontname="Helvetica", fontsize=10];']
    scored = [r["final_best_score"] for r in records if _is_num(r.get("final_best_score")) and r.get("status") != "crash"]
    best_v = min(scored) if scored else None
    for r in records:
        rid = r.get("run_id")
        fb = r.get("final_best_score")
        sc = f"{fb:.4f}" if _is_num(fb) and r.get("status") != "crash" else "crash"
        fill = _STATUS_FILL.get(r.get("status"), "#ffffff")
        is_best = _is_num(fb) and best_v is not None and abs(fb - best_v) < 1e-12 and r.get("status") != "crash"
        attrs = [f'fillcolor="{fill}"', f'label="{rid}\\n{r.get("candidate_name","")}\\n{sc}"']
        if r.get("tune"):
            attrs.append('peripheries=2')          # double circle = HPO'd
        if is_best:
            attrs.append('color="#d4a017"')         # gold border = best
            attrs.append('penwidth=3')
        if not _numeric_parents(r):
            attrs.append('shape=box')               # fresh roots are boxes
        lines.append(f'  "{rid}" [{", ".join(attrs)}];')
    for r in records:
        for p in _numeric_parents(r):
            lines.append(f'  "{p}" -> "{r.get("run_id")}";')
    lines.append("}")
    path.write_text("\n".join(lines))


def render_png(records: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return False

    def xy(r):
        return int(r["run_id"]), r.get("final_best_score")

    fig, ax = plt.subplots(figsize=(13, 7))
    pos = {}
    scored = [r for r in records if _is_num(r.get("final_best_score")) and r.get("status") != "crash"]
    ymin = min(r["final_best_score"] for r in scored) if scored else -1.0
    ymax = max(r["final_best_score"] for r in scored) if scored else 0.0
    crash_y = ymax + 0.02 * (ymax - ymin + 1e-9) + 0.005
    for r in records:
        x = int(r["run_id"])
        y = r.get("final_best_score")
        y = y if (_is_num(y) and r.get("status") != "crash") else crash_y
        pos[r["run_id"]] = (x, y)

    # lineage edges (parent -> child)
    for r in records:
        cx, cy = pos[r["run_id"]]
        for p in _numeric_parents(r):
            if p in pos:
                px, py = pos[p]
                ax.annotate("", xy=(cx, cy), xytext=(px, py),
                            arrowprops=dict(arrowstyle="-|>", color="#b9b9b9", lw=0.8, alpha=0.7))

    # running-best frontier over run_id order
    best_so_far, fx, fy = float("inf"), [], []
    for r in sorted(records, key=lambda r: int(r["run_id"])):
        v = r.get("final_best_score")
        if _is_num(v) and r.get("status") != "crash" and v < best_so_far:
            best_so_far = v
        if best_so_far != float("inf"):
            fx.append(int(r["run_id"]))
            fy.append(best_so_far)
    if fx:
        ax.step(fx, fy, where="post", color="#d4a017", lw=1.6, alpha=0.9, label="running best", zorder=1)

    colors = {"keep": "#3a9d23", "discard": "#9a9a9a", "crash": "#cc3333", "pending": "#caa800"}
    best_v = min((r["final_best_score"] for r in scored), default=None)
    for r in records:
        x, y = pos[r["run_id"]]
        st = r.get("status")
        ax.scatter([x], [y], s=130, c=colors.get(st, "#444"),
                   marker="X" if st == "crash" else ("s" if not _numeric_parents(r) else "o"),
                   edgecolors="black", linewidths=0.6, zorder=3)
        if r.get("tune"):  # HPO'd: ring
            ax.scatter([x], [y], s=320, facecolors="none", edgecolors="#1f4ed8", linewidths=2.0, zorder=2)
        if best_v is not None and _is_num(r.get("final_best_score")) and abs(r["final_best_score"] - best_v) < 1e-12 and st != "crash":
            ax.scatter([x], [y], s=420, marker="*", facecolors="#ffd400", edgecolors="black", linewidths=0.8, zorder=4)
        ax.annotate(str(r["run_id"]), (x, y), fontsize=7, ha="center", va="center", zorder=5)

    ax.set_xlabel("run_id (chronological)")
    ax.set_ylabel("final_best_score (lower = better)")
    ax.set_title("Idea search graph — lineage edges, running-best, HPO'd nodes ringed (blue), best starred")
    ax.grid(True, alpha=0.25)
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3a9d23", markeredgecolor="k", markersize=10, label="keep"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9a9a9a", markeredgecolor="k", markersize=10, label="discard"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#cc3333", markeredgecolor="k", markersize=10, label="crash"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#9a9a9a", markeredgecolor="k", markersize=10, label="fresh (root)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#1f4ed8", markersize=14, markeredgewidth=2, label="HPO'd (deep-tuned)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffd400", markeredgecolor="k", markersize=16, label="best"),
        Line2D([0], [0], color="#d4a017", lw=1.6, label="running best"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path, help="runs/<task>/<tag>")
    ap.add_argument("--out-dir", type=Path, default=None, help="defaults to --run-dir")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data, records = load_records(run_dir)
    if not records:
        sys.exit("ledger has no records")

    s = summarize(data, records)
    print_report(s)
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2))

    dot_path = out_dir / "idea_graph.dot"
    write_dot(records, dot_path)

    png_path = out_dir / "idea_graph.png"
    png_ok = render_png(records, png_path)

    print(f"wrote: {out_dir/'summary.json'}")
    print(f"wrote: {dot_path}  (render: dot -Tpng idea_graph.dot -o idea_graph.png)")
    print(f"wrote: {png_path}" if png_ok else "PNG skipped (matplotlib unavailable — run via the task uv env, or use the .dot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
