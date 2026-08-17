"""Cross-cell aggregation for the inner-tuner benchmark (PLAN §九).

Discovers per-cell artifacts (``manifest.json`` + ``result.json`` pairs,
as written by cell.py / runner.run_cell), attaches each cell's checkpoint
stratum from the frozen checkpoints, and builds the comparison report:

- stratification: ``first`` / ``cont_improved`` / ``cont_not_improved``
  reported separately (§九: never assume an arm must win both regimes);
- the per-checkpoint unit first: per (stratum, arm, checkpoint) medians
  over seeds, THEN across-checkpoint median/mean/min/max — different
  candidates' raw difficulties are never averaged together (their
  relative improvements only ever meet as per-checkpoint summaries);
- replicate spread: median across checkpoints of the across-seed range;
- paired comparison: per (checkpoint, seed) metric delta vs the baseline arm
  (``--baseline-arm``, default ``current``; only pairs where both cells
  exist);
- attrition: status counts per arm (``ok`` / ``arm_error`` /
  ``unsupported``). Metric summaries skip cells whose metrics are absent
  (zero-evaluation cells carry ``auc: null``); the counts make the
  attrition visible instead of silently padding or dropping.

Checkpoint matching is by ``checkpoint_id`` verified against the
manifest's ``checkpoint_hash`` (sha256 of checkpoint.json) — a mismatch
flags the cell as misconfigured (§八 排除误配) rather than letting a
stale checkpoint silently relabel a cell's stratum.

Usage:
    uv run python tools/inner_benchmark/aggregate.py \
        --cells <cells-root> --checkpoints <ckpts-root-machineA> [<ckpts-root-machineB> ...] \
        [--out report.json] [--md report.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import artifacts  # noqa: E402
import checkpoint as checkpoint_mod  # noqa: E402

BASELINE_ARM = "current"  # default; --baseline-arm overrides

# Trajectory metrics summarized everywhere; "@k" keys come from
# result.json's relative_improvement_at (stringified int keys).
TRAJECTORY_METRICS = ("auc", "final_relative_improvement")
AT_KEYS = ("2", "4", "6", "8", "10", "24")
COST_KEYS = ("llm_calls", "llm_input_tokens", "llm_output_tokens")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_cells(cells_root) -> list[dict]:
    """Every dir under cells_root holding both manifest.json and result.json
    (any depth — the sweep's own directory layout is the operator's)."""
    cells = []
    for result_path in sorted(Path(cells_root).rglob(artifacts.RESULT_FILENAME)):
        cell_dir = result_path.parent
        manifest_path = cell_dir / artifacts.MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue  # partial cell dir (interrupted before manifest?)
        cells.append(
            {
                "cell_dir": str(cell_dir),
                "manifest": json.loads(manifest_path.read_text()),
                "result": json.loads(result_path.read_text()),
            }
        )
    return cells


def attach_checkpoint_info(cells: list[dict], checkpoints_roots) -> None:
    """Attach regime/stratum from the frozen checkpoint, hash-verified.

    ``checkpoints_roots`` is a list of roots (one per execution machine's
    remeasured copy). Adds ``regime`` / ``stratum`` / ``checkpoint_ok`` to
    each cell dict; cells whose checkpoint is missing or hash-mismatched
    keep ``checkpoint_ok=False`` and stratum "unknown" (they still appear
    in the report's per-cell listing, never in a stratum they don't own).
    """
    # Multiple roots: every execution machine remeasures its OWN checkpoint
    # copy (PLAN §七/§八 same-machine scores), so the same checkpoint_id
    # legitimately carries a different hash per machine. A cell verifies when
    # its manifest hash matches ANY root's copy.
    index: dict[str, list[Path]] = {}
    for checkpoints_root in checkpoints_roots:
        for path in sorted(Path(checkpoints_root).rglob(checkpoint_mod.CHECKPOINT_FILENAME)):
            data = json.loads(path.read_text())
            checkpoint_id = data.get("checkpoint_id")
            if checkpoint_id:
                index.setdefault(checkpoint_id, []).append(path.parent)
    for cell in cells:
        manifest = cell["manifest"]
        cell["checkpoint_ok"] = False
        cell["regime"] = "unknown"
        cell["stratum"] = "unknown"
        for checkpoint_dir in index.get(manifest.get("checkpoint_id"), []):
            digest = hashlib.sha256(
                (checkpoint_dir / checkpoint_mod.CHECKPOINT_FILENAME).read_bytes()
            ).hexdigest()
            if digest != manifest.get("checkpoint_hash"):
                continue  # not the copy this cell ran against
            frozen = checkpoint_mod.load_checkpoint(checkpoint_dir)
            cell["checkpoint_ok"] = True
            cell["regime"] = frozen.regime
            cell["stratum"] = frozen.stratum
            break


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _cell_metrics(cell: dict) -> dict | None:
    """The flat metric view of one cell, or None when the cell produced no
    evaluations (zero-eval arm_error / unsupported: nothing to summarize)."""
    result = cell["result"]
    if result.get("auc") is None:
        return None
    at = result.get("relative_improvement_at") or {}
    metrics = {
        "auc": result["auc"],
        "final_relative_improvement": result.get("final_relative_improvement"),
    }
    for key in AT_KEYS:
        metrics[f"at_{key}"] = at.get(key)
    metrics["beat"] = 1.0 if result.get("beat_initial_incumbent") else 0.0
    return metrics


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(cells: list[dict], *, baseline_arm: str = BASELINE_ARM) -> dict:
    """Build the §九 report over cells carrying regime/stratum/strand.

    Paired deltas are computed against ``baseline_arm`` and reported under
    ``paired_delta_vs_<baseline_arm>``."""
    metric_names = list(TRAJECTORY_METRICS) + [f"at_{k}" for k in AT_KEYS] + ["beat"]
    paired_key = f"paired_delta_vs_{baseline_arm}"

    # per-cell rows (the report's audit surface)
    rows = []
    for cell in cells:
        result, manifest = cell["result"], cell["manifest"]
        rows.append(
            {
                "cell_dir": cell["cell_dir"],
                "checkpoint_id": manifest.get("checkpoint_id"),
                "checkpoint_ok": cell.get("checkpoint_ok", False),
                "stratum": cell.get("stratum", "unknown"),
                "arm": manifest.get("arm"),
                "seed": manifest.get("seed"),
                "status": result.get("status"),
                "evaluations": result.get("evaluations"),
                "auc": result.get("auc"),
                "final_best_score": result.get("final_best_score"),
                "final_relative_improvement": result.get(
                    "final_relative_improvement"
                ),
            }
        )

    # group: stratum -> arm -> cells
    groups: dict = {}
    for cell in cells:
        manifest = cell["manifest"]
        arm_bucket = groups.setdefault(cell.get("stratum", "unknown"), {}).setdefault(
            manifest.get("arm"), {"cells": [], "statuses": {}}
        )
        arm_bucket["cells"].append(cell)
        status = cell["result"].get("status")
        arm_bucket["statuses"][status] = arm_bucket["statuses"].get(status, 0) + 1

    # Stratum supply: distinct hash-verified checkpoints. PLAN §七: a stratum
    # with fewer than 3 checkpoints reports per-checkpoint results ONLY — no
    # cross-checkpoint statistics and no arm ranking.
    supply = {
        stratum: len(
            {
                cell["manifest"].get("checkpoint_id")
                for bucket in arms.values()
                for cell in bucket["cells"]
                if cell.get("checkpoint_ok")
            }
        )
        for stratum, arms in groups.items()
    }

    by_stratum: dict = {}
    for stratum, arms in sorted(groups.items()):
        eligible = supply[stratum] >= 3
        # Baseline pair index for this stratum: (checkpoint, seed) -> metrics.
        baseline_by_pair = {}
        for cell in arms.get(baseline_arm, {}).get("cells", []):
            metrics = _cell_metrics(cell)
            if metrics is None:
                continue
            key = (
                cell["manifest"].get("checkpoint_id"),
                cell["manifest"].get("seed"),
            )
            baseline_by_pair[key] = metrics

        stratum_report: dict = {}
        for arm, bucket in sorted(arms.items()):
            per_checkpoint: dict = {}
            for cell in bucket["cells"]:
                metrics = _cell_metrics(cell)
                if metrics is None:
                    continue
                ckpt = cell["manifest"].get("checkpoint_id")
                per_checkpoint.setdefault(ckpt, []).append((cell, metrics))

            checkpoint_summaries: dict = {}
            for ckpt, entries in sorted(per_checkpoint.items()):
                metric_lists = {
                    name: [m[name] for _, m in entries if m[name] is not None]
                    for name in metric_names
                }
                summary_block = {
                    "seeds": len(entries),
                    **{
                        name: statistics.median(values)
                        for name, values in metric_lists.items()
                        if values
                    },
                }
                # Raw scores are meaningful within one checkpoint only.  Keep
                # E2's final-best primary result on the per-cell and
                # per-checkpoint audit surfaces, but never pool it across
                # candidates or compute direction-ambiguous paired deltas.
                final_best_scores = [
                    cell["result"].get("final_best_score")
                    for cell, _ in entries
                    if isinstance(cell["result"].get("final_best_score"), (int, float))
                ]
                if final_best_scores:
                    summary_block["final_best_score"] = statistics.median(
                        final_best_scores
                    )
                # Within-checkpoint paired comparison (same seed): always
                # allowed, in every stratum (§九).
                if arm != baseline_arm:
                    deltas: dict = {name: [] for name in metric_names}
                    for cell, metrics in entries:
                        other = baseline_by_pair.get(
                            (ckpt, cell["manifest"].get("seed"))
                        )
                        if other is None:
                            continue
                        for name in metric_names:
                            if metrics[name] is not None and other[name] is not None:
                                deltas[name].append(metrics[name] - other[name])
                    summary_block[paired_key] = {
                        name: _summary(values)
                        for name, values in deltas.items()
                        if values
                    }
                checkpoint_summaries[ckpt] = summary_block

            across = None
            replicate_spread = None
            paired = None
            if eligible:
                across = {}
                for name in metric_names:
                    values = [
                        summary[name]
                        for summary in checkpoint_summaries.values()
                        if name in summary
                    ]
                    if values:
                        across[name] = _summary(values)
                replicate_spread = {}
                for name in metric_names:
                    spreads = [
                        _spread([m[name] for _, m in entries if m[name] is not None])
                        for entries in per_checkpoint.values()
                        if len(entries) > 1
                    ]
                    if spreads:
                        replicate_spread[name] = statistics.median(spreads)
                if arm != baseline_arm:
                    # Cross-checkpoint pooled paired comparison — only for an
                    # eligible stratum (PLAN §七 supply rule).
                    deltas = {name: [] for name in metric_names}
                    for cell in bucket["cells"]:
                        metrics = _cell_metrics(cell)
                        if metrics is None:
                            continue
                        other = baseline_by_pair.get(
                            (
                                cell["manifest"].get("checkpoint_id"),
                                cell["manifest"].get("seed"),
                            )
                        )
                        if other is None:
                            continue
                        for name in metric_names:
                            if metrics[name] is not None and other[name] is not None:
                                deltas[name].append(metrics[name] - other[name])
                    paired = {
                        name: _summary(values)
                        for name, values in deltas.items()
                        if values
                    }

            # costs + operational counters (sums over ALL cells of the arm)
            totals = {}
            for key in COST_KEYS + ("ranker_fallback_count",):
                values = [
                    cell["result"].get(key)
                    for cell in bucket["cells"]
                    if isinstance(cell["result"].get(key), (int, float))
                ]
                if values:
                    totals[key] = sum(values)
            first_improvements = [
                cell["result"]["first_improvement_eval"]
                for cell in bucket["cells"]
                if isinstance(cell["result"].get("first_improvement_eval"), int)
            ]

            stratum_report[arm] = {
                "cells": len(bucket["cells"]),
                "statuses": bucket["statuses"],
                "per_checkpoint": checkpoint_summaries,
                "across_checkpoints": across,
                "replicate_spread": replicate_spread,
                paired_key: paired,
                "first_improvement_eval_median": (
                    statistics.median(first_improvements)
                    if first_improvements
                    else None
                ),
                "totals": totals,
            }
        by_stratum[stratum] = {
            "checkpoint_count": supply[stratum],
            "ranking_eligible": eligible,
            "arms": stratum_report,
        }

    return {"cells": rows, "by_stratum": by_stratum}


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: dict, *, baseline_arm: str = BASELINE_ARM) -> str:
    """One table per stratum. Eligible strata (>= 3 checkpoints) get the arm
    ranking table; under-supplied strata get per-checkpoint rows only
    (PLAN §七). The JSON report remains the full audit surface."""
    paired_key = f"paired_delta_vs_{baseline_arm}"
    lines = ["# Inner-tuner benchmark report", ""]
    for stratum, block in sorted(report["by_stratum"].items()):
        arms = block["arms"]
        if not block["ranking_eligible"]:
            lines.append(
                f"## stratum: {stratum} — per-checkpoint results only "
                f"({block['checkpoint_count']} checkpoints < 3; PLAN §七 "
                "forbids cross-checkpoint arm ranking here)"
            )
            lines.append("")
            lines.append(
                "| arm | checkpoint | seeds | median AUC | median final "
                f"| ΔAUC vs {baseline_arm} (within-checkpoint, median/n) |"
            )
            lines.append("|---|---|---|---|---|---|")
            for arm, bucket in sorted(arms.items()):
                per_ckpt = bucket["per_checkpoint"]
                if not per_ckpt:
                    lines.append(f"| {arm} | — | 0 | — | — | — |")
                    continue
                for ckpt, summary in sorted(per_ckpt.items()):
                    auc = summary.get("auc")
                    final = summary.get("final_relative_improvement")
                    paired = summary.get(paired_key, {}).get("auc")
                    paired_text = (
                        "—"
                        if not paired
                        else f"{paired['median']:+.2f} (n={paired['n']})"
                    )
                    lines.append(
                        f"| {arm} | {ckpt} | {summary['seeds']} | "
                        f"{'—' if auc is None else f'{auc:.2f}'} | "
                        f"{'—' if final is None else f'{final:.2f}'} | "
                        f"{paired_text} |"
                    )
            lines.append("")
            continue
        lines.append(f"## stratum: {stratum}")
        lines.append("")
        lines.append(
            "| arm | cells | statuses | median AUC | mean AUC | median final "
            f"| beat rate | paired ΔAUC vs {baseline_arm} (median/mean, n) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for arm, bucket in sorted(arms.items()):
            across = bucket["across_checkpoints"] or {}
            statuses = ", ".join(
                f"{status}:{count}" for status, count in sorted(bucket["statuses"].items())
            )

            def _fmt(metric, key):
                block_ = across.get(metric)
                if not block_:
                    return "—"
                value = block_.get(key)
                return "—" if value is None else f"{value:.2f}"

            paired = (bucket[paired_key] or {}).get("auc")
            paired_text = "—"
            if paired:
                paired_text = (
                    f"{paired['median']:+.2f} / {paired['mean']:+.2f} "
                    f"(n={paired['n']})"
                )
            lines.append(
                f"| {arm} | {bucket['cells']} | {statuses} | "
                f"{_fmt('auc', 'median')} | {_fmt('auc', 'mean')} | "
                f"{_fmt('final_relative_improvement', 'median')} | "
                f"{_fmt('beat', 'mean')} | {paired_text} |"
            )
        lines.append("")
    flagged = [row for row in report["cells"] if not row["checkpoint_ok"]]
    if flagged:
        lines.append(f"## flagged cells (checkpoint missing/hash mismatch): {len(flagged)}")
        lines.append("")
        for row in flagged:
            lines.append(f"- {row['cell_dir']} (checkpoint {row['checkpoint_id']})")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inner_benchmark.aggregate")
    parser.add_argument("--cells", required=True, help="cells output root")
    parser.add_argument(
        "--checkpoints",
        required=True,
        nargs="+",
        help="frozen checkpoints roots — one per execution machine's "
        "remeasured copy (stratum attachment + hash verify against ANY root)",
    )
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--md", default=None, help="write the markdown summary here")
    parser.add_argument(
        "--baseline-arm",
        default=BASELINE_ARM,
        help="arm the paired deltas are computed against (default: "
        f"{BASELINE_ARM}; E1/E2 arm sets without `current` use their "
        "reference arm, e.g. pool_hebo_mace)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = load_cells(args.cells)
    attach_checkpoint_info(cells, args.checkpoints)
    report = summarize(cells, baseline_arm=args.baseline_arm)
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if args.md:
        Path(args.md).write_text(
            render_markdown(report, baseline_arm=args.baseline_arm), encoding="utf-8"
        )
    if not args.out and not args.md:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
