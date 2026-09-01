#!/usr/bin/env python3
"""Import a candidate from a past experiment run into a fresh run directory.

Scaffolding for the rewrite-operator loop: the imported candidate keeps its
full context (idea/change/semantic point, tuning state, failure artifacts and
evaluation traces) so the rewrite-editor starts from measured history instead
of a bare train.py. Pure file copying plus a manifest — no ledger synthesis:
neither run's ledger.json is ever written by this tool.

Usage:
    python tools/import_candidate.py --source <src_run_dir> --run-id <NNN> \
        --target <dst_run_dir> [--seed-experience]

stdout: {"candidate_dir": "<path>"}.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
from pathlib import Path

from apply_base_params import _find_base_params

ROOT = Path(__file__).resolve().parents[1]

# Copy-only-if-present manifest of a candidate directory; anything else in the
# source candidate (logs, scratch) stays behind. prepare.py is NOT copied from
# the source: it is re-copied from tasks/<task>/prepare.py so the imported
# candidate is bound to the current fixed evaluation surface.
COPY_IF_PRESENT = (
    "train.py",
    "_search_space.json",
    "_warm_configs.json",
    "tune_report.json",
    "_candidate_brief.json",
)
COPY_DIR_IF_PRESENT = ("_failures",)
IMPORT_MANIFEST = "_import.json"


def _load_source_record(source: Path, run_id: str) -> tuple[dict, dict]:
    ledger_path = source / "ledger.json"
    if not ledger_path.is_file():
        raise SystemExit(f"source ledger not found: {ledger_path}")
    ledger = json.loads(ledger_path.read_text())
    for record in ledger.get("records", []):
        if isinstance(record, dict) and record.get("run_id") == run_id:
            return ledger, record
    raise SystemExit(f"no ledger record for candidate {run_id} in {ledger_path}")


def _check_base_params(train_path: Path) -> None:
    """The imported candidate must keep a pure-literal BASE_PARAMS: the rewrite
    loop reads parameters straight out of train.py, and a computed default
    would make the imported baseline score unattributable."""
    try:
        tree = ast.parse(train_path.read_text())
    except SyntaxError as exc:
        raise SystemExit(f"{train_path}: cannot parse: {exc}") from None
    node = _find_base_params(tree)
    if node is None:
        raise SystemExit(f"{train_path}: no module-level BASE_PARAMS assignment")
    if not isinstance(node, ast.Dict):
        raise SystemExit(f"{train_path}: BASE_PARAMS value is not a dict literal")
    try:
        ast.literal_eval(node)
    except (ValueError, SyntaxError):
        raise SystemExit(
            f"{train_path}: BASE_PARAMS is not a pure literal dict; refusing to import"
        ) from None


def _tune_summary(candidate_src: Path) -> dict | None:
    """Compact subset of the source candidate's tune_report.json (phase,
    method and best fields); None when the candidate was never tuned."""
    report_path = candidate_src / "tune_report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict):
        return None
    summary = {}
    for key in ("inner_policy", "final_best_params", "final_best_score"):
        if key in report:
            summary[key] = report[key]
    phase_a = report.get("phase_a")
    if isinstance(phase_a, dict):
        summary["phase_a"] = {
            key: phase_a[key]
            for key in ("status", "best_warm_score", "best_warm_params", "k_evaluated")
            if key in phase_a
        }
    phase_c = report.get("phase_c")
    if isinstance(phase_c, dict) and isinstance(phase_c.get("stages"), list):
        summary["phase_c"] = {
            "stages": [
                {key: stage[key] for key in ("method", "status") if key in stage}
                for stage in phase_c["stages"]
                if isinstance(stage, dict)
            ]
        }
    return summary or None


def import_candidate(
    source: Path,
    run_id: str,
    target: Path,
    *,
    seed_experience: bool = False,
    repo_root: Path = ROOT,
) -> dict:
    source = Path(source)
    target = Path(target)
    ledger, record = _load_source_record(source, run_id)

    task = ledger.get("task")
    target_task = target.parent.name
    if task != target_task:
        raise SystemExit(
            f"task mismatch: source ledger task {task!r}, "
            f"target run sits under {target_task!r}"
        )

    final_best = record.get("final_best_score")
    if not (
        isinstance(final_best, (int, float))
        and not isinstance(final_best, bool)
        and math.isfinite(final_best)
    ):
        raise SystemExit(
            f"candidate {run_id} has no finite final_best_score "
            f"({final_best!r}); the rewrite loop would have no reference "
            "score to adjudicate against — refusing to import"
        )

    candidate_src = source / "candidates" / run_id
    train_src = candidate_src / "train.py"
    if not train_src.is_file():
        raise SystemExit(f"source candidate entrypoint not found: {train_src}")
    _check_base_params(train_src)

    prepare_src = repo_root / "tasks" / task / "prepare.py"
    if not prepare_src.is_file():
        raise SystemExit(f"task prepare.py not found: {prepare_src}")

    candidate_dst = target / "candidates" / run_id
    if candidate_dst.exists():
        raise SystemExit(f"target candidate directory already exists: {candidate_dst}")

    candidate_dst.mkdir(parents=True)
    for name in COPY_IF_PRESENT:
        src_file = candidate_src / name
        if src_file.is_file():
            shutil.copy2(src_file, candidate_dst / name)
    for name in COPY_DIR_IF_PRESENT:
        src_dir = candidate_src / name
        if src_dir.is_dir():
            shutil.copytree(src_dir, candidate_dst / name)
    # The new run numbers its own attempts in _traces/; the source history
    # moves aside so the two never collide.
    traces_src = candidate_src / "_traces"
    if traces_src.is_dir():
        shutil.copytree(traces_src, candidate_dst / "_traces_src")
    shutil.copy2(prepare_src, candidate_dst / "prepare.py")

    best_warm = record.get("best_warm_score")
    delta = None
    if (
        isinstance(best_warm, (int, float))
        and not isinstance(best_warm, bool)
        and math.isfinite(best_warm)
    ):
        # scores are lower-is-better: positive delta means tuning improved
        delta = best_warm - final_best
    manifest = {
        "source": str(source),
        "baseline_score": final_best,
        "warm_to_tuned_delta": delta,
        "idea": record.get("idea"),
        "change": record.get("change"),
        "semantic_point": record.get("semantic_point"),
        "tune_summary": _tune_summary(candidate_src),
    }
    (candidate_dst / IMPORT_MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    # Run-level context is seeded at most once: later imports into the same
    # run must not overwrite what is already there.
    for name in ("background.md", "background_retrieval.json"):
        src_file = source / name
        dst_file = target / name
        if src_file.is_file() and not dst_file.exists():
            shutil.copy2(src_file, dst_file)
    experience = ledger.get("experience")
    if seed_experience and experience is not None:
        seed_path = target / "experience.seed.json"
        if not seed_path.exists():
            seed_path.write_text(json.dumps(experience, indent=2, ensure_ascii=False) + "\n")

    return {"candidate_dir": str(candidate_dst)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="source run directory (read-only)")
    parser.add_argument("--run-id", required=True,
                        help="candidate id in the source run, e.g. 004")
    parser.add_argument("--target", required=True, type=Path,
                        help="target run directory")
    parser.add_argument("--seed-experience", action="store_true",
                        help="seed <target>/experience.seed.json from the source "
                             "ledger's experience field when absent")
    args = parser.parse_args()
    result = import_candidate(
        args.source,
        args.run_id,
        args.target,
        seed_experience=args.seed_experience,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
