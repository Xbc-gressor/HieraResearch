#!/usr/bin/env python3
"""Rebase a candidate's tuning state after a kept rewrite.

A kept rewrite changes train.py, so every Phase-C observation in
tune_report.json describes code that no longer exists. This tool archives
the old report under ``_rewrite/`` and writes a fresh Phase-A-only report
whose single warm row is the rewritten code at its current BASE_PARAMS with
the rewrite's reference score. The candidate's next tuning bout is therefore
a FIRST/INITIAL bout warm-started from those parameters, and the report
again binds to the candidate on disk (execution revision, SEARCH_SPACE,
BASE_PARAMS == incumbent), which is what the tuner's admission checks read.

Usage:
    python tools/rewrite_rebase.py --candidate <dir> --bout <N> --score REF \
        [--attempts K]

stdout: {"status": "ok", "archived": <path>, "report": <path>}. Any
validation failure exits non-zero with the reason; the candidate's code is
never touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS / "tuners"))
sys.path.insert(0, str(TOOLS))

from _common import write_tune_report  # noqa: E402
from tune_tools import (  # noqa: E402
    _candidate_execution_revision,
    _read_literal_mapping,
    _read_search_space,
    validate_phase_a_candidate_state,
)

REWRITE_DIR = "_rewrite"


def rebase(candidate_dir: Path, *, bout: int, score: float, attempts: int) -> dict:
    candidate_dir = Path(candidate_dir)
    train_py = candidate_dir / "train.py"
    report_path = candidate_dir / "tune_report.json"
    if not report_path.is_file():
        raise ValueError(f"{report_path} does not exist")
    old = json.loads(report_path.read_text(encoding="utf-8"))
    old_phase_a = old.get("phase_a") if isinstance(old.get("phase_a"), dict) else {}

    params = _read_literal_mapping(train_py, "BASE_PARAMS")
    search_space = json.loads(json.dumps(_read_search_space(train_py)))
    revision = _candidate_execution_revision(train_py)
    row = {
        "params": params,
        "score": float(score),
        "status": "ok",
        "role": "rewrite_incumbent",
        "rewrite_bout": int(bout),
    }
    phase_a = {
        "warm_start_configs": [row],
        "warm_config_selection": old_phase_a.get("warm_config_selection"),
        "deferred_configs": [],
        "search_space": search_space,
        "trials_attempted": max(1, int(attempts)),
        "status": "ok",
        "candidate_code_revision": revision,
        "warm_score_cache": {
            "schema_version": 1,
            "candidate_execution_revision": revision,
            "rows": [],
        },
        "best_warm_score": float(score),
        "best_warm_params": params,
        "k_evaluated": 1,
        "k_survived": 1,
        "k_deferred": 0,
        "elapsed_seconds": 0.0,
        "rebased_from_rewrite_bout": int(bout),
    }
    report = {
        key: value
        for key, value in old.items()
        if key not in {
            "phase_a", "phase_c", "preflight", "final_best_params",
            "final_best_score", "applied_to_base_params",
            "last_finalized_stage_index",
        }
    }
    report["phase_a"] = phase_a
    report["phase_c"] = {"stages": []}
    # Fail before touching anything: the new report must bind to the code.
    validate_phase_a_candidate_state(report, train_py)

    archive_dir = candidate_dir / REWRITE_DIR
    archive_dir.mkdir(exist_ok=True)
    archived = archive_dir / f"tune_report.bout-{int(bout):03d}.json"
    shutil.copy2(report_path, archived)
    write_tune_report(report_path, report)
    return {"status": "ok", "archived": str(archived), "report": str(report_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--bout", required=True, type=int)
    parser.add_argument("--score", required=True, type=float,
                        help="the rewrite's reference score (post-confirmation)")
    parser.add_argument("--attempts", type=int, default=1,
                        help="objective attempts behind the reference score")
    args = parser.parse_args()
    try:
        result = rebase(args.candidate, bout=args.bout, score=args.score,
                        attempts=args.attempts)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
