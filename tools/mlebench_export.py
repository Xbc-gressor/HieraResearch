#!/usr/bin/env python3
"""Refit the selected experiment incumbent through its task's export hook."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tomllib


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--run-dir", required=True, type=Path)
    a = p.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    run_dir = a.run_dir.resolve()
    records = json.loads((run_dir / "ledger.json").read_text())["records"]
    candidates = [r for r in records if r.get("final_best_score") is not None
                  and math.isfinite(r["final_best_score"])]
    if not candidates:
        raise SystemExit("no finite-score candidate to export")
    best = min(candidates, key=lambda r: r["final_best_score"])
    params = (best.get("applied_incumbent") or {}).get("params")
    if params is None:
        raise SystemExit("selected incumbent has no applied parameters")
    public = run_dir / "run_input" / "public"
    if not public.is_dir():
        raise SystemExit(f"missing staged public data: {public}")
    candidate = run_dir / "candidates" / best["run_id"]
    task = repo / "tasks" / a.task
    os.environ["MLEBENCH_PUBLIC_DATA"] = str(public)
    sys.path[:0] = [str(task), str(candidate), str(repo)]
    from driver.resources import task_resource_lease

    cfg = tomllib.loads((task / "task.toml").read_text())
    with task_resource_lease(cfg, owner={"run_dir": str(run_dir), "kind": "export"}) as lease:
        os.environ.update(lease["env"])
        # Pin the device before any candidate or task import can initialize CUDA.
        import prepare

        spec = importlib.util.spec_from_file_location("candidate_train", candidate / "train.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        print(f"exporting {best['run_id']} score={best['final_best_score']}", flush=True)
        prepare.export_submission(module.make_model, params, run_dir / "submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
