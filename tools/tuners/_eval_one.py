"""Evaluate ONE config in a fresh subprocess, for the per_runtime_limit timeout.

Usage (invoked by _common.timed_eval, not by hand):
    python _eval_one.py <candidate_train_path> <params_json> [expected_revision_json]

Imports the candidate's train.py + the task's prepare.py exactly as the tuners do
(load_candidate_modules + resolve_score_fn), runs the single score_fn(make_model,
params), and prints `RESULT:<float>` on stdout. Run in its own process group so the
parent can hard-kill the whole tree on timeout. A non-finishing / erroring run
prints nothing -> the parent raises so the caller records a failed/crashed
trial with a failure receipt.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tools/tuners (for _common)
from _common import load_candidate_modules, resolve_score_fn  # noqa: E402


def main() -> int:
    candidate_path = Path(sys.argv[1])
    params = json.loads(sys.argv[2])
    expected_revision = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    train_module, prepare_module = load_candidate_modules(
        candidate_path,
        expected_execution_revision=expected_revision,
    )
    evaluate = resolve_score_fn(prepare_module, candidate_path)
    score = float(evaluate(train_module.make_model, params))
    print(f"RESULT:{score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
