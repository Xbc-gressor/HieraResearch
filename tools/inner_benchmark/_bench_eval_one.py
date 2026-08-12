"""Evaluate ONE benchmark config in a fresh subprocess, for per_runtime_limit.

Mirror of tools/tuners/_eval_one.py with ONE sanctioned deviation (PLAN §5.1):
the score-fn name is passed explicitly (from the checkpoint spec) instead of
being resolved from tasks/<task>/task.toml via candidate-path inference —
frozen benchmark checkpoints live at arbitrary paths, so path-based task
inference is impossible. Module loading, revision pinning, and the RESULT
protocol are identical to production.

Usage (invoked by objective.evaluate, not by hand):
    python _bench_eval_one.py <candidate_train_path> <params_json> \\
        <expected_revision_json> <score_fn_name>

Prints `RESULT:<float>` on stdout. A non-finishing / erroring run prints
nothing -> the parent records a crash. Run in its own process group so the
parent can hard-kill the whole tree on timeout.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tuners"))  # for _common
from _common import load_candidate_modules  # noqa: E402


def main() -> int:
    candidate_path = Path(sys.argv[1])
    params = json.loads(sys.argv[2])
    expected_revision = json.loads(sys.argv[3])
    score_fn = sys.argv[4]
    train_module, prepare_module = load_candidate_modules(
        candidate_path,
        expected_execution_revision=expected_revision,
    )
    if not hasattr(prepare_module, score_fn):
        raise RuntimeError(
            f"prepare.py missing score fn {score_fn!r} (checkpoint spec)"
        )
    evaluate = getattr(prepare_module, score_fn)
    score = float(evaluate(train_module.make_model, params))
    print(f"RESULT:{score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
