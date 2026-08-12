"""Run one task-owned, no-score candidate preflight in a fresh subprocess.

Mirror of tools/tuners/_preflight_one.py with sanctioned deviations
(PLAN §5.1): the preflight-fn name is passed explicitly (from the checkpoint
spec) instead of being resolved from tasks/<task>/task.toml via candidate-path
inference — frozen benchmark checkpoints live at arbitrary paths; the
``probe_mode`` axis is dropped (the benchmark runs only the correctness
preflight, never the resource probe); the expected execution revision is
mandatory (the benchmark always pins).

Usage (invoked by objective.preflight, not by hand):
    python _bench_preflight_one.py <candidate_train_path> <params_json> \\
        <expected_revision_json> <preflight_fn_name>

Prints `PREFLIGHT:<json>` on stdout. A non-finishing / erroring run prints
nothing -> the parent records a rejection. Never consumes objective budget.
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
    preflight_fn = sys.argv[4]
    train_module, prepare_module = load_candidate_modules(
        candidate_path,
        expected_execution_revision=expected_revision,
    )
    if not hasattr(prepare_module, preflight_fn):
        raise RuntimeError(
            f"prepare.py missing preflight fn {preflight_fn!r} (checkpoint spec)"
        )
    preflight = getattr(prepare_module, preflight_fn)
    result = preflight(train_module.make_model, params)
    if result is None:
        result = {"status": "ok"}
    print("PREFLIGHT:" + json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
