"""Run one task-owned, no-score candidate preflight in a fresh subprocess."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_candidate_modules, resolve_preflight_fn  # noqa: E402


def main() -> int:
    candidate_path = Path(sys.argv[1])
    params = json.loads(sys.argv[2])
    expected_revision = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    if expected_revision is None:
        train_module, prepare_module = load_candidate_modules(
            candidate_path,
            required_symbols=("make_model",),
        )
    else:
        train_module, prepare_module = load_candidate_modules(
            candidate_path,
            expected_execution_revision=expected_revision,
        )
    preflight = resolve_preflight_fn(prepare_module, candidate_path)
    if preflight is None:
        print("PREFLIGHT:null")
        return 0
    result = preflight(train_module.make_model, params)
    if result is None:
        result = {"status": "ok"}
    print("PREFLIGHT:" + json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
