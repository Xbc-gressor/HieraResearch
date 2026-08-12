"""Per-cell logging artifacts for the inner-tuner benchmark (PLAN §十).

Per-cell output directory (path supplied by the caller) holding:

- manifest.json : checkpoint id+hash, arm, seed, model/decoding placeholder,
  dependency versions, hardware info, arm active-dimension count, and an open
  ``extra`` slot for arm-specific calibration values. Write-once at cell start.
- events.jsonl  : append-only JSONL intermediate trajectory; every event
  carries the common envelope (see make_event / EVENT_KEYS).
- result.json   : final metrics + best configuration.

These artifacts are intentionally NOT compatible with the old benchmark
remnant format.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tuners"))

from _common import _to_native  # noqa: E402

MANIFEST_FILENAME = "manifest.json"
EVENTS_FILENAME = "events.jsonl"
RESULT_FILENAME = "result.json"


def _json_default(value):
    """json default hook: production _to_native, then repr as a last resort.

    _to_native is a NARROWING helper — it returns unknown types unchanged, so
    json re-enters this hook on the same object and raises "Circular reference
    detected". An arm logging a np.bool_ (the natural result of `new < best`)
    would otherwise take down a cell whose objective had already run.
    """
    native = _to_native(value)
    if type(native) is type(value) and not isinstance(
        native, (str, int, float, bool, list, dict, type(None))
    ):
        return repr(value)
    return native


MANIFEST_KEYS = (
    "checkpoint_id",
    "checkpoint_hash",
    "candidate_execution_revision",  # tune_tools revision the objective pins
    "arm",
    "seed",
    "model",  # placeholder dict: model/decoding config, filled by the runner task
    "dependencies",  # dependency versions dict
    "hardware",  # hardware info dict
    "active_dimensions",  # arm active-dimension count
    "extra",  # open slot for arm-specific calibration values (e.g. SPSA s / a_0)
)

EVENT_KEYS = (
    "eval_index",
    "transaction_id",
    "proposal",
    "source",
    "rationale",
    "preflight_status",
    "status",
    "score",
    "incumbent_before",
    "incumbent_after",
    "arm_state",
)


def write_manifest(cell_dir, manifest: dict) -> Path:
    """Write manifest.json at cell start. Write-once: FileExistsError on a
    second call. All MANIFEST_KEYS must be present."""
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"manifest missing required keys: {missing}")
    path = _cell_dir(cell_dir) / MANIFEST_FILENAME
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    return path


def make_event(*, eval_index=None, transaction_id=None, proposal=None,
               source=None, rationale=None, preflight_status=None, status=None,
               score=None, incumbent_before=None, incumbent_after=None,
               arm_state=None, **extra) -> dict:
    """Build the common event envelope; arms add their specifics via extra
    keyword args (merged at top level) or inside ``arm_state``."""
    event = {
        "eval_index": eval_index,
        "transaction_id": transaction_id,
        "proposal": proposal,
        "source": source,
        "rationale": rationale,
        "preflight_status": preflight_status,
        "status": status,
        "score": score,
        "incumbent_before": incumbent_before,
        "incumbent_after": incumbent_after,
        "arm_state": {} if arm_state is None else dict(arm_state),
    }
    event.update(extra)
    return event


def append_event(cell_dir, event: dict) -> Path:
    """Append one event to events.jsonl. The common envelope (EVENT_KEYS)
    must be present; build events with make_event."""
    missing = [key for key in EVENT_KEYS if key not in event]
    if missing:
        raise ValueError(f"event missing envelope keys {missing}; use make_event")
    path = _cell_dir(cell_dir) / EVENTS_FILENAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(event, ensure_ascii=False, default=_json_default) + "\n"
        )
    return path


def write_result(cell_dir, result: dict) -> Path:
    """Write result.json: final metrics + best configuration (dict shape is
    defined by the runner task)."""
    path = _cell_dir(cell_dir) / RESULT_FILENAME
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    return path


def _cell_dir(cell_dir) -> Path:
    path = Path(cell_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
