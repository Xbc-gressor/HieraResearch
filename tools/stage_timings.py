#!/usr/bin/env python3
"""Deterministic stage-boundary receipts for the background-research stage.

The background stage has run for over three hours with nothing written to the
run directory between the last retrieval receipt and the finished
``background.md``, leaving its largest time block unattributable. The
deterministic helpers that stage already calls append a boundary event here on
every invocation, so a later diagnosis can separate a long generation from a
validator fix loop without asking the agent what it did.

Only helper invocations are recorded. ``plan`` and ``distill`` have no helper
call site of their own and are derived from these events plus artifact mtimes;
an agent-reported timestamp is never accepted into this file.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TIMINGS_FILENAME = "background_stage_timings.json"
STAGES = ("plan", "retrieve", "distill", "write")
RUN_MARKER = "framework_cfg.json"
BACKGROUND_FILENAME = "background.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def run_dir_for(path: Path) -> Path | None:
    """Locate the run directory owning an artifact path, or None."""
    try:
        candidate = Path(path).resolve()
    except OSError:
        return None
    for directory in [candidate.parent, *candidate.parents]:
        if (directory / RUN_MARKER).exists():
            return directory
    return None


def load(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / TIMINGS_FILENAME
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "events": []}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        raise ValueError(f"{path}: stage timings must be an object with an events list")
    return data


def record(
    artifact: Path, *, stage: str, action: str, ok: bool, detail: str | None = None
) -> None:
    """Append one boundary event. Never raises — this is a receipt, not a gate."""
    try:
        run_dir = run_dir_for(artifact)
        if run_dir is None:
            return
        payload = load(run_dir)
        payload["schema_version"] = SCHEMA_VERSION
        payload["events"].append(
            {"stage": stage, "action": action, "at": _now(), "ok": bool(ok),
             "detail": detail}
        )
        path = run_dir / TIMINGS_FILENAME
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=run_dir, delete=False
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return


def _span(start: str | None, end: str | None) -> dict[str, Any]:
    """A boundary pair, with seconds only when both ends are known."""
    seconds: float | None = None
    if start and end:
        try:
            seconds = (
                datetime.fromisoformat(end) - datetime.fromisoformat(start)
            ).total_seconds()
        except ValueError:
            seconds = None
    return {"start": start, "end": end, "seconds": seconds}


def derive(run_dir: Path) -> dict[str, Any]:
    """Reconstruct stage spans from recorded events and artifact mtimes.

    A boundary that cannot be established stays null and its span reports null
    seconds; no duration is ever imputed from a partial pair.
    """
    run_dir = Path(run_dir)
    events = load(run_dir).get("events", [])
    retrieve = [event for event in events if event.get("stage") == "retrieve"]
    validate = [event for event in events if event.get("stage") == "write"]
    run_start = _mtime(run_dir / RUN_MARKER)
    first_retrieve = retrieve[0]["at"] if retrieve else None
    last_retrieve = retrieve[-1]["at"] if retrieve else None
    background = _mtime(run_dir / BACKGROUND_FILENAME)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "stages": {
            "plan": _span(run_start, first_retrieve),
            "retrieve": _span(first_retrieve, last_retrieve),
            "distill": _span(last_retrieve, background),
            "write": _span(background, validate[-1]["at"] if validate else None),
        },
        "retrieval_calls": len(retrieve),
        "validator_invocations": len(validate),
        "validator_failures": sum(1 for event in validate if not event.get("ok")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("derive", help="render stage spans for a run directory")
    show.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(derive(args.run_dir), indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
