"""Driver progress events: one stdout line + one JSONL row per event.

This is the observability substrate that replaces the retired Claude Code
status lines: a human (or later tooling) can watch a run live from the
terminal or from <run_dir>/driver_events.jsonl.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class EventsLog:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "driver_events.jsonl"

    def emit(self, kind: str, **fields: object) -> None:
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **fields}
        print(f"[driver] {kind}: {json.dumps(fields, ensure_ascii=False, sort_keys=True)}", flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
