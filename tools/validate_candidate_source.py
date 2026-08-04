#!/usr/bin/env python3
"""Validate one agent-authored candidate Python source file.

This is the single deterministic gate for authored candidate sources: the file
exists, holds non-empty strict UTF-8 text, and compiles — nothing more, nothing
less. The coordinator runs it as the post-edit gate, and bounded agent edit
invocations may run the exact same command through their allow-listed Bash
boundary, so both sides always agree on accept/reject. Deliberately stricter
than ``python -m py_compile``: py_compile honors PEP 263 coding cookies and
skips a UTF-8 BOM, while this gate decodes the bytes as strict UTF-8 and
compiles the decoded text, matching what the coordinator considers a legal
authored source.

Emits the standard typed verdict JSON ({"ok", "failure_kind", "errors"}) and
exits 0 on accept, 1 on rejection; unexpected I/O failures propagate as an
ordinary traceback so the caller classifies them as operational, not authored,
failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REJECTION_KIND = "candidate_source_validation"


def source_errors(path: Path) -> list[str]:
    """First-failure verdict for one authored source, mirroring the gate."""
    if not path.is_file():
        return [f"candidate writer did not create {path}"]
    try:
        source = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return [str(exc)]
    if not source.strip():
        return [f"candidate source is empty: {path}"]
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        return [str(exc)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True, type=Path)
    args = parser.parse_args()
    errors = source_errors(args.path)
    if errors:
        print(
            json.dumps(
                {"ok": False, "failure_kind": REJECTION_KIND, "errors": errors},
                indent=2,
            )
        )
        return 1
    print(json.dumps({"ok": True, "errors": []}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
