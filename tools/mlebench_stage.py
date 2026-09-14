#!/usr/bin/env python3
"""Stage one MLE-bench public dataset for a run.

The canonical data root is an operator-owned cache.  Agents receive only a
copy of the task's prepared/public tree under the run directory; raw and
private siblings never get copied into that tree.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import fnmatch


FORBIDDEN_PARTS = {"raw", "private", "solution", "answers", "answer"}
FORBIDDEN_GLOBS = ("*answer*", "*solution*", "*private*")
FORBIDDEN_NAMES = {"manifest.json", "download.log", "staging_manifest.json"}


class StageError(RuntimeError):
    pass


def _source_dir(data_root: Path, task: str, competition_id: str | None = None) -> Path:
    names = [task]
    if competition_id and competition_id not in names:
        names.append(competition_id)
    task_root = next((data_root / name for name in names if (data_root / name).is_dir()), None)
    if task_root is None:
        expected = ", ".join(str(data_root / name) for name in names)
        raise StageError(f"canonical MLE-bench task directory is missing: {expected}")
    task_root = task_root.resolve()
    candidates = (task_root / "prepared" / "public", task_root / "public", task_root)
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if candidate == task_root and any(
            child.name.lower() in FORBIDDEN_PARTS
            for child in task_root.iterdir()
            if child.is_dir()
        ):
            continue
        return candidate
    raise StageError(f"no public data tree found for {task}: {task_root}")


def _allowed(relative: Path) -> bool:
    parts = {part.lower() for part in relative.parts}
    if parts & FORBIDDEN_PARTS:
        return False
    if relative.name.lower() in FORBIDDEN_NAMES:
        return False
    return not any(
        fnmatch.fnmatch(relative.name.lower(), pattern)
        for pattern in FORBIDDEN_GLOBS
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_public(task: str, run_dir: Path, *, data_root: Path | None = None,
                 competition_id: str | None = None) -> Path:
    run_dir = Path(run_dir).resolve()
    raw_root = str(data_root) if data_root is not None else os.environ.get("MLEBENCH_DATA_ROOT")
    if not raw_root:
        raise StageError("MLEBENCH_DATA_ROOT is required for an MLE-bench run")
    data_root = Path(raw_root).resolve()
    source = _source_dir(data_root, task, competition_id)
    staging_root = run_dir / "run_input"
    public = staging_root / "public"
    if public.exists():
        for path in sorted(public.rglob("*"), reverse=True):
            try:
                path.chmod(0o755 if path.is_dir() else 0o644)
            except OSError:
                pass
        public.chmod(0o755)
        shutil.rmtree(public)
    public.mkdir(parents=True)
    records = []
    for src in sorted(source.rglob("*")):
        if not src.is_file():
            continue
        relative = src.relative_to(source)
        if not _allowed(relative):
            raise StageError(f"forbidden file in public source tree: {relative}")
        dst = public / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        # Candidate processes receive a public snapshot, not a writable cache.
        dst.chmod(0o444)
        records.append({
            "path": relative.as_posix(),
            "bytes": dst.stat().st_size,
            "sha256": _sha256(dst),
        })
    if not records:
        raise StageError(f"public data tree is empty: {source}")
    manifest = {
        "schema_version": 1,
        "task": task,
        "source_kind": "prepared/public",
        "private_exposed": False,
        "files": records,
    }
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = staging_root / "staging_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(staging_root / "staging_manifest.json")
    for directory in (public, *[p for p in public.rglob("*") if p.is_dir()]):
        directory.chmod(0o555)
    return public


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--competition-id")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(stage_public(args.task, args.run_dir, competition_id=args.competition_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
