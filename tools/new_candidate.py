#!/usr/bin/env python3
"""Create a run-local candidate directory for an autoresearch task."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
from pathlib import Path
from typing import Optional

from validate_tasks import ROOT, parse_task_toml


DEFAULT_COPY_FILES = ["prepare.py", "train.py"]
DEFAULT_ENTRYPOINT = "train.py"
DEFAULT_TEMPLATE = "runs/{task_name}/{tag}/candidates/{run_id}"
BRIEF_FILENAME = "_candidate_brief.json"


def candidate_path(template: str, task_name: str, tag: str, run_id: str) -> Path:
    return ROOT / template.format(task_name=task_name, tag=tag, run_id=run_id)


def resolve_source_candidate(
    template: str,
    task_name: str,
    tag: str,
    from_candidate: Optional[str],
) -> Optional[Path]:
    if not from_candidate:
        return None
    raw_path = Path(from_candidate)
    if raw_path.exists():
        return raw_path.resolve()
    if raw_path.is_absolute() or "/" in from_candidate:
        return (ROOT / raw_path).resolve()
    return candidate_path(template, task_name, tag, from_candidate)


def candidate_brief(ledger_path: Path, run_id: str) -> dict | None:
    """Return the immutable implementation fields for one persisted record."""
    if not ledger_path.is_file():
        return None
    try:
        data = json.loads(ledger_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for record in data.get("records", []):
        if record.get("run_id") == run_id:
            return {
                "schema_version": 1,
                "run_id": run_id,
                "op": record.get("op"),
                "idea": record.get("idea"),
                "change": record.get("change"),
                "source_run_ids": record.get("source_run_ids") or [],
                "candidate_name": record.get("candidate_name"),
            }
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name", help="Task folder name under tasks/.")
    parser.add_argument("tag", help="Run tag under runs/<task-name>/.")
    parser.add_argument("run_id", help="Candidate/run id, e.g. 000 or 001.")
    parser.add_argument(
        "--from-candidate",
        help="Candidate id or path whose train.py should seed this candidate.",
    )
    parser.add_argument(
        "--skip-entrypoint",
        action="store_true",
        help=(
            "Do not copy the entrypoint; require its ledger record and write "
            f"{BRIEF_FILENAME} for candidate-writer. Mutually exclusive with "
            "--from-candidate."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite copied files if the candidate directory already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sources and print paths without creating files.",
    )
    args = parser.parse_args()

    task_dir = ROOT / "tasks" / args.task_name
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        parser.error(f"missing task.toml: {task_toml}")

    config = parse_task_toml(task_toml)
    candidate = config.get("candidate", {})
    template = candidate.get("root_template", DEFAULT_TEMPLATE)
    copy_files = candidate.get("copy_files", DEFAULT_COPY_FILES)
    entrypoint = candidate.get("entrypoint", DEFAULT_ENTRYPOINT)

    if not isinstance(template, str):
        parser.error("candidate.root_template must be a string")
    if not isinstance(copy_files, list) or not all(isinstance(item, str) for item in copy_files):
        parser.error("candidate.copy_files must be a list of strings")
    if not isinstance(entrypoint, str):
        parser.error("candidate.entrypoint must be a string")
    if args.skip_entrypoint and args.from_candidate:
        parser.error("--skip-entrypoint and --from-candidate are mutually exclusive")
    if args.skip_entrypoint:
        copy_files = [item for item in copy_files if item != entrypoint]

    ledger_path = ROOT / "runs" / args.task_name / args.tag / "ledger.json"
    brief = candidate_brief(ledger_path, args.run_id) if args.skip_entrypoint else None
    if args.skip_entrypoint and brief is None:
        parser.error(
            f"--skip-entrypoint requires ledger record {args.run_id}: {ledger_path}"
        )

    dest = candidate_path(template, args.task_name, args.tag, args.run_id)
    if dest.exists() and any(dest.iterdir()) and not args.force:
        parser.error(f"candidate directory already exists and is not empty: {dest}")
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    source_candidate = resolve_source_candidate(
        template,
        args.task_name,
        args.tag,
        args.from_candidate,
    )

    for relative in copy_files:
        if relative == entrypoint and source_candidate is not None:
            source = source_candidate / entrypoint
        else:
            source = task_dir / relative
        if not source.is_file():
            parser.error(f"missing source file for {relative}: {source}")
        target = dest / relative
        if args.dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not args.force:
            parser.error(f"target file already exists: {target}")
        shutil.copy2(source, target)

    brief_path = dest / BRIEF_FILENAME
    if brief is not None and not args.dry_run:
        brief_path.write_text(json.dumps(brief, indent=2) + "\n")

    entrypoint_path = dest / entrypoint
    task_project = config.get("env", {}).get("project", f"tasks/{args.task_name}")
    print(f"candidate_dir: {dest.relative_to(ROOT)}")
    print(f"entrypoint:    {entrypoint_path.relative_to(ROOT)}")
    if brief is not None:
        print(f"brief:         {brief_path.relative_to(ROOT)}")
    print(
        "run_command:   "
        f"uv --directory {shlex.quote(str(task_project))} "
        f"run python {shlex.quote(str(entrypoint_path))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
