#!/usr/bin/env python3
"""Create a run-local candidate directory for an autoresearch task."""

from __future__ import annotations

import argparse
import hashlib
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


def provided_seed_entrypoint(config: dict, task_dir: Path) -> Path | None:
    """Resolve the task-declared provided entrypoint, if one exists.

    The candidate entrypoint and seed entrypoint must agree so the normal copy
    and evaluation paths cannot silently score a different file.
    """
    seed = config.get("seed")
    if not isinstance(seed, dict):
        return None
    entrypoint = seed.get("entrypoint", DEFAULT_ENTRYPOINT)
    provided = seed.get("provided", [])
    if (
        not isinstance(entrypoint, str)
        or not entrypoint
        or not isinstance(provided, list)
        or any(not isinstance(item, str) for item in provided)
    ):
        return None
    candidate = config.get("candidate", {})
    if not isinstance(candidate, dict):
        candidate = {}
    candidate_entrypoint = candidate.get("entrypoint", DEFAULT_ENTRYPOINT)
    if candidate_entrypoint != entrypoint:
        raise ValueError(
            "seed.entrypoint must match candidate.entrypoint for provided-baseline admission"
        )

    expected = (task_dir / entrypoint).resolve()
    for item in provided:
        raw = Path(item)
        candidates = [raw] if raw.is_absolute() else [task_dir / raw, ROOT / raw]
        for source in candidates:
            if source.is_file() and source.resolve() == expected:
                return source.resolve()
    return None


def _implementation_source(path: Path | None) -> dict:
    if path is None:
        return {"kind": "generated"}
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        display_path = resolved.as_posix()
    return {
        "kind": "provided_entrypoint",
        "path": display_path,
        "sha256": "sha256:" + hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _content_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _primary_parent_receipt(
    ledger_path: Path,
    source_run_ids: list[str],
    *,
    entrypoint: str = DEFAULT_ENTRYPOINT,
) -> dict | None:
    """Pin the first numeric parent as the child's structural/tuning base."""
    if not source_run_ids:
        return None
    parent_run_id = source_run_ids[0]
    parent_train = ledger_path.parent / "candidates" / parent_run_id / entrypoint
    if not parent_train.is_file():
        return None
    return {
        "schema_version": 1,
        "run_id": parent_run_id,
        "path": _display_path(parent_train),
        "sha256": _content_sha256(parent_train),
    }


def _primary_parent_tuning(data: dict, parent_run_id: str) -> dict | None:
    """The parent's ledger-recorded tuning state at inheritance time.

    Inherited parameters came from the parent's current revision; when that
    parent was already tuned, its incumbent is often near-optimal for the
    child implementation too, which changes the child's tuning potential.
    """
    for record in data.get("records", []):
        if isinstance(record, dict) and record.get("run_id") == parent_run_id:
            return {
                "tune": bool(record.get("tune")),
                "evaluation_depth": record.get("evaluation_depth"),
            }
    return None


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


def candidate_brief(
    ledger_path: Path,
    run_id: str,
    *,
    implementation_source: dict | None = None,
    entrypoint: str = DEFAULT_ENTRYPOINT,
) -> dict | None:
    """Return the immutable implementation fields for one persisted record."""
    if not ledger_path.is_file():
        return None
    try:
        data = json.loads(ledger_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for record in data.get("records", []):
        if record.get("run_id") == run_id:
            parents = record.get("source_run_ids")
            expected_parents = {"fresh": 0, "improve": 1, "crossover": 2}.get(
                record.get("op")
            )
            if (
                expected_parents is None
                or not isinstance(parents, list)
                or len(parents) != expected_parents
                or len(parents) != len(set(parents))
                or any(not isinstance(parent, str) or not parent.isdigit() for parent in parents)
                or not isinstance(record.get("idea"), str)
                or not record["idea"].strip()
                or not isinstance(record.get("change"), str)
                or not record["change"].strip()
                or not isinstance(record.get("semantic_point"), dict)
                or not isinstance(record.get("policy_receipt"), dict)
            ):
                return None
            primary_parent = _primary_parent_receipt(
                ledger_path,
                parents,
                entrypoint=entrypoint,
            )
            if parents and primary_parent is None:
                return None
            return {
                "schema_version": 4,
                "run_id": run_id,
                "op": record.get("op"),
                "idea": record.get("idea"),
                "change": record.get("change"),
                "source_run_ids": parents,
                "semantic_point": record.get("semantic_point"),
                "policy_receipt": record.get("policy_receipt"),
                "candidate_name": record.get("candidate_name"),
                "primary_parent": primary_parent,
                "primary_parent_tuning": (
                    _primary_parent_tuning(data, primary_parent["run_id"])
                    if primary_parent is not None
                    else None
                ),
                "implementation_source": (
                    implementation_source
                    or (
                        {
                            "kind": "primary_parent_snapshot",
                            "parent_run_id": primary_parent["run_id"],
                            "path": primary_parent["path"],
                            "sha256": primary_parent["sha256"],
                        }
                        if primary_parent is not None
                        else _implementation_source(None)
                    )
                ),
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
            "Require its ledger record and write the candidate-writer brief; "
            "non-fresh candidates start from an exact primary-parent entrypoint "
            "snapshot, while fresh candidates start without an entrypoint. "
            "Mutually exclusive with --from-candidate."
        ),
    )
    parser.add_argument(
        "--provided-baseline",
        action="store_true",
        help=(
            "copy the task-declared seed entrypoint for an already-admitted "
            "baseline record and stamp its source receipt into the candidate brief"
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
    selected_modes = sum(
        bool(value)
        for value in (
            args.skip_entrypoint,
            args.provided_baseline,
            args.from_candidate,
        )
    )
    if selected_modes > 1:
        parser.error(
            "--skip-entrypoint, --provided-baseline, and --from-candidate "
            "are mutually exclusive"
        )
    if args.skip_entrypoint:
        copy_files = [item for item in copy_files if item != entrypoint]
    provided_entrypoint = None
    if args.provided_baseline:
        try:
            provided_entrypoint = provided_seed_entrypoint(config, task_dir)
        except ValueError as exc:
            parser.error(str(exc))
        if provided_entrypoint is None:
            parser.error(
                "task does not declare its candidate entrypoint in [seed].provided"
            )
        if entrypoint not in copy_files:
            parser.error(
                "candidate.copy_files must include candidate.entrypoint for "
                "provided-baseline admission"
            )

    ledger_path = ROOT / "runs" / args.task_name / args.tag / "ledger.json"
    if args.provided_baseline:
        if args.run_id != "000":
            parser.error("--provided-baseline requires run_id 000")
        try:
            records = json.loads(ledger_path.read_text()).get("records", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            records = []
        if (
            not isinstance(records, list)
            or len(records) != 1
            or records[0].get("run_id") != "000"
        ):
            parser.error(
                "--provided-baseline requires run 000 to be the ledger's only record"
            )
    needs_brief = args.skip_entrypoint or args.provided_baseline
    brief = (
        candidate_brief(
            ledger_path,
            args.run_id,
            implementation_source=(
                _implementation_source(provided_entrypoint)
                if args.provided_baseline
                else None
            ),
            entrypoint=entrypoint,
        )
        if needs_brief
        else None
    )
    if needs_brief and brief is None:
        parser.error(
            "candidate creation requires a complete admitted ledger record "
            f"{args.run_id}: {ledger_path}"
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
        elif relative == entrypoint and provided_entrypoint is not None:
            source = provided_entrypoint
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

    # Normal non-fresh generation starts from an exact, helper-pinned snapshot
    # of the primary parent.  candidate-writer edits this local copy in place;
    # it no longer reconstructs a parent program from prose and references.
    if (
        args.skip_entrypoint
        and brief is not None
        and isinstance(brief.get("primary_parent"), dict)
    ):
        parent_path = Path(brief["primary_parent"]["path"])
        if not parent_path.is_absolute():
            parent_path = ROOT / parent_path
        if not parent_path.is_file():
            parser.error(f"missing primary-parent entrypoint: {parent_path}")
        if _content_sha256(parent_path) != brief["primary_parent"]["sha256"]:
            parser.error("primary-parent entrypoint changed while materializing candidate")
        target = dest / entrypoint
        if not args.dry_run:
            if target.exists() and not args.force:
                parser.error(f"target file already exists: {target}")
            shutil.copy2(parent_path, target)

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
