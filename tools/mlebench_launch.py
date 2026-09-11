#!/usr/bin/env python3
"""Launch a driver command with one MLE-bench public-data mount.

This is an operator boundary, not a replacement for the driver.  It stages
the selected task before entering bubblewrap, then exposes only the repository
and that task's public tree.  The driver consequently fails closed if it is
run directly as root without this wrapper.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

from mlebench_stage import stage_public


def _optional_bind(args: list[str], source: Path, target: str) -> None:
    if source.exists():
        args.extend(["--ro-bind", str(source), target])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a driver command is required after --")
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SystemExit("bwrap is required; refusing an unisolated MLE-bench run")
    repo = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    if not (repo / "tasks" / args.task / "task.toml").is_file():
        raise SystemExit(f"unknown task surface: {args.task}")
    try:
        run_dir.relative_to(repo)
    except ValueError:
        raise SystemExit("run-dir must be inside repo-root for the isolated launcher")
    with (repo / "tasks" / args.task / "task.toml").open("rb") as stream:
        competition_id = (tomllib.load(stream).get("mlebench") or {}).get("competition_id")
    public = stage_public(args.task, run_dir, competition_id=competition_id)
    command = list(args.command)
    if command[0] == "--":
        command = command[1:]
    env = os.environ.copy()
    env.update({
        "MLEBENCH_PRESTAGED": "1",
        "MLEBENCH_NAMESPACE_READY": "1",
        "MLEBENCH_PUBLIC_DATA": "/mnt/mle-public",
    })
    sandbox = [
        bwrap, "--die-with-parent", "--new-session", "--unshare-pid",
        "--unshare-user-try", "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/tmp", "--bind", str(repo), "/workspace",
        "--bind", str(public), "/mnt/mle-public",
        "--chdir", "/workspace",
    ]
    for path in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        _optional_bind(sandbox, Path(path), path)
    _optional_bind(sandbox, Path.home() / ".local", "/root/.local")
    _optional_bind(sandbox, Path.home() / ".cache", "/root/.cache")
    _optional_bind(sandbox, Path.home() / "miniconda3", "/root/miniconda3")
    sandbox += ["--setenv", "MLEBENCH_PRESTAGED", "1",
                "--setenv", "MLEBENCH_NAMESPACE_READY", "1",
                "--setenv", "MLEBENCH_PUBLIC_DATA", "/mnt/mle-public",
                "--setenv", "PATH", "/root/miniconda3/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin"]
    sandbox.extend(command)
    return subprocess.call(sandbox, cwd=repo, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
