#!/usr/bin/env python3
"""Launch a HieraResearch agent on the kimi-cli runtime.

Usage:
    python3 tools/kimi_run.py --agent <name> [kimi args...]
    python3 tools/kimi_run.py --agent autoresearch-experiment \\
        --print -p "task_name=tabular-blind tag=0717-demo max_evaluations=50"
    python3 tools/kimi_run.py --print-config      # inspect the merged config

kimi-cli has no project-level config file and `--config` *replaces* the user's
config rather than merging into it, so this launcher:

1. reads the user's `~/.kimi/config.toml` (providers, models, services),
2. appends the repo's hook registrations from `.kimi/kimi-hooks.toml`
   (expanding `${HIERA_ROOT}` to this repo's absolute root),
3. execs `kimi --agent-file .kimi/agents/<name>.yaml --config <merged JSON>`
   with any remaining arguments forwarded verbatim.

The kimi binary is located via `$KIMI_BIN`, then `PATH`, then the common
VS Code extension install locations. Candidates are probed for `--agent-file`
support (agent files were added after kimi-cli 0.x), so a stale `kimi` on
PATH is skipped with a note instead of failing at launch.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / ".kimi" / "agents"
HOOKS_TOML = ROOT / ".kimi" / "kimi-hooks.toml"
USER_CONFIG = Path.home() / ".kimi" / "config.toml"


def supports_agent_files(binary: str) -> bool:
    """Probe whether a kimi candidate knows `--agent-file` (post-0.x feature)."""
    try:
        result = subprocess.run(
            [binary, "--help"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "--agent-file" in result.stdout + result.stderr


def path_hits(name: str) -> list[str]:
    """All matches for `name` on PATH, in PATH order (`which -a` semantics).

    shutil.which returns only the first hit; here a stale 0.x binary earlier in
    PATH must not hide a current install later in PATH (e.g. a uv-tool kimi-cli
    in ~/.local/bin shadowed by an old ~/.kimi-code/bin/kimi).
    """
    hits = []
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(directory or ".", name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            hits.append(candidate)
    return hits


def find_kimi() -> str:
    """Locate a kimi executable that supports agent files."""
    candidates = [os.environ.get("KIMI_BIN"), *path_hits("kimi")]
    candidates += sorted(
        glob.glob(
            os.path.expanduser(
                "~/.vscode*/data/User/globalStorage/moonshot-ai.kimi-code/bin/kimi/kimi"
            )
        )
    )
    seen: set[str] = set()
    rejected: list[str] = []
    for candidate in candidates:
        if (
            not candidate
            or candidate in seen
            or not os.path.isfile(candidate)
            or not os.access(candidate, os.X_OK)
        ):
            continue
        seen.add(candidate)
        if supports_agent_files(candidate):
            return candidate
        rejected.append(candidate)
        print(
            f"kimi_run.py: skipping {candidate} (no --agent-file support; "
            "outdated kimi-cli)",
            file=sys.stderr,
        )
    detail = ""
    if rejected:
        detail = (
            "\nskipped outdated kimi binaries without --agent-file support:\n"
            + "\n".join(f"  - {path}" for path in rejected)
            + "\nupgrade kimi-cli or point KIMI_BIN at a newer binary"
        )
    sys.exit(
        "kimi_run.py: cannot find a kimi binary with --agent-file support; "
        "set KIMI_BIN or put a current `kimi` on PATH" + detail
    )


def build_merged_config() -> dict:
    """User config + this repo's hook registrations (idempotent)."""
    config: dict = {}
    if USER_CONFIG.is_file():
        with USER_CONFIG.open("rb") as handle:
            config = tomllib.load(handle)
    with HOOKS_TOML.open("rb") as handle:
        repo_hooks = tomllib.load(handle).get("hooks", [])
    for entry in repo_hooks:
        if "command" in entry:
            entry["command"] = entry["command"].replace("${HIERA_ROOT}", str(ROOT))
    hooks = config.setdefault("hooks", [])
    if not isinstance(hooks, list):
        raise TypeError(f"{USER_CONFIG}: 'hooks' is not an array")
    known = {(h.get("event"), h.get("matcher"), h.get("command")) for h in hooks}
    for entry in repo_hooks:
        key = (entry.get("event"), entry.get("matcher"), entry.get("command"))
        if key not in known:
            hooks.append(entry)
    return config


def main() -> int:
    # allow_abbrev=False: forwarded kimi flags like `--print` must not be
    # captured as abbreviations of this launcher's own `--print-config`.
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], allow_abbrev=False
    )
    parser.add_argument(
        "--agent",
        choices=sorted(p.stem for p in AGENTS_DIR.glob("*.yaml") if p.stem != "_common"),
        help="HieraResearch agent to run as the main thread",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the merged config JSON and exit (no kimi launch)",
    )
    args, forwarded = parser.parse_known_args()

    merged = build_merged_config()
    if args.print_config:
        print(json.dumps(merged, indent=2))
        return 0
    if not args.agent:
        parser.error("--agent is required unless --print-config is given")

    command = [
        find_kimi(),
        "--agent-file",
        str(AGENTS_DIR / f"{args.agent}.yaml"),
        "--config",
        json.dumps(merged),
        *forwarded,
    ]
    os.execvp(command[0], command)
    return 127  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
