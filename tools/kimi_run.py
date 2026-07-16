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
VS Code extension install locations.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / ".kimi" / "agents"
HOOKS_TOML = ROOT / ".kimi" / "kimi-hooks.toml"
USER_CONFIG = Path.home() / ".kimi" / "config.toml"


def find_kimi() -> str:
    """Locate the kimi executable."""
    candidates = [os.environ.get("KIMI_BIN"), shutil.which("kimi")]
    candidates += sorted(
        glob.glob(
            os.path.expanduser(
                "~/.vscode*/data/User/globalStorage/moonshot-ai.kimi-code/bin/kimi/kimi"
            )
        )
    )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    sys.exit(
        "kimi_run.py: cannot find the kimi binary; set KIMI_BIN or put `kimi` on PATH"
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
