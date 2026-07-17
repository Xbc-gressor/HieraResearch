#!/usr/bin/env python3
"""Launch one Claude background-research retrieval condition.

The launcher keeps the prompt, adapter disable switches, and Claude native-web
permissions aligned. It never installs a backend or makes an optional live
service part of the repository contract.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFLINE_ENV = "HIERA_RETRIEVAL_OFFLINE"
DISABLED_ENV = "HIERA_RETRIEVAL_DISABLE_BACKENDS"
DISABLE_NATIVE_WEB_ENV = "HIERA_RETRIEVAL_DISABLE_NATIVE_WEB"


def _set_disabled(env: dict[str, str], *names: str) -> None:
    env[DISABLED_ENV] = ",".join(sorted({name.lower() for name in names}))


def _native_web_args(enabled: bool) -> list[str]:
    flag = "--allowedTools" if enabled else "--disallowedTools"
    # Use --flag=value because these Claude CLI options are variadic; a separate
    # value token could consume the positional prompt as another tool pattern.
    return [f"{flag}=WebSearch,WebFetch"]


def build_launch(
    task_name: str,
    tag: str,
    condition: str,
    frozen_corpus: Path | None,
    base_env: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str], list[str]]:
    env = dict(os.environ if base_env is None else base_env)
    for name in (OFFLINE_ENV, DISABLED_ENV, DISABLE_NATIVE_WEB_ENV):
        env.pop(name, None)
    run_dir = f"runs/{task_name}/{tag}"
    common = (
        f"Run background research only for {run_dir}. "
        "Write and validate background.md plus background_retrieval.json, then stop "
        "without generating candidates. Treat tools/search_backends.py as the only "
        "DeepXiv interface: never run which/import/glob/read/find/curl against an "
        "installed or sibling DeepXiv checkout."
    )
    shown_env: list[str] = []

    if condition == "full":
        native_web = True
        prompt = (
            common
            + " Use the full open-world condition: call "
            "`python tools/search_backends.py probe --backend deepxiv`, then search "
            "through the adapter with `--backend deepxiv`. Use native WebSearch/WebFetch "
            "only for a documented evidence gap. Record native searches with "
            "`record-search` and cited native fetches with `record-visit`."
        )
    elif condition == "no-deepxiv":
        native_web = True
        _set_disabled(env, "deepxiv", "jina", "direct")
        shown_env.append(f"{DISABLED_ENV}={env[DISABLED_ENV]}")
        prompt = (
            common
            + " Use the native-web-only ablation. DeepXiv, Jina, and direct visiting "
            "are disabled. Run native WebSearch for each planned question, retain its "
            "exact URL/title/snippet rows with `record-search`, fetch selected sources "
            "with native WebFetch, and retain cited content with `record-visit`."
        )
    elif condition == "no-native-web":
        native_web = False
        env[DISABLE_NATIVE_WEB_ENV] = "1"
        _set_disabled(env, "jina", "direct")
        shown_env.extend(
            [
                f"{DISABLE_NATIVE_WEB_ENV}=1",
                f"{DISABLED_ENV}={env[DISABLED_ENV]}",
            ]
        )
        prompt = (
            common
            + " Use the DeepXiv-only ablation: probe DeepXiv, search it through the "
            "adapter, and visit selected arXiv sources through the adapter. Native web, "
            "Jina, and direct visiting are disabled. Retain any DeepXiv failure as a "
            "coverage gap instead of substituting another backend."
        )
    elif condition == "frozen":
        if frozen_corpus is None:
            raise ValueError("the frozen condition requires --frozen-corpus")
        corpus = frozen_corpus.resolve()
        if not corpus.exists():
            raise ValueError(f"frozen corpus does not exist: {corpus}")
        native_web = False
        env[OFFLINE_ENV] = "1"
        env[DISABLE_NATIVE_WEB_ENV] = "1"
        shown_env.extend([f"{OFFLINE_ENV}=1", f"{DISABLE_NATIVE_WEB_ENV}=1"])
        prompt = (
            common
            + f" Use only the frozen corpus at {corpus}. Pass `--frozen-corpus {corpus}` "
            "on every adapter search and visit. All live retrieval and external receipt "
            "recording are disabled."
        )
    else:
        raise ValueError(f"unknown condition: {condition}")

    shown_env.append(f"Claude native web tools: {'allow' if native_web else 'deny'}")
    command = [
        "claude",
        "--agent",
        "background-researcher",
        "-p",
        *_native_web_args(native_web),
        prompt,
    ]
    return command, env, shown_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name")
    parser.add_argument("tag")
    parser.add_argument(
        "--condition",
        required=True,
        choices=["full", "no-deepxiv", "no-native-web", "frozen"],
    )
    parser.add_argument("--frozen-corpus", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the effective launch without running Claude"
    )
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.task_name / args.tag
    if not run_dir.is_dir():
        print(
            f"Run directory is missing: {run_dir}\n"
            f"Initialize it first with: python tools/init_run.py {args.task_name} {args.tag}",
            file=sys.stderr,
        )
        return 2
    try:
        command, env, shown_env = build_launch(
            args.task_name, args.tag, args.condition, args.frozen_corpus
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.dry_run:
        print("environment:")
        for item in shown_env:
            print(f"  {item}")
        print("command:")
        print("  " + shlex.join(command))
        return 0
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
