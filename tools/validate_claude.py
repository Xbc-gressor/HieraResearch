#!/usr/bin/env python3
"""Offline contract checks for the Claude agent and retrieval launch surface."""

from __future__ import annotations

import os
from pathlib import Path

from harness_guard import background_tool_decision, runtime_bash_decision
from run_background import (
    DISABLED_ENV,
    DISABLE_NATIVE_WEB_ENV,
    OFFLINE_ENV,
    build_launch,
)


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = ROOT / ".claude" / "agents" / "background-researcher.md"
SETTINGS = ROOT / ".claude" / "settings.json"


def main() -> int:
    errors: list[str] = []
    text = BACKGROUND.read_text()
    for required in (
        "tools/search_backends.py` is the sole interface to DeepXiv",
        "python tools/search_backends.py probe --backend deepxiv",
        "python tools/search_backends.py record-search",
        'matcher: "Bash|WebSearch|WebFetch"',
        '"--background-tools"',
        "HIERA_RETRIEVAL_OFFLINE=1",
    ):
        if required not in text:
            errors.append(f"background-researcher is missing {required!r}")
    if "--runtime-bash" not in SETTINGS.read_text():
        errors.append("Claude project settings do not install the runtime Bash guard")

    safe = background_tool_decision(
        "Bash",
        {
            "command": "python tools/search_backends.py probe --backend deepxiv",
        },
    )
    if not safe or safe[0] != "allow":
        errors.append(f"sanctioned DeepXiv probe was not allowed: {safe}")

    safe_search = background_tool_decision(
        "Bash",
        {
            "command": (
                "python tools/search_backends.py search "
                "--manifest runs/tabular-blind/probe/background_retrieval.json "
                "--backend deepxiv --query trees"
            )
        },
    )
    if not safe_search or safe_search[0] != "allow":
        errors.append(f"sanctioned run-local search was not allowed: {safe_search}")

    unsafe_commands = (
        "python train.py",
        "python -c 'import deepxiv_sdk'",
        "which deepxiv-cli; find / -iname '*deepxiv*' | head",
        "curl https://arxiv.org/",
        "python tools/search_backends.py validate --manifest /tmp/outside.json",
    )
    for command in unsafe_commands:
        decision = background_tool_decision("Bash", {"command": command})
        if not decision or decision[0] != "deny":
            errors.append(f"unsafe background command was not denied: {command!r}: {decision}")

    direct_train = runtime_bash_decision(
        {"command": "python runs/tabular-blind/probe/candidates/001/train.py"}
    )
    if not direct_train or direct_train[0] != "deny":
        errors.append(f"direct candidate execution was not denied: {direct_train}")
    optimized_train = runtime_bash_decision(
        {"command": "python -OO runs/tabular-blind/probe/candidates/001/train.py"}
    )
    if not optimized_train or optimized_train[0] != "deny":
        errors.append(f"flagged direct candidate execution was not denied: {optimized_train}")
    compound_train = runtime_bash_decision(
        {
            "command": (
                "python tools/validate_tasks.py && "
                "python runs/tabular-blind/probe/candidates/001/train.py"
            )
        }
    )
    if not compound_train or compound_train[0] != "deny":
        errors.append(f"compound direct candidate execution was not denied: {compound_train}")
    wrapped_train = runtime_bash_decision(
        {
            "command": (
                "python tools/timed_run.py 60 python "
                "runs/tabular-blind/probe/candidates/001/train.py"
            )
        }
    )
    if not wrapped_train or wrapped_train[0] != "deny":
        errors.append(f"wrapped direct candidate execution was not denied: {wrapped_train}")
    shell_wrapped_train = runtime_bash_decision(
        {
            "command": (
                "bash -lc 'python runs/tabular-blind/probe/candidates/001/train.py'"
            )
        }
    )
    if not shell_wrapped_train or shell_wrapped_train[0] != "deny":
        errors.append(f"shell-wrapped candidate execution was not denied: {shell_wrapped_train}")
    indirect_train = runtime_bash_decision(
        {
            "command": (
                "python tools/tuners/tune_tools.py lint-schema --candidate-path "
                "runs/tabular-blind/probe/candidates/001/train.py"
            )
        }
    )
    if not indirect_train or indirect_train[0] != "allow":
        errors.append(f"sanctioned candidate inspection was not allowed: {indirect_train}")
    arbitrary_python = runtime_bash_decision({"command": "python unknown_script.py"})
    if not arbitrary_python or arbitrary_python[0] != "ask":
        errors.append(f"arbitrary Python did not require review: {arbitrary_python}")
    interpreter_snippet = runtime_bash_decision({"command": "python -c 'print(1)'"})
    if not interpreter_snippet or interpreter_snippet[0] != "ask":
        errors.append(f"Python -c did not require review: {interpreter_snippet}")

    old_offline = os.environ.get(OFFLINE_ENV)
    old_native_disable = os.environ.get(DISABLE_NATIVE_WEB_ENV)
    try:
        os.environ.pop(OFFLINE_ENV, None)
        os.environ.pop(DISABLE_NATIVE_WEB_ENV, None)
        native = background_tool_decision("WebSearch", {"query": "balanced forests"})
        if not native or native[0] != "allow":
            errors.append(f"enabled native WebSearch was not allowed: {native}")
        os.environ[OFFLINE_ENV] = "1"
        native = background_tool_decision("WebSearch", {"query": "balanced forests"})
        if not native or native[0] != "deny":
            errors.append(f"offline native WebSearch was not denied: {native}")
        receipt = background_tool_decision(
            "Bash",
            {
                "command": (
                    "python tools/search_backends.py record-search "
                    "--manifest runs/tabular-blind/probe/background_retrieval.json "
                    "--query trees --results-file "
                    "runs/tabular-blind/probe/.retrieval-tmp/results.json"
                )
            },
        )
        if not receipt or receipt[0] != "deny":
            errors.append(f"offline native search receipt was not denied: {receipt}")
        dangerous_delete = background_tool_decision(
            "Bash", {"command": "rm runs/tabular-blind/probe/background_retrieval.json"}
        )
        if not dangerous_delete or dangerous_delete[0] != "deny":
            errors.append(f"run artifact deletion was not denied: {dangerous_delete}")
    finally:
        if old_offline is None:
            os.environ.pop(OFFLINE_ENV, None)
        else:
            os.environ[OFFLINE_ENV] = old_offline
        if old_native_disable is None:
            os.environ.pop(DISABLE_NATIVE_WEB_ENV, None)
        else:
            os.environ[DISABLE_NATIVE_WEB_ENV] = old_native_disable

    poisoned = {
        OFFLINE_ENV: "1",
        DISABLED_ENV: "deepxiv",
        DISABLE_NATIVE_WEB_ENV: "1",
    }
    full_command, full_env, _ = build_launch(
        "tabular-blind", "probe", "full", None, base_env=poisoned
    )
    _, no_deepxiv_env, _ = build_launch(
        "tabular-blind", "probe", "no-deepxiv", None, base_env=poisoned
    )
    no_native_command, no_native_env, _ = build_launch(
        "tabular-blind", "probe", "no-native-web", None, base_env=poisoned
    )
    if any(name in full_env for name in (OFFLINE_ENV, DISABLED_ENV, DISABLE_NATIVE_WEB_ENV)):
        errors.append("full launch retained inherited ablation controls")
    if set(no_deepxiv_env.get(DISABLED_ENV, "").split(",")) != {
        "deepxiv",
        "direct",
        "jina",
    }:
        errors.append("no-deepxiv launch has the wrong disabled adapters")
    if no_native_env.get(DISABLE_NATIVE_WEB_ENV) != "1":
        errors.append("no-native-web launch did not disable Claude native web tools")
    if "deepxiv" in no_native_env.get(DISABLED_ENV, "").split(","):
        errors.append("no-native-web launch accidentally disabled DeepXiv")
    if "--allowedTools=WebSearch,WebFetch" not in full_command:
        errors.append("full launch does not explicitly allow Claude native web tools")
    if "--disallowedTools=WebSearch,WebFetch" not in no_native_command:
        errors.append("no-native-web launch does not explicitly deny Claude native web tools")
    if not full_command[-1].startswith("Run background research only for runs/"):
        errors.append("Claude launcher prompt was consumed or misplaced")

    if errors:
        print("Claude integration validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "Validated Claude background adapter boundary, safe tool gate, native-web "
        "condition controls, and retrieval launcher."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
