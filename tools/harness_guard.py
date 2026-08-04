#!/usr/bin/env python3
"""Deterministic delegation boundary checks for Claude agent calls."""

from __future__ import annotations

import argparse
import json
import sys


def delegation_violation(agent: str, prompt: str) -> str | None:
    """Return a narrow, actionable violation or None.

    The checks intentionally target observed positive assignments, not generic
    words such as "evaluate" that may occur in a negative boundary reminder.
    """
    normalized = " ".join(prompt.lower().split())
    if agent == "candidate-writer" and (
        "also perform step 0+1" in normalized
        or "then also perform step 0+1" in normalized
        or ("after writing" in normalized and "perform step 0+1" in normalized)
        or ("also perform" in normalized and "warm-start" in normalized)
    ):
        return (
            "candidate-writer may only create train.py. Spawn "
            "tunable-contract-extractor separately for step 0+1; an evaluation "
            "budget is not permission to collapse agent roles."
        )
    return None


def _claude_hook() -> int:
    payload = json.load(sys.stdin)
    if payload.get("tool_name") != "Agent":
        return 0
    tool_input = payload.get("tool_input") or {}
    agent = str(tool_input.get("subagent_type") or "")
    prompt = str(tool_input.get("prompt") or "")
    reason = delegation_violation(agent, prompt)
    if reason is None:
        return 0
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"HieraResearch delegation guard: {reason}",
        }
    }))
    return 0


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    return _claude_hook()


if __name__ == "__main__":
    raise SystemExit(main())
