#!/usr/bin/env python3
"""Deterministic delegation boundary checks for Claude/OpenCode agent calls."""

from __future__ import annotations

import argparse
import json
import re
import sys


RECEIPT_FIELDS = {
    "idea-generator": (
        ("status", "generation_run_ids", "actions", "risk_flags"),
        ("status", "generation_run_ids", "actions"),
    ),
    "candidate-writer": (
        ("status", "candidate_path", "candidate_name", "wrote", "risk_flags", "confidence"),
        ("status", "candidate_path", "wrote"),
    ),
    "tunable-contract-extractor": (
        ("status", "train_py", "ledger_recorded", "best_warm", "trials_completed",
         "n_dims", "checks", "fixes", "risk_flags", "confidence"),
        ("status", "train_py", "ledger_recorded"),
    ),
    "tuner-orchestrator": (
        ("tuned_run_id", "selection_reason", "phase_c_method", "best_warm_score",
         "final_best_score", "trials_completed", "trials_attempted", "timeout_count",
         "elapsed_seconds", "applied", "report_path", "ledger_updated", "risks"),
        ("tuned_run_id", "ledger_updated"),
    ),
    "experience-extractor": (
        ("status", "run_dir", "records_seen", "levers", "lessons", "directions",
         "bottlenecks", "notes"),
        ("status", "records_seen"),
    ),
}

RECEIPT_ENUMS = {
    ("idea-generator", "status"): {"recorded"},
    ("candidate-writer", "status"): {"written", "existing", "blocked"},
    ("tunable-contract-extractor", "status"): {"ok", "crash"},
    ("experience-extractor", "status"): {"written", "no-data"},
}


def _field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def compact_task_result(agent: str, raw: str) -> str:
    """Reduce a child result to the agent's receipt contract.

    The rich payload already lives in run-local artifacts. This is deliberately
    deterministic: an ignored output prompt cannot inject code, diffs, logs, or
    reports back into the coordinator.
    """
    contract = RECEIPT_FIELDS.get(agent)
    if contract is None:
        return raw
    allowed, required = contract
    values: dict[str, list[str]] = {}
    pattern = re.compile(r"^\s*(?:[-*]\s*)?([A-Za-z][A-Za-z0-9 _-]{0,40}):\s*(.*?)\s*$")
    for line in raw.splitlines():
        match = pattern.match(line.strip("`"))
        if not match:
            continue
        key = _field_name(match.group(1))
        if key not in allowed:
            continue
        value = match.group(2).strip()
        if value and value not in values.setdefault(key, []):
            values[key].append(value[:500])

    child = re.search(r'<task\s+id="([^"]+)"', raw)
    missing = [key for key in required if not values.get(key)]
    invalid_values = []
    for key in allowed:
        expected = RECEIPT_ENUMS.get((agent, key))
        if expected and values.get(key) and any(value.lower() not in expected for value in values[key]):
            invalid_values.append(key)
    scope_violation = None
    if agent == "candidate-writer" and re.search(
        r"\b(?:warmstart_eval|best_warm|trials_completed|ledger_recorded|set-tuning)\b",
        raw,
        re.IGNORECASE,
    ):
        scope_violation = "candidate-writer returned evaluation/tuning evidence"
    invalid = bool(missing or invalid_values or scope_violation)
    lines = [
        f"agent: {agent}",
        f"child_session_id: {child.group(1) if child else 'unknown'}",
        f"original_chars: {len(raw)}",
        f"receipt_contract: {'invalid' if invalid else 'ok'}",
    ]
    for key in allowed:
        if values.get(key):
            joined = "; ".join(values[key])
            lines.append(f"{key}: {joined[:1000]}")
    if missing:
        lines.append(f"missing_fields: {','.join(missing)}")
    if invalid_values:
        lines.append(f"invalid_fields: {','.join(invalid_values)}")
    if scope_violation:
        lines.append(f"scope_violation: {scope_violation}")
    if invalid:
        diagnostic_lines = []
        for line in raw.splitlines():
            clean = " ".join(line.strip("` <>\t").split())
            if clean and re.search(
                r"\b(?:error|exception|failed|blocked|timeout|permission|missing)\b",
                clean,
                re.IGNORECASE,
            ):
                diagnostic_lines.append(clean[:240])
            if len(diagnostic_lines) == 3:
                break
        if diagnostic_lines:
            lines.append(f"diagnostic: {' | '.join(diagnostic_lines)}")
        lines.append("recover_from: run-local artifacts and ledger; do not request the full child output")
    return "\n".join(lines)


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencode-check", metavar="AGENT")
    parser.add_argument("--compact-result", metavar="AGENT")
    args = parser.parse_args()
    if args.compact_result is not None:
        print(compact_task_result(args.compact_result, sys.stdin.read()))
        return 0
    if args.opencode_check is not None:
        reason = delegation_violation(args.opencode_check, sys.stdin.read())
        if reason:
            print(reason)
            return 3
        return 0
    return _claude_hook()


if __name__ == "__main__":
    raise SystemExit(main())
