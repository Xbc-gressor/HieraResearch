#!/usr/bin/env python3
"""Deterministic delegation boundary checks for Claude/OpenCode agent calls."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path


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

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = (ROOT / "runs").resolve()
OFFLINE_ENV = "HIERA_RETRIEVAL_OFFLINE"
DISABLE_NATIVE_WEB_ENV = "HIERA_RETRIEVAL_DISABLE_NATIVE_WEB"
SHELL_CONTROL_RE = re.compile(r"(?:&&|\|\||[;|&<>`\n]|\$\(|\$\{)")
SANCTIONED_PYTHON_TOOLS = {
    "tools/apply_base_params.py",
    "tools/apply_search_space.py",
    "tools/background_contract.py",
    "tools/compare_runs.py",
    "tools/got_graph.py",
    "tools/got_select.py",
    "tools/init_run.py",
    "tools/ledger.py",
    "tools/new_candidate.py",
    "tools/search_backends.py",
    "tools/summarize_run.py",
    "tools/tuners/bo_search.py",
    "tools/tuners/cmaes_search.py",
    "tools/tuners/grid_search.py",
    "tools/tuners/tune_tools.py",
    "tools/tuners/warmstart_eval.py",
    "tools/validate_background.py",
    "tools/validate_claude.py",
    "tools/validate_got.py",
    "tools/validate_kimi.py",
    "tools/validate_search_backends.py",
    "tools/validate_skills.py",
    "tools/validate_tasks.py",
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


def spawn_closure_violation(agent: str, prompt: str, allowed: set[str] | None) -> str | None:
    """Enforce the orchestrator's exact spawn set when `allowed` is given.

    kimi-cli's Agent tool description always mentions the built-in coder/explore
    types in its static prose, which can invite out-of-closure spawns; deny them
    deterministically. A bare `resume` (no subagent_type) is allowed through.
    """
    if allowed is None:
        return None
    if not agent:
        if not prompt:  # resume-only call carries no spawn intent
            return None
        return (
            "Agent calls must name subagent_type explicitly (one of: "
            + ", ".join(sorted(allowed))
            + "); the built-in default would escape the role closure."
        )
    if agent not in allowed:
        return (
            f"subagent_type {agent!r} is not one of the HieraResearch role "
            f"agents ({', '.join(sorted(allowed))}); spawn only the bounded "
            "role agents declared by the orchestrator."
        )
    return None


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _run_local_path(raw: str) -> bool:
    path = Path(raw)
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(RUNS_ROOT)
    except ValueError:
        return False
    return True


def _temporary_receipt_path(raw: str) -> bool:
    path = Path(raw)
    return (
        _run_local_path(raw)
        and ".retrieval-tmp" in path.parts
        and path.suffix in {".json", ".txt"}
        and not any(char in raw for char in "*?[]")
    )


def _option_value(tokens: list[str], option: str) -> str | None:
    try:
        index = tokens.index(option)
    except ValueError:
        return None
    return tokens[index + 1] if index + 1 < len(tokens) else ""


def background_tool_decision(
    tool_name: str, tool_input: dict[str, object]
) -> tuple[str, str] | None:
    """Return an allow/deny decision for the background researcher's tools.

    This is an agent-scoped capability gate, not a general shell allowlist. The
    background researcher can call the retrieval and validation control planes,
    but cannot execute candidates, probe the machine, or create a second web
    path with curl/wget.
    """
    if tool_name in {"WebSearch", "WebFetch"}:
        if _enabled(OFFLINE_ENV) or _enabled(DISABLE_NATIVE_WEB_ENV):
            return (
                "deny",
                "native web tools are disabled for this retrieval condition; retain the "
                "coverage gap instead of bypassing the adapter condition",
            )
        return "allow", "native web is enabled for the background-research role"

    if tool_name != "Bash":
        return None
    command = str(tool_input.get("command") or "").strip()
    if not command:
        return "deny", "empty Bash calls are not part of the background-research contract"
    if SHELL_CONTROL_RE.search(command):
        return (
            "deny",
            "compound commands, redirects, substitutions, and pipelines are not allowed; "
            "invoke one sanctioned adapter command at a time",
        )
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return "deny", f"shell command could not be parsed safely: {exc}"
    if not tokens:
        return "deny", "empty Bash calls are not part of the background-research contract"

    if tokens[0] in {"python", "python3"} and len(tokens) >= 3:
        script = tokens[1]
        action = tokens[2]
        if script == "tools/search_backends.py" and action in {
            "probe",
            "search",
            "record-search",
            "visit",
            "record-visit",
            "validate",
        }:
            if action in {"record-search", "record-visit"} and (
                _enabled(OFFLINE_ENV) or _enabled(DISABLE_NATIVE_WEB_ENV)
            ):
                return (
                    "deny",
                    f"{action} is disabled because this retrieval condition forbids native web",
                )
            if action != "probe":
                manifest = _option_value(tokens, "--manifest")
                if not manifest or not _run_local_path(manifest):
                    return "deny", "retrieval manifests must be under the repository runs/ tree"
            for option in ("--results-file", "--content-file"):
                value = _option_value(tokens, option)
                if value is not None and (not value or not _temporary_receipt_path(value)):
                    return (
                        "deny",
                        f"{option} must reference a file under run-local .retrieval-tmp/",
                    )
            return "allow", f"sanctioned retrieval adapter action: {action}"
        if script == "tools/background_contract.py" and action == "validate":
            for option in ("--background", "--retrieval-manifest"):
                value = _option_value(tokens, option)
                if not value or not _run_local_path(value):
                    return "deny", f"{option} must reference a run-local artifact"
            return "allow", "sanctioned background-contract validation"

    if tokens[0] == "mkdir" and tokens[1:2] == ["-p"] and len(tokens) == 3:
        target = Path(tokens[2])
        if _run_local_path(tokens[2]) and target.name == ".retrieval-tmp":
            return "allow", "create the run-local retrieval receipt directory"
    if tokens[0] == "rm" and len(tokens) == 2 and _temporary_receipt_path(tokens[1]):
        return "allow", "delete one temporary run-local receipt input"

    return (
        "deny",
        "background-research Bash is capability-limited: use Read/Glob, WebSearch/WebFetch, "
        "or the project-local search_backends/background_contract commands; direct Python, "
        "candidate execution, environment inspection, filesystem-wide discovery, curl, and "
        "wget are outside this role",
    )


def _background_hook() -> int:
    payload = json.load(sys.stdin)
    tool_name = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input") or {}
    decision = background_tool_decision(
        tool_name, tool_input if isinstance(tool_input, dict) else {}
    )
    if decision is None:
        return 0
    behavior, reason = decision
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": behavior,
            "permissionDecisionReason": f"HieraResearch background guard: {reason}",
        }
    }))
    return 0


def _script_after_python(tokens: list[str], position: int) -> str | None:
    index = position + 1
    while index < len(tokens):
        token = tokens[index]
        if token in {"-c", "-m", "-"}:
            return None
        if token == "--":
            return tokens[index + 1] if index + 1 < len(tokens) else None
        if token in {"-W", "-X", "--check-hash-based-pycs"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return None


def _python_script(tokens: list[str]) -> str | None:
    """Return the script passed to the first direct Python invocation."""
    wrappers = {"timeout", "time", "nice", "nohup", "stdbuf"}
    index = 0
    if tokens and Path(tokens[0]).name in wrappers:
        index = 1
        while index < len(tokens) and tokens[index].startswith("-"):
            index += 1
    for position in range(index, len(tokens)):
        if Path(tokens[position]).name not in {"python", "python3"}:
            continue
        return _script_after_python(tokens, position)
    return None


def _shell_segments(command: str) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    segments: list[list[str]] = [[]]
    for token in lexer:
        if token and all(char in ";&|<>" for char in token):
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(token)
    return [segment for segment in segments if segment]


def _executes_train(tokens: list[str]) -> bool:
    for position, token in enumerate(tokens):
        if Path(token).name not in {"python", "python3"}:
            continue
        script = _script_after_python(tokens, position)
        if script is not None and Path(script).name == "train.py":
            return True
    wrappers = {"timeout", "time", "nice", "nohup", "stdbuf"}
    index = 0
    if tokens and Path(tokens[0]).name in wrappers:
        index = 1
        while index < len(tokens) and tokens[index].startswith("-"):
            index += 1
    return index < len(tokens) and Path(tokens[index]).name == "train.py"


def _segments_execute_train(segments: list[list[str]], depth: int = 0) -> bool:
    if depth > 2:
        return False
    for segment in segments:
        if _executes_train(segment):
            return True
        for index, token in enumerate(segment[:-1]):
            if Path(token).name not in {"bash", "sh", "zsh"}:
                continue
            if "c" not in segment[index + 1].lstrip("-") or index + 2 >= len(segment):
                continue
            try:
                nested = _shell_segments(segment[index + 2])
            except ValueError:
                continue
            if _segments_execute_train(nested, depth + 1):
                return True
    return False


def runtime_bash_decision(tool_input: dict[str, object]) -> tuple[str, str] | None:
    """Protect direct candidate execution and approve only named control-plane scripts."""
    command = str(tool_input.get("command") or "").strip()
    if not command:
        return None
    try:
        segments = _shell_segments(command)
    except ValueError:
        return None
    if _segments_execute_train(segments):
        return (
            "deny",
            "direct train.py execution is outside the HieraResearch evaluation contract; "
            "use tunable-contract-extractor or tuner-orchestrator through the sanctioned tools",
        )
    has_python = any(
        Path(token).name in {"python", "python3"}
        for segment in segments
        for token in segment
    )
    if len(segments) != 1 or SHELL_CONTROL_RE.search(command):
        return (
            ("ask", "compound Python commands require explicit review")
            if has_python
            else None
        )
    tokens = segments[0]
    script = _python_script(tokens)
    if script in SANCTIONED_PYTHON_TOOLS:
        return "allow", f"sanctioned HieraResearch control-plane script: {script}"
    if has_python:
        return (
            "ask",
            "this Python invocation is not a named HieraResearch control-plane script",
        )
    return None


def _runtime_bash_hook() -> int:
    payload = json.load(sys.stdin)
    if payload.get("tool_name") != "Bash":
        return 0
    tool_input = payload.get("tool_input") or {}
    decision = runtime_bash_decision(tool_input if isinstance(tool_input, dict) else {})
    if decision is None:
        return 0
    behavior, reason = decision
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": behavior,
            "permissionDecisionReason": f"HieraResearch runtime guard: {reason}",
        }
    }))
    return 0


def _claude_hook(allowed: set[str] | None = None) -> int:
    payload = json.load(sys.stdin)
    if payload.get("tool_name") != "Agent":
        return 0
    tool_input = payload.get("tool_input") or {}
    agent = str(tool_input.get("subagent_type") or "")
    prompt = str(tool_input.get("prompt") or "")
    reason = spawn_closure_violation(agent, prompt, allowed) or delegation_violation(
        agent, prompt
    )
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
    parser.add_argument(
        "--background-tools",
        action="store_true",
        help="apply the background researcher's agent-scoped tool capability policy",
    )
    parser.add_argument(
        "--runtime-bash",
        action="store_true",
        help="protect direct candidate execution and approve named harness scripts",
    )
    parser.add_argument(
        "--allowed-subagents",
        metavar="CSV",
        help="comma-separated spawn closure; deny any other subagent_type "
        "(used by the kimi-cli hook; omit for the Claude runtime)",
    )
    args = parser.parse_args()
    if args.background_tools:
        return _background_hook()
    if args.runtime_bash:
        return _runtime_bash_hook()
    if args.compact_result is not None:
        print(compact_task_result(args.compact_result, sys.stdin.read()))
        return 0
    if args.opencode_check is not None:
        reason = delegation_violation(args.opencode_check, sys.stdin.read())
        if reason:
            print(reason)
            return 3
        return 0
    allowed = (
        {name.strip() for name in args.allowed_subagents.split(",") if name.strip()}
        if args.allowed_subagents
        else None
    )
    return _claude_hook(allowed)


if __name__ == "__main__":
    raise SystemExit(main())
