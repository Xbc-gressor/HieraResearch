"""Session layer: one role invocation = one SDK session + verify-repair loop.

The loop: run the session → check the accepted receipt + postconditions →
corrective follow-up in the SAME session with concrete diagnostics → repeat
up to role.corrective_attempts → InvocationFailed. Escalation beyond that
is the loops' job (role-specific, see spec Error handling).

A drain is interrupted as soon as a schema-valid receipt is accepted: the
receipt is the role's final artifact, so post-acceptance turns can only be
waste (observed: models rambling or resubmitting for 10-30 extra turns).
Corrective follow-ups may still resubmit (latest wins, see receipts.py).
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import time
from collections import Counter

import anyio
from pathlib import Path
from typing import Callable, Protocol

from .events import EventsLog
from .receipts import ReceiptStore, build_receipt_server
from .roles import (
    PROMPT_DIR,
    REPO_ROOT,
    InvocationContext,
    RoleDefinition,
    driver_job_handoff_problem,
)
from tools.evaluation_budget import find_run_dir, time_budget
from tools.task_contract import contract_context

RECEIPT_TOOL = "mcp__receipts__submit_receipt"

# Every session is bounded by the run's single cutoff, deadline − final_reserve
# (evaluation_budget.time_budget["usable_seconds"]): no session starts past
# it, and a running one is cancelled at it regardless of whether the SDK is
# streaming messages. The whole reserve stays free for submission export.
TIME_REACHED_PROBLEM = "time budget reached"

# Frozen failure-signature prefixes: the ONLY machine-readable problem
# taxonomy consumed by run-level failure escalation (consecutive-seat-failure
# trip, PLAN-block-escalation Wave 0/1). Order matters — the first matching
# prefix wins, so the specific error-result subtypes must precede the generic
# transport catch-all. New problem classes must be registered here before any
# consumer classifies them; raw string matching at consumers is forbidden.
# Role-postcondition strings deliberately carry no class of their own: they
# fall through to "postcondition" and isomorphism is judged by role name.
PROBLEM_CLASS_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("repetition_breaker", ("repetition breaker tripped:",)),
    ("max_turns",
     ("session ended with error result: error_max_turns",)),
    ("idle_timeout",
     ("session ended with error result: idle_timeout",)),
    ("wall_limit", ("wall-clock limit",)),
    ("time_budget", (TIME_REACHED_PROBLEM,)),
    # First-turn error results that are not a bounded-role cutoff (API 402 at
    # start, relay transients) share the generic error-result prefix.
    ("transport", ("session ended with error result:",)),
)


def problem_class(problem: str) -> str:
    """Classify one InvocationFailed problem string into the frozen enum."""
    text = str(problem)
    for name, prefixes in PROBLEM_CLASS_PREFIXES:
        if any(text.startswith(prefix) for prefix in prefixes):
            return name
    return "postcondition"


def invocation_problem_class(problems: list[str]) -> str:
    """The class of an invocation's problem list: the first classified
    non-postcondition problem wins (session-level cutoffs outrank the
    "no accepted receipt" postcondition that accompanies them)."""
    classes = [problem_class(str(problem)) for problem in problems]
    for name in classes:
        if name != "postcondition":
            return name
    return classes[0] if classes else "postcondition"

# One "submit what you have" turn for soft-rescue roles at their wall limit.
SOFT_RESCUE_GRACE_SECONDS = 180.0
SOFT_RESCUE_MESSAGE = ("Wall-clock limit reached. Submit your current best "
                       f"result immediately by calling {RECEIPT_TOOL}.")
# Transport-class failures (API 402 at start, relay "error result: success")
# end the session on its first turn with nothing done; they are retried with
# the same context (resume id intact) instead of spending a corrective.
TRANSPORT_BACKOFF_SECONDS = (5.0, 15.0, 30.0)
TRANSPORT_RETRIES = 3

# Relay key rotation: ANTHROPIC_AUTH_TOKEN is the active key and
# ANTHROPIC_AUTH_TOKEN_BACKUPS a comma-separated standby list. These
# api_error_status values are key/quota-class failures where swapping the key
# can succeed while backoff alone cannot (e.g. a relay's 402/503 pool
# exhaustion); rotation skips the dead key for the rest of the process.
AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
AUTH_TOKEN_BACKUPS_ENV = "ANTHROPIC_AUTH_TOKEN_BACKUPS"
KEY_ROTATION_STATUSES = frozenset({401, 402, 403, 503})


class AuthTokenPool:
    """Process-wide relay key state: the active key plus backups. Rotation
    never logs token material; ``index`` identifies the key in events."""

    def __init__(self) -> None:
        self.index = 0

    def tokens(self) -> list[str]:
        primary = os.environ.get(AUTH_TOKEN_ENV, "").strip()
        backups = (t.strip()
                   for t in os.environ.get(AUTH_TOKEN_BACKUPS_ENV, "").split(","))
        return list(dict.fromkeys(t for t in (primary, *backups) if t))

    def active(self) -> str | None:
        tokens = self.tokens()
        if not tokens:
            return None
        return tokens[min(self.index, len(tokens) - 1)]

    def rotate(self) -> bool:
        if self.index + 1 < len(self.tokens()):
            self.index += 1
            return True
        return False


AUTH_TOKEN_POOL = AuthTokenPool()

# Consecutive identical (tool, input) calls before the repetition breaker
# trips. Observed incident: a model retried one hallucinated Edit verbatim
# 880+ times until the context overflowed. Healthy retries re-read the file
# or change the input, so the fingerprint necessarily changes.
REPETITION_LIMIT = 5

# Bounded transcript dump (diagnostics only), written for every session.
# Per-message and whole-buffer caps keep the JSONL small even for
# max_turns-size sessions.
_DUMP_MSG_CHARS = 4000

# The ledger has exactly one mutator (tools/ledger.py). Sessions read it
# freely; any direct write path is denied at the tool layer, so "agents never
# hand-edit the ledger" stops being a prompt convention.
_LEDGER_FILENAME = "ledger.json"
_FILE_EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
_BASH_WRITE_MARKERS = (">", "sed -i", "tee ", "python -c", "python3 -c",
                       "truncate ", "mv ", "cp ", "rm ")


def _ledger_write_problem(name: str, tool_input: dict | None) -> str | None:
    tool_input = tool_input or {}
    if name in _FILE_EDIT_TOOLS:
        target = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        if Path(target).name == _LEDGER_FILENAME:
            return (f"{name} on {_LEDGER_FILENAME} is not allowed: the ledger "
                    "is written only through `python tools/ledger.py` "
                    "subcommands")
        return None
    if name == "Bash":
        command = str(tool_input.get("command") or "")
        if _LEDGER_FILENAME in command \
                and not command.lstrip().startswith(("python tools/ledger.py",
                                                     "python3 tools/ledger.py")) \
                and any(marker in command for marker in _BASH_WRITE_MARKERS):
            return (f"Bash may not write {_LEDGER_FILENAME}: the ledger is "
                    "written only through `python tools/ledger.py` subcommands")
    return None


# Retrieval-adapter confinement for bash_adapter_only roles (the background
# researcher): Bash runs exactly one allowlisted adapter invocation — no
# operators, substitution, or redirects — and --manifest stays inside the
# run's own directory so a sibling run's record cannot be replayed.
_PUNCTUATION_CHARS = "|&;()<>"
_ADAPTER_ALLOWLIST = {
    "tools/search_backends.py": ("search", "status", "results", "visit",
                                 "read", "validate"),
    "tools/background_contract.py": ("catalog", "normalize", "validate"),
}
_ADAPTER_USAGE = ("run a single adapter command: `python "
                  "tools/search_backends.py search|status|results|visit|read|"
                  "validate ...` or `python tools/background_contract.py "
                  "catalog|normalize|validate ...`")


def _adapter_tokens(command: str) -> list[str] | None:
    """Tokenize an adapter command, returning None for shell syntax errors."""
    lexer = shlex.shlex(command, posix=True,
                        punctuation_chars=_PUNCTUATION_CHARS)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return None
    return tokens


def _adapter_poll_command(command: str, run_dir: Path | None) -> bool:
    """Whether *command* is a valid status/results adapter poll."""
    tokens = _adapter_tokens(command)
    if tokens is None or len(tokens) < 3:
        return False
    if tokens[0] not in ("python", "python3"):
        return False
    if tokens[2] not in ("status", "results"):
        return False
    if tokens[2] not in _ADAPTER_ALLOWLIST.get(tokens[1], ()):
        return False
    # Reuse the complete confinement/verdict path so an escaped manifest is
    # never classified as a legitimate poll.
    return adapter_command_verdict(command, Path(run_dir or ".")) is None


def adapter_command_verdict(command: str, run_dir: Path) -> str | None:
    """None when the command is one allowlisted adapter invocation, else the
    deny reason. Quoted arguments tokenize to a single word, so JSON payloads
    carrying shell metacharacters stay legal."""
    tokens = _adapter_tokens(command)
    if tokens is None:
        return f"unparseable command; {_ADAPTER_USAGE}"
    if not tokens:
        return f"empty command; {_ADAPTER_USAGE}"
    for token in tokens:
        if all(char in _PUNCTUATION_CHARS for char in token):
            return (f"shell operator {token!r} is not allowed — no pipes, "
                    f"compound commands, subshells, or redirects; "
                    f"{_ADAPTER_USAGE}")
        if "`" in token or token.startswith("$("):
            return f"command substitution is not allowed; {_ADAPTER_USAGE}"
    if len(tokens) < 3 or tokens[0] not in ("python", "python3") \
            or tokens[2] not in _ADAPTER_ALLOWLIST.get(tokens[1], ()):
        return f"off the adapter allowlist; {_ADAPTER_USAGE}"
    for index, token in enumerate(tokens):
        flag, _, value = token.partition("=")
        if flag not in ("--manifest", "--background"):
            continue
        if not value:
            if index + 1 >= len(tokens):
                return f"{flag} requires a value"
            value = tokens[index + 1]
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = Path(run_dir) / candidate
        if not candidate.resolve().is_relative_to(Path(run_dir).resolve()):
            return (f"{flag} must stay inside this run's directory "
                    f"({run_dir}); {_ADAPTER_USAGE}")
    return None


_DUMP_MAX_ENTRIES = 200
_DUMP_MAX_BYTES = 256 * 1024


_REPEAT_HISTORY_LIMIT = 10
_WINDOW_REPEAT_LIMIT = 4
# Consecutive tool calls that only revisit earlier (tool, input, revision)
# fingerprints of this invocation. Liveness is judged by novelty: a new
# input, a Read of a changed file, or any Edit/Write is progress.
_STALL_LIMIT = 8
# The same trip over a window, so sparse novel calls cannot reset it:
# _STALL_WINDOW_REVISITS revisits among the last _STALL_WINDOW calls.
_STALL_WINDOW = 20
_STALL_WINDOW_REVISITS = 16
# Reads of one unchanged file beyond this count are revisits whatever their
# offset/limit, so paging through the same content is not novelty.
_READ_REVISIT_AFTER = 8
# Every PreToolUse guard: the invariant it protects, who consumes that
# invariant, its consequence class (hard | repairable | advisory; see the
# design's guard discipline) and the legal exit it offers. guard_denied /
# guard_degraded events carry the id; a guard that blocks a run is the
# first suspect.
GUARDS = {
    "capability": ("role tool set", "role isolation", "hard",
                   "the role's available tools"),
    "ledger_write": ("only tools/ledger.py mutates ledger.json",
                     "ledger integrity", "hard", "tools/ledger.py"),
    "bash_patterns": ("role Bash prefix allowlist", "role scope",
                      "repairable", "allowed prefixes; file tools"),
    "driver_job": ("long objective work is driver-owned",
                   "GPU lease and evaluation accounting", "repairable",
                   "a driver_job receipt"),
    "bash_adapter": ("retrieval goes through the adapter",
                     "retrieval compliance", "hard",
                     "adapter commands (benign shell forms rewritten); "
                     "file tools"),
    "repetition_breaker": ("the session makes progress", "liveness",
                           "repairable", "fresh session with a handoff"),
    "repetition_hint": ("the session makes progress", "liveness",
                        "advisory", "the call proceeds with a hint"),
}
COMPACTION_HINT = (
    "Your context was just compacted and its summary may be inaccurate. "
    "Re-read a file before editing it; the invocation context in your "
    "system prompt (paths, ids) is authoritative.")


def _read_revision(name: str, tool_input: dict | None) -> tuple | None:
    """The (path, mtime) a Read observes, so a Read after an edit is new."""
    if name != "Read" or not isinstance(tool_input, dict):
        return None
    raw_path = tool_input.get("file_path")
    if not raw_path:
        return None
    path = str(Path(str(raw_path)).expanduser().resolve(strict=False))
    try:
        mtime = Path(path).stat().st_mtime_ns
    except OSError:
        mtime = None
    return (path, mtime)


def _strip_benign_shell(command: str, run_dir: Path) -> str | None:
    """An equivalent adapter command without trailing benign shell forms
    (``2>&1``, ``2>/dev/null``, ``| head/tail [-n N]``), or None when the
    command is not one allowlisted adapter call once they are removed."""
    tokens = _adapter_tokens(command)
    if not tokens:
        return None
    stripped = list(tokens)
    while True:
        if stripped[-3:] == ["2", ">&", "1"] or stripped[-3:] == ["2", ">", "/dev/null"]:
            del stripped[-3:]
            continue
        pipe = max((i for i, t in enumerate(stripped) if t == "|"), default=-1)
        tail = stripped[pipe + 1:] if pipe >= 0 else []
        if tail and tail[0] in ("head", "tail") and (
                tail[1:] == []
                or (len(tail) == 2 and tail[1].startswith("-")
                    and tail[1].lstrip("-n").isdigit())
                or (len(tail) == 3 and tail[1] == "-n" and tail[2].isdigit())):
            del stripped[pipe:]
            continue
        break
    if stripped == tokens:
        return None
    rewritten = shlex.join(stripped)
    return rewritten if adapter_command_verdict(rewritten, run_dir) is None else None


def new_breaker() -> dict:
    return {
        "fp": None,
        "count": 0,
        "tripped": None,
        # PreToolUse sees denied intentions too. Keep the bounded history in
        # the invocation breaker so a model cannot evade the window by
        # repeating a call that was refused.
        "history": [],
        "seen": set(),
        "stall": 0,
        "revisits": [],
        "reads": Counter(),
        # Set by the PreCompact hook; the next tool call carries the hint.
        "compacted": False,
    }


def _jsonish(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _clip(text: str) -> str:
    if len(text) <= _DUMP_MSG_CHARS:
        return text
    return text[:_DUMP_MSG_CHARS] + "…[truncated]"


def _summarize_message(msg) -> dict | None:
    """One compact JSON-able row per streamed message; None when the message
    carries no diagnostic content (the init handshake). Assistant text, tool
    inputs, and tool results all serialize through .content; the final
    ResultMessage's text lands via .result."""
    if getattr(msg, "subtype", None) == "init":
        return None
    row: dict = {"msg": type(msg).__name__}
    content = getattr(msg, "content", None)
    if content is not None:
        row["content"] = _clip(_jsonish(content))
    result = getattr(msg, "result", None)
    if result:
        row["result"] = _clip(_jsonish(result))
    return row if len(row) > 1 else None


class _BoundedTranscript:
    """Invocation-scoped, size-capped buffer of streamed session messages.

    Persisted for every session (2026-09-16: the failed slate-plan-writer
    sessions left 107B stubs and no transcript; the notune3 extractor
    sessions that hand-edited the ledger ended "successfully" and left no
    transcript either — both blind spots need the same dump).
    """

    def __init__(self) -> None:
        self.entries: list[dict] = []
        self.dropped = 0
        self._bytes = 0

    def add(self, msg) -> None:
        if len(self.entries) >= _DUMP_MAX_ENTRIES:
            self.dropped += 1
            return
        entry = _summarize_message(msg)
        if entry is None:
            return
        size = len(json.dumps(entry, ensure_ascii=False))
        if self._bytes + size > _DUMP_MAX_BYTES:
            self.dropped += 1
            return
        self.entries.append(entry)
        self._bytes += size

    def rows(self) -> list[dict]:
        rows = list(self.entries)
        if self.dropped:
            rows.append({"truncated": True, "dropped_messages": self.dropped})
        return rows


class InvocationFailed(Exception):
    def __init__(self, role: str, problems: list[str], *, invocation_id: int | None = None):
        self.role = role
        self.problems = problems
        self.invocation_id = invocation_id
        super().__init__(f"{role}: " + "; ".join(problems))


class _TransportFailure(Exception):
    def __init__(self, problems: list[str], api_error_status=None):
        self.problems = problems
        self.api_error_status = api_error_status
        super().__init__("; ".join(problems))


def budget_run_dir(ctx: InvocationContext) -> Path:
    """The run whose deadline bounds this session: the explicit binding, the
    session directory itself when it is a run, else the enclosing run
    (per-candidate session directories such as candidates/<id>/_hebo_llm)."""
    if ctx.budget_run_dir is not None:
        return Path(ctx.budget_run_dir)
    run_dir = Path(ctx.run_dir)
    if (run_dir / "framework_cfg.json").is_file():
        return run_dir
    return find_run_dir(run_dir) or run_dir


def session_time_budget(ctx: InvocationContext) -> dict:
    view = time_budget(budget_run_dir(ctx))
    if ctx.deadline_epoch is not None:
        remaining = ctx.deadline_epoch-time.time()
        usable = view.get("usable_seconds")
        view["usable_seconds"] = remaining if usable is None else min(usable, remaining)
        view["time_reached"] = view["usable_seconds"] <= 0
    return view


def admit_session(role: RoleDefinition, ctx: InvocationContext,
                  events: EventsLog | None = None) -> InvocationContext:
    """The shared session entry's budget gate: refuse past the cutoff, else
    return the context with the remaining usable seconds injected (rendered
    into the user message as ``time_budget_remaining_seconds``)."""
    budget = session_time_budget(ctx)
    if budget["time_reached"]:
        if events is not None:
            events.emit("session_refused", role=role.name,
                        invocation_id=ctx.invocation_id, reason="time_reached")
        raise InvocationFailed(role.name, [TIME_REACHED_PROBLEM],
                               invocation_id=ctx.invocation_id)
    extra = {**ctx.extra, **contract_context(budget_run_dir(ctx))}
    usable = budget.get("usable_seconds")
    if usable is not None:
        extra["time_budget_remaining_seconds"] = int(max(0.0, usable))
    return dataclasses.replace(ctx, extra=extra)


def _has_tool_use(msg) -> bool:
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return False
    return any(type(block).__name__ == "ToolUseBlock"
               or (isinstance(block, dict) and block.get("type") == "tool_use")
               for block in content)


class SessionRunner(Protocol):
    def run(self, role: RoleDefinition, ctx: InvocationContext) -> dict:
        """Return the accepted receipt payload; raise InvocationFailed."""
        ...


class SDKSessionRunner:
    def __init__(
        self,
        model: str,
        events: EventsLog,
        cli_path: str | None = None,
        client_factory: Callable | None = None,
        token_pool: AuthTokenPool | None = None,
    ):
        self.model = model
        self.events = events
        self.cli_path = cli_path
        self._client_factory = client_factory
        self._token_pool = token_pool or AUTH_TOKEN_POOL

    # -- public -------------------------------------------------------------

    def run(self, role: RoleDefinition, ctx: InvocationContext) -> dict:
        ctx = admit_session(role, ctx, self.events)
        attempt = 0
        restarted = False
        while True:
            try:
                return anyio.run(self._run_async, role, ctx)
            except InvocationFailed as exc:
                # A tripped session's context is degenerate; one fresh
                # session with a driver-built handoff is the role-level
                # recovery. On-disk work persists across the restart.
                trip = next((p for p in exc.problems
                             if p.startswith("repetition breaker tripped")), None)
                if trip is None or restarted \
                        or session_time_budget(ctx)["time_reached"]:
                    raise
                restarted = True
                self.events.emit("recovery_choice", layer="role", role=role.name,
                                 invocation_id=ctx.invocation_id,
                                 signature="repetition_breaker",
                                 frontier=["fresh_session", "abandon"],
                                 choice="fresh_session", chooser="default")
                ctx = admit_session(role, dataclasses.replace(
                    ctx, resume_session_id=None, extra={
                        **ctx.extra,
                        "restart_handoff": (
                            f"a previous session of this invocation was stopped "
                            f"({'; '.join(exc.problems)}). Files it wrote persist; "
                            "inspect the current on-disk state once, then finish "
                            "the task without repeating earlier tool calls."),
                    }), self.events)
            except _TransportFailure as exc:
                if (exc.api_error_status in KEY_ROTATION_STATUSES
                        and self._token_pool.rotate()):
                    # Key/quota-class failure: swap to a backup key and retry
                    # at once — no backoff, and the transport retries stay
                    # unspent for genuine transients on the new key.
                    self.events.emit("api_key_fallback", role=role.name,
                                     invocation_id=ctx.invocation_id,
                                     api_error_status=exc.api_error_status,
                                     key_index=self._token_pool.index)
                    continue
                if attempt >= TRANSPORT_RETRIES:
                    raise InvocationFailed(role.name, exc.problems,
                                           invocation_id=ctx.invocation_id)
                backoff = TRANSPORT_BACKOFF_SECONDS[
                    min(attempt, len(TRANSPORT_BACKOFF_SECONDS) - 1)]
                usable = session_time_budget(ctx).get("usable_seconds")
                if usable is not None and usable <= backoff:
                    raise InvocationFailed(
                        role.name, [TIME_REACHED_PROBLEM, *exc.problems],
                        invocation_id=ctx.invocation_id)
                attempt += 1
                self.events.emit("transport_retry", role=role.name,
                                 invocation_id=ctx.invocation_id,
                                 attempt=attempt,
                                 api_error_status=exc.api_error_status,
                                 backoff_seconds=backoff)
                self._sleep(backoff)

    _sleep = staticmethod(time.sleep)

    # -- internals ------------------------------------------------------------

    def _capability_hook(self, role: RoleDefinition, breaker: dict | None = None,
                         run_dir: Path | None = None):
        """Fail-closed: deny every tool outside the role's positive set.

        A PreToolUse hook (not canUseTool — that callback is shadowed under
        bypassPermissions and never reached; hooks still run). When the role
        declares bash_patterns, Bash commands must also start with one of
        those prefixes; bash_adapter_only instead confines Bash to the
        retrieval-adapter allowlist (adapter_command_verdict): benign trailing
        shell forms are rewritten away via updatedInput, other refusals name
        the tools that do the job. Every deny names a legal alternative and
        emits guard_denied with its GUARDS id.

        Repetition: the strict consecutive breaker and the novelty stall
        (``_STALL_LIMIT`` consecutive, or ``_STALL_WINDOW_REVISITS`` of the
        last ``_STALL_WINDOW`` calls that only revisit earlier fingerprints)
        trip the invocation; early-repeat and window repeats only attach a
        hint (additionalContext) and the call proceeds.
        """
        if breaker is None:
            breaker = new_breaker()
        allowed = set(role.tools) | {RECEIPT_TOOL}
        adapter_run_dir = Path(run_dir or ".")
        alternatives = "; ".join(
            text for tool, text in (
                ("Glob", "list files with Glob"),
                ("Read", "read files with Read"),
                ("Edit", "modify files with Edit"),
                ("Write", "write files with Write"),
            ) if tool in allowed)

        def deny(guard: str, reason: str, name: str, context,
                 **fields) -> dict:
            self.events.emit("guard_denied", guard=guard,
                             cls=GUARDS[guard][2], role=role.name,
                             invocation_id=context.get("invocation_id")
                             if isinstance(context, dict) else None,
                             tool=name, reason=reason, **fields)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }

        def allow(hints: list[str], updated_input: dict | None = None) -> dict:
            if not hints and updated_input is None:
                return {}
            output = {"hookEventName": "PreToolUse"}
            if updated_input is not None:
                output["permissionDecision"] = "allow"
                output["updatedInput"] = updated_input
            if hints:
                output["additionalContext"] = "\n".join(hints)
            return {"hookSpecificOutput": output}

        def emit_trip(name: str, context) -> dict:
            reason = f"repetition breaker tripped: {breaker['tripped']}"
            return deny("repetition_breaker", reason, name, context)

        async def hook(input_data, tool_use_id, context):
            name = input_data.get("tool_name", "")
            tool_input = input_data.get("tool_input")
            hints: list[str] = []
            if breaker["compacted"]:
                # Re-reading after compaction is legitimate, not a loop.
                breaker["compacted"] = False
                breaker["history"].clear()
                breaker["seen"].clear()
                breaker["stall"] = 0
                breaker["revisits"].clear()
                breaker["reads"].clear()
                hints.append(COMPACTION_HINT)
            # All repetition layers see the same content revision: a Read
            # after a real edit is not a repetition of the earlier result.
            fp = (name, json.dumps(tool_input, sort_keys=True, default=str),
                  _read_revision(name, tool_input))
            if fp == breaker["fp"]:
                breaker["count"] += 1
            else:
                breaker["fp"], breaker["count"] = fp, 1

            history = breaker["history"]
            history.append(fp)
            del history[:-_REPEAT_HISTORY_LIMIT]
            window_count = Counter(history)[fp]

            # Adapter status/results are intentionally poll-like: the same
            # command can be issued after a round changes without changing
            # its input. Keep the strict consecutive breaker, but neither
            # the window hint nor the novelty stall applies.
            adapter_poll = (
                name == "Bash" and role.bash_adapter_only
                and _adapter_poll_command((tool_input or {}).get("command", ""),
                                          adapter_run_dir)
            )
            if not adapter_poll:
                revisit = fp in breaker["seen"]
                breaker["seen"].add(fp)
                if fp[2] is not None:
                    breaker["reads"][fp[2]] += 1
                    revisit = revisit or breaker["reads"][fp[2]] > _READ_REVISIT_AFTER
                breaker["stall"] = breaker["stall"] + 1 if revisit else 0
                breaker["revisits"].append(revisit)
                del breaker["revisits"][:-_STALL_WINDOW]

            if breaker["tripped"] is None and breaker["count"] >= REPETITION_LIMIT:
                breaker["tripped"] = (f"{fp[0]} invoked {breaker['count']}x "
                                      "with identical input")
            if breaker["tripped"] is None and breaker["stall"] >= _STALL_LIMIT:
                breaker["tripped"] = (f"no progress: {breaker['stall']} "
                                      "consecutive calls repeated earlier calls")
            if breaker["tripped"] is None \
                    and sum(breaker["revisits"]) >= _STALL_WINDOW_REVISITS:
                breaker["tripped"] = (f"no progress: {sum(breaker['revisits'])} "
                                      f"of the last {_STALL_WINDOW} calls "
                                      "repeated earlier calls")
            if breaker["tripped"] is not None:
                return emit_trip(name, context)

            repeat_hint = None
            if role.early_repeat_correct and breaker["count"] >= 2:
                repeat_hint = (
                    f"repeated identical call #{breaker['count']} to {fp[0]} "
                    "with the same input; the earlier result is already in "
                    "your context — proceed to produce your receipt/output.")
            elif not adapter_poll and window_count >= _WINDOW_REPEAT_LIMIT:
                repeat_hint = (
                    f"repeated call to {fp[0]} appears {window_count} times "
                    f"in the last {_REPEAT_HISTORY_LIMIT} tool calls; use the "
                    "context already obtained and proceed to produce your "
                    "receipt/output.")
            if repeat_hint is not None:
                hints.append(repeat_hint)
                self.events.emit("guard_degraded", guard="repetition_hint",
                                 role=role.name,
                                 invocation_id=context.get("invocation_id")
                                 if isinstance(context, dict) else None,
                                 tool=name, hint=repeat_hint)

            if name not in allowed:
                return deny("capability",
                            f"role {role.name} may not use tool {name}; "
                            f"available tools: {', '.join(sorted(allowed))}",
                            name, context)
            ledger_problem = _ledger_write_problem(name, tool_input)
            if ledger_problem:
                return deny("ledger_write", ledger_problem, name, context)
            if name == "Bash" and role.bash_patterns:
                # Coarse prefix matching — compound commands (&&, ;, |) can
                # evade it. This is minimal containment, not a sandbox.
                command = (tool_input or {}).get("command", "")
                if not any(command.startswith(p) for p in role.bash_patterns):
                    return deny(
                        "bash_patterns",
                        f"role {role.name} may only run Bash commands starting "
                        f"with: {', '.join(role.bash_patterns)}; {alternatives}",
                        name, context)
            if name == "Bash" and role.forbidden_bash_substrings:
                command = (tool_input or {}).get("command", "")
                forbidden = next(
                    (part for part in role.forbidden_bash_substrings if part in command),
                    None,
                )
                if forbidden is not None:
                    return deny(
                        "driver_job",
                        f"role {role.name} may not launch long objective work via "
                        f"Bash ({forbidden!r}); submit a driver_job receipt",
                        name, context)
            if name == "Bash" and role.bash_adapter_only:
                command = (tool_input or {}).get("command", "")
                reason = adapter_command_verdict(command, adapter_run_dir)
                if reason is not None:
                    rewritten = _strip_benign_shell(command, adapter_run_dir)
                    if rewritten is not None:
                        self.events.emit("guard_degraded", guard="bash_adapter",
                                         role=role.name,
                                         command_head=command[:120],
                                         rewritten_head=rewritten[:120])
                        hints.append(
                            f"the driver ran `{rewritten}` (trailing shell "
                            "redirects/pipes are removed; adapter output is "
                            "already bounded)")
                        return allow(hints, {**tool_input, "command": rewritten})
                    return deny("bash_adapter",
                                f"{reason}; for anything else, {alternatives}",
                                name, context, command_head=command[:120])
            return allow(hints)

        return hook

    def _compaction_hook(self, role: RoleDefinition, breaker: dict,
                         invocation_id: int):
        """PreCompact: flag the breaker so the next tool call carries the
        re-read hint (the Python SDK has no PostCompact/SessionStart hook)."""

        async def hook(input_data, tool_use_id, context):
            breaker["compacted"] = True
            self.events.emit("session_compacted", role=role.name,
                             invocation_id=invocation_id,
                             trigger=input_data.get("trigger"))
            return {}

        return hook

    def _system_prompt(self, role: RoleDefinition, ctx: InvocationContext) -> str:
        prompt_path = PROMPT_DIR / role.prompt_file
        # driver/prompts/ lands in a later task; tolerate its absence so the
        # runner is testable before the prompt files exist.
        body = (prompt_path.read_text(encoding="utf-8")
                if prompt_path.exists() else "")
        if ctx.extra.get("task_contract_dir"):
            body += ("\n\nRead TASK.md and task.toml from task_contract_dir supplied below. "
                     "These are the effective task contract for this run, including its evaluation budget. "
                     "References to TASK.md/task.toml in task paths or task packets refer to these copies. "
                     "Continue using the original task directory for prepare.py, code, data and the uv environment. "
                     "Keep the staged contract read-only.")
        return (
            body
            + "\n\n---\n\n## Invocation context\n\n"
            + ctx.user_message()
            + "\n\n"
            + ctx.postcondition_checklist(role)
        )

    def _default_client_factory(self, options):
        from claude_agent_sdk import ClaudeSDKClient

        return ClaudeSDKClient(options=options)

    def _build_options(self, role: RoleDefinition, ctx: InvocationContext,
                       server, breaker: dict | None = None):
        if breaker is None:
            breaker = new_breaker()
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        kwargs = {}
        if self.cli_path:
            kwargs["cli_path"] = self.cli_path
        if ctx.resume_session_id:
            kwargs["resume"] = ctx.resume_session_id
        if role.max_turns is not None:
            kwargs["max_turns"] = role.max_turns
        env = {"IS_SANDBOX": "1"}
        token = self._token_pool.active()
        if token:
            # Overrides the inherited ANTHROPIC_AUTH_TOKEN once rotation has
            # advanced the pool; identical to it otherwise.
            env[AUTH_TOKEN_ENV] = token
        task_toml = REPO_ROOT / "tasks" / ctx.task / "task.toml"
        if task_toml.is_file():
            import tomllib
            with task_toml.open("rb") as stream:
                mlebench = tomllib.load(stream).get("mlebench") or {}
            public_env = mlebench.get("public_data_env")
            if isinstance(public_env, str) and public_env:
                env[public_env] = str(
                    ctx.run_dir.resolve() / "run_input" / "public")
        return ClaudeAgentOptions(
            system_prompt=self._system_prompt(role, ctx),
            cwd=REPO_ROOT,
            model=self.model,
            permission_mode="bypassPermissions",
            # Claude Code refuses bypassPermissions under root unless it is
            # told the session runs in a sandbox; our runs do (dedicated
            # server/container). Harmless for non-root users.
            env=env,
            setting_sources=[],
            disallowed_tools=list(role.disallowed),
            mcp_servers={"receipts": server},
            hooks={
                "PreToolUse": [HookMatcher(matcher=None, hooks=[
                    self._capability_hook(role, breaker,
                                          run_dir=Path(ctx.run_dir))])],
                "PreCompact": [HookMatcher(matcher=None, hooks=[
                    self._compaction_hook(role, breaker, ctx.invocation_id)])],
            },
            **kwargs,
        )

    async def _drain(self, client, role: RoleDefinition, ctx: InvocationContext,
                     store: ReceiptStore, accepted: list[dict],
                     breaker: dict | None = None,
                     transcript: _BoundedTranscript | None = None,
                     invocation_start: float | None = None) -> dict | None:
        """Consume one turn's message stream.

        Returns the ResultMessage summary ``{"is_error", "subtype",
        "num_turns", "api_error_status", "tool_use", "wall_limited"}`` or
        None when the stream ended without one. Two bounds live here: the
        role's idle timeout between messages (a hung CLI produces none) and,
        when ``invocation_start`` is given, the role's backstop wall limit —
        exceeding it without a new receipt interrupts the session and flags
        ``wall_limited`` for the caller to rescue or fail.
        """
        if breaker is None:
            breaker = new_breaker()
        from claude_agent_sdk import ResultMessage, SystemMessage

        interrupted = False
        wall_limited = False
        saw_tool_use = False
        # {"is_error", "subtype"} of the session's ResultMessage, so the caller
        # can tell an error-terminated session (e.g. error_max_turns) apart
        # from an ordinary no-receipt turn.
        result_info = None
        # Receipts accepted by EARLIER drains (e.g. the one whose postcondition
        # failure triggered this corrective turn) must not interrupt this
        # drain — only a NEW acceptance ends this turn's useful work.
        accepted_baseline = len(accepted)
        stream = client.receive_response().__aiter__()
        idle = role.idle_timeout_seconds
        while True:
            try:
                if idle is None:
                    msg = await stream.__anext__()
                else:
                    with anyio.fail_after(idle):
                        msg = await stream.__anext__()
            except StopAsyncIteration:
                break
            except TimeoutError:
                self.events.emit("idle_timeout", role=role.name,
                                 invocation_id=ctx.invocation_id,
                                 silent_seconds=idle)
                store.mark_session_ended(role.name, ctx.invocation_id, False,
                                         "idle_timeout")
                result_info = {"is_error": True, "subtype": "idle_timeout",
                               "num_turns": None, "api_error_status": None,
                               "tool_use": saw_tool_use, "wall_limited": False}
                break
            if transcript is not None:
                transcript.add(msg)
            saw_tool_use = saw_tool_use or _has_tool_use(msg)
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                session_id = msg.data.get("session_id")
                if session_id:
                    # Persist IMMEDIATELY on init — a later kill can then
                    # resume this exact conversation via resume=<session_id>.
                    store.persist_session_id(
                        role.name, ctx.invocation_id, session_id,
                        run_id=ctx.run_id,
                        parent_session_id=ctx.resume_session_id)
            elif isinstance(msg, ResultMessage):
                is_error = self._result_is_error(msg.is_error, interrupted,
                                                 breaker)
                result_info = {"is_error": is_error, "subtype": msg.subtype,
                               "num_turns": msg.num_turns,
                               "api_error_status": msg.api_error_status,
                               "tool_use": saw_tool_use,
                               "wall_limited": wall_limited}
                store.mark_session_ended(role.name, ctx.invocation_id,
                                         not is_error, msg.subtype)
                self.events.emit(
                    "session_end",
                    role=role.name,
                    invocation_id=ctx.invocation_id,
                    session_id=msg.session_id,
                    is_error=msg.is_error,
                    num_turns=msg.num_turns,
                    total_cost_usd=msg.total_cost_usd,
                    usage=msg.usage,
                    # error detail matters most when is_error is true; these
                    # are None/empty on success and cheap to carry always
                    stop_reason=msg.stop_reason,
                    api_error_status=msg.api_error_status,
                    errors=msg.errors,
                    result_excerpt=(msg.result or "")[:500] or None,
                )
            # duck-typed test fakes
            elif getattr(msg, "subtype", None) == "init":
                session_id = msg.data.get("session_id")
                if session_id:
                    store.persist_session_id(
                        role.name, ctx.invocation_id, session_id,
                        run_id=ctx.run_id,
                        parent_session_id=ctx.resume_session_id)
            elif hasattr(msg, "num_turns"):
                is_error = self._result_is_error(msg.is_error, interrupted,
                                                 breaker)
                result_info = {
                    "is_error": is_error,
                    "subtype": getattr(msg, "subtype", None),
                    "num_turns": getattr(msg, "num_turns", None),
                    "api_error_status": getattr(msg, "api_error_status", None),
                    "tool_use": saw_tool_use,
                    "wall_limited": wall_limited,
                }
                store.mark_session_ended(role.name, ctx.invocation_id,
                                         not is_error,
                                         getattr(msg, "subtype", None))
                self.events.emit(
                    "session_end",
                    role=role.name,
                    invocation_id=ctx.invocation_id,
                    session_id=msg.session_id,
                    is_error=msg.is_error,
                    num_turns=msg.num_turns,
                    total_cost_usd=msg.total_cost_usd,
                    usage=msg.usage,
                    stop_reason=getattr(msg, "stop_reason", None),
                    api_error_status=getattr(msg, "api_error_status", None),
                    errors=getattr(msg, "errors", None),
                    result_excerpt=(getattr(msg, "result", None) or "")[:500]
                                   or None,
                )
            # A schema-valid receipt ends the invocation's useful work — the
            # receipt protocol defines it as the role's FINAL artifact
            # ("Call after ALL on-disk work is complete"). Without an
            # interrupt the model keeps going after "receipt accepted":
            # observed up to 31 extra turns and ~$5 per invocation in
            # degenerate repetition loops. Keep draining after the interrupt
            # so the ResultMessage (usage accounting) still lands.
            if (len(accepted) > accepted_baseline
                    or breaker["tripped"] is not None) and not interrupted:
                if breaker["tripped"] is not None:
                    self.events.emit("repetition_breaker", role=role.name,
                                     invocation_id=ctx.invocation_id,
                                     description=breaker["tripped"])
                interrupted = True
                interrupt = getattr(client, "interrupt", None)
                if interrupt is not None:
                    try:
                        await interrupt()
                    except Exception:
                        # The receipt is already persisted; a failed
                        # interrupt must not fail the invocation.
                        pass
            elif (not interrupted and invocation_start is not None
                    and role.wall_limit_seconds is not None
                    and time.monotonic() - invocation_start
                    > role.wall_limit_seconds):
                # Backstop bound, not a trimming threshold (set ≥ 2× the
                # observed maximum). Interrupt now; the caller decides
                # between one rescue turn and failing the invocation.
                wall_limited = interrupted = True
                if result_info is not None:
                    result_info["wall_limited"] = True
                await self._best_effort_interrupt(client)
        if transcript is not None and transcript.entries:
            try:
                store.persist_session_messages(role.name, ctx.invocation_id,
                                               transcript.rows())
            except Exception:
                # Diagnostics must never change the invocation's outcome.
                self.events.emit("session_dump_failed", role=role.name,
                                 invocation_id=ctx.invocation_id)
        return result_info

    @staticmethod
    def _result_is_error(is_error: bool, interrupted: bool, breaker: dict) -> bool:
        """The CLI reports a turn we interrupted after receipt acceptance as
        ``error_during_execution`` (``errors=["[ede_diagnostic] ..."]``): the
        abort lands mid tool-use, so its last message is not a clean stop.
        That is our own stop signal, not a poisoned session — treating it as
        an error would skip corrective turns and, via the resume guard, turn
        every driver_job / repair continuation into a fresh session. A
        breaker-tripped interrupt keeps the error verdict: that context IS
        degenerate."""
        if interrupted and breaker["tripped"] is None:
            return False
        return bool(is_error)

    @staticmethod
    def _latest_receipt(store: ReceiptStore, role: RoleDefinition,
                        ctx: InvocationContext, accepted: list[dict]) -> dict | None:
        """Newest accepted receipt payload, or None.

        The real SDK path appends to ``accepted`` via the receipts MCP tool;
        injected test clients persist directly through the store, so fall
        back to the on-disk receipt — the durable artifact is the source of
        truth either way.
        """
        if accepted:
            return accepted[-1]
        path = store.receipt_path(role.name, ctx.invocation_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _problems(self, role: RoleDefinition, ctx: InvocationContext,
                  receipt: dict | None) -> list[str]:
        problems = [] if receipt is not None else ["no accepted receipt"]
        # A typed long-job request is an intermediate handoff, not a completed
        # role invocation. The driver executes it synchronously and resumes the
        # same session; terminal postconditions are checked on that later turn.
        if receipt is not None and isinstance(receipt.get("driver_job"), dict):
            handoff_problem = driver_job_handoff_problem(role.name, receipt)
            if handoff_problem:
                problems.append(handoff_problem)
            return problems
        for check in role.postconditions:
            error = check(ctx)
            if error:
                problems.append(error)
        return problems

    @staticmethod
    def _corrective_message(problems: list[str]) -> str:
        items = "\n".join(f"- {p}" for p in problems)
        return (
            "Your previous turn did NOT satisfy this invocation's requirements. "
            "Fix each problem below using your tools, then call "
            f"{RECEIPT_TOOL} again:\n{items}"
        )

    @staticmethod
    async def _best_effort_interrupt(client) -> None:
        interrupt = getattr(client, "interrupt", None)
        if interrupt is None:
            return
        try:
            with anyio.CancelScope(shield=True), anyio.move_on_after(10):
                await interrupt()
        except Exception:
            pass

    @staticmethod
    def _transport_failure(result: dict | None) -> bool:
        """First-turn error with nothing done: API refusal at start (402)
        or a relay transient. A max_turns cutoff is never one."""
        return bool(
            result and result.get("is_error")
            and result.get("subtype") not in ("error_max_turns", "idle_timeout")
            and (result.get("num_turns") or 0) <= 1
            and not result.get("tool_use")
        )

    def _emit_wall_limit(self, role, ctx, started, *, source, had_receipt,
                         rescued) -> None:
        self.events.emit("session_wall_limit", role=role.name,
                         invocation_id=ctx.invocation_id,
                         elapsed_seconds=round(time.monotonic() - started, 1),
                         limit_source=source, had_receipt=had_receipt,
                         rescued=rescued)

    async def _run_async(self, role: RoleDefinition, ctx: InvocationContext) -> dict:
        store = ReceiptStore(ctx.run_dir)
        server, accepted = build_receipt_server(
            role.name, role.receipt_schema, store, ctx.invocation_id
        )
        breaker = new_breaker()
        options = self._build_options(role, ctx, server, breaker)
        factory = self._client_factory or self._default_client_factory
        self.events.emit("session_start", role=role.name,
                         invocation_id=ctx.invocation_id,
                         resume=bool(ctx.resume_session_id))
        transcript = _BoundedTranscript()
        started = time.monotonic()
        usable = session_time_budget(ctx).get("usable_seconds")
        outcome: tuple[dict | None, list[str]] | None = None
        async with factory(options) as client:
            # The run's single cutoff bounds the whole conversation — every
            # drain, idle wait, corrective turn, and rescue grace — by an
            # absolute clock, independent of whether messages arrive.
            with anyio.move_on_after(
                    None if usable is None else max(0.0, float(usable))
            ) as deadline:
                outcome = await self._converse(
                    client, role, ctx, store, accepted, breaker, transcript,
                    started)
            if deadline.cancelled_caught:
                await self._best_effort_interrupt(client)
        if outcome is None:
            self._emit_wall_limit(role, ctx, started, source="deadline",
                                  had_receipt=bool(accepted), rescued=False)
            raise InvocationFailed(role.name,
                                   [f"{TIME_REACHED_PROBLEM} mid-session"],
                                   invocation_id=ctx.invocation_id)
        receipt, problems = outcome
        if problems:
            raise InvocationFailed(role.name, problems,
                                   invocation_id=ctx.invocation_id)
        return receipt

    async def _converse(self, client, role, ctx, store, accepted, breaker,
                        transcript, started) -> tuple[dict | None, list[str]]:
        await client.query(ctx.user_message())
        result = await self._drain(client, role, ctx, store, accepted,
                                   breaker, transcript, started)
        receipt = self._latest_receipt(store, role, ctx, accepted)
        if receipt is None and self._transport_failure(result):
            raise _TransportFailure(
                [f"session ended with error result: {result['subtype']}"],
                api_error_status=result.get("api_error_status"))
        attempts = 0
        while True:
            # Every drain shares the same wall-limit exit, including the
            # last corrective turn. Rescue returns immediately, so it can
            # run at most once per invocation.
            if result and result.get("wall_limited") and receipt is None:
                receipt = await self._soft_rescue(client, role, ctx, store,
                                                  accepted, breaker, transcript,
                                                  started)
                return receipt, self._problems(role, ctx, receipt)
            problems = self._problems(role, ctx, receipt)
            if breaker["tripped"] is not None:
                problems.append(
                    f"repetition breaker tripped: {breaker['tripped']}")
            if (not problems or attempts >= role.corrective_attempts
                    or breaker["tripped"] is not None):
                return receipt, problems
            if result and result["is_error"]:
                # The session ended on an error result (e.g.
                # error_max_turns from a role's max_turns cap): the CLI
                # process may be dead, and a corrective turn cannot
                # produce the receipt anyway. Fail directly instead of
                # querying a spent session, so callers' InvocationFailed
                # handlers see every bounded-role cutoff.
                problems.append(
                    f"session ended with error result: {result['subtype']}"
                )
                break
            if result and result.get("wall_limited"):
                self._emit_wall_limit(role, ctx, started, source="role",
                                      had_receipt=receipt is not None,
                                      rescued=False)
                problems.append(
                    f"wall-clock limit {role.wall_limit_seconds:g}s reached")
                break
            attempts += 1
            self.events.emit("corrective_followup", role=role.name,
                             invocation_id=ctx.invocation_id,
                             attempt=attempts, problems=problems)
            await client.query(self._corrective_message(problems))
            result = await self._drain(client, role, ctx, store, accepted,
                                       breaker, transcript, started)
            receipt = self._latest_receipt(store, role, ctx, accepted)
        return receipt, problems

    async def _soft_rescue(self, client, role, ctx, store, accepted, breaker,
                           transcript, started) -> dict | None:
        """At the wall limit without a receipt: soft-rescue roles get one
        'submit now' turn under a short grace (their partial result — an
        edit, a draft — is valid); terminal roles fail, since a coerced
        receipt there would be a fabricated verdict."""
        if not role.soft_rescue:
            self._emit_wall_limit(role, ctx, started, source="role",
                                  had_receipt=False, rescued=False)
            raise InvocationFailed(
                role.name,
                [f"wall-clock limit {role.wall_limit_seconds:g}s reached "
                 "without a receipt"],
                invocation_id=ctx.invocation_id)
        await client.query(SOFT_RESCUE_MESSAGE)
        with anyio.move_on_after(SOFT_RESCUE_GRACE_SECONDS) as grace:
            # invocation_start omitted: the wall limit already fired; only
            # the grace bounds this turn.
            await self._drain(client, role, ctx, store, accepted, breaker,
                              transcript)
        receipt = self._latest_receipt(store, role, ctx, accepted)
        if grace.cancelled_caught:
            await self._best_effort_interrupt(client)
        self._emit_wall_limit(role, ctx, started, source="role",
                              had_receipt=receipt is not None,
                              rescued=receipt is not None)
        if receipt is None:
            raise InvocationFailed(
                role.name,
                [f"wall-clock limit {role.wall_limit_seconds:g}s reached; "
                 "rescue turn produced no receipt"],
                invocation_id=ctx.invocation_id)
        return receipt


class FakeSessionRunner:
    """Scripted runner for loop tests. Each script entry:

    {"receipt": {...},
     "side_effects": Callable[[InvocationContext], None] | None,
     "fail": list[str] | None}
    """

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.calls: list[tuple[str, InvocationContext]] = []

    def run(self, role: RoleDefinition, ctx: InvocationContext) -> dict:
        ctx = admit_session(role, ctx)  # same gate as the SDK runner
        self.calls.append((role.name, ctx))
        # Mirror the real runner's init-time persistence so loop tests can
        # exercise same-session resume (resume=<session_id>).
        ReceiptStore(ctx.run_dir).persist_session_id(
            role.name, ctx.invocation_id, f"fake-sess-{ctx.invocation_id:04d}"
        )
        if not self.script:
            raise InvocationFailed(role.name, ["fake runner: script exhausted"])
        entry = self.script.pop(0)
        if entry.get("side_effects"):
            entry["side_effects"](ctx)
        if entry.get("fail"):
            raise InvocationFailed(role.name, entry["fail"])
        ReceiptStore(ctx.run_dir).persist_receipt(
            role.name, ctx.invocation_id, entry["receipt"]
        )
        return entry["receipt"]
