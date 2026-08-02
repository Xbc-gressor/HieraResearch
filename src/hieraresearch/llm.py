"""Revision-bound Claude Messages and Agent SDK adapters."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, TypeVar

from .artifacts import (
    InvocationJournal,
    StaleInferenceError,
    file_revision,
    json_revision,
    paths_revision,
)


T = TypeVar("T")

# Documented max output of claude-sonnet-5 on the Messages API
# (https://docs.claude.com/en/docs/about-claude/models/overview). This is a
# cap, not an allocation: billing is by tokens actually generated, so a
# smaller cap saves nothing and only truncates valid responses. Adaptive
# thinking is always on for Sonnet 5 and thinking tokens count against this
# budget, which is why "conservative" values like 4096 truncate real calls.
# Brevity pressure belongs in prompts, never in this number.
MODEL_MAX_OUTPUT_TOKENS = 128_000


class InferenceError(RuntimeError):
    pass


class InferenceContractError(InferenceError):
    pass


class StructuredBackend(Protocol):
    def generate(
        self,
        *,
        purpose: str,
        model: str,
        system_prompt: str,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> tuple[Any, dict[str, Any]]: ...


class EditBackend(Protocol):
    def edit(
        self,
        *,
        purpose: str,
        model: str,
        cwd: Path,
        system_prompt: str,
        prompt: str,
        tools: Sequence[str],
        policy: "PathPolicy",
        max_turns: int,
    ) -> tuple[str, dict[str, Any]]: ...


@dataclass(frozen=True)
class AgentEditSpec:
    purpose: str
    schema_version: int
    cwd: Path
    system_prompt: str
    prompt: str
    tools: tuple[str, ...]
    read_roots: tuple[Path, ...]
    write_paths: tuple[Path, ...]
    input_paths: tuple[Path, ...]
    immutable_input_paths: tuple[Path, ...]
    derived_output_paths: tuple[Path, ...] = ()
    max_turns: int = 24

    def __post_init__(self) -> None:
        authored = {Path(path).resolve() for path in self.write_paths}
        derived = {Path(path).resolve() for path in self.derived_output_paths}
        overlap = sorted(authored & derived, key=str)
        if overlap:
            raise ValueError(
                "agent-authored and Python-derived outputs must be disjoint: "
                + ", ".join(map(str, overlap))
            )


class ModelGateway:
    def __init__(
        self,
        *,
        model: str,
        journal: InvocationJournal,
        structured_backend: StructuredBackend,
        edit_backend: EditBackend,
    ):
        if not model.strip():
            raise ValueError("model must be non-empty")
        self.model = model
        self.journal = journal
        self.structured_backend = structured_backend
        self.edit_backend = edit_backend

    def infer(
        self,
        *,
        purpose: str,
        schema_version: int,
        system_prompt: str,
        prompt: str,
        schema: dict[str, Any],
        input_paths: Sequence[Path],
        parser: Callable[[Any], T],
        max_tokens: int = MODEL_MAX_OUTPUT_TOKENS,
    ) -> T:
        path_revision = paths_revision(input_paths)
        request = {
            "kind": "structured",
            "purpose": purpose,
            "schema_version": schema_version,
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "schema": schema,
            "input_paths": [str(Path(path).resolve()) for path in input_paths],
            "input_path_revision": path_revision,
            "max_tokens": max_tokens,
        }
        input_revision = json_revision(request)
        previous = self.journal.completed(
            purpose=purpose,
            schema_version=schema_version,
            input_revision=input_revision,
            model=self.model,
        )
        if previous is not None:
            try:
                return parser(previous.response)
            except (TypeError, ValueError, KeyError) as exc:
                raise InferenceContractError(
                    f"recorded {purpose} response no longer satisfies schema {schema_version}: {exc}"
                ) from exc

        invocation = self.journal.begin(
            purpose=purpose,
            schema_version=schema_version,
            model=self.model,
            input_revision=input_revision,
            request=request,
        )
        try:
            response, metadata = self.structured_backend.generate(
                purpose=purpose,
                model=self.model,
                system_prompt=system_prompt,
                prompt=prompt,
                schema=schema,
                max_tokens=max_tokens,
            )
            if paths_revision(input_paths) != path_revision:
                raise StaleInferenceError(
                    f"{purpose} inputs changed while Claude was running; discard the response"
                )
            try:
                parsed = parser(response)
            except (TypeError, ValueError, KeyError) as exc:
                raise InferenceContractError(
                    f"{purpose} returned an invalid schema-{schema_version} response: {exc}"
                ) from exc
            self.journal.complete(invocation, response=response, metadata=metadata)
            return parsed
        except BaseException as exc:
            self.journal.fail(invocation, exc)
            raise

    def edit(self, spec: AgentEditSpec, *, validate: Callable[[], T]) -> T:
        policy = PathPolicy(
            cwd=spec.cwd,
            read_roots=spec.read_roots,
            write_paths=spec.write_paths,
            allowed_tools=spec.tools,
        )
        input_path_revision = paths_revision(spec.input_paths)
        immutable_revision = paths_revision(spec.immutable_input_paths)
        request = {
            "kind": "agent_edit",
            "purpose": spec.purpose,
            "schema_version": spec.schema_version,
            "model": self.model,
            "system_prompt": spec.system_prompt,
            "prompt": spec.prompt,
            "tools": list(spec.tools),
            "cwd": str(spec.cwd.resolve()),
            "read_roots": [str(path.resolve()) for path in spec.read_roots],
            "write_paths": [str(path.resolve()) for path in spec.write_paths],
            "derived_output_paths": [
                str(path.resolve()) for path in spec.derived_output_paths
            ],
            "input_paths": [str(path.resolve()) for path in spec.input_paths],
            "input_path_revision": input_path_revision,
            "immutable_input_revision": immutable_revision,
            "max_turns": spec.max_turns,
        }
        input_revision = json_revision(request)
        invocation = self.journal.begin(
            purpose=spec.purpose,
            schema_version=spec.schema_version,
            model=self.model,
            input_revision=input_revision,
            request=request,
        )
        try:
            result, metadata = self.edit_backend.edit(
                purpose=spec.purpose,
                model=self.model,
                cwd=spec.cwd,
                system_prompt=spec.system_prompt,
                prompt=spec.prompt,
                tools=spec.tools,
                policy=policy,
                max_turns=spec.max_turns,
            )
            if paths_revision(spec.immutable_input_paths) != immutable_revision:
                raise StaleInferenceError(
                    f"{spec.purpose} immutable inputs changed during the edit"
                )
            validated = validate()
            missing_derived = [
                path.resolve()
                for path in spec.derived_output_paths
                if not path.resolve().is_file()
            ]
            if missing_derived:
                raise InferenceContractError(
                    f"{spec.purpose} validation produced no derived outputs: "
                    + ", ".join(map(str, missing_derived))
                )
            output_revisions = {
                str(path.resolve()): file_revision(path.resolve())
                for path in (*spec.write_paths, *spec.derived_output_paths)
                if path.resolve().is_file()
            }
            response = {"result": result, "output_revisions": output_revisions}
            self.journal.complete(invocation, response=response, metadata=metadata)
            return validated
        except BaseException as exc:
            self.journal.fail(invocation, exc)
            raise


class AnthropicMessagesBackend:
    def __init__(
        self,
        *,
        # Headroom for adaptive thinking plus a long structured generation;
        # matches the Anthropic SDK's own default request timeout.
        timeout_seconds: float = 600.0,
        max_retries: int = 2,
        client: Any | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = client

    def _client_instance(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - installation boundary
            raise InferenceError(
                "Anthropic SDK is not installed; install the project dependencies"
            ) from exc
        self._client = Anthropic(
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        return self._client

    def generate(
        self,
        *,
        purpose: str,
        model: str,
        system_prompt: str,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> tuple[Any, dict[str, Any]]:
        del purpose
        try:
            message = self._client_instance().messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "format": {"type": "json_schema", "schema": schema}
                },
            )
        except Exception as exc:
            raise InferenceError(f"Claude Messages request failed: {exc}") from exc
        if getattr(message, "stop_reason", None) in {"refusal", "max_tokens"}:
            raise InferenceContractError(
                f"Claude Messages stopped with {message.stop_reason!r}"
            )
        text = "".join(
            str(block.text)
            for block in getattr(message, "content", [])
            if getattr(block, "type", None) == "text"
        )
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InferenceContractError(
                "Claude Messages did not return a complete JSON object"
            ) from exc
        usage = getattr(message, "usage", None)
        metadata = {
            "request_id": getattr(message, "_request_id", None),
            "stop_reason": getattr(message, "stop_reason", None),
            "usage": usage.to_dict() if hasattr(usage, "to_dict") else str(usage),
        }
        return value, metadata


class PathPolicy:
    """Hard file/tool boundary used by every Agent SDK edit call."""

    READ_TOOLS = {"Read", "Glob", "Grep"}
    WRITE_TOOLS = {"Write", "Edit"}
    WEB_TOOLS = {"WebSearch", "WebFetch"}

    def __init__(
        self,
        *,
        cwd: Path,
        read_roots: Sequence[Path],
        write_paths: Sequence[Path],
        allowed_tools: Sequence[str],
    ):
        self.cwd = Path(cwd).resolve()
        self.read_roots = tuple(Path(path).resolve() for path in read_roots)
        self.write_paths = frozenset(Path(path).resolve() for path in write_paths)
        self.allowed_tools = frozenset(allowed_tools)

    @staticmethod
    def _within(path: Path, root: Path) -> bool:
        if root.is_file():
            return path == root
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _resolve(self, raw: Any) -> Path | None:
        if not isinstance(raw, str) or not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.cwd / path
        return path.resolve()

    def decision(self, tool_name: str, tool_input: dict[str, Any]) -> tuple[bool, str]:
        if tool_name not in self.allowed_tools:
            return False, f"tool {tool_name} is outside this invocation's allow-list"
        if tool_name in self.WEB_TOOLS:
            return True, "web tool allowed for this invocation"
        if tool_name in self.WRITE_TOOLS:
            path = self._resolve(tool_input.get("file_path"))
            if path is None or path not in self.write_paths:
                return False, f"write target is not explicitly allowed: {path}"
            return True, "write target allowed"
        if tool_name in self.READ_TOOLS:
            raw_path = tool_input.get("file_path")
            if tool_name in {"Glob", "Grep"}:
                raw_path = tool_input.get("path") or str(self.cwd)
                pattern = tool_input.get("pattern")
                if tool_name == "Glob" and isinstance(pattern, str):
                    pattern_path = Path(pattern)
                    if pattern_path.is_absolute() or ".." in pattern_path.parts:
                        return False, "glob pattern may not escape its bounded search root"
            path = self._resolve(raw_path)
            roots = (*self.read_roots, *self.write_paths)
            if path is None or not any(self._within(path, root) for root in roots):
                return False, f"read target is outside bounded roots: {path}"
            return True, "read target allowed"
        return False, f"unsupported tool for bounded editor: {tool_name}"


class ClaudeAgentBackend:
    def edit(
        self,
        *,
        purpose: str,
        model: str,
        cwd: Path,
        system_prompt: str,
        prompt: str,
        tools: Sequence[str],
        policy: PathPolicy,
        max_turns: int,
    ) -> tuple[str, dict[str, Any]]:
        del purpose
        return asyncio.run(
            self._edit_async(
                model=model,
                cwd=cwd,
                system_prompt=system_prompt,
                prompt=prompt,
                tools=tools,
                policy=policy,
                max_turns=max_turns,
            )
        )

    async def _edit_async(
        self,
        *,
        model: str,
        cwd: Path,
        system_prompt: str,
        prompt: str,
        tools: Sequence[str],
        policy: PathPolicy,
        max_turns: int,
    ) -> tuple[str, dict[str, Any]]:
        try:
            from claude_agent_sdk import (
                ClaudeAgentOptions,
                HookMatcher,
                ResultMessage,
                query,
            )
        except ImportError as exc:  # pragma: no cover - installation boundary
            raise InferenceError(
                "Claude Agent SDK is not installed; install the project dependencies"
            ) from exc

        async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
            del tool_use_id, context
            allowed, reason = policy.decision(
                str(input_data.get("tool_name", "")),
                input_data.get("tool_input", {}),
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow" if allowed else "deny",
                    "permissionDecisionReason": reason,
                }
            }

        options = ClaudeAgentOptions(
            tools=list(tools),
            allowed_tools=list(tools),
            system_prompt=system_prompt,
            strict_mcp_config=True,
            permission_mode="default",
            model=model,
            max_turns=max_turns,
            cwd=cwd,
            setting_sources=[],
            hooks={"PreToolUse": [HookMatcher(hooks=[guard])]},
            sandbox={
                "enabled": True,
                "autoAllowBashIfSandboxed": False,
                "allowUnsandboxedCommands": False,
            },
        )
        final: Any | None = None
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    final = message
        except Exception as exc:
            raise InferenceError(f"Claude Agent SDK edit failed: {exc}") from exc
        if final is None:
            raise InferenceError("Claude Agent SDK returned no final result")
        if final.is_error:
            raise InferenceError(
                f"Claude Agent SDK edit ended with {final.subtype}: {final.result or ''}"
            )
        metadata = {
            "session_id": final.session_id,
            "subtype": final.subtype,
            "stop_reason": final.stop_reason,
            "num_turns": final.num_turns,
            "duration_ms": final.duration_ms,
            "total_cost_usd": final.total_cost_usd,
            "usage": final.usage,
        }
        return final.result or "", metadata


def _recording_name(purpose: str) -> str:
    value = re.sub(r"[^a-z0-9_.-]+", "-", purpose.lower()).strip("-.")
    return value[:120] or "inference"


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RecordedBackend:
    """Replay purpose-keyed structured responses and exact edit snapshots."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"recording directory does not exist: {self.root}")

    def generate(
        self,
        *,
        purpose: str,
        model: str,
        system_prompt: str,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> tuple[Any, dict[str, Any]]:
        del model, system_prompt, prompt, schema, max_tokens
        source = self.root / f"{_recording_name(purpose)}.json"
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InferenceError(
                f"missing or invalid structured recording for {purpose}: {source}: {exc}"
            ) from exc
        return value, {"backend": "recorded", "source": str(source)}

    def edit(
        self,
        *,
        purpose: str,
        model: str,
        cwd: Path,
        system_prompt: str,
        prompt: str,
        tools: Sequence[str],
        policy: PathPolicy,
        max_turns: int,
    ) -> tuple[str, dict[str, Any]]:
        del model, cwd, system_prompt, prompt, tools, max_turns
        source_dir = self.root / _recording_name(purpose)
        for target in sorted(policy.write_paths, key=str):
            allowed, reason = policy.decision("Write", {"file_path": str(target)})
            if not allowed:
                raise InferenceError(f"recorded edit target rejected: {reason}")
            source = source_dir / target.name
            try:
                payload = source.read_bytes()
            except OSError as exc:
                raise InferenceError(
                    f"missing edit recording for {purpose} target {target.name}: {source}"
                ) from exc
            _atomic_write_bytes(target, payload)
        return "recorded edit applied", {
            "backend": "recorded",
            "source": str(source_dir),
        }
