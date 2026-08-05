"""Receipt protocol: schema validation, persistence, MCP server factory.

Roles never return free-text JSON. Each session gets an in-process MCP
server exposing exactly one tool, surfaced as
``mcp__receipts__submit_receipt``. Exactly one ACCEPTED receipt per
invocation: rejected attempts and post-repair replacements are normal.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


class ReceiptError(ValueError):
    pass


_TYPE_CHECKS = {
    "str": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
}


def validate_receipt(schema: dict[str, Any], payload: object) -> list[str]:
    """Return validation problems; empty list means valid.

    Field spec: a type name ("str"/"int"/"float"/"bool"/"list"/"dict"),
    optionally "?" prefixed for optional fields, or ("enum", v1, v2, ...).
    """
    if not isinstance(payload, dict):
        return ["receipt is not a JSON object"]
    problems: list[str] = []
    for key, spec in schema.items():
        optional = isinstance(spec, str) and spec.startswith("?")
        if key not in payload:
            if not optional:
                problems.append(f"missing field: {key}")
            continue
        value = payload[key]
        if isinstance(spec, tuple) and spec and spec[0] == "enum":
            if value not in spec[1:]:
                problems.append(f"field {key}: {value!r} not one of {list(spec[1:])}")
            continue
        type_name = spec[1:] if optional else spec
        check = _TYPE_CHECKS.get(type_name)
        if check is None:
            raise ReceiptError(f"unknown schema spec for {key}: {spec!r}")
        if not check(value):
            problems.append(
                f"field {key}: expected {type_name}, got {type(value).__name__}"
            )
    return problems


_ID_SUFFIX = re.compile(r"-(\d{4})\.(?:session\.)?json$")


class ReceiptStore:
    """Persists accepted receipts and SDK session ids under <run_dir>/receipts/.

    invocation_id is driver-assigned, monotonic across BOTH receipt and
    session files, and links receipt ↔ session ↔ driver_events rows.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def _dir(self) -> Path:
        d = self.run_dir / "receipts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def receipt_path(self, role: str, invocation_id: int) -> Path:
        return self._dir() / f"{role}-{invocation_id:04d}.json"

    def session_path(self, role: str, invocation_id: int) -> Path:
        return self._dir() / f"{role}-{invocation_id:04d}.session.json"

    def next_invocation_id(self) -> int:
        highest = 0
        for path in self._dir().iterdir():
            match = _ID_SUFFIX.search(path.name)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def persist_receipt(self, role: str, invocation_id: int, payload: dict) -> Path:
        path = self.receipt_path(role, invocation_id)
        if path.exists():
            raise ReceiptError(f"receipt already exists: {path}")
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)
        return path

    def persist_session_id(self, role: str, invocation_id: int, session_id: str) -> None:
        self.session_path(role, invocation_id).write_text(
            json.dumps({"session_id": session_id}) + "\n", encoding="utf-8"
        )

    def load_session_id(self, role: str, invocation_id: int) -> str | None:
        path = self.session_path(role, invocation_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("session_id")


def build_receipt_server(
    role_name: str,
    schema: dict[str, Any],
    store: ReceiptStore,
    invocation_id: int,
):
    """In-process MCP server exposing exactly one tool: submit_receipt.

    Returns (server, accepted) where accepted collects validated payloads
    (at most one is expected; the driver reads accepted[-1]).
    Lazily imports claude_agent_sdk so validation/store tests need no SDK.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    accepted: list[dict] = []

    @tool(
        "submit_receipt",
        "Submit this role's final structured receipt. Call after ALL on-disk "
        "work is complete. Rejected submissions return an error listing the "
        "problems; fix them and call again.",
        {"receipt": dict},
    )
    async def submit_receipt(args: dict) -> dict:
        payload = args.get("receipt")
        problems = validate_receipt(schema, payload)
        if problems:
            return {
                "content": [{"type": "text",
                             "text": "receipt rejected: " + "; ".join(problems)}],
                "isError": True,
            }
        store.persist_receipt(role_name, invocation_id, payload)
        accepted.append(payload)
        return {
            "content": [{"type": "text",
                         "text": f"receipt accepted "
                                 f"({store.receipt_path(role_name, invocation_id).name})"}]
        }

    server = create_sdk_mcp_server(name="receipts", version="1.0.0",
                                   tools=[submit_receipt])
    return server, accepted
