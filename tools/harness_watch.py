#!/usr/bin/env python3
"""Live token, delegation, and run-progress monitor for HieraResearch.

OpenCode mode reads its local SQLite store read-only and attributes usage to the
root and every descendant session. Claude mode reads persisted JSONL transcripts.
The two status-line modes consume Claude Code's JSON on stdin and do no I/O.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWN_AGENTS = {
    "autoresearch-experiment",
    "background-researcher",
    "idea-generator",
    "experience-extractor",
    "candidate-writer",
    "tunable-contract-extractor",
    "tuner-orchestrator",
}


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0

    def add(self, other: "Usage") -> None:
        for name in ("input", "output", "reasoning", "cache_read", "cache_write"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.cost += other.cost

    @property
    def processed(self) -> int:
        return self.input + self.cache_read + self.cache_write


@dataclass
class SessionView:
    id: str
    parent_id: str | None
    depth: int
    agent: str
    title: str
    status: str
    current_tool: str | None
    updated_ms: int
    usage: Usage
    latest_context: int


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _fmt_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _fmt_age(updated_ms: int) -> str:
    seconds = max(0, int(time.time() - updated_ms / 1000))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _run_snapshot(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    run_dir = run_dir.resolve()
    ledger_path = run_dir / "ledger.json"
    cfg_path = run_dir / "framework_cfg.json"
    state_path = run_dir / "loop_state.md"
    try:
        ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        ledger = {}
    try:
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        cfg = {}
    state: dict[str, str] = {}
    if state_path.is_file():
        for line in state_path.read_text(errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator:
                state[key.strip()] = value.strip()
    records = ledger.get("records", []) if isinstance(ledger.get("records", []), list) else []
    attempted = 0
    for record in records:
        value = record.get("trials_attempted")
        if value is None:
            value = record.get("trials_completed")
        if value is None:
            value = record.get("warm_start_K")
        attempted += _integer(value)
    stored = ledger.get("run_state") if isinstance(ledger.get("run_state"), dict) else {}
    budget = cfg.get("max_evaluations")
    if not isinstance(budget, int) or isinstance(budget, bool):
        budget = stored.get("evaluation_budget")
    budget = budget if isinstance(budget, int) and not isinstance(budget, bool) else None
    if stored.get("phase") == "blocked" or state.get("phase") == "blocked":
        phase = "blocked"
        stop_condition = stored.get("active_stop_condition") or state.get("active_stop_condition")
    elif budget is not None and attempted >= budget:
        phase = "completed"
        stop_condition = "evaluation_budget_reached"
    else:
        phase = "running"
        stop_condition = "none"
    mtimes = [path.stat().st_mtime for path in (ledger_path, state_path) if path.exists()]
    return {
        "run_dir": str(run_dir),
        "task": ledger.get("task") or state.get("task"),
        "tag": ledger.get("tag") or state.get("tag") or run_dir.name,
        "phase": phase,
        "active_stop_condition": stop_condition,
        "candidates": len(records),
        "pending_run_ids": [r.get("run_id") for r in records if r.get("status") == "pending"],
        "evaluations_attempted": attempted,
        "budget": budget,
        "remaining": None if budget is None else max(0, budget - attempted),
        "last_progress_ms": int(max(mtimes) * 1000) if mtimes else 0,
    }


def _default_opencode_db() -> Path:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "opencode" / "opencode.db"


class OpenCodeReader:
    def __init__(self, db_path: Path):
        self.db_path = db_path.expanduser().resolve()
        self.db = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=1)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        self.db.close()

    def _root_for_id(self, session_id: str) -> str:
        seen: set[str] = set()
        current = session_id
        while current not in seen:
            seen.add(current)
            row = self.db.execute("SELECT parent_id FROM session WHERE id=?", (current,)).fetchone()
            if row is None:
                raise SystemExit(f"OpenCode session not found: {session_id}")
            if not row["parent_id"]:
                return current
            current = row["parent_id"]
        raise SystemExit(f"cycle in OpenCode session ancestry: {session_id}")

    def select_root(self, project_dir: Path, session_id: str | None, run_dir: Path | None) -> str:
        if session_id:
            return self._root_for_id(session_id)
        rows = self.db.execute(
            "SELECT id FROM session WHERE parent_id IS NULL AND directory=? "
            "ORDER BY time_updated DESC LIMIT 40",
            (str(project_dir.resolve()),),
        ).fetchall()
        if not rows:
            raise SystemExit(f"no OpenCode root sessions for {project_dir.resolve()}")
        if run_dir is not None:
            needles = {str(run_dir.resolve()), str(run_dir), run_dir.as_posix()}
            try:
                needles.add(str(run_dir.resolve().relative_to(project_dir.resolve())))
            except ValueError:
                pass
            for row in rows:
                parts = self.db.execute(
                    "SELECT data FROM part WHERE session_id=?", (row["id"],)
                ).fetchall()
                text = "\n".join(part["data"] for part in parts)
                if any(re.search(re.escape(needle) + r"(?=[/\s`\"']|$)", text) for needle in needles):
                    return row["id"]
        return rows[0]["id"]

    def _sessions(self, root_id: str) -> list[SessionView]:
        rows = self.db.execute(
            """
            WITH RECURSIVE tree(id, depth) AS (
              SELECT ?, 0
              UNION ALL
              SELECT s.id, tree.depth + 1 FROM session s JOIN tree ON s.parent_id=tree.id
            )
            SELECT s.*, tree.depth FROM session s JOIN tree ON tree.id=s.id
            ORDER BY s.time_created
            """,
            (root_id,),
        ).fetchall()
        result: list[SessionView] = []
        for row in rows:
            latest = self.db.execute(
                "SELECT data FROM message WHERE session_id=? "
                "AND json_extract(data,'$.role')='assistant' ORDER BY time_created DESC LIMIT 1",
                (row["id"],),
            ).fetchone()
            latest_data = json.loads(latest["data"]) if latest else {}
            completed = (latest_data.get("time") or {}).get("completed")
            status = "active" if latest and completed is None else ("idle" if row["depth"] == 0 else "done")
            running_tool = self.db.execute(
                "SELECT data FROM part WHERE session_id=? AND json_extract(data,'$.type')='tool' "
                "AND json_extract(data,'$.state.status')='running' ORDER BY time_updated DESC LIMIT 1",
                (row["id"],),
            ).fetchone()
            current_tool = None
            if running_tool:
                part = json.loads(running_tool["data"])
                current_tool = part.get("tool")
                description = ((part.get("state") or {}).get("input") or {}).get("description")
                if description:
                    current_tool = f"{current_tool}: {description}"
            tokens = latest_data.get("tokens") or {}
            cache = tokens.get("cache") or {}
            latest_context = _integer(tokens.get("input")) + _integer(cache.get("read")) + _integer(cache.get("write"))
            result.append(SessionView(
                id=row["id"], parent_id=row["parent_id"], depth=row["depth"],
                agent=row["agent"] or "unknown", title=row["title"], status=status,
                current_tool=current_tool, updated_ms=row["time_updated"],
                usage=Usage(
                    input=_integer(row["tokens_input"]), output=_integer(row["tokens_output"]),
                    reasoning=_integer(row["tokens_reasoning"]),
                    cache_read=_integer(row["tokens_cache_read"]),
                    cache_write=_integer(row["tokens_cache_write"]), cost=float(row["cost"] or 0),
                ), latest_context=latest_context,
            ))
        return result

    def _role_violations(self, root_id: str) -> list[str]:
        rows = self.db.execute(
            "SELECT data FROM part WHERE session_id=? AND json_extract(data,'$.type')='tool' "
            "AND json_extract(data,'$.tool')='task' "
            "AND json_extract(data,'$.state.status')='completed'",
            (root_id,),
        ).fetchall()
        violations: list[str] = []
        for row in rows:
            part = json.loads(row["data"])
            tool_input = (part.get("state") or {}).get("input") or {}
            agent = tool_input.get("subagent_type")
            prompt = str(tool_input.get("prompt") or "").lower()
            if agent == "candidate-writer" and (
                "perform step 0+1" in prompt
                or ("after writing" in prompt and "step 0+1" in prompt)
                or "also perform" in prompt and "warm-start" in prompt
            ):
                violations.append(
                    f"candidate-writer was assigned contract/evaluation work: "
                    f"{tool_input.get('description') or 'unnamed task'}"
                )
        return violations

    def report(
        self,
        project_dir: Path,
        session_id: str | None,
        run_dir: Path | None,
        max_session_input: int,
        max_root_context: int,
    ) -> dict[str, Any]:
        root_id = self.select_root(project_dir, session_id, run_dir)
        sessions = self._sessions(root_id)
        root = sessions[0]
        run = _run_snapshot(run_dir)
        total = Usage()
        by_agent: dict[str, Usage] = defaultdict(Usage)
        counts = Counter()
        for session in sessions:
            total.add(session.usage)
            by_agent[session.agent].add(session.usage)
            counts[session.agent] += 1
        alerts = self._role_violations(root_id)
        active = [session for session in sessions if session.status == "active"]
        if any(session.depth > 1 for session in sessions):
            alerts.append("recursive delegation detected (a child session spawned another child)")
        unexpected = sorted({s.agent for s in sessions if s.agent not in KNOWN_AGENTS})
        if unexpected:
            alerts.append(f"unexpected agent types: {', '.join(unexpected)}")
        if root.latest_context >= max_root_context:
            alerts.append(
                f"root context is {_fmt_tokens(root.latest_context)} tokens "
                f"(threshold {_fmt_tokens(max_root_context)})"
            )
        expensive = [s for s in sessions[1:] if s.usage.input >= max_session_input]
        if expensive:
            worst = max(expensive, key=lambda s: s.usage.input)
            alerts.append(
                f"{len(expensive)} child session(s) exceed fresh-input threshold; "
                f"worst {worst.agent} {_fmt_tokens(worst.usage.input)}"
            )
        if run and not active and run["phase"] == "running":
            remaining = run["remaining"]
            suffix = "unbounded run" if remaining is None else f"{remaining} evaluations remain"
            alerts.append(f"root is idle while run phase is running ({suffix})")
        if run and run["pending_run_ids"] and not active:
            alerts.append(f"pending candidates without an active worker: {run['pending_run_ids']}")
        return {
            "source": "opencode", "root_session_id": root_id,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "root_status": root.status, "run": run, "total": asdict(total),
            "by_agent": {
                agent: {"sessions": counts[agent], **asdict(usage)}
                for agent, usage in sorted(by_agent.items(), key=lambda item: item[1].input, reverse=True)
            },
            "sessions": [{**asdict(session), "usage": asdict(session.usage)} for session in sessions],
            "alerts": alerts,
        }


def _claude_usage(path: Path) -> tuple[Usage, int]:
    messages: dict[str, Usage] = {}
    if not path.is_file():
        return Usage(), 0
    for index, raw in enumerate(path.read_text(errors="replace").splitlines()):
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        message = item.get("message") if isinstance(item.get("message"), dict) else {}
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else item.get("usage")
        if item.get("type") != "assistant" or not isinstance(usage, dict):
            continue
        cache_creation = usage.get("cache_creation_input_tokens")
        if isinstance(cache_creation, dict):
            cache_creation = sum(_integer(value) for value in cache_creation.values())
        key = str(message.get("id") or item.get("uuid") or index)
        messages[key] = Usage(
            input=_integer(usage.get("input_tokens")),
            output=_integer(usage.get("output_tokens")),
            cache_read=_integer(usage.get("cache_read_input_tokens")),
            cache_write=_integer(cache_creation),
        )
    total = Usage()
    for usage in messages.values():
        total.add(usage)
    return total, len(messages)


def claude_report(transcript: Path, run_dir: Path | None, max_session_input: int) -> dict[str, Any]:
    transcript = transcript.expanduser().resolve()
    candidates = [transcript]
    for folder in (transcript.parent / transcript.stem / "subagents", transcript.parent / "subagents"):
        if folder.is_dir():
            candidates.extend(sorted(folder.glob("*.jsonl")))
    sessions = []
    total = Usage()
    alerts = []
    for path in dict.fromkeys(candidates):
        usage, messages = _claude_usage(path)
        total.add(usage)
        role = "main" if path == transcript else path.stem
        sessions.append({
            "id": role, "agent": role, "messages": messages, "usage": asdict(usage),
            "updated_ms": int(path.stat().st_mtime * 1000) if path.exists() else 0,
        })
        if path != transcript and usage.input >= max_session_input:
            alerts.append(f"{role} used {_fmt_tokens(usage.input)} fresh input tokens")
    return {
        "source": "claude", "transcript": str(transcript),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": _run_snapshot(run_dir), "total": asdict(total), "sessions": sessions,
        "alerts": alerts,
    }


def _print_report(report: dict[str, Any], all_sessions: bool) -> None:
    total = Usage(**report["total"])
    print(
        f"HieraResearch live | {report['source']} | {report['generated_at']}\n"
        f"tokens: fresh={_fmt_tokens(total.input)} cache-read={_fmt_tokens(total.cache_read)} "
        f"cache-write={_fmt_tokens(total.cache_write)} output={_fmt_tokens(total.output)} "
        f"reasoning={_fmt_tokens(total.reasoning)} processed={_fmt_tokens(total.processed)} "
        f"cost=${total.cost:.3f}"
    )
    run = report.get("run")
    if run:
        budget = "unbounded" if run["budget"] is None else f"{run['evaluations_attempted']}/{run['budget']}"
        print(
            f"run: {run['task']}/{run['tag']} phase={run['phase']} evals={budget} "
            f"candidates={run['candidates']} pending={run['pending_run_ids']}"
        )
    if report["source"] == "opencode":
        print("\nby agent (fresh/cache/output/cost):")
        for agent, row in report["by_agent"].items():
            print(
                f"  {agent:28} n={row['sessions']:<3} {_fmt_tokens(row['input']):>7} "
                f"{_fmt_tokens(row['cache_read']):>8} {_fmt_tokens(row['output']):>7} ${row['cost']:.3f}"
            )
        views = report["sessions"]
        active = [row for row in views if row["status"] == "active"]
        expensive = sorted(views, key=lambda row: row["usage"]["input"], reverse=True)
        shown = views if all_sessions else list({row["id"]: row for row in active + expensive[:8]}.values())
        print("\nsessions (status/fresh/latest-context/age):")
        for row in shown:
            tool = f" | {row['current_tool']}" if row.get("current_tool") else ""
            print(
                f"  {'  ' * row['depth']}{row['agent'][:26]:26} {row['status']:6} "
                f"{_fmt_tokens(row['usage']['input']):>7} {_fmt_tokens(row['latest_context']):>7} "
                f"age={_fmt_age(row['updated_ms']):>5}{tool}"
            )
    else:
        print("\ntranscripts (fresh/cache/output/age):")
        for row in report["sessions"]:
            usage = row["usage"]
            print(
                f"  {row['agent'][:36]:36} {_fmt_tokens(usage['input']):>7} "
                f"{_fmt_tokens(usage['cache_read']):>8} {_fmt_tokens(usage['output']):>7} "
                f"age={_fmt_age(row['updated_ms'])}"
            )
    print("\nalerts:")
    if report["alerts"]:
        for alert in report["alerts"]:
            print(f"  ! {alert}")
    else:
        print("  none")


def claude_statusline() -> int:
    data = json.load(sys.stdin)
    current = (data.get("context_window") or {}).get("current_usage") or {}
    context = data.get("context_window") or {}
    model = (data.get("model") or {}).get("display_name") or "Claude"
    pct = int(context.get("used_percentage") or 0)
    total = _integer(context.get("total_input_tokens"))
    fresh = _integer(current.get("input_tokens"))
    cache = _integer(current.get("cache_read_input_tokens"))
    output = _integer(current.get("output_tokens"))
    cost = float((data.get("cost") or {}).get("total_cost_usd") or 0)
    flag = " ALERT" if pct >= 80 else (" WARN" if pct >= 65 else "")
    print(
        f"Hiera [{model}] ctx={_fmt_tokens(total)} ({pct}%){flag} | "
        f"fresh={_fmt_tokens(fresh)} cache={_fmt_tokens(cache)} out={_fmt_tokens(output)} | ${cost:.3f}"
    )
    return 0


def claude_subagent_statusline() -> int:
    data = json.load(sys.stdin)
    now_ms = int(time.time() * 1000)
    for task in data.get("tasks", []):
        tokens = _integer(task.get("tokenCount"))
        elapsed = max(0, (now_ms - _integer(task.get("startTime"))) // 1000)
        flag = " !" if tokens >= 50_000 else ""
        label = task.get("label") or task.get("name") or task.get("type") or "agent"
        content = f"{label} · {task.get('status', '?')} · {_fmt_tokens(tokens)} tok · {elapsed}s{flag}"
        print(json.dumps({"id": task.get("id"), "content": content}, separators=(",", ":")))
    return 0


def check_run_idle(run_dir: Path) -> int:
    run = _run_snapshot(run_dir)
    if run and run["phase"] == "running":
        remaining = run["remaining"]
        message = "run is still active" if remaining is None else f"run has {remaining} evaluations remaining"
        print(message)
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "opencode", "claude"], default="auto")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--session", help="OpenCode session id (a child id resolves to its root)")
    parser.add_argument("--db", type=Path, default=_default_opencode_db())
    parser.add_argument("--transcript", type=Path, help="Claude Code main transcript JSONL")
    parser.add_argument("--watch", nargs="?", const=2.0, type=float, help="Refresh every N seconds")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--all-sessions", action="store_true")
    parser.add_argument("--max-session-input", type=int, default=50_000)
    parser.add_argument("--max-root-context", type=int, default=150_000)
    parser.add_argument("--claude-statusline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--claude-subagent-statusline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--check-run-idle", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.claude_statusline:
        return claude_statusline()
    if args.claude_subagent_statusline:
        return claude_subagent_statusline()
    if args.check_run_idle:
        if args.run_dir is None:
            raise SystemExit("--check-run-idle requires --run-dir")
        return check_run_idle(args.run_dir)
    source = args.source
    if source == "auto":
        source = "claude" if args.transcript else "opencode"
    while True:
        if source == "opencode":
            reader = OpenCodeReader(args.db)
            try:
                report = reader.report(
                    args.project_dir, args.session, args.run_dir,
                    args.max_session_input, args.max_root_context,
                )
            finally:
                reader.close()
        else:
            if args.transcript is None:
                raise SystemExit("Claude mode requires --transcript")
            report = claude_report(args.transcript, args.run_dir, args.max_session_input)
        if args.watch and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_report(report, args.all_sessions)
        if not args.watch:
            return 2 if report.get("alerts") else 0
        time.sleep(max(0.5, args.watch))


if __name__ == "__main__":
    raise SystemExit(main())
