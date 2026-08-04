#!/usr/bin/env python3
"""Live Claude token and run-progress monitor for HieraResearch.

Transcript mode reads persisted Claude Code JSONL. The two status-line modes
consume Claude Code's JSON on stdin and do no I/O.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

from evaluation_budget import ATTEMPT_LOG, budget_status
from run_cfg import RunConfigError
from semantic_evidence import LIFECYCLE_TERMINAL_STATUSES


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
    try:
        strict_budget = budget_status(run_dir)
        attempted = max(attempted, strict_budget["evaluations_done"])
        cfg_error = None
    except RunConfigError as exc:
        # The watchdog must keep observing a run whose config is broken; surface
        # the error explicitly instead of pretending no budget is configured.
        cfg_error = str(exc)
    stored = ledger.get("run_state") if isinstance(ledger.get("run_state"), dict) else {}
    budget = cfg.get("max_evaluations")
    if not isinstance(budget, int) or isinstance(budget, bool):
        budget = stored.get("evaluation_budget")
    budget = budget if isinstance(budget, int) and not isinstance(budget, bool) else None
    if stored.get("phase") == "blocked" or state.get("phase") == "blocked":
        phase = "blocked"
        stop_condition = stored.get("active_stop_condition") or state.get("active_stop_condition")
    elif budget is not None and attempted >= budget:
        lifecycle_terminal = bool(records) and all(
            isinstance(record, dict)
            and record.get("status") in LIFECYCLE_TERMINAL_STATUSES
            for record in records
        )
        dag_revision = ledger.get("dag_revision", 0)
        experience = (
            ledger.get("experience")
            if isinstance(ledger.get("experience"), dict)
            else {}
        )
        experience_cursor = experience.get("dag_revision", 0)
        stale_experience = (
            isinstance(dag_revision, int)
            and not isinstance(dag_revision, bool)
            and isinstance(experience_cursor, int)
            and not isinstance(experience_cursor, bool)
            and dag_revision > experience_cursor
        )
        if not lifecycle_terminal:
            phase = "running"
            stop_condition = "budget_reached_pending_resolution"
        elif stale_experience:
            phase = "running"
            stop_condition = "final_experience_refresh_required"
        else:
            phase = "completed"
            stop_condition = "evaluation_budget_reached"
    else:
        phase = "running"
        stop_condition = "none"
    tracked_paths = (
        ledger_path,
        state_path,
        run_dir / ATTEMPT_LOG,
        run_dir / "environment_preflight.json",
    )
    mtimes = [path.stat().st_mtime for path in tracked_paths if path.exists()]
    return {
        "run_dir": str(run_dir),
        "task": ledger.get("task") or state.get("task"),
        "tag": ledger.get("tag") or state.get("tag") or run_dir.name,
        "phase": phase,
        "active_stop_condition": stop_condition,
        "candidates": len(records),
        "pending_run_ids": [r.get("run_id") for r in records if r.get("status") == "pending"],
        "evaluations_attempted": attempted,
        "preflight_attempts": sum(
            _integer(record.get("preflight_attempts")) for record in records
        ),
        "preflight_failures": sum(
            _integer(record.get("preflight_failures")) for record in records
        ),
        "feasibility_rejections": sum(
            _integer(record.get("feasibility_rejections")) for record in records
        ),
        "budget": budget,
        "remaining": None if budget is None else max(0, budget - attempted),
        "config_error": cfg_error,
        "last_progress_ms": int(max(mtimes) * 1000) if mtimes else 0,
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


def _print_report(report: dict[str, Any]) -> None:
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
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--transcript", type=Path, help="Claude Code main transcript JSONL")
    parser.add_argument("--watch", nargs="?", const=2.0, type=float, help="Refresh every N seconds")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-session-input", type=int, default=50_000)
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
    if args.transcript is None:
        raise SystemExit("transcript reporting requires --transcript")
    while True:
        report = claude_report(args.transcript, args.run_dir, args.max_session_input)
        if args.watch and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_report(report)
        if not args.watch:
            return 2 if report.get("alerts") else 0
        time.sleep(max(0.5, args.watch))


if __name__ == "__main__":
    raise SystemExit(main())
