"""Round-level recovery: a failed work unit is closed and excluded, the run goes on.

A unit (generation seat, rewrite climb, tune bout, refresh) that raises an
unexpected exception is caught at its own boundary instead of blocking the
run. The default policy keys the failure by signature (exception type + the
failing tool subcommand or innermost repo frame): the first occurrence
excludes the failed target from the action for the rest of this process;
the same signature on a different target disables the whole action. The
scheduler then picks from what remains. GENERATION is never disabled for
the run: disabling it only pauses it until the next optimization round has
changed the pool (``lift``), so semantic search always resumes.
``RunBlocked`` is never caught here — only the hard classes block.

A signature this run has not seen yet may first go to a registered triage
chooser (the ``failure-triage`` role), which picks the scope from the same
menu plus ``retry_once``. Its pick is checked against the menu; anything
else, or a triage failure, falls back to the default policy. Recurrences
of a signature always follow the default escalation.
"""

from __future__ import annotations

import json
import subprocess
import threading
import traceback
from collections import Counter
from pathlib import Path

from tools.scheduler.store import SchedulerStore

REPO_ROOT = Path(__file__).resolve().parents[2]

GENERATION = "GENERATION"
REWRITE = "REWRITE"
TUNE = "TUNE"
REFRESH = "REFRESH"

SCOPES = ("exclude_target", "disable_action", "retry_once")

_lock = threading.Lock()
_state: dict[str, dict] = {}
_triage: dict[str, object] = {}


def _run_state(run_dir) -> dict:
    with _lock:
        return _state.setdefault(
            str(run_dir), {"excluded": {}, "disabled": set(), "signatures": {}})


def reset(run_dir) -> None:
    """Fresh entry into a run starts with every action available."""
    _state.pop(str(run_dir), None)
    _triage.pop(str(run_dir), None)


def set_triage(run_dir, chooser) -> None:
    """``chooser(payload) -> {"scope", "rationale"} | None`` for unseen
    signatures; it must not raise past its own failures."""
    _triage[str(run_dir)] = chooser


def signature(action: str, exc: BaseException) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        argv = [str(part) for part in (exc.cmd if isinstance(exc.cmd, (list, tuple))
                                        else str(exc.cmd).split())]
        script = next((i for i, part in enumerate(argv) if part.endswith(".py")), None)
        where = (" ".join(Path(p).name if i == 0 else p
                          for i, p in enumerate(argv[script:script + 2]))
                 if script is not None else (argv[0] if argv else "?"))
    else:
        frames = traceback.extract_tb(exc.__traceback__)
        inside = [f for f in frames if f.filename.startswith(str(REPO_ROOT))]
        frame = (inside or frames or [None])[-1]
        where = f"{Path(frame.filename).name}:{frame.name}" if frame else "?"
    return f"{action}:{type(exc).__name__}@{where}"


def unit_failed(run_dir, events, action: str, target: str | None,
                exc: BaseException) -> str:
    """Apply the recovery policy to one failed unit; returns the chosen
    scope (one of ``SCOPES``). Seats and rewrite channels fail concurrently:
    state changes hold the lock, the triage session does not."""
    state = _run_state(run_dir)
    sig = signature(action, exc)
    with _lock:
        unseen = sig not in state["signatures"]
        targets = state["signatures"].setdefault(sig, set())
        repeated = bool(targets - {target})
        targets.add(target)
        disabled_now = sorted(state["disabled"])
        excluded_now = {a: sorted(t) for a, t in state["excluded"].items()}
    detail = str(getattr(exc, "stderr", None) or exc)[-2000:]
    trace = "".join(traceback.format_exception(exc))[-3000:]
    events.emit("unit_failed", action=action, target=target, signature=sig,
                error=detail, traceback=trace)
    frontier = [scope for scope in SCOPES
                if target is not None or scope != "exclude_target"]
    choice, chooser, rationale = None, "default", None
    triage = _triage.get(str(run_dir))
    if unseen and triage is not None:
        picked = triage({
            "action": action, "target": target, "signature": sig,
            "error": detail, "traceback": trace, "menu": frontier,
            "disabled": disabled_now, "excluded": excluded_now,
        })
        if isinstance(picked, dict) and picked.get("scope") in frontier:
            choice, chooser = picked["scope"], "triage"
            rationale = picked.get("rationale")
    if choice is None:
        choice = ("disable_action" if repeated or target is None
                  else "exclude_target")
    with _lock:
        if choice != "retry_once" and target is not None:
            state["excluded"].setdefault(action, set()).add(str(target))
        if choice == "disable_action":
            state["disabled"].add(action)
        state["version"] = state.get("version", 0) + 1
    events.emit("recovery_choice", layer="round", signature=sig,
                frontier=frontier, choice=choice,
                chooser=chooser, action=action, target=target,
                rationale=rationale)
    return choice


def disable(run_dir, events, action: str, sig: str, reason: str) -> None:
    """Disable an action whose failures form a systemic pattern (e.g. a streak
    of isomorphic seat skips); the run goes on with the remaining actions."""
    state = _run_state(run_dir)
    with _lock:
        state["disabled"].add(action)
        state["version"] = state.get("version", 0) + 1
    events.emit("recovery_choice", layer="round", signature=sig,
                frontier=["disable_action"],
                choice="disable_action", chooser="default", action=action,
                reason=reason)


def lift(run_dir, events, action: str) -> None:
    """End a pause (the GENERATION contract: optimization has run since)."""
    state = _run_state(run_dir)
    with _lock:
        if action not in state["disabled"]:
            return
        state["disabled"].discard(action)
        state["version"] = state.get("version", 0) + 1
    events.emit("recovery_lifted", layer="round", action=action)


def excluded(run_dir, action: str) -> set[str]:
    state = _run_state(run_dir)
    with _lock:
        return set(state["excluded"].get(action, ()))


def disabled(run_dir, action: str) -> bool:
    return action in _run_state(run_dir)["disabled"]


def version(run_dir) -> int:
    """Bumped on every unit failure; a round that changed it made progress
    in the recovery sense (the frontier shrank)."""
    return _run_state(run_dir).get("version", 0)


def close_decision(run_dir, repo_root, cmd, selection: dict, action: str,
                   consumed: int = 0) -> None:
    """Close a failed unit's scheduler decision as infra_failure unless its
    outcome is already bound; ``consumed`` is the objective evaluations the
    unit ran before failing. A failure here is ledger-integrity (hard):
    it propagates."""
    decision_id = selection.get("decision_id")
    if not decision_id:
        return
    still_open = {row.get("decision_id")
                  for row in SchedulerStore(run_dir).unbound_decisions()}
    if decision_id not in still_open:
        return
    cmd(["python", "tools/scheduler/cli.py", "record",
         "--ledger", Path(run_dir) / "ledger.json",
         "--decision-id", decision_id, "--action", action,
         "--run-id", str(selection.get("run_id")), "--consumed", str(consumed),
         "--status", "infra_failure"], repo_root)


def summarize(run_dir, events) -> dict:
    """Run-end cost summary over driver_events.jsonl: guard denies and
    degradations per guard/role, unit failures per signature, and recovery
    choices per (layer, chooser, choice). Written to failure_summary.json."""
    counts = {key: Counter() for key in
              ("guard_denied", "guard_degraded", "unit_failed",
               "recovery_choice")}
    path = Path(run_dir) / "driver_events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    for line in lines:
        row = json.loads(line)
        kind = row.get("kind")
        if kind in ("guard_denied", "guard_degraded"):
            counts[kind][f"{row.get('guard')}/{row.get('role')}"] += 1
        elif kind == "unit_failed":
            counts[kind][row.get("signature")] += 1
        elif kind == "recovery_choice":
            counts[kind][f"{row.get('layer')}/{row.get('chooser')}/"
                         f"{row.get('choice')}"] += 1
    summary = {kind: dict(counter.most_common())
               for kind, counter in counts.items()}
    (Path(run_dir) / "failure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    events.emit("failure_summary", **{kind: sum(c.values())
                                      for kind, c in counts.items()})
    return summary
