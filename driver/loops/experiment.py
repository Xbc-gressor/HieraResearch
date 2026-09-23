"""Run the deterministic autoresearch experiment protocol.

Responsibility is deliberately split three ways:

* this module owns sequencing, lifecycle checks, and recovery;
* ``tools/`` owns deterministic decisions and durable state changes;
* role sessions own generation and judgment.

The high-level lifecycle is:

    setup or resume
        -> reconcile an optional provided baseline
        -> repeat rounds:
             recover pending candidates
             refresh bounded experience when required
             round_v1 (default): generate one candidate generation until the
               cycle's candidate threshold is met, then run one optimization
               phase (rewrite climbs, then tune bouts) over the whole pool
             other schedulers: generate one generation, then at most one
               decoupled tuning bout
        -> complete on budget exhaustion (evaluations or wall clock) or
           quiescence

Within one generation the admitted seats are implemented on a bounded
session channel (``pipeline.session_concurrency`` driver threads): while one
seat's extractor is blocked on its warm-screening job, another seat's writer
or extractor runs. The GPU channel stays serial (the device lease), each
candidate's own chain stays ordered, and the admission gates are unchanged,
so evaluation facts and attribution are exactly those of the serial loop.

The helpers below are grouped by responsibility. Recovery policy stays close
to the operation it recovers, while ``run_experiment`` remains a compact map of
the complete lifecycle.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..events import EventsLog
from ..jobs import DriverJobError, _arm_exit_hooks, execute_driver_job
from ..metadata import warn_on_mismatch, write_metadata
from ..receipts import ReceiptStore
from ..resources import ResourceUnavailable
from ..roles import (
    REPO_ROOT,
    ROLES,
    InvocationContext,
    driver_job_handoff_problem,
    record_status,
)
from ..session import InvocationFailed, invocation_problem_class
from ..status import budget_status, compact_status
from . import background_audit
from . import common
from . import phase_c
from . import rounds
from .common import RunBlocked
from tools import competition_policy, mlebench_finalize
from tools.evaluation_budget import budget_status as objective_budget_status
from tools.evaluation_budget import phase_c_attempts
from tools.evaluation_budget import time_budget as run_time_budget
from tools.objective_brief import (
    build_brief,
    compact_line,
    ensure_brief,
    render_block,
)
from tools.scheduler.contract import (
    DEFAULT_K_EVAL,
    MIN_GENERATION_K_EVAL,
    ResourceContract,
)
from tools.scheduler.donor import build_donor_snapshot, donors_dir
from tools.semantic_routes import is_not_applicable, validate_route_provenance

TRANSFER_SCHEDULER_POLICY = "anchor_transfer_challenger_v1"
# Mirrors tune_tools.GLOBAL_DONOR_TRANSFER_FILENAME; the tuners package is a
# script-level package the driver cannot import in-process.
_DONOR_RECEIPT_FILENAME = "_global_donor_transfer.json"

_RECONCILE_GUIDANCE = (
    " Reconcile from the AUTHORITATIVE artifacts (ledger, "
    "tune_report.json, attempt log, finalization artifacts) — never "
    "the receipt. Complete any valid pending finalization ONLY via "
    "tools/finalize_tuning.py, or submit a corrected truthful no-op "
    "receipt.")


# =============================================================================
# Shared artifact and role-session helpers
# =============================================================================


def _objective_line(task_toml: dict, **scores) -> str:
    """The one-line objective view every proposing/implementing role sees."""
    return compact_line(build_brief(task_toml, **scores))


def _objective_block(run_dir: Path) -> str | None:
    """The rendered brief block for payload prefixes, when the run wrote one."""
    try:
        brief = json.loads(
            (run_dir / "objective_brief.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return render_block(brief) if isinstance(brief, dict) else None


def _brief(run_dir: Path, repo_root: Path, cmd) -> dict:
    out = cmd(["python", "tools/ledger.py", "brief",
               "--ledger", run_dir / "ledger.json"], repo_root)
    return json.loads(out.stdout)


def _ledger_records(run_dir: Path) -> list[dict]:
    path = run_dir / "ledger.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("records", [])


def _tune_flag(run_dir: Path, run_id: str) -> bool:
    for record in _ledger_records(run_dir):
        if record.get("run_id") == run_id:
            return bool(record.get("tune"))
    return False


def _scheduler_stopped(run_dir: Path) -> bool:
    """True when a stored scheduler's latest decision is a terminal STOP.

    STOP is absorbing: it requires both no affordable bout and no budget
    for another generation round, and the remaining budget only shrinks.
    Waiting for two zero-progress rounds after that only spends more role
    sessions confirming a state the scheduler has already declared final.
    Legacy runs have no `.scheduler/` store, so this reads as False there.
    """
    path = run_dir / ".scheduler" / "decisions.jsonl"
    if not path.is_file():
        return False
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "scheduler_decision":
            last = row
    return bool(last) and last.get("selected_action") == "STOP"


def _invoke(runner, store, role_name, task, tag, run_dir, *,
            run_id=None, round_no=None, extra=None, resume_from=None,
            inline_payload=None, writer_attempt_dir=None) -> dict:
    """Invoke one role and return its persisted receipt plus invocation id."""
    # Admission and attempt reservation are atomic with respect to a sibling
    # blocking the run. Never spend an attempt on a refused session, and never
    # hold the block lock while the admitted session runs.
    with _block_lock:
        reason = _block_reason.get(str(run_dir))
        if reason is not None:
            raise RunBlocked(reason)
        if writer_attempt_dir is not None:
            _register_candidate_writer_attempt(writer_attempt_dir)
    inv_id = store.issue_invocation_id()
    resume = (store.load_session_id(role_name, resume_from)
              if resume_from is not None else None)
    ctx = InvocationContext(task=task, tag=tag, run_dir=run_dir,
                            invocation_id=inv_id, run_id=run_id,
                            round_no=round_no, extra=extra or {},
                            resume_session_id=resume,
                            inline_payload=inline_payload)
    try:
        runner.run(ROLES[role_name], ctx)
    except InvocationFailed as exc:
        # The failed invocation may still have a valid persisted SDK session.
        # Callers that own a repair path need its identity even though no
        # terminal receipt was accepted.
        if exc.invocation_id is None:
            exc.invocation_id = inv_id
        raise
    receipt_path = store.receipt_path(role_name, inv_id)
    return json.loads(receipt_path.read_text(encoding="utf-8")), inv_id


def _invoke_with_driver_jobs(
    runner,
    store,
    role_name,
    task,
    tag,
    run_dir,
    *,
    run_id=None,
    round_no=None,
    extra=None,
    resume_from=None,
    repo_root=REPO_ROOT,
    job_runner=execute_driver_job,
) -> tuple[dict, int]:
    """Run a role, synchronously executing each typed objective-job handoff."""
    receipt, inv_id = _invoke(
        runner,
        store,
        role_name,
        task,
        tag,
        run_dir,
        run_id=run_id,
        round_no=round_no,
        extra=extra,
        resume_from=resume_from,
    )
    jobs_run = 0
    while isinstance(receipt.get("driver_job"), dict):
        jobs_run += 1
        if jobs_run > 12:
            raise InvocationFailed(
                role_name,
                ["driver job handoff exceeded 12 requests without a terminal receipt"],
                invocation_id=inv_id,
            )
        ctx = InvocationContext(
            task=task,
            tag=tag,
            run_dir=run_dir,
            invocation_id=inv_id,
            run_id=run_id,
            round_no=round_no,
            extra=extra or {},
        )
        handoff_problem = driver_job_handoff_problem(role_name, receipt)
        _refuse_if_blocked(run_dir)
        try:
            if handoff_problem:
                raise DriverJobError(handoff_problem)
            result = job_runner(
                role_name,
                ctx,
                receipt["driver_job"],
                repo_root=repo_root,
            )
        except ResourceUnavailable as exc:
            # The GPU never became available before the deadline. That is not
            # a candidate observation and not a job result for the session:
            # the invocation ends here and the caller settles the candidate
            # from the budget state (zero attempts at a reached stop).
            raise InvocationFailed(
                role_name,
                [f"objective job could not start before the deadline: {exc}"],
                invocation_id=inv_id,
            ) from exc
        except (DriverJobError, OSError, subprocess.SubprocessError) as exc:
            result = {
                "kind": receipt["driver_job"].get("kind"),
                "accepted": False,
                "error": str(exc),
            }
        receipt, next_inv_id = _invoke(
            runner,
            store,
            role_name,
            task,
            tag,
            run_dir,
            run_id=run_id,
            round_no=round_no,
            extra={
                **(extra or {}),
                "driver_job_result": json.dumps(result, sort_keys=True),
            },
            resume_from=inv_id,
        )
        inv_id = next_inv_id
    return receipt, inv_id


# =============================================================================
# Blocking and reconciliation
# =============================================================================


# Run-wide block state. Seats run on driver threads; the first channel to
# block persists the blocked phase, every other channel stops accepting new
# sessions or job handoffs at its next boundary (an in-flight GPU job always
# drains to completion — evaluations are expensive and their facts stay
# valid), and run_experiment unwinds once the channels have joined.
_block_lock = threading.Lock()
_block_reason: dict[str, str] = {}  # run_dir -> first block reason
_block_persisted: set[str] = set()


def _reset_block_state(run_dir) -> None:
    with _block_lock:
        _block_reason.pop(str(run_dir), None)
        _block_persisted.discard(str(run_dir))


def _refuse_if_blocked(run_dir) -> None:
    with _block_lock:
        reason = _block_reason.get(str(run_dir))
    if reason is not None:
        raise RunBlocked(reason)


def _or_block(run_dir, repo_root, cmd, events, reason: str):
    """Stop new work and persist the first blocker before unwinding.

    Delivery belongs to the joined run boundary, never a seat's stack:
    another channel may still be producing objective evidence. A failed
    persistence is not a persisted block; the boundary may retry it.
    """
    key = str(run_dir)
    with _block_lock:
        original = _block_reason.setdefault(key, reason)
        if key not in _block_persisted:
            common.block(run_dir, repo_root, cmd, events, original)
            _block_persisted.add(key)
        else:
            events.emit("blocked_secondary", reason=reason)
    raise RunBlocked(original)


def _finish_blocked_run(run_dir, repo_root, cmd, events) -> None:
    """After all channels join, attempt delivery without losing the block."""
    key = str(run_dir)
    with _block_lock:
        reason = _block_reason.get(key)
        persisted = key in _block_persisted
    if reason is None:
        return
    if not persisted:
        # A worker's persistence failure can accompany another worker's
        # RunBlocked. Do not let the latter mask a non-durable stop.
        try:
            _or_block(run_dir, repo_root, cmd, events, reason)
        except RunBlocked:
            pass
    try:
        annotation = _attempt_degraded_delivery(run_dir, repo_root, cmd, events)
    except Exception as exc:  # best effort, including settlement/formatting
        events.emit("degraded_submit", status="failed", error=repr(exc))
        return
    if annotation:
        # The original block is already durable even if annotation fails.
        try:
            common.set_phase(run_dir, repo_root, cmd, "blocked",
                             stop_condition=f"{reason}; {annotation}")
        except Exception as exc:  # best-effort annotation, block is durable
            detail = getattr(exc, "stderr", None) or str(exc)
            events.emit("degraded_submit_annotation_failed",
                        error=str(detail)[-1500:])


# =============================================================================
# Seat skips, run-level consecutive-failure trip, degraded delivery
# =============================================================================
#
# P1/P3: one failed session settles deterministically — its seat is skipped
# (ledger `aborted`: terminal, no observation, no DAG bump) and the run
# continues. Only a STREAK of consecutive skips with isomorphic failure
# signatures (same role + same frozen problem class, see
# session.problem_class) is systemic degradation and blocks the run. A seat
# that completes normally resets the streak.

_SEAT_SKIP_TRIP_LIMIT = 2
_seat_skip_state: dict[str, dict] = {}
# run_dir -> {"task", "submission_command", "data_dir"}: the operator's
# finalization contract, registered by run_experiment when automatic
# finalization is configured.
_delivery_cfg: dict[str, dict] = {}


def _seat_skip_trip_limit(run_dir: Path) -> int:
    """Consecutive isomorphic skips that block the run
    (framework_cfg ``pipeline.seat_skip_trip_limit``; default 2)."""
    try:
        section = json.loads((run_dir / "framework_cfg.json").read_text(
            encoding="utf-8")).get("pipeline")
    except (OSError, json.JSONDecodeError):
        return _SEAT_SKIP_TRIP_LIMIT
    value = section.get("seat_skip_trip_limit") if isinstance(section, dict) \
        else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return _SEAT_SKIP_TRIP_LIMIT
    return value


def _reset_seat_skip_state(run_dir) -> None:
    with _block_lock:
        _seat_skip_state.pop(str(run_dir), None)
    _delivery_cfg.pop(str(run_dir), None)


def _seat_skip_count(run_dir) -> int:
    with _block_lock:
        return int(_seat_skip_state.get(str(run_dir), {}).get("count", 0))


def _note_seat_progress(run_dir) -> None:
    """One seat completed normally: the consecutive-skip streak restarts."""
    with _block_lock:
        state = _seat_skip_state.get(str(run_dir))
        if state is not None:
            state["signature"] = None
            state["streak"] = 0


def _note_seat_skip(run_dir, repo_root, cmd, events, *, role, problems,
                    run_id=None) -> None:
    """Register one skipped seat. A streak of consecutive isomorphic skips
    (same role + problem class) trips the run-level breaker and blocks."""
    problems = [str(p) for p in problems]
    signature = (str(role),
                 invocation_problem_class(problems or ["postcondition"]))
    with _block_lock:
        state = _seat_skip_state.setdefault(
            str(run_dir), {"signature": None, "streak": 0, "count": 0})
        state["count"] += 1
        if state["signature"] == signature:
            state["streak"] += 1
        else:
            state["signature"] = signature
            state["streak"] = 1
        streak = state["streak"]
    if streak >= _seat_skip_trip_limit(run_dir):
        _or_block(
            run_dir, repo_root, cmd, events,
            f"consecutive seat failures ({streak}x) with isomorphic "
            f"signature (role={signature[0]}, "
            f"problem_class={signature[1]}): systemic degradation, "
            "not an isolated failure")


def _skip_candidate(run_dir, repo_root, cmd, events, run_id, *, role,
                    problems, count_failure=True) -> None:
    """End a failed seat from its current evidence, then register the failure.

    A retry may have run objectives or even settled the record before its
    session failed. Only a still-pending seat with no objective evidence can
    be aborted. Block cleanup uses the same settlement without adding a new
    failure to the run's streak.
    """
    problems = [str(p) for p in problems]
    if record_status(run_dir, run_id) in ("pending", None):
        view = budget_status(run_dir, repo_root, cmd)
        row = next((r for r in view.get("per_candidate", [])
                    if r.get("run_id") == str(run_id)), {})
        candidate_dir = run_dir / "candidates" / str(run_id)
        if int(row.get("evals") or 0) > 0 \
                or _finite_warm_score(candidate_dir) is not None:
            _settle_at_deadline(run_dir, run_id, repo_root, cmd, events)
        else:
            try:
                cmd(["python", "tools/ledger.py", "resolve-aborted",
                     "--ledger", run_dir / "ledger.json", "--run-id", str(run_id),
                     "--problems", json.dumps(problems)], repo_root)
            except subprocess.CalledProcessError as exc:
                _or_block(run_dir, repo_root, cmd, events,
                          f"resolve-aborted refused for {run_id}: "
                          f"{(exc.stderr or str(exc)).strip()[-1500:]}")
    events.emit("candidate_skipped", run_id=str(run_id), role=str(role),
                problem_class=invocation_problem_class(
                    problems or ["postcondition"]),
                outcome=record_status(run_dir, run_id), problems=problems[:5])
    if count_failure:
        _note_seat_skip(run_dir, repo_root, cmd, events, role=role,
                       problems=problems, run_id=str(run_id))


def _settle_for_delivery(run_dir, run_id, repo_root, cmd, events) -> None:
    """Close a pending seat after channels drain; cleanup is not a new fault."""
    if record_status(run_dir, run_id) in ("pending", None):
        _skip_candidate(run_dir, repo_root, cmd, events, run_id,
                        role="driver", count_failure=False,
                        problems=["seat pending at block: settled for "
                                  "degraded delivery"])


def _attempt_degraded_delivery(run_dir, repo_root, cmd, events) -> str | None:
    """One best-effort export after the block persists and channels drain.

    Requires an operator finalization contract, a settled keep/discard
    incumbent, and enough remaining deadline; everything else is skipped
    loudly (event) and the persisted block remains unchanged. Returns the
    ``degraded_submit`` annotation on success."""
    delivery = _delivery_cfg.get(str(run_dir))
    if delivery is None:
        return None
    try:
        cfg = json.loads((run_dir / "framework_cfg.json").read_text(
            encoding="utf-8"))
        deadline = float(cfg.get("deadline"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        deadline = float("nan")
    remaining = deadline - time.time()
    records = _ledger_records(run_dir)
    if not math.isfinite(remaining) or remaining < 60:
        settled = [r for r in records
                   if r.get("status") in ("keep", "discard")]
        events.emit("degraded_submit", status="skipped",
                    remaining_seconds=(remaining if math.isfinite(remaining)
                                       else None),
                    settled_candidates=len(settled))
        return None
    for record in records:
        if str(record.get("status") or "pending") == "pending":
            _settle_for_delivery(run_dir, str(record.get("run_id")),
                                 repo_root, cmd, events)
    records = _ledger_records(run_dir)
    settled = [r for r in records
               if r.get("status") in ("keep", "discard")]
    if not settled:
        events.emit("degraded_submit", status="skipped",
                    remaining_seconds=remaining, settled_candidates=0)
        return None
    remaining = deadline - time.time()
    if remaining <= 0:
        events.emit("degraded_submit", status="skipped", remaining_seconds=remaining)
        return None
    _, result = mlebench_finalize.run_submission(
        task=delivery["task"],
        run_dir=run_dir,
        data_dir=delivery["data_dir"],
        submission_command=delivery["submission_command"],
        deadline=deadline,
    )
    status = result["status"]
    if status == "deadline_expired":
        status = "skipped"
    events.emit(
        "degraded_submit",
        status=status,
        returncode=result.get("returncode"),
        error=result.get("error"),
        started_at_unix=result.get("started_at_unix"),
        ended_at_unix=result.get("ended_at_unix"),
    )
    return "degraded_submit" if status == "success" else None


def _complete_run(run_dir, repo_root, cmd, events, terminal_leftover=False,
                  stop_condition=None) -> None:
    """Persist normal completion, translating a refusal into a blocked run.

    ``stop_condition`` names the actual early-stop cause (quiescent,
    scheduler stop, unspendable budget tail); the None default keeps the
    budget/clock-derived reason for normal exhaustion completions. A run
    that skipped seats is always suffixed so the completion annotation can
    be audited (P4)."""
    if _seat_skip_count(run_dir):
        if stop_condition is None:
            try:
                cfg = json.loads((run_dir / "framework_cfg.json").read_text(
                    encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cfg = {}
            budget = cfg.get("max_evaluations")
            deadline = cfg.get("deadline")
            stop_condition = (
                "time_budget_reached"
                if (not isinstance(budget, int) or isinstance(budget, bool))
                and isinstance(deadline, (int, float))
                and not isinstance(deadline, bool)
                else "evaluation_budget_reached"
            )
        stop_condition = f"{stop_condition}_seats_skipped"
    try:
        common.set_phase(run_dir, repo_root, cmd, "completed",
                         stop_condition=stop_condition,
                         terminal_leftover=terminal_leftover)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or str(exc)).strip()
        _or_block(run_dir, repo_root, cmd, events,
                  f"set-phase completed refused: {detail}")


def _time_reached(run_dir) -> bool:
    """Past deadline − final_reserve: no session may start (D1)."""
    return bool(run_time_budget(run_dir).get("time_reached"))


def _refresh(runner, store, task, tag, run_dir, repo_root, cmd, events) -> None:
    """Refresh bounded experience, with one artifact-aware retry.

    Skipped once the run's cutoff has passed: the final refresh is the one
    session that used to run past the deadline (two deadline_expired
    submissions), and completion no longer depends on it (set-phase
    tolerates the unprocessed terminal delta at the cutoff)."""
    if _time_reached(run_dir):
        events.emit("refresh_skipped", reason="time_reached")
        return
    try:
        _invoke(runner, store, "experience-extractor", task, tag, run_dir)
        return
    except InvocationFailed:
        # A refresh may have started before cutoff and been cancelled at it.
        # Re-entering the session gate would turn normal exhaustion into a
        # blocked run, consuming finalization time on an external resume.
        if _time_reached(run_dir):
            events.emit("refresh_skipped", reason="time_reached")
            return
    try:  # one retry with reconciliation context, then block
        brief = _brief(run_dir, repo_root, cmd)
        _invoke(runner, store, "experience-extractor", task, tag, run_dir,
                extra={"reconcile_note":
                       "prior refresh failed postconditions; current ledger brief: "
                       + json.dumps(brief, sort_keys=True)})
        return
    except InvocationFailed as exc:
        # Also covers cutoff during reconciliation or the retry itself.
        if _time_reached(run_dir):
            events.emit("refresh_skipped", reason="time_reached")
            return
        _or_block(run_dir, repo_root, cmd, events,
                  f"experience refresh failed: {exc.problems}")


def _ideate(runner, store, task, tag, run_dir, round_no, repo_root, cmd,
            events, task_toml=None) -> list[dict]:
    """Generate one action batch and ensure every returned action was admitted.

    Degradation (P3/P4): a failed ideation retry no longer blocks — the
    generation degrades to its admitted subset (an empty generation flows
    into the zero-progress → quiescent completion), the event records it,
    and the run-level streak breaker catches a systemic pattern. The
    judged-slate arm is deliberately NOT covered: its admission is atomic,
    so partial degradation there would violate the manifest contract."""
    objective = _objective_line(task_toml or {})

    def admitted_missing(actions: list[dict]) -> list[str]:
        admitted = {r.get("run_id") for r in _ledger_records(run_dir)}
        return [a.get("run_id") for a in actions if a.get("run_id") not in admitted]

    def degrade(reason: str, problems: list[str]) -> list[dict]:
        events.emit("generation_degraded", round_no=round_no, reason=reason)
        _note_seat_skip(run_dir, repo_root, cmd, events,
                        role="idea-generator", problems=problems)
        # Empty generation: flows into zero-progress → quiescent completion.
        return []

    try:
        receipt, _ = _invoke(runner, store, "idea-generator", task, tag,
                             run_dir, round_no=round_no,
                             extra={"objective": objective})
    except InvocationFailed:
        receipt = None
    else:
        if not admitted_missing(receipt.get("actions", [])):
            _note_seat_progress(run_dir)
            return receipt.get("actions", [])
    # one retry whose context reconciles against what is already admitted
    note = {
        "reconcile_note":
            "Records already admitted for this generation stand; never "
            "re-admit them. Complete only the missing work.",
        "admitted_run_ids": [r.get("run_id") for r in _ledger_records(run_dir)],
        "objective": objective,
    }
    try:
        receipt, _ = _invoke(runner, store, "idea-generator", task, tag,
                             run_dir, round_no=round_no, extra=note)
    except InvocationFailed as exc:
        return degrade(f"idea-generator failed after retry: {exc.problems}",
                       [str(p) for p in exc.problems])
    missing = admitted_missing(receipt.get("actions", []))
    if missing:
        admitted = {r.get("run_id") for r in _ledger_records(run_dir)}
        subset = [action for action in receipt.get("actions", [])
                  if action.get("run_id") in admitted]
        if not subset:
            return degrade(
                f"idea actions not admitted after retry: {missing}",
                [f"actions not admitted after retry: {missing}"])
        events.emit("generation_degraded", round_no=round_no,
                    missing_run_ids=missing)
        _note_seat_skip(run_dir, repo_root, cmd, events,
                        role="idea-generator",
                        problems=[f"actions not admitted after retry: "
                                  f"{missing}"])
        return subset
    _note_seat_progress(run_dir)
    return receipt.get("actions", [])


# =============================================================================
# Judged-slate generation (semantic_search.policy == "judged_slate")
# =============================================================================
#
# One generation: action lanes -> per-lane proposals -> pool + shared A1
# context -> two independent judge rollouts (+ one boundary rollout on
# disagreement; coverage fallback on failure) -> immutable manifest -> one
# PLAN per seat -> one atomic two-seat admission -> the regular
# materialize/implement pipeline.  Every decision lives in tools/ CLIs; this
# section owns lifecycle, sessions, and resume.  The manifest is the commit
# point: before it the same gen_no is rebuilt, after it judges never re-run.


_SLATE_REGULAR_STAGES = ("regular-0", "regular-1")


def _semantic_policy(run_dir: Path) -> str | None:
    """Read the semantic policy frozen into this run's framework config."""
    config_path = run_dir / "framework_cfg.json"
    section = json.loads(config_path.read_text(encoding="utf-8")).get(
        "semantic_search")
    return section.get("policy") if isinstance(section, dict) else None


def _scheduler_policy(run_dir: Path) -> str | None:
    """Read the scheduler policy frozen into this run's framework config."""
    config_path = run_dir / "framework_cfg.json"
    section = json.loads(config_path.read_text(encoding="utf-8")).get("tuner")
    return section.get("scheduler_policy") if isinstance(section, dict) else None


def _slate_route_arm(run_dir: Path) -> int:
    """n_route_sketches from the frozen framework config (0 = arm inactive)."""
    config_path = run_dir / "framework_cfg.json"
    section = json.loads(config_path.read_text(encoding="utf-8")).get(
        "semantic_search")
    sketches = section.get("n_route_sketches", 0) if isinstance(section, dict) else 0
    return sketches if isinstance(sketches, int) and not isinstance(
        sketches, bool) else 0


def _slate_cmd(run_dir, repo_root, cmd, events, args, what: str):
    """A tools/ subprocess whose failure blocks the run (fail closed)."""
    try:
        return cmd(args, repo_root)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or str(exc)).strip()
        _or_block(run_dir, repo_root, cmd, events,
                  f"judged-slate {what} failed: {detail}")


def _generation_abort_path(gen_dir: Path) -> Path:
    return gen_dir / "generation.aborted.json"


def _generation_aborted(gen_dir: Path) -> bool:
    return _generation_abort_path(gen_dir).is_file()


def _abort_slate_generation(run_dir, gen_dir, repo_root, cmd, events, *,
                            reason: str, role: str,
                            problems: list[str]) -> None:
    """Abort one generation whose plan phase cannot complete within its
    attempt budget, instead of blocking the run on a permanent re-block loop.

    At this point the ledger was never touched (admission is atomic and
    comes later), so the generation's reserved run ids are safely reusable
    by the next generation. The marker artifact closes the generation for
    `_find_open_slate_generation`; the skip feeds the run-level streak
    breaker, so a systemically dead plan-writer still blocks."""
    manifest = json.loads(
        (gen_dir / "generation.json").read_text(encoding="utf-8"))
    problems = [str(p) for p in problems]
    _generation_abort_path(gen_dir).write_text(
        json.dumps({
            "schema_version": 1,
            "kind": "slate_generation_aborted",
            "gen_no": manifest.get("gen_no"),
            "generation_id": manifest.get("generation_id"),
            "reason": reason,
            "problems": problems[:10],
        }, indent=2) + "\n", encoding="utf-8")
    events.emit("slate_generation_aborted", gen_no=manifest.get("gen_no"),
                generation_id=manifest.get("generation_id"), reason=reason,
                problems=problems[:5])
    _note_seat_skip(run_dir, repo_root, cmd, events, role=role,
                    problems=problems)


def _find_open_slate_generation(run_dir: Path) -> tuple[Path, int, dict | None]:
    """The generation this round resumes into: (gen_dir, gen_no, manifest).

    ``manifest`` is the newest committed generation.json whose seats are not
    all in the ledger with a matching schema-8 binding — that generation must
    be resumed (or, for a binding violation, blocked by the caller).  When no
    manifest exists or the newest is fully admitted, ``manifest`` is None and
    (gen_dir, gen_no) name the next generation, whose provisional artifacts
    may be overwritten.  A tail generation carrying the abort marker is
    closed: the next generation starts fresh (its reserved ids never reached
    the ledger and are reusable).
    """
    semantic = run_dir / ".semantic"
    count = (
        sum(1 for _ in semantic.glob("gen-*/generation.json"))
        if semantic.is_dir() else 0
    )
    if count:
        gen_dir = semantic / f"gen-{count:04d}"
        if _generation_aborted(gen_dir):
            gen_no = count + 1
            return semantic / f"gen-{gen_no:04d}", gen_no, None
        manifest = json.loads(
            (gen_dir / "generation.json").read_text(encoding="utf-8"))
        seats = _slate_seat_records(run_dir, manifest)
        if any(seat is None for seat in seats) \
                or _slate_binding_errors(manifest, seats):
            return gen_dir, count, manifest
    gen_no = count + 1
    return semantic / f"gen-{gen_no:04d}", gen_no, None


def _construct_slate_generation(run_dir, gen_dir, repo_root, cmd,
                                events) -> dict | None:
    """lanes -> per-lane proposals -> pool + shared A1 context.

    Returns the pool document, or None when the admission cap is zero (a
    valid budget-boundary no-op: nothing is admitted this generation).
    """
    lanes_path = gen_dir / "lanes.json"
    _slate_cmd(run_dir, repo_root, cmd, events,
               ["python", "tools/got_select.py", "decide",
                "--ledger", run_dir / "ledger.json",
                "--mode", "lanes", "--output", lanes_path], "lanes decide")
    lanes_doc = json.loads(lanes_path.read_text(encoding="utf-8"))
    if (lanes_doc.get("budget") or {}).get("admission_cap") == 0:
        return None
    proposals_dir = gen_dir / "proposals"
    for lane in lanes_doc.get("lanes") or []:
        _slate_cmd(run_dir, repo_root, cmd, events,
                   ["python", "tools/semantic_search.py", "propose",
                    "--background", run_dir / "background.md",
                    "--ledger", run_dir / "ledger.json",
                    "--op", lane["op"],
                    "--parents", ",".join(lane["parents"]),
                    "--max-points", "24",
                    "--output", proposals_dir / f"{lane['lane_id']}.json"],
                   f"propose {lane['lane_id']}")
    _slate_cmd(run_dir, repo_root, cmd, events,
               ["python", "tools/slate.py", "construct",
                "--lanes", lanes_path,
                "--proposals-dir", proposals_dir,
                "--ledger", run_dir / "ledger.json",
                "--background", run_dir / "background.md",
                "--pool-output", gen_dir / "pool.json",
                "--context-output", gen_dir / "context.json"], "construct")
    pool_doc = json.loads((gen_dir / "pool.json").read_text(encoding="utf-8"))
    events.emit("slate_pool_built", gen_no=pool_doc["gen_no"],
                pool_size=len(pool_doc["pool"]),
                pool_digest=pool_doc["pool_digest"],
                lanes_without_proposals=pool_doc["lanes_without_proposals"])
    return pool_doc


def _slate_judge_needed(pool_doc: dict) -> bool:
    """Whether this generation spawns judge sessions at all.

    Lifecycle mirror of the cardinality gate tools/slate.py enforces at
    aggregate time: judges run only when the pool can fill past one seat and
    the admission cap allows two.  The selection itself is never recomputed
    here; `aggregate` stays the decider.
    """
    pool_n = len(pool_doc.get("pool") or [])
    cap = (pool_doc.get("budget") or {}).get("admission_cap")
    return pool_n > 2 and (cap is None or cap >= 2)


def _validate_judge_stage(run_dir, gen_dir, stage, receipt_path, session_id,
                          model, repo_root, cmd, events) -> dict:
    """Write the stage artifact; an invalid permutation (exit 1) still writes."""
    out_path = gen_dir / "judgments" / f"{stage}.json"
    args = ["python", "tools/slate.py", "validate-judge",
            "--input", gen_dir / "judgments" / f"{stage}.input.json",
            "--output", out_path]
    if receipt_path is not None and Path(receipt_path).exists():
        args += ["--receipt", receipt_path]
    if session_id:
        args += ["--session-id", session_id]
    if model:
        args += ["--model", model]
    cmd(args, repo_root, check=False)
    if not out_path.exists():
        _or_block(run_dir, repo_root, cmd, events,
                  f"validate-judge produced no stage artifact for {stage}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def _invoke_slate_judge(runner, store, task, tag, run_dir, gen_dir, stage, *,
                        round_no, model, repo_root, cmd, events,
                        labels=None) -> None:
    """One judge rollout: fresh session, validate, ONE corrective resume.

    Both failure kinds — no accepted receipt, or a receipt the deterministic
    permutation check rejects — share the single resume allowance, which
    chains onto this rollout's own session with the deterministic errors.
    A second failure persists a failed stage artifact; `aggregate` then takes
    the coverage fallback.  Regular stages always start fresh
    (resume_from=None); so does the boundary stage.
    """
    judgments = gen_dir / "judgments"
    judgments.mkdir(parents=True, exist_ok=True)
    input_path = judgments / f"{stage}.input.json"
    args = ["python", "tools/slate.py", "prepare-judge",
            "--pool", gen_dir / "pool.json",
            "--context", gen_dir / "context.json",
            "--stage", stage,
            "--output", input_path]
    if labels:
        args += ["--labels", ",".join(labels)]
    from tools.task_contract import task_brief_path
    task_brief = task_brief_path(repo_root, run_dir, task)
    if task_brief.is_file():
        args += ["--task-brief", task_brief]
    objective_path = run_dir / "objective_brief.json"
    if objective_path.is_file():
        args += ["--objective-brief", objective_path]
    _slate_cmd(run_dir, repo_root, cmd, events, args, f"prepare-judge {stage}")
    prompt_text = json.loads(input_path.read_text(encoding="utf-8"))[
        "prompt_text"]
    extra = {"stage": stage, "gen_dir": str(gen_dir)}

    inv_id = None
    problems = None
    try:
        _, inv_id = _invoke(runner, store, "slate-judge", task, tag, run_dir,
                            round_no=round_no, extra=extra,
                            inline_payload=prompt_text)
    except InvocationFailed as exc:
        inv_id = exc.invocation_id
        problems = [str(p) for p in exc.problems]
    artifact = _validate_judge_stage(
        run_dir, gen_dir, stage,
        store.receipt_path("slate-judge", inv_id) if inv_id is not None else None,
        store.load_session_id("slate-judge", inv_id)
        if inv_id is not None else None,
        model, repo_root, cmd, events)
    if artifact.get("status") == "valid":
        return
    reason = "; ".join(problems or artifact.get("errors")
                       or ["judge receipt invalid"])
    note = ("Your previous judge receipt was rejected by the deterministic "
            "validator: " + reason + ". Re-submit the receipt with `ranking` "
            "holding every presented candidate label exactly once.")
    try:
        _, inv_id = _invoke(runner, store, "slate-judge", task, tag, run_dir,
                            round_no=round_no,
                            extra={**extra, "correction_note": note},
                            resume_from=inv_id, inline_payload=prompt_text)
    except InvocationFailed as exc:
        inv_id = exc.invocation_id
    _validate_judge_stage(
        run_dir, gen_dir, stage,
        store.receipt_path("slate-judge", inv_id) if inv_id is not None else None,
        store.load_session_id("slate-judge", inv_id)
        if inv_id is not None else None,
        model, repo_root, cmd, events)


def _slate_aggregate(run_dir, gen_dir, repo_root, cmd, events) -> dict:
    """Run `slate.py aggregate`; the small stdout status drives orchestration."""
    out = _slate_cmd(run_dir, repo_root, cmd, events,
                     ["python", "tools/slate.py", "aggregate",
                      "--pool", gen_dir / "pool.json",
                      "--context", gen_dir / "context.json",
                      "--judgments-dir", gen_dir / "judgments",
                      "--output", gen_dir / "judge.json"], "aggregate")
    return json.loads(out.stdout)


def _run_slate_judges(runner, store, task, tag, run_dir, gen_dir, pool_doc,
                      round_no, model, repo_root, cmd, events) -> dict:
    """Run the judged (or degraded) aggregation; return the final status."""
    if not _slate_judge_needed(pool_doc):
        return _slate_aggregate(run_dir, gen_dir, repo_root, cmd, events)
    for stage in _SLATE_REGULAR_STAGES:
        _invoke_slate_judge(runner, store, task, tag, run_dir, gen_dir, stage,
                            round_no=round_no, model=model,
                            repo_root=repo_root, cmd=cmd, events=events)
    status = _slate_aggregate(run_dir, gen_dir, repo_root, cmd, events)
    if status.get("status") == "boundary_required":
        _invoke_slate_judge(runner, store, task, tag, run_dir, gen_dir,
                            "boundary", round_no=round_no, model=model,
                            repo_root=repo_root, cmd=cmd, events=events,
                            labels=status["boundary_labels"])
        status = _slate_aggregate(run_dir, gen_dir, repo_root, cmd, events)
    if status.get("status") == "fallback":
        events.emit("slate_judge_fallback", gen_no=pool_doc["gen_no"],
                    reason=status.get("reason"))
    elif status.get("status") == "selected":
        events.emit("slate_judge_completed", gen_no=pool_doc["gen_no"],
                    path=status.get("path"), slate=status.get("slate"))
    return status


def _reserved_run_ids(run_dir: Path, count: int) -> list[str]:
    """Consecutive run ids, mirroring tools/ledger_core.next_run_id's rule."""
    records = _ledger_records(run_dir)
    numeric = [int(r["run_id"]) for r in records
               if str(r.get("run_id", "")).isdigit()]
    width = max([3] + [len(str(r["run_id"])) for r in records
                       if str(r.get("run_id", "")).isdigit()])
    start = max(numeric) + 1 if numeric else 0
    return [f"{start + index:0{width}d}" for index in range(count)]


def _generation_donor_args(run_dir, repo_root, cmd, events) -> list[str]:
    """Freeze this generation's donor snapshot once, before the manifest (§3.1).

    The snapshot is content-addressed, so a retried commit binds the same
    artifact; ``no_donor`` is the normal pre-anchor state.  A construction
    failure (unreadable ledger, an unreproducible applied incumbent) blocks
    the run — a corrupt donor state must never silently degrade the
    generation into an ordinary warm pool (§8).
    """
    try:
        result = build_donor_snapshot(run_dir)
    except (ValueError, OSError) as exc:
        _or_block(run_dir, repo_root, cmd, events,
                  f"donor snapshot construction failed: {exc}")
    if result["status"] == "no_donor":
        return ["--no-donor"]
    return ["--donor-snapshot", result["path"]]


def _commit_slate_manifest(run_dir, gen_dir, slate_size, repo_root, cmd,
                           events) -> dict:
    """Reserve run ids and atomically write the immutable generation.json."""
    reserved = _reserved_run_ids(run_dir, slate_size)
    donor_args = []
    if _scheduler_policy(run_dir) == TRANSFER_SCHEDULER_POLICY:
        donor_args = _generation_donor_args(run_dir, repo_root, cmd, events)
    _slate_cmd(run_dir, repo_root, cmd, events,
               ["python", "tools/slate.py", "build-manifest",
                "--lanes", gen_dir / "lanes.json",
                "--pool", gen_dir / "pool.json",
                "--context", gen_dir / "context.json",
                "--judge", gen_dir / "judge.json",
                "--reserved-run-ids", ",".join(reserved),
                *donor_args,
                "--output", gen_dir / "generation.json"], "build-manifest")
    manifest = json.loads(
        (gen_dir / "generation.json").read_text(encoding="utf-8"))
    events.emit("slate_manifested", gen_no=manifest["gen_no"],
                generation_id=manifest["generation_id"],
                reserved_run_ids=reserved,
                aggregation=manifest["aggregation"].get("path"))
    return manifest


def _slate_plan_problems(plan, slot: dict, route_memory: dict | None) -> list[str]:
    """Validate the writer-owned fields before leaving its repair loop."""
    if not isinstance(plan, dict):
        return ["plan is not a JSON object"]
    problems = []
    if plan.get("slot") != slot["slot"]:
        problems.append(
            f"plan slot {plan.get('slot')!r} != seat slot {slot['slot']}")
    for field_name in ("idea", "change", "candidate_name"):
        if not isinstance(plan.get(field_name), str) \
                or not plan[field_name].strip():
            problems.append(f"plan needs a non-empty {field_name}")
    if route_memory is not None:
        provenance = plan.get("route_provenance")
        if is_not_applicable(provenance):
            problems.append(
                "a judged-slate candidate is never the task-provided baseline; "
                "route provenance cannot be not_applicable")
        else:
            problems.extend(validate_route_provenance(provenance, memory=route_memory))
    return problems


def _slate_plan_payload(slot: dict, pool_doc: dict, context_doc: dict,
                        route_memory_path, objective_text: str | None = None) -> str:
    """The bounded plan input, bound to the manifest's point and carrier."""
    entry = next(entry for entry in pool_doc["pool"]
                 if entry["label"] == slot["label"])
    parts = []
    if objective_text:
        parts.append(objective_text)
    parts.extend([
        "Frozen slot assignment (binding; the slate decision is final):",
        json.dumps({
            "slot": slot["slot"],
            "run_id": slot["run_id"],
            "label": slot["label"],
            "candidate_id": slot["candidate_id"],
            "point_id": slot["point_id"],
            "point": slot["point"],
            "carrier": slot["carrier"],
        }, indent=2),
        "Candidate summary exactly as the judges saw it:",
        json.dumps(entry["summary"], indent=2),
        str(context_doc.get("rendered_text") or ""),
    ])
    if route_memory_path is not None:
        parts.append(
            f"Route memory for this seat (read before planning routes): "
            f"{route_memory_path}"
        )
    return "\n\n".join(parts) + "\n"


def _slate_plan_retry_note(problems: list[str]) -> str:
    """Decorrelated retry appendix: the prior attempt's actual failure plus,
    for the repetition failure mode, an explicitly worded admonition."""
    items = "\n".join(f"- {p}" for p in problems)
    if any("repetition breaker" in p for p in problems):
        admonition = (
            "A previous planning attempt for this seat was terminated for "
            "repeating an identical file Read. Use the supplied context, "
            "read any required referenced file once, then complete the plan "
            "and call submit_receipt.")
    else:
        admonition = (
            "A previous planning attempt for this seat failed as listed "
            "below. Address its failure, then complete the plan and call "
            "submit_receipt.")
    return ("--- Prior attempt note ---\n"
            f"{admonition}\nFailure problems from that attempt:\n{items}\n")


# slate-plan-writer attempt budget: 1 initial + 2 decorrelated inline
# retries per (generation, slot). Persisted before each invocation so a
# crash mid-attempt still consumes budget, and never reset by restarts or
# external resumes (2026-09-16 incident: 75 blocked resumes re-drew the
# same failing payload for ~$4 before the budget ran out elsewhere).
_SLATE_WRITER_ATTEMPTS = 3


# candidate-writer gets one initial invocation and one decorrelated retry.
# The counter lives under the candidate so a process restart or external
# resume cannot silently mint another writer session.
_CANDIDATE_WRITER_ATTEMPTS = 2


def _candidate_writer_attempts_path(candidate_dir: Path) -> Path:
    return candidate_dir / "writer.attempts.json"


def _candidate_writer_failure_path(candidate_dir: Path) -> Path:
    return candidate_dir / "writer.failure.json"


def _candidate_writer_attempts(candidate_dir: Path) -> dict:
    path = _candidate_writer_attempts_path(candidate_dir)
    if not path.exists():
        return {"attempts": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"WARNING: corrupted writer attempts file "
              f"{path}: {exc}", file=sys.stderr)
        return {"attempts": 0}
    return data if isinstance(data, dict) else {"attempts": 0}


def _candidate_writer_failure(candidate_dir: Path) -> list[str]:
    path = _candidate_writer_failure_path(candidate_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"WARNING: corrupted writer failure file "
              f"{path}: {exc}", file=sys.stderr)
        return []
    return [str(problem) for problem in data] if isinstance(data, list) else []


def _register_candidate_writer_attempt(candidate_dir: Path) -> int:
    """Persist the next writer attempt before starting its session."""
    path = _candidate_writer_attempts_path(candidate_dir)
    data = _candidate_writer_attempts(candidate_dir)
    attempts = int(data.get("attempts", 0)) + 1
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"attempts": attempts}) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return attempts


def _record_candidate_writer_success(candidate_dir: Path, invocation_id: int) -> None:
    path = _candidate_writer_attempts_path(candidate_dir)
    data = _candidate_writer_attempts(candidate_dir)
    data["completed_invocation_id"] = invocation_id
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _record_candidate_writer_failure(candidate_dir: Path, problems: list[str]) -> None:
    path = _candidate_writer_failure_path(candidate_dir)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps([str(problem) for problem in problems]) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _candidate_writer_retry_note(problems: list[str]) -> str:
    items = "\n".join(f"- {problem}" for problem in problems)
    return (
        "\n\n--- Prior candidate-writer attempt note ---\n"
        "The previous candidate-writer attempt failed for these reasons:\n"
        f"{items}\n"
        "Complete the checks required by the current write mode, then carry "
        "out that mode and submit the receipt; do not repeat reads of the "
        "same context already obtained.\n"
    )


def _slot_attempts_path(plans_dir, slot_no) -> Path:
    return plans_dir / f"slot-{slot_no}.attempts.json"


def _slate_writer_attempts_used(plans_dir, slot_no) -> int:
    path = _slot_attempts_path(plans_dir, slot_no)
    if not path.exists():
        return 0
    try:
        return int(json.loads(
            path.read_text(encoding="utf-8")).get("attempts", 0))
    except (OSError, json.JSONDecodeError, ValueError):
        return 0


def _register_slate_writer_attempt(plans_dir, slot_no) -> int:
    """Increment-and-persist the per-slot attempt counter; returns the
    1-based attempt number just registered."""
    path = _slot_attempts_path(plans_dir, slot_no)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    attempts = int(data.get("attempts", 0)) + 1
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"attempts": attempts}) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return attempts


def _ensure_slate_plans(runner, store, task, tag, run_dir, gen_dir, manifest,
                        round_no, repo_root, cmd, events) -> bool:
    """One plan per seat; only missing or invalid plans are (re)written.

    Returns False when the generation was aborted: the attempt budget for
    some seat's plan ran out — invocation failures and schema-invalid
    receipts both consume it — the abort marker closes the generation, and
    the caller proceeds with a fresh generation instead of re-blocking on
    every resume (the reserved ids never reached the ledger)."""
    plans_dir = gen_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    route_arm = _slate_route_arm(run_dir)
    pool_doc = json.loads((gen_dir / "pool.json").read_text(encoding="utf-8"))
    context_doc = json.loads(
        (gen_dir / "context.json").read_text(encoding="utf-8"))
    objective_text = _objective_block(run_dir)
    for slot in manifest["slate"]:
        plan_path = plans_dir / f"slot-{slot['slot']}.json"
        route_memory_path = None
        route_memory = None
        if route_arm:
            point_path = plans_dir / f"slot-{slot['slot']}.point.json"
            point_path.write_text(
                json.dumps(slot["point"], indent=2) + "\n", encoding="utf-8")
            route_memory_path = plans_dir / f"slot-{slot['slot']}.route-memory.json"
            _slate_cmd(run_dir, repo_root, cmd, events,
                       ["python", "tools/semantic_routes.py", "memory",
                        "--ledger", run_dir / "ledger.json",
                        "--point", point_path,
                        "--op", slot["carrier"]["op"],
                        "--output", route_memory_path], "route memory")
            route_memory = json.loads(route_memory_path.read_text(encoding="utf-8"))
        if plan_path.exists():
            try:
                existing = json.loads(plan_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing is not None and not _slate_plan_problems(
                    existing, slot, route_memory):
                continue
        if _slate_writer_attempts_used(plans_dir, slot["slot"]) \
                >= _SLATE_WRITER_ATTEMPTS:
            reason = (f"slate-plan-writer attempt budget exhausted for slot "
                      f"{slot['slot']} ({_SLATE_WRITER_ATTEMPTS} attempts, "
                      f"no valid plan)")
            _abort_slate_generation(run_dir, gen_dir, repo_root, cmd, events,
                                    reason=reason,
                                    role="slate-plan-writer",
                                    problems=[reason])
            return False
        extra = {"slot": slot["slot"], "candidate_id": slot["candidate_id"],
                 "gen_dir": str(gen_dir)}
        if route_arm:
            extra["n_route_sketches"] = route_arm
            extra["route_memory"] = str(route_memory_path)
        payload = _slate_plan_payload(
            slot, pool_doc, context_doc, route_memory_path,
            objective_text=objective_text)
        while True:
            attempt = _register_slate_writer_attempt(plans_dir, slot["slot"])
            try:
                receipt, _ = _invoke(
                    runner, store, "slate-plan-writer", task, tag, run_dir,
                    run_id=slot["run_id"], round_no=round_no, extra=extra,
                    inline_payload=payload)
            except InvocationFailed as exc:
                problems = [str(p) for p in exc.problems]
            else:
                problems = _slate_plan_problems(receipt, slot, route_memory)
                if not problems:
                    break  # a valid plan for this seat
            if attempt >= _SLATE_WRITER_ATTEMPTS:
                reason = (f"slate plan for slot {slot['slot']} could not be "
                          f"produced within {_SLATE_WRITER_ATTEMPTS} "
                          f"attempts: {problems}")
                _abort_slate_generation(run_dir, gen_dir, repo_root, cmd,
                                        events, reason=reason,
                                        role="slate-plan-writer",
                                        problems=problems)
                return False
            payload = payload + _slate_plan_retry_note(problems)
        plan = {key: receipt[key]
                for key in ("slot", "idea", "change", "candidate_name",
                            "route_provenance")
                if key in receipt}
        tmp = plan_path.with_name(plan_path.name + ".tmp")
        tmp.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, plan_path)
        _note_seat_progress(run_dir)
    return True


def _admit_slate(run_dir, gen_dir, repo_root, cmd, events) -> None:
    """The single atomic two-seat admission (tools/ledger.py owns the write)."""
    _slate_cmd(run_dir, repo_root, cmd, events,
               ["python", "tools/ledger.py", "admit-slate",
                "--ledger", run_dir / "ledger.json",
                "--background", run_dir / "background.md",
                "--manifest", gen_dir / "generation.json",
                "--plans-dir", gen_dir / "plans"], "admit-slate")


def _slate_seat_records(run_dir: Path, manifest: dict) -> list:
    """Each seat's ledger record, or None; binding errors on present seats."""
    by_id = {str(r.get("run_id")): r for r in _ledger_records(run_dir)}
    return [by_id.get(slot["run_id"]) for slot in manifest["slate"]]


def _slate_binding_errors(manifest: dict, seats: list) -> list[str]:
    """A present seat must carry this manifest's schema-8 judge binding."""
    errors = []
    for slot, record in zip(manifest["slate"], seats):
        if record is None:
            continue
        receipt = record.get("policy_receipt")
        judge = receipt.get("judge") if isinstance(receipt, dict) else None
        if not isinstance(judge, dict) \
                or receipt.get("schema_version") != 8 \
                or receipt.get("generation_id") != manifest["generation_id"] \
                or judge.get("candidate_id") != slot["candidate_id"]:
            errors.append(f"seat {slot['run_id']} does not carry the "
                          "manifest's schema-8 judge binding")
    return errors


def _evaluate_admitted_slate(runner, store, task, tag, run_dir, manifest,
                             repo_root, cmd, events, job_runner) -> None:
    """Implement the seats on the session channel; a slot-0 crash never
    refills it."""
    _implement_seats(runner, store, task, tag, run_dir,
                     [slot["run_id"] for slot in manifest["slate"]],
                     repo_root, cmd, events, job_runner)


# =============================================================================
# Session channel: bounded overlap of seat implementations
# =============================================================================


def _session_concurrency(run_dir: Path) -> int:
    """Seats implemented at once (framework_cfg ``pipeline.session_concurrency``,
    absent = 1 = the serial loop)."""
    config_path = run_dir / "framework_cfg.json"
    try:
        section = json.loads(config_path.read_text(encoding="utf-8")).get(
            "pipeline")
    except (OSError, json.JSONDecodeError):
        return 1
    value = section.get("session_concurrency") if isinstance(section, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1
    return value


def _implement_seats(runner, store, task, tag, run_dir, run_ids, repo_root,
                     cmd, events, job_runner) -> None:
    """Materialize + implement each admitted seat, up to
    ``session_concurrency`` at a time.

    Serial (concurrency 1) runs exactly the historical loop. Otherwise each
    seat is one task on a small thread pool: the stop-condition check moves
    into the seat so a seat that becomes runnable after the budget landed is
    skipped, and a RunBlocked raised on one seat lets the others finish their
    current boundary before it unwinds. Per-seat waits are emitted for
    throughput diagnostics only — never as a scoring input.
    """
    concurrency = _session_concurrency(run_dir)

    def seat(run_id, ready):
        started = time.monotonic()
        if budget_status(run_dir, repo_root, cmd).get("reached"):
            events.emit("seat_skipped", run_id=run_id, reason="budget_reached")
            return
        events.emit("seat_started", run_id=run_id,
                    session_wait_seconds=round(started - ready, 3))
        _materialize_candidate(task, tag, run_dir, run_id, repo_root, cmd)
        _implement_candidate(runner, store, task, tag, run_dir, run_id,
                             repo_root, cmd, events, job_runner)
        events.emit("seat_finished", run_id=run_id,
                    seconds=round(time.monotonic() - started, 3))

    if concurrency <= 1 or len(run_ids) <= 1:
        for run_id in run_ids:
            if budget_status(run_dir, repo_root, cmd).get("reached"):
                break
            _materialize_candidate(task, tag, run_dir, run_id, repo_root, cmd)
            _implement_candidate(runner, store, task, tag, run_dir, run_id,
                                 repo_root, cmd, events, job_runner)
        return

    batch_started = time.monotonic()
    ready = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency,
                            thread_name_prefix="seat") as pool:
        futures = [pool.submit(seat, run_id, ready) for run_id in run_ids]
        blocked = None
        error = None
        for future in futures:
            try:
                future.result()
            except RunBlocked as exc:
                blocked = blocked or exc
            except BaseException as exc:  # noqa: BLE001 - surfaced after join
                error = error or exc
    events.emit("seats_completed", run_ids=list(run_ids),
                session_concurrency=concurrency,
                wall_seconds=round(time.monotonic() - batch_started, 3))
    if blocked is not None:
        if error is not None:  # planned stops win, but never silently
            events.emit("seat_error_masked", error=repr(error))
        raise blocked
    if error is not None:
        raise error


def _evaluate_judged_generation(runner, store, task, tag, run_dir, round_no,
                                repo_root, cmd, events, model,
                                job_runner) -> list[dict]:
    """One judged-slate generation; the manifest is the only resume anchor."""
    gen_dir, gen_no, manifest = _find_open_slate_generation(run_dir)
    if manifest is None:
        # No committed manifest means this generation is provisional.  Any
        # judge artifacts left by an interrupted attempt belong to the old
        # pool/context and must not be mixed into a rebuilt generation.
        judgments_dir = gen_dir / "judgments"
        if judgments_dir.is_dir():
            for artifact in judgments_dir.iterdir():
                if artifact.is_file() and artifact.suffix == ".json":
                    artifact.unlink()
        pool_doc = _construct_slate_generation(run_dir, gen_dir, repo_root,
                                               cmd, events)
        if pool_doc is None:
            return []  # admission cap 0: budget-boundary no-op
        status = _run_slate_judges(runner, store, task, tag, run_dir, gen_dir,
                                   pool_doc, round_no, model, repo_root, cmd,
                                   events)
        if not (status.get("slate") or []):
            return []  # no lane produced a proposal; gen_no stays unconsumed
        manifest = _commit_slate_manifest(run_dir, gen_dir,
                                          len(status["slate"]),
                                          repo_root, cmd, events)
    seats = _slate_seat_records(run_dir, manifest)
    if any(seat is not None for seat in seats):
        # The atomic batch admits every seat or none, and a manifest whose
        # seats all landed is closed by _find_open_slate_generation.  Any
        # ledger presence here violates that contract: fail closed, never
        # guess the missing seat.
        problems = _slate_binding_errors(manifest, seats)
        detail = "; ".join(problems) if problems else "partial admission"
        _or_block(run_dir, repo_root, cmd, events,
                  f"judged-slate generation {manifest['gen_no']} violates the "
                  f"atomic-admission contract: {detail}")
    if not _ensure_slate_plans(runner, store, task, tag, run_dir, gen_dir,
                               manifest, round_no, repo_root, cmd, events):
        return []  # generation aborted: seats were never admitted
    _admit_slate(run_dir, gen_dir, repo_root, cmd, events)
    events.emit("slate_admitted", gen_no=manifest["gen_no"],
                generation_id=manifest["generation_id"],
                run_ids=[slot["run_id"] for slot in manifest["slate"]])
    _evaluate_admitted_slate(runner, store, task, tag, run_dir, manifest,
                             repo_root, cmd, events, job_runner)
    return [{"run_id": slot["run_id"], "op": slot["carrier"]["op"]}
            for slot in manifest["slate"]]


# =============================================================================
# Candidate materialization and step 0+1 evaluation
# =============================================================================


def _failure_evidence(candidate_dir: Path,
                      problems: list[str] | None = None) -> str | None:
    """The diagnosis input: the durable report path when it exists, plus the
    failed invocation's own problems, so the diagnoser sees both the artifact
    and why the session could not finish."""
    report = candidate_dir / "tune_report.json"
    if not report.exists():
        return None
    evidence = str(report)
    if problems:
        evidence += ("\n\nextractor invocation problems:\n"
                     + "\n".join(f"- {p}" for p in problems))
    return evidence


def _resolve_unevaluated(run_dir, run_id, repo_root, cmd) -> bool:
    """Call-and-catch: the helper itself enforces all four preconditions."""
    try:
        cmd(["python", "tools/ledger.py", "resolve-unevaluated",
             "--ledger", run_dir / "ledger.json", "--run-id", run_id], repo_root)
        return True
    except subprocess.CalledProcessError:
        return False


def _record_crash(run_dir, run_id, repo_root, cmd) -> None:
    cmd(["python", "tools/ledger.py", "record-run",
         "--ledger", run_dir / "ledger.json", "--run-id", run_id,
         "--status", "crash"], repo_root)


def _finite_warm_score(candidate_dir: Path) -> float | None:
    try:
        report = json.loads((candidate_dir / "tune_report.json")
                            .read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = ((report or {}).get("phase_a") or {}).get("best_warm_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _settle_at_deadline(run_dir, run_id, repo_root, cmd, events) -> None:
    """Deterministic settlement of a pending candidate — ledger subcommands
    only, no session.

    Three-way, by on-disk evidence: no objective attempt → unevaluated;
    a finite warm score in the tune report → set-tuning + record-run (the
    extractor's own step 3c replayed; auto status keep/discard); attempts
    but no finite score → unevaluated when every attempt was cut off by the
    time budget, else crash. Originally the run-cutoff settlement, now also
    the deterministic close for a repair-exhausted extractor seat with
    evidence: only durable facts decide, never a session (fidelity before
    attribution).
    """
    if record_status(run_dir, run_id) in ("keep", "discard", "crash",
                                         "unevaluated"):
        return
    candidate_dir = run_dir / "candidates" / run_id
    view = budget_status(run_dir, repo_root, cmd)
    row = next((r for r in view.get("per_candidate", [])
                if r.get("run_id") == run_id), {})
    attempts = int(row.get("evals") or 0)
    warm = _finite_warm_score(candidate_dir)
    ledger = run_dir / "ledger.json"
    if warm is None and (attempts == 0 or int(
            row.get("time_cutoff_evals") or 0) == attempts):
        if _resolve_unevaluated(run_dir, run_id, repo_root, cmd):
            events.emit("candidate_settled_at_deadline", run_id=run_id,
                        outcome="unevaluated", attempts=attempts)
            return
        # Refused (e.g. the record already carries a score): a record left
        # pending would block set-phase completed at the cutoff, so fall
        # through to the crash settlement like a refused set-tuning.
    if warm is not None:
        try:
            cmd(["python", "tools/ledger.py", "set-tuning", "--ledger", ledger,
                 "--run-id", run_id,
                 "--from-report", candidate_dir / "tune_report.json"], repo_root)
            cmd(["python", "tools/ledger.py", "record-run", "--ledger", ledger,
                 "--run-id", run_id, "--final-best-score", str(warm)], repo_root)
        except subprocess.CalledProcessError as exc:
            events.emit("candidate_settlement_failed", run_id=run_id,
                        detail=(exc.stderr or str(exc))[-2000:])
        else:
            events.emit("candidate_settled_at_deadline", run_id=run_id,
                        outcome=record_status(run_dir, run_id),
                        best_warm_score=warm, attempts=attempts)
            return
    _record_crash(run_dir, run_id, repo_root, cmd)
    events.emit("candidate_settled_at_deadline", run_id=run_id,
                outcome="crash", attempts=attempts)


def _materialize_candidate(task, tag, run_dir, run_id, repo_root, cmd) -> None:
    """new_candidate.py refuses a non-empty candidate dir; on resume the dir
    may already be materialized, so only run the helper when it is not."""
    candidate_dir = run_dir / "candidates" / run_id
    if candidate_dir.exists() and any(candidate_dir.iterdir()):
        return
    cmd(["python", "tools/new_candidate.py", task, tag, run_id,
         "--skip-entrypoint"], repo_root)


def _screening_contract(run_dir: Path, run_id: str) -> tuple[int, int]:
    """Return the resume-stable (actual, target) Phase-A screening width."""
    config_path = run_dir / "framework_cfg.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    tuner = config.get("tuner") if isinstance(config, dict) else {}
    target = max(
        MIN_GENERATION_K_EVAL,
        int((tuner or {}).get("K_eval", DEFAULT_K_EVAL)),
    )

    report_path = run_dir / "candidates" / run_id / "tune_report.json"
    if report_path.is_file():
        selection = json.loads(report_path.read_text(encoding="utf-8")).get(
            "phase_a", {}
        ).get("warm_config_selection", {})
        persisted = selection.get("k_eval")
        if isinstance(persisted, int) and not isinstance(persisted, bool) \
                and persisted > 0:
            return persisted, target

    remaining = objective_budget_status(run_dir).get("remaining")
    actual = ResourceContract(k_eval=target).screening_reservation(
        remaining if isinstance(remaining, int) else target,
        allow_terminal_degrade=isinstance(remaining, int),
    )
    return actual, target


def _manifest_donor_binding(run_dir, run_id, repo_root, cmd, events):
    """A judged-slate seat's generation-manifest donor binding, if any."""
    record = next(
        (r for r in _ledger_records(run_dir)
         if str(r.get("run_id")) == str(run_id)),
        None,
    )
    receipt = (record or {}).get("policy_receipt")
    judge = receipt.get("judge") if isinstance(receipt, dict) else None
    manifest_rel = judge.get("manifest_path") if isinstance(judge, dict) else None
    if manifest_rel is None:
        return None  # not a judged-slate seat
    manifest_path = run_dir / manifest_rel
    if not manifest_path.is_file():
        _or_block(run_dir, repo_root, cmd, events,
                  f"seat {run_id}'s generation manifest {manifest_rel} is "
                  "missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = manifest.get("donor_snapshot")
    if not isinstance(binding, dict):
        _or_block(run_dir, repo_root, cmd, events,
                  f"seat {run_id}'s generation manifest lacks the donor "
                  "binding the transfer scheduler policy requires")
    return binding


def _candidate_donor_binding(run_dir, run_id, repo_root, cmd, events) -> dict:
    """The coverage arm's per-candidate donor binding (design §3.1 middle arm).

    There is no generation manifest on this arm, so the candidate-local
    receipt is the binding carrier: an existing receipt re-binds its frozen
    snapshot even when the current frontier has moved on (§8); otherwise the
    frontier is frozen once, now, for this candidate.
    """
    receipt_path = run_dir / "candidates" / run_id / _DONOR_RECEIPT_FILENAME
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _or_block(run_dir, repo_root, cmd, events,
                      f"candidate {run_id} donor receipt is unreadable: {exc}")
        donor = receipt.get("donor") if isinstance(receipt, dict) else None
        snapshot_id = donor.get("snapshot_id") if isinstance(donor, dict) else None
        if not isinstance(snapshot_id, str) or not snapshot_id:
            _or_block(run_dir, repo_root, cmd, events,
                      f"candidate {run_id} donor receipt lacks "
                      "donor.snapshot_id")
        return {
            "status": "bound",
            "snapshot_id": snapshot_id,
            "path": (donors_dir(run_dir) / f"{snapshot_id}.json")
            .relative_to(run_dir)
            .as_posix(),
        }
    try:
        result = build_donor_snapshot(run_dir)
    except (ValueError, OSError) as exc:
        _or_block(run_dir, repo_root, cmd, events,
                  f"donor snapshot construction failed: {exc}")
    if result["status"] == "no_donor":
        return {"status": "no_donor", "snapshot_id": None, "path": None}
    return {
        "status": "bound",
        "snapshot_id": result["snapshot_id"],
        "path": result["path"],
    }


def _resolve_donor_extra(run_dir, run_id, repo_root, cmd, events) -> dict:
    """The extractor invocation's donor binding for one candidate.

    Empty under every other policy pair.  A judged-slate seat reads its
    generation manifest's immutable binding, so a donor frontier that moves
    between seats cannot drift the generation (§3.1); the coverage arm binds
    per candidate.  A missing bound artifact blocks the run rather than
    silently degrading to an ordinary warm pool (§8).
    """
    if _scheduler_policy(run_dir) != TRANSFER_SCHEDULER_POLICY:
        return {}
    binding = _manifest_donor_binding(run_dir, run_id, repo_root, cmd, events)
    if binding is None:
        binding = _candidate_donor_binding(
            run_dir, run_id, repo_root, cmd, events)
    if binding.get("status") == "no_donor":
        return {"donor_binding": "no_donor"}
    path = binding.get("path")
    snapshot_path = run_dir / path if isinstance(path, str) else None
    if snapshot_path is None or not snapshot_path.is_file():
        _or_block(run_dir, repo_root, cmd, events,
                  f"candidate {run_id} is bound to donor snapshot {path!r} "
                  "but the artifact is missing")
    return {"donor_binding": "bound", "donor_snapshot": str(snapshot_path)}


def _extractor_extra(run_dir: Path, run_id: str, candidate_dir: Path,
                     **extra) -> dict:
    actual, target = _screening_contract(run_dir, run_id)
    return {
        "candidate_dir": str(candidate_dir),
        "screening_k_eval": actual,
        "screening_target_k_eval": target,
        **extra,
    }


def _settle_unevaluated(runner, store, task, tag, run_dir, run_id,
                        candidate_dir, donor_extra, extractor_inv, repo_root,
                        cmd, events, job_runner) -> None:
    """The extractor reported a zero-attempt candidate at a reached stop
    condition; the driver owns that lifecycle resolution.

    ``resolve-unevaluated`` proves the stop condition and the zero attempts
    itself. When it refuses (the budget is not actually exhausted), the
    session is resumed once with that fact so it re-requests its job; a
    second unsupported claim blocks the run rather than leaving a pending
    seat behind.
    """
    if record_status(run_dir, run_id) in ("keep", "discard", "crash",
                                         "unevaluated"):
        return
    if _resolve_unevaluated(run_dir, run_id, repo_root, cmd):
        events.emit("candidate_unevaluated", run_id=run_id)
        return
    note = ("Your receipt claimed status=unevaluated, but the run's stop "
            "condition has not been reached and resolve-unevaluated refused. "
            "Continue the extractor procedure: request the warmstart "
            "driver_job (or record the candidate's real outcome) and submit "
            "a terminal receipt.")
    try:
        receipt, _ = _invoke_with_driver_jobs(
            runner, store, "tunable-contract-extractor", task, tag, run_dir,
            run_id=run_id,
            extra=_extractor_extra(run_dir, run_id, candidate_dir,
                                   reconcile_note=note, **donor_extra),
            resume_from=extractor_inv, repo_root=repo_root,
            job_runner=job_runner)
    except InvocationFailed as exc:
        _or_block(run_dir, repo_root, cmd, events,
                  f"extractor could not settle unevaluated claim for "
                  f"{run_id}: {exc.problems}")
    if receipt.get("status") == "unevaluated" and record_status(
            run_dir, run_id) == "pending" and not _resolve_unevaluated(
            run_dir, run_id, repo_root, cmd):
        _or_block(run_dir, repo_root, cmd, events,
                  f"extractor repeated an unsupported unevaluated claim for "
                  f"{run_id}")


def _implement_candidate(runner, store, task, tag, run_dir, run_id, repo_root,
                          cmd, events, job_runner=execute_driver_job,
                          task_toml=None) -> None:
    """candidate-writer + extractor with evidence-branched escalation."""
    candidate_dir = run_dir / "candidates" / run_id
    objective = _objective_line(task_toml or common.load_task_toml(task, repo_root))
    writer_extra = {"candidate_dir": str(candidate_dir), "objective": objective}
    writer_attempts = _candidate_writer_attempts(candidate_dir)
    writer_inv = writer_attempts.get("completed_invocation_id")
    if writer_inv is None and int(writer_attempts.get("attempts", 0)) \
            >= _CANDIDATE_WRITER_ATTEMPTS:
        # No successful writer stage survived the previous process. Close
        # from current evidence (aborted only for a zero-product seat);
        # the run-level streak breaker catches repeated failures.
        _skip_candidate(
            run_dir, repo_root, cmd, events, run_id,
            role="candidate-writer",
            problems=_candidate_writer_failure(candidate_dir)
            or ["candidate-writer attempt budget exhausted "
                f"({_CANDIDATE_WRITER_ATTEMPTS} attempts)"],
        )
        return
    # Resolved once per candidate implementation so every extractor retry of
    # this candidate sees the identical donor binding.
    donor_extra = _resolve_donor_extra(run_dir, run_id, repo_root, cmd, events)
    while writer_inv is None:
        attempt = int(_candidate_writer_attempts(candidate_dir).get("attempts", 0)) + 1
        retry_problems = _candidate_writer_failure(candidate_dir)
        extra = writer_extra
        if attempt > 1:
            extra = dict(writer_extra)
            extra["retry_note"] = _candidate_writer_retry_note(
                [str(problem) for problem in retry_problems]
                or ["the previous attempt did not produce an accepted receipt"]
            )
        try:
            _, writer_inv = _invoke(
                runner, store, "candidate-writer", task, tag, run_dir,
                run_id=run_id, extra=extra, writer_attempt_dir=candidate_dir)
        except InvocationFailed as exc:
            _record_candidate_writer_failure(candidate_dir, exc.problems)
            if attempt >= _CANDIDATE_WRITER_ATTEMPTS:
                _skip_candidate(
                    run_dir, repo_root, cmd, events, run_id,
                    role="candidate-writer", problems=exc.problems)
                return
        else:
            _record_candidate_writer_success(candidate_dir, writer_inv)
    try:
        receipt, extractor_inv = _invoke_with_driver_jobs(
            runner, store, "tunable-contract-extractor", task, tag, run_dir,
            run_id=run_id,
            extra=_extractor_extra(
                run_dir, run_id, candidate_dir, **donor_extra
            ),
            repo_root=repo_root, job_runner=job_runner)
    except InvocationFailed as exc:
        problems = exc.problems
        extractor_inv = exc.invocation_id
    else:
        if receipt.get("status") == "unevaluated":
            _settle_unevaluated(runner, store, task, tag, run_dir, run_id,
                                candidate_dir, donor_extra, extractor_inv,
                                repo_root, cmd, events, job_runner)
        _note_seat_progress(run_dir)
        return

    # Branch on durable evidence (spec Error handling):
    # 0. past the run's cutoff no session (diagnosis, repair) may start:
    #    settle from the on-disk report instead of diagnosing.
    if _time_reached(run_dir):
        _settle_at_deadline(run_dir, run_id, repo_root, cmd, events)
        _note_seat_progress(run_dir)
        return
    # 1. stop condition reached + zero attempts → resolve-unevaluated (call+catch)
    if budget_status(run_dir, repo_root, cmd).get("reached") and \
            _resolve_unevaluated(run_dir, run_id, repo_root, cmd):
        _note_seat_progress(run_dir)
        return
    # 2. actual failure receipt → crash diagnosis
    evidence = _failure_evidence(candidate_dir, problems)
    if evidence:
        try:
            verdict = common.crash_diagnose(
                runner, store, task, tag, run_dir, evidence)["verdict"]
        except InvocationFailed as exc:
            events.emit("crash_diagnosis_failed", run_id=run_id,
                        problems=exc.problems)
            verdict = "abandon"
        if verdict == "abandon":
            _record_crash(run_dir, run_id, repo_root, cmd)
            _note_seat_progress(run_dir)
            return
        try:  # fix verdicts go back to the repair-capable extractor session
            _invoke_with_driver_jobs(
                runner, store, "tunable-contract-extractor", task, tag,
                run_dir, run_id=run_id,
                extra=_extractor_extra(
                    run_dir, run_id, candidate_dir,
                    diagnosis_verdict=verdict, **donor_extra
                ),
                resume_from=extractor_inv, repo_root=repo_root,
                job_runner=job_runner)
            _note_seat_progress(run_dir)
            return
        except InvocationFailed as exc2:
            _skip_candidate(run_dir, repo_root, cmd, events, run_id,
                            role="tunable-contract-extractor",
                            problems=exc2.problems)
            return
    # 3. no evidence yet → one fresh retry, then settle its current evidence
    try:
        _invoke_with_driver_jobs(
            runner, store, "tunable-contract-extractor", task, tag,
            run_dir, run_id=run_id,
            extra=_extractor_extra(
                run_dir, run_id, candidate_dir, **donor_extra
            ), repo_root=repo_root,
            job_runner=job_runner)
    except InvocationFailed as exc3:
        _skip_candidate(run_dir, repo_root, cmd, events, run_id,
                        role="tunable-contract-extractor",
                        problems=exc3.problems)
        return
    _note_seat_progress(run_dir)


# =============================================================================
# Decoupled tuning
# =============================================================================


def _tuner_reconcile(runner, store, task, tag, run_dir, round_no, reason: str,
                     repo_root=REPO_ROOT, job_runner=execute_driver_job,
                     selection=None):
    receipt, _ = _invoke_with_driver_jobs(
        runner, store, "tuner-orchestrator", task, tag, run_dir,
        round_no=round_no,
        run_id=(str(selection["run_id"]) if selection else None),
        extra=_tune_invocation_extra(selection,
                                     {"reconcile_note": reason + _RECONCILE_GUIDANCE}),
        repo_root=repo_root, job_runner=job_runner)
    return receipt


def _phase_c_recover_close(run_dir: Path, run_id: str, repo_root: Path, cmd,
                           events, task: str) -> dict | None:
    """Close/finalize a terminal Phase-C report using deterministic actions.

    A tuner receipt is a session handoff and may truthfully say ``tuned=false``
    even though the objective job exhausted its bout.  The report/action pair
    is authoritative for deciding whether that bout can be closed.  Every
    no-op return first emits ``tuning_driver_finalize_deferred`` with the
    refusing step's own detail, so a persistent refusal is diagnosable.
    """
    candidate_dir = run_dir / "candidates" / str(run_id)

    def defer(what: str, proc=None) -> None:
        detail = ""
        if proc is not None:
            detail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        events.emit("tuning_driver_finalize_deferred", run_id=run_id,
                    reason=f"{what}: {detail}" if detail else what)

    action_cmd = ["python", "tools/tuners/tune_tools.py", "phase-c-action",
                  "--candidate-path", candidate_dir / "train.py",
                  "--tune-report-json", candidate_dir / "tune_report.json"]
    action = cmd(action_cmd, repo_root, check=False)
    if getattr(action, "returncode", 1) != 0:
        defer("phase-c-action refused", action)
        return None
    decision = json.loads(action.stdout)
    if decision.get("action") == "close_exhausted_stage":
        closed = cmd(["python", "tools/tuners/tune_tools.py",
                      "close-exhausted-stage",
                      "--candidate-path", candidate_dir / "train.py",
                      "--tune-report-json", candidate_dir / "tune_report.json"],
                     repo_root, check=False)
        if getattr(closed, "returncode", 1) != 0:
            defer("close-exhausted-stage refused", closed)
            return None
        action = cmd(action_cmd, repo_root, check=False)
        if getattr(action, "returncode", 1) != 0:
            defer("phase-c-action re-check refused", action)
            return None
        decision = json.loads(action.stdout)
    if decision.get("action") != "finalize":
        defer(f"phase-c-action decided {decision.get('action')!r}")
        return None
    result = cmd(["python", "tools/finalize_tuning.py",
                  "--candidate-path", candidate_dir / "train.py",
                  "--tune-report-json", candidate_dir / "tune_report.json",
                  "--ledger", run_dir / "ledger.json",
                  "--run-id", str(run_id), "--task", task],
                 repo_root, check=False)
    if getattr(result, "returncode", 1) != 0:
        defer("finalize_tuning refused", result)
        return None
    return json.loads(result.stdout)


def _close_failed_tune(run_dir, repo_root, cmd, events, selection, *,
                       phase_c_before: int = 0) -> None:
    """A pinned tune bout is ending in a blocked run: close its decision as
    an infrastructure failure so it cannot be silently re-executed later.

    ``phase_c_before`` is the selected candidate's admitted Phase-C attempt
    count when the bout started: the outcome must charge the decision with
    what the failed attempts actually consumed, not an unconditional zero.
    """
    if selection is None:
        return
    run_id = str(selection["run_id"])
    consumed = max(
        0, phase_c_attempts(run_dir, run_id) - int(phase_c_before))
    try:
        rounds._record(run_dir, repo_root, cmd, selection["decision_id"],
                       run_id, action="TUNE", consumed=consumed,
                       status="infra_failure")
    except subprocess.CalledProcessError as exc:
        events.emit("tune_outcome_record_failed",
                    decision_id=selection.get("decision_id"),
                    detail=(exc.stderr or str(exc))[-1500:])


def _tune_invocation_extra(selection, base=None):
    """Session context for a pinned tune invocation (round_v1 handoff).

    The handoff is the committed scheduler decision; `run_id` travels as a
    separate invocation pin so every driver-owned Phase-C job is fail-closed
    to the selected candidate before any objective evaluation launches.
    """
    extra = dict(base or {})
    if selection is not None:
        extra["scheduler_selection"] = json.dumps(
            rounds.tune_handoff(selection), sort_keys=True)
    return extra or None


def _tuner_inner_policy(run_dir: Path) -> str | None:
    """Read the inner tuner policy frozen into this run's framework config."""
    try:
        section = json.loads((run_dir / "framework_cfg.json").read_text(
            encoding="utf-8")).get("tuner")
    except (OSError, json.JSONDecodeError):
        return None
    return section.get("inner_policy") if isinstance(section, dict) else None


def _tuning_snapshot(run_dir: Path, run_id: str) -> dict:
    """Persistent-state fingerprint of one candidate's tuning progress.

    Zero progress is judged from this, never from receipt shape, exception
    type, or mtimes: attempts, Phase-C stage states, and the report's
    closing fields all unchanged means the bout produced nothing.
    """
    candidate_dir = run_dir / "candidates" / str(run_id)
    try:
        report = json.loads((candidate_dir / "tune_report.json").read_text(
            encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = {}
    stages = (report.get("phase_c") or {}).get("stages") or []
    return {
        "phase_c_attempts": phase_c_attempts(run_dir, str(run_id)),
        "stages": [
            {"method": stage.get("method"), "status": stage.get("status"),
             "trials_attempted": stage.get("trials_attempted"),
             "trials_completed": stage.get("trials_completed"),
             "trials": len(stage.get("trials") or [])}
            for stage in stages if isinstance(stage, dict)
        ],
        "applied_to_base_params": report.get("applied_to_base_params"),
        "last_finalized_stage_index": report.get("last_finalized_stage_index"),
    }


def _snapshot_counts(snapshot: dict) -> dict:
    """The compact counts that travel on outcome events (never the report)."""
    stages = snapshot.get("stages") or []
    return {
        "phase_c_attempts": snapshot.get("phase_c_attempts"),
        "stages": len(stages),
        "trials": sum(int(stage.get("trials") or 0) for stage in stages),
    }


def _tune(runner, store, task, tag, run_dir, round_no, repo_root, cmd,
          events, job_runner=execute_driver_job, selection=None) -> dict:
    """Run the decoupled tuning step; return the effective tuner receipt.

    ``selection`` is the round scheduler's committed TUNE decision
    (round_v1): the bout is driver-owned and deterministic — no
    tuner-orchestrator session sits between ``phase-c-action`` and the job
    (the legacy inner policy keeps the session, because Phase-R re-warm
    proposals exist only there). ``None`` keeps the orchestrator-side
    selection and recovery used by the complete-bout and legacy policies.
    """
    if selection is None:
        receipt = _tune_session(runner, store, task, tag, run_dir, round_no,
                                repo_root, cmd, events, job_runner)
    else:
        receipt = _tune_pinned(runner, store, task, tag, run_dir, round_no,
                               repo_root, cmd, events, job_runner, selection)
    if receipt.get("outcome_status") != "infra_failure":
        _note_seat_progress(run_dir)
    return receipt


def _tune_pinned(runner, store, task, tag, run_dir, round_no, repo_root, cmd,
                 events, job_runner, selection) -> dict:
    """One pinned round_v1 TUNE bout; every ending funnels through one settle:
    deterministic close, then persistent-state zero-progress classification,
    then at most one reconciliation for a progressed ambiguity."""
    _refuse_if_blocked(run_dir)
    run_id = str(selection["run_id"])
    before = _tuning_snapshot(run_dir, run_id)
    receipt = None
    failure = None
    if _tuner_inner_policy(run_dir) == "legacy":
        origin = "session"
        try:
            receipt, _ = _invoke_with_driver_jobs(
                runner, store, "tuner-orchestrator", task, tag, run_dir,
                round_no=round_no, run_id=run_id,
                extra=_tune_invocation_extra(selection),
                repo_root=repo_root, job_runner=job_runner)
        except (InvocationFailed, DriverJobError) as exc:
            failure = "; ".join(
                str(p) for p in getattr(exc, "problems", [str(exc)]))
    else:
        origin = "bout"
        try:
            phase_c.run_single_bout(
                task, tag, run_dir, run_id, repo_root, cmd, job_runner,
                next_invocation_id=store.issue_invocation_id,
                trial_cap_fn=lambda action: int(action.get("bout_trials") or 0),
                round_no=round_no, finalize_task=task, resume_once=True)
            return {"tuned_run_id": run_id, "tuned": True,
                    "ledger_updated": True, "outcome_status": "valid"}
        except phase_c.PhaseCBoutFailure as exc:
            failure = str(exc)
    return _settle_pinned_tune(runner, store, task, tag, run_dir, round_no,
                               repo_root, cmd, events, job_runner, selection,
                               receipt, failure, before, origin)


def _settle_pinned_tune(runner, store, task, tag, run_dir, round_no, repo_root,
                        cmd, events, job_runner, selection, receipt, failure,
                        before, origin) -> dict:
    run_id = str(selection["run_id"])
    tuned_id = (receipt or {}).get("tuned_run_id", "none")
    # Deterministic close first: the report/action pair, not the receipt,
    # decides whether a bout landed — an exhausted bout must be closed and
    # applied even under a missing or tuned=false ending.
    if not _tune_flag(run_dir, run_id):
        finalized = _phase_c_recover_close(run_dir, run_id, repo_root, cmd,
                                           events, task)
        if finalized is not None:
            return {**(receipt or {}), **finalized, "tuned": True,
                    "tuned_run_id": run_id, "ledger_updated": True,
                    "outcome_status": "valid"}
    after = _tuning_snapshot(run_dir, run_id)
    if after == before:
        # Zero progress by persistent state: infra failure, no
        # reconciliation, whatever shape the ending took (breaker trip,
        # DriverJobError, clean empty receipt, truthful no-op).
        return {
            "tuned_run_id": run_id, "tuned": False, "ledger_updated": False,
            "outcome_status": "infra_failure",
            "outcome_reason": f"zero_progress_{origin}_failure",
            "outcome_detail": failure,
            "snapshot_counts": _snapshot_counts(after),
        }
    if (failure is None and receipt is not None and tuned_id == run_id
            and receipt.get("tuned") and _tune_flag(run_dir, run_id)):
        return {**receipt, "outcome_status": "valid"}
    if (failure is None and receipt is not None and tuned_id == run_id
            and not receipt.get("tuned")):
        # A truthful in-progress receipt: the stage stays open and resumable,
        # its paid trials preserved for a later decision.
        return receipt
    if failure is not None:
        note = f"tuner bout failed with progressed tuning state: {failure}"
    elif receipt is not None and receipt.get("tuned"):
        note = f"receipt claims tuned {tuned_id} but ledger has tune: false."
    else:
        note = ("session disowned the bout while the tuning state "
                "progressed")
    try:
        reconciled = _tuner_reconcile(runner, store, task, tag, run_dir,
                                      round_no, note, repo_root, job_runner,
                                      selection=selection)
    except (InvocationFailed, DriverJobError) as exc:
        problems = [str(p) for p in getattr(exc, "problems", [str(exc)])]
        events.emit("tune_reconcile_failed",
                    decision_id=selection.get("decision_id"), run_id=run_id,
                    first_failure=note, problems=problems,
                    snapshot_before=_snapshot_counts(before),
                    snapshot_after=_snapshot_counts(after))
        # Demotion (P3): the bout ended in a progressed-but-failed state and
        # its reconciliation session failed too. The decision closes as an
        # infra failure through the normal outcome path — which also feeds
        # the per-candidate tune backoff — and the round continues; the
        # run-level streak breaker catches a systemic pattern.
        receipt = {
            "tuned_run_id": run_id, "tuned": False, "ledger_updated": False,
            "outcome_status": "infra_failure",
            "outcome_reason": "tune_reconcile_failed",
            "outcome_detail": f"{note}; reconciliation failed: {problems}",
            "snapshot_counts": _snapshot_counts(after),
        }
        try:
            _note_seat_skip(run_dir, repo_root, cmd, events,
                            role="tuner-orchestrator", problems=problems,
                            run_id=run_id)
        except RunBlocked:
            # The streak trip wins: close the bout's decision before
            # unwinding so the blocked run leaves no open scheduler
            # decision behind.
            _close_failed_tune(run_dir, repo_root, cmd, events, selection,
                               phase_c_before=before["phase_c_attempts"])
            raise
        return receipt
    if reconciled.get("tuned") and not _tune_flag(
            run_dir, reconciled.get("tuned_run_id", "none")):
        _close_failed_tune(run_dir, repo_root, cmd, events, selection,
                           phase_c_before=before["phase_c_attempts"])
        _or_block(run_dir, repo_root, cmd, events,
                  "authoritative artifacts still contradict after tuner "
                  "reconciliation")
    return reconciled


def _tune_session(runner, store, task, tag, run_dir, round_no, repo_root, cmd,
                  events, job_runner=execute_driver_job) -> dict:
    """The orchestrator-side tuning step (complete-bout and legacy policies):
    the session selects its own candidate and owns the bout handoffs."""
    tuner_inv = None  # set only on a successful first invocation
    try:
        receipt, tuner_inv = _invoke_with_driver_jobs(
            runner, store, "tuner-orchestrator", task, tag, run_dir,
            round_no=round_no, repo_root=repo_root, job_runner=job_runner)
    except (InvocationFailed, DriverJobError) as exc:
        # The reconcile session is the recovery path: hand it the first
        # session's problems, and keep them for the block reason too.
        first_problems = [str(p) for p in
                          getattr(exc, "problems", [str(exc)])]
        try:
            receipt = _tuner_reconcile(
                runner, store, task, tag, run_dir, round_no,
                "tuner session failed: " + "; ".join(first_problems),
                repo_root, job_runner)
        except (InvocationFailed, DriverJobError) as exc:
            problems = [str(p) for p in
                        getattr(exc, "problems", [str(exc)])]
            _or_block(run_dir, repo_root, cmd, events,
                      f"tuner session failed: {first_problems}; "
                      f"reconciliation failed: {problems}")
    tuned_id = receipt.get("tuned_run_id", "none")
    # The receipt is only a handoff.  If it names a candidate, consult the
    # deterministic phase-C action even when tuned=false; an exhausted bout
    # must be closed and applied before the round can end.
    if tuned_id != "none" and not _tune_flag(run_dir, tuned_id):
        finalized = _phase_c_recover_close(
            run_dir, tuned_id, repo_root, cmd, events, task)
        if finalized is not None:
            receipt = {**receipt, **finalized, "tuned": True,
                       "tuned_run_id": tuned_id, "ledger_updated": True}
    # contradiction: receipt claims applied but the ledger still disagrees
    if receipt.get("tuned") and tuned_id != "none" and \
            not _tune_flag(run_dir, tuned_id):
        note = (f"receipt claims tuned {tuned_id} but ledger has tune: false.")
        # spec: corrective follow-up in the SAME tuner session first
        corrected = None
        reconcile_note = note
        try:
            corrected, _ = _invoke_with_driver_jobs(
                runner, store, "tuner-orchestrator", task, tag, run_dir,
                round_no=round_no, resume_from=tuner_inv,
                extra={"reconcile_note": note + _RECONCILE_GUIDANCE},
                repo_root=repo_root, job_runner=job_runner)
        except InvocationFailed as exc:
            corrected = None
            reconcile_note = note + (
                " The same-session corrective attempt failed: "
                + "; ".join(str(p) for p in exc.problems))
        if corrected is None or (corrected.get("tuned") and not _tune_flag(
                run_dir, corrected.get("tuned_run_id", "none"))):
            try:
                corrected = _tuner_reconcile(
                    runner, store, task, tag, run_dir, round_no,
                    reconcile_note, repo_root, job_runner)
            except (InvocationFailed, DriverJobError) as exc:
                problems = getattr(exc, "problems", [str(exc)])
                _or_block(run_dir, repo_root, cmd, events,
                      f"tuner receipt/ledger contradiction unresolved: "
                          f"{problems}")
        if corrected.get("tuned") and \
                not _tune_flag(run_dir, corrected.get("tuned_run_id", "none")):
            _or_block(run_dir, repo_root, cmd, events,
                      "authoritative artifacts still contradict after "
                      "tuner reconciliation")
        return corrected
    return receipt


# =============================================================================
# Run setup and provided-baseline reconciliation
# =============================================================================


def _dimension_strategy(run_dir: Path) -> str | None:
    """Read the strategy frozen into this run's framework config."""
    config_path = run_dir / "framework_cfg.json"
    return json.loads(config_path.read_text(encoding="utf-8")).get(
        "dimension_strategy")


def _init_run_extra(dimension_strategy, llm_intelligence_score,
                    semantic_policy, scheduler_policy, inner_policy,
                    k_warm, k_eval, proposer_arm=None, time_budget=None,
                    deadline=None, final_reserve=None,
                    round_options=None, session_concurrency=None,
                    rewrite_concurrency=None, no_eval_timeout=False) -> list[str]:
    extra = ["--no-eval-timeout"] if no_eval_timeout else []
    if session_concurrency is not None:
        extra += ["--session-concurrency", str(session_concurrency)]
    if rewrite_concurrency is not None:
        extra += ["--rewrite-concurrency", str(rewrite_concurrency)]
    if time_budget is not None:
        extra += ["--time-budget", str(time_budget)]
    if deadline is not None:
        extra += ["--deadline", str(deadline)]
    if final_reserve is not None:
        extra += ["--final-reserve", str(final_reserve)]
    for key, value in sorted((round_options or {}).items()):
        if value is not None:
            flag = key.replace("_", "-")
            if not flag.startswith("round-"):
                flag = f"round-{flag}"
            extra += [f"--{flag}", str(value)]
    if dimension_strategy:
        extra += ["--dimension-strategy", dimension_strategy]
    if llm_intelligence_score is not None:
        extra += ["--llm-intelligence-score", str(llm_intelligence_score)]
    if semantic_policy is not None:
        extra += ["--semantic-policy", semantic_policy]
    if scheduler_policy is not None:
        extra += ["--scheduler-policy", scheduler_policy]
    if inner_policy is not None:
        extra += ["--inner-tuner-policy", inner_policy]
    if proposer_arm is not None:
        extra += ["--proposer-arm", proposer_arm]
    if k_warm is not None:
        extra += ["--k-warm", str(k_warm)]
    if k_eval is not None:
        extra += ["--k-eval", str(k_eval)]
    return extra


def _setup(runner, store, task, tag, run_dir, task_toml, repo_root, cmd,
           events, max_evaluations, timeout, dimension_strategy,
           llm_intelligence_score, semantic_policy, scheduler_policy,
           inner_policy, k_warm, k_eval, model, cli_path,
           proposer_arm=None, time_budget=None, deadline=None,
           final_reserve=None, round_options=None,
           session_concurrency=None, rewrite_concurrency=None, no_eval_timeout=False) -> None:
    extra = _init_run_extra(
        dimension_strategy,
        llm_intelligence_score,
        semantic_policy,
        scheduler_policy,
        inner_policy,
        k_warm,
        k_eval,
        proposer_arm=proposer_arm,
        time_budget=time_budget,
        deadline=deadline,
        final_reserve=final_reserve,
        round_options=round_options,
        session_concurrency=session_concurrency,
        rewrite_concurrency=rewrite_concurrency,
        no_eval_timeout=no_eval_timeout,
    )
    common.init_run(task, tag, repo_root, cmd, max_evaluations, timeout,
                    extra=extra)
    cmd(["uv", "--directory", common.task_project(task, task_toml), "sync"], repo_root)
    # the protocol gates this on "required assets absent"; the task
    # contract exposes no deterministic signal for that, so a declared
    # prepare command runs on fresh setup (same choice as the hillclimb
    # port). A non-idempotent prepare command is a task-contract bug.
    try:
        common.run_prepare(task, task_toml, repo_root, cmd)
    except RuntimeError as exc:
        _or_block(run_dir, repo_root, cmd, events, str(exc))
    common.preflight_env(task, run_dir, repo_root, cmd)
    competition_id = (task_toml.get("mlebench") or {}).get("competition_id")
    write_metadata(run_dir, model, cli_path, competition_id=competition_id)
    _ensure_identity_profile(run_dir, task, competition_id)
    brief = ensure_brief(run_dir, task_toml)
    events.emit("objective_brief", path=str(run_dir / "objective_brief.json"),
                metric=brief.get("metric"),
                aspirational_target_score=brief.get(
                    "aspirational_target_score"),
                target_source=brief.get("target_source"))

    # background-researcher runs ONCE, never in the loop. A run dir pre-seeded
    # with a frozen background (background.md + retrieval manifest) skips
    # generation entirely; the same deterministic validators gate it, and a
    # failure blocks rather than letting the researcher rewrite the frozen
    # artifacts.
    strategy = _dimension_strategy(run_dir)
    objective = _objective_line(task_toml)
    preseeded = (
        (run_dir / "background.md").exists()
        and (run_dir / "background_retrieval.json").exists()
    )
    if preseeded:
        errors = _background_validation_errors(run_dir, repo_root, cmd,
                                               strategy)
        if errors:
            _or_block(run_dir, repo_root, cmd, events,
                      f"pre-seeded background validation failed: {errors}")
        _background_faithfulness_gate(
            runner, store, task, tag, run_dir, repo_root, cmd, events,
            strategy,
            repairable=_background_researcher_invoked(run_dir))
    else:
        try:
            _invoke(runner, store, "background-researcher", task, tag, run_dir,
                    extra={"objective": objective})
        except InvocationFailed as exc:
            _or_block(run_dir, repo_root, cmd, events,
                      f"background-researcher failed: {exc.problems}")
        _validate_background(runner, store, task, tag, run_dir, repo_root, cmd,
                             events, strategy, number_gate=True)
        _background_faithfulness_gate(
            runner, store, task, tag, run_dir, repo_root, cmd, events,
            strategy,
            repairable=_background_researcher_invoked(run_dir))


def _ensure_identity_profile(run_dir, task, competition_id) -> None:
    """Write <run_dir>/task_identity_profile.json for MLE tasks (idempotent).

    The retrieval adapter fails closed when a competition_id is known but no
    parseable profile sits beside the manifest; non-MLE tasks get no file.
    """
    profile = competition_policy.derive_profile(task, competition_id)
    if profile is None:
        return
    path = run_dir / "task_identity_profile.json"
    if path.exists():
        try:
            json.loads(path.read_text(encoding="utf-8"))
            return
        except (OSError, json.JSONDecodeError):
            pass
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _background_researcher_invoked(run_dir) -> bool:
    """Whether this run ever invoked the background researcher.

    A receipt or a session file (persisted at session init, so the kill
    window "background written, receipt not yet delivered" counts too) under
    ``receipts/`` both prove it; a pre-seeded run dir has neither.
    """
    receipts = run_dir / "receipts"
    if not receipts.is_dir():
        return False
    return any(
        path.name.startswith("background-researcher-") for path in receipts.iterdir()
    )


def _validator_error_messages(result) -> list[str]:
    """Named validator errors for in-session repair and the block reason.

    ``background_contract.py validate`` and ``search_backends.py validate``
    print JSON with an ``errors`` list on stdout and leave stderr empty on a
    clean reject.  Prefer those named strings; stderr is only a supplement
    (or the fallback when stdout is not that JSON).
    """
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    named: list[str] = []
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
        else:
            if isinstance(payload, dict):
                raw = payload.get("errors")
                if isinstance(raw, list):
                    named = [str(item) for item in raw if str(item).strip()]
    if named:
        if stderr:
            named.append(stderr)
        return named
    if stderr:
        return [stderr]
    if stdout:
        return [stdout]
    return ["validation failed"]


def _background_validation_errors(run_dir, repo_root, cmd,
                                  strategy, *, number_gate=False) -> list[str]:
    """Run all deterministic validators for background-research artifacts."""
    checks = []
    if strategy == "llm_induced":
        checks.append([
            "python", "tools/background_contract.py", "catalog",
            "--path", run_dir / "dimension_catalog.json",
        ])
    contract_validate = [
        "python", "tools/background_contract.py", "validate",
        "--background", run_dir / "background.md",
        "--retrieval-manifest", run_dir / "background_retrieval.json",
    ]
    if number_gate:
        contract_validate.append("--number-gate")
    checks.extend([
        ["python", "tools/search_backends.py", "validate",
         "--manifest", run_dir / "background_retrieval.json"],
        contract_validate,
    ])

    errors = []
    for check_args in checks:
        result = cmd(check_args, repo_root, check=False)
        if result.returncode != 0:
            errors.extend(_validator_error_messages(result))
    return errors


def _validate_background(runner, store, task, tag, run_dir, repo_root, cmd,
                         events, strategy, *, number_gate=False) -> None:
    """Validate background artifacts and allow one in-session repair."""
    errors = _background_validation_errors(run_dir, repo_root, cmd, strategy,
                                           number_gate=number_gate)
    if not errors:
        return
    # feed validator errors back in-session once; never migrate the frozen space
    try:
        _invoke(runner, store, "background-researcher", task, tag, run_dir,
                extra={"validation_errors": "\n".join(errors)})
    except InvocationFailed as exc:
        _or_block(run_dir, repo_root, cmd, events,
                  f"background validation repair failed: {exc.problems}")
    errors = _background_validation_errors(run_dir, repo_root, cmd, strategy,
                                           number_gate=number_gate)
    if errors:
        _or_block(run_dir, repo_root, cmd, events,
                  f"background validation failed: {errors}")


def _background_faithfulness_gate(runner, store, task, tag, run_dir,
                                  repo_root, cmd, events, strategy, *,
                                  repairable: bool) -> None:
    """Synchronous faithfulness audit after validation, before the space freezes.

    A repairable run (the researcher ran on this run dir) gets one researcher
    repair round on unfaithful findings, followed by a narrowed re-audit; a
    pre-seeded frozen background cannot be rewritten, so its unfaithful
    findings are recorded as a terminal warning instead of blocking.
    """
    repair = None
    if repairable:
        def repair(findings_text: str) -> None:
            try:
                _invoke(runner, store, "background-researcher", task, tag,
                        run_dir,
                        extra={"faithfulness_findings": findings_text})
            except InvocationFailed as exc:
                _or_block(run_dir, repo_root, cmd, events,
                          f"background faithfulness repair failed: {exc.problems}")
            _validate_background(
                runner, store, task, tag, run_dir, repo_root, cmd, events,
                strategy,
                number_gate=_background_researcher_invoked(run_dir))

    background_audit.run_faithfulness_gate(
        runner, store, task, tag, run_dir, events,
        invoke=_invoke,
        or_block=lambda reason: _or_block(run_dir, repo_root, cmd, events,
                                          reason),
        repair=repair,
    )


def _resume_setup(runner, store, task, tag, run_dir, repo_root, cmd, events,
                  model, cli_path, task_toml=None) -> None:
    """Finish any interrupted setup work and restore a runnable phase."""
    task_toml = task_toml or common.load_task_toml(task, repo_root)
    competition_id = (task_toml.get("mlebench") or {}).get("competition_id")
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        for warning in warn_on_mismatch(run_dir, model, cli_path):
            events.emit("metadata_mismatch", warning=warning)
    else:
        # The run was killed before fresh setup reached write_metadata().
        write_metadata(run_dir, model, cli_path, competition_id=competition_id)
    _ensure_identity_profile(run_dir, task, competition_id)
    ensure_brief(run_dir, task_toml)

    # A prior block may have named prepare itself as the failure. The resume
    # path otherwise never re-runs prepare, so a block+resume cycle would
    # silently bypass the failed premise: re-run it under the same failure
    # condition and block again if it still fails (Wave 0.4).
    ledger_path = run_dir / "ledger.json"
    if ledger_path.exists():
        try:
            prior_state = (json.loads(ledger_path.read_text(
                encoding="utf-8")).get("run_state") or {})
        except (OSError, json.JSONDecodeError):
            prior_state = {}
        if prior_state.get("phase") == "blocked" and str(
                prior_state.get("active_stop_condition") or ""
        ).startswith("prepare_command failed"):
            try:
                common.run_prepare(task, task_toml, repo_root, cmd)
            except RuntimeError as exc:
                _or_block(run_dir, repo_root, cmd, events, str(exc))

    common.preflight_env(task, run_dir, repo_root, cmd)

    # Status consumers must not keep seeing a stale terminal phase after an
    # explicit resume has started making progress again.
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        stale_phase = ledger.get("run_state", {}).get("phase")
        if stale_phase in ("blocked", "completed"):
            common.set_phase(run_dir, repo_root, cmd, "running")
            events.emit("resumed_from_terminal", phase=stale_phase)

    background_missing = (
        not (run_dir / "background.md").exists()
        or not (run_dir / "background_retrieval.json").exists()
    )
    if background_missing:
        # The run was killed while the one-time background-research step was
        # in progress. Resume that setup phase before entering ideation.
        try:
            _invoke(runner, store, "background-researcher", task, tag, run_dir,
                    extra={"objective": _objective_line(task_toml)})
        except InvocationFailed as exc:
            _or_block(run_dir, repo_root, cmd, events,
                      f"background-researcher failed: {exc.problems}")

    # A resume may follow a kill after the researcher wrote its files but
    # before setup validated them.  Establish the frozen-space invariant once
    # at this process boundary; rounds trust it.  The number gate applies iff
    # this run ever invoked the researcher (a receipt or an init-time session
    # file proves it); a pre-seeded frozen background stays gate-off.
    _validate_background(
        runner, store, task, tag, run_dir, repo_root, cmd, events,
        _dimension_strategy(run_dir),
        number_gate=_background_researcher_invoked(run_dir),
    )

    # The same kill window covers the faithfulness audit.  Re-run the gate
    # unless a prior audit reached a terminal-ok outcome on these artifacts;
    # a blocked audit is not terminal, so every manual resume gets a fresh
    # audit+repair round (the resume is the decision to retry).  Repairable
    # follows the same researcher-invoked criterion as setup; a pre-seeded
    # frozen background records unfaithful findings as a terminal warning
    # rather than blocking.
    if not background_audit.audit_completed(run_dir):
        _background_faithfulness_gate(
            runner, store, task, tag, run_dir, repo_root, cmd, events,
            _dimension_strategy(run_dir),
            repairable=_background_researcher_invoked(run_dir))


def _provided_baseline(runner, store, task, tag, run_dir, repo_root, cmd,
                       events, seed, job_runner=execute_driver_job) -> None:
    semantic = run_dir / ".semantic" / "000"
    semantic.mkdir(parents=True, exist_ok=True)
    cmd(["python", "tools/semantic_search.py", "propose",
         "--background", run_dir / "background.md",
         "--ledger", run_dir / "ledger.json",
         "--op", "fresh", "--baseline-only",
         "--output", semantic / "proposals.json"], repo_root)
    cmd(["python", "tools/semantic_search.py", "select",
         "--proposals", semantic / "proposals.json",
         "--policy", "coverage_experience",
         "--ledger", run_dir / "ledger.json",
         "--point-output", semantic / "point.json",
         "--receipt-output", semantic / "policy.json"], repo_root)
    entrypoint = seed.get("entrypoint", "train.py")
    # Nobody planned this candidate: it is the task's own file, copied in. The
    # route arm records that explicitly rather than inventing a planned route.
    route_provenance = semantic / "route-provenance.json"
    route_provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "not_applicable",
                "reason": f"task-provided baseline entrypoint {entrypoint}",
            },
            indent=2,
        )
        + "\n"
    )
    cmd(["python", "tools/ledger.py", "add-record",
         "--ledger", run_dir / "ledger.json", "--task", task,
         "--run-id", "000", "--kind", "optimization", "--op", "fresh",
         "--source-run-ids", "",
         "--idea", "Use the unchanged task-provided baseline implementation.",
         "--change", "provided baseline at all-baselines point",
         "--background", run_dir / "background.md",
         "--semantic-point", semantic / "point.json",
         "--policy-receipt", semantic / "policy.json",
         "--candidate-name-hint", "provided_baseline",
         "--role", "task_provided_baseline",
         "--route-provenance", route_provenance,
         "--description", f"Task-provided baseline: {entrypoint}"], repo_root)
    cmd(["python", "tools/new_candidate.py", task, tag, "000",
         "--provided-baseline"], repo_root)
    try:
        receipt, _ = _invoke(runner, store, "candidate-writer", task, tag,
                             run_dir, run_id="000",
                             extra={"candidate_dir":
                                    str(run_dir / "candidates" / "000"),
                                    "expect": "status: existing, wrote: false"})
        if receipt.get("status") != "existing" or receipt.get("wrote"):
            _or_block(run_dir, repo_root, cmd, events,
                      "provided-baseline writer receipt violates existing/false")
    except InvocationFailed as exc:
        _or_block(run_dir, repo_root, cmd, events,
                  f"provided-baseline writer failed: {exc.problems}")
    _implement_candidate_extractor_only(runner, store, task, tag, run_dir,
                                        "000", repo_root, cmd, events,
                                        job_runner)


def _implement_candidate_extractor_only(runner, store, task, tag, run_dir,
                                        run_id, repo_root, cmd, events,
                                        job_runner=execute_driver_job) -> None:
    """Provided baseline: extractor step 0+1 only; a crash BLOCKS the run
    (the protocol forbids searching on without the control)."""
    candidate_dir = run_dir / "candidates" / run_id
    try:
        _invoke_with_driver_jobs(
            runner, store, "tunable-contract-extractor", task, tag, run_dir,
            run_id=run_id,
            extra={"candidate_dir": str(candidate_dir)}, repo_root=repo_root,
            job_runner=job_runner)
        return
    except InvocationFailed as exc:
        problems = [str(p) for p in exc.problems]
    evidence = _failure_evidence(candidate_dir, problems)
    repair_problems = None
    if evidence:
        try:
            verdict = common.crash_diagnose(
                runner, store, task, tag, run_dir, evidence)["verdict"]
        except InvocationFailed as exc:
            events.emit("crash_diagnosis_failed", run_id=run_id,
                        problems=exc.problems)
            verdict = "abandon"
        if verdict != "abandon":
            try:
                _invoke_with_driver_jobs(
                    runner, store, "tunable-contract-extractor", task,
                    tag, run_dir, run_id=run_id,
                    extra={"candidate_dir": str(candidate_dir),
                           "diagnosis_verdict": verdict},
                    repo_root=repo_root, job_runner=job_runner)
                return
            except InvocationFailed as exc:
                repair_problems = [str(p) for p in exc.problems]
    _record_crash(run_dir, run_id, repo_root, cmd)
    detail = "; ".join(problems)
    if repair_problems:
        detail += f"; repair failed: {'; '.join(repair_problems)}"
    _or_block(run_dir, repo_root, cmd, events,
              "provided baseline could not be evaluated; crash recorded: "
              + detail)


def _ensure_provided_baseline(runner, store, task, tag, run_dir, task_toml,
                              repo_root, cmd, events,
                              job_runner=execute_driver_job) -> None:
    """Install or reconcile the task-provided control before ideation."""
    seed = task_toml.get("seed", {})
    if not seed.get("provided"):
        return

    records = _ledger_records(run_dir)
    if not records:
        _provided_baseline(runner, store, task, tag, run_dir,
                           repo_root, cmd, events, seed, job_runner)
        return

    if all(record.get("run_id") != "000" for record in records):
        _or_block(
            run_dir, repo_root, cmd, events,
            "seed.provided baseline missing: the ledger has records but no "
            "000; the protocol forbids retrofitting the control after "
            "ideation",
        )


# =============================================================================
# One round of the experiment loop
# =============================================================================


def _resume_pending_candidates(runner, store, task, tag, run_dir, brief,
                               repo_root, cmd, events,
                               job_runner=execute_driver_job) -> tuple[
                                   dict | None, list[str]]:
    """Resume admitted-but-unimplemented candidates before new ideation."""
    pending_ids = (brief or {}).get("pending_run_ids") or []
    if not pending_ids:
        return brief, pending_ids

    for pending_id in pending_ids:
        if budget_status(run_dir, repo_root, cmd).get("reached"):
            break
        _materialize_candidate(task, tag, run_dir, pending_id, repo_root, cmd)
        _implement_candidate(runner, store, task, tag, run_dir, pending_id,
                             repo_root, cmd, events, job_runner)

    return _brief(run_dir, repo_root, cmd), pending_ids


def _evaluate_generation(runner, store, task, tag, run_dir, round_no,
                         repo_root, cmd, events,
                         job_runner=execute_driver_job, model=None,
                         task_toml=None) -> list[dict]:
    """Generate one bounded action batch and take each action through step 0+1."""
    if _semantic_policy(run_dir) == "judged_slate":
        return _evaluate_judged_generation(runner, store, task, tag, run_dir,
                                           round_no, repo_root, cmd, events,
                                           model, job_runner)
    actions = _ideate(runner, store, task, tag, run_dir, round_no,
                      repo_root, cmd, events, task_toml or {})
    _implement_seats(runner, store, task, tag, run_dir,
                     [str(action["run_id"]) for action in actions],
                     repo_root, cmd, events, job_runner)
    return actions


def _round_step(runner, store, task, tag, run_dir, round_no, task_toml,
                repo_root, cmd, events, job_runner, model, *,
                ledger_exists) -> tuple[list[dict], bool]:
    """One round_v1 iteration: a generation, or an optimization phase.

    The generation phase runs while the cycle's threshold is unmet and the
    remaining usable time still covers the next optimization quota; otherwise
    the optimization phase runs on the whole candidate pool and starts the
    next cycle's count.
    """
    view = (rounds.status(run_dir, repo_root, cmd) if ledger_exists
            else {"generate": True})
    if view["generate"]:
        actions = _evaluate_generation(
            runner, store, task, tag, run_dir, round_no,
            repo_root, cmd, events, job_runner, model=model,
            task_toml=task_toml,
        )
        return actions, False
    events.emit("round_optimize", round_no=round_no,
                produced=view.get("produced"), threshold=view.get("threshold"),
                final_round=view.get("final_round"))

    def tune(no, selection):
        return _tune(runner, store, task, tag, run_dir, no, repo_root, cmd,
                     events, job_runner, selection=selection)

    progressed = rounds.optimization_phase(
        runner, store, task, tag, run_dir, round_no, task_toml, repo_root,
        cmd, events, tune=tune, config=view["config"])
    return [], progressed


# =============================================================================
# Public entry point
# =============================================================================


def run_experiment(task, tag, *, runner, model, repo_root=REPO_ROOT,
                   max_evaluations=None, timeout=None, dimension_strategy=None,
                   llm_intelligence_score=None, semantic_policy=None,
                   scheduler_policy=None, inner_policy=None, k_warm=None,
                   k_eval=None, proposer_arm=None, time_budget=None,
                   deadline=None, final_reserve=None, round_options=None,
                   session_concurrency=None, rewrite_concurrency=None,
                   cli_path=None, finalization=None, no_eval_timeout=False,
                   cmd=common.run_cmd, job_runner=execute_driver_job) -> dict:
    """Set up or resume a run, then advance it until blocked or complete.

    ``finalization`` ({"submission_command", "data_dir"}) registers the
    operator's export contract for the degraded-delivery attempt. After a
    persisted block and all channels have joined, the run boundary attempts
    a best-effort submission.csv from the settled evidence.

    Boundary rule (P1): every failure ends as either a deterministic
    continue or a persisted block — an unhandled exception is converted to a
    blocked phase here, never a traceback exit with the ledger still
    claiming running."""
    run_dir = repo_root / "runs" / task / tag
    events = EventsLog(run_dir)
    task_toml = common.load_task_toml(task, repo_root)
    store = ReceiptStore(run_dir)
    _reset_block_state(run_dir)
    _reset_seat_skip_state(run_dir)
    rounds.reset_tune_backoff(run_dir)
    if finalization and finalization.get("submission_command"):
        _delivery_cfg[str(run_dir)] = {
            "task": task,
            "submission_command": finalization["submission_command"],
            "data_dir": str(finalization.get("data_dir") or ""),
        }
    _arm_exit_hooks()  # main thread: SIGTERM must reach leased child groups

    try:
        if not (run_dir / "framework_cfg.json").exists():
            _setup(runner, store, task, tag, run_dir, task_toml, repo_root,
                   cmd, events, max_evaluations, timeout, dimension_strategy,
                   llm_intelligence_score, semantic_policy, scheduler_policy,
                   inner_policy, k_warm, k_eval, model, cli_path,
                   proposer_arm=proposer_arm, time_budget=time_budget,
                   deadline=deadline, final_reserve=final_reserve,
                   round_options=round_options,
                   session_concurrency=session_concurrency,
                   rewrite_concurrency=rewrite_concurrency,
                   no_eval_timeout=no_eval_timeout)
        else:
            # Explicit CLI overrides must never disappear merely because the
            # run directory already exists. init_run applies mutable limits,
            # accepts idempotent frozen values, and rejects policy/space
            # changes once their artifacts exist. An existing deadline is
            # frozen: do not re-pass --time-budget (that would be now+budget).
            cfg_path = run_dir / "framework_cfg.json"
            existing_deadline = None
            try:
                existing_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing_cfg = {}
            raw_deadline = existing_cfg.get("deadline") if isinstance(
                existing_cfg, dict) else None
            if isinstance(raw_deadline, (int, float)) and not isinstance(
                    raw_deadline, bool):
                existing_deadline = float(raw_deadline)
            extra = _init_run_extra(
                dimension_strategy,
                llm_intelligence_score,
                semantic_policy,
                scheduler_policy,
                inner_policy,
                k_warm,
                k_eval,
                proposer_arm=proposer_arm,
                time_budget=None if existing_deadline is not None else time_budget,
                deadline=None if existing_deadline is not None else deadline,
                final_reserve=final_reserve,
                round_options=round_options,
                session_concurrency=session_concurrency,
                rewrite_concurrency=rewrite_concurrency,
                no_eval_timeout=no_eval_timeout,
            )
            if max_evaluations is not None or timeout is not None or extra:
                common.init_run(
                    task,
                    tag,
                    repo_root,
                    cmd,
                    max_evaluations,
                    timeout,
                    extra=extra,
                )
            _resume_setup(runner, store, task, tag, run_dir, repo_root, cmd,
                          events, model, cli_path, task_toml=task_toml)

        # Applies to both fresh setup and resume. A kill between init_run and
        # add-record must not let a provided control silently become seedless.
        _ensure_provided_baseline(runner, store, task, tag, run_dir, task_toml,
                                  repo_root, cmd, events, job_runner)

        round_no = 0
        zero_progress_rounds = 0
        while True:
            # -----------------------------------------------------------------
            # Round step 0: recover in-flight work, then honor the budget.
            # Seedless first rounds have no ledger and proceed to ideation.
            # -----------------------------------------------------------------
            ledger_exists = (run_dir / "ledger.json").exists()
            brief = _brief(run_dir, repo_root, cmd) if ledger_exists else None
            brief, pending_ids = _resume_pending_candidates(
                runner, store, task, tag, run_dir, brief,
                repo_root, cmd, events, job_runner,
            )

            if brief is not None and budget_status(
                    run_dir, repo_root, cmd).get("reached"):
                # Only drain records that are still pending after any work
                # completed before the reservation boundary.  A candidate
                # may have become terminal on the final available call.
                unresolved = (_brief(run_dir, repo_root, cmd) or {}).get(
                    "pending_run_ids", [])
                at_deadline = _time_reached(run_dir)
                for pending_id in unresolved:
                    if at_deadline:
                        _settle_at_deadline(run_dir, pending_id, repo_root,
                                            cmd, events)
                    else:
                        _resolve_unevaluated(run_dir, pending_id, repo_root, cmd)
                brief = _brief(run_dir, repo_root, cmd)
                if brief.get("experience_refresh_required"):
                    _refresh(runner, store, task, tag, run_dir, repo_root,
                             cmd, events)
                _complete_run(run_dir, repo_root, cmd, events,
                              stop_condition=(
                                  "budget_reached_tune_backoff_no_success"
                                  if rounds.tune_backoff_no_success(run_dir)
                                  else None))
                break

            # -----------------------------------------------------------------
            # Round step 1: refresh experience at a revision boundary.
            # -----------------------------------------------------------------
            refreshed = False
            if brief is not None and brief.get("experience_refresh_required"):
                _refresh(runner, store, task, tag, run_dir, repo_root, cmd,
                         events)
                refreshed = True

            # -----------------------------------------------------------------
            # Round steps 2+3 under round_v1: generate until the cycle's
            # candidate threshold is met (the whole seed set first, then N new
            # candidates), then one optimization phase over the full pool.
            # -----------------------------------------------------------------
            if _scheduler_policy(run_dir) == rounds.POLICY_ID:
                actions, tuner_progressed = _round_step(
                    runner, store, task, tag, run_dir, round_no, task_toml,
                    repo_root, cmd, events, job_runner, model,
                    ledger_exists=ledger_exists)
                progressed = (bool(actions) or tuner_progressed or refreshed
                              or bool(pending_ids))
                zero_progress_rounds = 0 if progressed else zero_progress_rounds + 1
                if zero_progress_rounds >= 2:
                    events.emit("quiescent", round_no=round_no,
                                reason="two consecutive zero-progress rounds")
                    _complete_run(run_dir, repo_root, cmd, events,
                                  stop_condition=(
                                      "quiescent_tune_backoff_no_success"
                                      if rounds.tune_backoff_no_success(run_dir)
                                      else "quiescent"))
                    break
                round_no += 1
                continue

            # -----------------------------------------------------------------
            # Round step 2: generate candidates; evaluate each through step 0+1.
            # Full background validation belongs to setup/resume; the
            # high-frequency helpers enforce only the dynamic invariants they
            # consume.
            # -----------------------------------------------------------------
            actions = _evaluate_generation(
                runner, store, task, tag, run_dir, round_no,
                repo_root, cmd, events, job_runner, model=model,
                task_toml=task_toml,
            )

            # -----------------------------------------------------------------
            # Round step 3: run at most one decoupled deep-tuning bout.
            # -----------------------------------------------------------------
            tuner_progressed = False
            if not budget_status(run_dir, repo_root, cmd).get("reached"):
                tuner_receipt = _tune(runner, store, task, tag, run_dir,
                                      round_no, repo_root, cmd, events,
                                      job_runner)
                tuner_progressed = bool(tuner_receipt.get("tuned"))
                if not tuner_progressed and _scheduler_stopped(run_dir):
                    events.emit("quiescent", round_no=round_no,
                                reason="scheduler terminal STOP")
                    remaining = objective_budget_status(run_dir).get("remaining")
                    _complete_run(
                        run_dir, repo_root, cmd, events,
                        stop_condition="scheduler_stop",
                        terminal_leftover=(isinstance(remaining, int)
                                           and 0 < remaining < MIN_GENERATION_K_EVAL),
                    )
                    break

            # -----------------------------------------------------------------
            # Completion guard: stop if no operation can spend the budget.
            # -----------------------------------------------------------------
            # One leftover objective call cannot form the schema-4
            # control/treatment minimum, so close it immediately with an
            # explicit reason. Other zero-progress states still need two
            # observations because scheduler eligibility may depend on
            # non-budget facts repaired by the intervening role invocation.
            progressed = (bool(actions) or tuner_progressed or refreshed
                          or bool(pending_ids))
            remaining = objective_budget_status(run_dir).get("remaining")
            if not progressed and isinstance(remaining, int) \
                    and 0 < remaining < MIN_GENERATION_K_EVAL:
                events.emit(
                    "quiescent",
                    round_no=round_no,
                    reason=(
                        "remaining objective budget is below the minimum "
                        f"meaningful action cost ({MIN_GENERATION_K_EVAL})"
                    ),
                    unused_budget=remaining,
                )
                _complete_run(run_dir, repo_root, cmd, events,
                              stop_condition="insufficient_remaining_budget",
                              terminal_leftover=True)
                break
            zero_progress_rounds = 0 if progressed else zero_progress_rounds + 1
            if zero_progress_rounds >= 2:
                events.emit("quiescent", round_no=round_no,
                            reason="two consecutive zero-progress rounds")
                _complete_run(run_dir, repo_root, cmd, events,
                              stop_condition="quiescent")
                break
            round_no += 1
    except RunBlocked:
        pass
    except Exception as exc:  # noqa: BLE001 - the boundary rule (P1)
        # Every unhandled failure becomes a persisted block; the traceback
        # rides along in the event stream so an external observer can tell
        # "died with a traceback" apart from "died into blocked".
        events.emit("unhandled_exception", error=repr(exc),
                    traceback=traceback.format_exc()[-4000:])
        try:
            _or_block(run_dir, repo_root, cmd, events,
                      f"unhandled {type(exc).__name__}: {exc}")
        except RunBlocked:
            pass

    _finish_blocked_run(run_dir, repo_root, cmd, events)
    return compact_status(task, tag, run_dir, repo_root=repo_root, cmd=cmd)
