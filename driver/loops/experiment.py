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
             generate and evaluate one candidate generation
             run at most one decoupled tuning bout
        -> complete on budget exhaustion or quiescence

The helpers below are grouped by responsibility. Recovery policy stays close
to the operation it recovers, while ``run_experiment`` remains a compact map of
the complete lifecycle.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..events import EventsLog
from ..jobs import DriverJobError, execute_driver_job
from ..metadata import warn_on_mismatch, write_metadata
from ..receipts import ReceiptStore
from ..roles import (
    REPO_ROOT,
    ROLES,
    InvocationContext,
    driver_job_handoff_problem,
)
from ..session import InvocationFailed
from ..status import budget_status, compact_status
from . import background_audit
from . import common
from .common import RunBlocked
from tools.evaluation_budget import budget_status as objective_budget_status
from tools.scheduler.contract import (
    DEFAULT_K_EVAL,
    MIN_GENERATION_K_EVAL,
    ResourceContract,
)
from tools.scheduler.donor import build_donor_snapshot, donors_dir

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
            inline_payload=None) -> dict:
    """Invoke one role and return its persisted receipt plus invocation id."""
    inv_id = store.next_invocation_id()
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
        try:
            if handoff_problem:
                raise DriverJobError(handoff_problem)
            result = job_runner(
                role_name,
                ctx,
                receipt["driver_job"],
                repo_root=repo_root,
            )
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


def _or_block(run_dir, repo_root, cmd, events, reason: str):
    """Persist a blocked phase, then unwind to ``run_experiment``."""
    common.block(run_dir, repo_root, cmd, events, reason)
    raise RunBlocked(reason)


def _complete_run(run_dir, repo_root, cmd, events, terminal_leftover=False) -> None:
    """Persist normal completion, translating a refusal into a blocked run."""
    try:
        common.set_phase(run_dir, repo_root, cmd, "completed",
                         terminal_leftover=terminal_leftover)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or str(exc)).strip()
        _or_block(run_dir, repo_root, cmd, events,
                  f"set-phase completed refused: {detail}")


def _refresh(runner, store, task, tag, run_dir, repo_root, cmd, events) -> None:
    """Refresh bounded experience, with one artifact-aware retry."""
    try:
        _invoke(runner, store, "experience-extractor", task, tag, run_dir)
        return
    except InvocationFailed:
        pass
    try:  # one retry with reconciliation context, then block
        brief = _brief(run_dir, repo_root, cmd)
        _invoke(runner, store, "experience-extractor", task, tag, run_dir,
                extra={"reconcile_note":
                       "prior refresh failed postconditions; current ledger brief: "
                       + json.dumps(brief, sort_keys=True)})
        return
    except InvocationFailed as exc:
        _or_block(run_dir, repo_root, cmd, events,
                  f"experience refresh failed: {exc.problems}")


def _ideate(runner, store, task, tag, run_dir, round_no, repo_root, cmd,
            events) -> list[dict]:
    """Generate one action batch and ensure every returned action was admitted."""
    def admitted_missing(actions: list[dict]) -> list[str]:
        admitted = {r.get("run_id") for r in _ledger_records(run_dir)}
        return [a.get("run_id") for a in actions if a.get("run_id") not in admitted]

    try:
        receipt, _ = _invoke(runner, store, "idea-generator", task, tag,
                             run_dir, round_no=round_no)
    except InvocationFailed:
        receipt = None
    if receipt is not None and not admitted_missing(receipt.get("actions", [])):
        return receipt.get("actions", [])
    # one retry whose context reconciles against what is already admitted
    note = {
        "reconcile_note":
            "Records already admitted for this generation stand; never "
            "re-admit them. Complete only the missing work.",
        "admitted_run_ids": [r.get("run_id") for r in _ledger_records(run_dir)],
    }
    try:
        receipt, _ = _invoke(runner, store, "idea-generator", task, tag,
                             run_dir, round_no=round_no, extra=note)
    except InvocationFailed as exc:
        _or_block(run_dir, repo_root, cmd, events,
                  f"idea-generator failed: {exc.problems}")
    missing = admitted_missing(receipt.get("actions", []))
    if missing:
        _or_block(run_dir, repo_root, cmd, events,
                  f"idea actions not admitted after retry: {missing}")
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


def _find_open_slate_generation(run_dir: Path) -> tuple[Path, int, dict | None]:
    """The generation this round resumes into: (gen_dir, gen_no, manifest).

    ``manifest`` is the newest committed generation.json whose seats are not
    all in the ledger with a matching schema-8 binding — that generation must
    be resumed (or, for a binding violation, blocked by the caller).  When no
    manifest exists or the newest is fully admitted, ``manifest`` is None and
    (gen_dir, gen_no) name the next generation, whose provisional artifacts
    may be overwritten.
    """
    semantic = run_dir / ".semantic"
    count = (
        sum(1 for _ in semantic.glob("gen-*/generation.json"))
        if semantic.is_dir() else 0
    )
    if count:
        gen_dir = semantic / f"gen-{count:04d}"
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
    task_brief = repo_root / "tasks" / task / "TASK.md"
    if task_brief.is_file():
        args += ["--task-brief", task_brief]
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


def _slate_plan_problems(plan, slot: dict, route_arm: int) -> list[str]:
    """Driver-side receipt/plan checks: non-empty fields and slot binding."""
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
    if slot["carrier"]["op"] == "fresh":
        expected = f"from scratch at {slot['point_id']}"
        if isinstance(plan.get("change"), str) \
                and plan["change"].strip() != expected:
            problems.append(f"fresh-seat change must be {expected!r}")
    if route_arm and not isinstance(plan.get("route_provenance"), dict):
        problems.append(
            "route arm is active: plan needs a route_provenance object")
    return problems


def _slate_plan_payload(slot: dict, pool_doc: dict, context_doc: dict,
                        route_memory_path) -> str:
    """The bounded plan input, bound to the manifest's point and carrier."""
    entry = next(entry for entry in pool_doc["pool"]
                 if entry["label"] == slot["label"])
    parts = [
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
    ]
    if route_memory_path is not None:
        parts.append(
            f"Route memory for this seat (read before planning routes): "
            f"{route_memory_path}")
    return "\n\n".join(parts) + "\n"


def _ensure_slate_plans(runner, store, task, tag, run_dir, gen_dir, manifest,
                        round_no, repo_root, cmd, events) -> None:
    """One plan per seat; only missing or invalid plans are (re)written."""
    plans_dir = gen_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    route_arm = _slate_route_arm(run_dir)
    pool_doc = json.loads((gen_dir / "pool.json").read_text(encoding="utf-8"))
    context_doc = json.loads(
        (gen_dir / "context.json").read_text(encoding="utf-8"))
    for slot in manifest["slate"]:
        plan_path = plans_dir / f"slot-{slot['slot']}.json"
        if plan_path.exists():
            try:
                existing = json.loads(plan_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing is not None and not _slate_plan_problems(
                    existing, slot, route_arm):
                continue
        route_memory_path = None
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
        extra = {"slot": slot["slot"], "candidate_id": slot["candidate_id"],
                 "gen_dir": str(gen_dir)}
        if route_arm:
            extra["n_route_sketches"] = route_arm
            extra["route_memory"] = str(route_memory_path)
        try:
            receipt, _ = _invoke(
                runner, store, "slate-plan-writer", task, tag, run_dir,
                run_id=slot["run_id"], round_no=round_no, extra=extra,
                inline_payload=_slate_plan_payload(
                    slot, pool_doc, context_doc, route_memory_path))
        except InvocationFailed as exc:
            _or_block(run_dir, repo_root, cmd, events,
                      f"slate-plan-writer failed for slot {slot['slot']}: "
                      f"{exc.problems}")
        problems = _slate_plan_problems(receipt, slot, route_arm)
        if problems:
            _or_block(run_dir, repo_root, cmd, events,
                      f"slate plan for slot {slot['slot']} is invalid: "
                      f"{problems}")
        plan = {key: receipt[key]
                for key in ("slot", "idea", "change", "candidate_name",
                            "route_provenance")
                if key in receipt}
        tmp = plan_path.with_name(plan_path.name + ".tmp")
        tmp.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, plan_path)


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
    """Evaluate the seats in slot order; a slot-0 crash never refills it."""
    for slot in manifest["slate"]:
        if budget_status(run_dir, repo_root, cmd).get("reached"):
            break
        run_id = slot["run_id"]
        _materialize_candidate(task, tag, run_dir, run_id, repo_root, cmd)
        _implement_candidate(runner, store, task, tag, run_dir, run_id,
                             repo_root, cmd, events, job_runner)


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
    _ensure_slate_plans(runner, store, task, tag, run_dir, gen_dir, manifest,
                        round_no, repo_root, cmd, events)
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


def _failure_evidence(candidate_dir: Path) -> str | None:
    report = candidate_dir / "tune_report.json"
    return str(report) if report.exists() else None


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


def _implement_candidate(runner, store, task, tag, run_dir, run_id, repo_root,
                         cmd, events, job_runner=execute_driver_job) -> None:
    """candidate-writer + extractor with evidence-branched escalation."""
    candidate_dir = run_dir / "candidates" / run_id
    # Resolved once per candidate implementation so every extractor retry of
    # this candidate sees the identical donor binding.
    donor_extra = _resolve_donor_extra(run_dir, run_id, repo_root, cmd, events)
    try:
        _, writer_inv = _invoke(runner, store, "candidate-writer", task, tag,
                                run_dir, run_id=run_id,
                                extra={"candidate_dir": str(candidate_dir)})
    except InvocationFailed as exc:
        # invocation failure: no crash evidence exists — retry once, then block
        try:
            _, writer_inv = _invoke(runner, store, "candidate-writer", task,
                                    tag, run_dir, run_id=run_id,
                                    extra={"candidate_dir": str(candidate_dir)})
        except InvocationFailed:
            _or_block(run_dir, repo_root, cmd, events,
                      f"candidate-writer failed for {run_id}: {exc.problems}")
    try:
        _, extractor_inv = _invoke_with_driver_jobs(
            runner, store, "tunable-contract-extractor", task, tag, run_dir,
            run_id=run_id,
            extra=_extractor_extra(
                run_dir, run_id, candidate_dir, **donor_extra
            ),
            repo_root=repo_root, job_runner=job_runner)
        return
    except InvocationFailed as exc:
        problems = exc.problems
        extractor_inv = exc.invocation_id

    # Branch on durable evidence (spec Error handling):
    # 1. budget exhausted + zero attempts → resolve-unevaluated (call+catch)
    if budget_status(run_dir, repo_root, cmd).get("reached") and \
            _resolve_unevaluated(run_dir, run_id, repo_root, cmd):
        return
    # 2. actual failure receipt → crash diagnosis
    evidence = _failure_evidence(candidate_dir)
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
            return
        except InvocationFailed as exc2:
            _or_block(run_dir, repo_root, cmd, events,
                      f"extractor repair failed for {run_id}: {exc2.problems}")
    # 3. no evidence → one fresh retry, then block
    try:
        _invoke_with_driver_jobs(
            runner, store, "tunable-contract-extractor", task, tag,
            run_dir, run_id=run_id,
            extra=_extractor_extra(
                run_dir, run_id, candidate_dir, **donor_extra
            ), repo_root=repo_root,
            job_runner=job_runner)
    except InvocationFailed as exc3:
        _or_block(run_dir, repo_root, cmd, events,
                  f"extractor failed for {run_id}: {exc3.problems}")


# =============================================================================
# Decoupled tuning
# =============================================================================


def _tuner_reconcile(runner, store, task, tag, run_dir, round_no, reason: str,
                     repo_root=REPO_ROOT, job_runner=execute_driver_job):
    receipt, _ = _invoke_with_driver_jobs(
        runner, store, "tuner-orchestrator", task, tag, run_dir,
        round_no=round_no,
        extra={"reconcile_note": reason + _RECONCILE_GUIDANCE},
        repo_root=repo_root, job_runner=job_runner)
    return receipt


def _tune(runner, store, task, tag, run_dir, round_no, repo_root, cmd,
          events, job_runner=execute_driver_job) -> dict:
    """Run the decoupled tuning step; return the effective tuner receipt."""
    tuner_inv = None  # set only on a successful first invocation
    try:
        receipt, tuner_inv = _invoke_with_driver_jobs(
            runner, store, "tuner-orchestrator", task, tag, run_dir,
            round_no=round_no, repo_root=repo_root, job_runner=job_runner)
    except (InvocationFailed, DriverJobError):
        try:
            receipt = _tuner_reconcile(runner, store, task, tag, run_dir,
                                       round_no, "tuner session failed.",
                                       repo_root, job_runner)
        except (InvocationFailed, DriverJobError) as exc:
            problems = getattr(exc, "problems", [str(exc)])
            _or_block(run_dir, repo_root, cmd, events,
                      f"tuner reconciliation failed: {problems}")
    # contradiction: receipt claims applied but the ledger disagrees
    tuned_id = receipt.get("tuned_run_id", "none")
    if receipt.get("tuned") and tuned_id != "none" and \
            not _tune_flag(run_dir, tuned_id):
        note = (f"receipt claims tuned {tuned_id} but ledger has tune: false.")
        # spec: corrective follow-up in the SAME tuner session first
        corrected = None
        try:
            corrected, _ = _invoke_with_driver_jobs(
                runner, store, "tuner-orchestrator", task, tag, run_dir,
                round_no=round_no, resume_from=tuner_inv,
                extra={"reconcile_note": note + _RECONCILE_GUIDANCE},
                repo_root=repo_root, job_runner=job_runner)
        except InvocationFailed:
            corrected = None
        if corrected is None or (corrected.get("tuned") and not _tune_flag(
                run_dir, corrected.get("tuned_run_id", "none"))):
            try:
                corrected = _tuner_reconcile(
                    runner, store, task, tag, run_dir, round_no, note,
                    repo_root, job_runner)
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
                    k_warm, k_eval) -> list[str]:
    extra = []
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
    if k_warm is not None:
        extra += ["--k-warm", str(k_warm)]
    if k_eval is not None:
        extra += ["--k-eval", str(k_eval)]
    return extra


def _setup(runner, store, task, tag, run_dir, task_toml, repo_root, cmd,
           events, max_evaluations, timeout, dimension_strategy,
           llm_intelligence_score, semantic_policy, scheduler_policy,
           inner_policy, k_warm, k_eval, model, cli_path) -> None:
    extra = _init_run_extra(
        dimension_strategy,
        llm_intelligence_score,
        semantic_policy,
        scheduler_policy,
        inner_policy,
        k_warm,
        k_eval,
    )
    common.init_run(task, tag, repo_root, cmd, max_evaluations, timeout,
                    extra=extra)
    cmd(["uv", "--directory", f"tasks/{task}", "sync"], repo_root)
    # the protocol gates this on "required assets absent"; the task
    # contract exposes no deterministic signal for that, so a declared
    # prepare command runs on fresh setup (same choice as the hillclimb
    # port). A non-idempotent prepare command is a task-contract bug.
    try:
        common.run_prepare(task, task_toml, repo_root, cmd)
    except RuntimeError as exc:
        _or_block(run_dir, repo_root, cmd, events, str(exc))
    common.preflight_env(task, run_dir, repo_root, cmd)
    write_metadata(run_dir, model, cli_path)

    # background-researcher runs ONCE, never in the loop. A run dir pre-seeded
    # with a frozen background (background.md + retrieval manifest) skips
    # generation entirely; the same deterministic validators gate it, and a
    # failure blocks rather than letting the researcher rewrite the frozen
    # artifacts.
    strategy = _dimension_strategy(run_dir)
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
        _background_faithfulness_gate(runner, store, task, tag, run_dir,
                                      repo_root, cmd, events, strategy,
                                      repairable=False)
    else:
        try:
            _invoke(runner, store, "background-researcher", task, tag, run_dir)
        except InvocationFailed as exc:
            _or_block(run_dir, repo_root, cmd, events,
                      f"background-researcher failed: {exc.problems}")
        _validate_background(runner, store, task, tag, run_dir, repo_root, cmd,
                             events, strategy)
        _background_faithfulness_gate(runner, store, task, tag, run_dir,
                                      repo_root, cmd, events, strategy,
                                      repairable=True)


def _background_validation_errors(run_dir, repo_root, cmd,
                                  strategy) -> list[str]:
    """Run all deterministic validators for background-research artifacts."""
    checks = []
    if strategy == "llm_induced":
        checks.append([
            "python", "tools/background_contract.py", "catalog",
            "--path", run_dir / "dimension_catalog.json",
        ])
    checks.extend([
        ["python", "tools/search_backends.py", "validate",
         "--manifest", run_dir / "background_retrieval.json"],
        ["python", "tools/background_contract.py", "validate",
         "--background", run_dir / "background.md",
         "--retrieval-manifest", run_dir / "background_retrieval.json"],
    ])

    errors = []
    for check_args in checks:
        result = cmd(check_args, repo_root, check=False)
        if result.returncode != 0:
            errors.append(result.stderr or "validation failed")
    return errors


def _validate_background(runner, store, task, tag, run_dir, repo_root, cmd,
                         events, strategy) -> None:
    """Validate background artifacts and allow one in-session repair."""
    errors = _background_validation_errors(run_dir, repo_root, cmd, strategy)
    if not errors:
        return
    # feed validator errors back in-session once; never migrate the frozen space
    try:
        _invoke(runner, store, "background-researcher", task, tag, run_dir,
                extra={"validation_errors": "\n".join(errors)})
    except InvocationFailed as exc:
        _or_block(run_dir, repo_root, cmd, events,
                  f"background validation repair failed: {exc.problems}")
    errors = _background_validation_errors(run_dir, repo_root, cmd, strategy)
    if errors:
        _or_block(run_dir, repo_root, cmd, events,
                  f"background validation failed: {errors}")


def _background_faithfulness_gate(runner, store, task, tag, run_dir,
                                  repo_root, cmd, events, strategy, *,
                                  repairable: bool) -> None:
    """Synchronous faithfulness audit after validation, before the space freezes.

    The generated path gets exactly one researcher repair round on unfaithful
    findings (followed by a fresh random re-audit); a pre-seeded frozen
    background cannot be rewritten, so unfaithful findings block it directly.
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
            _validate_background(runner, store, task, tag, run_dir, repo_root,
                                 cmd, events, strategy)

    background_audit.run_faithfulness_gate(
        runner, store, task, tag, run_dir, events,
        invoke=_invoke,
        or_block=lambda reason: _or_block(run_dir, repo_root, cmd, events,
                                          reason),
        repair=repair,
    )


def _resume_setup(runner, store, task, tag, run_dir, repo_root, cmd, events,
                  model, cli_path) -> None:
    """Finish any interrupted setup work and restore a runnable phase."""
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        for warning in warn_on_mismatch(run_dir, model, cli_path):
            events.emit("metadata_mismatch", warning=warning)
    else:
        # The run was killed before fresh setup reached write_metadata().
        write_metadata(run_dir, model, cli_path)

    common.preflight_env(task, run_dir, repo_root, cmd)

    # Status consumers must not keep seeing a stale "blocked" phase after an
    # explicit resume has started making progress again.
    ledger_path = run_dir / "ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("run_state", {}).get("phase") == "blocked":
            common.set_phase(run_dir, repo_root, cmd, "running")
            events.emit("resumed_from_blocked")

    background_missing = (
        not (run_dir / "background.md").exists()
        or not (run_dir / "background_retrieval.json").exists()
    )
    if background_missing:
        # The run was killed while the one-time background-research step was
        # in progress. Resume that setup phase before entering ideation.
        try:
            _invoke(runner, store, "background-researcher", task, tag, run_dir)
        except InvocationFailed as exc:
            _or_block(run_dir, repo_root, cmd, events,
                      f"background-researcher failed: {exc.problems}")

    # A resume may follow a kill after the researcher wrote its files but
    # before setup validated them.  Establish the frozen-space invariant once
    # at this process boundary; rounds trust it.
    _validate_background(
        runner, store, task, tag, run_dir, repo_root, cmd, events,
        _dimension_strategy(run_dir),
    )

    # The same kill window covers the faithfulness audit.  Re-run the gate
    # unless a prior audit reached a terminal-ok outcome on these artifacts;
    # a background (re)generated by this resume gets the same one repair
    # round as setup, while pre-existing artifacts are frozen — unfaithful
    # findings block rather than rewrite them.
    if not background_audit.audit_completed(run_dir):
        _background_faithfulness_gate(runner, store, task, tag, run_dir,
                                      repo_root, cmd, events,
                                      _dimension_strategy(run_dir),
                                      repairable=background_missing)


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
            runner, store, "tunable-contract-extractor", task, tag,
            run_dir, run_id=run_id,
            extra={"candidate_dir": str(candidate_dir)}, repo_root=repo_root,
            job_runner=job_runner)
        return
    except InvocationFailed:
        evidence = _failure_evidence(candidate_dir)
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
                except InvocationFailed:
                    pass
        _record_crash(run_dir, run_id, repo_root, cmd)
        _or_block(run_dir, repo_root, cmd, events,
                  "provided baseline could not be evaluated; crash recorded")


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
                         job_runner=execute_driver_job, model=None) -> list[dict]:
    """Generate one bounded action batch and take each action through step 0+1."""
    if _semantic_policy(run_dir) == "judged_slate":
        return _evaluate_judged_generation(runner, store, task, tag, run_dir,
                                           round_no, repo_root, cmd, events,
                                           model, job_runner)
    actions = _ideate(runner, store, task, tag, run_dir, round_no,
                      repo_root, cmd, events)
    for action in actions:
        if budget_status(run_dir, repo_root, cmd).get("reached"):
            break
        run_id = str(action["run_id"])
        _materialize_candidate(task, tag, run_dir, run_id, repo_root, cmd)
        _implement_candidate(runner, store, task, tag, run_dir, run_id,
                             repo_root, cmd, events, job_runner)
    return actions


# =============================================================================
# Public entry point
# =============================================================================


def run_experiment(task, tag, *, runner, model, repo_root=REPO_ROOT,
                   max_evaluations=None, timeout=None, dimension_strategy=None,
                   llm_intelligence_score=None, semantic_policy=None,
                   scheduler_policy=None, inner_policy=None, k_warm=None,
                   k_eval=None,
                   cli_path=None, cmd=common.run_cmd,
                   job_runner=execute_driver_job) -> dict:
    """Set up or resume a run, then advance it until blocked or complete."""
    run_dir = repo_root / "runs" / task / tag
    events = EventsLog(run_dir)
    task_toml = common.load_task_toml(task, repo_root)
    store = ReceiptStore(run_dir)

    try:
        if not (run_dir / "framework_cfg.json").exists():
            _setup(runner, store, task, tag, run_dir, task_toml, repo_root,
                   cmd, events, max_evaluations, timeout, dimension_strategy,
                   llm_intelligence_score, semantic_policy, scheduler_policy,
                   inner_policy, k_warm, k_eval, model, cli_path)
        else:
            # Explicit CLI overrides must never disappear merely because the
            # run directory already exists. init_run applies mutable limits,
            # accepts idempotent frozen values, and rejects policy/space
            # changes once their artifacts exist.
            extra = _init_run_extra(
                dimension_strategy,
                llm_intelligence_score,
                semantic_policy,
                scheduler_policy,
                inner_policy,
                k_warm,
                k_eval,
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
                          events, model, cli_path)

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
                for pending_id in unresolved:
                    _resolve_unevaluated(run_dir, pending_id, repo_root, cmd)
                brief = _brief(run_dir, repo_root, cmd)
                if brief.get("experience_refresh_required"):
                    _refresh(runner, store, task, tag, run_dir, repo_root,
                             cmd, events)
                _complete_run(run_dir, repo_root, cmd, events)
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
            # Round step 2: generate candidates; evaluate each through step 0+1.
            # Full background validation belongs to setup/resume; the
            # high-frequency helpers enforce only the dynamic invariants they
            # consume.
            # -----------------------------------------------------------------
            actions = _evaluate_generation(
                runner, store, task, tag, run_dir, round_no,
                repo_root, cmd, events, job_runner, model=model,
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
                              terminal_leftover=True)
                break
            zero_progress_rounds = 0 if progressed else zero_progress_rounds + 1
            if zero_progress_rounds >= 2:
                events.emit("quiescent", round_no=round_no,
                            reason="two consecutive zero-progress rounds")
                _complete_run(run_dir, repo_root, cmd, events)
                break
            round_no += 1
    except RunBlocked:
        pass

    return compact_status(task, tag, run_dir, repo_root=repo_root, cmd=cmd)
