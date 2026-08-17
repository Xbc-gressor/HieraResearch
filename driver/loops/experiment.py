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
from . import common
from .common import RunBlocked

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
            run_id=None, round_no=None, extra=None, resume_from=None) -> dict:
    """Invoke one role and return its persisted receipt plus invocation id."""
    inv_id = store.next_invocation_id()
    resume = (store.load_session_id(role_name, resume_from)
              if resume_from is not None else None)
    ctx = InvocationContext(task=task, tag=tag, run_dir=run_dir,
                            invocation_id=inv_id, run_id=run_id,
                            round_no=round_no, extra=extra or {},
                            resume_session_id=resume)
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


def _complete_run(run_dir, repo_root, cmd, events) -> None:
    """Persist normal completion, translating a refusal into a blocked run."""
    try:
        common.set_phase(run_dir, repo_root, cmd, "completed")
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


def _implement_candidate(runner, store, task, tag, run_dir, run_id, repo_root,
                         cmd, events, job_runner=execute_driver_job) -> None:
    """candidate-writer + extractor with evidence-branched escalation."""
    candidate_dir = run_dir / "candidates" / run_id
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
            run_id=run_id, extra={"candidate_dir": str(candidate_dir)},
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
        verdict = common.crash_diagnose(
            runner, store, task, tag, run_dir, evidence)["verdict"]
        if verdict == "abandon":
            _record_crash(run_dir, run_id, repo_root, cmd)
            return
        try:  # fix verdicts go back to the repair-capable extractor session
            _invoke_with_driver_jobs(
                runner, store, "tunable-contract-extractor", task, tag,
                run_dir, run_id=run_id,
                extra={"candidate_dir": str(candidate_dir),
                       "diagnosis_verdict": verdict},
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
            extra={"candidate_dir": str(candidate_dir)}, repo_root=repo_root,
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
    else:
        try:
            _invoke(runner, store, "background-researcher", task, tag, run_dir)
        except InvocationFailed as exc:
            _or_block(run_dir, repo_root, cmd, events,
                      f"background-researcher failed: {exc.problems}")
        _validate_background(runner, store, task, tag, run_dir, repo_root, cmd,
                             events, strategy)


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
            verdict = common.crash_diagnose(
                runner, store, task, tag, run_dir, evidence)["verdict"]
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
                         job_runner=execute_driver_job) -> list[dict]:
    """Generate one bounded action batch and take each action through step 0+1."""
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
                repo_root, cmd, events, job_runner,
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
                    _complete_run(run_dir, repo_root, cmd, events)
                    break

            # -----------------------------------------------------------------
            # Completion guard: stop if no operation can spend the budget.
            # -----------------------------------------------------------------
            # Quiescence guard: got_select caps actions by
            # floor(remaining_slots / max(2, K_eval)), so a nearly-exhausted
            # budget yields empty ideation rounds forever. Two consecutive
            # rounds with no new candidates, no tuning, no refresh, and no
            # pending resolutions mean no progress is possible within the
            # remaining budget — complete normally instead of spinning.
            progressed = (bool(actions) or tuner_progressed or refreshed
                          or bool(pending_ids))
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
