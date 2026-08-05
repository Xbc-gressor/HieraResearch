"""Deterministic port of the autoresearch-experiment protocol.

Python owns sequencing, budget/lifecycle checks, and escalation. tools/
helpers own deterministic decisions; LLM sessions own generation/judgment.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..events import EventsLog
from ..metadata import warn_on_mismatch, write_metadata
from ..receipts import ReceiptStore
from ..roles import REPO_ROOT, ROLES, InvocationContext
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


def _invoke(runner, store, role_name, task, tag, run_dir, *,
            run_id=None, round_no=None, extra=None, resume_from=None) -> dict:
    inv_id = store.next_invocation_id()
    resume = (store.load_session_id(role_name, resume_from)
              if resume_from is not None else None)
    ctx = InvocationContext(task=task, tag=tag, run_dir=run_dir,
                            invocation_id=inv_id, run_id=run_id,
                            round_no=round_no, extra=extra or {},
                            resume_session_id=resume)
    runner.run(ROLES[role_name], ctx)
    receipt_path = store.receipt_path(role_name, inv_id)
    return json.loads(receipt_path.read_text(encoding="utf-8")), inv_id


# --- escalations -------------------------------------------------------------


def _or_block(run_dir, repo_root, cmd, events, reason: str):
    common.block(run_dir, repo_root, cmd, events, reason)
    raise RunBlocked(reason)


def _refresh(runner, store, task, tag, run_dir, repo_root, cmd, events) -> None:
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


def _implement_candidate(runner, store, task, tag, run_dir, run_id, repo_root,
                         cmd, events) -> None:
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
        _, extractor_inv = _invoke(
            runner, store, "tunable-contract-extractor", task, tag, run_dir,
            run_id=run_id, extra={"candidate_dir": str(candidate_dir)})
        return
    except InvocationFailed as exc:
        problems = exc.problems

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
            _invoke(runner, store, "tunable-contract-extractor", task, tag,
                    run_dir, run_id=run_id,
                    extra={"candidate_dir": str(candidate_dir),
                           "diagnosis_verdict": verdict},
                    resume_from=extractor_inv)
            return
        except InvocationFailed as exc2:
            _or_block(run_dir, repo_root, cmd, events,
                      f"extractor repair failed for {run_id}: {exc2.problems}")
    # 3. no evidence → one fresh retry, then block
    try:
        _invoke(runner, store, "tunable-contract-extractor", task, tag,
                run_dir, run_id=run_id,
                extra={"candidate_dir": str(candidate_dir)})
    except InvocationFailed as exc3:
        _or_block(run_dir, repo_root, cmd, events,
                  f"extractor failed for {run_id}: {exc3.problems}")


def _tuner_reconcile(runner, store, task, tag, run_dir, round_no, reason: str):
    receipt, _ = _invoke(
        runner, store, "tuner-orchestrator", task, tag, run_dir,
        round_no=round_no,
        extra={"reconcile_note": reason + _RECONCILE_GUIDANCE})
    return receipt


def _tune(runner, store, task, tag, run_dir, round_no, repo_root, cmd,
          events) -> None:
    try:
        receipt, tuner_inv = _invoke(runner, store, "tuner-orchestrator",
                                     task, tag, run_dir, round_no=round_no)
    except InvocationFailed:
        try:
            receipt = _tuner_reconcile(runner, store, task, tag, run_dir,
                                       round_no, "tuner session failed.")
        except InvocationFailed as exc:
            _or_block(run_dir, repo_root, cmd, events,
                      f"tuner reconciliation failed: {exc.problems}")
    # contradiction: receipt claims applied but the ledger disagrees
    tuned_id = receipt.get("tuned_run_id", "none")
    if receipt.get("tuned") and tuned_id != "none" and \
            not _tune_flag(run_dir, tuned_id):
        note = (f"receipt claims tuned {tuned_id} but ledger has tune: false.")
        # spec: corrective follow-up in the SAME tuner session first
        corrected = None
        try:
            corrected, _ = _invoke(
                runner, store, "tuner-orchestrator", task, tag, run_dir,
                round_no=round_no, resume_from=tuner_inv,
                extra={"reconcile_note": note + _RECONCILE_GUIDANCE})
        except InvocationFailed:
            corrected = None
        if corrected is None or (corrected.get("tuned") and not _tune_flag(
                run_dir, corrected.get("tuned_run_id", "none"))):
            try:
                corrected = _tuner_reconcile(
                    runner, store, task, tag, run_dir, round_no, note)
            except InvocationFailed as exc:
                _or_block(run_dir, repo_root, cmd, events,
                          f"tuner receipt/ledger contradiction unresolved: "
                          f"{exc.problems}")
        if corrected.get("tuned") and \
                not _tune_flag(run_dir, corrected.get("tuned_run_id", "none")):
            _or_block(run_dir, repo_root, cmd, events,
                      "authoritative artifacts still contradict after "
                      "tuner reconciliation")


# --- setup ---------------------------------------------------------------------


def _setup(runner, store, task, tag, run_dir, task_toml, repo_root, cmd,
           events, max_evaluations, timeout, dimension_strategy,
           llm_intelligence_score, model, cli_path) -> None:
    extra = []
    if dimension_strategy:
        extra += ["--dimension-strategy", dimension_strategy]
    if llm_intelligence_score is not None:
        extra += ["--llm-intelligence-score", str(llm_intelligence_score)]
    common.init_run(task, tag, repo_root, cmd, max_evaluations, timeout,
                    extra=extra)
    cmd(["uv", "--directory", f"tasks/{task}", "sync"], repo_root)
    prepare = task_toml.get("run", {}).get("prepare_command")
    if prepare:
        # the protocol gates this on "required assets absent"; the task
        # contract exposes no deterministic signal for that, so a declared
        # prepare command runs on fresh setup (same choice as the hillclimb
        # port). A non-idempotent prepare command is a task-contract bug.
        cmd(prepare.split(), repo_root)
    common.preflight_env(task, run_dir, repo_root, cmd)
    write_metadata(run_dir, model, cli_path)

    # background-researcher runs ONCE, never in the loop
    strategy = json.loads((run_dir / "framework_cfg.json")
                          .read_text(encoding="utf-8")).get("dimension_strategy")
    try:
        _invoke(runner, store, "background-researcher", task, tag, run_dir)
    except InvocationFailed as exc:
        _or_block(run_dir, repo_root, cmd, events,
                  f"background-researcher failed: {exc.problems}")
    _validate_background(runner, store, task, tag, run_dir, repo_root, cmd,
                         events, strategy)

    seed = task_toml.get("seed", {})
    if seed.get("provided"):
        _provided_baseline(runner, store, task, tag, run_dir, repo_root, cmd,
                           events, seed)


def _validate_background(runner, store, task, tag, run_dir, repo_root, cmd,
                         events, strategy) -> None:
    def validators():
        checks = []
        if strategy == "llm_induced":
            checks.append(["python", "tools/background_contract.py", "catalog",
                           "--path", run_dir / "dimension_catalog.json"])
        checks.append(["python", "tools/search_backends.py", "validate",
                       "--manifest", run_dir / "background_retrieval.json"])
        checks.append(["python", "tools/background_contract.py", "validate",
                       "--background", run_dir / "background.md",
                       "--retrieval-manifest",
                       run_dir / "background_retrieval.json"])
        errors = []
        for check_args in checks:
            result = cmd(check_args, repo_root, check=False)
            if result.returncode != 0:
                errors.append(result.stderr or "validation failed")
        return errors

    errors = validators()
    if not errors:
        return
    # feed validator errors back in-session once; never migrate the frozen space
    try:
        _invoke(runner, store, "background-researcher", task, tag, run_dir,
                extra={"validation_errors": "\n".join(errors)})
    except InvocationFailed as exc:
        _or_block(run_dir, repo_root, cmd, events,
                  f"background validation repair failed: {exc.problems}")
    errors = validators()
    if errors:
        _or_block(run_dir, repo_root, cmd, events,
                  f"background validation failed: {errors}")


def _provided_baseline(runner, store, task, tag, run_dir, repo_root, cmd,
                       events, seed) -> None:
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
                                        "000", repo_root, cmd, events)


def _implement_candidate_extractor_only(runner, store, task, tag, run_dir,
                                        run_id, repo_root, cmd, events) -> None:
    """Provided baseline: extractor step 0+1 only; a crash BLOCKS the run
    (the protocol forbids searching on without the control)."""
    candidate_dir = run_dir / "candidates" / run_id
    try:
        _invoke(runner, store, "tunable-contract-extractor", task, tag,
                run_dir, run_id=run_id,
                extra={"candidate_dir": str(candidate_dir)})
        return
    except InvocationFailed:
        evidence = _failure_evidence(candidate_dir)
        if evidence:
            verdict = common.crash_diagnose(
                runner, store, task, tag, run_dir, evidence)["verdict"]
            if verdict != "abandon":
                try:
                    _invoke(runner, store, "tunable-contract-extractor", task,
                            tag, run_dir, run_id=run_id,
                            extra={"candidate_dir": str(candidate_dir),
                                   "diagnosis_verdict": verdict})
                    return
                except InvocationFailed:
                    pass
        _record_crash(run_dir, run_id, repo_root, cmd)
        _or_block(run_dir, repo_root, cmd, events,
                  "provided baseline could not be evaluated; crash recorded")


# --- main loop -------------------------------------------------------------------


def run_experiment(task, tag, *, runner, model, repo_root=REPO_ROOT,
                   max_evaluations=None, timeout=None, dimension_strategy=None,
                   llm_intelligence_score=None, cli_path=None,
                   cmd=common.run_cmd) -> dict:
    run_dir = repo_root / "runs" / task / tag
    events = EventsLog(run_dir)
    task_toml = common.load_task_toml(task, repo_root)
    store = ReceiptStore(run_dir)

    try:
        if not (run_dir / "framework_cfg.json").exists():
            _setup(runner, store, task, tag, run_dir, task_toml, repo_root,
                   cmd, events, max_evaluations, timeout, dimension_strategy,
                   llm_intelligence_score, model, cli_path)
        else:  # resume: re-run the env gate; warn on metadata drift
            for warning in warn_on_mismatch(run_dir, model, cli_path):
                events.emit("metadata_mismatch", warning=warning)
            common.preflight_env(task, run_dir, repo_root, cmd)

        round_no = 0
        while True:
            # step 0: lifecycle + budget (only when a ledger exists — seedless
            # first rounds skip straight to ideation)
            ledger_exists = (run_dir / "ledger.json").exists()
            brief = _brief(run_dir, repo_root, cmd) if ledger_exists else None
            if brief is not None and budget_status(run_dir, repo_root, cmd).get("reached"):
                if brief.get("experience_refresh_required"):
                    _refresh(runner, store, task, tag, run_dir, repo_root,
                             cmd, events)
                common.set_phase(run_dir, repo_root, cmd, "completed")
                break

            # step 1: bounded belief refresh at a refresh boundary
            if brief is not None and brief.get("experience_refresh_required"):
                _refresh(runner, store, task, tag, run_dir, repo_root, cmd,
                         events)

            # step 2: generate + evaluate candidates
            result = cmd(["python", "tools/background_contract.py", "preflight",
                          "--background", run_dir / "background.md"] +
                         (["--ledger", run_dir / "ledger.json"]
                          if ledger_exists else []),
                         repo_root, check=False)
            action = ""
            try:
                action = json.loads(result.stdout).get("action", "")
            except (json.JSONDecodeError, AttributeError):
                pass
            if result.returncode != 0 or action != "none":
                _or_block(run_dir, repo_root, cmd, events,
                          "background preflight rejected the frozen space")

            actions = _ideate(runner, store, task, tag, run_dir, round_no,
                              repo_root, cmd, events)
            for action_item in actions:
                if budget_status(run_dir, repo_root, cmd).get("reached"):
                    break  # mid-round budget stop; step 0 finalizes
                run_id = str(action_item["run_id"])
                cmd(["python", "tools/new_candidate.py", task, tag, run_id,
                     "--skip-entrypoint"], repo_root)
                _implement_candidate(runner, store, task, tag, run_dir,
                                     run_id, repo_root, cmd, events)

            # step 3: decoupled tuning (skipped when the budget ran out)
            if not budget_status(run_dir, repo_root, cmd).get("reached"):
                _tune(runner, store, task, tag, run_dir, round_no, repo_root,
                      cmd, events)
            round_no += 1
    except RunBlocked:
        pass

    return compact_status(task, tag, run_dir, repo_root=repo_root, cmd=cmd)
