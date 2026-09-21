"""round_v1 optimization phase: rewrite climbs, then tune bouts.

The experiment loop alternates a generation phase (judged-slate generations
until the round's candidate threshold is met) with this optimization phase
over the whole candidate pool. Every decision lives tool-side
(``tools/scheduler/cli.py round ...``); this module sequences the bouts,
reuses the rewrite loop's bout runner, and commits a kept rewrite through
``tools/rewrite_rebase.py`` and ``tools/ledger.py record-rewrite``.

One REWRITE decision is one *climb* on the selected candidate: repeated
edit -> evaluate -> keep/revert steps (each journaled as its own bout, the
editor session and the adjudication reference chained across steps) until
the candidate stalls, reaches its bout cap, or the phase quota / run
budget runs out. Depth on one candidate is where hillclimbing's kept
chains come from; rotating single edits across candidates never climbs.

GPU work is strictly serial: each bout's evaluations run under the task
resource lease, one at a time. Rewrite climbs, however, may run on several
channels (``pipeline.rewrite_concurrency`` threads) over DIFFERENT
candidates, so one channel's editor session overlaps another's evaluation:
the GPU no longer idles while an LLM thinks. Channels share the round's
climb count, exclude each other's candidate at selection (part of the
decision's identity), and account for each other's in-flight bout when
admitting against the phase quota.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import subprocess
import threading
import time
import os

from ..resources import ResourceUnavailable
from ..session import InvocationFailed
from ..status import budget_status
from . import rewrite
from tools.evaluation_budget import time_remaining

POLICY_ID = "round_v1"

# The selection domain production evaluations land in: every objective path
# publishes its scalar through the legacy adapter as a (proxy, fast) record
# indexed on the ledger. The driver CLI sets EVALUATION_STAGE/FIDELITY to
# this pair unless the operator already did.
DEFAULT_EVALUATION_DOMAIN = ("proxy", "fast")


def _round(run_dir: Path, repo_root: Path, cmd, *args) -> dict:
    proc = cmd(["python", "tools/scheduler/cli.py", "round",
                "--ledger", run_dir / "ledger.json", *args], repo_root)
    return json.loads(proc.stdout)


def _record(run_dir, repo_root, cmd, decision_id, run_id, *, action, consumed,
            status, gain=None) -> None:
    args = ["python", "tools/scheduler/cli.py", "record",
            "--ledger", run_dir / "ledger.json",
            "--decision-id", decision_id, "--action", action,
            "--run-id", run_id, "--consumed", str(consumed),
            "--status", status]
    if gain is not None:
        args += ["--gain", str(gain)]
    cmd(args, repo_root)


def _load_ledger_view(run_dir: Path) -> dict | None:
    """One ledger load with evaluator records materialized; None when the
    view cannot be built (missing ledger or unreadable records)."""
    path = run_dir / "ledger.json"
    if not path.exists():
        return None
    ledger = json.loads(path.read_text(encoding="utf-8"))
    try:
        from tools.scheduler.state import materialize_evaluation_records
        ledger = materialize_evaluation_records(ledger)
    except (ImportError, OSError, ValueError, TypeError):
        return None
    return ledger


def _record_score(record: dict) -> float | None:
    """One record's current best, tuned score first (like the scheduler)."""
    stage = os.environ.get("EVALUATION_STAGE")
    fidelity = os.environ.get("EVALUATION_FIDELITY")
    if stage or fidelity:
        if not stage or not fidelity:
            return None
        try:
            from tools.evaluation_records import best_score
            value = best_score(record.get("evaluation_records", ()),
                               stage=stage, fidelity=fidelity)
        except (ImportError, ValueError):
            return None
        return float(value) if value is not None and math.isfinite(float(value)) else None
    keys = ["best_warm_score", "final_best_score"]
    if record.get("tune"):
        keys.insert(0, "final_best_score")
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and math.isfinite(value):
            return float(value)
    return None


def _ledger_score(run_dir: Path, run_id: str) -> float | None:
    """The candidate's current best, tuned score first (like the scheduler)."""
    ledger = _load_ledger_view(run_dir)
    if ledger is None:
        return None
    for record in ledger.get("records", []):
        if str(record.get("run_id")) != run_id:
            continue
        return _record_score(record)
    return None


def _run_best(run_dir: Path) -> float | None:
    """The best finite score over every candidate in the run (the editor's bar)."""
    ledger = _load_ledger_view(run_dir)
    if ledger is None:
        return None
    scores = []
    for record in ledger.get("records", []):
        score = _record_score(record)
        if score is not None:
            scores.append(score)
    return min(scores) if scores else None


def status(run_dir, repo_root, cmd) -> dict:
    """The generate-or-optimize switch for this loop iteration."""
    stage = os.environ.get("EVALUATION_STAGE")
    fidelity = os.environ.get("EVALUATION_FIDELITY")
    args = ["status"]
    if stage or fidelity:
        if not stage or not fidelity:
            raise RuntimeError("EVALUATION_STAGE and EVALUATION_FIDELITY must be set together")
        args += ["--stage", stage, "--fidelity", fidelity]
    return _round(run_dir, repo_root, cmd, *args)


def _eval_seconds(run_dir, repo_root, cmd, run_id) -> tuple[int, float | None]:
    """(admitted attempts of ``run_id``, its mean evaluation seconds).

    Per-candidate on purpose: a bout's consumption is the delta of ITS
    candidate's attempts, so evaluations another channel runs concurrently
    are never charged to this decision.
    """
    view = budget_status(run_dir, repo_root, cmd)
    row = next((r for r in view.get("per_candidate", [])
                if r.get("run_id") == run_id), {})
    return int(row.get("evals") or 0), row.get("mean_seconds")


def _overhead(run_dir, repo_root, cmd, kind, run_id, started, evals_before,
              lease_wait=0.0):
    """Record the bout's non-evaluation seconds (session + tooling).

    ``lease_wait`` — time this bout queued for the device behind another
    channel's evaluation — is neither this candidate's cost nor this
    channel's overhead, so it is taken out of the wall clock too.
    """
    evals_after, mean_seconds = _eval_seconds(run_dir, repo_root, cmd, run_id)
    wall = time.monotonic() - started - float(lease_wait or 0.0)
    eval_seconds = (evals_after - evals_before) * (mean_seconds or 0.0)
    _round(run_dir, repo_root, cmd, "overhead", "--kind", kind,
           "--seconds", str(max(0.0, wall - eval_seconds)))


def _space_revision(run_dir: Path) -> int | None:
    path = run_dir / "ledger.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8")).get(
            "search_space_state")
    except (OSError, json.JSONDecodeError):
        return None
    revision = (state or {}).get("revision") if isinstance(state, dict) else None
    return revision if isinstance(revision, int) else None


def _rewrite_concurrency(run_dir: Path) -> int:
    """Rewrite channels of one optimization phase (framework_cfg
    ``pipeline.rewrite_concurrency``; absent = 1 = the serial climb loop)."""
    try:
        section = json.loads((run_dir / "framework_cfg.json").read_text(
            encoding="utf-8")).get("pipeline")
    except (OSError, json.JSONDecodeError):
        return 1
    value = section.get("rewrite_concurrency") if isinstance(section, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1
    return value


def _lease_wait_timeout(run_dir: Path, task_toml: dict, concurrency: int) -> float:
    """How long one rewrite step may queue for the device: the task's lease
    wait scaled by the channel count (each other channel may hold the GPU
    for one full evaluation), never past the run's or the phase's clock."""
    resources = (task_toml or {}).get("resources") or {}
    try:
        base = float(resources.get("lease_wait_timeout", 600))
    except (TypeError, ValueError):
        base = 600.0
    limit = base * max(1, int(concurrency))
    left = time_remaining(run_dir)
    if left is not None:
        limit = min(limit, max(1.0, left))
    return limit


class _ClimbCoordinator:
    """Shared state of one phase's rewrite channels.

    ``cond`` doubles as the scheduler lock: every ``round select`` /
    ``round record`` subprocess call and every in_flight / climb-count /
    commitment update happens while holding it, so two channels never
    double-select a candidate and the scheduler store's check-then-append
    is serialized driver-side (``record_outcome`` appends without a lock).
    """

    def __init__(self, climbs: int):
        self.cond = threading.Condition()
        self.remaining = int(climbs)
        self.in_flight: set[str] = set()
        # channel -> (expected_seconds, started_monotonic): the bout a
        # channel admitted and has not closed yet.
        self.commitments: dict[int, tuple[float, float]] = {}
        self.halt = False
        self.progressed = False
        # Priced once per phase, on the first REWRITE selection (the reserve
        # protects the phase's tail for one tune bout; it is not per channel).
        self.tune_reserve: float | None = None

    def other_commitments(self, channel: int) -> float:
        """Remaining priced cost of the other channels' open bouts. Decays
        with the wall clock: elapsed seconds are already gone from the
        phase quota, charging them again would double-count."""
        now = time.monotonic()
        return sum(max(0.0, expected - (now - started))
                   for other, (expected, started) in self.commitments.items()
                   if other != channel)


def _tune_reserve(run_dir: Path, repo_root: Path, cmd,
                  *, enabled: bool) -> float:
    """Return the currently selected tune bout's priced cost.

    Rewrite runs first, so keep one tune bout's admission ticket aside. The
    peek selection is read-only; the actual tune selection is repeated after
    rewrite, when the candidate pool may have changed.
    """
    if not enabled:
        return 0.0
    selection = _round(run_dir, repo_root, cmd, "select", "--kind", "tune", "--peek")
    if selection.get("action") != "TUNE":
        return 0.0
    mode = selection.get("evidence_mode") or {}
    ranked = mode.get("ranked") or []
    run_id = str(selection.get("run_id"))
    row = next((row for row in ranked if str(row.get("run_id")) == run_id), None)
    expected = row.get("expected_seconds") if row else None
    try:
        return max(0.0, float(expected)) if expected is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _rewrite_climb(runner, store, task, tag, run_dir, selection, task_toml,
                   config, repo_root, cmd, events, *, tune_reserve=0.0,
                   coord: _ClimbCoordinator | None = None, channel: int = 0,
                   exclude_run_ids=(), concurrency: int = 1) -> dict:
    """One scheduler REWRITE decision: hillclimb the selected candidate.

    Repeated edit -> evaluate -> keep/revert steps, each journaled as its
    own bout and each kept step committed, until the candidate stalls
    (``rewrite_stall_after`` consecutive non-kept steps), reaches its
    ``rewrite_max_bouts`` cap, or the phase quota / run budget runs out.
    The adjudication reference chains across steps, so every step is
    measured against the candidate's current best. The decision is
    recorded once, with the climb's total attempts and reference gain.

    Under concurrency, ``coord`` carries the other channels' open bouts:
    each step is admitted against ``quota − tune_reserve − Σ their remaining
    commitments`` and registers its own; a step squeezed out only by those
    commitments waits for them to close or decay instead of ending the
    climb. ``exclude_run_ids`` and the per-step reference snapshot are
    journaled on the climb event so stale-state decisions can be told apart
    afterwards.
    """
    run_id = str(selection["run_id"])
    lock = coord.cond if coord is not None else nullcontext()
    candidate = run_dir / "candidates" / run_id
    metric = task_toml["result"]["metric"]
    noise_margin = float(config["noise_margin"])
    max_bouts = int(config["rewrite_max_bouts"])
    stall_after = int(config["rewrite_stall_after"])
    overhead = float((selection.get("evidence_mode") or {}).get(
        "overhead_seconds") or config["session_overhead_seconds"])
    initial_reference = selection.get("reference")
    if initial_reference is None:
        initial_reference = _ledger_score(run_dir, run_id)
    reference = initial_reference
    steps = 0
    kept = 0
    attempts = 0
    bouts = rewrite._load_bouts(candidate)
    # Seed the stall streak from the journal so a climb resumed after a
    # kill does not re-spend the non-kept steps it already paid for.
    streak = rewrite._consecutive_non_kept(bouts)
    status = "done"
    stop = None
    snapshots = []
    while stop is None:
        if len(bouts) >= max_bouts:
            stop = "bout_cap"
            break
        if streak >= stall_after:
            stop = "stalled"
            break
        admitted = _admit_step(run_dir, run_id, repo_root, cmd, overhead,
                               tune_reserve, coord, channel)
        if admitted != "ok":
            status, stop = admitted
            break
        run_best = _run_best(run_dir)
        snapshots.append({"step": steps + 1, "run_best": run_best,
                          "space_revision": _space_revision(run_dir)})
        started = time.monotonic()
        evals_before, _ = _eval_seconds(run_dir, repo_root, cmd, run_id)
        try:
            result = rewrite._run_bout(
                task, tag, run_dir, candidate, bouts, runner, store, metric,
                noise_margin, "full", task_toml, repo_root, cmd, events,
                reference=reference, confirm=True, run_best=run_best,
                lease_wait_timeout=_lease_wait_timeout(
                    run_dir, task_toml, concurrency))
        except InvocationFailed as exc:
            # The failed session's edit is already rolled back; one
            # candidate's dead editor session does not stop the run.
            events.emit("rewrite_editor_failed", run_id=run_id,
                        problems=[str(p) for p in exc.problems])
            status, stop = "failed", "editor_failed"
            break
        except ResourceUnavailable as exc:
            # The device never came free within this step's bounded wait
            # (another channel's evaluations, or the clock). The unjudged
            # edit is already reverted; only this climb ends.
            events.emit("resource_unavailable", candidate=candidate.name,
                        stage="rewrite_step", reason=str(exc))
            status, stop = "failed", "lease_timeout"
            break
        finally:
            if coord is not None:
                with coord.cond:
                    coord.commitments.pop(channel, None)
                    coord.cond.notify_all()
        _overhead(run_dir, repo_root, cmd, "rewrite", run_id, started,
                  evals_before, lease_wait=result.get("lease_wait_seconds", 0.0))
        steps += 1
        attempts += result["attempts"]
        if result["status"] == "budget":
            status, stop = "budget", "run_budget"
            break
        if result["status"] == "no_reference":
            stop = "no_reference"
            break
        if result["outcome"] == "kept":
            kept += 1
            streak = 0
            _commit_kept(run_dir, run_id, candidate, len(bouts) + 1, result,
                         repo_root, cmd, events)
        else:
            streak += 1
        if result["reference"] is not None:
            reference = result["reference"]
        bouts = rewrite._load_bouts(candidate)
    with lock:
        _record(run_dir, repo_root, cmd, selection["decision_id"], run_id,
                action="REWRITE",
                consumed=attempts,
                status="valid" if steps > 0 else "infra_failure",
                gain=(None if kept == 0 or initial_reference is None
                      or reference is None else initial_reference - reference))
    events.emit("rewrite_climb", run_id=run_id, steps=steps, kept=kept,
                attempts=attempts, stop=stop, reference=reference,
                channel=channel, exclude_run_ids=sorted(exclude_run_ids),
                reference_snapshot=snapshots)
    return {"status": status, "steps": steps, "kept": kept,
            "attempts": attempts, "stop": stop, "reference": reference}


def _admit_step(run_dir, run_id, repo_root, cmd, overhead, tune_reserve,
                coord, channel):
    """Admit one climb step against the phase quota.

    Returns "ok" (the step's expected cost is registered as this channel's
    commitment) or a ``(status, stop)`` pair ending the climb. The quota
    check is ``quota − tune_reserve``; under concurrency the other channels'
    decaying commitments are subtracted too, and a step that only THEY
    squeeze out waits for a bout to close or the commitments to decay,
    rather than ending the climb — the reservation refusal stays the
    backstop.
    """
    while True:
        view = budget_status(run_dir, repo_root, cmd)
        if view.get("reached"):
            return "budget", "run_budget"
        quota = view.get("phase_quota_remaining_seconds")
        _, mean = _eval_seconds(run_dir, repo_root, cmd, run_id)
        expected = None if mean is None else 2.0 * float(mean) + overhead
        base = None if quota is None else max(0.0, quota - tune_reserve)
        if base is not None and (
                base <= 0 or (expected is not None and expected > base)):
            return "done", "round_quota"
        if coord is None:
            return "ok"
        with coord.cond:
            others = coord.other_commitments(channel)
            if base is not None and others > 0 and (
                    base - others <= 0
                    or (expected is not None and expected > base - others)):
                coord.cond.wait(timeout=30.0)
                continue
            if expected is not None:
                coord.commitments[channel] = (expected, time.monotonic())
        return "ok"


def _commit_kept(run_dir, run_id, candidate, bout, result, repo_root, cmd,
                 events) -> None:
    """Rebase the tuning report onto the new code, then commit the score."""
    reference = result["reference"]
    report_args = []
    rebase = cmd(["python", "tools/rewrite_rebase.py",
                  "--candidate", candidate, "--bout", str(bout),
                  "--score", str(reference),
                  "--attempts", str(max(1, result["attempts"]))],
                 repo_root, check=False)
    if rebase.returncode == 0:
        report_args = ["--tune-report", candidate / "tune_report.json"]
    else:
        # The code improvement stands; the candidate merely stays untunable
        # until a later rebase succeeds (round_policy checks report currency).
        events.emit("rewrite_rebase_failed", run_id=run_id, bout=bout,
                    detail=(rebase.stderr or rebase.stdout or "")[-2000:])
    # A refused record-rewrite would fork the journal (kept bout) from the
    # ledger silently. One immediate retry absorbs a transient refusal; a
    # repeated refusal is a tools-side failure and escalates (fail closed).
    for attempt in range(2):
        try:
            cmd(["python", "tools/ledger.py", "record-rewrite",
                 "--ledger", run_dir / "ledger.json", "--run-id", run_id,
                 "--score", str(reference), *report_args], repo_root)
            break
        except subprocess.CalledProcessError as exc:
            events.emit("rewrite_ledger_failed", run_id=run_id, bout=bout,
                        attempt=attempt + 1,
                        detail=(exc.stderr or str(exc))[-2000:])
            if attempt == 1:
                raise
    events.emit("rewrite_kept", run_id=run_id, bout=bout,
                score=result["score"], reference=reference)


def tune_handoff(selection: dict) -> dict:
    """The compact exact-target contract the tuner session must obey.

    Budget is the contract bout cost (evaluation count); round_v1's mid-bout
    truncation semantics (reservation refusal + `close_exhausted_stage`)
    are unchanged by the handoff.
    """
    keys = ("decision_id", "run_id", "policy_version", "state_snapshot_id",
            "evidence_cursor", "bout_trials")
    return {key: selection.get(key) for key in keys}


# In-process tune backoff: consecutive zero-progress/infra failures per
# candidate. `select_tune` sees none of them (score, rank, expected seconds
# and report currency are all unchanged by a consumed=0 failure), so without
# this the same candidate is re-selected every round. A driver restart
# forgets the streaks; the first re-failure re-excludes.
TUNE_BACKOFF_LIMIT = 2
_tune_backoff: dict[str, dict] = {}


def _backoff_state(run_dir) -> dict:
    return _tune_backoff.setdefault(
        str(run_dir), {"streak": {}, "excluded": set(), "succeeded": 0})


def _note_tune_outcome(run_dir, events, run_id: str, status: str) -> None:
    state = _backoff_state(run_dir)
    if status == "valid":
        state["succeeded"] += 1
        state["streak"][run_id] = 0
        return
    streak = state["streak"].get(run_id, 0) + 1
    state["streak"][run_id] = streak
    if streak >= TUNE_BACKOFF_LIMIT and run_id not in state["excluded"]:
        state["excluded"].add(run_id)
        events.emit("tune_candidate_excluded", run_id=run_id,
                    consecutive_infra_failures=streak)


def tune_excluded(run_dir) -> list[str]:
    """Candidates backed off from this run's tune selection (this process)."""
    return sorted(_backoff_state(run_dir)["excluded"])


def reset_tune_backoff(run_dir) -> None:
    """Fresh entry into a run (run_experiment start) starts with no streaks."""
    _tune_backoff.pop(str(run_dir), None)


def tune_backoff_no_success(run_dir) -> bool:
    """True when backoff excluded candidates and no tune bout ever succeeded:
    a completion in that state must not look like a normal close."""
    state = _tune_backoff.get(str(run_dir))
    return bool(state and state["excluded"] and state["succeeded"] == 0)


def close_tune_outcome(run_dir, repo_root, cmd, events, selection, receipt,
                       *, evals_before, reference_before) -> None:
    """Bind the tune bout's result to the decision that selected it.

    Every terminal path of a TUNE selection ends here: a kept improvement, a
    valid no-op, a zero-progress/infra failure (`outcome_status` on the
    receipt), or a receipt whose `tuned_run_id` names another candidate.
    The bout's objective jobs are pinned to the selected candidate, so a
    mismatching id can only be a stale or fabricated handoff — the result is
    never re-attributed to the candidate the receipt names.
    """
    run_id = str(selection["run_id"])
    executed = str(receipt.get("tuned_run_id", "none"))
    if executed not in (run_id, "none"):
        events.emit("tune_target_mismatch",
                    decision_id=selection.get("decision_id"),
                    selected_run_id=run_id, executed_run_id=executed)
        _record(run_dir, repo_root, cmd, selection["decision_id"], run_id,
                action="TUNE", consumed=0, status="infra_failure")
        _note_tune_outcome(run_dir, events, run_id, "infra_failure")
        return
    evals_after, _ = _eval_seconds(run_dir, repo_root, cmd, run_id)
    consumed = max(0, evals_after - evals_before)
    if receipt.get("outcome_status") == "infra_failure":
        _record(run_dir, repo_root, cmd, selection["decision_id"], run_id,
                action="TUNE", consumed=consumed, status="infra_failure")
        events.emit("tune_infra_failure",
                    decision_id=selection.get("decision_id"),
                    selected_run_id=run_id, executed_run_id=executed,
                    consumed=consumed,
                    reason=receipt.get("outcome_reason"),
                    detail=receipt.get("outcome_detail"),
                    snapshot=receipt.get("snapshot_counts"))
        _note_tune_outcome(run_dir, events, run_id, "infra_failure")
        return
    reference_after = _ledger_score(run_dir, run_id)
    gain = (None if reference_before is None or reference_after is None
            else reference_before - reference_after)
    _record(run_dir, repo_root, cmd, selection["decision_id"], run_id,
            action="TUNE", consumed=consumed, status="valid", gain=gain)
    events.emit("tune_bout", decision_id=selection.get("decision_id"),
                selected_run_id=run_id, executed_run_id=executed,
                tuned=bool(receipt.get("tuned")),
                consumed_evaluations=consumed, realized_gain=gain)
    _note_tune_outcome(run_dir, events, run_id, "valid")


def _rewrite_channel(channel, coord, runner, store, task, tag, run_dir,
                     task_toml, config, repo_root, cmd, events, *,
                     concurrency) -> None:
    """One rewrite channel: select → climb, until the shared climb count
    is spent, the run budget lands, or nothing is selectable.

    Selection excludes the candidates other channels are climbing. A STOP
    under a non-empty exclude set does not end the channel — the pool may
    change when the other climb closes (a kept commit, a freed candidate)
    — so it waits for that and reselects; a STOP with nothing excluded is
    the phase's own verdict and ends the channel.
    """
    while True:
        with coord.cond:
            while True:
                if coord.halt or coord.remaining <= 0:
                    return
                if budget_status(run_dir, repo_root, cmd).get("reached"):
                    coord.halt = True
                    coord.cond.notify_all()
                    return
                exclude = sorted(coord.in_flight)
                args = ["select", "--kind", "rewrite"]
                if exclude:
                    args += ["--exclude", ",".join(exclude)]
                selection = _round(run_dir, repo_root, cmd, *args)
                events.emit("round_select", bout_kind="rewrite",
                            action=selection["action"],
                            run_id=selection.get("run_id"),
                            reason=selection.get("reason"),
                            channel=channel, exclude_run_ids=exclude)
                if selection["action"] == "REWRITE":
                    run_id = str(selection["run_id"])
                    coord.remaining -= 1
                    coord.in_flight.add(run_id)
                    if coord.tune_reserve is None:
                        coord.tune_reserve = _tune_reserve(
                            run_dir, repo_root, cmd,
                            enabled=int(config["tune_bouts"]) > 0)
                    tune_reserve = coord.tune_reserve
                    break
                if not exclude:
                    return
                coord.cond.wait()
        try:
            result = _rewrite_climb(
                runner, store, task, tag, run_dir, selection, task_toml,
                config, repo_root, cmd, events, tune_reserve=tune_reserve,
                coord=coord, channel=channel, exclude_run_ids=exclude,
                concurrency=concurrency)
        finally:
            with coord.cond:
                coord.in_flight.discard(run_id)
                coord.commitments.pop(channel, None)
                coord.cond.notify_all()
        with coord.cond:
            coord.progressed = coord.progressed or result["steps"] > 0
            if result["status"] == "budget":
                coord.halt = True
                coord.cond.notify_all()


def _rewrite_phase(runner, store, task, tag, run_dir, task_toml, config,
                   repo_root, cmd, events) -> bool:
    """The rewrite sub-phase: ``rewrite_bouts`` climbs shared by
    ``rewrite_concurrency`` channels; True when any step ran."""
    climbs = int(config["rewrite_bouts"])
    if climbs <= 0:
        return False
    concurrency = max(1, min(_rewrite_concurrency(run_dir), climbs))
    coord = _ClimbCoordinator(climbs)
    if concurrency == 1:
        _rewrite_channel(0, coord, runner, store, task, tag, run_dir,
                         task_toml, config, repo_root, cmd, events,
                         concurrency=1)
        return coord.progressed
    events.emit("rewrite_channels", concurrency=concurrency, climbs=climbs)
    threads = []
    errors: list[BaseException] = []

    def worker(channel):
        try:
            _rewrite_channel(channel, coord, runner, store, task, tag, run_dir,
                             task_toml, config, repo_root, cmd, events,
                             concurrency=concurrency)
        except BaseException as exc:  # noqa: BLE001 - re-raised after join
            with coord.cond:
                errors.append(exc)
                coord.halt = True  # other channels stop at their boundary
                coord.cond.notify_all()

    for channel in range(concurrency):
        thread = threading.Thread(target=worker, args=(channel,),
                                  name=f"rewrite-{channel}")
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return coord.progressed


def optimization_phase(runner, store, task, tag, run_dir, round_no, task_toml,
                       repo_root, cmd, events, *, tune, config) -> bool:
    """Run one optimization round; True when any bout ran."""
    begun = _round(run_dir, repo_root, cmd, "begin")
    events.emit("round_begin", round_no=round_no, cycle=begun.get("cycle"),
                phase_deadline=begun.get("phase_deadline"))
    progressed = False
    try:
        progressed = _rewrite_phase(runner, store, task, tag, run_dir,
                                    task_toml, config, repo_root, cmd, events)
        for _ in range(int(config["tune_bouts"])):
            if budget_status(run_dir, repo_root, cmd).get("reached"):
                break
            excluded = tune_excluded(run_dir)
            select_args = ["select", "--kind", "tune"]
            if excluded:
                select_args += ["--exclude", ",".join(excluded)]
            selection = _round(run_dir, repo_root, cmd, *select_args)
            events.emit("round_select", bout_kind="tune",
                        action=selection["action"],
                        run_id=selection.get("run_id"),
                        decision_id=selection.get("decision_id"),
                        reason=selection.get("reason"),
                        exclude_run_ids=excluded)
            if selection["action"] != "TUNE":
                if excluded:
                    events.emit("tune_backoff_stop",
                                exclude_run_ids=excluded,
                                reason=selection.get("reason"))
                break
            run_id = str(selection["run_id"])
            reference_before = _ledger_score(run_dir, run_id)
            started = time.monotonic()
            evals_before, _ = _eval_seconds(run_dir, repo_root, cmd, run_id)
            receipt = tune(round_no, selection)
            progressed = progressed or bool(receipt.get("tuned"))
            _overhead(run_dir, repo_root, cmd, "tune", run_id, started,
                      evals_before)
            close_tune_outcome(run_dir, repo_root, cmd, events, selection,
                               receipt, evals_before=evals_before,
                               reference_before=reference_before)
    finally:
        args = ["end"]
        stage = os.environ.get("EVALUATION_STAGE")
        fidelity = os.environ.get("EVALUATION_FIDELITY")
        if stage or fidelity:
            args += ["--stage", stage, "--fidelity", fidelity]
        ended = _round(run_dir, repo_root, cmd, *args)
        events.emit("round_end", round_no=round_no, cycle=ended.get("cycle"),
                    cycle_start_count=ended.get("cycle_start_count"))
    return progressed
