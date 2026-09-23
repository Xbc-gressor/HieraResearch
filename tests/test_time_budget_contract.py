"""Session time budget (B) and the final-deadline contract (D).

One cutoff, deadline − final_reserve, bounds everything: no session starts
past it, a running one is cancelled at it, driver jobs refuse/terminate at
it, and pending candidates settle deterministically from on-disk evidence.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from driver import jobs  # noqa: E402
from driver.events import EventsLog  # noqa: E402
from driver.loops import experiment  # noqa: E402
from driver.receipts import ReceiptStore  # noqa: E402
from driver.roles import InvocationContext, RoleDefinition  # noqa: E402
from driver.session import (  # noqa: E402
    SOFT_RESCUE_MESSAGE,
    AuthTokenPool,
    FakeSessionRunner,
    InvocationFailed,
    SDKSessionRunner,
)
from ledger import cmd_set_phase, resolve_aborted, resolve_unevaluated  # noqa: E402
from driver.status import _derive_state  # noqa: E402
from process_group import terminate_group  # noqa: E402
from search_space_state import empty_search_space_state  # noqa: E402
from semantic_space import complete_point, space_receipt  # noqa: E402
from tests.fixtures import background_text, fixture_registry, record  # noqa: E402
from tests.test_search_space_state import empty_experience  # noqa: E402


def _run_dir(tmp: Path, *, usable: float, reserve: float = 0.0) -> Path:
    """A run whose cutoff (deadline − reserve) is ``usable`` seconds away."""
    run_dir = tmp / "runs" / "toy" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "framework_cfg.json").write_text(json.dumps({
        "deadline": time.time() + usable + reserve,
        "final_reserve_seconds": reserve,
    }))
    return run_dir


def _role(**overrides) -> RoleDefinition:
    base = dict(name="hillclimb-editor", prompt_file="hillclimb-editor.md",
                tools=("Read", "Write"), disallowed=("Agent",),
                receipt_schema={"edited": "bool", "summary": "str"},
                corrective_attempts=1)
    base.update(overrides)
    return RoleDefinition(**base)


class RefreshCutoffTests(unittest.TestCase):
    def test_refresh_is_skipped_at_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with mock.patch.object(experiment, "_time_reached", return_value=True), \
                    mock.patch.object(experiment, "_invoke") as invoke:
                self.assertFalse(experiment._refresh(None, None, "toy", "r1", run_dir,
                                                    ROOT, None, EventsLog(run_dir)))
            invoke.assert_not_called()

    def test_refresh_failure_preserves_search_without_retry(self):
        for reason in ("no receipt", "API failure", "time budget reached"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp)
                (run_dir / "ledger.json").write_text('{"dag_revision": 1}')
                cmd = mock.Mock(return_value=subprocess.CompletedProcess([], 0, '{"ready": true}', ''))
                with mock.patch.object(experiment, "_time_reached", return_value=False), \
                        mock.patch.object(experiment, "_invoke", side_effect=InvocationFailed(
                            "experience-extractor", [reason])) as invoke, \
                        mock.patch.object(experiment, "_or_block") as block:
                    self.assertFalse(experiment._refresh(None, None, "toy", "r1", run_dir,
                                                        ROOT, cmd, EventsLog(run_dir)))
                invoke.assert_called_once()
                block.assert_not_called()
                self.assertIn("--error", cmd.call_args.args[0])
                self.assertEqual(len(_events(run_dir, "experience_update_failed")), 1)


class _Result:
    def __init__(self, *, is_error=False, subtype=None, num_turns=1):
        self.session_id = "sess-fake"
        self.is_error = is_error
        self.subtype = subtype
        self.num_turns = num_turns
        self.total_cost_usd = 0.0
        self.usage = {}


class _Init:
    subtype = "init"

    def __init__(self):
        self.data = {"session_id": "sess-fake"}


class _Client:
    """Scripted client: ``turns`` is a list of async callables, one per
    query, each yielding the turn's messages."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.queries = []
        self.interrupts = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        self.queries.append(prompt)

    def receive_response(self):
        return self.turns.pop(0)()

    async def interrupt(self):
        self.interrupts += 1


def _events(run_dir: Path, kind: str) -> list[dict]:
    path = run_dir / "driver_events.jsonl"
    if not path.exists():
        return []
    return [row for row in (json.loads(line) for line in
                            path.read_text().splitlines() if line.strip())
            if row.get("kind") == kind]


class SessionBudgetTests(unittest.TestCase):
    def test_refused_past_cutoff_at_the_shared_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_dir(Path(tmp), usable=-5)
            factory = mock.Mock()
            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir),
                                      client_factory=factory)
            ctx = InvocationContext(task="toy", tag="r1", run_dir=run_dir,
                                    invocation_id=1)
            with self.assertRaises(InvocationFailed) as cm:
                runner.run(_role(), ctx)
            self.assertIn("time budget reached", cm.exception.problems)
            factory.assert_not_called()
            self.assertEqual(_events(run_dir, "session_refused")[0]["reason"],
                             "time_reached")
            with self.assertRaises(InvocationFailed):  # fake mirrors the gate
                FakeSessionRunner([{"receipt": {}}]).run(_role(), ctx)

    def test_remaining_seconds_injected_and_budget_follows_the_main_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_dir(Path(tmp), usable=1000)
            # A proposer session stored under the candidate: no cfg of its own.
            session_dir = run_dir / "candidates" / "007" / "_hebo_llm"
            session_dir.mkdir(parents=True)
            runner = FakeSessionRunner([{"receipt": {"ok": True}}])
            runner.run(_role(), InvocationContext(
                task="toy", tag="r1", run_dir=session_dir, invocation_id=1))
            extra = runner.calls[0][1].extra
            self.assertTrue(900 <= extra["time_budget_remaining_seconds"] <= 1000)

    def test_deadline_cancels_a_silent_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_dir(Path(tmp), usable=0.5)

            async def silent():
                yield _Init()
                await asyncio.sleep(30)  # would only wake far past the cutoff
                yield _Result()

            client = _Client([silent])
            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir),
                                      client_factory=lambda options: client)
            started = time.monotonic()
            with self.assertRaises(InvocationFailed) as cm:
                runner.run(_role(), InvocationContext(
                    task="toy", tag="r1", run_dir=run_dir, invocation_id=1))
            self.assertLess(time.monotonic() - started, 5)
            self.assertIn("mid-session", cm.exception.problems[0])
            self.assertEqual(client.interrupts, 1)
            limit = _events(run_dir, "session_wall_limit")[0]
            self.assertEqual(limit["limit_source"], "deadline")

    def test_idle_timeout_ends_the_drain_as_an_error_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            async def hung():
                yield _Init()
                await asyncio.sleep(30)
                yield _Result()

            client = _Client([hung])
            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir),
                                      client_factory=lambda options: client)
            role = _role(idle_timeout_seconds=0.2)
            with self.assertRaises(InvocationFailed) as cm:
                runner.run(role, InvocationContext(
                    task="toy", tag="r1", run_dir=run_dir, invocation_id=1))
            self.assertIn("idle_timeout", " ".join(cm.exception.problems))
            self.assertEqual(len(_events(run_dir, "idle_timeout")), 1)
            self.assertEqual(_events(run_dir, "transport_retry"), [])

    def _wall_limit_runner(self, run_dir, store, *, rescue_accepts,
                           during_correction=False):
        async def missing_receipt():
            yield _Init()
            yield _Result()

        async def slow_turn():
            yield _Init()
            await asyncio.sleep(0.15)
            yield object()  # first message after the limit: trips it
            yield _Result(is_error=True, subtype="error_during_execution",
                          num_turns=3)

        async def rescue_turn():
            yield _Init()
            if rescue_accepts:
                (run_dir / "train.py").write_text("# partial edit\n")
                store.persist_receipt("hillclimb-editor", 1,
                                      {"edited": True, "summary": "partial"})
            yield _Result(num_turns=1)

        turns = [slow_turn, rescue_turn]
        if during_correction:
            turns.insert(0, missing_receipt)
        client = _Client(turns)
        runner = SDKSessionRunner(model="m", events=EventsLog(run_dir),
                                  client_factory=lambda options: client)
        return runner, client

    def test_wall_limit_soft_rescue_role_gets_one_submit_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            store = ReceiptStore(run_dir)
            runner, client = self._wall_limit_runner(run_dir, store,
                                                     rescue_accepts=True)
            role = _role(wall_limit_seconds=0.05, soft_rescue=True)
            receipt = runner.run(role, InvocationContext(
                task="toy", tag="r1", run_dir=run_dir, invocation_id=1))
            self.assertEqual(receipt["summary"], "partial")
            self.assertEqual(client.queries[1], SOFT_RESCUE_MESSAGE)
            self.assertEqual(client.interrupts, 1)
            limit = _events(run_dir, "session_wall_limit")[0]
            self.assertEqual((limit["limit_source"], limit["rescued"]),
                             ("role", True))

    def test_wall_limit_terminal_role_fails_without_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            store = ReceiptStore(run_dir)
            runner, client = self._wall_limit_runner(run_dir, store,
                                                     rescue_accepts=True)
            role = _role(wall_limit_seconds=0.05, soft_rescue=False)
            with self.assertRaises(InvocationFailed) as cm:
                runner.run(role, InvocationContext(
                    task="toy", tag="r1", run_dir=run_dir, invocation_id=1))
            self.assertIn("wall-clock limit", cm.exception.problems[0])
            self.assertEqual(len(client.queries), 1)
            self.assertFalse(_events(run_dir, "session_wall_limit")[0]["rescued"])

    def test_wall_limit_during_last_correction_gets_one_rescue(self) -> None:
        for accepts in (True, False):
            with self.subTest(rescue_accepts=accepts), tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp)
                runner, client = self._wall_limit_runner(
                    run_dir, ReceiptStore(run_dir), rescue_accepts=accepts,
                    during_correction=True)
                role = _role(wall_limit_seconds=0.05, soft_rescue=True,
                             corrective_attempts=1)
                ctx = InvocationContext(task="toy", tag="r1", run_dir=run_dir,
                                        invocation_id=1)
                if accepts:
                    self.assertEqual(runner.run(role, ctx)["summary"], "partial")
                else:
                    with self.assertRaises(InvocationFailed) as cm:
                        runner.run(role, ctx)
                    self.assertIn("rescue turn produced no receipt", str(cm.exception))
                self.assertEqual(len(client.queries), 3)
                self.assertEqual(client.queries[-1], SOFT_RESCUE_MESSAGE)
                limits = _events(run_dir, "session_wall_limit")
                self.assertEqual(len(limits), 1)
                self.assertEqual(limits[0]["rescued"], accepts)

    def test_transport_failure_retries_with_the_same_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            store = ReceiptStore(run_dir)

            async def relay_glitch():
                yield _Init()
                yield _Result(is_error=True, subtype="error_during_execution",
                              num_turns=1)

            async def success():
                yield _Init()
                (run_dir / "train.py").write_text("# edited\n")
                store.persist_receipt("hillclimb-editor", 1,
                                      {"edited": True, "summary": "ok"})
                yield _Result(num_turns=2)

            clients = []

            def factory(options):
                clients.append(options)
                turns = [relay_glitch] if len(clients) < 3 else [success]
                return _Client(turns)

            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir),
                                      client_factory=factory)
            runner._sleep = lambda seconds: None
            ctx = InvocationContext(task="toy", tag="r1", run_dir=run_dir,
                                    invocation_id=1,
                                    resume_session_id="sess-parent")
            with mock.patch.object(runner, "_build_options",
                                   side_effect=lambda role, c, *a, **k: c):
                receipt = runner.run(_role(), ctx)
            self.assertTrue(receipt["edited"])
            retries = _events(run_dir, "transport_retry")
            self.assertEqual([r["attempt"] for r in retries], [1, 2])
            self.assertEqual([r["backoff_seconds"] for r in retries], [5.0, 15.0])
            # every retry carried the same resume id (the options ARE the ctx)
            self.assertEqual({c.resume_session_id for c in clients},
                             {"sess-parent"})
            self.assertEqual(_events(run_dir, "corrective_followup"), [])

    def test_transport_failures_exhaust_into_invocation_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            async def glitch():
                yield _Init()
                yield _Result(is_error=True, subtype="error_during_execution",
                              num_turns=0)

            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir),
                                      client_factory=lambda o: _Client([glitch]))
            runner._sleep = lambda seconds: None
            with self.assertRaises(InvocationFailed):
                runner.run(_role(), InvocationContext(
                    task="toy", tag="r1", run_dir=run_dir, invocation_id=1))
            self.assertEqual(len(_events(run_dir, "transport_retry")), 3)

    def test_key_class_failure_rotates_to_the_backup_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            store = ReceiptStore(run_dir)

            class _ApiResult(_Result):
                def __init__(self, *, api_error_status=None, **kw):
                    super().__init__(**kw)
                    self.api_error_status = api_error_status

            async def quota_fail():
                yield _Init()
                yield _ApiResult(is_error=True,
                                 subtype="error_during_execution",
                                 num_turns=1, api_error_status=402)

            async def success():
                yield _Init()
                (run_dir / "train.py").write_text("# edited\n")
                store.persist_receipt("hillclimb-editor", 1,
                                      {"edited": True, "summary": "ok"})
                yield _Result(num_turns=2)

            options_seen = []

            def factory(options):
                options_seen.append(options)
                turns = [quota_fail] if len(options_seen) == 1 else [success]
                return _Client(turns)

            pool = AuthTokenPool()
            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir),
                                      client_factory=factory, token_pool=pool)
            runner._sleep = lambda seconds: None
            env = {"ANTHROPIC_AUTH_TOKEN": "tok-a",
                   "ANTHROPIC_AUTH_TOKEN_BACKUPS": "tok-b,tok-c"}
            with mock.patch.dict(os.environ, env):
                receipt = runner.run(_role(), InvocationContext(
                    task="toy", tag="r1", run_dir=run_dir, invocation_id=1))
            self.assertTrue(receipt["edited"])
            # the retry carried the rotated key via options env
            self.assertEqual([o.env["ANTHROPIC_AUTH_TOKEN"]
                              for o in options_seen], ["tok-a", "tok-b"])
            self.assertEqual(pool.index, 1)
            fallback = _events(run_dir, "api_key_fallback")
            self.assertEqual([(f["api_error_status"], f["key_index"])
                              for f in fallback], [(402, 1)])
            # rotation spent no transport retry
            self.assertEqual(_events(run_dir, "transport_retry"), [])


# --- driver jobs and process groups -----------------------------------------


def _job_fixture(tmp: Path, *, usable: float):
    repo = tmp / "repo"
    (repo / "tasks" / "toy").mkdir(parents=True)
    (repo / "tasks" / "toy" / "task.toml").write_text(
        '[env]\ntype = "uv"\nproject = "tasks/toy"\n')
    run_dir = repo / "runs" / "toy" / "r1"
    candidate = run_dir / "candidates" / "007"
    candidate.mkdir(parents=True)
    (candidate / "train.py").write_text("# candidate\n")
    (run_dir / "framework_cfg.json").write_text(json.dumps({
        "deadline": time.time() + usable, "final_reserve_seconds": 0}))
    ctx = InvocationContext(task="toy", tag="r1", run_dir=run_dir,
                            invocation_id=3, run_id="007")
    return repo, ctx


IGNORE_TERM = [sys.executable, "-c",
               "import signal, time; signal.signal(signal.SIGTERM, "
               "signal.SIG_IGN); time.sleep(60)"]


class DriverJobDeadlineTests(unittest.TestCase):
    def test_handoff_refused_when_lease_wait_crosses_the_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, ctx = _job_fixture(Path(tmp), usable=0.3)

            @contextlib.contextmanager
            def slow_lease(task_toml, *, owner=None, wait_timeout=...):
                time.sleep(0.5)  # the queue itself crosses the cutoff
                yield dict(jobs.NO_LEASE) if hasattr(jobs, "NO_LEASE") else {
                    "devices": [], "env": {}}

            argv = [sys.executable, "-c", "print('must not run')"]
            with mock.patch.object(jobs, "task_resource_lease", slow_lease), \
                    mock.patch.object(jobs, "build_driver_job",
                                      return_value=(argv, ctx.run_dir / "j.log",
                                                    "007")), \
                    mock.patch.object(jobs.subprocess, "Popen") as popen:
                result = jobs.execute_driver_job(
                    "tuner-orchestrator", ctx,
                    {"kind": "phase_c", "run_id": "007"}, repo_root=repo)
            popen.assert_not_called()
            self.assertFalse(result["accepted"])
            record = json.loads(Path(result["job_record"]).read_text())
            self.assertEqual(record["status"], "refused_time_reached")

    def test_running_job_is_killed_at_the_cutoff_after_escalation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, ctx = _job_fixture(Path(tmp), usable=0.5)
            released = []

            @contextlib.contextmanager
            def lease(task_toml, *, owner=None, wait_timeout=...):
                try:
                    yield {"devices": [], "env": {}}
                finally:
                    released.append(time.monotonic())

            with mock.patch.object(jobs, "task_resource_lease", lease), \
                    mock.patch.object(jobs, "build_driver_job",
                                      return_value=(IGNORE_TERM,
                                                    ctx.run_dir / "j.log",
                                                    "007")), \
                    mock.patch.object(jobs, "JOB_TERMINATE_GRACE_SECONDS", 0.3):
                started = time.monotonic()
                result = jobs.execute_driver_job(
                    "tuner-orchestrator", ctx,
                    {"kind": "phase_c", "run_id": "007"}, repo_root=repo)
            self.assertLess(time.monotonic() - started, 10)
            self.assertFalse(result["accepted"])
            record = json.loads(Path(result["job_record"]).read_text())
            self.assertEqual(record["status"], "deadline_killed")
            self.assertEqual(record["returncode"], -signal.SIGKILL)
            with self.assertRaises(ProcessLookupError):  # reaped before release
                os.kill(record["pid"], 0)
            self.assertEqual(len(released), 1)
            self.assertEqual(jobs._live_children, [])


class ProcessGroupTests(unittest.TestCase):
    def test_terminate_group_escalates_and_waits(self) -> None:
        proc = subprocess.Popen(IGNORE_TERM, start_new_session=True)
        time.sleep(0.7)  # let the child install its SIGTERM handler
        started = time.monotonic()
        code = terminate_group(proc, grace=0.3)
        self.assertEqual(code, -signal.SIGKILL)
        self.assertLess(time.monotonic() - started, 10)
        self.assertIsNotNone(proc.poll())

    def test_timed_run_wrapper_kills_its_child_group_on_sigterm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            child = ("import os, time, pathlib; "
                     f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
                     "time.sleep(60)")
            wrapper = subprocess.Popen(
                [sys.executable, str(ROOT / "tools" / "timed_run.py"), "60",
                 sys.executable, "-c", child],
                start_new_session=True)
            for _ in range(200):
                if pid_file.exists() and pid_file.read_text().strip():
                    break
                time.sleep(0.05)
            child_pid = int(pid_file.read_text())
            os.kill(wrapper.pid, signal.SIGTERM)
            wrapper.wait(timeout=40)
            for _ in range(100):  # the wrapper exits only after its child
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                os.kill(child_pid, signal.SIGKILL)
                self.fail("evaluation child outlived its wrapper")


# --- ledger settlement at the cutoff -----------------------------------------


class _SettleCmd:
    """The ledger/budget surface _settle_at_deadline drives, scripted."""

    def __init__(self, evals: int, cutoffs: int, *, resolve_ok: bool):
        self.evals, self.cutoffs, self.resolve_ok = evals, cutoffs, resolve_ok
        self.ledger_calls: list[str] = []

    def __call__(self, args, repo_root, check=True, **kw):
        args = [str(a) for a in args]
        joined = " ".join(args)
        if "evaluation_budget.py" in joined:
            return subprocess.CompletedProcess(args, 0, json.dumps({
                "reached": True, "per_candidate": [
                    {"run_id": "001", "evals": self.evals,
                     "time_cutoff_evals": self.cutoffs}]}), "")
        if "ledger.py" in joined:
            sub = args[args.index("tools/ledger.py") + 1]
            self.ledger_calls.append(sub)
            if sub == "resolve-unevaluated" and not self.resolve_ok:
                raise subprocess.CalledProcessError(1, args, stderr="refused")
            return subprocess.CompletedProcess(args, 0, "{}", "")
        return subprocess.CompletedProcess(args, 0, "{}", "")


class DeadlineSettlementTests(unittest.TestCase):
    def _run(self, tmp: Path, cmd, warm) -> tuple[Path, list[dict]]:
        run_dir = tmp / "run"
        candidate = run_dir / "candidates" / "001"
        candidate.mkdir(parents=True)
        (run_dir / "ledger.json").write_text(json.dumps({
            "records": [{"run_id": "001", "status": "pending"}]}))
        if warm is not None:
            (candidate / "tune_report.json").write_text(json.dumps({
                "phase_a": {"status": "ok", "best_warm_score": warm}}))
        events = EventsLog(run_dir)
        experiment._settle_at_deadline(run_dir, "001", tmp, cmd, events)
        return run_dir, (_events(run_dir, "candidate_settled_at_deadline")
                         + _events(run_dir, "candidate_settled"))

    def test_zero_attempts_resolve_unevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = _SettleCmd(0, 0, resolve_ok=True)
            _, settled = self._run(Path(tmp), cmd, warm=None)
            self.assertEqual(cmd.ledger_calls, ["resolve-unevaluated"])
            self.assertEqual(settled[0]["outcome"], "unevaluated")

    def test_valid_warm_score_settles_without_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = _SettleCmd(4, 0, resolve_ok=False)
            _, settled = self._run(Path(tmp), cmd, warm=0.42)
            self.assertEqual(cmd.ledger_calls, ["set-tuning", "record-run"])
            self.assertEqual(settled[0]["best_warm_score"], 0.42)
            self.assertNotIn("crash", cmd.ledger_calls)

    def test_all_attempts_time_cut_off_resolve_unevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = _SettleCmd(2, 2, resolve_ok=True)
            _, settled = self._run(Path(tmp), cmd, warm=None)
            self.assertEqual(cmd.ledger_calls, ["resolve-unevaluated"])
            self.assertEqual(settled[0]["outcome"], "unevaluated")

    def test_attempts_without_result_or_cutoff_record_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = _SettleCmd(2, 0, resolve_ok=False)
            _, settled = self._run(Path(tmp), cmd, warm=float("inf"))
            self.assertEqual(cmd.ledger_calls, ["record-run"])
            self.assertEqual(settled[0]["outcome"], "crash")


class LedgerDeadlineTests(unittest.TestCase):
    def _ledger(self):
        registry = fixture_registry()
        baseline = complete_point(registry)
        terminal = record("000", "fresh", [], baseline, score=0.5, status="keep")
        terminal["dag_revision"] = 1
        pending = record("001", "improve", ["000"], baseline,
                         score=float("inf"), status="pending",
                         prior_records=[terminal])
        pending["final_best_score"] = None
        return registry, {
            "task": "hard-interactions", "tag": "t", "metric": "validation_loss",
            "search_space": space_receipt(registry),
            "search_space_state": empty_search_space_state(),
            "dag_revision": 1,
            "records": [terminal, pending],
            "experience": {**empty_experience("000"), "dag_revision": 0},
        }

    def _attempt_rows(self, cutoff: bool) -> str:
        rows = [{"schema_version": 1, "kind": "score_attempt",
                 "attempt_id": "eval-000001", "run_id": "001",
                 "phase": "phase_a", "method": "warmstart", "params": {}},
                {"schema_version": 1, "kind": "score_completion",
                 "attempt_id": "eval-000001", "run_id": "001",
                 "duration_seconds": 12.0, **({"time_cutoff": True} if cutoff else {})}]
        return "".join(json.dumps(r) + "\n" for r in rows)

    def test_time_cut_off_attempts_resolve_unevaluated_with_their_own_kind(self) -> None:
        registry, ledger = self._ledger()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            ledger_path = run_dir / "ledger.json"
            ledger_path.write_text(json.dumps(ledger))
            (run_dir / "background.md").write_text(background_text(registry))
            (run_dir / "framework_cfg.json").write_text(json.dumps({
                "max_evaluations": None, "deadline": time.time() - 10}))
            (run_dir / "evaluation_attempts.jsonl").write_text(
                self._attempt_rows(cutoff=False))
            with self.assertRaisesRegex(ValueError, "objective attempt"):
                resolve_unevaluated(ledger_path, "hard-interactions", "001")
            (run_dir / "evaluation_attempts.jsonl").write_text(
                self._attempt_rows(cutoff=True))
            resolved = resolve_unevaluated(ledger_path, "hard-interactions", "001")
            receipt = resolved["unevaluated_receipt"]
            self.assertEqual(receipt["kind"],
                             "time_budget_reached_before_finite_result")
            self.assertEqual(receipt["candidate_objective_attempts"], 1)

    def test_set_phase_completed_tolerates_the_terminal_delta_at_the_cutoff(self) -> None:
        registry, ledger = self._ledger()
        ledger["records"][1]["status"] = "unevaluated"
        ledger["records"][1]["final_best_score"] = None
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            ledger_path = run_dir / "ledger.json"
            ledger_path.write_text(json.dumps(ledger))
            (run_dir / "evaluation_attempts.jsonl").write_text("")
            args = types.SimpleNamespace(
                ledger=str(ledger_path), task="hard-interactions",
                phase="completed", stop_condition=None, budget=None)
            (run_dir / "framework_cfg.json").write_text(json.dumps({
                "max_evaluations": None, "deadline": time.time() + 3600}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_set_phase(args), 0)
            (run_dir / "framework_cfg.json").write_text(json.dumps({
                "max_evaluations": None, "deadline": time.time() - 10}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_set_phase(args), 0)
            stored = json.loads(ledger_path.read_text())
            self.assertEqual(stored["run_state"]["phase"], "completed")
            self.assertEqual(stored["run_state"]["active_stop_condition"],
                             "time_budget_reached")
            # The guard-vetted cutoff completion (with its unrefreshed
            # terminal delta) reads back as completed — no downgrade to
            # final_experience_refresh_required.
            cfg = {"max_evaluations": None, "deadline": time.time() - 10}
            self.assertEqual(
                _derive_state(stored, cfg, 0),
                ("completed", "time_budget_reached"))


class AbortedSettlementTests(unittest.TestCase):
    """resolve-aborted: the evidence-neutral seat-skip settlement
    (PLAN-block-escalation Wave 1)."""

    def _pending_ledger(self, tmp: Path) -> Path:
        registry = fixture_registry()
        terminal = record("000", "fresh", [], complete_point(registry),
                          score=0.5, status="keep")
        pending = record("001", "improve", ["000"], complete_point(registry),
                         score=float("inf"), status="pending",
                         prior_records=[terminal])
        pending["final_best_score"] = None
        ledger = {
            "task": "hard-interactions", "tag": "t",
            "metric": "validation_loss",
            "search_space": space_receipt(registry),
            "search_space_state": empty_search_space_state(),
            "dag_revision": 3,
            "records": [terminal, pending],
            "experience": {"dag_revision": 3},
        }
        ledger_path = tmp / "ledger.json"
        ledger_path.write_text(json.dumps(ledger))
        return ledger_path

    def test_pending_becomes_aborted_without_observation_or_dag_bump(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = self._pending_ledger(Path(tmp))
            before = json.loads(ledger_path.read_text())
            resolved = resolve_aborted(ledger_path, "hard-interactions",
                                       "001", ["writer down"])
            self.assertEqual(resolved["status"], "aborted")
            self.assertIsNone(resolved["final_best_score"])
            self.assertEqual(resolved["aborted_receipt"]["kind"],
                             "seat_aborted_infra_failure")
            self.assertEqual(resolved["aborted_receipt"]["problems"],
                             ["writer down"])
            after = json.loads(ledger_path.read_text())
            # No DAG bump and no attempt observation, although the record
            # carries a semantic point (a keep/discard/crash settlement
            # would have captured one): a skipped seat is not evidence.
            self.assertEqual(after["dag_revision"], before["dag_revision"])
            self.assertEqual(after["records"][1].get("dag_revision"),
                             before["records"][1].get("dag_revision"))
            self.assertEqual(after.get("attempt_observations", []), [])
            self.assertIsNotNone(
                after["records"][1]["semantic_point"].get("point_id"))
            # The aborted record counts as lifecycle-terminal for phase
            # derivation (completions are not held hostage by a skip).
            cfg = {"max_evaluations": 1}
            self.assertEqual(_derive_state(after, cfg, 1)[0], "completed")

    def test_aborted_refuses_existing_objective_evidence(self) -> None:
        for evidence in ("warm_score", "attempt"):
            with self.subTest(evidence=evidence), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                ledger_path = self._pending_ledger(run)
                before = ledger_path.read_bytes()
                if evidence == "warm_score":
                    candidate = run / "candidates/001"
                    candidate.mkdir(parents=True)
                    (candidate / "tune_report.json").write_text(
                        '{"phase_a": {"best_warm_score": 0.1}}')
                else:
                    from evaluation_budget import ATTEMPT_KIND, ATTEMPT_LOG
                    (run / ATTEMPT_LOG).write_text(json.dumps({
                        "schema_version": 1, "kind": ATTEMPT_KIND, "run_id": "001",
                        "attempt_id": "a", "phase": "warmstart", "method": "direct", "params": {}}) + "\n")
                with self.assertRaisesRegex(ValueError, "evidence|attempt|score"):
                    resolve_aborted(ledger_path, "hard-interactions", "001", ["session failed"])
                self.assertEqual(ledger_path.read_bytes(), before)

    def test_aborted_is_idempotent_and_refuses_terminal_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = self._pending_ledger(Path(tmp))
            resolve_aborted(ledger_path, "hard-interactions", "001", ["a"])
            again = resolve_aborted(ledger_path, "hard-interactions",
                                    "001", ["a"])
            self.assertEqual(again["status"], "aborted")
            with self.assertRaisesRegex(ValueError, "must be pending"):
                resolve_aborted(ledger_path, "hard-interactions", "000",
                                ["a"])


if __name__ == "__main__":
    unittest.main()
