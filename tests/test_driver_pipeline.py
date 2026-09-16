"""Session-channel pipelining: bounded overlap of seat implementations,
serial GPU channel, driver-owned settlement of `unevaluated` receipts, and
the tool-layer ledger write protection."""

import asyncio
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.events import EventsLog  # noqa: E402
from driver.loops import rounds  # noqa: E402
from driver.loops.experiment import run_experiment  # noqa: E402
from driver.receipts import ReceiptStore  # noqa: E402
from driver.roles import ROLES, InvocationContext, record_is_terminal  # noqa: E402
from driver.session import FakeSessionRunner, SDKSessionRunner  # noqa: E402
from tests.test_driver_experiment import (  # noqa: E402
    ExperimentCmd,
    write_background,
    write_task,
    writer_effect,
)


class PipelineCmd(ExperimentCmd):
    """ExperimentCmd whose fresh framework config enables two seats at once."""

    def __init__(self, repo: Path, session_concurrency: int):
        super().__init__(repo)
        self.session_concurrency = session_concurrency
        self.lock = threading.Lock()

    def __call__(self, args, repo_root, check=True, capture=True, **kw):
        with self.lock:  # the fake ledger is a plain JSON file
            result = super().__call__(args, repo_root, check=check,
                                      capture=capture, **kw)
        if "init_run.py" in " ".join(str(a) for a in args):
            cfg_path = self.run_dir / "framework_cfg.json"
            cfg = json.loads(cfg_path.read_text())
            cfg["pipeline"] = {"session_concurrency": self.session_concurrency}
            cfg_path.write_text(json.dumps(cfg))
        return result


class RoleRunner:
    """Thread-safe scripted runner keyed by role (seat order is not fixed
    under the session channel, so a positional script cannot be used)."""

    def __init__(self, handlers: dict):
        self.handlers = handlers
        self.calls: list[tuple[str, InvocationContext]] = []
        self._lock = threading.Lock()

    def run(self, role, ctx):
        with self._lock:
            self.calls.append((role.name, ctx))
        store = ReceiptStore(ctx.run_dir)
        store.persist_session_id(role.name, ctx.invocation_id,
                                 f"fake-sess-{ctx.invocation_id:04d}")
        receipt = self.handlers[role.name](ctx)
        store.persist_receipt(role.name, ctx.invocation_id, receipt)
        return receipt


WRITER_SECONDS = 0.6
EXTRACTOR_TURN_SECONDS = 0.1
JOB_SECONDS = 0.2


def _seat_handlers(cmd: PipelineCmd, seat_ids: list[str]) -> dict:
    def background(ctx):
        write_background(ctx.run_dir)
        return {"status": "ok", "background": "background.md",
                "retrieval_manifest": "background_retrieval.json"}

    def ideas(ctx):
        for run_id in seat_ids:
            cmd(["python", "tools/ledger.py", "add-record", "--run-id", run_id],
                cmd.repo)
        return {"actions": [{"run_id": r, "op": "fresh"} for r in seat_ids]}

    def writer(ctx):
        time.sleep(WRITER_SECONDS)  # API-bound work: overlaps another seat's job
        writer_effect(ctx)
        return {"status": "written", "wrote": True,
                "candidate_dir": str(ctx.run_dir / "candidates" / ctx.run_id)}

    def extractor(ctx):
        time.sleep(EXTRACTOR_TURN_SECONDS)
        if "driver_job_result" not in ctx.extra:
            return {"run_id": ctx.run_id, "status": "driver_job",
                    "ledger_updated": False,
                    "driver_job": {"kind": "warmstart", "run_id": ctx.run_id,
                                   "k_eval": 3}}
        cmd(["python", "tools/ledger.py", "record-run", "--run-id", ctx.run_id,
             "--status", "keep"], cmd.repo)
        return {"run_id": ctx.run_id, "status": "keep", "ledger_updated": True}

    def tuner(ctx):
        return {"tuned_run_id": "none", "tuned": False, "ledger_updated": False}

    return {"background-researcher": background, "idea-generator": ideas,
            "candidate-writer": writer, "tunable-contract-extractor": extractor,
            "tuner-orchestrator": tuner}


class GpuChannel:
    """A fake objective job behind a capacity-1 lock, standing in for the
    device lease execute_driver_job takes: jobs queue, never overlap."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jobs: list[str] = []
        self.busy_seconds = 0.0

    def __call__(self, role, ctx, request, *, repo_root):
        with self.lock:
            self.jobs.append(request["run_id"])
            time.sleep(JOB_SECONDS)
            self.busy_seconds += JOB_SECONDS
        return {"kind": request["kind"], "run_id": request["run_id"],
                "returncode": 0}


class SessionChannelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        write_task(self.repo)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, concurrency: int) -> tuple[PipelineCmd, RoleRunner, GpuChannel, float]:
        cmd = PipelineCmd(self.repo, concurrency)
        # two seat checks, pre-tuner, then the next round's start check
        cmd.reached = [False, False, False, True]
        runner = RoleRunner(_seat_handlers(cmd, ["000", "001"]))
        gpu = GpuChannel()
        started = time.monotonic()
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd, job_runner=gpu,
                       semantic_policy="coverage_attempt",
                       scheduler_policy="v3_2")
        return cmd, runner, gpu, time.monotonic() - started

    def test_two_seats_overlap_llm_and_gpu_time(self) -> None:
        cmd, runner, gpu, wall = self._run(concurrency=2)
        self.assertEqual(cmd._ledger().get("phase"), "completed")
        statuses = {r["run_id"]: r["status"] for r in cmd._ledger()["records"]}
        self.assertEqual(statuses, {"000": "keep", "001": "keep"})
        self.assertEqual(sorted(gpu.jobs), ["000", "001"])
        serial = 2 * (WRITER_SECONDS + 2 * EXTRACTOR_TURN_SECONDS + JOB_SECONDS)
        self.assertLess(wall, serial * 0.8,
                        f"wall {wall:.2f}s should be well below the serial "
                        f"sum {serial:.2f}s")
        roles = [name for name, _ in runner.calls]
        self.assertEqual(roles[:2], ["background-researcher", "idea-generator"])
        self.assertEqual(roles.count("candidate-writer"), 2)
        self.assertEqual(roles.count("tunable-contract-extractor"), 4)
        events = [json.loads(line)["kind"] for line in
                  (cmd.run_dir / "driver_events.jsonl").read_text().splitlines()]
        self.assertEqual(events.count("seat_started"), 2)
        self.assertIn("seats_completed", events)
        # invocation ids issued from two threads never collide
        inv_ids = [ctx.invocation_id for _, ctx in runner.calls]
        self.assertEqual(len(inv_ids), len(set(inv_ids)))

    def test_concurrency_one_is_the_serial_loop(self) -> None:
        cmd, runner, gpu, wall = self._run(concurrency=1)
        self.assertEqual(cmd._ledger().get("phase"), "completed")
        roles = [name for name, _ in runner.calls]
        self.assertEqual(
            roles[2:8],
            ["candidate-writer", "tunable-contract-extractor",
             "tunable-contract-extractor", "candidate-writer",
             "tunable-contract-extractor", "tunable-contract-extractor"])
        self.assertEqual(gpu.jobs, ["000", "001"])


class UnevaluatedReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        write_task(self.repo)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_driver_settles_a_zero_attempt_candidate_at_the_stop(self) -> None:
        cmd = ExperimentCmd(self.repo)
        # round start F, seat check F, then the budget is reached for the
        # seat check F, then the budget is reached for every later check
        cmd.reached = [False, True, True]
        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: write_background(ctx.run_dir)},
            {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
             "side_effects": lambda ctx: cmd([
                 "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                 self.repo)},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            # the extractor never got an objective slot: it says so instead
            # of hand-editing the ledger, and the driver resolves the record
            {"receipt": {"run_id": "000", "status": "unevaluated",
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd,
                       semantic_policy="coverage_attempt",
                       scheduler_policy="v3_2")
        record = cmd._ledger()["records"][0]
        self.assertEqual(record["status"], "unevaluated")
        self.assertEqual(cmd._ledger().get("phase"), "completed")
        self.assertTrue(any("resolve-unevaluated" in call for call in cmd.calls))

    def test_postcondition_accepts_the_unevaluated_receipt(self) -> None:
        run_dir = self.repo / "runs" / "fake-task" / "t1"
        run_dir.mkdir(parents=True)
        (run_dir / "ledger.json").write_text(json.dumps(
            {"records": [{"run_id": "000", "status": "pending"}]}))
        ctx = InvocationContext(task="fake-task", tag="t1", run_dir=run_dir,
                                invocation_id=4, run_id="000")
        self.assertIsNotNone(record_is_terminal(ctx))
        ReceiptStore(run_dir).persist_receipt(
            "tunable-contract-extractor", 4,
            {"run_id": "000", "status": "unevaluated", "ledger_updated": False})
        self.assertIsNone(record_is_terminal(ctx))
        # a receipt that claims a terminal write the ledger does not show
        ReceiptStore(run_dir).persist_receipt(
            "tunable-contract-extractor", 4,
            {"run_id": "000", "status": "keep", "ledger_updated": True},
            allow_replace=True)
        self.assertIsNotNone(record_is_terminal(ctx))


class LedgerWriteProtectionTests(unittest.TestCase):
    def _hook(self):
        runner = SDKSessionRunner(model="m", events=EventsLog(Path(tempfile.mkdtemp())))
        return runner._capability_hook(ROLES["tunable-contract-extractor"])

    def _verdict(self, hook, name, tool_input):
        return asyncio.run(hook({"tool_name": name, "tool_input": tool_input},
                                None, {}))

    def test_edit_and_write_on_the_ledger_are_denied(self) -> None:
        hook = self._hook()
        for name in ("Edit", "Write"):
            verdict = self._verdict(hook, name, {
                "file_path": "/x/runs/t/tag/ledger.json", "content": "{}"})
            self.assertEqual(
                verdict["hookSpecificOutput"]["permissionDecision"], "deny", name)
        allowed = self._verdict(hook, "Write", {
            "file_path": "/x/runs/t/tag/candidates/000/_warm_configs.json"})
        self.assertEqual(allowed, {})

    def test_bash_may_use_the_ledger_cli_but_not_rewrite_the_file(self) -> None:
        hook = self._hook()
        ok = self._verdict(hook, "Bash", {
            "command": "python tools/ledger.py record-run --ledger "
                       "runs/t/tag/ledger.json --run-id 000 --status keep"})
        self.assertEqual(ok, {})
        read = self._verdict(hook, "Bash", {
            "command": "cat runs/t/tag/ledger.json"})
        self.assertEqual(read, {})
        for command in ("sed -i 's/pending/keep/' runs/t/tag/ledger.json",
                        "python3 -c 'open(\"runs/t/tag/ledger.json\",\"w\")'",
                        "echo '{}' > runs/t/tag/ledger.json"):
            verdict = self._verdict(hook, "Bash", {"command": command})
            self.assertEqual(
                verdict["hookSpecificOutput"]["permissionDecision"], "deny",
                command)


class PerCandidateAttributionTests(unittest.TestCase):
    def test_eval_seconds_counts_only_the_candidates_attempts(self) -> None:
        import subprocess

        def cmd(args, repo_root, **kw):
            return subprocess.CompletedProcess(args, 0, json.dumps({
                "evaluations_done": 9,
                "per_candidate": [
                    {"run_id": "000", "evals": 2, "mean_seconds": 30.0},
                    {"run_id": "001", "evals": 7, "mean_seconds": 12.0},
                ]}), "")

        self.assertEqual(rounds._eval_seconds(Path("/r"), Path("/repo"), cmd, "000"),
                         (2, 30.0))
        self.assertEqual(rounds._eval_seconds(Path("/r"), Path("/repo"), cmd, "002"),
                         (0, None))


class JobReconcileTests(unittest.TestCase):
    def test_live_job_of_another_candidate_does_not_refuse(self) -> None:
        import subprocess as sp
        from driver.jobs import DriverJobError, _reconcile_running_jobs
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp) / "driver_jobs"
            jobs_dir.mkdir()
            live = sp.Popen(["sleep", "30"])
            try:
                (jobs_dir / "tunable-contract-extractor-0001.json").write_text(
                    json.dumps({"status": "running", "pid": live.pid,
                                "run_id": "007"}))
                _reconcile_running_jobs(jobs_dir, "008")  # other candidate: queue
                with self.assertRaises(DriverJobError):
                    _reconcile_running_jobs(jobs_dir, "007")  # same candidate
            finally:
                live.kill()
                live.wait()


if __name__ == "__main__":
    unittest.main()
