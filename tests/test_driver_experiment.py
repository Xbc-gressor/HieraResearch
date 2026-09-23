import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
import threading
from unittest import mock
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.loops import experiment
from driver.receipts import ReceiptStore
from driver.events import EventsLog  # noqa: E402
from driver.loops.background_audit import audit_completed  # noqa: E402
from driver.loops.common import RunBlocked  # noqa: E402
from driver.loops.experiment import (  # noqa: E402
    _note_seat_progress,
    _note_seat_skip,
    _phase_c_recover_close,
    _reset_seat_skip_state,
    _validator_error_messages,
    run_experiment,
)
from driver.session import FakeSessionRunner  # noqa: E402
from tests.fixtures import (  # noqa: E402
    background_text,
    fixture_registry,
    retrieval_hit_manifest,
)


def write_task(repo: Path, provided: bool = False) -> None:
    task_dir = repo / "tasks" / "fake-task"
    task_dir.mkdir(parents=True)
    seed = '\n[seed]\nprovided = true\nentrypoint = "train.py"\n' if provided else ""
    (task_dir / "task.toml").write_text(
        """
[env]
project = "tasks/fake-task"
[result]
metric = "neg_acc"
[run]
working_dir = "tasks/fake-task"
prepare_command = "uv run python prepare.py"
[constraints]
editable_files = ["train.py"]
readonly_files = ["prepare.py"]
allow_dependencies = false
""" + seed
    )
    (task_dir / "prepare.py").write_text("# fixed\n")
    (task_dir / "train.py").write_text("# provided baseline\n")


_run_experiment = run_experiment


def run_experiment(*args, **kwargs):
    cmd = kwargs.get("cmd")
    if isinstance(cmd, ExperimentCmd):
        kwargs.setdefault("job_runner", cmd.evaluate)
    return _run_experiment(*args, **kwargs)


class ExperimentCmd:
    """Fakes tools/ helpers. brief_queue/reached_queue drive loop progress."""

    def __init__(self, repo: Path):
        self.repo = repo
        self.calls: list[str] = []
        self.cwds: list = []
        self.brief_queue: list[dict] = []
        self.reached: list[bool] = []
        self.fail_next: set[str] = set()
        self.fail_payload: dict[str, tuple[int, str, str]] = {}
        self.raise_once: set[str] = set()

    @property
    def run_dir(self) -> Path:
        return self.repo / "runs" / "fake-task" / "t1"

    def _ledger(self) -> dict:
        path = self.run_dir / "ledger.json"
        return json.loads(path.read_text()) if path.exists() else {"records": []}

    def _save_ledger(self, data: dict) -> None:
        (self.run_dir / "ledger.json").write_text(json.dumps(data))

    def __call__(self, args, repo_root, check=True, capture=True, **kw):
        args = [str(a) for a in args]
        joined = " ".join(args)
        self.calls.append(joined)
        self.cwds.append(kw.get("cwd"))
        for marker in self.fail_next:
            if marker in joined:
                self.fail_next.discard(marker)
                return subprocess.CompletedProcess(args, 1, "", "boom")
        for marker, payload in self.fail_payload.items():
            if marker in joined:
                code, stdout, stderr = payload
                return subprocess.CompletedProcess(args, code, stdout, stderr)
        for marker in list(self.raise_once):
            if marker in joined:
                self.raise_once.discard(marker)
                raise subprocess.CalledProcessError(
                    1, args, "", f"helper refused: {marker}")

        if "ledger.py experience-context" in joined:
            output = Path(args[args.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}")
            return self._ok('{"ready": true}')
        if "ledger.py experience-attempt" in joined:
            return self._ok('{}')
        if "ledger.py set-experience" in joined:
            return self._ok('{"ok": true, "decision_ids": []}')
        if "init_run.py" in joined:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": 3, "per_runtime_limit": None,
                            "dimension_strategy": "catalog_subset"}))
            return self._ok("")
        if "ledger.py" in joined and "brief" in joined:
            brief = self.brief_queue.pop(0) if self.brief_queue else {
                "records": self._ledger()["records"],
                "pending_run_ids": [r["run_id"] for r in self._ledger()["records"]
                                    if r.get("status") == "pending"],
                "experience_refresh_required": False}
            return self._ok(json.dumps(brief))
        if "evaluation_budget.py" in joined and "status" in joined:
            reached = self.reached.pop(0) if self.reached else False
            return self._ok(json.dumps({"evaluations_done": 0, "reached": reached}))
        if "add-record" in joined:
            run_id = args[args.index("--run-id") + 1]
            ledger = self._ledger()
            ledger["records"].append({"run_id": run_id, "status": "pending"})
            self._save_ledger(ledger)
            return self._ok("")
        if "record-run" in joined:
            run_id = args[args.index("--run-id") + 1]
            status = args[args.index("--status") + 1] if "--status" in args else "keep"
            ledger = self._ledger()
            for record in ledger["records"]:
                if record["run_id"] == run_id:
                    record["status"] = status
            self._save_ledger(ledger)
            return self._ok("")
        if "resolve-unevaluated" in joined:
            run_id = args[args.index("--run-id") + 1]
            ledger = self._ledger()
            for record in ledger["records"]:
                if record["run_id"] == run_id:
                    record["status"] = "unevaluated"
            self._save_ledger(ledger)
            return self._ok("")
        if "resolve-aborted" in joined:
            run_id = args[args.index("--run-id") + 1]
            ledger = self._ledger()
            for record in ledger["records"]:
                if record["run_id"] == run_id:
                    record["status"] = "aborted"
            self._save_ledger(ledger)
            return self._ok("")
        if "set-phase" in joined:
            ledger = self._ledger()
            phase = args[args.index("--phase") + 1]
            run_state = {"phase": phase}
            if "--stop-condition" in args:
                run_state["active_stop_condition"] = \
                    args[args.index("--stop-condition") + 1]
            elif phase == "completed":
                # Mirrors the real default for the fake's capped framework_cfg.
                run_state["active_stop_condition"] = "evaluation_budget_reached"
            else:
                run_state["active_stop_condition"] = "none"
            ledger["run_state"] = run_state
            self._save_ledger(ledger)
            return self._ok("")
        if "new_candidate.py" in joined:
            run_id = args[4]  # ["python", script, task, tag, run_id, ...]
            (self.run_dir / "candidates" / run_id).mkdir(parents=True,
                                                         exist_ok=True)
            if "--provided-baseline" in args:
                (self.run_dir / "candidates" / run_id / "train.py").write_text(
                    (self.repo / "tasks/fake-task/train.py").read_text())
            return self._ok("")
        return self._ok("{}")

    def evaluate(self, role, ctx, request, *, repo_root):
        assert role == "driver" and request["kind"] == "warmstart"
        candidate = ctx.run_dir / "candidates" / ctx.run_id
        (candidate / "tune_report.json").write_text(json.dumps({
            "phase_a": {"status": "ok", "best_warm_score": 0.1}}))
        return {"kind": "warmstart", "run_id": ctx.run_id, "returncode": 0}

    @staticmethod
    def _ok(stdout: str):
        return subprocess.CompletedProcess([], 0, stdout, "")


def writer_effect(ctx):
    target = ctx.run_dir / "candidates" / str(ctx.run_id) / "train.py"
    if not ctx.extra.get("expect"):
        target.write_text("# implemented\n")
    (target.parent / "_warm_configs.json").write_text('[{"x": 1}]')
    (target.parent / "_search_space.json").write_text('{"x": ["int", 1, 3]}')
    (target.parent / "_candidate_brief.json").write_text('{"source_run_ids": []}')


def write_background(run_dir: Path) -> None:
    """Minimal background artifacts the in-process faithfulness gate accepts.

    The gate must be able to load the registry; an evidence-free registry
    yields an empty audit sample ("no_mappings"), so no judge invocation
    enters the scripted session sequence.
    """
    registry = {
        "schema_version": 3,
        "kind": "semantic_search_space",
        "space_id": "fake-loop-space",
        "dimensions": [],
        "relations": [],
        "guidance": [],
        "sources": [],
    }
    (run_dir / "background.md").write_text(
        "# bg\n\n## Search space registry\n```json\n"
        + json.dumps(registry)
        + "\n```\n"
    )
    (run_dir / "background_retrieval.json").write_text(
        json.dumps({"schema_version": 4, "rounds": [], "visits": []})
    )


def write_audited_background(run_dir: Path) -> None:
    """Background artifacts carrying real claim mappings the gate will audit."""
    (run_dir / "background.md").write_text(background_text(fixture_registry()))
    (run_dir / "background_retrieval.json").write_text(
        json.dumps(retrieval_hit_manifest())
    )


def judge_entry(unfaithful_ids: set[str]) -> dict:
    """Script entry for the faithfulness judge: one verdict per presented
    label, ``unfaithful`` for the named item ids, ``faithful`` otherwise."""
    entry: dict = {"receipt": {}}

    def effect(ctx) -> None:
        verdicts = []
        label = None
        for line in ctx.inline_payload.splitlines():
            match = re.fullmatch(r"## (M\d+)", line)
            if match:
                label = match.group(1)
            elif line.startswith("- item: ") and label is not None:
                ids = re.findall(r"`([^`]+)`", line)
                verdict = (
                    "unfaithful"
                    if any(item_id in unfaithful_ids for item_id in ids)
                    else "faithful"
                )
                verdicts.append({"mapping": label, "verdict": verdict,
                                 "rationale": f"{label} judged"})
                label = None
        entry["receipt"] = {"verdicts": verdicts}

    entry["side_effects"] = effect
    return entry


class ValidatorErrorMessageTests(unittest.TestCase):
    def test_stdout_json_errors_win_over_empty_stderr(self) -> None:
        result = subprocess.CompletedProcess(
            [], 1,
            json.dumps({"ok": False, "errors": [
                "hypothesis hyp-x number 0.839 is missing",
            ]}),
            "",
        )
        self.assertEqual(
            _validator_error_messages(result),
            ["hypothesis hyp-x number 0.839 is missing"],
        )

    def test_stderr_is_the_fallback_when_stdout_is_not_named_json(self) -> None:
        result = subprocess.CompletedProcess([], 1, "", "boom")
        self.assertEqual(_validator_error_messages(result), ["boom"])
        empty = subprocess.CompletedProcess([], 1, "", "")
        self.assertEqual(_validator_error_messages(empty), ["validation failed"])


class ExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_seedless_full_round_then_budget_completion(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, True]   # ideation check, pre-tuner check, step-0 check
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
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd,
                       semantic_policy="coverage_attempt",
                       scheduler_policy="v3_2")
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "completed")
        roles = [name for name, _ in runner.calls]
        self.assertEqual(roles, ["background-researcher", "idea-generator",
                                 "candidate-writer",
                                 "tuner-orchestrator"])
        self.assertEqual(
            sum("background_contract.py validate" in call for call in cmd.calls),
            1,
        )
        init_call = next(call for call in cmd.calls if "init_run.py" in call)
        self.assertIn("--semantic-policy coverage_attempt", init_call)
        self.assertIn("--scheduler-policy v3_2", init_call)
        self.assertNotIn("--inner-tuner-policy", init_call)

    def test_candidate_writer_retry_is_decorrelated_and_persistent(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, True]
        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: write_background(ctx.run_dir)},
            {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
             "side_effects": lambda ctx: cmd([
                 "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                 self.repo)},
            {"fail": ["writer repetition breaker tripped"]},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        writer_calls = [ctx for name, ctx in runner.calls
                        if name == "candidate-writer"]
        self.assertEqual(len(writer_calls), 2)
        self.assertNotIn("retry_note", writer_calls[0].extra)
        self.assertIn("retry_note", writer_calls[1].extra)
        self.assertIn("writer repetition breaker tripped",
                      writer_calls[1].extra["retry_note"])
        self.assertIn("current write mode", writer_calls[1].extra["retry_note"])
        self.assertNotIn("each read once", writer_calls[1].extra["retry_note"])
        attempts = self.repo / "runs" / "fake-task" / "t1" / "candidates" / "000" / "writer.attempts.json"
        self.assertEqual(json.loads(attempts.read_text())["attempts"], 2)

    def test_tuner_contradiction_corrected_in_same_session(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, True]   # ideation check, pre-tuner check, step-0 check
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
            # contradiction: claims tuned 000 but the ledger has no tune flag
            {"receipt": {"tuned_run_id": "000", "tuned": True,
                         "ledger_updated": True}},
            # same-session corrective follow-up still needs one driver job
            {"receipt": {"tuned_run_id": "000", "tuned": False,
                         "ledger_updated": False,
                         "driver_job": {"kind": "phase_c", "run_id": "000",
                                        "method": "bo", "trial_cap": 10}}},
            {"receipt": {"tuned_run_id": "000", "tuned": True,
                         "ledger_updated": True}},
        ])
        jobs = []

        def job_runner(role, ctx, request, *, repo_root):
            if request["kind"] == "warmstart":
                return cmd.evaluate(role, ctx, request, repo_root=repo_root)
            jobs.append(request)
            ledger = cmd._ledger()
            ledger["records"][0]["tune"] = True
            cmd._save_ledger(ledger)
            return {"kind": "phase_c", "run_id": "000", "returncode": 0}

        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd, job_runner=job_runner)
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "completed")
        self.assertEqual(len(jobs), 1)
        tuner_calls = [ctx for name, ctx in runner.calls
                       if name == "tuner-orchestrator"]
        self.assertEqual(len(tuner_calls), 3)
        self.assertNotIn("reconcile_note", tuner_calls[0].extra)
        self.assertIn("reconcile_note", tuner_calls[1].extra)
        self.assertIn("driver_job_result", tuner_calls[2].extra)
        self.assertEqual(
            tuner_calls[1].resume_session_id,
            f"fake-sess-{tuner_calls[0].invocation_id:04d}")
        self.assertEqual(
            tuner_calls[2].resume_session_id,
            f"fake-sess-{tuner_calls[1].invocation_id:04d}")

    def test_tuner_contradiction_can_correct_to_truthful_noop(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, True]
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
            {"receipt": {"tuned_run_id": "000", "tuned": True,
                         "ledger_updated": True}},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "completed")
        tuner_calls = [ctx for name, ctx in runner.calls
                       if name == "tuner-orchestrator"]
        self.assertEqual(len(tuner_calls), 2)
        self.assertEqual(
            tuner_calls[1].resume_session_id,
            f"fake-sess-{tuner_calls[0].invocation_id:04d}")

    def test_refresh_runs_before_ideation(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        (cmd.run_dir / "ledger.json").parent.mkdir(parents=True, exist_ok=True)
        write_background(cmd.run_dir)
        (cmd.run_dir / "framework_cfg.json").write_text(json.dumps(
            {"max_evaluations": 3, "dimension_strategy": "catalog_subset"}))
        cmd._save_ledger({"records": [{"run_id": "000", "status": "keep"}]})
        cmd.brief_queue = [
            {"records": [{"run_id": "000", "status": "keep"}],
             "experience_refresh_required": True},
        ]
        cmd.reached = [False, True, True]
        runner = FakeSessionRunner([
            {"receipt": {"updates": []}},
            {"receipt": {"actions": []}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        roles = [name for name, _ in runner.calls]
        self.assertEqual(roles[:2], ["experience-extractor", "idea-generator"])

    def test_provided_baseline_admitted_before_loop(self) -> None:
        write_task(self.repo, provided=True)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [True]  # budget already exhausted at step 0
        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: write_background(ctx.run_dir)},
            {"receipt": {"status": "existing", "wrote": False,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        ledger = cmd._ledger()
        self.assertEqual(ledger["records"][0]["run_id"], "000")
        add_call = next(call for call in cmd.calls if "add-record" in call)
        self.assertIn("--role task_provided_baseline", add_call)
        roles = [name for name, _ in runner.calls]
        self.assertNotIn("idea-generator", roles)

    def test_warmstart_budget_exhausted_resolves_unevaluated(self):
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.run_dir.mkdir(parents=True)
        candidate = cmd.run_dir / "candidates/001"
        candidate.mkdir(parents=True)
        cmd._save_ledger({"records": [{"run_id": "001", "status": "pending"}]})
        (cmd.run_dir / "framework_cfg.json").write_text("{}")
        runner = FakeSessionRunner([{"receipt": {"status": "written", "wrote": True,
             "candidate_dir": str(candidate)}, "side_effects": writer_effect}])
        experiment._implement_candidate(
            runner, ReceiptStore(cmd.run_dir), "fake-task", "t1", cmd.run_dir,
            "001", self.repo, cmd, EventsLog(cmd.run_dir),
            job_runner=lambda *a, **k: {"returncode": 4})
        self.assertEqual(cmd._ledger()["records"][0]["status"], "unevaluated")
        self.assertEqual(len(runner.calls), 1)

    def test_warmstart_failure_resumes_author_and_rechecks(self):
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        run = cmd.run_dir
        candidate = run / "candidates/001"
        candidate.mkdir(parents=True)
        cmd._save_ledger({"records": [{"run_id": "001", "status": "pending"}]})
        (cmd.run_dir / "framework_cfg.json").write_text("{}")
        def repair(ctx):
            (candidate / "train.py").write_text("# repaired\n")
        runner = FakeSessionRunner([
            {"receipt": {"status": "written", "wrote": True, "candidate_dir": str(candidate)},
             "side_effects": writer_effect},
            {"receipt": {"status": "written", "wrote": True, "candidate_dir": str(candidate)},
             "side_effects": repair}])
        calls = []
        def evaluate(role, ctx, request, **kw):
            calls.append(request)
            if len(calls) == 1:
                (candidate / "tune_report.json").write_text('{"phase_a": {"status": "crashed"}}')
                return {"returncode": 3, "log_tail": "invalid configuration"}
            return cmd.evaluate(role, ctx, request, **kw)
        experiment._implement_candidate(runner, ReceiptStore(run), "fake-task", "t1", run,
                                         "001", self.repo, cmd, EventsLog(run), job_runner=evaluate)
        self.assertEqual(cmd._ledger()["records"][0]["status"], "keep")
        self.assertEqual([name for name, ctx in runner.calls], ["candidate-writer"] * 2)
        self.assertEqual(runner.calls[1][1].resume_session_id, "fake-sess-0001")
        self.assertIn("invalid configuration", runner.calls[1][1].extra["repair_feedback"])
        self.assertEqual(len(calls), 2)

    def _seed_resumed_run(self, records: list[dict]) -> None:
        """A run killed after setup: cfg/background exist, no metadata."""
        write_task(self.repo)
        run_dir = self.repo / "runs" / "fake-task" / "t1"
        run_dir.mkdir(parents=True)
        (run_dir / "framework_cfg.json").write_text(json.dumps(
            {"max_evaluations": 3, "dimension_strategy": "catalog_subset"}))
        write_background(run_dir)
        (run_dir / "ledger.json").write_text(json.dumps({"records": records}))

    def test_pending_record_resumed_in_place_before_ideation(self) -> None:
        self._seed_resumed_run([{"run_id": "000", "status": "pending"}])
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, False, True]
        runner = FakeSessionRunner([
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"actions": []}},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        roles = [name for name, _ in runner.calls]
        # the pending record is implemented BEFORE any idea-generator run
        self.assertEqual(roles, ["candidate-writer",
                                 "idea-generator", "tuner-orchestrator"])
        writer_ctx = runner.calls[0][1]
        self.assertEqual(writer_ctx.run_id, "000")
        self.assertTrue(any("new_candidate.py" in c and " 000 " in c
                            for c in cmd.calls))
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "completed")

    def test_quiescent_rounds_complete_instead_of_spinning(self) -> None:
        self._seed_resumed_run([{"run_id": "000", "status": "keep"}])
        cmd = ExperimentCmd(self.repo)  # budget never reached
        runner = FakeSessionRunner([
            {"receipt": {"actions": []}},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
            {"receipt": {"actions": []}},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        status = run_experiment("fake-task", "t1", runner=runner, model="m",
                                repo_root=self.repo, cmd=cmd)
        roles = [name for name, _ in runner.calls]
        # exactly two zero-progress rounds, then normal completion
        self.assertEqual(roles, ["idea-generator", "tuner-orchestrator"] * 2)
        self.assertEqual(
            cmd._ledger().get("run_state", {}).get("phase"), "completed")
        # The persisted early-stop completion is what the loop returns:
        # phase completed with the actual stop cause, not a re-derived
        # "running" (the delivery-gap regression).
        self.assertEqual(status["phase"], "completed")
        self.assertEqual(status["stop_condition"], "quiescent")
        self.assertEqual(
            sum("background_contract.py validate" in call for call in cmd.calls),
            1,
        )

    def test_resume_resets_stale_blocked_phase(self) -> None:
        self._seed_resumed_run([{"run_id": "000", "status": "keep"}])
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [True]  # budget exhausted: completes immediately
        # a previously blocked run carries run_state.phase = "blocked"
        import json as _json
        ledger_path = self.repo / "runs" / "fake-task" / "t1" / "ledger.json"
        ledger = _json.loads(ledger_path.read_text())
        ledger["run_state"] = {"phase": "blocked"}
        ledger_path.write_text(_json.dumps(ledger))
        runner = FakeSessionRunner([])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        set_phases = [c for c in cmd.calls if "set-phase" in c]
        self.assertTrue(any("--phase running" in c for c in set_phases),
                        set_phases)

    def test_resume_resets_stale_completed_phase(self) -> None:
        self._seed_resumed_run([{"run_id": "000", "status": "keep"}])
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [True]  # budget exhausted: completes immediately
        import json as _json
        ledger_path = self.repo / "runs" / "fake-task" / "t1" / "ledger.json"
        ledger = _json.loads(ledger_path.read_text())
        ledger["run_state"] = {"phase": "completed",
                               "active_stop_condition": "quiescent"}
        ledger_path.write_text(_json.dumps(ledger))
        runner = FakeSessionRunner([])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        set_phases = [c for c in cmd.calls if "set-phase" in c]
        # A resumed run does not carry its stale completion through the new
        # session; it ends completed again through its own guard.
        self.assertTrue(any("--phase running" in c for c in set_phases),
                        set_phases)
        self.assertEqual(
            cmd._ledger().get("run_state", {}).get("phase"), "completed")

    def test_set_phase_completed_refusal_blocks_cleanly(self) -> None:
        self._seed_resumed_run([{"run_id": "000", "status": "keep"}])
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [True]
        cmd.raise_once = {"phase completed"}  # set-phase completed refused once
        runner = FakeSessionRunner([])
        status = run_experiment("fake-task", "t1", runner=runner, model="m",
                                repo_root=self.repo, cmd=cmd)
        self.assertEqual(
            cmd._ledger().get("run_state", {}).get("phase"), "blocked")
        self.assertEqual(status["phase"], "blocked")
        events = (cmd.run_dir / "driver_events.jsonl").read_text()
        self.assertIn("set-phase completed refused", events)

    def test_tuner_first_failure_then_contradiction_no_unbound_local(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, True]
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
            {"fail": ["tuner session died"]},
            # fresh reconciliation returns a CONTRADICTORY receipt (ledger has
            # no tune flag for 000); with no live tuner session the corrective
            # follow-up must run fresh (resume_from=None), not crash
            {"receipt": {"tuned_run_id": "000", "tuned": True,
                         "ledger_updated": True}},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "completed")
        tuner_calls = [ctx for name, ctx in runner.calls
                       if name == "tuner-orchestrator"]
        self.assertEqual(len(tuner_calls), 3)
        self.assertIsNone(tuner_calls[1].resume_session_id)
        self.assertIsNone(tuner_calls[2].resume_session_id)
        self.assertIn("reconcile_note", tuner_calls[1].extra)
        self.assertIn("tuner session died",
                      tuner_calls[1].extra["reconcile_note"])

    def test_provided_baseline_unrecoverable_failure_blocks(self):
        write_task(self.repo, provided=True)
        cmd = ExperimentCmd(self.repo)
        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: write_background(ctx.run_dir)},
            {"receipt": {"status": "existing", "wrote": False, "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"status": "abandon", "wrote": False, "candidate_dir": "candidates/000",
                         "reason": "cannot adapt the original baseline"}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m", repo_root=self.repo,
                       cmd=cmd, job_runner=lambda *a, **k: {"returncode": 3})
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "blocked")
        self.assertNotIn("idea-generator", [name for name, _ in runner.calls])
        self.assertIn("cannot adapt", (cmd.run_dir / "driver_events.jsonl").read_text())
        self.assertEqual(cmd._ledger()["records"][0]["status"], "aborted")

    def test_phase_c_recover_close_reports_refusal_detail(self) -> None:
        run_dir = self.repo / "runs" / "fake-task" / "t1"
        candidate = run_dir / "candidates" / "000"
        candidate.mkdir(parents=True)

        def refusing_cmd(args, repo_root, check=True, capture=True, **kw):
            return subprocess.CompletedProcess(
                [], 1, "", "stage still running: remaining > 0")

        emitted = []

        class RecordingEvents:
            def emit(self, kind, **fields):
                emitted.append((kind, fields))

        result = _phase_c_recover_close(
            run_dir, "000", self.repo, refusing_cmd, RecordingEvents(),
            "fake-task")
        self.assertIsNone(result)
        self.assertEqual(emitted[0][0], "tuning_driver_finalize_deferred")
        self.assertIn("phase-c-action refused", emitted[0][1]["reason"])
        self.assertIn("stage still running", emitted[0][1]["reason"])

    def test_resume_missing_background_reruns_researcher(self) -> None:
        self._seed_resumed_run([{"run_id": "000", "status": "keep"}])
        run_dir = self.repo / "runs" / "fake-task" / "t1"
        (run_dir / "background.md").unlink()  # killed mid-setup
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [True]
        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: write_background(ctx.run_dir)},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        roles = [name for name, _ in runner.calls]
        self.assertEqual(roles, ["background-researcher"])
        self.assertTrue((run_dir / "background.md").exists())
        self.assertTrue((run_dir / "run_metadata.json").exists())

    def test_preseeded_background_skips_researcher(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, True]
        cmd.run_dir.mkdir(parents=True, exist_ok=True)
        write_background(cmd.run_dir)
        frozen_text = (cmd.run_dir / "background.md").read_text()
        runner = FakeSessionRunner([
            {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
             "side_effects": lambda ctx: cmd([
                 "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                 self.repo)},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "completed")
        roles = [name for name, _ in runner.calls]
        self.assertEqual(roles, ["idea-generator", "candidate-writer",
                                 "tuner-orchestrator"])
        # the frozen artifacts are validated once and left untouched
        self.assertEqual(
            sum("background_contract.py validate" in call
                for call in cmd.calls),
            1)
        self.assertEqual((cmd.run_dir / "background.md").read_text(),
                         frozen_text)

    def test_generation_validation_repair_receives_stdout_json_errors(self) -> None:
        # Validators print named errors on stdout JSON and leave stderr empty.
        # The repair extra and the block reason must carry those strings, not
        # the collapsed fallback "validation failed".
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        named = (
            "hypothesis hyp-glove-embedding number 0.839 is in no cited "
            "source's retained content at preview tier or better "
            "(candidates: src-02); visit a cited source containing the "
            "number, cite a different source that carries it, or downgrade "
            "the claim to a qualitative statement"
        )
        payload = json.dumps({"ok": False, "errors": [named]})
        cmd.fail_payload = {
            "background_contract.py validate": (1, payload, ""),
        }
        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: write_background(ctx.run_dir)},
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "blocked")
        self.assertEqual(
            [name for name, _ in runner.calls],
            ["background-researcher", "background-researcher"],
        )
        extra = runner.calls[1][1].extra
        self.assertIn("0.839", extra["validation_errors"])
        self.assertIn("hyp-glove-embedding", extra["validation_errors"])
        self.assertNotIn("validation failed", extra["validation_errors"])
        events = (cmd.run_dir / "driver_events.jsonl").read_text()
        self.assertIn("0.839", events)
        self.assertNotIn("['validation failed']", events)

    def test_invalid_preseeded_background_blocks_without_repair(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.run_dir.mkdir(parents=True, exist_ok=True)
        (cmd.run_dir / "background.md").write_text("# frozen\n")
        (cmd.run_dir / "background_retrieval.json").write_text("{}")
        cmd.fail_next = {"background_contract.py validate"}
        runner = FakeSessionRunner([])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "blocked")
        self.assertEqual(runner.calls, [])
        self.assertEqual((cmd.run_dir / "background.md").read_text(),
                         "# frozen\n")

    def test_preseeded_unfaithful_background_records_warning(self) -> None:
        # repairable=False (no researcher receipt): unfaithful findings are
        # recorded as a terminal warning, not a block — the frozen background
        # cannot be rewritten, and blocking would dead-loop every resume.
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, True]
        cmd.run_dir.mkdir(parents=True, exist_ok=True)
        write_audited_background(cmd.run_dir)
        frozen_text = (cmd.run_dir / "background.md").read_text()
        runner = FakeSessionRunner([
            judge_entry({"hyp-data-filtered"}),
            {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
             "side_effects": lambda ctx: cmd([
                 "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                 self.repo)},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "completed")
        roles = [name for name, _ in runner.calls]
        self.assertEqual(roles, ["background-faithfulness-judge",
                                 "idea-generator", "candidate-writer",
                                 "tuner-orchestrator"])
        artifact = json.loads(
            (cmd.run_dir / "background_faithfulness.json").read_text())
        self.assertEqual(artifact["rounds"][-1]["outcome"],
                         "unfaithful_irreparable_warning")
        self.assertTrue(audit_completed(cmd.run_dir))  # terminal: no re-audit
        events = [json.loads(line) for line in
                  (cmd.run_dir / "driver_events.jsonl").read_text().splitlines()]
        audit_events = [row for row in events
                        if row.get("kind") == "background_faithfulness_audit"]
        self.assertEqual(audit_events[-1]["outcome"],
                         "unfaithful_irreparable_warning")
        self.assertTrue(audit_events[-1]["findings"])
        self.assertEqual((cmd.run_dir / "background.md").read_text(),
                         frozen_text)

    def test_resume_kill_window_audit_gets_repair_round(self) -> None:
        # Killed after the researcher wrote its files but before the audit:
        # the init-time session file proves the researcher ran, so the resume
        # audit is repairable and unfaithful findings get a repair round
        # instead of a permanent block.
        write_task(self.repo)
        run_dir = self.repo / "runs" / "fake-task" / "t1"
        run_dir.mkdir(parents=True)
        (run_dir / "framework_cfg.json").write_text(json.dumps(
            {"max_evaluations": 3, "dimension_strategy": "catalog_subset"}))
        write_audited_background(run_dir)
        (run_dir / "ledger.json").write_text(
            json.dumps({"records": [{"run_id": "000", "status": "keep"}]}))
        receipts = run_dir / "receipts"
        receipts.mkdir()
        (receipts / "background-researcher-0000.session.json").write_text(
            json.dumps({"session_id": "sess-killed"}))
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [True]
        runner = FakeSessionRunner([
            judge_entry({"hyp-data-filtered"}),  # attempt 1: unfaithful
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"}},
            judge_entry(set()),                    # attempt 2: all faithful
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "completed")
        roles = [name for name, _ in runner.calls]
        self.assertEqual(roles, ["background-faithfulness-judge",
                                 "background-researcher",
                                 "background-faithfulness-judge"])
        self.assertIn("faithfulness_findings", runner.calls[1][1].extra)
        artifact = json.loads(
            (run_dir / "background_faithfulness.json").read_text())
        self.assertEqual([r["outcome"] for r in artifact["rounds"]],
                         ["unfaithful", "passed"])

    def test_prepare_runs_in_task_working_dir(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, True]
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
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        idx = cmd.calls.index("uv run python prepare.py")
        self.assertEqual(cmd.cwds[idx], self.repo / "tasks" / "fake-task")

    def test_prepare_failure_blocks_run(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.raise_once = {"prepare.py"}
        runner = FakeSessionRunner([])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "blocked")
        events = (cmd.run_dir / "driver_events.jsonl").read_text()
        self.assertIn("prepare_command failed", events)

    def _seed_resumed_provided_run(self, ledger: dict | None) -> Path:
        """Killed after init_run/background but before add-record 000."""
        write_task(self.repo, provided=True)
        run_dir = self.repo / "runs" / "fake-task" / "t1"
        run_dir.mkdir(parents=True)
        (run_dir / "framework_cfg.json").write_text(json.dumps(
            {"max_evaluations": 3, "dimension_strategy": "catalog_subset"}))
        write_background(run_dir)
        if ledger is not None:
            (run_dir / "ledger.json").write_text(json.dumps(ledger))
        return run_dir

    def test_resume_reconciles_missing_provided_baseline(self) -> None:
        # resume-shaped run with a provided seed and NO ledger: the baseline
        # must be admitted before any ideation, not skipped
        self._seed_resumed_provided_run(ledger=None)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [True]  # budget exhausted at the first step-0 check
        runner = FakeSessionRunner([
            {"receipt": {"status": "existing", "wrote": False,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger()["records"][0]["run_id"], "000")
        roles = [name for name, _ in runner.calls]
        self.assertNotIn("idea-generator", roles)
        self.assertEqual(roles, ["candidate-writer"])

    def test_resume_blocks_when_baseline_record_lost(self) -> None:
        # ledger has records but no 000: retrofitting the control is
        # forbidden, so the run blocks instead of ideating on
        run_dir = self._seed_resumed_provided_run(
            ledger={"records": [{"run_id": "001", "status": "keep"}]})
        cmd = ExperimentCmd(self.repo)
        runner = FakeSessionRunner([])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("run_state", {}).get("phase"), "blocked")
        events = (run_dir / "driver_events.jsonl").read_text()
        self.assertIn("retrofitting the control", events)
        roles = [name for name, _ in runner.calls]
        self.assertNotIn("idea-generator", roles)

    def test_resume_does_not_forward_time_budget_when_deadline_exists(self) -> None:
        self._seed_resumed_run([{"run_id": "000", "status": "keep"}])
        cfg_path = self.repo / "runs" / "fake-task" / "t1" / "framework_cfg.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["deadline"] = 1_800_000_000
        cfg_path.write_text(json.dumps(cfg))
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [True]
        runner = FakeSessionRunner([])
        run_experiment(
            "fake-task", "t1", runner=runner, model="m",
            repo_root=self.repo, cmd=cmd, time_budget=3600,
            max_evaluations=3,
        )
        init_calls = [c for c in cmd.calls if "init_run.py" in c]
        self.assertTrue(init_calls)
        for call in init_calls:
            self.assertNotIn("--time-budget", call)
            self.assertNotIn("--deadline", call)
            self.assertIn("--max-evaluations 3", call)


class CandidateRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        write_task(self.repo)
        self.run = self.repo / "runs/fake-task/t1"
        self.candidate = self.run / "candidates/001"
        self.candidate.mkdir(parents=True)
        (self.run / "framework_cfg.json").write_text('{"max_evaluations": 10}')
        self.cmd = ExperimentCmd(self.repo)
        self.cmd._save_ledger({"records": [{"run_id": "001", "status": "pending"}]})
        experiment._reset_block_state(self.run)
        experiment._reset_seat_skip_state(self.run)
        self.store = ReceiptStore(self.run)
        self.events = EventsLog(self.run)

    def implement(self, runner):
        experiment._implement_candidate(
            runner, self.store, "fake-task", "t1", self.run, "001",
            self.repo, self.cmd, self.events, job_runner=self.cmd.evaluate)

    def test_valid_report_settles_before_any_failed_session_can_run(self):
        (self.candidate / "tune_report.json").write_text(
            '{"phase_a": {"status": "ok", "best_warm_score": 0.2161099781414916}}')
        runner = FakeSessionRunner([])
        self.implement(runner)
        self.assertEqual(self.cmd._ledger()["records"][0]["status"], "keep")
        self.assertEqual(runner.calls, [])

    def test_blocked_writer_retry_preserves_its_attempt_for_resume(self):
        def sibling_blocks(ctx):
            with self.assertRaises(RunBlocked):
                experiment._or_block(
                    self.run, self.repo, self.cmd, self.events, "sibling blocked")

        first = FakeSessionRunner([
            {"fail": ["no accepted receipt"], "side_effects": sibling_blocks},
        ])
        with self.assertRaisesRegex(RunBlocked, "sibling blocked"):
            self.implement(first)
        self.assertEqual(json.loads((self.candidate / "writer.attempts.json")
                                   .read_text())["attempts"], 1)
        self.assertEqual(self.cmd._ledger()["records"][0]["status"], "pending")

        experiment._reset_block_state(self.run)

        def settled(ctx):
            self.cmd(["python", "tools/ledger.py", "record-run", "--run-id", "001"],
                     self.repo)

        resumed = FakeSessionRunner([
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/001"}, "side_effects": writer_effect},
        ])
        self.implement(resumed)
        self.assertEqual(json.loads((self.candidate / "writer.attempts.json")
                                   .read_text())["attempts"], 2)
        self.assertEqual(self.cmd._ledger()["records"][0]["status"], "keep")

    def test_resume_skips_a_writer_that_already_delivered(self):
        first = FakeSessionRunner([
            {"fail": ["no accepted receipt"]},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/001"}, "side_effects": writer_effect}])
        def interrupted(*args, **kwargs):
            raise RuntimeError("process interrupted before evaluation")
        with self.assertRaisesRegex(RuntimeError, "process interrupted"):
            experiment._implement_candidate(first, self.store, "fake-task", "t1", self.run,
                "001", self.repo, self.cmd, self.events, job_runner=interrupted)
        self.assertEqual(json.loads((self.candidate / "writer.attempts.json").read_text())["attempts"], 2)
        resumed = FakeSessionRunner([])
        self.implement(resumed)
        self.assertEqual(resumed.calls, [])
        self.assertEqual(self.cmd._ledger()["records"][0]["status"], "keep")


class SeatSkipStreakTests(unittest.TestCase):
    """The run-level consecutive-failure breaker (P3): isolated skips settle
    as skips; a streak of isomorphic skips (same role + frozen problem
    class) blocks the run."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        write_task(self.repo)
        self.run_dir = self.repo / "runs" / "fake-task" / "t1"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "framework_cfg.json").write_text(
            json.dumps({"max_evaluations": 3}))
        self.cmd = ExperimentCmd(self.repo)
        self.events = EventsLog(self.run_dir)
        _reset_seat_skip_state(self.run_dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _note(self, role: str, problems: list[str]) -> None:
        _note_seat_skip(self.run_dir, self.repo, self.cmd, self.events,
                        role=role, problems=problems, run_id="001")

    def test_two_isomorphic_skips_trip_and_block(self) -> None:
        self._note("candidate-writer", ["writer down"])
        with self.assertRaises(RunBlocked):
            self._note("candidate-writer", ["writer still down"])
        self.assertEqual(
            self.cmd._ledger().get("run_state", {}).get("phase"), "blocked")
        self.assertIn("consecutive seat failures",
                      self.cmd._ledger()["run_state"]["active_stop_condition"])
        self.assertIn("problem_class=postcondition",
                      self.cmd._ledger()["run_state"]["active_stop_condition"])

    def test_success_resets_and_other_signatures_do_not_accumulate(self) -> None:
        # Same role, different frozen problem classes: separate streaks.
        self._note("candidate-writer",
                   ["session ended with error result: error_max_turns"])
        self._note("candidate-writer", ["writer down"])
        # A completed seat resets the running streak.
        _note_seat_progress(self.run_dir)
        self._note("candidate-writer", ["writer down"])
        # Different roles never share a streak.
        self._note("slate-plan-writer", ["planner down"])
        self.assertIsNone(self.cmd._ledger().get("run_state"))

    def test_successful_ideation_resets_the_failure_streak(self) -> None:
        self._note("idea-generator", ["generator unavailable"])
        actions = experiment._ideate(
            FakeSessionRunner([{"receipt": {"actions": []}}]),
            ReceiptStore(self.run_dir), "fake-task", "t1", self.run_dir, 0,
            self.repo, self.cmd, self.events,
            task_toml=experiment.common.load_task_toml(
                "fake-task", self.repo),
        )
        self.assertEqual(actions, [])
        self._note("idea-generator", ["generator unavailable again"])
        self.assertIsNone(self.cmd._ledger().get("run_state"))


class DegradedDeliveryTests(unittest.TestCase):
    """P6: persist block, drain channels, settle, then export best-effort."""

    SUBMISSION = (
        "python -c \"open('submission.csv', 'w').write("
        "'Id,Probability\\n1,0.5\\n')\""
    )
    MALFORMED_SUBMISSION = (
        "python -c \"open('submission.csv', 'w').write('ok')\""
    )
    FAILING = "python -c \"import sys; sys.exit(3)\""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        write_task(self.repo)
        self.run_dir = self.repo / "runs" / "fake-task" / "t1"
        self.run_dir.mkdir(parents=True)
        write_background(self.run_dir)
        (self.run_dir / "framework_cfg.json").write_text(
            json.dumps({"max_evaluations": 3, "per_runtime_limit": None,
                        "dimension_strategy": "catalog_subset",
                        "deadline": time.time() + 600}))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _ledger_with(self, statuses: dict[str, str]) -> None:
        self.run_dir.joinpath("ledger.json").write_text(json.dumps({
            "records": [
                {"run_id": run_id, "status": status,
                 **({"final_best_score": 0.5} if status == "keep" else {})}
                for run_id, status in statuses.items()
            ]}))

    def _block(self, finalization, statuses=None, cmd=None) -> dict:
        self._ledger_with(statuses or {"000": "keep", "001": "pending"})
        cmd = cmd or ExperimentCmd(self.repo)
        cmd.fail_next.add("ledger.py brief")
        status = run_experiment(
            "fake-task", "t1", runner=FakeSessionRunner([]), model="m",
            repo_root=self.repo, cmd=cmd, finalization=finalization)
        self.assertEqual(status["phase"], "blocked")
        return status

    def _events(self) -> list[dict]:
        path = self.run_dir / "driver_events.jsonl"
        return [json.loads(line) for line in
                path.read_text().splitlines() if line.strip()]

    def test_block_exports_submission_and_annotates(self) -> None:
        status = self._block({"submission_command": self.SUBMISSION,
                              "data_dir": str(self.repo / "data")})
        self.assertIn("degraded_submit", status["stop_condition"])
        self.assertTrue((self.run_dir / "submission.csv").is_file())
        # The pending seat settled deterministically: zero attempts →
        # aborted (never a crash observation on an innocent point).
        ledger = json.loads((self.run_dir / "ledger.json").read_text())
        self.assertEqual(
            [r["status"] for r in ledger["records"]], ["keep", "aborted"])
        rows = self._events()
        self.assertIn("candidate_skipped",
                      [row["kind"] for row in rows])
        delivery = [row for row in rows
                    if row["kind"] == "degraded_submit"][-1]
        self.assertEqual(delivery["status"], "success")

    def test_annotation_failure_keeps_the_original_block_durable(self) -> None:
        class AnnotationFailCmd(ExperimentCmd):
            def __init__(self, repo: Path) -> None:
                super().__init__(repo)
                self.block_writes = 0

            def __call__(self, args, repo_root, **kwargs):
                joined = " ".join(str(arg) for arg in args)
                if "set-phase" in joined and "--phase blocked" in joined:
                    self.block_writes += 1
                    if self.block_writes == 2:
                        raise subprocess.CalledProcessError(
                            1, args, stderr="annotation unavailable")
                return super().__call__(args, repo_root, **kwargs)

        status = self._block(
            {"submission_command": self.SUBMISSION, "data_dir": "x"},
            cmd=AnnotationFailCmd(self.repo),
        )
        self.assertEqual(status["phase"], "blocked")
        self.assertNotIn("degraded_submit", status["stop_condition"])
        self.assertTrue((self.run_dir / "submission.csv").is_file())
        failures = [row for row in self._events()
                    if row["kind"] == "degraded_submit_annotation_failed"]
        self.assertEqual(len(failures), 1)
        self.assertIn("annotation unavailable", failures[0]["error"])

    def test_malformed_submission_is_not_annotated_as_delivered(self) -> None:
        status = self._block({
            "submission_command": self.MALFORMED_SUBMISSION,
            "data_dir": str(self.repo / "data"),
        })
        self.assertNotIn("degraded_submit", status["stop_condition"])
        delivery = [row for row in self._events()
                    if row["kind"] == "degraded_submit"][-1]
        self.assertEqual(delivery["status"], "failed")
        self.assertIn("no usable header/rows", delivery["error"])

    def test_cleanup_of_two_pending_seats_still_exports(self) -> None:
        status = self._block(
            {"submission_command": self.SUBMISSION, "data_dir": "x"},
            {"000": "keep", "001": "pending", "002": "pending"})
        self.assertIn("degraded_submit", status["stop_condition"])
        self.assertTrue((self.run_dir / "submission.csv").is_file())
        self.assertEqual(experiment._seat_skip_count(self.run_dir), 0)

    def test_settlement_exception_keeps_the_original_block_durable(self) -> None:
        with mock.patch.object(experiment, "_settle_for_delivery",
                               side_effect=RuntimeError("settlement unavailable")):
            status = self._block(
                {"submission_command": self.SUBMISSION, "data_dir": "x"})
        self.assertEqual(status["phase"], "blocked")
        self.assertIn("JSONDecodeError", status["stop_condition"])
        failures = [r for r in self._events()
                    if r["kind"] == "degraded_submit" and r["status"] == "failed"]
        self.assertEqual(len(failures), 1)
        self.assertIn("settlement unavailable", failures[0]["error"])

    def test_delivery_waits_for_inflight_evaluation_before_settling(self) -> None:
        self._ledger_with({"000": "keep", "001": "pending", "002": "pending"})
        cfg_path = self.run_dir / "framework_cfg.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["pipeline"] = {"session_concurrency": 2}
        cfg_path.write_text(json.dumps(cfg))
        run_dir = self.run_dir

        class InflightCmd(ExperimentCmd):
            def __call__(self, args, repo_root, **kw):
                if "evaluation_budget.py" in str(args):
                    return self._ok(json.dumps({"reached": False,
                        "per_candidate": [{"run_id": "002", "evals": 1}]}))
                return super().__call__(args, repo_root, **kw)

        cmd = InflightCmd(self.repo)
        started, blocker_returned = threading.Event(), threading.Event()
        events = EventsLog(run_dir)

        def implement(runner, store, task, tag, run, run_id, *args, **kw):
            if run_id == "001":
                self.assertTrue(started.wait(5))
                try:
                    experiment._or_block(run, self.repo, cmd, events,
                                         "another channel blocked")
                finally:
                    blocker_returned.set()
            else:
                started.set()
                self.assertTrue(blocker_returned.wait(5))
                (run / "candidates/002/tune_report.json").write_text(
                    '{"phase_a": {"status": "ok", "best_warm_score": 0.1}}')
                experiment._refuse_if_blocked(run)

        def resume(*args, **kw):
            experiment._implement_seats(
                None, None, "fake-task", "t1", run_dir, ["001", "002"],
                self.repo, cmd, events, None)

        with mock.patch.object(experiment, "_resume_setup", side_effect=resume), \
                mock.patch.object(experiment, "_implement_candidate", side_effect=implement):
            status = run_experiment(
                "fake-task", "t1", runner=FakeSessionRunner([]), model="m",
                repo_root=self.repo, cmd=cmd,
                finalization={"submission_command": self.SUBMISSION, "data_dir": "x"})
        self.assertEqual(status["phase"], "blocked")
        self.assertEqual([r["status"] for r in cmd._ledger()["records"]],
                         ["keep", "aborted", "keep"])
        self.assertTrue((run_dir / "submission.csv").is_file())

    def test_no_settled_incumbent_skips_the_delivery(self) -> None:
        self._ledger_with({"001": "pending"})
        cmd = ExperimentCmd(self.repo)
        cmd.fail_next.add("ledger.py brief")
        status = run_experiment(
            "fake-task", "t1", runner=FakeSessionRunner([]), model="m",
            repo_root=self.repo, cmd=cmd,
            finalization={"submission_command": self.SUBMISSION,
                          "data_dir": "x"})
        self.assertEqual(status["phase"], "blocked")
        self.assertNotIn("degraded_submit", status["stop_condition"])
        self.assertFalse((self.run_dir / "submission.csv").exists())
        delivery = [row for row in self._events()
                    if row["kind"] == "degraded_submit"][-1]
        self.assertEqual(delivery["status"], "skipped")

    def test_pending_warm_score_settles_before_incumbent_check(self) -> None:
        self._ledger_with({"001": "pending"})
        candidate_dir = self.run_dir / "candidates" / "001"
        candidate_dir.mkdir(parents=True)
        candidate_dir.joinpath("tune_report.json").write_text(
            '{"phase_a": {"status": "ok", "best_warm_score": 0.1}}')
        cmd = ExperimentCmd(self.repo)
        cmd.fail_next.add("ledger.py brief")
        status = run_experiment(
            "fake-task", "t1", runner=FakeSessionRunner([]), model="m",
            repo_root=self.repo, cmd=cmd,
            finalization={"submission_command": self.SUBMISSION,
                          "data_dir": "x"})
        self.assertEqual(status["phase"], "blocked")
        self.assertIn("degraded_submit", status["stop_condition"])
        self.assertTrue((self.run_dir / "submission.csv").is_file())
        self.assertEqual(cmd._ledger()["records"][0]["status"], "keep")

    def test_failed_export_does_not_defer_the_block(self) -> None:
        status = self._block({"submission_command": self.FAILING,
                              "data_dir": "x"})
        self.assertNotIn("degraded_submit", status["stop_condition"])
        self.assertFalse((self.run_dir / "submission.csv").exists())
        delivery = [row for row in self._events()
                    if row["kind"] == "degraded_submit"][-1]
        self.assertEqual(delivery["status"], "failed")

    def test_timeout_terminates_the_submission_process_group(self) -> None:
        started = self.repo / "descendant-started"
        survived = self.repo / "descendant-survived"
        child = self.repo / "submission-child.py"
        child.write_text(
            "from pathlib import Path\n"
            "import sys, time\n"
            "Path(sys.argv[2]).touch()\n"
            "time.sleep(0.8)\n"
            "Path(sys.argv[1]).touch()\n"
        )
        parent = self.repo / "submission-parent.py"
        parent.write_text(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2], "
            "sys.argv[3]])\n"
            "time.sleep(60)\n"
        )
        command = shlex.join([
            sys.executable, str(parent), str(child), str(survived), str(started),
        ])
        self._ledger_with({"000": "keep"})
        experiment._delivery_cfg[str(self.run_dir)] = {
            "task": "fake-task",
            "submission_command": command,
            "data_dir": str(self.repo / "data"),
        }
        self.addCleanup(experiment._reset_seat_skip_state, self.run_dir)
        now = time.time()
        cfg_path = self.run_dir / "framework_cfg.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["deadline"] = now + 60.5
        cfg_path.write_text(json.dumps(cfg))
        ticks = iter((now, now + 60.0))

        with mock.patch.object(
                experiment.time, "time",
                side_effect=lambda: next(ticks, now + 60.0)):
            annotation = experiment._attempt_degraded_delivery(
                self.run_dir, self.repo, ExperimentCmd(self.repo),
                EventsLog(self.run_dir))

        self.assertIsNone(annotation)
        delivery = [row for row in self._events()
                    if row["kind"] == "degraded_submit"][-1]
        self.assertEqual(delivery["status"], "timeout")
        self.assertTrue(started.is_file())
        time.sleep(1.0)
        self.assertFalse(survived.exists())

    def test_no_finalization_contract_no_attempt(self) -> None:
        status = self._block(None)
        self.assertNotIn("degraded_submit", status["stop_condition"])
        self.assertFalse((self.run_dir / "submission.csv").exists())
        self.assertEqual(
            [row["kind"] for row in self._events()].count("degraded_submit"),
            0)


class UnhandledExceptionBoundaryTests(unittest.TestCase):
    """Wave 0 (P1): every unhandled exception ends as a persisted block —
    never a traceback exit with the ledger still claiming running."""

    def test_bare_exception_becomes_persisted_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_task(repo)
            run_dir = repo / "runs" / "fake-task" / "t1"
            run_dir.mkdir(parents=True)
            write_background(run_dir)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": 3, "per_runtime_limit": None,
                            "dimension_strategy": "catalog_subset"}))
            (run_dir / "ledger.json").write_text(json.dumps({
                "records": [{"run_id": "000", "status": "keep",
                             "final_best_score": 0.5}]}))

            class ExplodingCmd(ExperimentCmd):
                def __init__(self, repo: Path) -> None:
                    super().__init__(repo)
                    self.fired = False

                def __call__(self, args, repo_root, check=True,
                             capture=True, **kw):
                    joined = " ".join(str(a) for a in args)
                    if not self.fired and "ledger.py brief" in joined:
                        self.fired = True
                        raise RuntimeError("scheduler store exploded")
                    return super().__call__(args, repo_root, check=check,
                                            capture=capture, **kw)

            cmd = ExplodingCmd(repo)
            status = run_experiment("fake-task", "t1",
                                    runner=FakeSessionRunner([]), model="m",
                                    repo_root=repo, cmd=cmd)
            self.assertEqual(status["phase"], "blocked")
            self.assertIn("unhandled RuntimeError",
                          status["stop_condition"])
            events_path = run_dir / "driver_events.jsonl"
            rows = [json.loads(line) for line in
                    events_path.read_text().splitlines() if line.strip()]
            unhandled = [row for row in rows
                         if row["kind"] == "unhandled_exception"]
            self.assertEqual(len(unhandled), 1)
            self.assertIn("scheduler store exploded",
                          unhandled[0]["traceback"])


if __name__ == "__main__":
    unittest.main()


class IncrementalRefreshIntegrationTests(unittest.TestCase):
    def test_failed_updates_are_nonfatal_and_not_retried_on_same_revision(self):
        from tools.validate_background import Run
        from tests.fixtures import complete_point
        registry = fixture_registry()
        for outcome in ({'fail': ['missing receipt']},
                        {'receipt': {'updates': [{'op': 'delete', 'id': 'unknown'}]}},
                        {'receipt': {'updates': []}}):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = Run(root, registry)
                run.add_record('000', 'fresh', [], complete_point(registry))
                run.record_run('000', 0.45)
                before = json.loads(run.ledger.read_text())
                runner = FakeSessionRunner([outcome])
                def cmd(args, cwd, check=True):
                    return subprocess.run([sys.executable, *map(str, args[1:])],
                                          cwd=cwd, check=check, text=True, capture_output=True)
                store = ReceiptStore(root)
                published = experiment._refresh(runner, store, 'hard-interactions', 't',
                                                root, ROOT, cmd, EventsLog(root))
                after = json.loads(run.ledger.read_text())
                self.assertEqual(published, outcome == {'receipt': {'updates': []}})
                self.assertEqual(after['records'], before['records'])
                if not published:
                    self.assertEqual(after.get('experience'), before.get('experience'))
                    self.assertEqual(after['experience_update']['status'], 'failed')
                    self.assertEqual(after['search_space_state'], before['search_space_state'])
                self.assertFalse(experiment._refresh(runner, store, 'hard-interactions', 't',
                                                    root, ROOT, cmd, EventsLog(root)))
                self.assertEqual(len(runner.calls), 1)
                self.assertNotEqual(after.get('run_state', {}).get('phase'), 'blocked')
