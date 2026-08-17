import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.loops.experiment import run_experiment  # noqa: E402
from driver.session import FakeSessionRunner  # noqa: E402


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


class ExperimentCmd:
    """Fakes tools/ helpers. brief_queue/reached_queue drive loop progress."""

    def __init__(self, repo: Path):
        self.repo = repo
        self.calls: list[str] = []
        self.cwds: list = []
        self.brief_queue: list[dict] = []
        self.reached: list[bool] = []
        self.fail_next: set[str] = set()
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
        for marker in list(self.raise_once):
            if marker in joined:
                self.raise_once.discard(marker)
                raise subprocess.CalledProcessError(
                    1, args, "", f"helper refused: {marker}")

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
        if "set-phase" in joined:
            ledger = self._ledger()
            ledger["phase"] = args[args.index("--phase") + 1]
            self._save_ledger(ledger)
            return self._ok("")
        if "new_candidate.py" in joined:
            run_id = args[4]  # ["python", script, task, tag, run_id, ...]
            (self.run_dir / "candidates" / run_id).mkdir(parents=True,
                                                         exist_ok=True)
            return self._ok("")
        return self._ok("{}")

    @staticmethod
    def _ok(stdout: str):
        return subprocess.CompletedProcess([], 0, stdout, "")


def writer_effect(ctx):
    target = ctx.run_dir / "candidates" / str(ctx.run_id) / "train.py"
    target.write_text("# implemented\n")


class ExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _extractor_side_effect(self, cmd: ExperimentCmd, status: str):
        def effect(ctx):
            ledger = cmd._ledger()
            for record in ledger["records"]:
                if record["run_id"] == ctx.run_id:
                    record["status"] = status
            cmd._save_ledger(ledger)
        return effect

    def test_seedless_full_round_then_budget_completion(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, True]   # ideation check, pre-tuner check, step-0 check
        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: (
                 (ctx.run_dir / "background.md").write_text("# bg\n"),
                 (ctx.run_dir / "background_retrieval.json").write_text("{}"))},
            {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
             "side_effects": lambda ctx: cmd([
                 "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                 self.repo)},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"run_id": "000", "status": "keep", "ledger_updated": True},
             "side_effects": self._extractor_side_effect(cmd, "keep")},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd,
                       semantic_policy="coverage_attempt",
                       scheduler_policy="v3_2")
        self.assertEqual(cmd._ledger().get("phase"), "completed")
        roles = [name for name, _ in runner.calls]
        self.assertEqual(roles, ["background-researcher", "idea-generator",
                                 "candidate-writer", "tunable-contract-extractor",
                                 "tuner-orchestrator"])
        self.assertEqual(
            sum("background_contract.py validate" in call for call in cmd.calls),
            1,
        )
        init_call = next(call for call in cmd.calls if "init_run.py" in call)
        self.assertIn("--semantic-policy coverage_attempt", init_call)
        self.assertIn("--scheduler-policy v3_2", init_call)
        self.assertNotIn("--inner-tuner-policy", init_call)

    def test_tuner_contradiction_corrected_in_same_session(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, True]   # ideation check, pre-tuner check, step-0 check
        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: (
                 (ctx.run_dir / "background.md").write_text("# bg\n"),
                 (ctx.run_dir / "background_retrieval.json").write_text("{}"))},
            {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
             "side_effects": lambda ctx: cmd([
                 "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                 self.repo)},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"run_id": "000", "status": "keep", "ledger_updated": True},
             "side_effects": self._extractor_side_effect(cmd, "keep")},
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
            jobs.append(request)
            ledger = cmd._ledger()
            ledger["records"][0]["tune"] = True
            cmd._save_ledger(ledger)
            return {"kind": "phase_c", "run_id": "000", "returncode": 0}

        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd, job_runner=job_runner)
        self.assertEqual(cmd._ledger().get("phase"), "completed")
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
             "side_effects": lambda ctx: (
                 (ctx.run_dir / "background.md").write_text("# bg\n"),
                 (ctx.run_dir / "background_retrieval.json").write_text("{}"))},
            {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
             "side_effects": lambda ctx: cmd([
                 "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                 self.repo)},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"run_id": "000", "status": "keep", "ledger_updated": True},
             "side_effects": self._extractor_side_effect(cmd, "keep")},
            {"receipt": {"tuned_run_id": "000", "tuned": True,
                         "ledger_updated": True}},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("phase"), "completed")
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
        (cmd.run_dir / "background.md").write_text("# bg\n")
        (cmd.run_dir / "background_retrieval.json").write_text("{}")
        (cmd.run_dir / "framework_cfg.json").write_text(json.dumps(
            {"max_evaluations": 3, "dimension_strategy": "catalog_subset"}))
        cmd._save_ledger({"records": [{"run_id": "000", "status": "keep"}]})
        cmd.brief_queue = [
            {"records": [{"run_id": "000", "status": "keep"}],
             "experience_refresh_required": True},
        ]
        cmd.reached = [False, True, True]
        runner = FakeSessionRunner([
            {"receipt": {"search_space_state_revision": 1, "decision_ids": []}},
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
             "side_effects": lambda ctx: (
                 (ctx.run_dir / "background.md").write_text("# bg\n"),
                 (ctx.run_dir / "background_retrieval.json").write_text("{}"))},
            {"receipt": {"status": "existing", "wrote": False,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"run_id": "000", "status": "keep", "ledger_updated": True},
             "side_effects": self._extractor_side_effect(cmd, "keep")},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        ledger = cmd._ledger()
        self.assertEqual(ledger["records"][0]["run_id"], "000")
        add_call = next(call for call in cmd.calls if "add-record" in call)
        self.assertIn("--role task_provided_baseline", add_call)
        roles = [name for name, _ in runner.calls]
        self.assertNotIn("idea-generator", roles)

    def test_extractor_failure_budget_exhausted_resolves_unevaluated(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, True, True, True]
        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: (
                 (ctx.run_dir / "background.md").write_text("# bg\n"),
                 (ctx.run_dir / "background_retrieval.json").write_text("{}"))},
            {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
             "side_effects": lambda ctx: cmd([
                 "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                 self.repo)},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"fail": ["budget exhausted mid-extraction"]},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertTrue(any("resolve-unevaluated" in c for c in cmd.calls))
        self.assertNotEqual(cmd._ledger().get("phase"), "blocked")

    def test_extractor_failure_with_evidence_resumes_failed_session(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, False, True]

        def failed_extractor(ctx):
            report = ctx.run_dir / "candidates" / str(ctx.run_id) / "tune_report.json"
            report.write_text(json.dumps({"phase_a": {"status": "failed"}}))

        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: (
                 (ctx.run_dir / "background.md").write_text("# bg\n"),
                 (ctx.run_dir / "background_retrieval.json").write_text("{}"))},
            {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
             "side_effects": lambda ctx: cmd([
                 "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                 self.repo)},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"fail": ["warm evaluation failed"],
             "side_effects": failed_extractor},
            {"receipt": {"verdict": "code_incompatible", "summary": "repair",
                         "evidence": ["tune_report.json"]}},
            {"receipt": {"run_id": "000", "status": "keep",
                         "ledger_updated": True},
             "side_effects": self._extractor_side_effect(cmd, "keep")},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])

        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)

        extractor_calls = [ctx for name, ctx in runner.calls
                           if name == "tunable-contract-extractor"]
        self.assertEqual(len(extractor_calls), 2)
        self.assertEqual(
            extractor_calls[1].resume_session_id,
            f"fake-sess-{extractor_calls[0].invocation_id:04d}",
        )
        self.assertEqual(cmd._ledger().get("phase"), "completed")

    def _seed_resumed_run(self, records: list[dict]) -> None:
        """A run killed after setup: cfg/background exist, no metadata."""
        write_task(self.repo)
        run_dir = self.repo / "runs" / "fake-task" / "t1"
        run_dir.mkdir(parents=True)
        (run_dir / "framework_cfg.json").write_text(json.dumps(
            {"max_evaluations": 3, "dimension_strategy": "catalog_subset"}))
        (run_dir / "background.md").write_text("# bg\n")
        (run_dir / "background_retrieval.json").write_text("{}")
        (run_dir / "ledger.json").write_text(json.dumps({"records": records}))

    def test_pending_record_resumed_in_place_before_ideation(self) -> None:
        self._seed_resumed_run([{"run_id": "000", "status": "pending"}])
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, False, True]
        runner = FakeSessionRunner([
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"run_id": "000", "status": "keep", "ledger_updated": True},
             "side_effects": self._extractor_side_effect(cmd, "keep")},
            {"receipt": {"actions": []}},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        roles = [name for name, _ in runner.calls]
        # the pending record is implemented BEFORE any idea-generator run
        self.assertEqual(roles, ["candidate-writer", "tunable-contract-extractor",
                                 "idea-generator", "tuner-orchestrator"])
        writer_ctx = runner.calls[0][1]
        self.assertEqual(writer_ctx.run_id, "000")
        self.assertTrue(any("new_candidate.py" in c and " 000 " in c
                            for c in cmd.calls))
        self.assertEqual(cmd._ledger().get("phase"), "completed")

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
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        roles = [name for name, _ in runner.calls]
        # exactly two zero-progress rounds, then normal completion
        self.assertEqual(roles, ["idea-generator", "tuner-orchestrator"] * 2)
        self.assertEqual(cmd._ledger().get("phase"), "completed")
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

    def test_set_phase_completed_refusal_blocks_cleanly(self) -> None:
        self._seed_resumed_run([{"run_id": "000", "status": "keep"}])
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [True]
        cmd.raise_once = {"phase completed"}  # set-phase completed refused once
        runner = FakeSessionRunner([])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("phase"), "blocked")
        events = (cmd.run_dir / "driver_events.jsonl").read_text()
        self.assertIn("set-phase completed refused", events)

    def test_tuner_first_failure_then_contradiction_no_unbound_local(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, True]
        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: (
                 (ctx.run_dir / "background.md").write_text("# bg\n"),
                 (ctx.run_dir / "background_retrieval.json").write_text("{}"))},
            {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
             "side_effects": lambda ctx: cmd([
                 "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                 self.repo)},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"run_id": "000", "status": "keep", "ledger_updated": True},
             "side_effects": self._extractor_side_effect(cmd, "keep")},
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
        self.assertEqual(cmd._ledger().get("phase"), "completed")
        tuner_calls = [ctx for name, ctx in runner.calls
                       if name == "tuner-orchestrator"]
        self.assertEqual(len(tuner_calls), 3)
        self.assertIsNone(tuner_calls[1].resume_session_id)
        self.assertIsNone(tuner_calls[2].resume_session_id)
        self.assertIn("reconcile_note", tuner_calls[1].extra)

    def test_resume_missing_background_reruns_researcher(self) -> None:
        self._seed_resumed_run([{"run_id": "000", "status": "keep"}])
        run_dir = self.repo / "runs" / "fake-task" / "t1"
        (run_dir / "background.md").unlink()  # killed mid-setup
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [True]
        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: (
                 (ctx.run_dir / "background.md").write_text("# bg\n"),
                 (ctx.run_dir / "background_retrieval.json").write_text("{}"))},
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
        (cmd.run_dir / "background.md").write_text("# frozen\n")
        (cmd.run_dir / "background_retrieval.json").write_text("{}")
        runner = FakeSessionRunner([
            {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
             "side_effects": lambda ctx: cmd([
                 "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                 self.repo)},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"run_id": "000", "status": "keep",
                         "ledger_updated": True},
             "side_effects": self._extractor_side_effect(cmd, "keep")},
            {"receipt": {"tuned_run_id": "none", "tuned": False,
                         "ledger_updated": False}},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("phase"), "completed")
        roles = [name for name, _ in runner.calls]
        self.assertEqual(roles, ["idea-generator", "candidate-writer",
                                 "tunable-contract-extractor",
                                 "tuner-orchestrator"])
        # the frozen artifacts are validated once and left untouched
        self.assertEqual(
            sum("background_contract.py validate" in call
                for call in cmd.calls),
            1)
        self.assertEqual((cmd.run_dir / "background.md").read_text(),
                         "# frozen\n")

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
        self.assertEqual(cmd._ledger().get("phase"), "blocked")
        self.assertEqual(runner.calls, [])
        self.assertEqual((cmd.run_dir / "background.md").read_text(),
                         "# frozen\n")

    def test_prepare_runs_in_task_working_dir(self) -> None:
        write_task(self.repo)
        cmd = ExperimentCmd(self.repo)
        cmd.reached = [False, False, True]
        runner = FakeSessionRunner([
            {"receipt": {"status": "ok", "background": "background.md",
                         "retrieval_manifest": "background_retrieval.json"},
             "side_effects": lambda ctx: (
                 (ctx.run_dir / "background.md").write_text("# bg\n"),
                 (ctx.run_dir / "background_retrieval.json").write_text("{}"))},
            {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
             "side_effects": lambda ctx: cmd([
                 "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                 self.repo)},
            {"receipt": {"status": "written", "wrote": True,
                         "candidate_dir": "candidates/000"},
             "side_effects": writer_effect},
            {"receipt": {"run_id": "000", "status": "keep", "ledger_updated": True},
             "side_effects": self._extractor_side_effect(cmd, "keep")},
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
        self.assertEqual(cmd._ledger().get("phase"), "blocked")
        events = (cmd.run_dir / "driver_events.jsonl").read_text()
        self.assertIn("prepare_command failed", events)

    def _seed_resumed_provided_run(self, ledger: dict | None) -> Path:
        """Killed after init_run/background but before add-record 000."""
        write_task(self.repo, provided=True)
        run_dir = self.repo / "runs" / "fake-task" / "t1"
        run_dir.mkdir(parents=True)
        (run_dir / "framework_cfg.json").write_text(json.dumps(
            {"max_evaluations": 3, "dimension_strategy": "catalog_subset"}))
        (run_dir / "background.md").write_text("# bg\n")
        (run_dir / "background_retrieval.json").write_text("{}")
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
            {"receipt": {"run_id": "000", "status": "keep", "ledger_updated": True},
             "side_effects": self._extractor_side_effect(cmd, "keep")},
        ])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger()["records"][0]["run_id"], "000")
        roles = [name for name, _ in runner.calls]
        self.assertNotIn("idea-generator", roles)
        self.assertEqual(roles, ["candidate-writer",
                                 "tunable-contract-extractor"])

    def test_resume_blocks_when_baseline_record_lost(self) -> None:
        # ledger has records but no 000: retrofitting the control is
        # forbidden, so the run blocks instead of ideating on
        run_dir = self._seed_resumed_provided_run(
            ledger={"records": [{"run_id": "001", "status": "keep"}]})
        cmd = ExperimentCmd(self.repo)
        runner = FakeSessionRunner([])
        run_experiment("fake-task", "t1", runner=runner, model="m",
                       repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("phase"), "blocked")
        events = (run_dir / "driver_events.jsonl").read_text()
        self.assertIn("retrofitting the control", events)
        roles = [name for name, _ in runner.calls]
        self.assertNotIn("idea-generator", roles)


if __name__ == "__main__":
    unittest.main()
