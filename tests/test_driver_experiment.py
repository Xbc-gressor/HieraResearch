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
        self.brief_queue: list[dict] = []
        self.reached: list[bool] = []
        self.fail_next: set[str] = set()

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
        for marker in self.fail_next:
            if marker in joined:
                self.fail_next.discard(marker)
                return subprocess.CompletedProcess(args, 1, "", "boom")

        if "init_run.py" in joined:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": 3, "per_runtime_limit": None,
                            "dimension_strategy": "catalog_subset"}))
            return self._ok("")
        if "ledger.py" in joined and "brief" in joined:
            brief = self.brief_queue.pop(0) if self.brief_queue else {
                "records": self._ledger()["records"],
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
        if "background_contract.py" in joined and "preflight" in joined:
            return self._ok(json.dumps({"action": "none"}))
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
        status = run_experiment("fake-task", "t1", runner=runner, model="m",
                                repo_root=self.repo, cmd=cmd)
        self.assertEqual(cmd._ledger().get("phase"), "completed")
        roles = [name for name, _ in runner.calls]
        self.assertEqual(roles, ["background-researcher", "idea-generator",
                                 "candidate-writer", "tunable-contract-extractor",
                                 "tuner-orchestrator"])

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
        status = run_experiment("fake-task", "t1", runner=runner, model="m",
                                repo_root=self.repo, cmd=cmd)
        self.assertTrue(any("resolve-unevaluated" in c for c in cmd.calls))
        self.assertNotEqual(cmd._ledger().get("phase"), "blocked")


if __name__ == "__main__":
    unittest.main()
