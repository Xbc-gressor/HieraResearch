import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.loops.hillclimb import run_hillclimb  # noqa: E402
from driver.session import FakeSessionRunner  # noqa: E402


def write_task(repo: Path, with_entrypoint: bool = True) -> None:
    task_dir = repo / "tasks" / "fake-task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        """
[env]
project = "tasks/fake-task"
[result]
metric = "neg_acc"
required_patterns = ["^neg_acc:", "^best_model:"]
[run]
working_dir = "tasks/fake-task"
prepare_command = "uv run python prepare.py"
[constraints]
editable_files = ["train.py"]
readonly_files = ["prepare.py"]
allow_dependencies = false
"""
    )
    (task_dir / "prepare.py").write_text("# fixed eval surface\n")
    if with_entrypoint:
        (task_dir / "train.py").write_text("# baseline implementation\n")


class FakeCmd:
    """Fakes tools/ helper + entrypoint subprocesses keyed by argv content.

    scores: queue of outcomes the entrypoint 'prints' into run.log. Each item
    is a float/None (metric value) or a dict {"score", "exit", "companion"}
    for adversarial cases (nonzero exit, missing required companion line).
    reserve_exhausted: when True, reserve exits 4.
    """

    def __init__(self, repo: Path, scores: list[float | None]):
        self.repo = repo
        self.scores = list(scores)
        self.reserved = 0
        self.calls: list[list[str]] = []
        self.cwds: list = []
        self.raise_once: set[str] = set()

    def _budget_cap(self) -> float:
        cfg = self.run_dir / "framework_cfg.json"
        if cfg.exists():
            cap = json.loads(cfg.read_text()).get("max_evaluations")
            if cap is not None:
                return cap
        return float("inf")

    @property
    def run_dir(self) -> Path:
        return self.repo / "runs" / "fake-task" / "t1"

    def __call__(self, args, repo_root, check=True, capture=True, **kw):
        self.calls.append(list(args))
        self.cwds.append(kw.get("cwd"))
        joined = " ".join(str(a) for a in args)
        run_dir = self.run_dir
        for marker in list(self.raise_once):
            if marker in joined:
                self.raise_once.discard(marker)
                raise subprocess.CalledProcessError(
                    1, args, "", f"helper refused: {marker}")

        if "init_run.py" in joined:
            run_dir.mkdir(parents=True, exist_ok=True)
            cap = 3
            if "--max-evaluations" in args:
                cap = int(args[args.index("--max-evaluations") + 1])
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": cap, "per_runtime_limit": None}))
            return self._ok("")
        if "evaluation_budget.py" in joined and "reserve" in joined:
            self.reserved += 1
            return self._ok(json.dumps({"status": "reserved"}))
        if "evaluation_budget.py" in joined and "status" in joined:
            return self._ok(json.dumps({
                "evaluations_done": self.reserved,
                "reached": self.reserved >= self._budget_cap()}))
        if "preflight_candidate.py" in joined or "preflight_env.py" in joined:
            return self._ok("{}")
        if "timed_run.py" in joined or joined.endswith("train.py"):
            # the entrypoint run: write run.log with the next scripted outcome
            item = self.scores.pop(0) if self.scores else None
            if isinstance(item, dict):
                score = item.get("score")
                exit_code = item.get("exit", 0)
                companion = item.get("companion", True)
            else:
                score, exit_code, companion = item, 0, True
            text = "training...\n"
            if score is not None:
                text += f"neg_acc: {score}\n"
            if companion:
                text += "best_model: checkpoint.pt\n"
            kw["stdout"].write(text)
            return subprocess.CompletedProcess(args, exit_code, "", "")
        return self._ok("{}")

    @staticmethod
    def _ok(stdout: str):
        return subprocess.CompletedProcess([], 0, stdout, "")


def edit_train_py(source: str):
    def effect(ctx):
        (ctx.run_dir / "train.py").write_text(source)
    return effect


class HillclimbTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        write_task(self.repo, with_entrypoint=False)
        self.add_entrypoint()

    def add_entrypoint(self) -> None:
        (self.repo / "tasks" / "fake-task" / "train.py").write_text(
            "# baseline implementation\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_dir(self) -> Path:
        return self.repo / "runs" / "fake-task" / "t1"

    def rows(self) -> list[list[str]]:
        lines = (self.run_dir() / "results.tsv").read_text().strip().splitlines()
        return [line.split("\t") for line in lines[1:]]

    def test_baseline_then_improvement_kept(self) -> None:
        cmd = FakeCmd(self.repo, scores=[-0.50, -0.81])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "bigger model"},
             "side_effects": edit_train_py("# v2\n")},
        ])
        # exhaust budget after the second evaluation so the loop stops
        status = run_hillclimb(
            "fake-task", "t1", runner=runner, model="m",
            repo_root=self.repo, cmd=cmd, max_evaluations=3,
        )
        rows = self.rows()
        self.assertEqual(rows[0][1:3], ["-0.500000", "keep"])      # baseline
        self.assertEqual(rows[1][1:3], ["-0.810000", "keep"])      # improvement
        self.assertEqual((self.run_dir() / "best.py").read_text(), "# v2\n")
        self.assertTrue((self.run_dir() / "history" / "000.py").exists())

    def test_worse_score_reverted(self) -> None:
        cmd = FakeCmd(self.repo, scores=[-0.73, -0.58, -0.90])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "bad idea"},
             "side_effects": edit_train_py("# bad\n")},
            {"receipt": {"edited": True, "summary": "good idea"},
             "side_effects": edit_train_py("# good\n")},
        ])
        run_hillclimb("fake-task", "t1", runner=runner, model="m",
                      repo_root=self.repo, cmd=cmd, max_evaluations=3)
        rows = self.rows()
        self.assertEqual(rows[1][1:3], ["-0.580000", "discard"])
        self.assertEqual(rows[2][1:3], ["-0.900000", "keep"])
        # after the discard, the editor started from best.py content
        self.assertEqual((self.run_dir() / "best.py").read_text(), "# good\n")

    def test_crash_row_recorded_and_best_unchanged(self) -> None:
        cmd = FakeCmd(self.repo, scores=[-0.73, None])  # second run: no metric line
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "broken"},
             "side_effects": edit_train_py("# broken\n")},
        ])
        run_hillclimb("fake-task", "t1", runner=runner, model="m",
                      repo_root=self.repo, cmd=cmd, max_evaluations=3,
                      crash_repairs=0)
        rows = self.rows()
        self.assertEqual(rows[1][1:3], ["inf", "crash"])
        self.assertEqual((self.run_dir() / "best.py").read_text(),
                         "# baseline implementation\n")

    def test_successful_crash_repair_retries_repaired_copy(self) -> None:
        cmd = FakeCmd(self.repo, scores=[-0.73, None, -0.85])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "broken idea"},
             "side_effects": edit_train_py("# broken\n")},
            {"receipt": {"verdict": "code_incompatible", "summary": "typo",
                         "evidence": []}},
            {"receipt": {"edited": True, "summary": "fixed"},
             "side_effects": edit_train_py("# fixed\n")},
        ])
        run_hillclimb("fake-task", "t1", runner=runner, model="m",
                      repo_root=self.repo, cmd=cmd, max_evaluations=3,
                      crash_repairs=1)
        rows = self.rows()
        self.assertEqual(rows[1][1:3], ["inf", "crash"])
        self.assertEqual(rows[2][1:3], ["-0.850000", "keep"])
        self.assertEqual((self.run_dir() / "best.py").read_text(), "# fixed\n")

    def test_reserve_exit_4_stops_without_launching(self) -> None:
        cmd = FakeCmd(self.repo, scores=[-0.73])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "x"},
             "side_effects": edit_train_py("# v2\n")},
        ])
        status = run_hillclimb("fake-task", "t1", runner=runner, model="m",
                               repo_root=self.repo, cmd=cmd, max_evaluations=1)
        self.assertEqual(status["active_stop_condition"], "none")
        self.assertEqual(status["steps_done"], 1)

    def test_bootstrap_editor_failure_blocked(self) -> None:
        # task ships no entrypoint: the bootstrap editor session is required
        (self.repo / "tasks" / "fake-task" / "train.py").unlink()
        cmd = FakeCmd(self.repo, scores=[])
        runner = FakeSessionRunner([
            {"fail": ["postcondition: train.py missing"]},
        ])
        status = run_hillclimb("fake-task", "t1", runner=runner, model="m",
                               repo_root=self.repo, cmd=cmd, max_evaluations=3)
        self.assertIn("bootstrap", status["active_stop_condition"])
        self.assertFalse((self.run_dir() / "best.py").exists())

    def test_reconcile_appends_recovery_rows(self) -> None:
        # pre-create a run whose attempt log is ahead of results.tsv
        run_dir = self.run_dir()
        run_dir.mkdir(parents=True)
        (run_dir / "results.tsv").write_text(
            "step\tscore\tstatus\tdescription\n0\t-0.700000\tkeep\tbaseline\n")
        (run_dir / "train.py").write_text("# v1\n")
        (run_dir / "best.py").write_text("# v1\n")
        (run_dir / "framework_cfg.json").write_text(json.dumps({"max_evaluations": 5}))

        class ReconcileCmd(FakeCmd):
            def __call__(self, args, repo_root, check=True, capture=True, **kw):
                joined = " ".join(str(a) for a in args)
                if "evaluation_budget.py" in joined and "status" in joined:
                    done = 2 + self.reserved
                    return self._ok(json.dumps({
                        "evaluations_done": done,
                        "reached": done >= self._budget_cap()}))
                return super().__call__(args, repo_root, check, capture, **kw)

        cmd = ReconcileCmd(self.repo, scores=[-0.80])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "resume"},
             "side_effects": edit_train_py("# v2\n")},
        ])
        run_hillclimb("fake-task", "t1", runner=runner, model="m",
                      repo_root=self.repo, cmd=cmd)
        rows = self.rows()
        self.assertEqual(rows[1][1:3], ["inf", "crash"])  # recovery row
        self.assertIn("recovery", rows[1][3])

    def test_resume_restores_stale_best_from_history(self) -> None:
        # kill between the keep row (step 1) and the best.py snapshot:
        # results.tsv says step 1 is best, but best.py holds stale content
        run_dir = self.run_dir()
        run_dir.mkdir(parents=True)
        (run_dir / "results.tsv").write_text(
            "step\tscore\tstatus\tdescription\n"
            "0\t-0.700000\tkeep\tbaseline\n"
            "1\t-0.850000\tkeep\timprovement\n")
        (run_dir / "history").mkdir()
        (run_dir / "history" / "000.py").write_text("# v1\n")
        (run_dir / "history" / "001.py").write_text("# v2-best\n")
        (run_dir / "best.py").write_text("# stale\n")
        (run_dir / "train.py").write_text("# v2-best\n")
        (run_dir / "framework_cfg.json").write_text(json.dumps({"max_evaluations": 0}))
        cmd = FakeCmd(self.repo, scores=[])
        runner = FakeSessionRunner([])
        run_hillclimb("fake-task", "t1", runner=runner, model="m",
                      repo_root=self.repo, cmd=cmd)
        self.assertEqual((run_dir / "best.py").read_text(), "# v2-best\n")
        events = (run_dir / "driver_events.jsonl").read_text()
        self.assertIn("restore_best_step", events)
        # resume with no metadata file writes it (setup atomicity)
        self.assertTrue((run_dir / "run_metadata.json").exists())

    def test_baseline_crash_continues_with_repair_semantics(self) -> None:
        # fresh path: baseline crashes — the loop continues (same as resume)
        cmd = FakeCmd(self.repo, scores=[None, -0.60])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "fixed"},
             "side_effects": edit_train_py("# fixed\n")},
        ])
        status = run_hillclimb("fake-task", "t1", runner=runner, model="m",
                               repo_root=self.repo, cmd=cmd, max_evaluations=2)
        rows = self.rows()
        self.assertEqual(rows[0][1:3], ["inf", "crash"])      # crashed baseline
        self.assertEqual(rows[1][1:3], ["-0.600000", "keep"])  # first finite keeps
        self.assertEqual((self.run_dir() / "best.py").read_text(), "# fixed\n")
        self.assertEqual(status["active_stop_condition"], "none")

    def test_prepare_runs_in_task_working_dir(self) -> None:
        cmd = FakeCmd(self.repo, scores=[-0.50])
        runner = FakeSessionRunner([])
        run_hillclimb("fake-task", "t1", runner=runner, model="m",
                      repo_root=self.repo, cmd=cmd, max_evaluations=1)
        idx = next(i for i, c in enumerate(cmd.calls)
                   if " ".join(str(a) for a in c) == "uv run python prepare.py")
        self.assertEqual(cmd.cwds[idx], self.repo / "tasks" / "fake-task")

    def test_prepare_failure_blocks_without_traceback(self) -> None:
        cmd = FakeCmd(self.repo, scores=[])
        cmd.raise_once = {"prepare.py"}
        runner = FakeSessionRunner([])
        status = run_hillclimb("fake-task", "t1", runner=runner, model="m",
                               repo_root=self.repo, cmd=cmd, max_evaluations=3)
        self.assertIn("prepare_command failed",
                      status["active_stop_condition"])

    def test_missing_companion_pattern_scores_crash(self) -> None:
        # metric line present but the required best_model line absent: the
        # process did not complete its contract, so the score must not count
        cmd = FakeCmd(self.repo,
                      scores=[-0.73, {"score": -0.80, "companion": False}])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "x"},
             "side_effects": edit_train_py("# v2\n")},
        ])
        run_hillclimb("fake-task", "t1", runner=runner, model="m",
                      repo_root=self.repo, cmd=cmd, max_evaluations=2,
                      crash_repairs=0)
        rows = self.rows()
        self.assertEqual(rows[0][1:3], ["-0.730000", "keep"])
        self.assertEqual(rows[1][1:3], ["inf", "crash"])

    def test_nonzero_exit_despite_metric_line_scores_crash(self) -> None:
        cmd = FakeCmd(self.repo,
                      scores=[-0.73, {"score": -0.80, "exit": 1}])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "x"},
             "side_effects": edit_train_py("# v2\n")},
        ])
        run_hillclimb("fake-task", "t1", runner=runner, model="m",
                      repo_root=self.repo, cmd=cmd, max_evaluations=2,
                      crash_repairs=0)
        rows = self.rows()
        self.assertEqual(rows[0][1:3], ["-0.730000", "keep"])
        self.assertEqual(rows[1][1:3], ["inf", "crash"])

    def test_keep_snapshot_and_best_are_identical(self) -> None:
        # snapshot-before-row ordering: after any keep, history/<step>.py and
        # best.py hold the same content, so _restore_best can always repair
        cmd = FakeCmd(self.repo, scores=[-0.50, -0.81])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "bigger model"},
             "side_effects": edit_train_py("# v2\n")},
        ])
        run_hillclimb("fake-task", "t1", runner=runner, model="m",
                      repo_root=self.repo, cmd=cmd, max_evaluations=3)
        run_dir = self.run_dir()
        self.assertEqual((run_dir / "history" / "000.py").read_text(),
                         "# baseline implementation\n")
        self.assertEqual((run_dir / "history" / "001.py").read_text(),
                         (run_dir / "best.py").read_text())


if __name__ == "__main__":
    unittest.main()
