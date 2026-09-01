"""Tests for driver/loops/rewrite.py (rewrite-operator loop).

Mirrors test_driver_hillclimb.py's fake-runner approach: a scripted
FakeSessionRunner plays the rewrite-editor; FakeCmd fakes every tools/
subprocess — except rewrite_bout.py, whose REAL CLI runs in-process so
snapshot/revert/journal/finalize behavior (including byte-exact rollback)
is exercised for real.
"""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import rewrite_bout  # noqa: E402
from driver.loops.rewrite import run_rewrite  # noqa: E402
from driver.session import FakeSessionRunner  # noqa: E402


def write_task(repo: Path) -> None:
    task_dir = repo / "tasks" / "fake-task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        """
[env]
project = "tasks/fake-task"
[result]
metric = "neg_acc"
[run]
working_dir = "tasks/fake-task"
prepare_command = "uv run python prepare.py"
"""
    )
    (task_dir / "prepare.py").write_text("# fixed eval surface\n")


V1 = 'BASE_PARAMS = {"x": 1}\n# v1\n'
V2 = 'BASE_PARAMS = {"x": 2}\n# v2\n'
V3 = 'BASE_PARAMS = {"x": 3}\n# v3\n'


class FakeCmd:
    """Fakes tools/ helper subprocesses keyed by argv content.

    eval_script: queue of rewrite_eval outcomes — {"score": float} or
    {"score": None, "error": str} (a spent evaluation), {"params_error":
    str} (BASE_PARAMS unreadable, no budget spent), {"exit": 4} (budget
    exhausted). preflight_script: queue of preflight returncodes.
    rewrite_bout.py dispatches to the real CLI in-process.
    """

    def __init__(self, repo: Path, eval_script: list[dict],
                 preflight_script: list[int] | None = None):
        self.repo = repo
        self.eval_script = list(eval_script)
        self.preflight_script = list(preflight_script or [])
        self.reserved = 0
        self.calls: list[list[str]] = []

    @property
    def run_dir(self) -> Path:
        return self.repo / "runs" / "fake-task" / "t1"

    def _budget_cap(self) -> float:
        cfg = self.run_dir / "framework_cfg.json"
        if cfg.exists():
            cap = json.loads(cfg.read_text()).get("max_evaluations")
            if cap is not None:
                return cap
        return float("inf")

    def _real_bout_cli(self, args: list[str]) -> subprocess.CompletedProcess:
        idx = next(i for i, a in enumerate(args) if a.endswith("rewrite_bout.py"))
        argv = ["rewrite_bout.py"] + args[idx + 1:]
        old_argv = sys.argv
        sys.argv = argv
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = rewrite_bout.main()
        except SystemExit as exc:  # argparse error or clean contract failure
            return subprocess.CompletedProcess(
                args, exc.code if isinstance(exc.code, int) else 1,
                buf.getvalue(), "" if isinstance(exc.code, int) else str(exc.code))
        finally:
            sys.argv = old_argv
        return subprocess.CompletedProcess(args, code, buf.getvalue(), "")

    def __call__(self, args, repo_root, check=True, capture=True, **kw):
        args = [str(a) for a in args]
        self.calls.append(args)
        joined = " ".join(args)
        run_dir = self.run_dir

        if "init_run.py" in joined:
            run_dir.mkdir(parents=True, exist_ok=True)
            cap = None
            if "--max-evaluations" in args:
                cap = int(args[args.index("--max-evaluations") + 1])
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": cap, "per_runtime_limit": None}))
            return self._ok("")
        if "evaluation_budget.py" in joined:
            return self._ok(json.dumps({
                "evaluations_done": self.reserved,
                "reached": self.reserved >= self._budget_cap()}))
        if "rewrite_bout.py" in joined:
            return self._real_bout_cli(args)
        if "rewrite_context.py" in joined:
            candidate = Path(args[args.index("--candidate") + 1])
            output = candidate / "_rewrite" / "context.md"
            output.parent.mkdir(exist_ok=True)
            output.write_text("# Rewrite context\n", encoding="utf-8")
            return self._ok(json.dumps({"context_md": str(output)}))
        if "rewrite_eval.py" in joined:
            candidate = Path(args[args.index("--candidate") + 1])
            item = self.eval_script.pop(0) if self.eval_script else {"score": 0.0}
            if item.get("exit") == 4:
                return subprocess.CompletedProcess(
                    args, 4, json.dumps({"attempt_id": None, "score": None,
                                         "error": "evaluation budget exhausted",
                                         "stage": "eval"}), "")
            if "params_error" in item:
                return self._ok(json.dumps(
                    {"attempt_id": None, "score": None,
                     "error": item["params_error"], "stage": "params"}))
            self.reserved += 1
            attempt_id = f"eval-{self.reserved:06d}"
            traces = candidate / "_traces"
            traces.mkdir(exist_ok=True)
            (traces / f"{attempt_id}.log").write_text("trace\n", encoding="utf-8")
            return self._ok(json.dumps({
                "attempt_id": attempt_id,
                "score": item.get("score"),
                "error": item.get("error"),
                "stage": "eval"}))
        if "preflight_candidate.py" in joined:
            rc = self.preflight_script.pop(0) if self.preflight_script else 0
            return subprocess.CompletedProcess(
                args, rc, "", "SyntaxError: bad edit" if rc else "")
        return self._ok("{}")  # uv sync, prepare.py, preflight_env.py

    @staticmethod
    def _ok(stdout: str):
        return subprocess.CompletedProcess([], 0, stdout, "")


def edit_train_py(source: str):
    def effect(ctx):
        candidate = Path(ctx.extra["candidate_dir"])
        (candidate / "train.py").write_text(source)
    return effect


class RewriteLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        write_task(self.repo)
        candidate = self.candidate()
        candidate.mkdir(parents=True)
        (candidate / "train.py").write_text(V1)
        (candidate / "_import.json").write_text(json.dumps({
            "source": "runs/fake-task/src",
            "baseline_score": 1.0,
            "warm_to_tuned_delta": 0.2,
            "idea": "idea text",
            "change": "change text",
            "semantic_point": {"assignments": []},
            "tune_summary": {"bouts": 3},
        }))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def candidate(self) -> Path:
        return self.repo / "runs" / "fake-task" / "t1" / "candidates" / "src-001"

    def bouts(self) -> list[dict]:
        path = self.candidate() / "_rewrite" / "bouts.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def editor_calls(self, runner) -> list:
        return [ctx for name, ctx in runner.calls if name == "rewrite-editor"]

    def test_kept_bout_then_reverted_bout(self) -> None:
        cmd = FakeCmd(self.repo, eval_script=[{"score": 0.7}, {"score": 0.9}])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "lever a", "basis": "hyp-1"},
             "side_effects": edit_train_py(V2)},
            {"receipt": {"edited": True, "summary": "lever b", "basis": "g-2"},
             "side_effects": edit_train_py(V3)},
        ])
        status = run_rewrite(
            "fake-task", "t1", runner=runner, model="m", repo_root=self.repo,
            cmd=cmd, noise_margin=0.1, max_bouts=12, stall_after=1,
            max_evaluations=10,
        )

        # bout 1 kept (1.0 - 0.7 > margin): the v2 edit survives; bout 2
        # reverted_worse (0.9 > best 0.7): byte-exact rollback to v2
        self.assertEqual((self.candidate() / "train.py").read_text(), V2)
        bouts = self.bouts()
        self.assertEqual(len(bouts), 2)
        self.assertEqual(bouts[0]["bout"], 1)
        self.assertEqual(bouts[0]["outcome"], "kept")
        self.assertEqual(bouts[0]["score"], 0.7)
        self.assertEqual(bouts[0]["attempt_id"], "eval-000001")
        self.assertEqual(bouts[0]["summary"], "lever a")
        self.assertEqual(bouts[0]["basis"], "hyp-1")
        self.assertEqual(bouts[1]["bout"], 2)
        self.assertEqual(bouts[1]["outcome"], "reverted_worse")
        self.assertEqual(bouts[1]["score"], 0.9)
        self.assertEqual(bouts[1]["attempt_id"], "eval-000002")

        # session resume chain: bout 2 resumed bout 1's session; the
        # per-candidate pointer tracks the latest invocation
        calls = self.editor_calls(runner)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0].resume_session_id)
        self.assertEqual(calls[0].extra["current_best"], 1.0)
        self.assertEqual(calls[0].extra["metric"], "neg_acc")
        self.assertEqual(calls[0].extra["candidate_dir"], str(self.candidate()))
        self.assertNotIn("last_outcome", calls[0].extra)
        self.assertEqual(calls[1].resume_session_id, "fake-sess-0001")
        pointer = json.loads(
            (self.candidate() / "_rewrite" / "session.json").read_text())
        self.assertEqual(pointer, {"invocation_id": 2})

        # the second bout's extras carry the first bout's outcome
        self.assertEqual(calls[1].extra["current_best"], 0.7)
        self.assertEqual(calls[1].extra["last_outcome"], "kept")
        self.assertEqual(calls[1].extra["last_score"], 0.7)
        self.assertTrue(calls[1].extra["last_trace"].endswith(
            "_traces/eval-000001.log"))

        # stalled derivation: stall_after=1 and one consecutive non-kept bout
        # ended the loop with budget and bouts to spare
        self.assertEqual(status["active_stop_condition"], "none")
        self.assertEqual(status["steps_done"], 2)
        summary = status["candidates"]["src-001"]
        self.assertEqual(summary["bouts"], 2)
        self.assertEqual(summary["consecutive_non_kept"], 1)
        self.assertEqual(summary["best"], 0.7)

    def test_preflight_failure_repair_failure_reverts_without_budget(self) -> None:
        cmd = FakeCmd(self.repo, eval_script=[], preflight_script=[1, 1])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "broken", "basis": "hyp-1"},
             "side_effects": edit_train_py(V2)},
            {"receipt": {"edited": True, "summary": "still broken", "basis": "hyp-1"},
             "side_effects": edit_train_py(V3)},
        ])
        status = run_rewrite(
            "fake-task", "t1", runner=runner, model="m", repo_root=self.repo,
            cmd=cmd, stall_after=1, max_evaluations=10,
        )

        # repair resumed the same session with the error tail, then the edit
        # was rolled back and journaled reverted_crash — no budget spent
        calls = self.editor_calls(runner)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1].resume_session_id, "fake-sess-0001")
        self.assertIn("SyntaxError", calls[1].extra["preflight_error"])
        self.assertEqual((self.candidate() / "train.py").read_text(), V1)
        bouts = self.bouts()
        self.assertEqual(len(bouts), 1)
        self.assertEqual(bouts[0]["outcome"], "reverted_crash")
        self.assertIsNone(bouts[0]["score"])
        self.assertIsNone(bouts[0]["attempt_id"])
        self.assertEqual(bouts[0]["summary"], "still broken")
        self.assertEqual(status["steps_done"], 0)
        self.assertEqual(status["active_stop_condition"], "none")

    def test_unedited_bout_journaled_noop(self) -> None:
        cmd = FakeCmd(self.repo, eval_script=[])
        runner = FakeSessionRunner([
            {"receipt": {"edited": False, "summary": "no lever", "basis": "none"}},
        ])
        status = run_rewrite(
            "fake-task", "t1", runner=runner, model="m", repo_root=self.repo,
            cmd=cmd, stall_after=1, max_evaluations=10,
        )

        self.assertEqual((self.candidate() / "train.py").read_text(), V1)
        bouts = self.bouts()
        self.assertEqual(len(bouts), 1)
        self.assertEqual(bouts[0]["outcome"], "noop")
        self.assertIsNone(bouts[0]["score"])
        self.assertIsNone(bouts[0]["attempt_id"])
        self.assertEqual(status["steps_done"], 0)
        # noop counts toward the stall derivation: the loop stopped here
        self.assertEqual(len(self.editor_calls(runner)), 1)
        self.assertEqual(status["candidates"]["src-001"]["consecutive_non_kept"], 1)

    def test_invocation_failed_reverts_to_bout_snapshot(self) -> None:
        cmd = FakeCmd(self.repo, eval_script=[])
        runner = FakeSessionRunner([
            {"side_effects": edit_train_py(V2),
             "fail": ["session died mid-edit"]},
        ])
        status = run_rewrite(
            "fake-task", "t1", runner=runner, model="m", repo_root=self.repo,
            cmd=cmd, stall_after=1, max_evaluations=10,
        )

        # the failed session's partial edit is rolled back to the bout
        # snapshot before the run blocks, so the replayed bout re-snapshots
        # the clean file
        self.assertEqual((self.candidate() / "train.py").read_text(), V1)
        self.assertIn("editor invocation failed",
                      status["active_stop_condition"])
        self.assertFalse(
            (self.candidate() / "_rewrite" / "bouts.jsonl").exists())

    def test_budget_exhausted_mid_bout_reverts_edit(self) -> None:
        cmd = FakeCmd(self.repo, eval_script=[{"exit": 4}])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "lever a", "basis": "hyp-1"},
             "side_effects": edit_train_py(V2)},
        ])
        status = run_rewrite(
            "fake-task", "t1", runner=runner, model="m", repo_root=self.repo,
            cmd=cmd, stall_after=1, max_evaluations=10,
        )

        # rewrite_eval exit 4 = budget out: the unverified edit is rolled
        # back, no bout is journaled, and the run ends normally (not blocked)
        self.assertEqual((self.candidate() / "train.py").read_text(), V1)
        self.assertEqual(status["active_stop_condition"], "none")
        self.assertEqual(status["candidates"]["src-001"]["bouts"], 0)

    def test_unedited_receipt_but_dirty_file_reverts_and_journals_noop(self) -> None:
        cmd = FakeCmd(self.repo, eval_script=[])
        runner = FakeSessionRunner([
            {"receipt": {"edited": False, "summary": "no lever", "basis": "none"},
             "side_effects": edit_train_py(V2)},
        ])
        status = run_rewrite(
            "fake-task", "t1", runner=runner, model="m", repo_root=self.repo,
            cmd=cmd, stall_after=1, max_evaluations=10,
        )

        # receipt claims no edit but train.py changed: the anomaly is
        # reverted byte-exactly and journaled noop; no preflight, no eval
        self.assertEqual((self.candidate() / "train.py").read_text(), V1)
        bouts = self.bouts()
        self.assertEqual(len(bouts), 1)
        self.assertEqual(bouts[0]["outcome"], "noop")
        self.assertIn("differed", bouts[0]["summary"])
        self.assertIsNone(bouts[0]["score"])
        self.assertEqual(cmd.reserved, 0)
        self.assertFalse(any("preflight_candidate.py" in " ".join(call)
                             for call in cmd.calls))
        self.assertEqual(status["steps_done"], 0)

    def test_edited_receipt_but_unchanged_file_journals_noop(self) -> None:
        cmd = FakeCmd(self.repo, eval_script=[])
        runner = FakeSessionRunner([
            {"receipt": {"edited": True, "summary": "claimed edit",
                         "basis": "hyp-1"}},
        ])
        status = run_rewrite(
            "fake-task", "t1", runner=runner, model="m", repo_root=self.repo,
            cmd=cmd, stall_after=1, max_evaluations=10,
        )

        # receipt claims an edit but train.py is byte-identical: journaled
        # noop with the receipt's own summary; no preflight, no eval
        self.assertEqual((self.candidate() / "train.py").read_text(), V1)
        bouts = self.bouts()
        self.assertEqual(len(bouts), 1)
        self.assertEqual(bouts[0]["outcome"], "noop")
        self.assertEqual(bouts[0]["summary"], "claimed edit")
        self.assertEqual(cmd.reserved, 0)
        self.assertFalse(any("preflight_candidate.py" in " ".join(call)
                             for call in cmd.calls))
        self.assertEqual(status["steps_done"], 0)

    def test_context_flag_passes_through_to_rewrite_context(self) -> None:
        cmd = FakeCmd(self.repo, eval_script=[])
        runner = FakeSessionRunner([
            {"receipt": {"edited": False, "summary": "no lever", "basis": "none"}},
        ])
        run_rewrite(
            "fake-task", "t1", runner=runner, model="m", repo_root=self.repo,
            cmd=cmd, stall_after=1, max_evaluations=10,
            context="bout_history,traces",
        )

        ctx_calls = [call for call in cmd.calls
                     if any(a.endswith("rewrite_context.py") for a in call)]
        self.assertEqual(len(ctx_calls), 1)
        argv = ctx_calls[0]
        self.assertEqual(argv[argv.index("--sections") + 1],
                         "bout_history,traces")


if __name__ == "__main__":
    unittest.main()
