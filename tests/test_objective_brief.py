"""Tests for the objective brief (target-score visibility upgrade).

Covers the plan's phase-A/C contracts: parsing, missing target, coexistence
with the baseline-relative proxy target, lower-is-better gap, context
assembly into role invocations, and the scheduler invariability (an
objective brief present in the run dir never changes deterministic
selection).
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))

import objective_brief as ob  # noqa: E402
import got_select  # noqa: E402
import checkpoint as checkpoint_mod  # noqa: E402
import llm  # noqa: E402
from driver.loops import rewrite as rewrite_loop  # noqa: E402
from driver.session import FakeSessionRunner  # noqa: E402
from driver.status import _objective_view  # noqa: E402

SPOOKY_TOML = {
    "result": {
        "metric": "validation_log_loss",
        "required_patterns": ["^score:"],
        "target_score": 0.26996,
    }
}

NO_TARGET_TOML = {"result": {"metric": "neg_acc", "required_patterns": []}}


class BriefBuildTests(unittest.TestCase):
    def test_declared_target(self) -> None:
        brief = ob.build_brief(SPOOKY_TOML, run_best=0.45)
        self.assertEqual(brief["metric"], "validation_log_loss")
        self.assertEqual(brief["direction"], "minimize")
        self.assertEqual(brief["aspirational_target_score"], 0.26996)
        self.assertEqual(brief["target_source"],
                         "task.toml[result].target_score")
        self.assertEqual(brief["target_semantics"], "anti_slop_aspiration")
        self.assertEqual(brief["run_best"], 0.45)
        self.assertIsNone(brief["baseline_score"])

    def test_missing_target_still_a_valid_brief(self) -> None:
        brief = ob.build_brief(NO_TARGET_TOML)
        self.assertIsNone(brief["aspirational_target_score"])
        self.assertIsNone(brief["target_source"])
        self.assertIn("no declared target", ob.compact_line(brief))

    def test_compact_line_carries_target_semantics(self) -> None:
        line = ob.compact_line(ob.build_brief(SPOOKY_TOML))
        self.assertIn("minimize validation_log_loss", line)
        self.assertIn("aspirational target 0.26996", line)
        self.assertIn("not an official score or a stop line", line)

    def test_render_block_is_bounded_and_labeled(self) -> None:
        block = ob.render_block(ob.build_brief(SPOOKY_TOML))
        self.assertTrue(block.startswith("## Objective"))
        self.assertIn("aspirational_target_score: 0.26996", block)
        self.assertIn("target_source: task.toml[result].target_score", block)
        self.assertIn("anti_slop_aspiration", block)

    def test_non_finite_target_is_a_contract_error(self) -> None:
        for bad in (float("inf"), float("nan"), True, "0.5"):
            with self.assertRaises(ValueError):
                ob.validated_target({"target_score": bad})
        with self.assertRaises(ValueError):
            ob.validated_target([])

    def test_gap_is_lower_is_better(self) -> None:
        brief = ob.build_brief(SPOOKY_TOML)
        self.assertEqual(ob.gap_to_target(brief, 0.45),
                         0.45 - 0.26996)
        self.assertLess(ob.gap_to_target(brief, 0.2), 0)  # below the bar
        self.assertIsNone(ob.gap_to_target(brief, None))
        self.assertIsNone(ob.gap_to_target(ob.build_brief(NO_TARGET_TOML), 0.4))

    def test_ensure_brief_writes_once_and_rereads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            first = ob.ensure_brief(run_dir, SPOOKY_TOML)
            path = run_dir / "objective_brief.json"
            self.assertTrue(path.is_file())
            self.assertIn("generated_at", first)
            # a later, different task_toml must not overwrite the stored brief
            again = ob.ensure_brief(run_dir, NO_TARGET_TOML)
            self.assertEqual(again["aspirational_target_score"], 0.26996)


class InnerTunerRenderingTests(unittest.TestCase):
    def _checkpoint(self, **task_fields):
        return checkpoint_mod.Checkpoint(
            checkpoint_id="c",
            regime="continuation",
            stratum="cont_improved",
            source={"candidate_id": "c", "kind": "improve"},
            checkpoint_dir=Path("."),
            candidate_relpath="candidate",
            candidate_path=Path("candidate/train.py"),
            task=checkpoint_mod.TaskSpec(
                score_fn="evaluate_config",
                preflight_fn=None,
                per_runtime_limit=None,
                **task_fields,
            ),
            incumbent=checkpoint_mod.Incumbent(
                params={"x": 1}, score=3.0),
            incumbent_is_inherited_control=False,
            items={"task_baseline": {"kind": "observed_metric",
                                     "metric": "val_bpb", "value": 4.0,
                                     "direction": "minimize"}},
        )

    def test_both_targets_render_with_distinct_labels(self) -> None:
        items = llm._experiment_items(self._checkpoint(
            relative_improvement_over_baseline=0.075,
            aspirational_target_score=0.26996,
        ))
        self.assertIn("required_target_score: 3.7", items)
        self.assertIn("aspirational_target_score: 0.26996", items)
        self.assertIn("aspirational_target_semantics", items)
        # the two labels never alias each other
        self.assertNotIn("required_target_score: 0.26996", items)

    def test_no_aspirational_target_renders_unchanged(self) -> None:
        items = llm._experiment_items(self._checkpoint(
            relative_improvement_over_baseline=0.075))
        self.assertIn("required_target_score: 3.7", items)
        self.assertNotIn("aspirational_target_score", items)

    def test_task_spec_round_trip(self) -> None:
        raw = {
            "score_fn": "evaluate_config",
            "preflight_fn": None,
            "per_runtime_limit": None,
            "aspirational_target_score": 0.26996,
        }
        spec = checkpoint_mod._task_spec(Path("checkpoint.json"), raw)
        self.assertEqual(spec.aspirational_target_score, 0.26996)
        spec2 = checkpoint_mod._task_spec(
            Path("checkpoint.json"),
            {k: v for k, v in raw.items() if k != "aspirational_target_score"},
        )
        self.assertIsNone(spec2.aspirational_target_score)
        with self.assertRaises(ValueError):
            checkpoint_mod._task_spec(Path("checkpoint.json"), {
                **raw, "aspirational_target_score": float("inf")})


class CrossLayerSourceTests(unittest.TestCase):
    """Phase-A acceptance: driver, inner benchmark, and fixtures derive the
    same target from the same task contract (real repo tasks)."""

    def test_freeze_task_block_reads_spooky_target(self) -> None:
        import freeze
        block = freeze._task_block(
            ROOT / "runs" / "mle-spooky" / "tag", {"per_runtime_limit": None})
        self.assertEqual(block["aspirational_target_score"], 0.26996)
        self.assertEqual(block["relative_improvement_over_baseline"], None)

    def test_hebo_production_checkpoint_reads_spooky_target(self) -> None:
        import hebo_search
        target = hebo_search._configured_target_score(
            ROOT / "runs" / "mle-spooky" / "tag" / "candidates" / "001"
            / "train.py")
        self.assertEqual(target, 0.26996)
        # a task without runs-path context (no task.toml resolution) degrades
        # to None rather than guessing
        self.assertIsNone(
            hebo_search._configured_target_score(Path("/tmp/x/train.py")))

    def test_local_tr_production_checkpoint_reads_spooky_target(self) -> None:
        import local_tr_search
        target = local_tr_search._configured_target_score(
            ROOT / "runs" / "mle-spooky" / "tag" / "candidates" / "001"
            / "train.py")
        self.assertEqual(target, 0.26996)
        self.assertIsNone(
            local_tr_search._configured_target_score(Path("/tmp/x/train.py")))

    def test_driver_brief_matches_task_contract(self) -> None:
        brief = ob.build_brief(SPOOKY_TOML)
        self.assertEqual(brief["aspirational_target_score"],
                         ob.validated_target(
                             SPOOKY_TOML["result"]))


class SelectionInvariabilityTests(unittest.TestCase):
    def test_objective_brief_never_changes_got_select_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "framework_cfg.json").write_text(json.dumps(
                {"max_evaluations": 10, "tuner": {"K_eval": 3}}))
            ledger = run_dir / "ledger.json"

            def decide() -> str:
                out = io.StringIO()
                with redirect_stdout(out):
                    got_select.cmd_decide(
                        SimpleNamespace(ledger=str(ledger), cfg=None))
                return out.getvalue()

            baseline = decide()
            ob.ensure_brief(run_dir, SPOOKY_TOML)
            self.assertEqual(decide(), baseline)
            (run_dir / "objective_brief.json").write_text(json.dumps(
                ob.build_brief({**SPOOKY_TOML,
                                "result": {**SPOOKY_TOML["result"],
                                           "target_score": 0.01}})))
            self.assertEqual(decide(), baseline)


class StatusViewTests(unittest.TestCase):
    def test_objective_view_reports_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            ob.ensure_brief(run_dir, SPOOKY_TOML)
            view = _objective_view(run_dir, 0.45)
            self.assertEqual(view["aspirational_target_score"], 0.26996)
            self.assertEqual(view["gap_to_target"],
                             0.45 - 0.26996)
            self.assertIsNone(_objective_view(run_dir / "missing", 0.45))


# --- rewrite-loop context assembly (fake runner, real bout CLI) ---------------


TARGET_TASK_TOML = """
[env]
project = "tasks/fake-task"
[result]
metric = "neg_acc"
target_score = 0.26996
[run]
working_dir = "tasks/fake-task"
prepare_command = "uv run python prepare.py"
"""


def _write_target_task(repo: Path) -> None:
    task_dir = repo / "tasks" / "fake-task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(TARGET_TASK_TOML)
    (task_dir / "prepare.py").write_text("# fixed eval surface\n")


class _RewriteFakeCmd:
    """The subset of FakeCmd one kept bout needs (init/budget/context/
    eval/preflight); rewrite_bout.py runs the real CLI in-process."""

    def __init__(self, repo: Path):
        self.repo = repo
        self.reserved = 0

    def __call__(self, args, repo_root, check=True, capture=True, **kw):
        args = [str(a) for a in args]
        joined = " ".join(args)
        run_dir = self.repo / "runs" / "fake-task" / "t1"
        if "init_run.py" in joined:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": 10,
                            "per_runtime_limit": None}))
            return subprocess.CompletedProcess([], 0, "", "")
        if "evaluation_budget.py" in joined:
            return subprocess.CompletedProcess([], 0, json.dumps(
                {"evaluations_done": self.reserved, "reached": False}), "")
        if "rewrite_bout.py" in joined:
            import rewrite_bout
            idx = next(i for i, a in enumerate(args)
                       if a.endswith("rewrite_bout.py"))
            old_argv = sys.argv
            sys.argv = ["rewrite_bout.py"] + args[idx + 1:]
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    code = rewrite_bout.main()
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
            finally:
                sys.argv = old_argv
            return subprocess.CompletedProcess(
                args, code, buf.getvalue(), "")
        if "rewrite_context.py" in joined:
            candidate = Path(args[args.index("--candidate") + 1])
            output = candidate / "_rewrite" / "context.md"
            output.parent.mkdir(exist_ok=True)
            output.write_text("# Rewrite context\n", encoding="utf-8")
            return subprocess.CompletedProcess([], 0, "", "")
        if "rewrite_eval.py" in joined:
            candidate = Path(args[args.index("--candidate") + 1])
            self.reserved += 1
            attempt = f"eval-{self.reserved:06d}"
            traces = candidate / "_traces"
            traces.mkdir(exist_ok=True)
            (traces / f"{attempt}.log").write_text("trace\n")
            return subprocess.CompletedProcess([], 0, json.dumps({
                "attempt_id": attempt, "score": 0.5, "stage": "eval"}), "")
        if "preflight_candidate.py" in joined:
            return subprocess.CompletedProcess([], 0, "", "")
        return subprocess.CompletedProcess([], 0, "{}", "")


class RewriteContextAssemblyTests(unittest.TestCase):
    def test_editor_extras_carry_declared_target_from_brief(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_target_task(repo)
            candidate = repo / "runs" / "fake-task" / "t1" / "candidates" / "src-001"
            candidate.mkdir(parents=True)
            (candidate / "train.py").write_text(
                'BASE_PARAMS = {"x": 1}\n# v1\n')
            (candidate / "_import.json").write_text(json.dumps({
                "source": "runs/fake-task/src", "baseline_score": 1.0,
                "warm_to_tuned_delta": 0.2, "idea": "i", "change": "c",
                "semantic_point": {"assignments": []},
                "tune_summary": {"bouts": 1},
            }))
            cmd = _RewriteFakeCmd(repo)
            runner = FakeSessionRunner([
                {"receipt": {"edited": True, "summary": "s", "basis": "b"},
                 "side_effects": lambda ctx: (
                     Path(ctx.extra["candidate_dir"], "train.py").write_text(
                         'BASE_PARAMS = {"x": 1}\n# v2\n'))},
            ])
            rewrite_loop.run_rewrite(
                "fake-task", "t1", runner=runner, model="m", repo_root=repo,
                cmd=cmd, noise_margin=0.1, max_bouts=12, stall_after=1,
                max_evaluations=10)
            editor = next(ctx for name, ctx in runner.calls
                          if name == "rewrite-editor")
            # byte-compatible with the old ad-hoc key: the value now comes
            # from the objective brief's validated extractor
            self.assertEqual(editor.extra["target_score"], 0.26996)
            self.assertEqual(editor.extra["metric"], "neg_acc")
            # observability: setup wrote the run's objective brief once
            stored = json.loads(
                (repo / "runs" / "fake-task" / "t1"
                 / "objective_brief.json").read_text())
            self.assertEqual(stored["aspirational_target_score"], 0.26996)


if __name__ == "__main__":
    unittest.main()
