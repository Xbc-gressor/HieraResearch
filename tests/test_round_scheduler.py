"""round_v1: time budget, cycle state, rewrite/tune selection, kept-rewrite
commit, and the driver's two-phase loop."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import evaluation_budget  # noqa: E402
import rewrite_bout  # noqa: E402
import rewrite_rebase  # noqa: E402
from ledger import cmd_add_record, record_rewrite, record_run  # noqa: E402
from semantic_space import complete_point  # noqa: E402
from tools.scheduler import cli as scheduler_cli  # noqa: E402
from tools.scheduler import round_policy  # noqa: E402
from tools.scheduler.session import contract_for  # noqa: E402
from tools.scheduler.state import load_state  # noqa: E402
from tune_tools import _candidate_execution_revision  # noqa: E402
from driver.loops.experiment import run_experiment  # noqa: E402
from driver.session import FakeSessionRunner  # noqa: E402
from tests.fixtures import (  # noqa: E402
    background_text,
    fixture_registry,
    policy_receipt,
)
from tests.test_driver_experiment import (  # noqa: E402
    ExperimentCmd,
    write_background,
    write_task,
    writer_effect,
)


TRAIN_PY = (
    'PARAM_SCHEMA = {"x": "float"}\n'
    'SEARCH_SPACE = {"x": ("float", 0.0, 1.0)}\n'
    'BASE_PARAMS = {"x": 0.5}\n'
    "def make_model(params):\n"
    "    return params\n"
)
PREPARE_PY = "def evaluate_config(make_model, params):\n    return 0.0\n"


def _candidate(run_dir: Path, run_id: str, score: float,
               *, in_progress: bool = False) -> Path:
    directory = run_dir / "candidates" / run_id
    directory.mkdir(parents=True)
    (directory / "train.py").write_text(TRAIN_PY)
    (directory / "prepare.py").write_text(PREPARE_PY)
    report = {
        "inner_policy": "hebo24-hebo20",
        "phase_a": {
            "status": "ok",
            "candidate_code_revision": _candidate_execution_revision(
                directory / "train.py"),
            "search_space": {"x": ["float", 0.0, 1.0]},
            "warm_start_configs": [{"params": {"x": 0.5}, "score": score}],
            "deferred_configs": [],
            "best_warm_params": {"x": 0.5},
            "best_warm_score": score,
            "trials_attempted": 1,
        },
    }
    if in_progress:
        report["phase_c"] = {"stages": [
            {"method": "hebo", "status": "running", "bout_index": 0, "trials": []}
        ]}
    (directory / "tune_report.json").write_text(json.dumps(report))
    return directory


def _record(run_id: str, score: float) -> dict:
    return {"run_id": run_id, "status": "keep", "op": "fresh",
            "source_run_ids": [], "tune": False, "tuning_bouts": 0,
            "best_warm_score": score, "final_best_score": score}


def _run_dir(tmp: Path, *, deadline: float, round_cfg: dict | None = None,
             n_seed: int = 2) -> Path:
    run_dir = tmp / "runs" / "fake-task" / "t1"
    run_dir.mkdir(parents=True)
    (run_dir / "framework_cfg.json").write_text(json.dumps({
        "got": {"n_seed": n_seed},
        "tuner": {"scheduler_policy": "round_v1", "inner_policy": "hebo24-hebo20",
                  "K_eval": 2},
        "round": round_cfg or {},
        "deadline": deadline,
        "final_reserve_seconds": 60,
    }))
    return run_dir


class TimeBudgetTests(unittest.TestCase):
    def test_status_reservation_and_durations_follow_the_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_dir(Path(tmp), deadline=time.time() + 3600)
            train = _candidate(run_dir, "000", 0.9) / "train.py"
            view = evaluation_budget.budget_status(run_dir)
            self.assertIsNone(view["remaining"])
            self.assertFalse(view["reached"])
            self.assertGreater(view["time"]["usable_seconds"], 3000)

            receipt = evaluation_budget.reserve_evaluation(
                train, params={}, phase="rewrite", method="rewrite")
            evaluation_budget.record_evaluation_completion(
                train, attempt_id=receipt["attempt_id"], duration_seconds=12.0)
            view = evaluation_budget.budget_status(run_dir)
            self.assertEqual(view["evaluations_done"], 1)
            self.assertEqual(view["per_candidate"][0]["mean_seconds"], 12.0)

            # An open phase quota bounds reservations before the deadline does.
            round_policy.save_round_state(run_dir, {
                **round_policy.load_round_state(run_dir),
                "phase": "optimize", "phase_deadline": time.time() - 1})
            with self.assertRaises(evaluation_budget.EvaluationBudgetExhausted) as ctx:
                evaluation_budget.reserve_evaluation(
                    train, params={}, phase="rewrite", method="rewrite")
            self.assertEqual(ctx.exception.scope, "round_quota")

            cfg = json.loads((run_dir / "framework_cfg.json").read_text())
            cfg["deadline"] = time.time() - 1
            (run_dir / "framework_cfg.json").write_text(json.dumps(cfg))
            self.assertTrue(evaluation_budget.budget_status(run_dir)["reached"])


class RoundPolicyTests(unittest.TestCase):
    def _cli(self, ledger: Path, *args) -> dict:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = scheduler_cli.main_with(["round", "--ledger", str(ledger), *args]) \
                if hasattr(scheduler_cli, "main_with") else self._main(ledger, args, out)
        self.assertEqual(code, 0)
        return json.loads(out.getvalue())

    @staticmethod
    def _main(ledger, args, out):
        argv = sys.argv
        sys.argv = ["cli.py", "round", "--ledger", str(ledger), *args]
        try:
            return scheduler_cli.main()
        finally:
            sys.argv = argv

    def test_cycle_switch_and_selections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_dir(
                Path(tmp), deadline=time.time() + 86400,
                round_cfg={"new_candidates": 4, "rewrite_top_k": 2,
                           "round_seconds": 3600, "session_overhead_seconds": 10})
            ledger = run_dir / "ledger.json"
            ledger.write_text(json.dumps({"records": [
                _record("000", 0.9), _record("001", 0.8), _record("002", 0.7)]}))
            _candidate(run_dir, "000", 0.9)
            _candidate(run_dir, "001", 0.8)
            _candidate(run_dir, "002", 0.7, in_progress=True)
            # 001 is slow: one timed evaluation of 1000 s (also the run-wide mean).
            train = run_dir / "candidates" / "001" / "train.py"
            receipt = evaluation_budget.reserve_evaluation(
                train, params={}, phase="phase_a", method="warm")
            evaluation_budget.record_evaluation_completion(
                train, attempt_id=receipt["attempt_id"], duration_seconds=1000.0)

            status = self._cli(ledger, "status")
            self.assertEqual((status["threshold"], status["produced"]), (2, 3))
            self.assertFalse(status["generate"])

            begun = self._cli(ledger, "begin")
            self.assertEqual(begun["phase"], "optimize")
            self.assertAlmostEqual(
                begun["phase_deadline"] - begun["phase_started_at"], 3600, delta=1)

            # 002 is the best but mid-tuning; top-2 by score is then 001/000
            # and both have zero rewrite bouts, so the better score wins.
            rewrite = self._cli(ledger, "select", "--kind", "rewrite")
            self.assertEqual((rewrite["action"], rewrite["run_id"]), ("REWRITE", "001"))
            self.assertEqual(rewrite["reference"], 0.8)
            # Same state again reuses the open rewrite decision.
            again = self._cli(ledger, "select", "--kind", "rewrite")
            self.assertEqual(again["decision_id"], rewrite["decision_id"])
            self.assertTrue(again["reused_open_decision"])

            # Tune: a 24-eval INITIAL bout at the 1000 s mean (24010 s) does
            # not fit the 3600 s quota for any candidate.
            tune = self._cli(ledger, "select", "--kind", "tune")
            self.assertEqual(tune["action"], "STOP")
            self.assertIn("quota", tune["reason"])

            state = load_state(ledger, contract=contract_for(ledger))
            decision = round_policy.select_tune(
                state, run_dir, now=time.time())
            self.assertEqual(decision.action, "STOP")

            # Widen the quota: the best-ranked candidate wins on priority.
            round_policy.save_round_state(run_dir, {
                **round_policy.load_round_state(run_dir),
                "phase_deadline": time.time() + 10 ** 6})
            decision = round_policy.select_tune(state, run_dir)
            self.assertEqual((decision.action, decision.run_id), ("TUNE", "002"))

            ended = self._cli(ledger, "end")
            self.assertEqual((ended["cycle"], ended["cycle_start_count"]), (1, 3))
            status = self._cli(ledger, "status")
            self.assertEqual((status["threshold"], status["produced"]), (4, 0))
            self.assertTrue(status["generate"])

            # Outside a phase the orchestrator's tune query defers.
            self.assertEqual(round_policy.decide(state, run_dir).action, "DEFER")


class KeptRewriteTests(unittest.TestCase):
    def test_confirmation_rebase_and_ledger_commit(self) -> None:
        registry = fixture_registry()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "hard-interactions" / "t1"
            run_dir.mkdir(parents=True)
            (run_dir / "framework_cfg.json").write_text(json.dumps({
                "tuner": {"scheduler_policy": "round_v1",
                          "inner_policy": "hebo24-hebo20"},
                "deadline": time.time() + 3600}))
            (run_dir / "background.md").write_text(background_text(registry))
            point = complete_point(registry)
            (run_dir / "point.json").write_text(json.dumps(point))
            (run_dir / "policy.json").write_text(json.dumps(
                policy_receipt("fresh", [], point, selection_index=1,
                               schema_version=6)))
            ledger = run_dir / "ledger.json"
            args = types.SimpleNamespace(
                ledger=str(ledger), task="hard-interactions", run_id="000",
                kind="optimization", op="fresh", source_run_ids="",
                background=str(run_dir / "background.md"), catalog=None,
                semantic_point=str(run_dir / "point.json"),
                policy_receipt=str(run_dir / "policy.json"),
                idea="Fixture solution at the selected point.",
                change="fixture change", candidate_name_hint="fixture_000",
                description=None, route_provenance=None)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_add_record(args), 0)
            record_run(ledger, "hard-interactions", "000", final_best_score=1.0)
            candidate = _candidate(run_dir, "000", 1.0)
            report = json.loads((candidate / "tune_report.json").read_text())
            report["phase_c"] = {"stages": [
                {"method": "hebo", "status": "ok", "bout_index": 0,
                 "trials": [{"params": {"x": 0.5}, "score": 1.0}]}]}
            (candidate / "tune_report.json").write_text(json.dumps(report))

            # Adjudicate against the ledger reference, then confirm.
            rewrite_bout.snapshot(candidate, 1)
            (candidate / "train.py").write_text(TRAIN_PY + "# better\n")
            result = rewrite_bout.finalize(
                candidate, 1, 0.8, 0.0, "eval-000001", "edit", "basis",
                reference=1.0)
            self.assertEqual(result["outcome"], "kept")
            entry = rewrite_bout.confirm(candidate, 1, 0.9, "eval-000002")
            self.assertAlmostEqual(entry["reference"], 0.85)
            self.assertAlmostEqual(
                rewrite_bout.current_best(candidate, baseline=1.0), 0.85)
            self.assertEqual(len(rewrite_bout.load_bouts(candidate)), 1)

            rebased = rewrite_rebase.rebase(candidate, bout=1, score=0.85, attempts=2)
            self.assertTrue(Path(rebased["archived"]).is_file())
            fresh = json.loads((candidate / "tune_report.json").read_text())
            self.assertEqual(fresh["phase_c"], {"stages": []})
            self.assertAlmostEqual(fresh["phase_a"]["best_warm_score"], 0.85)
            self.assertTrue(round_policy.tune_report_current(run_dir, "000"))

            record = record_rewrite(
                ledger, "hard-interactions", "000", score=0.85,
                report_path=candidate / "tune_report.json")
            self.assertAlmostEqual(record["final_best_score"], 0.85)
            self.assertAlmostEqual(record["best_warm_score"], 0.85)
            self.assertEqual((record["tune"], record["tuning_bouts"]), (False, 0))
            self.assertEqual(record["rewrite_bouts"], 1)
            self.assertEqual(record["applied_incumbent"]["source"], "applied_phase_a")
            self.assertAlmostEqual(record["applied_incumbent"]["score"], 0.85)


class RoundCmd(ExperimentCmd):
    """ExperimentCmd plus the round_v1 tool surface, scripted."""

    def __init__(self, repo: Path):
        super().__init__(repo)
        self.generate_queue: list[bool] = []
        self.done = False
        self.round_calls: list[str] = []

    def __call__(self, args, repo_root, check=True, capture=True, **kw):
        joined = " ".join(str(a) for a in args)
        if "init_run.py" in joined:
            self.calls.append(joined)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "framework_cfg.json").write_text(json.dumps({
                "max_evaluations": None, "per_runtime_limit": None,
                "dimension_strategy": "catalog_subset",
                "tuner": {"scheduler_policy": "round_v1",
                          "inner_policy": "hebo24-hebo20"}}))
            return self._ok("")
        if "evaluation_budget.py" in joined and "status" in joined:
            self.calls.append(joined)
            return self._ok(json.dumps({"evaluations_done": 0,
                                        "reached": self.done}))
        if "scheduler/cli.py" in joined and " round " in joined:
            self.calls.append(joined)
            sub = joined.split(" round ")[1].split()[2:]
            self.round_calls.append(" ".join(sub))
            if sub[0] == "status":
                generate = self.generate_queue.pop(0) if self.generate_queue else True
                return self._ok(json.dumps({
                    "generate": generate, "produced": 1, "threshold": 1,
                    "final_round": False,
                    "config": {**round_policy.DEFAULTS, "rewrite_bouts": 1,
                               "tune_bouts": 1}}))
            if sub[0] == "select" and "rewrite" in sub:
                return self._ok(json.dumps({"action": "STOP", "run_id": None,
                                            "reason": "no eligible",
                                            "decision_id": "dec-0001"}))
            if sub[0] == "select":
                return self._ok(json.dumps({"action": "TUNE", "run_id": "000",
                                            "reason": "best", "decision_id": "dec-0002"}))
            if sub[0] == "end":
                self.done = True
                return self._ok(json.dumps({"cycle": 1, "cycle_start_count": 1}))
            return self._ok(json.dumps({"cycle": 0, "phase_deadline": None}))
        return super().__call__(args, repo_root, check=check, capture=capture, **kw)


class RoundLoopTests(unittest.TestCase):
    def test_seed_set_then_optimization_phase_then_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_task(repo)
            cmd = RoundCmd(repo)
            cmd.generate_queue = [False]   # the seed set is complete: optimize

            def extractor_keep(ctx):
                ledger = cmd._ledger()
                for record in ledger["records"]:
                    if record["run_id"] == ctx.run_id:
                        record["status"] = "keep"
                cmd._save_ledger(ledger)

            runner = FakeSessionRunner([
                {"receipt": {"status": "ok", "background": "background.md",
                             "retrieval_manifest": "background_retrieval.json"},
                 "side_effects": lambda ctx: write_background(ctx.run_dir)},
                {"receipt": {"actions": [{"run_id": "000", "op": "fresh"}]},
                 "side_effects": lambda ctx: cmd([
                     "python", "tools/ledger.py", "add-record", "--run-id", "000"],
                     repo)},
                {"receipt": {"status": "written", "wrote": True,
                             "candidate_dir": "candidates/000"},
                 "side_effects": writer_effect},
                {"receipt": {"run_id": "000", "status": "keep", "ledger_updated": True},
                 "side_effects": extractor_keep},
                {"receipt": {"tuned_run_id": "none", "tuned": False,
                             "ledger_updated": False}},
            ])
            run_experiment("fake-task", "t1", runner=runner, model="m",
                           repo_root=repo, cmd=cmd,
                           semantic_policy="coverage_attempt",
                           scheduler_policy="round_v1", time_budget=3600)
            self.assertEqual(cmd._ledger().get("phase"), "completed")
            roles = [name for name, _ in runner.calls]
            self.assertEqual(roles, ["background-researcher", "idea-generator",
                                     "candidate-writer", "tunable-contract-extractor",
                                     "tuner-orchestrator"])
            self.assertEqual(
                [call for call in cmd.round_calls if not call.startswith("overhead")],
                ["status", "begin", "select --kind rewrite", "select --kind tune",
                 "end"])
            init_call = next(call for call in cmd.calls if "init_run.py" in call)
            self.assertIn("--time-budget 3600", init_call)
            self.assertIn("--scheduler-policy round_v1", init_call)


if __name__ == "__main__":
    unittest.main()
