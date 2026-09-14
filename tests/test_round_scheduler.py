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


class LedgerViewTests(unittest.TestCase):
    def test_run_best_materializes_one_view_for_all_candidates(self) -> None:
        import os
        from unittest import mock

        import tools.evaluation_records as evaluation_records
        from tools.evaluation_records import EvaluationRecord, append_record

        from driver.loops import rounds as driver_rounds

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "unit" / "view"
            run_dir.mkdir(parents=True)
            records = []
            for candidate, scores in (("001", (0.5, 0.4)), ("002", (0.3, 0.2))):
                record_path = run_dir / f"{candidate}.jsonl"
                digests = []
                for score in scores:
                    row = EvaluationRecord(
                        task_id="unit", candidate_id=candidate,
                        input_revision="r", output_artifact_digest="o",
                        contract_version="1", stage="proxy", fidelity="fast",
                        metric_name="loss", score=score).to_dict()
                    append_record(record_path, row)
                    digests.append(row["record_digest"])
                records.append({
                    "run_id": candidate,
                    "evaluation_record_digests": digests,
                    "evaluation_record_paths": {
                        digest: str(record_path) for digest in digests},
                })
            (run_dir / "ledger.json").write_text(json.dumps(
                {"records": records}))
            calls = []
            real_read = evaluation_records.read_records

            def counting_read(path):
                calls.append(str(path))
                return real_read(path)

            with mock.patch.object(evaluation_records, "read_records",
                                   counting_read), \
                 mock.patch.dict(os.environ, {
                     "EVALUATION_STAGE": "proxy",
                     "EVALUATION_FIDELITY": "fast"}):
                self.assertEqual(driver_rounds._run_best(run_dir), 0.2)
            # One read per candidate file, not one per (record, digest).
            self.assertEqual(len(calls), 2)


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
            snapshot = rewrite_bout.snapshot(candidate, 1)
            # A structural rewrite may introduce a new tunable dimension.
            (candidate / "train.py").write_text(
                'PARAM_SCHEMA = {"x": "float", "offset": "float"}\n'
                'SEARCH_SPACE = {"x": ("float", 0.0, 1.0), '
                '"offset": ("float", 0.0, 2.0)}\n'
                'BASE_PARAMS: dict = {"x": 0.7, "offset": 1.0}\n'
                'def make_model(params):\n'
                '    return params["x"] + params["offset"]\n')
            self.assertTrue(rewrite_bout.params_equal(candidate, snapshot))
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
            self.assertEqual(fresh["phase_a"]["best_warm_params"],
                             {"x": 0.7, "offset": 1.0})
            self.assertIn("offset", fresh["phase_a"]["search_space"])
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


class ClimbCmd:
    """The round_v1 tool surface scripted for one rewrite climb.

    rewrite_bout.py dispatches to the real CLI in-process; everything else
    is canned. ``eval_script`` holds the rewrite_eval scores in call order
    (adjudication eval and the confirmation re-eval are separate calls).
    """

    V = ['BASE_PARAMS = {"x": 0.5}\n# v1\n',
         'BASE_PARAMS = {"x": 0.5}\n# v2\n',
         'BASE_PARAMS = {"x": 0.5}\n# v3\n',
         'BASE_PARAMS = {"x": 0.5}\n# v4\n']

    def __init__(self, repo: Path, run_dir: Path, eval_script: list[float],
                 quota: float | None = None):
        self.repo = repo
        self.run_dir = run_dir
        self.eval_script = list(eval_script)
        self.quota = quota
        self.reserved = 0
        self.calls: list[list[str]] = []
        self.selects = 0

    def _real_bout_cli(self, args: list[str]) -> "subprocess.CompletedProcess":
        import subprocess
        idx = next(i for i, a in enumerate(args) if a.endswith("rewrite_bout.py"))
        old_argv = sys.argv
        sys.argv = ["rewrite_bout.py"] + args[idx + 1:]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = rewrite_bout.main()
        except SystemExit as exc:
            return subprocess.CompletedProcess(
                args, exc.code if isinstance(exc.code, int) else 1,
                buf.getvalue(), "" if isinstance(exc.code, int) else str(exc.code))
        finally:
            sys.argv = old_argv
        return subprocess.CompletedProcess(args, code, buf.getvalue(), "")

    def __call__(self, args, repo_root, check=True, capture=True, **kw):
        import subprocess
        args = [str(a) for a in args]
        self.calls.append(args)
        joined = " ".join(args)
        ok = lambda stdout="": subprocess.CompletedProcess(args, 0, stdout, "")

        if "evaluation_budget.py" in joined:
            return ok(json.dumps({
                "evaluations_done": self.reserved, "reached": False,
                "phase_quota_remaining_seconds": self.quota,
                "phase_quota_reached": (self.quota is not None
                                        and self.quota <= 0),
                "per_candidate": []}))
        if "rewrite_bout.py" in joined:
            return self._real_bout_cli(args)
        if "scheduler/cli.py" in joined:
            if " begin" in joined:
                return ok(json.dumps({"cycle": 0, "phase": "optimize",
                                      "phase_deadline": None}))
            if "select" in joined and "rewrite" in joined:
                self.selects += 1
                return ok(json.dumps({
                    "action": "REWRITE", "run_id": "000",
                    "reason": "scripted", "decision_id": "dec-r1",
                    "reference": 1.0,
                    "evidence_mode": {"overhead_seconds": 10.0}}))
            if "select" in joined:
                return ok(json.dumps({"action": "STOP", "run_id": None,
                                      "reason": "no eligible",
                                      "decision_id": "dec-t1"}))
            if " end" in joined:
                return ok(json.dumps({"cycle": 1, "cycle_start_count": 1}))
            return ok(json.dumps({"ok": True}))  # record / overhead
        if "rewrite_context.py" in joined:
            candidate = Path(args[args.index("--candidate") + 1])
            (candidate / "_rewrite").mkdir(exist_ok=True)
            (candidate / "_rewrite" / "context.md").write_text("# ctx\n")
            return ok(json.dumps({"context_md": "context.md"}))
        if "rewrite_eval.py" in joined:
            self.reserved += 1
            attempt_id = f"eval-{self.reserved:06d}"
            score = self.eval_script.pop(0) if self.eval_script else 0.0
            return ok(json.dumps({"attempt_id": attempt_id, "score": score,
                                  "error": None, "stage": "eval"}))
        if "preflight_candidate.py" in joined:
            return ok("")
        # rewrite_rebase.py, ledger.py record-rewrite, uv sync, ...
        return ok("{}")


class RewriteClimbTests(unittest.TestCase):
    """One REWRITE decision is one climb: stepped hillclimbing on the
    selected candidate until it stalls, with a single decision record."""

    def _setup_run(self, tmp: Path):
        run_dir = tmp / "runs" / "fake-task" / "t1"
        run_dir.mkdir(parents=True)
        (run_dir / "ledger.json").write_text(json.dumps({
            "records": [_record("000", 1.0)]}))
        candidate = run_dir / "candidates" / "000"
        candidate.mkdir(parents=True)
        (candidate / "train.py").write_text(ClimbCmd.V[0])
        (candidate / "tune_report.json").write_text(json.dumps({
            "phase_a": {"best_warm_score": 1.0}, "phase_c": {"stages": []}}))
        return run_dir, candidate

    def _events(self, run_dir: Path) -> list[dict]:
        path = run_dir / "driver_events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]

    def test_climb_kept_then_stalls_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, candidate = self._setup_run(Path(tmp))
            cmd = ClimbCmd(Path(tmp), run_dir,
                           eval_script=[0.9, 0.9, 0.95, 0.95])
            from driver.events import EventsLog
            from driver.loops import rounds
            from driver.receipts import ReceiptStore
            runner = FakeSessionRunner([
                {"receipt": {"edited": True, "summary": "lever a",
                             "basis": "hyp-1"},
                 "side_effects": lambda ctx: (candidate / "train.py")
                 .write_text(ClimbCmd.V[1])},
                {"receipt": {"edited": True, "summary": "lever b",
                             "basis": "hyp-1"},
                 "side_effects": lambda ctx: (candidate / "train.py")
                 .write_text(ClimbCmd.V[2])},
                {"receipt": {"edited": True, "summary": "lever c",
                             "basis": "hyp-1"},
                 "side_effects": lambda ctx: (candidate / "train.py")
                 .write_text(ClimbCmd.V[3])},
            ])
            config = {**round_policy.DEFAULTS, "rewrite_bouts": 1,
                      "tune_bouts": 1, "rewrite_stall_after": 2,
                      "noise_margin": 0.0}
            progressed = rounds.optimization_phase(
                runner, ReceiptStore(run_dir), "fake-task", "t1", run_dir, 1,
                {"result": {"metric": "neg_acc"}}, Path(tmp), cmd,
                EventsLog(run_dir), tune=lambda no: {"tuned": False},
                config=config)

            self.assertTrue(progressed)
            self.assertEqual(cmd.selects, 1)  # one decision for the climb
            # step 1 kept (1.0 -> 0.9, confirmation folds to 0.9), steps 2
            # and 3 reverted_worse: stall_after=2 ends the climb
            bouts = rewrite_bout.load_bouts(candidate)
            self.assertEqual([b["outcome"] for b in bouts],
                             ["kept", "reverted_worse", "reverted_worse"])
            self.assertEqual([b["bout"] for b in bouts], [1, 2, 3])
            self.assertEqual((candidate / "train.py").read_text(),
                             ClimbCmd.V[1])
            # one decision record: 4 attempts (adjudication + confirmation
            # for the kept step), gain measured against the initial reference
            record = next(call for call in cmd.calls
                          if "record" in call and "scheduler" in " ".join(call))
            self.assertIn("dec-r1", record)
            self.assertEqual(record[record.index("--consumed") + 1], "4")
            self.assertEqual(record[record.index("--status") + 1], "valid")
            self.assertAlmostEqual(
                float(record[record.index("--gain") + 1]), 0.1)
            # the editor session persisted and resumed across steps
            calls = [ctx for name, ctx in runner.calls
                     if name == "rewrite-editor"]
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[1].resume_session_id, "fake-sess-0001")
            self.assertEqual(calls[2].extra["current_best"], 0.9)
            self.assertEqual(calls[2].extra["last_outcome"], "reverted_worse")
            # the kept step was committed; the climb event names the stop
            self.assertTrue(any("record-rewrite" in " ".join(call)
                                for call in cmd.calls))
            climb = next(e for e in self._events(run_dir)
                         if e.get("kind") == "rewrite_climb")
            self.assertEqual((climb["steps"], climb["kept"], climb["stop"]),
                             (3, 1, "stalled"))
            # one rewrite select, then the tune select, per-step overheads
            overheads = [call for call in cmd.calls if "overhead" in call]
            self.assertEqual(len(overheads), 3)

    def test_climb_stops_before_first_step_when_quota_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, candidate = self._setup_run(Path(tmp))
            cmd = ClimbCmd(Path(tmp), run_dir, eval_script=[], quota=0.0)
            from driver.events import EventsLog
            from driver.loops import rounds
            from driver.receipts import ReceiptStore
            runner = FakeSessionRunner([])
            config = {**round_policy.DEFAULTS, "rewrite_bouts": 1,
                      "tune_bouts": 1, "noise_margin": 0.0}
            progressed = rounds.optimization_phase(
                runner, ReceiptStore(run_dir), "fake-task", "t1", run_dir, 1,
                {"result": {"metric": "neg_acc"}}, Path(tmp), cmd,
                EventsLog(run_dir), tune=lambda no: {"tuned": False},
                config=config)

            # the selection was already committed, but no step fits the
            # quota: zero-step climb, no editor session, no journal
            self.assertFalse(progressed)
            self.assertEqual(runner.calls, [])
            self.assertFalse((candidate / "_rewrite" / "bouts.jsonl").exists())
            record = next(call for call in cmd.calls
                          if "record" in call and "scheduler" in " ".join(call))
            self.assertEqual(record[record.index("--consumed") + 1], "0")
            self.assertEqual(record[record.index("--status") + 1],
                             "infra_failure")
            climb = next(e for e in self._events(run_dir)
                         if e.get("kind") == "rewrite_climb")
            self.assertEqual((climb["steps"], climb["stop"]),
                             (0, "round_quota"))


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
