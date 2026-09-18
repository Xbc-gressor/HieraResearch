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
from unittest import mock

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
from tools.scheduler.store import SchedulerStore  # noqa: E402
from tune_tools import (  # noqa: E402
    _candidate_execution_revision,
    cmd_select_candidate,
)
from driver.loops.experiment import run_experiment  # noqa: E402
from driver.roles import InvocationContext, tuner_target_matches_handoff  # noqa: E402
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

    def test_tune_decision_reuse_is_scoped_to_the_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_dir(Path(tmp), deadline=time.time() + 86400,
                               round_cfg={"round_seconds": 3600})
            ledger = run_dir / "ledger.json"
            ledger.write_text(json.dumps({"records": [_record("000", 0.9)]}))
            _candidate(run_dir, "000", 0.9)
            self._cli(ledger, "begin")

            first = self._cli(ledger, "select", "--kind", "tune")
            self.assertEqual(first["action"], "TUNE")
            # State drift (a partially executed bout does the same): the
            # snapshot identity no longer matches, but this cycle's open
            # decision must be re-issued, not re-decided.
            ledger.write_text(json.dumps({"records": [
                _record("000", 0.9), _record("003", 0.95)]}))
            again = self._cli(ledger, "select", "--kind", "tune")
            self.assertEqual(again["decision_id"], first["decision_id"])
            self.assertTrue(again["reused_open_decision"])

            # Closing the decision frees the next selection.
            SchedulerStore(run_dir).record_outcome(
                first["decision_id"], executed_action="TUNE",
                executed_run_id=first["run_id"], realized_gain=None,
                consumed=0, status="valid")
            fresh = self._cli(ledger, "select", "--kind", "tune")
            self.assertNotEqual(fresh["decision_id"], first["decision_id"])
            self.assertFalse(fresh["reused_open_decision"])

            # A stale open decision from an earlier cycle is never reused.
            self._cli(ledger, "end")
            self._cli(ledger, "begin")
            ledger.write_text(json.dumps({"records": [
                _record("000", 0.9), _record("004", 0.96)]}))
            crossed = self._cli(ledger, "select", "--kind", "tune")
            self.assertNotEqual(crossed["decision_id"], fresh["decision_id"])
            self.assertFalse(crossed["reused_open_decision"])
            self.assertEqual(crossed["round_cycle"], 1)

    def test_tune_decision_not_reused_across_cycle_with_unchanged_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_dir(Path(tmp), deadline=time.time() + 86400,
                               round_cfg={"round_seconds": 3600})
            ledger = run_dir / "ledger.json"
            ledger.write_text(json.dumps({"records": [_record("000", 0.9)]}))
            _candidate(run_dir, "000", 0.9)
            self._cli(ledger, "begin")

            first = self._cli(ledger, "select", "--kind", "tune")
            self.assertEqual(first["action"], "TUNE")

            # Cross the phase boundary with the ledger and the evidence log
            # untouched: snapshot identity still matches the open decision,
            # but the earlier cycle's TUNE decision must not be resurrected.
            self._cli(ledger, "end")
            self._cli(ledger, "begin")
            second = self._cli(ledger, "select", "--kind", "tune")
            self.assertNotEqual(second["decision_id"], first["decision_id"])
            self.assertFalse(second["reused_open_decision"])
            self.assertEqual(second["round_cycle"], 1)


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
        self.rewrite_selection: dict = {
            "action": "REWRITE", "run_id": "000",
            "reason": "scripted", "decision_id": "dec-r1",
            "reference": 1.0,
            "evidence_mode": {"overhead_seconds": 10.0}}
        self.tune_selection: dict | None = None

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
                return ok(json.dumps(self.rewrite_selection))
            if "select" in joined:
                if self.tune_selection is not None:
                    return ok(json.dumps(self.tune_selection))
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
                EventsLog(run_dir), tune=lambda no, selection: {"tuned": False},
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
                EventsLog(run_dir), tune=lambda no, selection: {"tuned": False},
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


class ConcurrentClimbTests(unittest.TestCase):
    """Two rewrite channels: the exclude set is part of the decision identity
    and the channels never climb the same candidate."""

    def _cli(self, ledger: Path, *args) -> dict:
        out = io.StringIO()
        argv = sys.argv
        sys.argv = ["cli.py", "round", "--ledger", str(ledger), *args]
        try:
            with contextlib.redirect_stdout(out):
                code = scheduler_cli.main()
        finally:
            sys.argv = argv
        self.assertEqual(code, 0)
        return json.loads(out.getvalue())

    def test_exclude_set_is_part_of_the_decision_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_dir(
                Path(tmp), deadline=time.time() + 86400,
                round_cfg={"rewrite_top_k": 3, "round_seconds": 3600,
                           "session_overhead_seconds": 10})
            ledger = run_dir / "ledger.json"
            ledger.write_text(json.dumps({"records": [
                _record("000", 0.9), _record("001", 0.8)]}))
            _candidate(run_dir, "000", 0.9)
            _candidate(run_dir, "001", 0.8)
            self._cli(ledger, "begin")

            first = self._cli(ledger, "select", "--kind", "rewrite")
            self.assertEqual((first["action"], first["run_id"]), ("REWRITE", "001"))
            # channel B, with A's candidate in flight: a different decision
            second = self._cli(ledger, "select", "--kind", "rewrite",
                               "--exclude", "001")
            self.assertEqual((second["action"], second["run_id"]), ("REWRITE", "000"))
            self.assertNotEqual(second["decision_id"], first["decision_id"])
            self.assertEqual(second["exclude_run_ids"], ["001"])
            self.assertFalse(second["reused_open_decision"])
            row = next(r for r in second["evidence_mode"]["ranked"]
                       if r["run_id"] == "001")
            self.assertIn("in_flight", row["ineligible"])
            # both in flight: STOP, reused only for the identical exclude set
            stop = self._cli(ledger, "select", "--kind", "rewrite",
                             "--exclude", "000,001")
            self.assertEqual(stop["action"], "STOP")
            again = self._cli(ledger, "select", "--kind", "rewrite",
                              "--exclude", "001,000")
            self.assertEqual(again["decision_id"], stop["decision_id"])
            self.assertTrue(again["reused_open_decision"])
            # A's own re-query (nothing excluded) gets A's open decision back,
            # never B's STOP
            reissued = self._cli(ledger, "select", "--kind", "rewrite")
            self.assertEqual(reissued["decision_id"], first["decision_id"])
            self.assertTrue(reissued["reused_open_decision"])

    def test_two_channels_climb_different_candidates(self) -> None:
        import threading
        from driver.events import EventsLog
        from driver.loops import rounds
        from driver.receipts import ReceiptStore

        class TwoChannelCmd(ClimbCmd):
            lock = threading.Lock()

            def __call__(self, args, repo_root, check=True, capture=True, **kw):
                import subprocess
                args = [str(a) for a in args]
                joined = " ".join(args)
                if "scheduler/cli.py" in joined and "select" in joined \
                        and "rewrite" in joined:
                    self.selects += 1
                    self.calls.append(args)
                    exclude = (args[args.index("--exclude") + 1].split(",")
                               if "--exclude" in args else [])
                    free = [r for r in ("000", "001") if r not in exclude]
                    if not free:
                        view = {"action": "STOP", "run_id": None,
                                "reason": "all in flight", "decision_id": "dec-s"}
                    else:
                        view = {"action": "REWRITE", "run_id": free[0],
                                "reason": "scripted",
                                "decision_id": f"dec-{free[0]}",
                                "reference": 1.0,
                                "evidence_mode": {"overhead_seconds": 10.0}}
                    return subprocess.CompletedProcess(args, 0, json.dumps(view), "")
                with self.lock:  # the in-process bout CLI swaps sys.argv/stdout
                    return super().__call__(args, repo_root, check, capture, **kw)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "fake-task" / "t1"
            run_dir.mkdir(parents=True)
            (run_dir / "framework_cfg.json").write_text(json.dumps({
                "pipeline": {"rewrite_concurrency": 2}}))
            (run_dir / "ledger.json").write_text(json.dumps({
                "records": [_record("000", 1.0), _record("001", 1.0)]}))
            for run_id in ("000", "001"):
                candidate = run_dir / "candidates" / run_id
                candidate.mkdir(parents=True)
                (candidate / "train.py").write_text(ClimbCmd.V[0])
                (candidate / "tune_report.json").write_text(json.dumps({
                    "phase_a": {"best_warm_score": 1.0}, "phase_c": {"stages": []}}))
            cmd = TwoChannelCmd(Path(tmp), run_dir, eval_script=[0.9] * 4)

            def edit(ctx):
                (Path(ctx.extra["candidate_dir"]) / "train.py").write_text(
                    ClimbCmd.V[1])

            runner = FakeSessionRunner([
                {"receipt": {"edited": True, "summary": "a", "basis": "h"},
                 "side_effects": edit},
                {"receipt": {"edited": True, "summary": "b", "basis": "h"},
                 "side_effects": edit},
            ])
            config = {**round_policy.DEFAULTS, "rewrite_bouts": 2,
                      "rewrite_max_bouts": 1, "tune_bouts": 0,
                      "noise_margin": 0.0}
            # EventsLog's stdout line would land in the OTHER channel's
            # in-process bout-CLI capture (redirect_stdout is process-wide);
            # production runs the CLI as a subprocess with its own pipe.
            with mock.patch("driver.events.print", create=True):
                progressed = rounds.optimization_phase(
                    runner, ReceiptStore(run_dir), "fake-task", "t1", run_dir,
                    1, {"result": {"metric": "neg_acc"}}, Path(tmp), cmd,
                    EventsLog(run_dir),
                    tune=lambda no, selection: {"tuned": False},
                    config=config)
            self.assertTrue(progressed)
            events = [json.loads(line) for line in
                      (run_dir / "driver_events.jsonl").read_text().splitlines()
                      if line.strip()]
            climbs = [e for e in events if e.get("kind") == "rewrite_climb"]
            self.assertEqual(sorted(c["run_id"] for c in climbs), ["000", "001"])
            self.assertEqual([c["stop"] for c in climbs], ["bout_cap", "bout_cap"])
            selects = [e for e in events if e.get("kind") == "round_select"]
            self.assertEqual(sorted(len(e["exclude_run_ids"]) for e in selects),
                             [0, 1])
            for run_id in ("000", "001"):
                bouts = rewrite_bout.load_bouts(run_dir / "candidates" / run_id)
                self.assertEqual([b["outcome"] for b in bouts], ["kept"])
            self.assertEqual(
                next(e for e in events if e.get("kind") == "rewrite_channels")
                ["concurrency"], 2)

    def test_admission_counts_the_other_channels_decaying_commitment(self) -> None:
        import threading
        from driver.loops import rounds

        class QuotaCmd:
            def __call__(self, args, repo_root, check=True, **kw):
                import subprocess
                return subprocess.CompletedProcess(
                    [str(a) for a in args], 0, json.dumps({
                        "reached": False, "phase_quota_remaining_seconds": 90.0,
                        "per_candidate": [{"run_id": "000", "evals": 2,
                                           "mean_seconds": 20.0}]}), "")

        # expected = 2 × 20 + 10 = 50 against a 90 s quota
        coord = rounds._ClimbCoordinator(2)
        coord.commitments[1] = (50.0, time.monotonic() - 20.0)  # 30 s left
        self.assertEqual(rounds._admit_step(
            Path("."), "000", Path("."), QuotaCmd(), 10.0, 0.0, coord, 0), "ok")
        self.assertEqual(coord.commitments[0][0], 50.0)
        coord.commitments.pop(0)

        coord.commitments[1] = (50.0, time.monotonic())  # squeezes 0 out
        admitted = []

        def channel0():
            admitted.append(rounds._admit_step(
                Path("."), "000", Path("."), QuotaCmd(), 10.0, 0.0, coord, 0))

        thread = threading.Thread(target=channel0)
        thread.start()
        time.sleep(0.2)
        self.assertEqual(admitted, [])  # waiting, not ending the climb
        with coord.cond:
            coord.commitments.pop(1)  # the other bout closes
            coord.cond.notify_all()
        thread.join(timeout=5)
        self.assertEqual(admitted, ["ok"])


class SelectCandidateDispatchTests(unittest.TestCase):
    """Under round_v1, select-candidate answers from the scheduler, not the
    legacy percentile gate."""

    def test_round_v1_returns_scheduler_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_dir(Path(tmp), deadline=time.time() + 86400,
                               round_cfg={"round_seconds": 3600})
            ledger = run_dir / "ledger.json"
            ledger.write_text(json.dumps({"records": [
                _record("000", 0.9), _record("002", 0.7)]}))
            _candidate(run_dir, "000", 0.9)
            _candidate(run_dir, "002", 0.7)
            round_policy.begin_optimization(run_dir)

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cmd_select_candidate(types.SimpleNamespace(
                    ledger=str(ledger), top_percentile=None, n_min=None)), 0)
            result = json.loads(out.getvalue())
            self.assertEqual(result["run_id"], "002")
            self.assertEqual(result["scheduler"]["action"], "TUNE")
            self.assertTrue(result["scheduler"]["decision_id"])
            # the legacy selector's receipt shape is absent
            self.assertNotIn("percentile", result)
            self.assertNotIn("is_continuation", result)


class TunerHandoffPostconditionTests(unittest.TestCase):
    def _ctx(self, run_dir: Path, handoff: dict | None):
        extra = ({"scheduler_selection": json.dumps(handoff)}
                 if handoff is not None else {})
        return InvocationContext(
            task="fake-task", tag="t1", run_dir=run_dir, invocation_id=1,
            extra=extra)

    def test_receipt_must_echo_the_handoff_target(self) -> None:
        from driver.receipts import ReceiptStore

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            handoff = {"decision_id": "dec-0001", "run_id": "005"}
            ReceiptStore(run_dir).persist_receipt(
                "tuner-orchestrator", 1,
                {"tuned_run_id": "002", "tuned": True, "ledger_updated": True})
            problem = tuner_target_matches_handoff(self._ctx(run_dir, handoff))
            self.assertIsNotNone(problem)
            self.assertIn("005", problem)

            ReceiptStore(run_dir).persist_receipt(
                "tuner-orchestrator", 1,
                {"tuned_run_id": "005", "tuned": True, "ledger_updated": True},
                allow_replace=True)
            self.assertIsNone(
                tuner_target_matches_handoff(self._ctx(run_dir, handoff)))

    def test_unpinned_invocations_are_unchecked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                tuner_target_matches_handoff(self._ctx(Path(tmp), None)))


class TuneBindingTests(unittest.TestCase):
    """The tune selection the driver commits is the bout the tuner runs:
    handoff contents, outcome binding, and mismatch rejection."""

    def _setup_run(self, tmp: Path):
        run_dir = tmp / "runs" / "fake-task" / "t1"
        run_dir.mkdir(parents=True)
        (run_dir / "ledger.json").write_text(json.dumps(
            {"records": [_record("000", 1.0)]}))
        candidate = run_dir / "candidates" / "000"
        candidate.mkdir(parents=True)
        (candidate / "train.py").write_text(ClimbCmd.V[0])
        (candidate / "tune_report.json").write_text(json.dumps({
            "phase_a": {"best_warm_score": 1.0}, "phase_c": {"stages": []}}))
        return run_dir

    def _events(self, run_dir: Path) -> list[dict]:
        path = run_dir / "driver_events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]

    def _phase(self, tmp: Path, run_dir: Path, cmd, tune) -> bool:
        from driver.events import EventsLog
        from driver.loops import rounds
        from driver.receipts import ReceiptStore
        config = {**round_policy.DEFAULTS, "rewrite_bouts": 1,
                  "tune_bouts": 1, "noise_margin": 0.0}
        return rounds.optimization_phase(
            FakeSessionRunner([]), ReceiptStore(run_dir), "fake-task", "t1",
            run_dir, 1, {"result": {"metric": "neg_acc"}}, Path(tmp), cmd,
            EventsLog(run_dir), tune=tune, config=config)

    def _tune_cmd(self, tmp: Path, run_dir: Path) -> ClimbCmd:
        cmd = ClimbCmd(Path(tmp), run_dir, eval_script=[])
        cmd.rewrite_selection = {"action": "STOP", "run_id": None,
                                 "reason": "no eligible",
                                 "decision_id": "dec-r0"}
        cmd.tune_selection = {"action": "TUNE", "run_id": "000",
                              "reason": "scripted", "decision_id": "dec-t9",
                              "policy_version": "scheduler-round-v2",
                              "state_snapshot_id": "snap-1",
                              "evidence_cursor": 7, "bout_trials": 24}
        return cmd

    def test_handoff_and_outcome_bind_to_the_same_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._setup_run(Path(tmp))
            cmd = self._tune_cmd(Path(tmp), run_dir)
            seen = {}

            def tune(no, selection):
                seen.update(round_no=no, selection=selection)
                return {"tuned": True, "tuned_run_id": "000",
                        "ledger_updated": True}

            self.assertTrue(self._phase(Path(tmp), run_dir, cmd, tune))
            selection = seen["selection"]
            self.assertEqual(seen["round_no"], 1)
            self.assertEqual(
                (selection["decision_id"], selection["run_id"],
                 selection["policy_version"], selection["state_snapshot_id"],
                 selection["evidence_cursor"], selection["bout_trials"]),
                ("dec-t9", "000", "scheduler-round-v2", "snap-1", 7, 24))
            record = next(call for call in cmd.calls
                          if "record" in call and "dec-t9" in call)
            self.assertEqual(record[record.index("--action") + 1], "TUNE")
            self.assertEqual(record[record.index("--status") + 1], "valid")
            self.assertEqual(record[record.index("--consumed") + 1], "0")
            event = next(e for e in self._events(run_dir)
                         if e.get("kind") == "tune_bout")
            self.assertEqual(
                (event["selected_run_id"], event["executed_run_id"],
                 event["decision_id"], event["tuned"]),
                ("000", "000", "dec-t9", True))

    def test_blocked_session_still_closes_the_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._setup_run(Path(tmp))
            cmd = self._tune_cmd(Path(tmp), run_dir)
            from driver.events import EventsLog
            from driver.loops.common import RunBlocked
            from driver.loops.experiment import _tune
            from driver.receipts import ReceiptStore

            with self.assertRaises(RunBlocked):
                _tune(FakeSessionRunner([]), ReceiptStore(run_dir), "fake-task",
                      "t1", run_dir, 1, Path(tmp), cmd, EventsLog(run_dir),
                      selection=dict(cmd.tune_selection))
            record = next(call for call in cmd.calls
                          if "record" in call and "dec-t9" in call)
            self.assertEqual(record[record.index("--action") + 1], "TUNE")
            self.assertEqual(record[record.index("--status") + 1],
                             "infra_failure")

    def test_failed_tune_charges_the_attempts_the_bout_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._setup_run(Path(tmp))
            cmd = self._tune_cmd(Path(tmp), run_dir)
            from driver.events import EventsLog
            from driver.loops.common import RunBlocked
            from driver.loops.experiment import _tune
            from driver.receipts import ReceiptStore

            def admit_two_evals(ctx) -> None:
                # The pinned Phase-C job evaluated twice before the tuner
                # session collapsed; the outcome must charge both, not zero.
                with (run_dir / "evaluation_attempts.jsonl").open(
                        "w", encoding="utf-8") as handle:
                    for _ in range(2):
                        handle.write(json.dumps({
                            "schema_version": 1, "kind": "score_attempt",
                            "phase": "phase_c", "run_id": "000"}) + "\n")

            runner = FakeSessionRunner([
                {"receipt": None,
                 "side_effects": admit_two_evals,
                 "fail": ["fake runner: bout collapsed mid-evaluation"]},
            ])

            with self.assertRaises(RunBlocked):
                _tune(runner, ReceiptStore(run_dir), "fake-task", "t1",
                      run_dir, 1, Path(tmp), cmd, EventsLog(run_dir),
                      selection=dict(cmd.tune_selection))
            record = next(call for call in cmd.calls
                          if "record" in call and "dec-t9" in call)
            self.assertEqual(record[record.index("--status") + 1],
                             "infra_failure")
            self.assertEqual(record[record.index("--consumed") + 1], "2")

    def test_mismatched_receipt_is_never_reattributed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._setup_run(Path(tmp))
            cmd = self._tune_cmd(Path(tmp), run_dir)

            def tune(no, selection):
                return {"tuned": True, "tuned_run_id": "002",
                        "ledger_updated": True}

            self.assertTrue(self._phase(Path(tmp), run_dir, cmd, tune))
            record = next(call for call in cmd.calls
                          if "record" in call and "dec-t9" in call)
            self.assertEqual(record[record.index("--action") + 1], "TUNE")
            self.assertEqual(record[record.index("--status") + 1],
                             "infra_failure")
            self.assertEqual(record[record.index("--consumed") + 1], "0")
            event = next(e for e in self._events(run_dir)
                         if e.get("kind") == "tune_target_mismatch")
            self.assertEqual(
                (event["selected_run_id"], event["executed_run_id"]),
                ("000", "002"))
            # no tune_bout event: the result was not accepted as a bout
            self.assertFalse(any(e.get("kind") == "tune_bout"
                                 for e in self._events(run_dir)))


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
            # the tuner invocation was pinned to the selected candidate and
            # carried the committed decision as a structured handoff
            tuner_ctx = next(ctx for name, ctx in runner.calls
                             if name == "tuner-orchestrator")
            self.assertEqual(tuner_ctx.run_id, "000")
            handoff = json.loads(tuner_ctx.extra["scheduler_selection"])
            self.assertEqual(handoff["decision_id"], "dec-0002")
            self.assertEqual(handoff["run_id"], "000")
            # the tune decision was closed against the same id
            tune_record = next(call for call in cmd.calls
                               if "cli.py record" in call and "dec-0002" in call)
            self.assertIn("--action TUNE", tune_record)
            self.assertIn("--status valid", tune_record)


if __name__ == "__main__":
    unittest.main()
