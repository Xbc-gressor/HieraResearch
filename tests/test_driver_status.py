import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.status import budget_status, compact_status, derive_phase  # noqa: E402


class DerivePhaseTests(unittest.TestCase):
    def test_completed_from_budget(self) -> None:
        ledger = {"records": [{"run_id": "001", "status": "keep"}]}
        cfg = {"max_evaluations": 3}
        self.assertEqual(derive_phase(ledger, cfg, 3), "completed")

    def test_running_below_budget(self) -> None:
        ledger = {"records": [{"run_id": "001", "status": "keep"}]}
        cfg = {"max_evaluations": 3}
        self.assertEqual(derive_phase(ledger, cfg, 1), "running")

    def test_running_when_unbounded(self) -> None:
        self.assertEqual(derive_phase({"records": []}, {}, 100), "running")

    def test_blocked_from_run_state(self) -> None:
        ledger = {"run_state": {"phase": "blocked"}, "records": []}
        self.assertEqual(derive_phase(ledger, {}, 0), "blocked")

    def test_top_level_phase_key_is_not_read(self) -> None:
        # _run_snapshot only honors run_state.phase == "blocked"; a top-level
        # "phase" key and an explicit "completed" never win.
        self.assertEqual(
            derive_phase({"phase": "blocked", "records": []}, {}, 0), "running")
        self.assertEqual(
            derive_phase({"phase": "completed", "records": []}, {}, 0), "running")

    def test_no_ledger_is_running(self) -> None:
        self.assertEqual(derive_phase(None, {"max_evaluations": 3}, 0), "running")

    def test_completion_requires_terminal_records(self) -> None:
        # Budget reached but a record is still pending -> running
        # (_run_snapshot: budget_reached_pending_resolution).
        ledger = {"records": [{"run_id": "001", "status": "pending"}]}
        self.assertEqual(
            derive_phase(ledger, {"max_evaluations": 3}, 3), "running")

    def test_completion_requires_fresh_experience(self) -> None:
        # dag_revision ahead of the experience cursor -> running
        # (_run_snapshot: final_experience_refresh_required).
        ledger = {
            "records": [{"run_id": "001", "status": "keep"}],
            "dag_revision": 2,
            "experience": {"dag_revision": 1},
        }
        self.assertEqual(
            derive_phase(ledger, {"max_evaluations": 3}, 3), "running")
        ledger["experience"]["dag_revision"] = 2
        self.assertEqual(
            derive_phase(ledger, {"max_evaluations": 3}, 3), "completed")

    def test_attempted_sums_record_trials(self) -> None:
        # Ported from test_snapshot_derives_completion_from_budget: the record
        # trials sum counts even when the strict budget counter is lower.
        cfg = {"max_evaluations": 3}
        ledger = {"records": [
            {"run_id": "001", "status": "keep", "trials_attempted": 2}]}
        self.assertEqual(derive_phase(ledger, cfg, 0), "running")
        ledger["records"][0]["trials_attempted"] = 3
        self.assertEqual(derive_phase(ledger, cfg, 0), "completed")

    def test_attempted_falls_back_to_completed_then_warm_start(self) -> None:
        cfg = {"max_evaluations": 2}
        ledger = {"records": [
            {"run_id": "001", "status": "keep", "trials_completed": 2}]}
        self.assertEqual(derive_phase(ledger, cfg, 0), "completed")
        ledger = {"records": [
            {"run_id": "001", "status": "keep", "warm_start_K": 2}]}
        self.assertEqual(derive_phase(ledger, cfg, 0), "completed")

    def test_attempted_is_max_of_records_and_counter(self) -> None:
        ledger = {"records": [{"run_id": "001", "status": "keep"}]}
        self.assertEqual(
            derive_phase(ledger, {"max_evaluations": 3}, 3), "completed")

    def test_budget_falls_back_to_run_state(self) -> None:
        ledger = {
            "records": [{"run_id": "001", "status": "keep"}],
            "run_state": {"evaluation_budget": 2},
        }
        self.assertEqual(derive_phase(ledger, {}, 2), "completed")

    def test_bool_or_non_int_budget_is_unbounded(self) -> None:
        ledger = {"records": [{"run_id": "001", "status": "keep"}]}
        self.assertEqual(
            derive_phase(ledger, {"max_evaluations": True}, 100), "running")
        self.assertEqual(
            derive_phase(ledger, {"max_evaluations": "3"}, 100), "running")


def _fake_cmd(budget: dict, brief: dict):
    def cmd(argv, repo_root):
        if "evaluation_budget.py" in argv[1]:
            return SimpleNamespace(stdout=json.dumps(budget))
        return SimpleNamespace(stdout=json.dumps(brief))
    return cmd


class BudgetStatusTests(unittest.TestCase):
    def test_parses_injected_cmd_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = _fake_cmd({"evaluations_done": 2}, {})
            out = budget_status(Path(tmp), ROOT, cmd)
            self.assertEqual(out["evaluations_done"], 2)


class CompactStatusTests(unittest.TestCase):
    def test_completed_run_merges_ledger_brief(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": 3}))
            (run_dir / "ledger.json").write_text(json.dumps({
                "task": "unit", "tag": "t5",
                "records": [{"run_id": "001", "status": "keep",
                             "trials_attempted": 3}],
            }))
            brief = {"next_run_id": "002", "best_run_id": "001",
                     "best_score": 0.5, "last_run_id": "001",
                     "last_status": "keep"}
            cmd = _fake_cmd({"evaluations_done": 3}, brief)
            status = compact_status("unit", "t5", run_dir, ROOT, cmd)
            self.assertEqual(status["phase"], "completed")
            self.assertEqual(status["stop_condition"],
                             "evaluation_budget_reached")
            self.assertEqual(status["task"], "unit")
            self.assertEqual(status["tag"], "t5")
            self.assertEqual(status["run_dir"], str(run_dir))
            for key, value in brief.items():
                self.assertEqual(status[key], value)

    def test_blocked_from_loop_state_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "ledger.json").write_text(json.dumps({"records": []}))
            (run_dir / "loop_state.md").write_text(
                "phase: blocked\nactive_stop_condition: contract mismatch\n")
            cmd = _fake_cmd({"evaluations_done": 0}, {})
            status = compact_status("unit", "t5", run_dir, ROOT, cmd)
            self.assertEqual(status["phase"], "blocked")
            self.assertEqual(status["stop_condition"], "contract mismatch")

    def test_no_ledger_skips_brief(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cmd = _fake_cmd({"evaluations_done": 0}, {})
            status = compact_status("unit", "t5", run_dir, ROOT, cmd)
            self.assertEqual(status["phase"], "running")
            self.assertNotIn("next_run_id", status)


if __name__ == "__main__":
    unittest.main()
