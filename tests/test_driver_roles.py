import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from search_space_state import empty_search_space_state  # noqa: E402

from driver.roles import (  # noqa: E402
    ROLES,
    InvocationContext,
    ledger_brief,
    record_status,
)


def make_ctx(run_dir: Path, **kw) -> InvocationContext:
    return InvocationContext(task="t", tag="tag", run_dir=run_dir, invocation_id=1, **kw)


class ContextTests(unittest.TestCase):
    def test_user_message_carries_paths_and_ids_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp), run_id="003")
            msg = ctx.user_message()
            self.assertIn(str(Path(tmp)), msg)
            self.assertIn("003", msg)

    def test_checklist_lists_postcondition_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            role = ROLES["candidate-writer"]
            ctx = make_ctx(Path(tmp), run_id="003")
            text = ctx.postcondition_checklist(role)
            self.assertIn("train.py", text)


class RegistryTests(unittest.TestCase):
    EXPECTED = {
        "background-researcher", "idea-generator", "candidate-writer",
        "tunable-contract-extractor", "tuner-orchestrator",
        "experience-extractor", "crash-diagnosis", "hillclimb-editor",
    }

    def test_all_roles_registered(self) -> None:
        self.assertEqual(set(ROLES), self.EXPECTED)

    def test_no_role_has_agent_or_skill_capability(self) -> None:
        for role in ROLES.values():
            for banned in ("Agent", "Task", "Skill"):
                self.assertNotIn(banned, role.tools, role.name)
                self.assertIn(banned, role.disallowed, role.name)

    def test_read_only_roles_drop_write_tools(self) -> None:
        for name in ("crash-diagnosis",):
            role = ROLES[name]
            for write_tool in ("Edit", "Write"):
                self.assertNotIn(write_tool, role.tools, name)
                self.assertIn(write_tool, role.disallowed, name)


class PostconditionTests(unittest.TestCase):
    def _write_ledger(self, run_dir: Path, records: list[dict]) -> None:
        (run_dir / "ledger.json").write_text(json.dumps(
            {"task": "t", "tag": "tag", "records": records}))

    def test_candidate_writer_postcondition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            role = ROLES["candidate-writer"]
            ctx = make_ctx(run_dir, run_id="003")
            problems = [p for check in role.postconditions if (p := check(ctx))]
            self.assertTrue(any("train.py" in p for p in problems))
            target = run_dir / "candidates" / "003" / "train.py"
            target.parent.mkdir(parents=True)
            target.write_text("# candidate\n")
            problems = [p for check in role.postconditions if (p := check(ctx))]
            self.assertEqual(problems, [])


class LedgerAccessorTests(unittest.TestCase):
    """Pin the real `ledger.py brief` contract the accessors rely on.

    `brief` deliberately omits per-record data (no `records` key), so
    `record_status` and `actions_admitted` read `ledger.json` directly
    (reading is allowed; only hand-editing is banned). `brief` does carry
    `experience_refresh_required`, which `refresh_flag_cleared` consumes.
    """

    def _write_briefable_ledger(
        self, run_dir: Path, records: list[dict], **extra
    ) -> None:
        # A record-bearing ledger must carry search_space_state or
        # tools/ledger.py refuses to load it.
        (run_dir / "ledger.json").write_text(json.dumps(
            {"task": "t", "tag": "tag", "records": records,
             "search_space_state": empty_search_space_state(), **extra}))

    def test_brief_omits_per_record_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_briefable_ledger(run_dir, [
                {"run_id": "003", "status": "keep", "op": "improve"},
                {"run_id": "004", "status": "pending", "op": "improve"},
            ])
            brief = ledger_brief(run_dir)
            self.assertNotIn("records", brief)
            self.assertIn("experience_refresh_required", brief)
            self.assertEqual(brief["pending_run_ids"], ["004"])
            self.assertEqual(brief["status_counts"], {"keep": 1, "pending": 1})

    def test_record_status_reads_ledger_json_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_briefable_ledger(run_dir, [
                {"run_id": "003", "status": "keep"},
            ])
            self.assertEqual(record_status(run_dir, "003"), "keep")
            self.assertIsNone(record_status(run_dir, "999"))

    def test_record_is_terminal_postcondition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            role = ROLES["tunable-contract-extractor"]
            ctx = make_ctx(run_dir, run_id="003")
            self._write_briefable_ledger(run_dir, [
                {"run_id": "003", "status": "pending"},
            ])
            problems = [p for check in role.postconditions if (p := check(ctx))]
            self.assertTrue(any("003" in p for p in problems))
            self._write_briefable_ledger(run_dir, [
                {"run_id": "003", "status": "keep"},
            ])
            problems = [p for check in role.postconditions if (p := check(ctx))]
            self.assertEqual(problems, [])

    def test_refresh_flag_cleared_postcondition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            role = ROLES["experience-extractor"]
            ctx = make_ctx(run_dir)
            records = [{"run_id": "003", "status": "keep", "op": "improve"}]
            self._write_briefable_ledger(run_dir, records, dag_revision=1)
            problems = [p for check in role.postconditions if (p := check(ctx))]
            self.assertTrue(any("experience_refresh_required" in p for p in problems))
            self._write_briefable_ledger(
                run_dir, records, dag_revision=1,
                experience={"dag_revision": 1})
            problems = [p for check in role.postconditions if (p := check(ctx))]
            self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
