import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.receipts import (ReceiptError, ReceiptStore,  # noqa: E402
                             handle_submit_receipt, validate_receipt)

SCHEMA = {
    "status": ("enum", "keep", "discard", "crash"),
    "run_id": "str",
    "score": "?float",
    "ledger_updated": "bool",
}


class ValidateReceiptTests(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(
            validate_receipt(SCHEMA, {"status": "keep", "run_id": "003",
                                      "ledger_updated": True}),
            [],
        )

    def test_missing_required(self) -> None:
        problems = validate_receipt(SCHEMA, {"status": "keep"})
        self.assertIn("missing field: run_id", problems)
        self.assertIn("missing field: ledger_updated", problems)

    def test_optional_may_be_absent_but_typed_when_present(self) -> None:
        self.assertEqual(
            validate_receipt(SCHEMA, {"status": "keep", "run_id": "003",
                                      "ledger_updated": True}),
            [],
        )
        problems = validate_receipt(
            SCHEMA, {"status": "keep", "run_id": "003", "ledger_updated": True,
                     "score": "high"})
        self.assertEqual(len(problems), 1)
        self.assertIn("score", problems[0])

    def test_enum_rejects_unknown_value(self) -> None:
        problems = validate_receipt(
            SCHEMA, {"status": "maybe", "run_id": "003", "ledger_updated": True})
        self.assertEqual(len(problems), 1)
        self.assertIn("status", problems[0])

    def test_bool_is_not_int(self) -> None:
        problems = validate_receipt(
            {"n": "int"}, {"n": True})
        self.assertEqual(len(problems), 1)

    def test_non_object(self) -> None:
        self.assertEqual(validate_receipt(SCHEMA, [1, 2]),
                         ["receipt is not a JSON object"])


class ReceiptStoreTests(unittest.TestCase):
    def test_invocation_ids_are_monotonic_across_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp))
            first = store.next_invocation_id()
            store.persist_receipt("idea-generator", first, {"actions": []})
            second = store.next_invocation_id()
            store.persist_session_id("candidate-writer", second, "sess-1")
            third = store.next_invocation_id()
            self.assertEqual((first, second, third), (1, 2, 3))

    def test_persist_receipt_is_atomic_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp))
            path = store.persist_receipt("tuner-orchestrator", 7, {"tuned": False})
            self.assertEqual(path.name, "tuner-orchestrator-0007.json")
            with self.assertRaises(ReceiptError):
                store.persist_receipt("tuner-orchestrator", 7, {"tuned": True})
            stored = json.loads(path.read_text())
            self.assertEqual(stored, {"tuned": False})
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_persist_receipt_allow_replace_overwrites_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp))
            path = store.persist_receipt("tuner-orchestrator", 7, {"tuned": False})
            replaced = store.persist_receipt(
                "tuner-orchestrator", 7, {"tuned": True}, allow_replace=True)
            self.assertEqual(replaced, path)
            stored = json.loads(path.read_text())
            self.assertEqual(stored, {"tuned": True})
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_session_id_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp))
            self.assertIsNone(store.load_session_id("idea-generator", 2))
            store.persist_session_id("idea-generator", 2, "sess-abc")
            self.assertEqual(store.load_session_id("idea-generator", 2), "sess-abc")


class SubmitReceiptContractTests(unittest.TestCase):
    def test_rejection_sets_sdk_is_error_key(self) -> None:
        """Rejected receipts must surface is_error (snake_case) so the SDK
        reports a tool error instead of a successful result."""
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp))
            accepted: list[dict] = []
            result = asyncio.run(handle_submit_receipt(
                {"edited": "bool"}, store, "hillclimb-editor", 1, accepted,
                {"receipt": {"edited": "yes"}},
            ))
            self.assertTrue(result.get("is_error"))
            self.assertEqual(accepted, [])
            self.assertFalse(list(store._dir().glob("*.json")))
    def test_resubmission_replaces_receipt_latest_wins(self) -> None:
        """A corrective follow-up re-submission within one invocation must
        succeed: accepted keeps both payloads and the disk holds the last."""
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp))
            accepted: list[dict] = []
            first = {"status": "keep", "run_id": "003", "ledger_updated": False}
            second = {"status": "keep", "run_id": "003", "ledger_updated": True}
            result1 = asyncio.run(handle_submit_receipt(
                SCHEMA, store, "idea-generator", 3, accepted,
                {"receipt": first},
            ))
            result2 = asyncio.run(handle_submit_receipt(
                SCHEMA, store, "idea-generator", 3, accepted,
                {"receipt": second},
            ))
            self.assertFalse(result1.get("is_error"))
            self.assertFalse(result2.get("is_error"))
            self.assertEqual(accepted, [first, second])
            path = store.receipt_path("idea-generator", 3)
            self.assertEqual(json.loads(path.read_text()), second)
            self.assertFalse(list(path.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
