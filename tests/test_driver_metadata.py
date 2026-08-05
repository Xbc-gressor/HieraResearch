import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver import metadata  # noqa: E402


class MetadataTests(unittest.TestCase):
    def test_write_and_roundtrip_bundled_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            meta = metadata.write_metadata(run_dir, "claude-sonnet-4-5", None)
            stored = json.loads((run_dir / "run_metadata.json").read_text())
            self.assertEqual(stored["model"], "claude-sonnet-4-5")
            self.assertEqual(stored["cli_source"], "bundled")
            self.assertIn("bundled-with-claude-agent-sdk-", stored["cli_version"])
            self.assertEqual(
                stored["permission_policy"],
                "bypassPermissions+pre-tool-use-capability-hook",
            )
            self.assertIsInstance(stored["prompt_hashes"], dict)
            self.assertEqual(meta["model"], stored["model"])

    def test_warn_on_mismatch_model_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metadata.write_metadata(run_dir, "model-a", None)
            warnings = metadata.warn_on_mismatch(run_dir, "model-b", None)
            self.assertTrue(any("model" in w for w in warnings))
            self.assertEqual(metadata.warn_on_mismatch(run_dir, "model-a", None), [])

    def test_warn_on_mismatch_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(metadata.warn_on_mismatch(Path(tmp), "m", None), [])


if __name__ == "__main__":
    unittest.main()
