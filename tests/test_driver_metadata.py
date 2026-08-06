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


class ResolveModelTests(unittest.TestCase):
    def test_fresh_dir_returns_cli_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, warning = metadata.resolve_model("model-a", Path(tmp))
            self.assertEqual((model, warning), ("model-a", None))

    def test_fresh_dir_without_model_yields_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(metadata.resolve_model(None, Path(tmp)),
                             (None, None))

    def test_matching_cli_model_is_silent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metadata.write_metadata(run_dir, "model-a", None)
            self.assertEqual(metadata.resolve_model("model-a", run_dir),
                             ("model-a", None))

    def test_stored_model_wins_over_mismatching_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metadata.write_metadata(run_dir, "model-a", None)
            model, warning = metadata.resolve_model("model-b", run_dir)
            self.assertEqual(model, "model-a")
            self.assertIn("--model model-b ignored", warning)
            self.assertIn("model-a", warning)


if __name__ == "__main__":
    unittest.main()
