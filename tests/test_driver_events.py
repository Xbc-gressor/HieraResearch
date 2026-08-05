import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.events import EventsLog  # noqa: E402


class EventsLogTests(unittest.TestCase):
    def test_emit_writes_jsonl_and_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            log = EventsLog(run_dir)
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                log.emit("session_end", role="candidate-writer", invocation_id=3, ok=True)
            line = (run_dir / "driver_events.jsonl").read_text().strip()
            row = json.loads(line)
            self.assertEqual(row["kind"], "session_end")
            self.assertEqual(row["role"], "candidate-writer")
            self.assertEqual(row["invocation_id"], 3)
            self.assertIn("ts", row)
            self.assertIn("session_end", buf.getvalue())

    def test_creates_run_dir_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "new" / "run"
            log = EventsLog(run_dir)
            log.emit("init")
            self.assertTrue((run_dir / "driver_events.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
