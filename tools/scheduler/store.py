"""Immutable decision artifacts and their provenance (design §8).

Three durable things per decision: the state snapshot, the decision
receipt, and — after execution — the realized outcome. They live under
``<run_dir>/.scheduler/`` and are append-only.

Why a separate store rather than reusing the ledger: the ledger is mutable
and holds current state. `state_snapshot_id` has to point at what the
scheduler saw *at decision time*, and a ledger revision cannot supply that
after the fact. Snapshots are content-addressed, which is exactly the
"content-addressed id" case the repo's hashing rule allows: the digest is
the identity another artifact (the receipt) refers to.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .evidence import EVIDENCE_FILENAME, EvidenceLog

STORE_DIRNAME = ".scheduler"
DECISIONS_FILENAME = "decisions.jsonl"
SNAPSHOTS_DIRNAME = "snapshots"
COVERAGE_FILENAME = "coverage.json"


class SchedulerStore:
    """Append-only artifact store for one run's scheduler decisions."""

    def __init__(self, run_dir: Path):
        self.root = Path(run_dir) / STORE_DIRNAME

    @property
    def evidence(self) -> EvidenceLog:
        return EvidenceLog(self.root / EVIDENCE_FILENAME)

    def _decisions_path(self) -> Path:
        return self.root / DECISIONS_FILENAME

    def put_snapshot(self, snapshot: dict) -> str:
        """Persist an immutable snapshot, returning its content id."""
        payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        snapshot_id = (
            "snap-" + hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
        )
        path = self.root / SNAPSHOTS_DIRNAME / f"{snapshot_id}.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(payload + "\n", encoding="utf-8")
            tmp.replace(path)
        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> dict:
        path = self.root / SNAPSHOTS_DIRNAME / f"{snapshot_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def next_decision_id(self) -> str:
        # Count decisions only: the log interleaves outcome rows, and those
        # reference an existing decision id rather than claiming a new one.
        decisions = [
            row for row in self.decisions() if row.get("kind") == "scheduler_decision"
        ]
        return f"dec-{len(decisions) + 1:04d}"

    def append_decision(self, receipt: dict) -> None:
        path = self._decisions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, sort_keys=True) + "\n")

    def decisions(self) -> list[dict]:
        path = self._decisions_path()
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def coverage_spent(self) -> int:
        """Forced-evidence collections charged to the shared cap.

        Counts distinct decisions, which is only a faithful charge because
        `decisions()` holds one row per decision *point* rather than one per
        query: re-asking for the same state reuses the open decision instead
        of appending another. Counting bound outcomes instead would undercount
        — the execution layer binds an outcome only after the bout finishes,
        so a coverage bout would be free to re-force itself while running.
        """
        return sum(
            1
            for row in self.decisions()
            if row.get("kind") == "scheduler_decision"
            and row.get("reason", "").startswith("coverage:")
        )

    def unbound_decisions(self) -> list[dict]:
        """Decisions with no outcome row yet, oldest first.

        An unbound decision is one the execution layer has not been shown to
        have carried out. It is the unit both idempotence and outcome
        binding work on.
        """
        bound = {
            row.get("decision_id")
            for row in self.decisions()
            if row.get("kind") == "scheduler_outcome"
        }
        return [
            row
            for row in self.decisions()
            if row.get("kind") == "scheduler_decision"
            and row.get("decision_id") not in bound
            # STOP spends nothing and executes nothing, so it can never be
            # bound; treating it as open would block every later decision.
            and row.get("selected_action") != "STOP"
        ]

    def open_decision(self, snapshot_id: str, evidence_cursor: int) -> dict | None:
        """An already-issued decision for this exact state, if unexecuted.

        A decision is a commitment to spend budget, not a query result. The
        orchestrator may ask more than once for the same round — a retry, a
        corrective follow-up, a crash between the call and the bout — and
        each of those must get back the decision that was already made,
        not a fresh one. Identity is `(state snapshot, evidence cursor)`:
        the two inputs the policy is a pure function of. Once an outcome is
        bound, the decision is closed and an identical state (a bout that
        consumed nothing and changed nothing) is genuinely a new decision.
        """
        for row in reversed(self.unbound_decisions()):
            if (
                row.get("state_snapshot_id") == snapshot_id
                and row.get("evidence_cursor") == evidence_cursor
            ):
                return row
        return None

    def record_outcome(
        self,
        decision_id: str,
        *,
        executed_action: str,
        executed_run_id: str | None,
        realized_gain: float | None,
        consumed: int | None,
        status: str,
    ) -> None:
        """Close the loop: what the driver actually did and what it cost.

        Kept separate from the decision receipt because a decision can be
        superseded by the execution layer (a failed bout, a recovery), and
        the receipt must stay the immutable record of what was *chosen*.
        """
        self.append_decision(
            {
                "schema_version": 1,
                "kind": "scheduler_outcome",
                "decision_id": decision_id,
                "executed_action": executed_action,
                "executed_run_id": executed_run_id,
                "realized_gain": realized_gain,
                "consumed_evaluations": consumed,
                "status": status,
            }
        )
