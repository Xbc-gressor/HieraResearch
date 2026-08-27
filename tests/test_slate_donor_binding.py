"""Manifest donor binding (design §3.1, §7.3): the build-manifest CLI gate,
digest coverage, and replay consistency between the manifest binding and the
seats' candidate-local receipts.

The fixtures reuse the real judged-slate pipeline from test_slate_replay
(lanes -> propose -> construct -> scripted judges -> aggregate -> manifest ->
admission) and the real finalized-donor candidate factory from
test_global_donor; only the judge rankings are scripted.
"""

from __future__ import annotations

import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import slate  # noqa: E402
from background_contract import ContractError  # noqa: E402
from tests.test_global_donor import (  # noqa: E402
    DONOR_PARAMS,
    _write_finalized_candidate,
)
from tests.test_slate_replay import (  # noqa: E402
    build_admitted_generation,
    run_replay,
)
from tests.fixtures import (  # noqa: E402
    fixture_registry,
    record as fixture_record,
)
from semantic_space import complete_point  # noqa: E402


TRANSFER_CFG = {
    "max_evaluations": 100,
    "tuner": {
        "scheduler_policy": "anchor_transfer_challenger_v1",
        "inner_policy": "hebo24-transfer10-hebo10",
    },
}

LEGACY_MANIFEST_KEYS = {
    "schema_version",
    "gen_no",
    "ledger_snapshot",
    "policy",
    "budget",
    "cardinality",
    "lanes_digest",
    "pool_digest",
    "context_digest",
    "judge_digest",
    "proposal_set_revisions",
    "aggregation",
    "reserved_run_ids",
    "slate",
    "judge_cost",
    "generation_id",
}


def _donor_record(run_dir: Path, run_id: str, *, score: float) -> dict:
    """A tuned, finalized prefix candidate eligible for the donor frontier.

    Full ledger-record shape (admission revalidates the whole ledger) plus
    the finalized Phase-C donor facts; the candidate files come from the
    real factory so the applied-incumbent cross-check reproduces.
    """
    donor = _write_finalized_candidate(
        run_dir,
        run_id,
        warm_params=DONOR_PARAMS,
        warm_score=score + 0.2,
        final_params=DONOR_PARAMS,
        final_score=score,
    )
    record = fixture_record(
        run_id,
        "fresh",
        [],
        complete_point(fixture_registry()),
        score=score,
        status="keep",
    )
    record.update(
        {
            "best_warm_score": score + 0.2,
            "tune": True,
            "tuning_bouts": 1,
            "evaluation_depth": "tuned_lightly",
            "applied_incumbent": donor["applied_incumbent"],
        }
    )
    return record


def _rebuild_manifest(run_dir: Path, **overrides) -> dict:
    """Re-run build-manifest over the existing artifacts to an alt output."""
    gen = run_dir / ".semantic" / "gen-0001"
    args = SimpleNamespace(
        lanes=gen / "lanes.json",
        pool=gen / "pool.json",
        context=gen / "context.json",
        judge=gen / "judge.json",
        reserved_run_ids="005,006",
        donor_snapshot=None,
        no_donor=False,
        output=gen / "alt-manifest.json",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    with redirect_stdout(io.StringIO()):
        slate.cmd_build_manifest(args)
    return json.loads(args.output.read_text())


def _generation_artifacts(run_dir: Path) -> tuple[dict, dict, dict]:
    gen = run_dir / ".semantic" / "gen-0001"
    return (
        json.loads((gen / "pool.json").read_text()),
        json.loads((gen / "context.json").read_text()),
        json.loads((gen / "judge.json").read_text()),
    )


class BuildManifestDonorBindingTests(unittest.TestCase):
    def test_old_policy_manifest_carries_no_donor_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_admitted_generation(Path(tmp))
            self.assertNotIn("donor_snapshot", manifest)
            # The byte contract of an old-policy manifest is exactly the
            # historical key set; a leaked field would show up here.
            self.assertEqual(set(manifest), LEGACY_MANIFEST_KEYS)

    def test_binding_is_digested_into_the_generation_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            build_admitted_generation(run_dir)
            pool, _, judge = _generation_artifacts(run_dir)
            budget = pool["budget"]
            reserved = ["005", "006"]
            base = slate.build_manifest(pool, judge, budget, reserved)
            no_donor = slate.build_manifest(
                pool, judge, budget, reserved,
                donor_snapshot=slate.no_donor_binding(),
            )
            bound = slate.build_manifest(
                pool, judge, budget, reserved,
                donor_snapshot={
                    "status": "bound",
                    "snapshot_id": "donor-" + "0" * 24,
                    "path": ".scheduler/donors/donor-" + "0" * 24 + ".json",
                    "digest": "sha256:" + "0" * 64,
                },
            )
            self.assertNotIn("donor_snapshot", base)
            self.assertEqual(
                no_donor["donor_snapshot"],
                {
                    "status": "no_donor",
                    "snapshot_id": None,
                    "path": None,
                    "digest": None,
                },
            )
            # The binding is part of the manifest core: every distinct binding
            # is a distinct generation.
            ids = {
                base["generation_id"],
                no_donor["generation_id"],
                bound["generation_id"],
            }
            self.assertEqual(len(ids), 3)
            for manifest in (base, no_donor, bound):
                self.assertEqual(
                    manifest["generation_id"],
                    slate.recompute_generation_id(manifest),
                )
            with self.assertRaises(ContractError):
                slate.build_manifest(
                    pool, judge, budget, reserved,
                    donor_snapshot={
                        "status": "bound",
                        "snapshot_id": None,
                        "path": None,
                        "digest": None,
                    },
                )

    def test_transfer_policy_requires_an_explicit_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            build_admitted_generation(run_dir)
            (run_dir / "framework_cfg.json").write_text(json.dumps(TRANSFER_CFG))
            with self.assertRaisesRegex(ContractError, "explicit donor binding"):
                _rebuild_manifest(run_dir)

    def test_cli_binds_no_donor_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            build_admitted_generation(run_dir)
            (run_dir / "framework_cfg.json").write_text(json.dumps(TRANSFER_CFG))
            manifest = _rebuild_manifest(run_dir, no_donor=True)
            self.assertEqual(
                manifest["donor_snapshot"],
                {
                    "status": "no_donor",
                    "snapshot_id": None,
                    "path": None,
                    "digest": None,
                },
            )
            pool, context, judge = _generation_artifacts(run_dir)
            self.assertEqual(slate.verify_manifest(manifest, pool, context, judge), [])

    def test_cli_binds_a_verified_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = build_admitted_generation(
                run_dir,
                framework_cfg=TRANSFER_CFG,
                extra_records=(_donor_record(run_dir, "005", score=0.5),),
                reserved_run_ids="006,007",
                donor="auto",
            )
            binding = manifest["donor_snapshot"]
            self.assertEqual(binding["status"], "bound")
            snapshot_path = run_dir / binding["path"]
            self.assertTrue(snapshot_path.is_file())
            snapshot = json.loads(snapshot_path.read_text())
            self.assertEqual(binding["snapshot_id"], snapshot["snapshot_id"])
            self.assertEqual(
                binding["path"],
                f".scheduler/donors/{snapshot['snapshot_id']}.json",
            )
            self.assertEqual(
                binding["digest"],
                "sha256:" + hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(snapshot["selected"]["run_id"], "005")
            pool, context, judge = _generation_artifacts(run_dir)
            self.assertEqual(slate.verify_manifest(manifest, pool, context, judge), [])

    def test_cli_rejects_a_snapshot_outside_the_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            build_admitted_generation(run_dir)
            (run_dir / "framework_cfg.json").write_text(json.dumps(TRANSFER_CFG))
            elsewhere = Path(tmp) / "elsewhere"
            elsewhere.mkdir()
            (elsewhere / "ledger.json").write_text(json.dumps({"records": []}))
            snapshot_path = elsewhere / "donor.json"
            snapshot_path.write_text(json.dumps({"snapshot_id": "donor-x"}))
            with self.assertRaisesRegex(ContractError, "not inside the run"):
                _rebuild_manifest(run_dir, donor_snapshot=snapshot_path)

    def test_old_policy_rejects_donor_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            build_admitted_generation(run_dir)  # no framework_cfg: old policy
            with self.assertRaisesRegex(ContractError, "scheduler_policy"):
                _rebuild_manifest(run_dir, no_donor=True)
            with self.assertRaisesRegex(ContractError, "scheduler_policy"):
                _rebuild_manifest(
                    run_dir,
                    donor_snapshot=run_dir / ".scheduler/donors/donor-x.json",
                )


class ReplayDonorBindingTests(unittest.TestCase):
    """§9.6: manifest binding <-> snapshot artifact <-> seat receipts."""

    def _bound_run(self, tmp: str) -> tuple[Path, dict]:
        run_dir = Path(tmp)
        manifest = build_admitted_generation(
            run_dir,
            framework_cfg=TRANSFER_CFG,
            extra_records=(_donor_record(run_dir, "005", score=0.5),),
            reserved_run_ids="006,007",
            donor="auto",
        )
        return run_dir, manifest

    @staticmethod
    def _write_seat_receipt(run_dir: Path, run_id: str, snapshot_id: str) -> None:
        candidate = run_dir / "candidates" / run_id
        candidate.mkdir(parents=True, exist_ok=True)
        (candidate / "_global_donor_transfer.json").write_text(
            json.dumps({"donor": {"snapshot_id": snapshot_id}})
        )

    def test_matching_seat_receipts_replay_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, manifest = self._bound_run(tmp)
            binding = manifest["donor_snapshot"]
            for slot in manifest["slate"]:
                self._write_seat_receipt(
                    run_dir, slot["run_id"], binding["snapshot_id"]
                )
            code, report = run_replay(run_dir)
            self.assertEqual(code, 0, json.dumps(report, indent=2))
            checks = report["generations"][0]["checks"]
            self.assertTrue(checks.get("donor_binding"))

    def test_missing_seat_receipts_are_notes_not_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, manifest = self._bound_run(tmp)
            self.assertEqual(manifest["donor_snapshot"]["status"], "bound")
            code, report = run_replay(run_dir)
            self.assertEqual(code, 0, json.dumps(report, indent=2))
            generation = report["generations"][0]
            self.assertTrue(generation["checks"].get("donor_binding"))
            self.assertTrue(
                any("no donor receipt" in note for note in generation["notes"])
            )

    def test_wrong_seat_snapshot_fails_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, manifest = self._bound_run(tmp)
            binding = manifest["donor_snapshot"]
            seats = manifest["slate"]
            self._write_seat_receipt(
                run_dir, seats[0]["run_id"], "donor-" + "1" * 24
            )
            self._write_seat_receipt(
                run_dir, seats[1]["run_id"], binding["snapshot_id"]
            )
            code, report = run_replay(run_dir)
            self.assertEqual(code, 1)
            errors = report["generations"][0]["errors"]
            self.assertTrue(
                any(
                    seats[0]["run_id"] in error and "donor receipt" in error
                    for error in errors
                ),
                errors,
            )
            self.assertFalse(report["generations"][0]["checks"]["donor_binding"])

    def test_snapshot_byte_drift_fails_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, manifest = self._bound_run(tmp)
            binding = manifest["donor_snapshot"]
            snapshot_path = run_dir / binding["path"]
            snapshot = json.loads(snapshot_path.read_text())
            # Same payload, different bytes: the content id still verifies but
            # the manifest bound the original byte digest.
            snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
            code, report = run_replay(run_dir)
            self.assertEqual(code, 1)
            errors = report["generations"][0]["errors"]
            self.assertTrue(any("digest" in error for error in errors), errors)

    def test_no_donor_binding_rejects_a_seat_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = build_admitted_generation(
                run_dir, framework_cfg=TRANSFER_CFG, donor="no_donor"
            )
            self.assertEqual(
                manifest["donor_snapshot"]["status"], "no_donor"
            )
            code, report = run_replay(run_dir)
            self.assertEqual(code, 0, json.dumps(report, indent=2))
            self.assertTrue(
                report["generations"][0]["checks"].get("donor_binding")
            )
            self._write_seat_receipt(
                run_dir, "005", "donor-" + "1" * 24
            )
            code, report = run_replay(run_dir)
            self.assertEqual(code, 1)
            errors = report["generations"][0]["errors"]
            self.assertTrue(
                any("no_donor binding" in error for error in errors), errors
            )


if __name__ == "__main__":
    unittest.main()
