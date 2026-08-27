"""Tests for the global-donor snapshot and the inject-global-donor helper.

Covers only the new contracts (design §3.1, §3.2, §4.1, §8):
- donor frontier selection (tuned + finalized only, min(score, run_id)),
  content-addressed snapshot stability, and the no_donor path;
- candidate-side injection: fresh/non-fresh projection, append vs dedup,
  donor_incompatible, frozen snapshot binding, and interruption recovery.

Fixtures use real candidate files, real ledgers, and real tune reports; the
new policy pair is probed from a raw framework_cfg.json because it is not
registered anywhere yet.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import tune_tools  # noqa: E402
from tools.scheduler.donor import (  # noqa: E402
    build_donor_snapshot,
    donors_dir,
    load_donor_snapshot,
)
from tune_tools import (  # noqa: E402
    GLOBAL_DONOR_TRANSFER_FILENAME,
    PARAMETER_TRANSFER_FILENAME,
    _candidate_execution_revision,
    inject_global_donor,
    materialize_parameter_transfer,
)


NEW_POLICY_CFG = {
    "tuner": {
        "scheduler_policy": "anchor_transfer_challenger_v1",
        "inner_policy": "hebo24-transfer10-hebo10",
    }
}

DONOR_SCHEMA = {
    "same": "int",
    "category": ("categorical", ["a", "b"]),
    "changed_kind": "float",
    "dropped": "int",
}
DONOR_SPACE = {
    "same": ("int", 1, 10),
    "category": ("categorical", ["a", "b"]),
    "changed_kind": ("float", 0.1, 1.0),
    "dropped": ("int", 1, 10),
}
DONOR_PARAMS = {"same": 7, "category": "b", "changed_kind": 0.5, "dropped": 9}

RECIPIENT_SCHEMA = {
    "same": "int",
    "category": ("categorical", ["a", "c"]),
    "changed_kind": "int",
    "new_key": "float",
}
RECIPIENT_SPACE = {
    "same": ("int", 1, 10),
    "category": ("categorical", ["a", "c"]),
    "changed_kind": ("int", 1, 5),
    "new_key": ("float", 0.1, 0.5),
}
RECIPIENT_DEFAULTS = {"same": 2, "category": "a", "changed_kind": 3, "new_key": 0.2}
# The donor projection onto RECIPIENT_SCHEMA: same copied (7), category reset
# (b removed from the options), changed_kind reset (float -> int), new_key new.
DONOR_PROJECTION = {"same": 7, "category": "a", "changed_kind": 3, "new_key": 0.2}

ORDINARY_CONFIGS = [
    RECIPIENT_DEFAULTS,
    {"same": 4, "category": "c", "changed_kind": 5, "new_key": 0.4},
    {"same": 1, "category": "a", "changed_kind": 1, "new_key": 0.1},
    {"same": 9, "category": "c", "changed_kind": 2, "new_key": 0.3},
    {"same": 5, "category": "a", "changed_kind": 4, "new_key": 0.5},
]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _train_source(schema: dict, space: dict, base: dict | None) -> str:
    lines = [f"PARAM_SCHEMA = {schema!r}", f"SEARCH_SPACE = {space!r}"]
    if base is not None:
        lines.append(f"BASE_PARAMS = {base!r}")
    lines += ["", "def make_model(params):", "    return params", ""]
    return "\n".join(lines)


def _write_finalized_candidate(
    run_dir: Path,
    run_id: str,
    *,
    schema: dict = DONOR_SCHEMA,
    space: dict = DONOR_SPACE,
    warm_params: dict,
    warm_score: float,
    final_params: dict,
    final_score: float,
) -> dict:
    """A candidate whose one Phase-C bout is closed and applied to BASE_PARAMS.

    Returns the ledger record; the caller assembles and writes ledger.json
    once every candidate's bytes are final.
    """
    candidate_dir = run_dir / "candidates" / run_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    train = candidate_dir / "train.py"
    train.write_text(_train_source(schema, space, final_params))
    (candidate_dir / "prepare.py").write_text(
        "def evaluate_config(make_model, params):\n    return 0.0\n"
    )
    report = {
        "phase_a": {
            "status": "ok",
            "best_warm_params": warm_params,
            "best_warm_score": warm_score,
            "warm_start_configs": [{"params": warm_params, "score": warm_score}],
            "candidate_code_revision": _candidate_execution_revision(train),
        },
        "phase_c": {
            "stages": [
                {
                    "method": "bo",
                    "status": "ok",
                    "trials": [{"params": final_params, "score": final_score}],
                }
            ]
        },
        "final_best_params": final_params,
        "final_best_score": final_score,
        "applied_to_base_params": True,
    }
    report_path = candidate_dir / "tune_report.json"
    report_path.write_text(json.dumps(report))
    return {
        "run_id": run_id,
        "status": "keep",
        "tune": True,
        "tuning_bouts": 1,
        "evaluation_depth": "tuned_lightly",
        "final_best_score": final_score,
        "applied_incumbent": {
            "schema_version": 1,
            "source": "finalized_phase_c",
            "score": final_score,
            "params": final_params,
            "param_schema": json.loads(json.dumps(schema)),
            "entrypoint_sha256": _sha256(train),
            "tune_report_sha256": _sha256(report_path),
        },
    }


def _write_run(root: Path, records: list, cfg: dict | None = None) -> Path:
    run_dir = root / "runs" / "unit" / "tag"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "framework_cfg.json").write_text(
        json.dumps(NEW_POLICY_CFG if cfg is None else cfg)
    )
    (run_dir / "ledger.json").write_text(json.dumps({"records": records}))
    return run_dir


def _write_recipient(
    run_dir: Path,
    run_id: str = "010",
    *,
    configs: list | None = None,
    space: dict = RECIPIENT_SPACE,
    brief_extra: dict | None = None,
) -> tuple[Path, Path]:
    candidate_dir = run_dir / "candidates" / run_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    train = candidate_dir / "train.py"
    train.write_text(_train_source(RECIPIENT_SCHEMA, space, None))
    (candidate_dir / "prepare.py").write_text(
        "def evaluate_config(make_model, params):\n    return 0.0\n"
    )
    brief = {
        "schema_version": 4,
        "run_id": run_id,
        "op": "fresh",
        "source_run_ids": [],
        "primary_parent": None,
    }
    if brief_extra:
        brief.update(brief_extra)
    (candidate_dir / "_candidate_brief.json").write_text(json.dumps(brief))
    configs_path = candidate_dir / "_warm_configs.json"
    configs_path.write_text(
        json.dumps(list(ORDINARY_CONFIGS if configs is None else configs))
    )
    return train, configs_path


def _bound_snapshot(run_dir: Path) -> Path:
    result = build_donor_snapshot(run_dir)
    assert result["status"] == "ok", result
    return run_dir / result["path"]


class DonorSnapshotTests(unittest.TestCase):
    def test_selects_lowest_score_then_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "unit" / "tag"
            records = [
                _write_finalized_candidate(
                    run_dir,
                    "001",
                    warm_params=DONOR_PARAMS,
                    warm_score=0.8,
                    final_params=DONOR_PARAMS,
                    final_score=0.3,
                ),
                _write_finalized_candidate(
                    run_dir,
                    "002",
                    warm_params=DONOR_PARAMS,
                    warm_score=0.8,
                    final_params=DONOR_PARAMS,
                    final_score=0.5,
                ),
                _write_finalized_candidate(
                    run_dir,
                    "003",
                    warm_params=DONOR_PARAMS,
                    warm_score=0.8,
                    final_params=DONOR_PARAMS,
                    final_score=0.3,
                ),
            ]
            _write_run(root, records)

            result = build_donor_snapshot(run_dir)

            self.assertEqual(result["status"], "ok")
            # 001 and 003 tie on score; the lower run id wins.
            self.assertEqual(result["snapshot"]["selected"]["run_id"], "001")
            self.assertEqual(
                [entry["run_id"] for entry in result["snapshot"]["eligible"]],
                ["001", "003", "002"],
            )
            self.assertEqual(
                result["snapshot"]["selection_rule"],
                "min(final_best_score, run_id)",
            )
            selected = result["snapshot"]["selected"]
            self.assertEqual(selected["params"], DONOR_PARAMS)
            self.assertEqual(
                selected["param_schema"], json.loads(json.dumps(DONOR_SCHEMA))
            )
            self.assertEqual(
                selected["entrypoint_revision"]["path"],
                "candidates/001/train.py",
            )
            self.assertEqual(result["excluded"], [])
            load_donor_snapshot(run_dir / result["path"])

    def test_excludes_warm_only_and_crash_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "unit" / "tag"
            donor = _write_finalized_candidate(
                run_dir,
                "005",
                warm_params=DONOR_PARAMS,
                warm_score=0.8,
                final_params=DONOR_PARAMS,
                final_score=0.5,
            )
            # Warm-only winner: fully consistent files, the best score in the
            # run, but no finalized Phase-C bout.
            warm_only = _write_finalized_candidate(
                run_dir,
                "006",
                warm_params=DONOR_PARAMS,
                warm_score=0.8,
                final_params=DONOR_PARAMS,
                final_score=0.01,
            )
            warm_only["tuning_bouts"] = 0
            warm_only["evaluation_depth"] = "screening"
            crash = {
                "run_id": "007",
                "status": "crash",
                "tune": True,
                "tuning_bouts": 1,
                "evaluation_depth": None,
                "final_best_score": float("inf"),
                "applied_incumbent": None,
            }
            _write_run(root, [donor, warm_only, crash])

            result = build_donor_snapshot(run_dir)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["snapshot"]["selected"]["run_id"], "005")
            self.assertEqual(
                [entry["run_id"] for entry in result["snapshot"]["eligible"]],
                ["005"],
            )
            excluded = {entry["run_id"]: entry["reason"] for entry in result["excluded"]}
            self.assertIn("006", excluded)
            self.assertIn("tuning_bouts", excluded["006"])
            self.assertIn("007", excluded)

    def test_content_id_is_stable_for_the_same_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "unit" / "tag"
            records = [
                _write_finalized_candidate(
                    run_dir,
                    "005",
                    warm_params=DONOR_PARAMS,
                    warm_score=0.8,
                    final_params=DONOR_PARAMS,
                    final_score=0.5,
                )
            ]
            _write_run(root, records)

            first = build_donor_snapshot(run_dir)
            path = donors_dir(run_dir) / f"{first['snapshot_id']}.json"
            first_bytes = path.read_bytes()
            second = build_donor_snapshot(run_dir)

            self.assertEqual(first["snapshot_id"], second["snapshot_id"])
            self.assertEqual(first["path"], second["path"])
            self.assertEqual(first_bytes, path.read_bytes())
            self.assertEqual(len(list(donors_dir(run_dir).iterdir())), 1)

    def test_no_eligible_donor_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm_only = {
                "run_id": "001",
                "status": "keep",
                "tune": True,
                "tuning_bouts": 0,
                "evaluation_depth": "screening",
                "final_best_score": 0.4,
                "applied_incumbent": None,
            }
            run_dir = _write_run(root, [warm_only])

            result = build_donor_snapshot(run_dir)

            self.assertEqual(result["status"], "no_donor")
            self.assertIsNone(result["snapshot_id"])
            self.assertIsNone(result["path"])
            self.assertFalse(donors_dir(run_dir).exists())


class InjectGlobalDonorTests(unittest.TestCase):
    def _donor_run(
        self,
        root: Path,
        *,
        cfg: dict | None = None,
        donor_schema: dict = DONOR_SCHEMA,
        donor_space: dict = DONOR_SPACE,
        donor_params: dict = DONOR_PARAMS,
        donor_score: float = 0.3,
    ) -> tuple[Path, Path]:
        run_dir = root / "runs" / "unit" / "tag"
        record = _write_finalized_candidate(
            run_dir,
            "005",
            schema=donor_schema,
            space=donor_space,
            warm_params=donor_params,
            warm_score=0.8,
            final_params=donor_params,
            final_score=donor_score,
        )
        _write_run(root, [record], cfg=cfg)
        return run_dir, _bound_snapshot(run_dir)

    def test_fresh_candidate_appends_projected_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, snapshot_path = self._donor_run(Path(tmp))
            train, configs_path = _write_recipient(run_dir)

            result = inject_global_donor(train, configs_path, snapshot_path)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["ordinary_config_count"], 5)
            self.assertEqual(result["config_count"], 6)
            self.assertEqual(result["warm_config_index"], 5)
            self.assertFalse(result["deduplicated"])
            self.assertEqual(result["donor_run_id"], "005")
            configs = json.loads(configs_path.read_text())
            self.assertEqual(len(configs), 6)
            self.assertEqual(configs[:5], ORDINARY_CONFIGS)
            self.assertEqual(configs[5], DONOR_PROJECTION)

            receipt = json.loads(
                (train.parent / GLOBAL_DONOR_TRANSFER_FILENAME).read_text()
            )
            self.assertEqual(receipt["schema_version"], 1)
            self.assertEqual(receipt["kind"], "global_donor_transfer")
            self.assertEqual(receipt["status"], "ok")
            self.assertEqual(receipt["candidate"]["run_id"], "010")
            self.assertTrue(receipt["candidate"]["fresh"])
            self.assertEqual(receipt["ordinary_config_count"], 5)
            self.assertEqual(receipt["donor"]["snapshot_id"], result["donor_snapshot"])
            self.assertEqual(receipt["donor"]["selected"]["run_id"], "005")
            self.assertEqual(receipt["warm_config_index"], 5)
            self.assertFalse(receipt["deduplicated"])
            self.assertIsNone(receipt["dedup_ordinary_index"])
            self.assertEqual(receipt["violations"], [])
            # Reserved for the warm-selection step (T2), not computed here.
            self.assertIsNone(receipt["mandatory_role_indices"])
            self.assertIsNone(receipt["k_eval"])
            projection = receipt["projection"]
            self.assertEqual(projection["params"], DONOR_PROJECTION)
            self.assertEqual(
                [item["key"] for item in projection["copied"]], ["same"]
            )
            self.assertEqual(
                {item["key"]: item["reason"] for item in projection["reset"]},
                {
                    "category": "categorical_value_removed",
                    "changed_kind": "kind_changed",
                },
            )
            self.assertEqual(
                [item["key"] for item in projection["new"]], ["new_key"]
            )
            self.assertEqual(
                [item["key"] for item in projection["dropped"]], ["dropped"]
            )

    def _write_nonfresh_recipient(self, run_dir: Path) -> tuple[Path, Path]:
        parent_train = run_dir / "candidates" / "020" / "train.py"
        train, configs_path = _write_recipient(
            run_dir,
            "010",
            brief_extra={
                "op": "improve",
                "source_run_ids": ["020"],
                "primary_parent": {
                    "schema_version": 1,
                    "run_id": "020",
                    "path": str(parent_train),
                    "sha256": _sha256(parent_train),
                },
                "implementation_source": {
                    "kind": "primary_parent_snapshot",
                    "parent_run_id": "020",
                    "path": str(parent_train),
                    "sha256": _sha256(parent_train),
                },
            },
        )
        return train, configs_path

    def test_nonfresh_defaults_come_from_the_transfer_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # The donor's "same" is float-typed, so it resets to the child
            # default (2) — provably not to the lineage control (3) now
            # sitting in warm config 0.
            float_same_schema = {**DONOR_SCHEMA, "same": "float"}
            float_same_space = {**DONOR_SPACE, "same": ("float", 1.0, 10.0)}
            run_dir = root / "runs" / "unit" / "tag"
            donor_record = _write_finalized_candidate(
                run_dir,
                "005",
                schema=float_same_schema,
                space=float_same_space,
                warm_params={**DONOR_PARAMS, "same": 7.0},
                warm_score=0.8,
                final_params={**DONOR_PARAMS, "same": 7.0},
                final_score=0.3,
            )
            parent_record = _write_finalized_candidate(
                run_dir,
                "020",
                warm_params=DONOR_PARAMS,
                warm_score=0.95,
                final_params={**DONOR_PARAMS, "same": 3},
                final_score=0.9,
            )
            _write_run(root, [donor_record, parent_record])
            snapshot_path = _bound_snapshot(run_dir)
            train, configs_path = self._write_nonfresh_recipient(run_dir)
            materialize_parameter_transfer(train, configs_path)
            configs = json.loads(configs_path.read_text())
            # The lineage control now occupies config 0.
            self.assertEqual(
                configs[0],
                {"same": 3, "category": "a", "changed_kind": 3, "new_key": 0.2},
            )

            result = inject_global_donor(train, configs_path, snapshot_path)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["warm_config_index"], 5)
            receipt = json.loads(
                (train.parent / GLOBAL_DONOR_TRANSFER_FILENAME).read_text()
            )
            self.assertFalse(receipt["candidate"]["fresh"])
            projected = receipt["projection"]["params"]
            self.assertEqual(
                projected,
                {"same": 2, "category": "a", "changed_kind": 3, "new_key": 0.2},
            )
            self.assertEqual(
                {item["key"]: item["reason"] for item in receipt["projection"]["reset"]},
                {
                    "same": "kind_changed",
                    "category": "categorical_value_removed",
                    "changed_kind": "kind_changed",
                },
            )
            configs = json.loads(configs_path.read_text())
            self.assertEqual(len(configs), 6)
            self.assertEqual(configs[5], projected)
            # The lineage control at index 0 is untouched.
            self.assertEqual(configs[0]["same"], 3)

    def test_dedup_against_the_lineage_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "unit" / "tag"
            donor_record = _write_finalized_candidate(
                run_dir,
                "005",
                warm_params=DONOR_PARAMS,
                warm_score=0.8,
                final_params=DONOR_PARAMS,
                final_score=0.3,
            )
            # The primary parent projects to exactly the donor row, so the
            # lineage control at index 0 and the donor coincide.
            parent_record = _write_finalized_candidate(
                run_dir,
                "020",
                warm_params=DONOR_PARAMS,
                warm_score=0.95,
                final_params=DONOR_PARAMS,
                final_score=0.9,
            )
            _write_run(root, [donor_record, parent_record])
            snapshot_path = _bound_snapshot(run_dir)
            train, configs_path = self._write_nonfresh_recipient(run_dir)
            materialize_parameter_transfer(train, configs_path)
            configs_before = configs_path.read_bytes()

            result = inject_global_donor(train, configs_path, snapshot_path)

            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["deduplicated"])
            self.assertEqual(result["warm_config_index"], 0)
            self.assertEqual(result["config_count"], 5)
            self.assertEqual(configs_path.read_bytes(), configs_before)
            receipt = json.loads(
                (train.parent / GLOBAL_DONOR_TRANSFER_FILENAME).read_text()
            )
            self.assertTrue(receipt["deduplicated"])
            self.assertEqual(receipt["warm_config_index"], 0)
            self.assertEqual(receipt["dedup_ordinary_index"], 0)

    def test_dedup_against_an_ordinary_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, snapshot_path = self._donor_run(Path(tmp))
            configs = list(ORDINARY_CONFIGS)
            configs[2] = dict(DONOR_PROJECTION)
            train, configs_path = _write_recipient(run_dir, configs=configs)
            configs_before = configs_path.read_bytes()

            result = inject_global_donor(train, configs_path, snapshot_path)

            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["deduplicated"])
            self.assertEqual(result["warm_config_index"], 2)
            self.assertEqual(result["config_count"], 5)
            self.assertEqual(configs_path.read_bytes(), configs_before)
            receipt = json.loads(
                (train.parent / GLOBAL_DONOR_TRANSFER_FILENAME).read_text()
            )
            self.assertEqual(receipt["dedup_ordinary_index"], 2)

    def test_donor_incompatible_records_violations_without_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, snapshot_path = self._donor_run(Path(tmp))
            narrow_space = {**RECIPIENT_SPACE, "same": ("int", 1, 5)}
            configs = [
                {**config, "same": min(config["same"], 5)}
                for config in ORDINARY_CONFIGS
            ]
            train, configs_path = _write_recipient(
                run_dir, configs=configs, space=narrow_space
            )
            configs_before = configs_path.read_bytes()

            result = inject_global_donor(train, configs_path, snapshot_path)

            self.assertEqual(result["status"], "donor_incompatible")
            self.assertEqual(
                [violation["key"] for violation in result["violations"]],
                ["same"],
            )
            self.assertEqual(result["config_count"], 5)
            self.assertIsNone(result["warm_config_index"])
            # No append, no clamp, no SEARCH_SPACE edit.
            self.assertEqual(configs_path.read_bytes(), configs_before)
            self.assertIn(
                '"same": ("int", 1, 5)',
                train.read_text().replace("'", '"'),
            )
            receipt = json.loads(
                (train.parent / GLOBAL_DONOR_TRANSFER_FILENAME).read_text()
            )
            self.assertEqual(receipt["status"], "donor_incompatible")
            self.assertIsNone(receipt["warm_config_index"])
            self.assertEqual(
                [violation["key"] for violation in receipt["violations"]],
                ["same"],
            )
            self.assertEqual(receipt["ordinary_config_count"], 5)

    def test_receipt_binds_the_snapshot_across_refreshes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, snapshot_path = self._donor_run(root)
            train, configs_path = _write_recipient(run_dir)
            receipt_path = train.parent / GLOBAL_DONOR_TRANSFER_FILENAME

            first = inject_global_donor(train, configs_path, snapshot_path)
            receipt_bytes = receipt_path.read_bytes()
            configs_bytes = configs_path.read_bytes()

            # An idempotent re-run rebuilds the identical projection.
            again = inject_global_donor(train, configs_path, snapshot_path)
            self.assertEqual(again["status"], "ok")
            self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
            self.assertEqual(configs_path.read_bytes(), configs_bytes)

            # A code fix changes the execution revision; the same bound
            # snapshot rebuilds the projection in place.
            train.write_text(train.read_text() + "\n# comment tweak\n")
            refreshed = inject_global_donor(train, configs_path, snapshot_path)
            self.assertEqual(refreshed["status"], "ok")
            self.assertEqual(refreshed["warm_config_index"], 5)
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["projection"]["params"], DONOR_PROJECTION)
            self.assertNotEqual(receipt_path.read_bytes(), receipt_bytes)
            self.assertEqual(configs_path.read_bytes(), configs_bytes)

            # A PARAM_SCHEMA repair rebuilds from the same snapshot: the new
            # key is filled from the recipient defaults.
            train.write_text(
                train.read_text().replace(
                    "'new_key': 'float'",
                    "'new_key': 'float', 'another': 'int'",
                )
            )
            train.write_text(
                train.read_text().replace(
                    "'new_key': ('float', 0.1, 0.5)",
                    "'new_key': ('float', 0.1, 0.5), 'another': ('int', 1, 3)",
                )
            )
            configs = json.loads(configs_path.read_text())
            for config in configs:
                config["another"] = 1
            configs_path.write_text(json.dumps(configs))
            repaired = inject_global_donor(train, configs_path, snapshot_path)
            self.assertEqual(repaired["status"], "ok")
            self.assertEqual(repaired["warm_config_index"], 5)
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["projection"]["params"]["another"], 1)
            configs = json.loads(configs_path.read_text())
            self.assertEqual(len(configs), 6)
            self.assertEqual(configs[5], receipt["projection"]["params"])

            # The binding is frozen: a newer donor frontier must not be
            # switched in.
            better = _write_finalized_candidate(
                run_dir,
                "006",
                warm_params=DONOR_PARAMS,
                warm_score=0.8,
                final_params=DONOR_PARAMS,
                final_score=0.1,
            )
            ledger_path = run_dir / "ledger.json"
            ledger = json.loads(ledger_path.read_text())
            ledger["records"].append(better)
            ledger_path.write_text(json.dumps(ledger))
            newer_snapshot = _bound_snapshot(run_dir)
            self.assertNotEqual(newer_snapshot, snapshot_path)
            with self.assertRaisesRegex(ValueError, "frozen"):
                inject_global_donor(train, configs_path, newer_snapshot)

    def test_warm_selection_recorded_freezes_population_index_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, snapshot_path = self._donor_run(root)
            train, configs_path = _write_recipient(run_dir)
            inject_global_donor(train, configs_path, snapshot_path)
            receipt_path = train.parent / GLOBAL_DONOR_TRANSFER_FILENAME
            receipt_bytes = receipt_path.read_bytes()
            configs_bytes = configs_path.read_bytes()
            (train.parent / "tune_report.json").write_text(
                json.dumps(
                    {
                        "phase_a": {
                            "status": "ok",
                            "warm_config_selection": {
                                "schema_version": 2,
                                "mandatory_indices": [0, 5],
                            },
                        }
                    }
                )
            )

            unchanged = inject_global_donor(train, configs_path, snapshot_path)
            self.assertEqual(unchanged["status"], "ok")
            self.assertTrue(unchanged["unchanged"])
            self.assertEqual(unchanged["warm_config_index"], 5)
            self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
            self.assertEqual(configs_path.read_bytes(), configs_bytes)

            # A mutated donor row violates the frozen population.
            configs = json.loads(configs_path.read_text())
            configs[5]["same"] = 6
            configs_path.write_text(json.dumps(configs))
            with self.assertRaisesRegex(ValueError, "no longer match"):
                inject_global_donor(train, configs_path, snapshot_path)
            configs_path.write_bytes(configs_bytes)

            # A dropped donor row changes the population count.
            configs_path.write_text(
                json.dumps(json.loads(configs_bytes)[:5])
            )
            with self.assertRaisesRegex(ValueError, "refusing any donor"):
                inject_global_donor(train, configs_path, snapshot_path)
            configs_path.write_bytes(configs_bytes)

            # A different snapshot is refused even though everything else is
            # consistent.
            other = _write_finalized_candidate(
                run_dir,
                "006",
                warm_params=DONOR_PARAMS,
                warm_score=0.8,
                final_params={**DONOR_PARAMS, "same": 8},
                final_score=0.1,
            )
            ledger_path = run_dir / "ledger.json"
            ledger = json.loads(ledger_path.read_text())
            ledger["records"].append(other)
            ledger_path.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, "frozen"):
                inject_global_donor(train, configs_path, _bound_snapshot(run_dir))

    def test_interrupted_configs_replace_recovers_from_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, snapshot_path = self._donor_run(Path(tmp))
            train, configs_path = _write_recipient(run_dir)
            receipt_path = train.parent / GLOBAL_DONOR_TRANSFER_FILENAME
            configs_before = configs_path.read_bytes()
            original_replace = Path.replace

            def fail_configs_replace(path: Path, target: Path):
                if Path(target).resolve() == configs_path.resolve():
                    raise OSError("simulated configs replace interruption")
                return original_replace(path, target)

            with mock.patch.object(Path, "replace", new=fail_configs_replace):
                with self.assertRaisesRegex(
                    OSError, "simulated configs replace interruption"
                ):
                    inject_global_donor(train, configs_path, snapshot_path)

            self.assertTrue(receipt_path.is_file())
            self.assertEqual(configs_path.read_bytes(), configs_before)

            recovered = inject_global_donor(train, configs_path, snapshot_path)
            self.assertEqual(recovered["status"], "ok")
            self.assertEqual(recovered["warm_config_index"], 5)
            configs = json.loads(configs_path.read_text())
            self.assertEqual(len(configs), 6)
            self.assertEqual(configs[5], DONOR_PROJECTION)
            self.assertEqual(
                json.loads(receipt_path.read_text())["projection"]["params"],
                DONOR_PROJECTION,
            )

    def test_no_donor_and_snapshot_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, snapshot_path = self._donor_run(Path(tmp))
            train, configs_path = _write_recipient(run_dir)
            receipt_path = train.parent / GLOBAL_DONOR_TRANSFER_FILENAME
            configs_before = configs_path.read_bytes()

            # No snapshot path under the new policy is the no_donor binding:
            # nothing is written and the ordinary pool is kept.
            result = inject_global_donor(train, configs_path)
            self.assertEqual(result["status"], "no_donor")
            self.assertFalse(receipt_path.exists())
            self.assertEqual(configs_path.read_bytes(), configs_before)

            # A bound candidate cannot be refreshed without its snapshot.
            inject_global_donor(train, configs_path, snapshot_path)
            with self.assertRaisesRegex(ValueError, "bound"):
                inject_global_donor(train, configs_path)

            # A missing snapshot file blocks instead of degrading.
            with self.assertRaisesRegex(ValueError, "cannot read donor snapshot"):
                inject_global_donor(
                    train,
                    configs_path,
                    run_dir / ".scheduler" / "donors" / "donor-missing.json",
                )

            # A tampered payload fails the content-id check.
            snapshot = json.loads(snapshot_path.read_text())
            snapshot["selected"]["score"] = 0.01
            tampered = run_dir / "tampered_snapshot.json"
            tampered.write_text(json.dumps(snapshot))
            with self.assertRaisesRegex(ValueError, "canonical payload"):
                inject_global_donor(train, configs_path, tampered)

    def test_cli_inactive_policy_changes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, snapshot_path = self._donor_run(
                Path(tmp),
                cfg={
                    "tuner": {
                        "scheduler_policy": "v3_2",
                        "inner_policy": "hebo24-hebo20",
                    }
                },
            )
            train, configs_path = _write_recipient(run_dir)
            configs_before = configs_path.read_bytes()
            argv = [
                "tune_tools.py",
                "inject-global-donor",
                "--candidate-path",
                str(train),
                "--configs-json",
                str(configs_path),
                "--donor-snapshot",
                str(snapshot_path),
            ]
            stdout = io.StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("sys.stdout", new=stdout),
            ):
                self.assertEqual(tune_tools.main(), 0)

            result = json.loads(stdout.getvalue())
            self.assertEqual(result["status"], "inactive")
            self.assertEqual(configs_path.read_bytes(), configs_before)
            self.assertFalse(
                (train.parent / GLOBAL_DONOR_TRANSFER_FILENAME).exists()
            )

    def test_cli_happy_path_prints_status_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, snapshot_path = self._donor_run(Path(tmp))
            train, configs_path = _write_recipient(run_dir)
            argv = [
                "tune_tools.py",
                "inject-global-donor",
                "--candidate-path",
                str(train),
                "--configs-json",
                str(configs_path),
                "--donor-snapshot",
                str(snapshot_path),
            ]
            stdout = io.StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("sys.stdout", new=stdout),
            ):
                self.assertEqual(tune_tools.main(), 0)

            result = json.loads(stdout.getvalue())
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["warm_config_index"], 5)


if __name__ == "__main__":
    unittest.main()
