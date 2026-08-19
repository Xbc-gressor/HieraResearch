from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import warmstart_eval  # noqa: E402
from tune_tools import (  # noqa: E402
    _candidate_execution_revision,
    _read_literal_mapping,
    _read_param_schema,
    authoritative_parent_incumbent,
    materialize_parameter_transfer,
    tuning_record,
    validate_parameter_transfer,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _applied_snapshot(
    parent: Path,
    report_path: Path,
    *,
    source: str,
    score: float,
    params: dict,
) -> dict:
    schema = _read_param_schema(parent)
    return {
        "schema_version": 1,
        "source": source,
        "score": score,
        "params": params,
        "param_schema": schema,
        "entrypoint_sha256": _sha256(parent),
        "tune_report_sha256": _sha256(report_path),
    }


def _parent_source(base: dict) -> str:
    return f"""
PARAM_SCHEMA = {{
    "same": "int",
    "category": ("categorical", ["a", "b"]),
    "changed_kind": "float",
    "dropped": "int",
}}
SEARCH_SPACE = {{
    "same": ("int", 1, 10),
    "category": ("categorical", ["a", "b"]),
    "changed_kind": ("float", 0.1, 1.0),
    "dropped": ("int", 1, 10),
}}
BASE_PARAMS = {base!r}

def make_model(params):
    return params
""".lstrip()


def _child_source() -> str:
    return """
PARAM_SCHEMA = {
    "same": "int",
    "category": ("categorical", ["a", "c"]),
    "changed_kind": "int",
    "new_key": "float",
}

def make_model(params):
    return params
""".lstrip()


class ParameterInheritanceTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        run_dir = root / "runs" / "unit" / "tag"
        parent = run_dir / "candidates" / "001" / "train.py"
        child = run_dir / "candidates" / "002" / "train.py"
        parent.parent.mkdir(parents=True)
        child.parent.mkdir(parents=True)
        parent_params = {
            "same": 7,
            "category": "b",
            "changed_kind": 0.5,
            "dropped": 9,
        }
        parent.write_text(_parent_source(parent_params))
        child.write_text(_child_source())
        (parent.parent / "prepare.py").write_text(
            "def evaluate_config(make_model, params):\n"
            "    return 0.0\n"
        )
        report = {
            "phase_a": {
                "status": "ok",
                "best_warm_params": parent_params,
                "best_warm_score": 0.5,
                "warm_start_configs": [
                    {"params": parent_params, "score": 0.5},
                ],
                "candidate_code_revision": _candidate_execution_revision(parent),
            },
            "phase_c": {
                "stages": [
                    {
                        "method": "bo",
                        "status": "running",
                        "trials": [
                            {
                                "params": {
                                    **parent_params,
                                    "same": 8,
                                },
                                "score": 0.1,
                            }
                        ],
                    }
                ]
            },
        }
        report_path = parent.parent / "tune_report.json"
        report_path.write_text(json.dumps(report))
        (run_dir / "ledger.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "run_id": "001",
                            "final_best_score": 0.5,
                            "applied_incumbent": _applied_snapshot(
                                parent,
                                report_path,
                                source="applied_phase_a",
                                score=0.5,
                                params=parent_params,
                            ),
                        },
                        {
                            "run_id": "002",
                        },
                    ]
                }
            )
        )
        (child.parent / "_candidate_brief.json").write_text(
            json.dumps(
                {
                    "schema_version": 4,
                    "run_id": "002",
                    "op": "improve",
                    "source_run_ids": ["001"],
                    "primary_parent": {
                        "schema_version": 1,
                        "run_id": "001",
                        "path": str(parent),
                        "sha256": _sha256(parent),
                    },
                    "implementation_source": {
                        "kind": "primary_parent_snapshot",
                        "parent_run_id": "001",
                        "path": str(parent),
                        "sha256": _sha256(parent),
                    },
                }
            )
        )
        configs = child.parent / "_warm_configs.json"
        configs.write_text(
            json.dumps(
                [
                    {
                        "same": 2,
                        "category": "a",
                        "changed_kind": 3,
                        "new_key": 0.2,
                    },
                    {
                        "same": 4,
                        "category": "c",
                        "changed_kind": 5,
                        "new_key": 0.4,
                    },
                ]
            )
        )
        return parent, child, configs

    def test_running_phase_c_is_not_an_inheritance_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent, _, _ = self._fixture(Path(tmp))
            incumbent = authoritative_parent_incumbent(parent, "001")

            self.assertEqual(incumbent["source"], "applied_phase_a")
            self.assertEqual(incumbent["score"], 0.5)
            self.assertEqual(incumbent["params"]["same"], 7)

    def test_phase_a_snapshot_survives_later_phase_c_report_growth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent, _, _ = self._fixture(Path(tmp))
            report_path = parent.parent / "tune_report.json"
            report = json.loads(report_path.read_text())
            phase_c = report.pop("phase_c")
            report_path.write_text(json.dumps(report))

            ledger_path = parent.parent.parent.parent / "ledger.json"
            ledger = json.loads(ledger_path.read_text())
            params = report["phase_a"]["best_warm_params"]
            ledger["records"][0]["applied_incumbent"] = _applied_snapshot(
                parent,
                report_path,
                source="applied_phase_a",
                score=0.5,
                params=params,
            )
            ledger_path.write_text(json.dumps(ledger))
            pinned_hash = ledger["records"][0]["applied_incumbent"][
                "tune_report_sha256"
            ]

            report["phase_c"] = phase_c
            report_path.write_text(json.dumps(report))
            self.assertNotEqual(_sha256(report_path), pinned_hash)

            incumbent = authoritative_parent_incumbent(parent, "001")

            self.assertEqual(incumbent["source"], "applied_phase_a")
            self.assertEqual(incumbent["score"], 0.5)
            self.assertEqual(incumbent["tune_report_sha256"], pinned_hash)

    def test_finalized_applied_phase_c_becomes_the_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent, _, _ = self._fixture(Path(tmp))
            report_path = parent.parent / "tune_report.json"
            report = json.loads(report_path.read_text())
            final_params = report["phase_c"]["stages"][0]["trials"][0]["params"]
            report["phase_c"]["stages"][0]["status"] = "ok"
            report.update(
                {
                    "final_best_params": final_params,
                    "final_best_score": 0.1,
                    "applied_to_base_params": True,
                }
            )
            report_path.write_text(json.dumps(report))
            parent.write_text(_parent_source(final_params))
            ledger_path = parent.parent.parent.parent / "ledger.json"
            ledger = json.loads(ledger_path.read_text())
            ledger["records"][0]["final_best_score"] = 0.1
            ledger["records"][0]["applied_incumbent"] = _applied_snapshot(
                parent,
                report_path,
                source="finalized_phase_c",
                score=0.1,
                params=final_params,
            )
            ledger_path.write_text(json.dumps(ledger))

            incumbent = authoritative_parent_incumbent(parent, "001")

            self.assertEqual(incumbent["source"], "finalized_phase_c")
            self.assertEqual(incumbent["score"], 0.1)
            self.assertEqual(incumbent["params"]["same"], 8)

    def test_exact_projection_and_stale_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent, child, configs_path = self._fixture(Path(tmp))
            receipt = materialize_parameter_transfer(child, configs_path)
            configs = json.loads(configs_path.read_text())

            self.assertEqual(
                configs[0],
                {
                    "same": 7,
                    "category": "a",
                    "changed_kind": 3,
                    "new_key": 0.2,
                },
            )
            self.assertEqual(
                [item["key"] for item in receipt["projection"]["copied"]],
                ["same"],
            )
            self.assertEqual(
                {
                    item["key"]: item["reason"]
                    for item in receipt["projection"]["reset"]
                },
                {
                    "category": "categorical_value_removed",
                    "changed_kind": "kind_changed",
                },
            )
            self.assertEqual(
                [item["key"] for item in receipt["projection"]["new"]],
                ["new_key"],
            )
            self.assertEqual(
                [item["key"] for item in receipt["projection"]["dropped"]],
                ["dropped"],
            )
            validate_parameter_transfer(child, configs, receipt)

            mutated_configs = json.loads(configs_path.read_text())
            mutated_configs[0]["same"] = 6
            with self.assertRaisesRegex(ValueError, "config 0"):
                validate_parameter_transfer(child, mutated_configs, receipt)

            child.write_text(child.read_text() + "\nCHANGED = True\n")
            with self.assertRaisesRegex(ValueError, "stale|snapshot"):
                validate_parameter_transfer(child, configs, receipt)

            child.write_text(_child_source())
            report_path = parent.parent / "tune_report.json"
            report = json.loads(report_path.read_text())
            report["unrelated_revision"] = 1
            report_path.write_text(json.dumps(report))
            validate_parameter_transfer(child, configs, receipt)

            report["phase_a"]["best_warm_score"] = 0.4
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "finite warm best"):
                validate_parameter_transfer(child, configs, receipt)

    def test_interrupted_configs_replace_recovers_from_receipt_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, child, configs_path = self._fixture(Path(tmp))
            receipt_path = child.parent / "_parameter_transfer.json"
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
                    materialize_parameter_transfer(child, configs_path)

            self.assertTrue(receipt_path.is_file())
            self.assertEqual(configs_path.read_bytes(), configs_before)
            persisted_receipt = json.loads(receipt_path.read_text())
            self.assertEqual(
                persisted_receipt["candidate"]["defaults"],
                json.loads(configs_before)[0],
            )

            recovered = materialize_parameter_transfer(child, configs_path)
            configs = json.loads(configs_path.read_text())
            self.assertEqual(recovered, json.loads(receipt_path.read_text()))
            validate_parameter_transfer(child, configs, recovered)

    def test_invalid_existing_receipt_fails_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, child, configs_path = self._fixture(Path(tmp))
            receipt_path = child.parent / "_parameter_transfer.json"
            receipt_path.write_text("{broken\n")
            before = {
                configs_path: configs_path.read_bytes(),
                receipt_path: receipt_path.read_bytes(),
            }

            with self.assertRaisesRegex(ValueError, "cannot trust existing"):
                materialize_parameter_transfer(child, configs_path)

            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_same_run_id_receipt_from_another_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, child_a, configs_a = self._fixture(root / "run-a")
            receipt_a = materialize_parameter_transfer(child_a, configs_a)
            _, child_b, configs_b = self._fixture(root / "run-b")
            receipt_b_path = child_b.parent / "_parameter_transfer.json"
            receipt_b_path.write_text(json.dumps(receipt_a))
            before = {
                configs_b: configs_b.read_bytes(),
                receipt_b_path: receipt_b_path.read_bytes(),
            }

            with self.assertRaisesRegex(
                ValueError,
                "different candidate identity",
            ):
                materialize_parameter_transfer(child_b, configs_b)

            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_transfer_output_paths_must_be_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, child, configs_path = self._fixture(Path(tmp))
            before = configs_path.read_bytes()

            with self.assertRaisesRegex(ValueError, "must all be distinct"):
                materialize_parameter_transfer(
                    child,
                    configs_path,
                    receipt_path=configs_path,
                )

            self.assertEqual(configs_path.read_bytes(), before)

    def test_schema_change_cannot_reinterpret_old_projection_as_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, child, configs_path = self._fixture(Path(tmp))
            materialize_parameter_transfer(child, configs_path)
            before = {
                configs_path: configs_path.read_bytes(),
                child.parent / "_parameter_transfer.json": (
                    child.parent / "_parameter_transfer.json"
                ).read_bytes(),
            }
            child.write_text(
                child.read_text().replace(
                    '"new_key": "float"',
                    '"new_key": "float",\n    "another": "int"',
                )
            )

            with self.assertRaisesRegex(ValueError, "still the prior projection"):
                materialize_parameter_transfer(child, configs_path)

            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_tuning_record_preserves_control_receipts_and_observation(self) -> None:
        receipt = {
            "schema_version": 1,
            "receipt_sha256": "sha256:" + "1" * 64,
        }
        control = {
            "warm_config_index": 0,
            "params_sha256": "sha256:" + "2" * 64,
        }
        observation = {
            "params": {"x": 3},
            "score": 0.4,
            "role": "inherited_control",
        }
        report = {
            "phase_a": {
                "status": "ok",
                "trials_attempted": 1,
                "best_warm_score": 0.4,
                "best_warm_params": {"x": 3},
                "warm_start_configs": [observation],
                "search_space": {"x": ["int", 1, 5]},
                "parameter_transfer": receipt,
                "inherited_control": control,
            }
        }

        record = tuning_record(report)

        self.assertEqual(
            record["parameter_transfer"],
            {
                "receipt": receipt,
                "inherited_control": control,
                "warm_start_observations": [observation],
            },
        )

    def test_report_authored_paired_control_is_rejected(self) -> None:
        report = {
            "phase_a": {
                "status": "ok",
                "trials_attempted": 2,
                "best_warm_score": 0.4,
                "best_warm_params": {"x": 1},
                "warm_start_configs": [
                    {"params": {"x": 1}, "score": 0.4},
                    {"params": {"x": 2}, "score": 0.1},
                ],
                "parameter_transfer": {
                    "schema_version": 2,
                    "semantic_control": {"status": "paired"},
                },
            }
        }

        with self.assertRaisesRegex(
            ValueError,
            "report-authored paired semantic controls are not admissible",
        ):
            tuning_record(report)

    def test_failed_control_attempt_is_not_persisted_as_score_evidence(self) -> None:
        report = {
            "phase_a": {
                "status": "crashed",
                "trials_attempted": 1,
                "warm_start_configs": [
                    {
                        "params": {"x": 3},
                        "score": None,
                        "status": "failed",
                        "role": "inherited_control",
                    }
                ],
                "parameter_transfer": {
                    "schema_version": 2,
                    "receipt_sha256": "sha256:" + "1" * 64,
                },
                "inherited_control": {
                    "warm_config_index": 0,
                },
            }
        }

        record = tuning_record(report)

        self.assertEqual(
            record["parameter_transfer"]["warm_start_observations"],
            [],
        )
        self.assertEqual(record["trials_attempted"], 1)

    def test_warm_evaluator_can_apply_the_mandatory_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, child, configs_path = self._fixture(Path(tmp))
            child.write_text(
                _child_source().replace(
                    "\ndef make_model",
                    """
SEARCH_SPACE = {
    "same": ("int", 1, 10),
    "category": ("categorical", ["a", "c"]),
    "changed_kind": ("int", 1, 5),
    "new_key": ("float", 0.1, 0.5),
}

def make_model""",
                )
            )
            (child.parent / "prepare.py").write_text(
                """
def evaluate_config(make_model, params):
    make_model(params)
    return -float(params["same"])
""".lstrip()
            )
            receipt = materialize_parameter_transfer(child, configs_path)
            report_path = child.parent / "tune_report.json"
            argv = [
                "warmstart_eval.py",
                "--candidate-path",
                str(child),
                "--configs-json",
                str(configs_path),
                "--tune-report-json",
                str(report_path),
                "--k-eval",
                "2",
            ]

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                self.assertEqual(warmstart_eval.main(), 0)

            report = json.loads(report_path.read_text())
            phase_a = report["phase_a"]
            self.assertEqual(
                phase_a["warm_config_selection"]["mandatory_indices"],
                [0],
            )
            self.assertEqual(phase_a["parameter_transfer"], receipt)
            self.assertNotIn("param_schema_sha256", receipt["candidate"])
            self.assertNotIn("defaults_sha256", receipt["candidate"])
            self.assertNotIn("incumbent_params_sha256", receipt["primary_parent"])
            self.assertNotIn("param_schema_sha256", receipt["primary_parent"])
            self.assertEqual(
                phase_a["inherited_control"]["receipt_sha256"],
                receipt["receipt_sha256"],
            )
            self.assertEqual(
                phase_a["warm_start_configs"][0],
                {
                    "params": receipt["projection"]["params"],
                    "score": -7.0,
                    "proposed_index": 0,
                    "role": "inherited_control",
                    "parameter_transfer_receipt_sha256": receipt[
                        "receipt_sha256"
                    ],
                    "params_sha256": receipt["projection"]["params_sha256"],
                },
            )
            self.assertEqual(len(phase_a["warm_start_configs"]), 2)
            self.assertEqual(
                phase_a["best_warm_params"], receipt["projection"]["params"]
            )
            self.assertEqual(phase_a["best_warm_score"], -7.0)
            self.assertEqual(
                _read_literal_mapping(child, "BASE_PARAMS"),
                receipt["projection"]["params"],
            )

    def test_nonfresh_screen_still_requires_an_alternative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, child, configs_path = self._fixture(Path(tmp))
            child.write_text(
                _child_source().replace(
                    "\ndef make_model",
                    """
SEARCH_SPACE = {
    "same": ("int", 1, 10),
    "category": ("categorical", ["a", "c"]),
    "changed_kind": ("int", 1, 5),
    "new_key": ("float", 0.1, 0.5),
}

def make_model""",
                )
            )
            (child.parent / "prepare.py").write_text(
                "def evaluate_config(make_model, params):\n"
                "    return float(params['same'])\n"
            )
            materialize_parameter_transfer(child, configs_path)
            argv = [
                "warmstart_eval.py",
                "--candidate-path",
                str(child),
                "--configs-json",
                str(configs_path),
                "--tune-report-json",
                str(child.parent / "tune_report.json"),
                "--k-eval",
                "1",
            ]

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(warmstart_eval, "timed_eval") as evaluate,
                mock.patch("sys.stderr", new=io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    warmstart_eval.main()

            evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
