"""Tests for warmstart_eval under the global-donor policy pair.

Covers only the new warm-evaluation contracts (design §4.2, §4.3, §3.3, §8):
mandatory role indices for fresh/non-fresh/deduplicated donor rows, donor
receipt/snapshot validation before any objective call, donor-only failure
semantics (crash/preflight rejection continue the screening; a donor row that
also carries the lineage control stays fail-closed), resume replay, and the
`phase_a` donor sections (initialization_mode / embedded transfer /
observation).  Old policies must never look at the donor receipt.

Fixtures use real candidate files and receipts produced by the real
inject-global-donor helper; only the objective/preflight calls are mocked.
"""

from __future__ import annotations

import contextlib
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

from tools.scheduler.donor import build_donor_snapshot  # noqa: E402
from tune_tools import (  # noqa: E402
    GLOBAL_DONOR_TRANSFER_FILENAME,
    _candidate_execution_revision,
    inject_global_donor,
    materialize_parameter_transfer,
)
import warmstart_eval  # noqa: E402


NEW_POLICY_CFG = {
    "tuner": {
        "scheduler_policy": "anchor_transfer_challenger_v1",
        "inner_policy": "hebo24-transfer10-hebo10",
    }
}
OLD_POLICY_CFG = {
    "tuner": {
        "scheduler_policy": "v3_2",
        "inner_policy": "hebo24-hebo20",
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
    """A candidate whose one Phase-C bout is closed and applied to BASE_PARAMS."""
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
) -> tuple[Path, Path, Path]:
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
        "implementation_source": {"kind": "generated"},
    }
    if brief_extra:
        brief.update(brief_extra)
    (candidate_dir / "_candidate_brief.json").write_text(json.dumps(brief))
    configs_path = candidate_dir / "_warm_configs.json"
    configs_path.write_text(
        json.dumps(list(ORDINARY_CONFIGS if configs is None else configs))
    )
    return train, configs_path, candidate_dir / "tune_report.json"


def _write_nonfresh_recipient(run_dir: Path) -> tuple[Path, Path, Path]:
    parent_train = run_dir / "candidates" / "020" / "train.py"
    return _write_recipient(
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


def _donor_run(
    root: Path,
    *,
    cfg: dict | None = None,
    donor_schema: dict = DONOR_SCHEMA,
    donor_space: dict = DONOR_SPACE,
    donor_params: dict = DONOR_PARAMS,
    donor_score: float = 0.3,
    parent_params: dict | None = None,
) -> tuple[Path, Path]:
    """A run dir with one eligible donor ("005") and its bound snapshot."""
    run_dir = root / "runs" / "unit" / "tag"
    records = [
        _write_finalized_candidate(
            run_dir,
            "005",
            schema=donor_schema,
            space=donor_space,
            warm_params=donor_params,
            warm_score=0.8,
            final_params=donor_params,
            final_score=donor_score,
        )
    ]
    if parent_params is not None:
        records.append(
            _write_finalized_candidate(
                run_dir,
                "020",
                warm_params=parent_params,
                warm_score=0.95,
                final_params=parent_params,
                final_score=0.9,
            )
        )
    _write_run(root, records, cfg=cfg)
    result = build_donor_snapshot(run_dir)
    assert result["status"] == "ok", result
    return run_dir, run_dir / result["path"]


def _fresh_donor_candidate(root: Path):
    run_dir, snapshot_path = _donor_run(root)
    train, configs_path, report_path = _write_recipient(run_dir)
    result = inject_global_donor(train, configs_path, snapshot_path)
    assert result["status"] == "ok", result
    return run_dir, snapshot_path, train, configs_path, report_path


def _default_score(_evaluate, _make_model, params, *_args, **_kwargs) -> float:
    return float(params["same"])


def _crash_on_donor(_evaluate, _make_model, params, *_args, **_kwargs) -> float:
    if params == DONOR_PROJECTION:
        raise RuntimeError("donor row crashed")
    return float(params["same"])


def _reject_donor_preflight(params, *_args, **_kwargs) -> dict:
    if params == DONOR_PROJECTION:
        raise RuntimeError("donor row preflight rejected")
    return {"status": "ok"}


def _run_warmstart(
    train: Path,
    configs_path: Path,
    report_path: Path,
    *,
    snapshot_path: Path | None = None,
    k_eval: int = 3,
    score_fn=_default_score,
    preflight_fn=None,
) -> tuple[int, mock.Mock, str]:
    timed_eval = mock.Mock(side_effect=score_fn)
    argv = [
        "warmstart_eval.py",
        "--candidate-path", str(train),
        "--configs-json", str(configs_path),
        "--tune-report-json", str(report_path),
        "--k-eval", str(k_eval),
    ]
    if snapshot_path is not None:
        argv += ["--donor-snapshot", str(snapshot_path)]
    stdout = io.StringIO()
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(sys, "argv", argv))
        stack.enter_context(
            mock.patch.object(warmstart_eval, "timed_eval", timed_eval)
        )
        stack.enter_context(mock.patch("sys.stdout", stdout))
        stack.enter_context(mock.patch("sys.stderr", new=io.StringIO()))
        if preflight_fn is not None:
            stack.enter_context(
                mock.patch.object(
                    warmstart_eval,
                    "resolve_preflight_fn",
                    return_value=lambda *_a, **_k: None,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    warmstart_eval,
                    "timed_preflight",
                    mock.Mock(side_effect=preflight_fn),
                )
            )
        code = warmstart_eval.main()
    return code, timed_eval, stdout.getvalue()


def _assert_fails_before_objective(
    train: Path,
    configs_path: Path,
    report_path: Path,
    snapshot_path: Path | None,
) -> None:
    train_bytes = train.read_bytes()
    configs_bytes = configs_path.read_bytes()
    timed_eval = mock.Mock()
    argv = [
        "warmstart_eval.py",
        "--candidate-path", str(train),
        "--configs-json", str(configs_path),
        "--tune-report-json", str(report_path),
        "--k-eval", "3",
    ]
    if snapshot_path is not None:
        argv += ["--donor-snapshot", str(snapshot_path)]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(warmstart_eval, "timed_eval", timed_eval),
        mock.patch("sys.stdout", new=io.StringIO()),
        mock.patch("sys.stderr", new=io.StringIO()),
    ):
        try:
            warmstart_eval.main()
        except SystemExit:
            pass
        else:
            raise AssertionError("warmstart_eval.main() did not exit")
    timed_eval.assert_not_called()
    # No BASE_PARAMS write, no configs mutation, no report creation.
    assert train.read_bytes() == train_bytes
    assert configs_path.read_bytes() == configs_bytes
    assert not report_path.exists()


class WarmstartGlobalDonorTests(unittest.TestCase):
    def test_fresh_donor_append_mandatory_roles_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _run_dir, snapshot_path, train, configs_path, report_path = (
                _fresh_donor_candidate(Path(tmp))
            )
            receipt_path = train.parent / GLOBAL_DONOR_TRANSFER_FILENAME
            receipt_bytes = receipt_path.read_bytes()

            code, timed_eval, _stdout = _run_warmstart(
                train, configs_path, report_path, snapshot_path=snapshot_path
            )

            self.assertEqual(code, 0)
            self.assertEqual(timed_eval.call_count, 3)
            phase_a = json.loads(report_path.read_text())["phase_a"]
            selection = phase_a["warm_config_selection"]
            # fresh + donor: the donor row alone is mandatory; the remaining
            # two slots are sampled from the ordinary pool.
            self.assertEqual(selection["mandatory_indices"], [5])
            self.assertEqual(selection["population_size"], 6)
            self.assertEqual(len(selection["selected_indices"]), 3)
            self.assertEqual(len(selection["deferred_indices"]), 3)
            self.assertEqual(len(phase_a["deferred_configs"]), 3)
            self.assertEqual(selection["selected_indices"][0], 5)
            rows = phase_a["warm_start_configs"]
            self.assertEqual(rows[0]["proposed_index"], 5)
            self.assertEqual(rows[0]["role"], "global_donor")
            self.assertNotIn("role", rows[1])
            self.assertNotIn("role", rows[2])
            self.assertEqual(phase_a["initialization_mode"], "global_donor")
            self.assertEqual(
                phase_a["global_donor_observation"],
                {
                    "warm_config_index": 5,
                    "status": "finite",
                    "score": 7.0,
                    "failure_ref": None,
                },
            )
            embedded = phase_a["global_donor_transfer"]
            self.assertEqual(embedded["status"], "ok")
            self.assertEqual(embedded["warm_config_index"], 5)
            self.assertEqual(embedded["mandatory_role_indices"], [5])
            self.assertEqual(embedded["k_eval"], 3)
            self.assertEqual(embedded["projection"]["params"], DONOR_PROJECTION)
            self.assertEqual(phase_a["k_finite"], 3)
            self.assertEqual(phase_a["k_crashed"], 0)
            self.assertEqual(phase_a["k_preflight_rejected"], 0)
            self.assertEqual(
                phase_a["best_warm_score"],
                min(row["score"] for row in rows),
            )
            # The helper-owned receipt file is not rewritten by warmstart.
            self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
            self.assertIsNone(
                json.loads(receipt_path.read_text())["mandatory_role_indices"]
            )

    def test_nonfresh_donor_mandatory_includes_lineage_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # The donor's "same" is float-typed, so its projection resets to
            # the child default (2) — provably distinct from the lineage
            # control (3), so the donor row appends at index 5.
            float_same_schema = {**DONOR_SCHEMA, "same": "float"}
            float_same_space = {**DONOR_SPACE, "same": ("float", 1.0, 10.0)}
            run_dir, snapshot_path = _donor_run(
                root,
                donor_schema=float_same_schema,
                donor_space=float_same_space,
                donor_params={**DONOR_PARAMS, "same": 7.0},
                parent_params={**DONOR_PARAMS, "same": 3},
            )
            train, configs_path, report_path = _write_nonfresh_recipient(run_dir)
            materialize_parameter_transfer(train, configs_path)
            result = inject_global_donor(train, configs_path, snapshot_path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["warm_config_index"], 5)

            code, timed_eval, _stdout = _run_warmstart(
                train, configs_path, report_path, snapshot_path=snapshot_path
            )

            self.assertEqual(code, 0)
            self.assertEqual(timed_eval.call_count, 3)
            phase_a = json.loads(report_path.read_text())["phase_a"]
            selection = phase_a["warm_config_selection"]
            self.assertEqual(selection["mandatory_indices"], [0, 5])
            self.assertEqual(selection["population_size"], 6)
            self.assertEqual(len(selection["selected_indices"]), 3)
            self.assertEqual(len(selection["deferred_indices"]), 3)
            rows = phase_a["warm_start_configs"]
            self.assertEqual(rows[0]["proposed_index"], 0)
            self.assertEqual(rows[0]["role"], "inherited_control")
            self.assertEqual(rows[1]["proposed_index"], 5)
            self.assertEqual(rows[1]["role"], "global_donor")
            self.assertEqual(phase_a["initialization_mode"], "global_donor")
            self.assertIn("parameter_transfer", phase_a)
            self.assertEqual(
                phase_a["inherited_control"]["warm_config_index"], 0
            )
            self.assertEqual(
                phase_a["global_donor_observation"],
                {
                    "warm_config_index": 5,
                    "status": "finite",
                    "score": 2.0,
                    "failure_ref": None,
                },
            )
            self.assertEqual(
                phase_a["global_donor_transfer"]["mandatory_role_indices"],
                [0, 5],
            )

    def test_dedup_with_lineage_control_shares_one_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # The primary parent projects to exactly the donor row, so the
            # lineage control at index 0 and the donor coincide.
            run_dir, snapshot_path = _donor_run(root, parent_params=DONOR_PARAMS)
            train, configs_path, report_path = _write_nonfresh_recipient(run_dir)
            materialize_parameter_transfer(train, configs_path)
            result = inject_global_donor(train, configs_path, snapshot_path)
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["deduplicated"])
            self.assertEqual(result["warm_config_index"], 0)

            code, timed_eval, _stdout = _run_warmstart(
                train, configs_path, report_path, snapshot_path=snapshot_path
            )

            self.assertEqual(code, 0)
            self.assertEqual(timed_eval.call_count, 3)
            phase_a = json.loads(report_path.read_text())["phase_a"]
            selection = phase_a["warm_config_selection"]
            # One index carries both roles; the mandatory set dedups and the
            # remaining slots come from the frozen permutation.
            self.assertEqual(selection["mandatory_indices"], [0])
            self.assertEqual(selection["population_size"], 5)
            self.assertEqual(len(selection["selected_indices"]), 3)
            self.assertEqual(len(selection["deferred_indices"]), 2)
            self.assertEqual(len(phase_a["deferred_configs"]), 2)
            rows = phase_a["warm_start_configs"]
            self.assertEqual(rows[0]["proposed_index"], 0)
            self.assertEqual(
                rows[0]["role"], ["inherited_control", "global_donor"]
            )
            self.assertEqual(phase_a["initialization_mode"], "global_donor")
            self.assertEqual(
                phase_a["global_donor_observation"],
                {
                    "warm_config_index": 0,
                    "status": "finite",
                    "score": 7.0,
                    "failure_ref": None,
                },
            )

    def test_dedup_with_ordinary_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, snapshot_path = _donor_run(root)
            configs = list(ORDINARY_CONFIGS)
            configs[2] = dict(DONOR_PROJECTION)
            train, configs_path, report_path = _write_recipient(
                run_dir, configs=configs
            )
            result = inject_global_donor(train, configs_path, snapshot_path)
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["deduplicated"])
            self.assertEqual(result["warm_config_index"], 2)

            code, _timed_eval, _stdout = _run_warmstart(
                train, configs_path, report_path, snapshot_path=snapshot_path
            )

            self.assertEqual(code, 0)
            phase_a = json.loads(report_path.read_text())["phase_a"]
            selection = phase_a["warm_config_selection"]
            self.assertEqual(selection["mandatory_indices"], [2])
            self.assertEqual(selection["population_size"], 5)
            self.assertEqual(len(selection["selected_indices"]), 3)
            self.assertEqual(len(selection["deferred_indices"]), 2)
            donor_rows = [
                row
                for row in phase_a["warm_start_configs"]
                if row["proposed_index"] == 2
            ]
            self.assertEqual(len(donor_rows), 1)
            self.assertEqual(donor_rows[0]["role"], "global_donor")
            self.assertEqual(phase_a["initialization_mode"], "global_donor")
            self.assertEqual(
                phase_a["global_donor_observation"]["warm_config_index"], 2
            )

    def test_resume_replays_selection_and_keeps_donor_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, snapshot_path, train, configs_path, report_path = (
                _fresh_donor_candidate(Path(tmp))
            )

            code, _first_eval, _stdout = _run_warmstart(
                train, configs_path, report_path, snapshot_path=snapshot_path
            )
            self.assertEqual(code, 0)
            first_report = json.loads(report_path.read_text())
            first_selection = first_report["phase_a"]["warm_config_selection"]

            code, second_eval, _stdout = _run_warmstart(
                train, configs_path, report_path, snapshot_path=snapshot_path
            )
            self.assertEqual(code, 0)
            second_eval.assert_not_called()
            second_report = json.loads(report_path.read_text())
            # Resume replays the frozen selection and reuses cached scores;
            # no redraw, no re-evaluation.
            self.assertEqual(
                second_report["phase_a"]["warm_config_selection"],
                first_selection,
            )
            self.assertEqual(
                second_report["phase_a"]["warm_start_configs"],
                first_report["phase_a"]["warm_start_configs"],
            )
            self.assertEqual(
                second_report["phase_a"]["global_donor_observation"],
                first_report["phase_a"]["global_donor_observation"],
            )

            # A different snapshot after selection is a binding change and
            # blocks before any objective call.
            better = _write_finalized_candidate(
                run_dir,
                "006",
                warm_params=DONOR_PARAMS,
                warm_score=0.8,
                final_params={**DONOR_PARAMS, "same": 8},
                final_score=0.1,
            )
            ledger_path = run_dir / "ledger.json"
            ledger = json.loads(ledger_path.read_text())
            ledger["records"].append(better)
            ledger_path.write_text(json.dumps(ledger))
            newer = build_donor_snapshot(run_dir)
            self.assertEqual(newer["status"], "ok")
            self.assertNotEqual(
                newer["snapshot_id"],
                first_report["phase_a"]["global_donor_transfer"]["donor"][
                    "snapshot_id"
                ],
            )
            timed_eval = mock.Mock()
            argv = [
                "warmstart_eval.py",
                "--candidate-path", str(train),
                "--configs-json", str(configs_path),
                "--tune-report-json", str(report_path),
                "--k-eval", "3",
                "--donor-snapshot", str(run_dir / newer["path"]),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(warmstart_eval, "timed_eval", timed_eval),
                mock.patch("sys.stdout", new=io.StringIO()),
                mock.patch("sys.stderr", new=io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    warmstart_eval.main()
            timed_eval.assert_not_called()

    def test_donor_only_crash_continues_and_resume_keeps_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _run_dir, snapshot_path, train, configs_path, report_path = (
                _fresh_donor_candidate(Path(tmp))
            )

            code, timed_eval, stdout = _run_warmstart(
                train,
                configs_path,
                report_path,
                snapshot_path=snapshot_path,
                score_fn=_crash_on_donor,
            )

            # The donor treatment crashed; the remaining rows still ran and
            # the candidate finalizes from min(finite selected scores).
            self.assertEqual(code, 0)
            self.assertEqual(timed_eval.call_count, 3)
            phase_a = json.loads(report_path.read_text())["phase_a"]
            self.assertEqual(phase_a["status"], "ok")
            self.assertEqual(phase_a["k_evaluated"], 3)
            self.assertEqual(phase_a["k_finite"], 2)
            self.assertEqual(phase_a["k_crashed"], 1)
            self.assertEqual(phase_a["k_preflight_rejected"], 0)
            # The failed admitted objective still counts against the budget.
            self.assertEqual(phase_a["trials_attempted"], 3)
            rows = phase_a["warm_start_configs"]
            self.assertEqual(len(rows), 3)
            donor_row = rows[0]
            self.assertEqual(donor_row["proposed_index"], 5)
            self.assertEqual(donor_row["role"], "global_donor")
            self.assertIsNone(donor_row["score"])
            self.assertEqual(donor_row["status"], "failed")
            self.assertIn("failure_ref", donor_row)
            observation = phase_a["global_donor_observation"]
            self.assertEqual(observation["warm_config_index"], 5)
            self.assertEqual(observation["status"], "crash")
            self.assertIsNone(observation["score"])
            self.assertEqual(
                observation["failure_ref"], donor_row["failure_ref"]
            )
            finite_scores = [
                row["score"] for row in rows if row["score"] is not None
            ]
            self.assertEqual(len(finite_scores), 2)
            self.assertEqual(phase_a["best_warm_score"], min(finite_scores))
            self.assertEqual(phase_a["initialization_mode"], "global_donor")
            self.assertEqual(
                json.loads(stdout)["global_donor_observation"]["status"],
                "crash",
            )

            # Resume: the recorded donor failure is the transfer observation —
            # it is neither redrawn nor re-evaluated.
            code, second_eval, _stdout = _run_warmstart(
                train,
                configs_path,
                report_path,
                snapshot_path=snapshot_path,
                score_fn=_crash_on_donor,
            )
            self.assertEqual(code, 0)
            second_eval.assert_not_called()
            resumed = json.loads(report_path.read_text())["phase_a"]
            self.assertEqual(resumed["trials_attempted"], 3)
            self.assertEqual(
                resumed["global_donor_observation"]["status"], "crash"
            )
            self.assertEqual(
                resumed["warm_start_configs"],
                phase_a["warm_start_configs"],
            )

    def test_donor_preflight_rejection_continues_without_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _run_dir, snapshot_path, train, configs_path, report_path = (
                _fresh_donor_candidate(Path(tmp))
            )

            code, timed_eval, _stdout = _run_warmstart(
                train,
                configs_path,
                report_path,
                snapshot_path=snapshot_path,
                preflight_fn=_reject_donor_preflight,
            )

            self.assertEqual(code, 0)
            # Preflight rejection consumes no objective slot; the other two
            # selected rows are evaluated normally.
            self.assertEqual(timed_eval.call_count, 2)
            report = json.loads(report_path.read_text())
            phase_a = report["phase_a"]
            self.assertEqual(phase_a["status"], "ok")
            self.assertEqual(phase_a["trials_attempted"], 2)
            self.assertEqual(phase_a["k_evaluated"], 3)
            self.assertEqual(phase_a["k_finite"], 2)
            self.assertEqual(phase_a["k_crashed"], 0)
            self.assertEqual(phase_a["k_preflight_rejected"], 1)
            self.assertEqual(len(phase_a["warm_start_configs"]), 2)
            observation = phase_a["global_donor_observation"]
            self.assertEqual(observation["warm_config_index"], 5)
            self.assertEqual(observation["status"], "preflight_rejected")
            self.assertIsNone(observation["score"])
            self.assertIsNotNone(observation["failure_ref"])
            preflight_attempts = report["preflight"]["attempts"]
            self.assertEqual(len(preflight_attempts), 3)
            self.assertEqual(
                [attempt["status"] for attempt in preflight_attempts],
                ["failed", "ok", "ok"],
            )
            self.assertEqual(
                phase_a["best_warm_score"],
                min(row["score"] for row in phase_a["warm_start_configs"]),
            )

            # Resume keeps the rejection; no preflight or objective re-run
            # for the donor row.
            code, second_eval, _stdout = _run_warmstart(
                train,
                configs_path,
                report_path,
                snapshot_path=snapshot_path,
                preflight_fn=_reject_donor_preflight,
            )
            self.assertEqual(code, 0)
            second_eval.assert_not_called()
            resumed = json.loads(report_path.read_text())["phase_a"]
            self.assertEqual(
                resumed["global_donor_observation"]["status"],
                "preflight_rejected",
            )
            self.assertEqual(resumed["k_preflight_rejected"], 1)

    def test_donor_lineage_overlap_crash_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, snapshot_path = _donor_run(root, parent_params=DONOR_PARAMS)
            train, configs_path, report_path = _write_nonfresh_recipient(run_dir)
            materialize_parameter_transfer(train, configs_path)
            result = inject_global_donor(train, configs_path, snapshot_path)
            self.assertTrue(result["deduplicated"])
            self.assertEqual(result["warm_config_index"], 0)

            code, timed_eval, _stdout = _run_warmstart(
                train,
                configs_path,
                report_path,
                snapshot_path=snapshot_path,
                score_fn=_crash_on_donor,
            )

            # The shared row is the lineage fidelity control: its crash is
            # fatal even though it is also the donor.
            self.assertEqual(code, warmstart_eval.CRASHED)
            self.assertEqual(timed_eval.call_count, 1)
            phase_a = json.loads(report_path.read_text())["phase_a"]
            self.assertEqual(phase_a["status"], "crashed")
            self.assertEqual(phase_a["trials_attempted"], 1)
            rows = phase_a["warm_start_configs"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["role"], ["inherited_control", "global_donor"]
            )
            self.assertEqual(rows[0]["status"], "failed")
            self.assertEqual(
                phase_a["global_donor_observation"]["status"], "crash"
            )
            self.assertEqual(phase_a["k_crashed"], 1)

    def test_no_finite_rows_takes_the_candidate_crash_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _run_dir, snapshot_path, train, configs_path, report_path = (
                _fresh_donor_candidate(Path(tmp))
            )

            code, timed_eval, stdout = _run_warmstart(
                train,
                configs_path,
                report_path,
                snapshot_path=snapshot_path,
                k_eval=1,
                score_fn=_crash_on_donor,
            )

            self.assertEqual(code, warmstart_eval.CRASHED)
            self.assertEqual(timed_eval.call_count, 1)
            phase_a = json.loads(report_path.read_text())["phase_a"]
            self.assertEqual(phase_a["status"], "crashed")
            self.assertEqual(phase_a["k_finite"], 0)
            self.assertEqual(phase_a["k_crashed"], 1)
            self.assertNotIn("best_warm_score", phase_a)
            self.assertEqual(
                phase_a["global_donor_observation"]["status"], "crash"
            )
            self.assertEqual(json.loads(stdout)["status"], "crashed")

    def test_no_donor_binding_runs_ordinary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, _snapshot_path = _donor_run(root)
            train, configs_path, report_path = _write_recipient(run_dir)

            code, timed_eval, stdout = _run_warmstart(
                train, configs_path, report_path
            )

            self.assertEqual(code, 0)
            self.assertEqual(timed_eval.call_count, 3)
            phase_a = json.loads(report_path.read_text())["phase_a"]
            self.assertEqual(phase_a["initialization_mode"], "ordinary")
            self.assertIsNone(phase_a["global_donor_transfer"])
            self.assertIsNone(phase_a["global_donor_observation"])
            self.assertEqual(
                phase_a["warm_config_selection"]["mandatory_indices"], [0]
            )
            self.assertEqual(phase_a["k_finite"], 3)
            self.assertEqual(json.loads(stdout)["initialization_mode"], "ordinary")

    def test_donor_incompatible_runs_ordinary_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, snapshot_path = _donor_run(root)
            narrow_space = {**RECIPIENT_SPACE, "same": ("int", 1, 5)}
            configs = [
                {**config, "same": min(config["same"], 5)}
                for config in ORDINARY_CONFIGS
            ]
            train, configs_path, report_path = _write_recipient(
                run_dir, configs=configs, space=narrow_space
            )
            result = inject_global_donor(train, configs_path, snapshot_path)
            self.assertEqual(result["status"], "donor_incompatible")

            code, timed_eval, _stdout = _run_warmstart(
                train, configs_path, report_path, snapshot_path=snapshot_path
            )

            self.assertEqual(code, 0)
            self.assertEqual(timed_eval.call_count, 3)
            phase_a = json.loads(report_path.read_text())["phase_a"]
            self.assertEqual(phase_a["initialization_mode"], "ordinary")
            self.assertIsNone(phase_a["global_donor_observation"])
            embedded = phase_a["global_donor_transfer"]
            self.assertEqual(embedded["status"], "donor_incompatible")
            self.assertIsNone(embedded["mandatory_role_indices"])
            selection = phase_a["warm_config_selection"]
            self.assertEqual(selection["mandatory_indices"], [0])
            self.assertEqual(selection["population_size"], 5)
            self.assertEqual(len(selection["selected_indices"]), 3)
            self.assertEqual(len(selection["deferred_indices"]), 2)
            for row in phase_a["warm_start_configs"]:
                self.assertNotIn("role", row)

    def test_old_policy_never_reads_donor_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, snapshot_path = _donor_run(root, cfg=OLD_POLICY_CFG)
            train, configs_path, report_path = _write_recipient(run_dir)
            # Even a bogus receipt file is invisible to old policies.
            (train.parent / GLOBAL_DONOR_TRANSFER_FILENAME).write_text(
                json.dumps({"not": "a receipt"})
            )

            code, timed_eval, stdout = _run_warmstart(
                train, configs_path, report_path
            )

            self.assertEqual(code, 0)
            self.assertEqual(timed_eval.call_count, 3)
            phase_a = json.loads(report_path.read_text())["phase_a"]
            for key in (
                "initialization_mode",
                "global_donor_transfer",
                "global_donor_observation",
                "k_finite",
                "k_crashed",
                "k_preflight_rejected",
            ):
                self.assertNotIn(key, phase_a)
            self.assertNotIn("initialization_mode", json.loads(stdout))
            self.assertEqual(
                phase_a["warm_config_selection"]["mandatory_indices"], [0]
            )

            # Passing --donor-snapshot under an old policy is equally inert.
            report_path.unlink()
            code, _eval, _stdout = _run_warmstart(
                train, configs_path, report_path, snapshot_path=snapshot_path
            )
            self.assertEqual(code, 0)
            phase_a = json.loads(report_path.read_text())["phase_a"]
            self.assertNotIn("initialization_mode", phase_a)


class WarmstartGlobalDonorValidationTests(unittest.TestCase):
    def _candidate(self, root: Path):
        run_dir, snapshot_path = _donor_run(root)
        train, configs_path, report_path = _write_recipient(run_dir)
        inject_global_donor(train, configs_path, snapshot_path)
        return run_dir, snapshot_path, train, configs_path, report_path

    def test_missing_receipt_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _run_dir, snapshot_path = _donor_run(root)
            train, configs_path, report_path = _write_recipient(_run_dir)
            _assert_fails_before_objective(
                train, configs_path, report_path, snapshot_path
            )

    def test_stale_receipt_fails_until_helper_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _run_dir, snapshot_path, train, configs_path, report_path = (
                self._candidate(Path(tmp))
            )
            # A code edit after injection makes the receipt stale.
            train.write_text(train.read_text() + "\n# comment tweak\n")
            _assert_fails_before_objective(
                train, configs_path, report_path, snapshot_path
            )

            # Re-running the helper rebuilds the projection from the same
            # bound snapshot and warm evaluation proceeds.
            result = inject_global_donor(train, configs_path, snapshot_path)
            self.assertEqual(result["status"], "ok")
            code, timed_eval, _stdout = _run_warmstart(
                train, configs_path, report_path, snapshot_path=snapshot_path
            )
            self.assertEqual(code, 0)
            self.assertEqual(timed_eval.call_count, 3)

    def test_snapshot_id_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, snapshot_path, train, configs_path, report_path = (
                self._candidate(Path(tmp))
            )
            better = _write_finalized_candidate(
                run_dir,
                "006",
                warm_params=DONOR_PARAMS,
                warm_score=0.8,
                final_params={**DONOR_PARAMS, "same": 8},
                final_score=0.1,
            )
            ledger_path = run_dir / "ledger.json"
            ledger = json.loads(ledger_path.read_text())
            ledger["records"].append(better)
            ledger_path.write_text(json.dumps(ledger))
            newer = build_donor_snapshot(run_dir)
            self.assertNotEqual(
                newer["snapshot_id"],
                json.loads(snapshot_path.read_text())["snapshot_id"],
            )
            _assert_fails_before_objective(
                train,
                configs_path,
                report_path,
                run_dir / newer["path"],
            )

    def test_tampered_donor_row_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _run_dir, snapshot_path, train, configs_path, report_path = (
                self._candidate(Path(tmp))
            )
            configs = json.loads(configs_path.read_text())
            configs[5]["same"] = 6
            configs_path.write_text(json.dumps(configs))
            _assert_fails_before_objective(
                train, configs_path, report_path, snapshot_path
            )

    def test_receipt_without_snapshot_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _run_dir, _snapshot_path, train, configs_path, report_path = (
                self._candidate(Path(tmp))
            )
            # A receipt exists but the run passed no --donor-snapshot (the
            # no_donor binding): refusing beats silently dropping the donor.
            _assert_fails_before_objective(
                train, configs_path, report_path, None
            )


if __name__ == "__main__":
    unittest.main()
