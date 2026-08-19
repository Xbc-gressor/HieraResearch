"""Regime-conditioned inner-tuner policy tests.

Covers the policy module itself and its wiring through phase-c-action, stage
admission, validate-proposals, select-candidate, the driver job builder, and
the bout kernels that changed shape: bo --sampler random (FIRST),
hebo_search (CONTINUE, prompt-v2 pool + official HEBO MACE), and
spsa_search (DEEP). Legacy-pinned behavior lives in the pre-existing
test files (test_deep_tune_governance et al.); this file exercises the new
default policy.
"""

from __future__ import annotations

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

import inner_policy  # noqa: E402
from _common import (  # noqa: E402
    DeepTuneStageAdmissionError,
    deep_tune_time_budget,
)
from bo_search import main as bo_main  # noqa: E402
from driver.jobs import build_driver_job  # noqa: E402
from driver.roles import InvocationContext  # noqa: E402
from driver.session import FakeSessionRunner  # noqa: E402
import hebo_search  # noqa: E402
from hebo_search import main as hebo_main  # noqa: E402
from arms.llm_pool_self_rank import ARM as SELF_RANK_ARM  # noqa: E402
from arms.mixup_pool_hebo import ARM as MIXUP_ARM  # noqa: E402
from arms.turbo import ARM as TURBO_ARM  # noqa: E402
from spsa_search import main as spsa_main  # noqa: E402
from tune_tools import (  # noqa: E402
    _candidate_execution_revision,
    phase_c_action,
    select_candidate,
    validate_proposals,
)


POLICY = inner_policy.POLICY_ID


def _write_candidate(
    candidate: Path,
    *,
    space: dict,
    base: dict,
    schema: dict | None = None,
) -> Path:
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if schema is None:
        schema = {}
        for key, entry in space.items():
            if entry[0] == "float" and len(entry) >= 4 and entry[3] == "log":
                schema[key] = ("float", "log")
            elif entry[0] == "categorical":
                schema[key] = ("categorical", list(entry[1]))
            else:
                schema[key] = entry[0]
    candidate.write_text(
        f"PARAM_SCHEMA = {schema!r}\n"
        f"SEARCH_SPACE = {space!r}\n"
        f"BASE_PARAMS = {base!r}\n"
        "def make_model(params):\n"
        "    return params\n"
    )
    (candidate.parent / "prepare.py").write_text(
        "def evaluate_config(make_model, params):\n"
        "    return 0.0\n"
    )
    return candidate


def _fresh_report(candidate: Path, space: dict, base: dict) -> dict:
    return {
        "phase_a": {
            "status": "ok",
            "candidate_code_revision": _candidate_execution_revision(candidate),
            "search_space": {k: list(v) for k, v in space.items()},
            "warm_start_configs": [{"params": dict(base), "score": 1.0}],
            "best_warm_params": dict(base),
            "best_warm_score": 1.0,
        }
    }


def _finalize_bout(report: dict, candidate: Path, *, best: float) -> None:
    """Stamp a validated applied close over the report's current stages."""
    stages = report["phase_c"]["stages"]
    report["final_best_params"] = dict(report["phase_a"]["best_warm_params"])
    report["final_best_score"] = best
    report["applied_to_base_params"] = True
    report["last_finalized_stage_index"] = len(stages) - 1


def _fixture(
    root: Path,
    *,
    space: dict,
    base: dict,
    legacy: bool = False,
    inner_policy_id: str | None = None,
):
    candidate = _write_candidate(
        root / "candidates" / "001" / "train.py", space=space, base=base
    )
    policy_id = "legacy" if legacy else inner_policy_id
    if policy_id is not None:
        tuner = {"inner_policy": policy_id}
        config = {"tuner": tuner}
        if policy_id in inner_policy.INITIAL24_POLICY_IDS:
            config["max_evaluations"] = 100
            tuner.update(
                {
                    "scheduler_policy": "anchor_challenger_v1",
                    "deep_tune_budget_fraction": None,
                    "deep_tune_per_candidate_cap": 44,
                }
            )
        (root / "framework_cfg.json").write_text(
            json.dumps(config)
        )
    report = _fresh_report(candidate, space, base)
    if policy_id is not None:
        # Production stamps the admitting policy on the report (see
        # _common._deep_tune_time_budget_locked); chain validation reads it.
        report["inner_policy"] = policy_id
    report_path = candidate.parent / "tune_report.json"
    report_path.write_text(json.dumps(report))
    return candidate, report_path


FLOAT3 = {"a": ("float", 0.0, 1.0), "b": ("float", 1e-4, 1.0, "log"), "c": ("float", -1.0, 1.0)}
BASE3 = {"a": 0.5, "b": 0.01, "c": 0.0}
INT_ONLY = {"n": ("int", 1, 4)}
INT_BASE = {"n": 2}


class InnerPolicyUnitTest(unittest.TestCase):
    def test_regime_and_bout_size(self):
        self.assertEqual(
            [inner_policy.regime_for_bout_index(i) for i in range(4)],
            ["FIRST", "CONTINUE", "DEEP", "DEEP"],
        )
        self.assertEqual(
            [inner_policy.bout_size(r) for r in ("FIRST", "CONTINUE", "DEEP")],
            [8, 10, 10],
        )

    def test_expected_bout_trials(self):
        self.assertEqual(
            inner_policy.expected_bout_trials(POLICY, 0, 10), 8
        )
        self.assertEqual(
            inner_policy.expected_bout_trials(POLICY, 2, 10), 10
        )
        self.assertEqual(
            inner_policy.expected_bout_trials("legacy", 0, 10), 10
        )
        self.assertEqual(
            [
                inner_policy.expected_bout_trials(
                    inner_policy.MIXUP_TURBO_POLICY_ID, index, 10
                )
                for index in range(3)
            ],
            [24, 10, 10],
        )
        self.assertEqual(
            [
                inner_policy.expected_bout_trials(
                    inner_policy.HEBO_TURBO_POLICY_ID, index, 10
                )
                for index in range(3)
            ],
            [24, 10, 10],
        )
        self.assertEqual(
            [
                inner_policy.expected_bout_trials(
                    inner_policy.HEBO_HEBO_POLICY_ID, index, 10
                )
                for index in range(3)
            ],
            [24, 10, 10],
        )
        with self.assertRaisesRegex(ValueError, "exactly 3 bouts"):
            inner_policy.expected_bout_trials(
                inner_policy.MIXUP_TURBO_POLICY_ID, 3, 10
            )

    def test_method_chains(self):
        self.assertEqual(
            inner_policy.method_chain_for_bout(POLICY, 0, FLOAT3), ["bo"]
        )
        self.assertEqual(
            inner_policy.method_chain_for_bout(POLICY, 1, FLOAT3),
            ["hebo"],
        )
        self.assertEqual(
            inner_policy.method_chain_for_bout(POLICY, 1, INT_ONLY),
            ["hebo"],
        )
        self.assertEqual(
            inner_policy.method_chain_for_bout(POLICY, 2, FLOAT3), ["spsa"]
        )
        self.assertEqual(
            inner_policy.method_chain_for_bout(
                inner_policy.LOCAL_TR_POLICY_ID, 0, FLOAT3
            ),
            ["local_tr"],
        )
        self.assertEqual(
            inner_policy.method_chain_for_bout(
                inner_policy.LOCAL_TR_POLICY_ID, 1, FLOAT3
            ),
            ["hebo"],
        )
        self.assertEqual(
            inner_policy.method_chain_for_bout(
                inner_policy.LOCAL_TR_POLICY_ID, 2, FLOAT3
            ),
            ["spsa"],
        )
        # localtr8-hebo10-hebo10-v1: local_tr FIRST, hebo everywhere else.
        self.assertEqual(
            [
                inner_policy.method_chain_for_bout(
                    inner_policy.LOCAL_TR_HEBO_POLICY_ID, i, FLOAT3
                )
                for i in range(4)
            ],
            [["local_tr"], ["hebo"], ["hebo"], ["hebo"]],
        )
        self.assertEqual(
            inner_policy.method_chain_for_bout(
                inner_policy.LOCAL_TR_HEBO_POLICY_ID, 2, INT_ONLY
            ),
            ["hebo"],
        )
        self.assertEqual(
            [
                inner_policy.method_chain_for_bout(
                    inner_policy.SELF_RANK_HEBO_POLICY_ID, i, FLOAT3
                )
                for i in range(4)
            ],
            [["selfrank"], ["hebo"], ["hebo"], ["hebo"]],
        )
        self.assertEqual(
            [
                inner_policy.method_chain_for_bout(
                    inner_policy.MIXUP_TURBO_POLICY_ID, index, FLOAT3
                )
                for index in range(3)
            ],
            [["mixup"], ["turbo"], ["turbo"]],
        )
        self.assertEqual(
            [
                inner_policy.method_chain_for_bout(
                    inner_policy.HEBO_TURBO_POLICY_ID, index, FLOAT3
                )
                for index in range(3)
            ],
            [["hebo"], ["turbo"], ["turbo"]],
        )
        self.assertEqual(
            [
                inner_policy.method_chain_for_bout(
                    inner_policy.HEBO_HEBO_POLICY_ID, index, FLOAT3
                )
                for index in range(3)
            ],
            [["hebo"], ["hebo"], ["hebo"]],
        )
        with self.assertRaisesRegex(ValueError, "exactly 3 bouts"):
            inner_policy.method_chain_for_bout(
                inner_policy.MIXUP_TURBO_POLICY_ID, 3, FLOAT3
            )
        # legacy: every bout keeps the old production chain.
        self.assertEqual(
            inner_policy.method_chain_for_bout("legacy", 0, INT_ONLY),
            ["grid", "bo"],
        )
        self.assertEqual(
            inner_policy.method_chain_for_bout("legacy", 2, FLOAT3),
            ["bo", "cmaes"],
        )

    def test_sampler_and_rewarm_rules(self):
        self.assertEqual(inner_policy.bo_sampler_for_bout(POLICY, 0), "random")
        self.assertEqual(inner_policy.bo_sampler_for_bout(POLICY, 1), "tpe")
        self.assertEqual(inner_policy.bo_sampler_for_bout("legacy", 0), "tpe")
        self.assertEqual(
            [inner_policy.rewarm_allowed(POLICY, i) for i in range(4)],
            [False, False, False, False],
        )
        self.assertEqual(
            [inner_policy.rewarm_allowed("legacy", i) for i in range(4)],
            [False, True, True, True],
        )
        self.assertEqual(
            inner_policy.bo_sampler_for_bout(inner_policy.LOCAL_TR_POLICY_ID, 0),
            "tpe",
        )
        self.assertEqual(
            [
                inner_policy.rewarm_allowed(inner_policy.LOCAL_TR_POLICY_ID, i)
                for i in range(4)
            ],
            [False, False, False, False],
        )
        self.assertTrue(inner_policy.is_regime_policy(inner_policy.LOCAL_TR_POLICY_ID))
        self.assertEqual(
            inner_policy.expected_bout_trials(inner_policy.LOCAL_TR_POLICY_ID, 0, 10),
            8,
        )
        self.assertTrue(
            inner_policy.is_regime_policy(inner_policy.LOCAL_TR_HEBO_POLICY_ID)
        )
        self.assertEqual(
            [
                inner_policy.expected_bout_trials(
                    inner_policy.LOCAL_TR_HEBO_POLICY_ID, i, 10
                )
                for i in range(4)
            ],
            [8, 10, 10, 10],
        )
        self.assertEqual(
            [
                inner_policy.rewarm_allowed(inner_policy.LOCAL_TR_HEBO_POLICY_ID, i)
                for i in range(4)
            ],
            [False, False, False, False],
        )

    def test_deep_requires_movable_continuous_only_for_spsa(self):
        self.assertTrue(inner_policy.deep_requires_movable_continuous(POLICY))
        self.assertTrue(
            inner_policy.deep_requires_movable_continuous(
                inner_policy.LOCAL_TR_POLICY_ID
            )
        )
        # HEBO proposes over the whole space; no continuous dim required.
        self.assertFalse(
            inner_policy.deep_requires_movable_continuous(
                inner_policy.LOCAL_TR_HEBO_POLICY_ID
            )
        )
        self.assertFalse(
            inner_policy.deep_requires_movable_continuous(
                inner_policy.SELF_RANK_HEBO_POLICY_ID
            )
        )
        self.assertFalse(inner_policy.deep_requires_movable_continuous("legacy"))
        self.assertFalse(
            inner_policy.deep_requires_movable_continuous(
                inner_policy.MIXUP_TURBO_POLICY_ID
            )
        )
        self.assertEqual(
            inner_policy.numeric_required_from_bout_index(
                inner_policy.MIXUP_TURBO_POLICY_ID
            ),
            1,
        )
        self.assertFalse(
            inner_policy.deep_requires_movable_continuous(
                inner_policy.HEBO_TURBO_POLICY_ID
            )
        )
        self.assertEqual(
            inner_policy.numeric_required_from_bout_index(
                inner_policy.HEBO_TURBO_POLICY_ID
            ),
            1,
        )
        self.assertFalse(
            inner_policy.deep_requires_movable_continuous(
                inner_policy.HEBO_HEBO_POLICY_ID
            )
        )
        self.assertIsNone(
            inner_policy.numeric_required_from_bout_index(
                inner_policy.HEBO_HEBO_POLICY_ID
            )
        )

    def test_has_movable_continuous(self):
        self.assertTrue(inner_policy.has_movable_continuous(FLOAT3))
        self.assertFalse(inner_policy.has_movable_continuous(INT_ONLY))
        self.assertFalse(
            inner_policy.has_movable_continuous({"x": ("float", 1.0, 1.0)})
        )
        self.assertTrue(inner_policy.has_movable_numeric(INT_ONLY))
        self.assertFalse(
            inner_policy.has_movable_numeric(
                {"mode": ("categorical", ["a", "b"])}
            )
        )

    def test_load_movable_continuous_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, _ = _fixture(Path(tmp), space=INT_ONLY, base=INT_BASE)
            ledger = {"records": [{"run_id": "001"}, {"run_id": "404"}]}
            flags = inner_policy.load_movable_continuous_flags(tmp, ledger)
            self.assertEqual(flags, {"001": False})


class PhaseCActionRegimeTest(unittest.TestCase):
    def test_24_plus_10_plus_10_policy_methods(self):
        cases = (
            (inner_policy.MIXUP_TURBO_POLICY_ID, "mixup", "turbo"),
            (inner_policy.HEBO_TURBO_POLICY_ID, "hebo", "turbo"),
            (inner_policy.HEBO_HEBO_POLICY_ID, "hebo", "hebo"),
        )
        for policy_id, initial_method, later_method in cases:
            with self.subTest(policy_id=policy_id):
                with tempfile.TemporaryDirectory() as tmp:
                    candidate, report_path = _fixture(
                        Path(tmp),
                        space=FLOAT3,
                        base=BASE3,
                        inner_policy_id=policy_id,
                    )
                    self._assert_24_plus_10_plus_10(
                        candidate, report_path, initial_method, later_method
                    )

    def _assert_24_plus_10_plus_10(
        self,
        candidate: Path,
        report_path: Path,
        initial_method: str,
        later_method: str,
    ) -> None:
        report = json.loads(report_path.read_text())
        first = phase_c_action(report, candidate)
        self.assertEqual(
            (
                first["method"],
                first["bout_trials"],
                first["method_chain"],
            ),
            (initial_method, 24, [initial_method]),
        )

        report["phase_c"] = {
            "stages": [
                {
                    "method": initial_method,
                    "status": "ok",
                    "trials": [{"params": dict(BASE3), "score": 1.0}],
                }
            ]
        }
        _finalize_bout(report, candidate, best=1.0)
        second = phase_c_action(report, candidate)
        self.assertEqual(
            (
                second["method"],
                second["bout_trials"],
                second["bout_regime"],
            ),
            (later_method, 10, "CONTINUE"),
        )

        report["phase_c"]["stages"].append(
            {
                "method": later_method,
                "bout_index": 1,
                "status": "ok",
                "trials": [{"params": dict(BASE3), "score": 1.0}],
            }
        )
        _finalize_bout(report, candidate, best=1.0)
        third = phase_c_action(report, candidate)
        self.assertEqual(
            (
                third["method"],
                third["bout_trials"],
                third["bout_regime"],
            ),
            (later_method, 10, "DEEP"),
        )

    def test_first_bout_is_random_bo8(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(Path(tmp), space=FLOAT3, base=BASE3)
            report = json.loads(report_path.read_text())
            action = phase_c_action(report, candidate)
            self.assertEqual(
                (
                    action["action"],
                    action["method"],
                    action["sampler"],
                    action["bout_regime"],
                    action["bout_trials"],
                    action["method_chain"],
                    action["inner_policy"],
                ),
                ("run", "bo", "random", "FIRST", 8, ["bo"], POLICY),
            )

    def test_continue_bout_is_hebo(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(Path(tmp), space=FLOAT3, base=BASE3)
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "bo",
                        "status": "ok",
                        "trials": [{"params": dict(BASE3), "score": 0.9}],
                    }
                ]
            }
            _finalize_bout(report, candidate, best=0.9)
            action = phase_c_action(report, candidate)
            self.assertEqual(
                (
                    action["action"],
                    action["method"],
                    action["sampler"],
                    action["bout_regime"],
                    action["bout_trials"],
                ),
                ("run", "hebo", None, "CONTINUE", 10),
            )
            self.assertEqual(action["method_chain"], ["hebo"])

    def test_deep_bout_is_spsa(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(Path(tmp), space=FLOAT3, base=BASE3)
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "bo",
                        "status": "ok",
                        "trials": [{"params": dict(BASE3), "score": 0.9}],
                    },
                    {
                        "method": "hebo",
                        "bout_index": 1,
                        "status": "ok",
                        "trials": [{"params": dict(BASE3), "score": 0.8}],
                    },
                ]
            }
            _finalize_bout(report, candidate, best=0.8)
            action = phase_c_action(report, candidate)
            self.assertEqual(
                (
                    action["action"],
                    action["method"],
                    action["sampler"],
                    action["bout_regime"],
                    action["bout_trials"],
                ),
                ("run", "spsa", None, "DEEP", 10),
            )

    def test_deep_bout_without_movable_continuous_finalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(
                Path(tmp), space=INT_ONLY, base=INT_BASE
            )
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "bo",
                        "status": "ok",
                        "trials": [{"params": dict(INT_BASE), "score": 0.9}],
                    },
                    {
                        "method": "hebo",
                        "bout_index": 1,
                        "status": "ok",
                        "trials": [{"params": dict(INT_BASE), "score": 0.8}],
                    },
                ]
            }
            _finalize_bout(report, candidate, best=0.8)
            action = phase_c_action(report, candidate)
            self.assertEqual(action["action"], "finalize")
            self.assertEqual(
                action["reason"], "deep_bout_requires_movable_continuous"
            )
    def test_legacy_pin_keeps_grid_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(
                Path(tmp), space=INT_ONLY, base=INT_BASE, legacy=True
            )
            report = json.loads(report_path.read_text())
            action = phase_c_action(report, candidate)
            self.assertEqual((action["action"], action["method"]), ("run", "grid"))
            self.assertEqual(action["inner_policy"], "legacy")
            self.assertEqual(action["bout_trials"], 10)

    def test_local_tr_policy_first_bout_is_local_tr8(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(
                Path(tmp),
                space=FLOAT3,
                base=BASE3,
                inner_policy_id=inner_policy.LOCAL_TR_POLICY_ID,
            )
            report = json.loads(report_path.read_text())
            action = phase_c_action(report, candidate)
            self.assertEqual(
                (
                    action["action"],
                    action["method"],
                    action["sampler"],
                    action["bout_regime"],
                    action["bout_trials"],
                    action["method_chain"],
                    action["inner_policy"],
                ),
                (
                    "run",
                    "local_tr",
                    None,
                    "FIRST",
                    8,
                    ["local_tr"],
                    inner_policy.LOCAL_TR_POLICY_ID,
                ),
            )


    def test_local_tr_hebo_policy_deep_bout_is_hebo(self):
        for space, base in ((FLOAT3, BASE3), (INT_ONLY, INT_BASE)):
            with self.subTest(space=space), tempfile.TemporaryDirectory() as tmp:
                candidate, report_path = _fixture(
                    Path(tmp),
                    space=space,
                    base=base,
                    inner_policy_id=inner_policy.LOCAL_TR_HEBO_POLICY_ID,
                )
                report = json.loads(report_path.read_text())
                report["phase_c"] = {
                    "stages": [
                        {
                            "method": "local_tr",
                            "status": "ok",
                            "trials": [{"params": dict(base), "score": 0.9}],
                        },
                        {
                            "method": "hebo",
                            "bout_index": 1,
                            "status": "ok",
                            "trials": [{"params": dict(base), "score": 0.8}],
                        },
                    ]
                }
                _finalize_bout(report, candidate, best=0.8)
                action = phase_c_action(report, candidate)
                self.assertEqual(
                    (
                        action["action"],
                        action["method"],
                        action["bout_regime"],
                        action["bout_trials"],
                        action["method_chain"],
                    ),
                    ("run", "hebo", "DEEP", 10, ["hebo"]),
                )

    def test_selfrank_policy_first_bout_is_selfrank8(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(
                Path(tmp),
                space=FLOAT3,
                base=BASE3,
                inner_policy_id=inner_policy.SELF_RANK_HEBO_POLICY_ID,
            )
            action = phase_c_action(json.loads(report_path.read_text()), candidate)
            self.assertEqual(
                (
                    action["action"],
                    action["method"],
                    action["bout_regime"],
                    action["bout_trials"],
                    action["method_chain"],
                    action["inner_policy"],
                ),
                (
                    "run",
                    "selfrank",
                    "FIRST",
                    8,
                    ["selfrank"],
                    inner_policy.SELF_RANK_HEBO_POLICY_ID,
                ),
            )


class AdmissionRegimeTest(unittest.TestCase):
    def test_first_bout_admits_only_bo(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(Path(tmp), space=FLOAT3, base=BASE3)
            budget = deep_tune_time_budget(candidate, report_path, "bo")
            self.assertEqual(budget["bout_index"], 0)
            budget["_phase_c_lock_handle"].close()

        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(Path(tmp), space=FLOAT3, base=BASE3)
            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError, "outside deterministic chain"
            ):
                deep_tune_time_budget(candidate, report_path, "grid")

        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(Path(tmp), space=FLOAT3, base=BASE3)
            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError, "outside deterministic chain"
            ):
                deep_tune_time_budget(candidate, report_path, "local_tr")

        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(
                Path(tmp),
                space=FLOAT3,
                base=BASE3,
                inner_policy_id=inner_policy.LOCAL_TR_POLICY_ID,
            )
            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError, "outside deterministic chain"
            ):
                deep_tune_time_budget(candidate, report_path, "bo")
            budget = deep_tune_time_budget(candidate, report_path, "local_tr")
            try:
                self.assertEqual(budget["bout_index"], 0)
            finally:
                budget["_phase_c_lock_handle"].close()

    def test_deep_bout_admits_spsa_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(Path(tmp), space=FLOAT3, base=BASE3)
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "bo",
                        "status": "ok",
                        "trials": [{"params": dict(BASE3), "score": 0.9}],
                    },
                    {
                        "method": "hebo",
                        "bout_index": 1,
                        "status": "ok",
                        "trials": [{"params": dict(BASE3), "score": 0.8}],
                    },
                ]
            }
            _finalize_bout(report, candidate, best=0.8)
            report_path.write_text(json.dumps(report))
            # hebo already closed terminally inside bout 1, so re-requesting it
            # is refused before any bout-2 work can start.
            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError, "cannot be rerun"
            ):
                deep_tune_time_budget(candidate, report_path, "hebo")
            budget = deep_tune_time_budget(candidate, report_path, "spsa")
            try:
                self.assertEqual(budget["bout_index"], 2)
            finally:
                budget["_phase_c_lock_handle"].close()


class ValidateProposalsDeepGuardTest(unittest.TestCase):
    def _two_finalized_bouts(self, candidate, report_path):
        report = json.loads(report_path.read_text())
        report["phase_c"] = {
            "stages": [
                {
                    "method": "bo",
                    "status": "ok",
                    "trials": [{"params": dict(BASE3), "score": 0.9}],
                },
                {
                    "method": "hebo",
                    "bout_index": 1,
                    "status": "ok",
                    "trials": [{"params": dict(BASE3), "score": 0.8}],
                },
            ]
        }
        _finalize_bout(report, candidate, best=0.8)
        report_path.write_text(json.dumps(report))

    def test_continue_bout_rejects_proposals(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(Path(tmp), space=FLOAT3, base=BASE3)
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "bo",
                        "status": "ok",
                        "trials": [{"params": dict(BASE3), "score": 0.9}],
                    }
                ]
            }
            _finalize_bout(report, candidate, best=0.9)
            report_path.write_text(json.dumps(report))
            result = validate_proposals(
                candidate, report_path, [{"a": 0.7, "b": 0.5, "c": 0.2}]
            )
            self.assertFalse(result["ok"])
            self.assertEqual(
                [r["reason"] for r in result["rejected"]],
                ["hebo_bout_has_no_rewarm"],
            )

    def test_deep_bout_rejects_proposals(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(Path(tmp), space=FLOAT3, base=BASE3)
            self._two_finalized_bouts(candidate, report_path)
            result = validate_proposals(
                candidate, report_path, [{"a": 0.7, "b": 0.5, "c": 0.2}]
            )
            self.assertFalse(result["ok"])
            self.assertEqual(
                [r["reason"] for r in result["rejected"]],
                ["deep_bout_has_no_rewarm"],
            )


class SelectCandidateRegimeTest(unittest.TestCase):
    @staticmethod
    def _budget():
        return {
            "remaining": 500,
            "deep_tune": {
                "remaining": 200,
                "total_cap": 200,
                "per_candidate_cap": 40,
                "per_candidate": [],
            },
        }

    def test_first_bout_trial_cap_is_eight(self):
        ledger = {
            "records": [
                {
                    "run_id": "001",
                    "status": "keep",
                    "best_warm_score": 1.0,
                }
            ]
        }
        result = select_candidate(
            ledger,
            n_min=1,
            top_percentile=80,
            bout_trials=10,
            budget_allocation=self._budget(),
        )
        self.assertEqual(result["run_id"], "001")
        self.assertEqual(result["bout_regime"], "FIRST")
        self.assertEqual(result["budget_allocation"]["trial_cap"], 8)

    def test_deep_ineligible_candidate_is_not_selected(self):
        record = {
            "run_id": "001",
            "status": "keep",
            "best_warm_score": 1.0,
            "tune": True,
            "tuning_bouts": 2,
            "last_bout_improved": True,
            "final_best_score": 0.5,
        }
        ledger = {"records": [record]}
        result = select_candidate(
            ledger,
            n_min=1,
            top_percentile=80,
            bout_trials=10,
            budget_allocation=self._budget(),
            movable_continuous={"001": False},
        )
        self.assertIsNone(result["run_id"])
        result = select_candidate(
            ledger,
            n_min=1,
            top_percentile=80,
            bout_trials=10,
            budget_allocation=self._budget(),
            movable_continuous={"001": True},
        )
        self.assertEqual(result["run_id"], "001")
        self.assertEqual(result["bout_regime"], "DEEP")
        self.assertEqual(result["budget_allocation"]["trial_cap"], 10)


class BoRandomSamplerTest(unittest.TestCase):
    """FIRST-bout kernel: explicit RandomSampler, deferred inside the bout."""

    def _run_bo(self, tmp, *, legacy: bool, n_trials: int, deferred: int):
        candidate, report_path = _fixture(
            Path(tmp), space=FLOAT3, base=BASE3, legacy=legacy
        )
        report = json.loads(report_path.read_text())
        report["phase_a"]["deferred_configs"] = [
            {"params": {"a": 0.1, "b": 0.001, "c": -0.5}},
            {"params": {"a": 0.9, "b": 0.5, "c": 0.5}},
        ][:deferred]
        report_path.write_text(json.dumps(report))
        train_module = mock.Mock(
            SEARCH_SPACE={k: tuple(v) for k, v in FLOAT3.items()},
            BASE_PARAMS=dict(BASE3),
            make_model=object(),
        )
        scores = iter([0.9, 0.85, 0.8, 0.75, 0.7, 0.65])
        with mock.patch(
            "bo_search.load_candidate_modules",
            return_value=(train_module, object()),
        ), mock.patch(
            "bo_search.resolve_score_fn", return_value=object()
        ), mock.patch(
            "bo_search.resolve_preflight_fn", return_value=None
        ), mock.patch(
            "bo_search.timed_eval", side_effect=lambda *a, **k: next(scores)
        ) as eval_mock, mock.patch(
            "bo_search.write_json"
        ) as write_result, mock.patch.object(
            sys,
            "argv",
            [
                "bo_search.py",
                "--candidate-path",
                str(candidate),
                "--tune-report-json",
                str(report_path),
                "--n-trials",
                str(n_trials),
                "--sampler",
                "random",
            ],
        ):
            self.assertEqual(bo_main(), 0)
        return write_result.call_args.args[0], eval_mock

    def test_random_sampler_reports_zero_model_driven(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, eval_mock = self._run_bo(tmp, legacy=False, n_trials=4, deferred=0)
            self.assertEqual(result["sampler"], "random")
            self.assertEqual(result["model_driven_trials"], 0)
            self.assertEqual(result["random_fallback_trials"], 4)
            self.assertEqual(eval_mock.call_count, 4)

    def test_deferred_occupy_first_bout_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, eval_mock = self._run_bo(tmp, legacy=False, n_trials=4, deferred=2)
            # 2 deferred + 2 random draws = the bout's 4, never 4 + 2.
            self.assertEqual(eval_mock.call_count, 4)
            self.assertEqual(result["trials_completed"], 4)
        with tempfile.TemporaryDirectory() as tmp:
            result, eval_mock = self._run_bo(tmp, legacy=True, n_trials=4, deferred=2)
            # legacy: deferred are extra trials on top of the bout budget.
            self.assertEqual(eval_mock.call_count, 6)


class SpsaSearchTest(unittest.TestCase):
    def _seed_finalized_bouts(self, report, *, base, second_method):
        """Two finalized bouts so the spsa stage admits at bout 2."""
        report["phase_c"] = {
            "stages": [
                {
                    "method": "bo",
                    "status": "ok",
                    "trials": [{"params": dict(base), "score": 0.9}],
                },
                {
                    "method": second_method,
                    "bout_index": 1,
                    "status": "ok",
                    "trials": [{"params": dict(base), "score": 0.8}],
                },
            ]
        }
        _finalize_bout(report, None, best=0.8)

    def _run_spsa(
        self, tmp, *, n_evals=10, space=FLOAT3, base=BASE3, eval_fn=None
    ):
        candidate, report_path = _fixture(Path(tmp), space=space, base=base)
        report = json.loads(report_path.read_text())
        self._seed_finalized_bouts(report, base=base, second_method="hebo")
        report_path.write_text(json.dumps(report))
        train_module = mock.Mock(
            SEARCH_SPACE={k: tuple(v) for k, v in space.items()},
            BASE_PARAMS=dict(base),
            make_model=object(),
        )

        def fake_eval(evaluate, make_model, params, *args, **kwargs):
            return float(sum(float(v) for v in params.values()))

        with mock.patch(
            "spsa_search.load_candidate_modules",
            return_value=(train_module, object()),
        ), mock.patch(
            "spsa_search.resolve_score_fn", return_value=object()
        ), mock.patch(
            "spsa_search.resolve_preflight_fn", return_value=None
        ), mock.patch(
            "spsa_search.timed_eval",
            side_effect=eval_fn or fake_eval,
        ) as eval_mock, mock.patch(
            "spsa_search.write_json"
        ) as write_result, mock.patch.object(
            sys,
            "argv",
            [
                "spsa_search.py",
                "--candidate-path",
                str(candidate),
                "--tune-report-json",
                str(report_path),
                "--n-evals",
                str(n_evals),
            ],
        ):
            self.assertEqual(spsa_main(), 0)
        report = json.loads(report_path.read_text())
        return write_result.call_args.args[0], eval_mock, report

    def test_ten_evals_buy_five_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, eval_mock, report = self._run_spsa(tmp)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["trials_attempted"], 10)
            self.assertEqual(result["pairs_attempted"], 5)
            self.assertEqual(eval_mock.call_count, 10)
            stage = report["phase_c"]["stages"][-1]
            self.assertEqual(stage["method"], "spsa")
            self.assertEqual(stage["status"], "ok")
            self.assertEqual(stage["spsa_state"]["k"], 5)
            rows = stage["trials"]
            self.assertEqual(len(rows), 10)
            self.assertEqual(
                {(row["spsa_k"], row["spsa_side"]) for row in rows},
                {(k, side) for k in range(5) for side in ("plus", "minus")},
            )

    def test_interrupted_bout_resumes_the_exact_pair(self):
        class Boom(Exception):
            pass

        calls = {"plus": None}

        def flaky_eval(evaluate, make_model, params, *args, **kwargs):
            # Pair 0 completes; pair 1's plus leg dies loudly (infra flake:
            # not config-infeasible), leaving the stage running with its pair
            # state persisted.
            if len(calls["all"]) == 2:
                calls["plus"] = dict(params)
                raise Boom("infra death")
            calls["all"].append(dict(params))
            return float(sum(float(v) for v in params.values()))

        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(Path(tmp), space=FLOAT3, base=BASE3)
            report = json.loads(report_path.read_text())
            self._seed_finalized_bouts(report, base=BASE3, second_method="hebo")
            report_path.write_text(json.dumps(report))
            train_module = mock.Mock(
                SEARCH_SPACE={k: tuple(v) for k, v in FLOAT3.items()},
                BASE_PARAMS=dict(BASE3),
                make_model=object(),
            )

            def invoke(eval_fn, n_evals=10):
                calls["all"] = []
                with mock.patch(
                    "spsa_search.load_candidate_modules",
                    return_value=(train_module, object()),
                ), mock.patch(
                    "spsa_search.resolve_score_fn", return_value=object()
                ), mock.patch(
                    "spsa_search.resolve_preflight_fn", return_value=None
                ), mock.patch(
                    "spsa_search.timed_eval", side_effect=eval_fn
                ) as eval_mock, mock.patch(
                    "spsa_search.write_json"
                ), mock.patch.object(
                    sys,
                    "argv",
                    [
                        "spsa_search.py",
                        "--candidate-path",
                        str(candidate),
                        "--tune-report-json",
                        str(report_path),
                        "--n-evals",
                        str(n_evals),
                    ],
                ):
                    spsa_main()
                return eval_mock

            # Invocation 1 dies mid-pair-1 (its plus leg is a paid failure row).
            with self.assertRaises(Boom):
                invoke(flaky_eval)
            report = json.loads(report_path.read_text())
            stage = report["phase_c"]["stages"][-1]
            self.assertEqual(stage["status"], "running")
            self.assertEqual(stage["spsa_state"]["k"], 1)
            self.assertEqual(stage["spsa_state"]["pending_pair"]["k"], 1)

            # Invocation 2 resumes: the recorded failed plus leg is recovered,
            # never re-evaluated; the bout then runs to its 10-eval close.
            def normal_eval(evaluate, make_model, params, *args, **kwargs):
                calls["all"].append(dict(params))
                return float(sum(float(v) for v in params.values()))

            second = invoke(normal_eval)
            self.assertNotIn(calls["plus"], calls["all"])
            # pair-1's minus leg + pairs 2..4 = 7 fresh evaluations; the bout
            # closes at exactly 10 objective spends across both invocations.
            self.assertEqual(second.call_count, 7)
            report = json.loads(report_path.read_text())
            stage = report["phase_c"]["stages"][-1]
            self.assertEqual(stage["status"], "ok")
            self.assertEqual(stage["spsa_state"]["k"], 5)
            self.assertNotIn("pending_pair", stage["spsa_state"])

    def test_no_movable_continuous_rejects_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, eval_mock, report = self._run_spsa(
                tmp, space=INT_ONLY, base=INT_BASE
            )
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(eval_mock.call_count, 0)
            stage = report["phase_c"]["stages"][-1]
            self.assertEqual(stage["status"], "rejected")


class HeboSearchTest(unittest.TestCase):
    """CONTINUE kernel: prompt-v2 pool + injected MACE ranker, no real objective."""

    def _pool_receipt(self, offset: float) -> dict:
        return {
            "configs": [
                {
                    "a": 0.11 + offset + 0.01 * index,
                    "b": 0.002 * (index + 1),
                    "c": -0.4 + 0.05 * index,
                }
                for index in range(5)
            ],
            "order": [0, 1, 2, 3, 4],
            "rationale": "scripted pool",
        }

    def test_continue_bout_executes_ranked_pool_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(Path(tmp), space=FLOAT3, base=BASE3)
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "bo",
                        "status": "ok",
                        "trials": [
                            {
                                "params": {
                                    "a": 0.05 * index,
                                    "b": 0.001 * (index + 1),
                                    "c": 0.1 * index - 0.5,
                                },
                                "score": 1.1 + 0.01 * index,
                            }
                            for index in range(8)
                        ],
                    }
                ]
            }
            _finalize_bout(report, candidate, best=1.0)
            report_path.write_text(json.dumps(report))

            runner = FakeSessionRunner(
                [
                    {"receipt": self._pool_receipt(0.0)},
                    {"receipt": self._pool_receipt(0.2)},
                ]
            )
            rank_calls = []

            def rank_fn(*, search_space, history, pool, seed):
                rank_calls.append(pool)
                # Unique first-Pareto member at index 1.
                return [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

            scores = iter([0.42, 0.41])
            hebo_search._TEST_SESSION_RUNNER = runner
            hebo_search._TEST_RANK_FN = rank_fn
            try:
                with mock.patch(
                    "hebo_search.timed_eval",
                    side_effect=lambda *a, **k: next(scores),
                ) as eval_mock, mock.patch(
                    "hebo_search.write_json"
                ) as write_result, mock.patch.object(
                    sys,
                    "argv",
                    [
                        "hebo_search.py",
                        "--candidate-path",
                        str(candidate),
                        "--tune-report-json",
                        str(report_path),
                        "--n-evals",
                        "2",
                    ],
                ):
                    self.assertEqual(hebo_main(), 0)
            finally:
                hebo_search._TEST_SESSION_RUNNER = None
                hebo_search._TEST_RANK_FN = None

            receipt = write_result.call_args.args[0]
            self.assertEqual(receipt["method"], "hebo")
            self.assertEqual(receipt["status"], "ok")
            self.assertEqual(receipt["trials_completed"], 2)
            self.assertEqual(eval_mock.call_count, 2)
            self.assertEqual(len(rank_calls), 2)
            report = json.loads(report_path.read_text())
            stage = report["phase_c"]["stages"][-1]
            self.assertEqual(stage["method"], "hebo")
            self.assertEqual(stage["status"], "ok")
            self.assertEqual(len(stage["trials"]), 2)
            self.assertEqual(stage["trials"][0]["params"]["a"], 0.12)
            extras = runner.calls[0][1].extra
            self.assertIn("~0.005", extras["history"])
            self.assertIn("genuinely different regions", extras["protocol"])

    def test_hebo24_first_bout_gets_baseline_target_and_deferred_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "autoresearch-baseline" / "r1"
            candidate, report_path = _fixture(
                run_dir,
                space=FLOAT3,
                base=BASE3,
                inner_policy_id=inner_policy.HEBO_TURBO_POLICY_ID,
            )
            (run_dir / "ledger.json").write_text(
                json.dumps(
                    {
                        "items": {
                            "task_baseline": {
                                "kind": "observed_metric",
                                "metric": "val_bpb",
                                "value": 4.0,
                                "direction": "minimize",
                            }
                        }
                    }
                )
            )
            report = json.loads(report_path.read_text())
            report["phase_a"]["deferred_configs"] = [
                {"params": {"a": 0.7, "b": 0.02, "c": 0.2}}
            ]
            report_path.write_text(json.dumps(report))

            runner = FakeSessionRunner(
                [{"receipt": self._pool_receipt(0.0)}]
            )
            hebo_search._TEST_SESSION_RUNNER = runner
            try:
                with mock.patch(
                    "hebo_search.timed_preflight", return_value={"status": "ok"}
                ), mock.patch(
                    "hebo_search.timed_eval", side_effect=[0.9, 0.8]
                ) as eval_mock, mock.patch(
                    "hebo_search.write_json"
                ) as write_result, mock.patch.object(
                    sys,
                    "argv",
                    [
                        "hebo_search.py",
                        "--candidate-path",
                        str(candidate),
                        "--tune-report-json",
                        str(report_path),
                        "--n-evals",
                        "2",
                    ],
                ):
                    self.assertEqual(hebo_main(), 0)
            finally:
                hebo_search._TEST_SESSION_RUNNER = None

            receipt = write_result.call_args.args[0]
            self.assertEqual(receipt["method"], "hebo")
            self.assertEqual(receipt["deferred_evaluated"], 1)
            self.assertEqual(eval_mock.call_count, 2)
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][-1]
            self.assertEqual(
                [row["source"] for row in stage["trials"]],
                ["deferred", "pool_hebo_mace"],
            )
            extras = runner.calls[0][1].extra
            self.assertIn("7.5% relative", extras["task"])
            self.assertIn("task_baseline.value: 4.0", extras["items"])
            self.assertIn("required_relative_improvement: 0.075", extras["items"])
            self.assertIn("required_target_score: 3.7", extras["items"])
            self.assertIn("score: 0.9", extras["incumbent"])

    def test_first_selfrank_consumes_deferred_inside_eight_slot_bout(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(
                Path(tmp),
                space=FLOAT3,
                base=BASE3,
                inner_policy_id=inner_policy.SELF_RANK_HEBO_POLICY_ID,
            )
            report = json.loads(report_path.read_text())
            report["phase_a"]["deferred_configs"] = [
                {"params": {"a": 0.7, "b": 0.02, "c": 0.2}}
            ]
            report_path.write_text(json.dumps(report))
            runner = FakeSessionRunner(
                [{"receipt": self._pool_receipt(0.0)}]
            )
            old_method, old_arm = hebo_search.METHOD, hebo_search.ARM
            hebo_search.METHOD = "selfrank"
            hebo_search.ARM = SELF_RANK_ARM
            hebo_search._TEST_SESSION_RUNNER = runner
            try:
                with mock.patch(
                    "hebo_search.timed_eval", side_effect=[0.4, 0.3]
                ) as eval_mock, mock.patch(
                    "hebo_search.write_json"
                ) as write_result, mock.patch.object(
                    sys,
                    "argv",
                    [
                        "selfrank_search.py",
                        "--candidate-path",
                        str(candidate),
                        "--tune-report-json",
                        str(report_path),
                        "--n-evals",
                        "2",
                    ],
                ):
                    self.assertEqual(hebo_main(), 0)
            finally:
                hebo_search._TEST_SESSION_RUNNER = None
                hebo_search.METHOD, hebo_search.ARM = old_method, old_arm

            receipt = write_result.call_args.args[0]
            self.assertEqual(receipt["method"], "selfrank")
            self.assertEqual(receipt["deferred_evaluated"], 1)
            self.assertEqual(eval_mock.call_count, 2)
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][-1]
            self.assertEqual(stage["method"], "selfrank")
            self.assertEqual(
                [row["source"] for row in stage["trials"]],
                ["deferred", "pool_self_rank1"],
            )
            # The fresh proposer session sees the deferred factual outcome.
            self.assertIn("0.4", runner.calls[0][1].extra["incumbent"])
            # A FIRST bout must not be described to the proposer as a
            # continuation: this kernel now serves all three regimes.
            candidate_block = runner.calls[0][1].extra["candidate"]
            self.assertIn("regime: first", candidate_block)
            self.assertIn("stratum: first", candidate_block)

    def test_first_mixup_consumes_deferred_inside_total_slot_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(
                Path(tmp),
                space=FLOAT3,
                base=BASE3,
                inner_policy_id=inner_policy.MIXUP_TURBO_POLICY_ID,
            )
            report = json.loads(report_path.read_text())
            report["phase_a"]["deferred_configs"] = [
                {"params": {"a": 0.7, "b": 0.02, "c": 0.2}}
            ]
            report_path.write_text(json.dumps(report))

            def suggest(**_kwargs):
                return {
                    "suggestion": {"a": 0.3, "b": 0.03, "c": -0.2},
                    "mode": "quasi",
                    "quasi_consumed": 1,
                }

            old_method, old_arm = hebo_search.METHOD, hebo_search.ARM
            old_runner = hebo_search._TEST_SESSION_RUNNER
            old_suggest = hebo_search._TEST_SUGGEST_FN
            hebo_search.METHOD = "mixup"
            hebo_search.ARM = MIXUP_ARM
            hebo_search._TEST_SESSION_RUNNER = FakeSessionRunner([])
            hebo_search._TEST_SUGGEST_FN = suggest
            try:
                with mock.patch(
                    "hebo_search.timed_eval", side_effect=[0.4, 0.3]
                ) as eval_mock, mock.patch(
                    "hebo_search.write_json"
                ) as write_result, mock.patch.object(
                    sys,
                    "argv",
                    [
                        "mixup_search.py",
                        "--candidate-path",
                        str(candidate),
                        "--tune-report-json",
                        str(report_path),
                        "--n-evals",
                        "2",
                    ],
                ):
                    self.assertEqual(hebo_main(), 0)
            finally:
                hebo_search.METHOD, hebo_search.ARM = old_method, old_arm
                hebo_search._TEST_SESSION_RUNNER = old_runner
                hebo_search._TEST_SUGGEST_FN = old_suggest

            receipt = write_result.call_args.args[0]
            self.assertEqual(receipt["method"], "mixup")
            self.assertEqual(receipt["deferred_evaluated"], 1)
            self.assertEqual(eval_mock.call_count, 2)
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][-1]
            self.assertEqual(
                [row["source"] for row in stage["trials"]],
                ["deferred", "hebo_quasi"],
            )

    def test_turbo_second_segment_restores_first_segment_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(
                Path(tmp),
                space=FLOAT3,
                base=BASE3,
                inner_policy_id=inner_policy.MIXUP_TURBO_POLICY_ID,
            )
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "mixup",
                        "status": "ok",
                        "trials": [{"params": dict(BASE3), "score": 1.0}],
                    }
                ]
            }
            _finalize_bout(report, candidate, best=1.0)
            report_path.write_text(json.dumps(report))

            old_method, old_arm = hebo_search.METHOD, hebo_search.ARM
            old_extras = hebo_search._TEST_ARM_EXTRAS
            hebo_search.METHOD = "turbo"
            hebo_search.ARM = TURBO_ARM
            hebo_search._TEST_ARM_EXTRAS = {
                "turbo_fit_steps": 2,
                "turbo_n_candidates": 32,
            }
            try:
                with mock.patch(
                    "hebo_search.timed_eval", return_value=1.1
                ), mock.patch("hebo_search.write_json"), mock.patch.object(
                    sys,
                    "argv",
                    [
                        "turbo_search.py",
                        "--candidate-path",
                        str(candidate),
                        "--tune-report-json",
                        str(report_path),
                        "--n-evals",
                        "1",
                    ],
                ):
                    self.assertEqual(hebo_main(), 0)

                report = json.loads(report_path.read_text())
                first_turbo = report["phase_c"]["stages"][-1]
                first_state = first_turbo["turbo_state"]["turbo_final_state"]
                self.assertEqual(first_state["outcomes_seen"], 1)
                self.assertEqual(first_state["proposal_index"], 1)
                _finalize_bout(report, candidate, best=1.0)
                report_path.write_text(json.dumps(report))

                with mock.patch(
                    "hebo_search.timed_eval", return_value=1.2
                ), mock.patch("hebo_search.write_json"), mock.patch.object(
                    sys,
                    "argv",
                    [
                        "turbo_search.py",
                        "--candidate-path",
                        str(candidate),
                        "--tune-report-json",
                        str(report_path),
                        "--n-evals",
                        "1",
                    ],
                ):
                    self.assertEqual(hebo_main(), 0)
            finally:
                hebo_search.METHOD, hebo_search.ARM = old_method, old_arm
                hebo_search._TEST_ARM_EXTRAS = old_extras

            report = json.loads(report_path.read_text())
            second_turbo = report["phase_c"]["stages"][-1]
            second_state = second_turbo["turbo_state"]["turbo_final_state"]
            self.assertEqual(second_turbo["bout_index"], 2)
            self.assertEqual(second_state["outcomes_seen"], 2)
            self.assertEqual(second_state["proposal_index"], 2)
            self.assertEqual(
                second_state["policy_id"], "hotstart-turbo1-deep20-v1"
            )

    def test_interrupted_turbo_segment_recovers_consumed_proposal(self):
        class InfraDeath(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = _fixture(
                Path(tmp),
                space=FLOAT3,
                base=BASE3,
                inner_policy_id=inner_policy.MIXUP_TURBO_POLICY_ID,
            )
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "mixup",
                        "status": "ok",
                        "trials": [{"params": dict(BASE3), "score": 1.0}],
                    }
                ]
            }
            _finalize_bout(report, candidate, best=1.0)
            report_path.write_text(json.dumps(report))

            old_method, old_arm = hebo_search.METHOD, hebo_search.ARM
            old_extras = hebo_search._TEST_ARM_EXTRAS
            hebo_search.METHOD = "turbo"
            hebo_search.ARM = TURBO_ARM
            hebo_search._TEST_ARM_EXTRAS = {
                "turbo_fit_steps": 2,
                "turbo_n_candidates": 32,
            }
            argv = [
                "turbo_search.py",
                "--candidate-path",
                str(candidate),
                "--tune-report-json",
                str(report_path),
                "--n-evals",
                "2",
            ]
            try:
                with mock.patch(
                    "hebo_search.timed_eval", side_effect=InfraDeath("lost")
                ), mock.patch("hebo_search.write_json"), mock.patch.object(
                    sys, "argv", argv
                ):
                    with self.assertRaises(InfraDeath):
                        hebo_main()

                interrupted = json.loads(report_path.read_text())
                stage = interrupted["phase_c"]["stages"][-1]
                self.assertEqual(stage["status"], "running")
                self.assertIn(
                    "turbo_proposal_identity", stage["turbo_state"]
                )
                self.assertNotIn(
                    "turbo_final_state", stage["turbo_state"]
                )

                with mock.patch(
                    "hebo_search.timed_eval", return_value=1.2
                ) as eval_mock, mock.patch(
                    "hebo_search.write_json"
                ), mock.patch.object(sys, "argv", argv):
                    self.assertEqual(hebo_main(), 0)
                self.assertEqual(eval_mock.call_count, 1)
            finally:
                hebo_search.METHOD, hebo_search.ARM = old_method, old_arm
                hebo_search._TEST_ARM_EXTRAS = old_extras

            recovered = json.loads(report_path.read_text())
            state = recovered["phase_c"]["stages"][-1]["turbo_state"][
                "turbo_final_state"
            ]
            self.assertEqual(state["outcomes_seen"], 2)
            self.assertEqual(state["proposal_index"], 2)
            self.assertEqual(state["failure_counter"], 2)


class DriverJobLocalTrTest(unittest.TestCase):
    def test_local_tr_argv_uses_repo_root_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            task = repo / "tasks" / "toy"
            task.mkdir(parents=True)
            (task / "task.toml").write_text(
                '[env]\ntype = "uv"\nproject = "tasks/toy"\n'
            )
            run_dir = repo / "runs" / "toy" / "r1"
            candidate_dir = run_dir / "candidates" / "007"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "train.py").write_text("# candidate\n")
            (candidate_dir / "tune_report.json").write_text(
                json.dumps({"phase_a": {}})
            )
            ctx = InvocationContext(
                task="toy", tag="r1", run_dir=run_dir,
                invocation_id=3, run_id="007",
            )
            with mock.patch(
                "driver.jobs._phase_c_action",
                return_value={
                    "action": "run",
                    "method": "local_tr",
                    "bout_trials": 8,
                    "sampler": None,
                },
            ):
                argv, _, _ = build_driver_job(
                    "tuner-orchestrator",
                    ctx,
                    {
                        "kind": "phase_c",
                        "run_id": "007",
                        "method": "local_tr",
                        "trial_cap": 8,
                    },
                    repo_root=repo,
                )
            joined = " ".join(argv)
            self.assertIn("local_tr_search.py", joined)
            self.assertEqual(argv[argv.index("--n-evals") + 1], "8")
            self.assertEqual(argv[argv.index("--project") + 1], str(repo))
            self.assertNotIn("--directory", argv)


class LocalTrSearchTest(unittest.TestCase):
    """FIRST kernel of localtr8-hebo10-spsa10-v1: deferred inside the bout."""

    def _run_local_tr(self, tmp, *, n_evals: int, deferred: int):
        candidate, report_path = _fixture(
            Path(tmp),
            space=FLOAT3,
            base=BASE3,
            inner_policy_id=inner_policy.LOCAL_TR_POLICY_ID,
        )
        report = json.loads(report_path.read_text())
        report["phase_a"]["deferred_configs"] = [
            {"params": {"a": 0.1, "b": 0.001, "c": -0.5}},
            {"params": {"a": 0.9, "b": 0.5, "c": 0.5}},
        ][:deferred]
        report_path.write_text(json.dumps(report))
        scores = iter([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55])
        import local_tr_search

        with mock.patch(
            "local_tr_search.timed_eval",
            side_effect=lambda *a, **k: next(scores),
        ) as eval_mock, mock.patch(
            "local_tr_search.write_json"
        ) as write_result, mock.patch.object(
            sys,
            "argv",
            [
                "local_tr_search.py",
                "--candidate-path",
                str(candidate),
                "--tune-report-json",
                str(report_path),
                "--n-evals",
                str(n_evals),
                "--seed",
                "7",
            ],
        ):
            self.assertEqual(local_tr_search.main(), 0)
        return write_result.call_args.args[0], eval_mock, report_path

    def test_deferred_occupy_first_bout_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, eval_mock, report_path = self._run_local_tr(
                tmp, n_evals=4, deferred=2
            )
            self.assertEqual(result["method"], "local_tr")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(eval_mock.call_count, 4)
            self.assertEqual(result["trials_completed"], 4)
            self.assertEqual(result["deferred_evaluated"], 2)
            report = json.loads(report_path.read_text())
            stage = report["phase_c"]["stages"][-1]
            self.assertEqual(stage["method"], "local_tr")
            self.assertEqual(stage["trials"][0]["source"], "deferred")
            self.assertEqual(stage["trials"][2]["source"], "local_tr")


class DriverJobHeboTest(unittest.TestCase):
    def test_hebo_argv_uses_repo_root_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            task = repo / "tasks" / "toy"
            task.mkdir(parents=True)
            (task / "task.toml").write_text(
                '[env]\ntype = "uv"\nproject = "tasks/toy"\n'
            )
            run_dir = repo / "runs" / "toy" / "r1"
            candidate_dir = run_dir / "candidates" / "007"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "train.py").write_text("# candidate\n")
            (candidate_dir / "tune_report.json").write_text(
                json.dumps({"phase_a": {}})
            )
            ctx = InvocationContext(
                task="toy", tag="r1", run_dir=run_dir,
                invocation_id=3, run_id="007",
            )
            with mock.patch(
                "driver.jobs._phase_c_action",
                return_value={
                    "action": "run",
                    "method": "hebo",
                    "bout_trials": 10,
                    "sampler": None,
                },
            ):
                argv, _, _ = build_driver_job(
                    "tuner-orchestrator",
                    ctx,
                    {
                        "kind": "phase_c",
                        "run_id": "007",
                        "method": "hebo",
                        "trial_cap": 10,
                    },
                    repo_root=repo,
                )
            joined = " ".join(argv)
            self.assertIn("hebo_search.py", joined)
            self.assertEqual(argv[argv.index("--n-evals") + 1], "10")
            self.assertEqual(argv[argv.index("--project") + 1], str(repo))
            self.assertNotIn("--directory", argv)


class DriverJobSpsaTest(unittest.TestCase):
    def test_spsa_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            task = repo / "tasks" / "toy"
            task.mkdir(parents=True)
            (task / "task.toml").write_text(
                '[env]\ntype = "uv"\nproject = "tasks/toy"\n'
            )
            run_dir = repo / "runs" / "toy" / "r1"
            candidate_dir = run_dir / "candidates" / "007"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "train.py").write_text("# candidate\n")
            (candidate_dir / "tune_report.json").write_text(
                json.dumps({"phase_a": {}})
            )
            ctx = InvocationContext(
                task="toy", tag="r1", run_dir=run_dir,
                invocation_id=3, run_id="007",
            )
            with mock.patch(
                "driver.jobs._phase_c_action",
                return_value={
                    "action": "run",
                    "method": "spsa",
                    "bout_trials": 10,
                    "sampler": None,
                },
            ):
                argv, _, _ = build_driver_job(
                    "tuner-orchestrator",
                    ctx,
                    {
                        "kind": "phase_c",
                        "run_id": "007",
                        "method": "spsa",
                        "trial_cap": 10,
                    },
                    repo_root=repo,
                )
            self.assertIn("spsa_search.py", " ".join(argv))
            self.assertEqual(argv[argv.index("--n-evals") + 1], "10")


if __name__ == "__main__":
    unittest.main()
