from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

from failure_artifacts import record_failure, render_failure  # noqa: E402
from _common import (  # noqa: E402
    is_config_infeasible_error,
    is_finite_score,
    read_prior_infeasible_trials,
    read_prior_trials,
    timed_eval,
    write_tune_report,
)
from bo_search import (  # noqa: E402
    DeferredConfigError,
    INFEASIBLE_ATTR,
    _enqueue_unique_deferred,
    _inject_infeasible_trials,
    _inject_prior_trials,
    _infeasible_constraints,
    _is_infeasible,
    _set_feasibility,
    main as bo_main,
)
from cmaes_search import _encode_prior_seed, build_codec, main as cmaes_main  # noqa: E402
from tune_tools import (  # noqa: E402
    _candidate_execution_revision,
    finalizable_tuning_result,
    select_best,
    select_candidate,
    summarize,
)


TRACEBACK = """Traceback (most recent call last):
  File \"/tmp/run/candidates/007/train.py\", line 41, in make_model
    return build(depth)
  File \"/usr/lib/example.py\", line 9, in build
    raise ValueError(\"depth must be positive\")
ValueError: depth must be positive
"""


def _plain_make_model(params):
    return params


class _FakeTrial:
    def __init__(self, params=None):
        self.params = params or {}
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class _FakeStudy:
    def __init__(self, trials, *, enqueue_error=None, add_error=None):
        self.trials = trials
        self.enqueued = []
        self.added = []
        self.enqueue_error = enqueue_error
        self.add_error = add_error

    def enqueue_trial(self, params, *, skip_if_exists=False):
        if self.enqueue_error is not None:
            raise self.enqueue_error
        self.enqueued.append((params, skip_if_exists))

    def add_trial(self, trial):
        if self.add_error is not None:
            raise self.add_error
        self.added.append(trial)
        if isinstance(trial, dict) and isinstance(trial.get("params"), dict):
            self.trials.append(_FakeTrial(trial["params"]))


class FailureArtifactTests(unittest.TestCase):
    def test_bo_deferred_configs_skip_injected_priors_and_each_other(self) -> None:
        study = _FakeStudy([_FakeTrial({"depth": 3})])
        n_enqueued = _enqueue_unique_deferred(
            study,
            [{"depth": 3}, {"depth": 4}, {"depth": 4}],
            {"depth": ("int", 1, 5)},
            {"depth": object()},
        )

        self.assertEqual(n_enqueued, 1)
        self.assertEqual(study.enqueued, [({"depth": 4}, True)])

    def test_bo_deferred_backend_rejection_is_explicit_and_fatal(self) -> None:
        study = _FakeStudy([], enqueue_error=ValueError("backend refused"))

        with self.assertRaises(DeferredConfigError) as caught:
            _enqueue_unique_deferred(
                study,
                [{"depth": 4}],
                {"depth": ("int", 1, 5)},
                {"depth": object()},
            )

        self.assertEqual(caught.exception.rejections[0]["reason"], "backend_rejected")
        self.assertEqual(caught.exception.rejections[0]["error_type"], "ValueError")

    def test_bo_prior_backend_rejection_has_a_receipt(self) -> None:
        study = _FakeStudy([], add_error=ValueError("invalid prior"))
        injected, rejections = _inject_prior_trials(
            study,
            [{"params": {"depth": 4}, "score": 0.5}],
            {"depth": object()},
            lambda **kwargs: kwargs,
            {},
        )

        self.assertEqual(injected, 0)
        self.assertEqual(rejections[0]["reason"], "backend_rejected")
        self.assertEqual(rejections[0]["error"], "invalid prior")

    def test_cma_prior_encoding_fallback_has_a_receipt(self) -> None:
        search_space = {"mode": ("categorical", ["a", "b"])}
        _, lower, upper, x0_default, encode, _ = build_codec(
            search_space,
            {"mode": "a"},
        )

        x0, rejection = _encode_prior_seed(
            {"params": {"mode": "unknown"}},
            encode=encode,
            lower=lower,
            upper=upper,
            x0_default=x0_default,
        )

        self.assertEqual(x0, x0_default)
        self.assertEqual(rejection["reason"], "seed_encoding_failed")
        self.assertEqual(rejection["error_type"], "ValueError")

    def test_cma_codec_omits_fixed_coordinates_and_restores_them(self) -> None:
        search_space = {
            "fixed_float": ("float", 1.0, 1.0),
            "active_float": ("float", 0.0, 2.0),
            "fixed_int": ("int", 3, 3),
            "fixed_mode": ("categorical", ["only"]),
        }
        keys, lower, upper, _, encode, decode = build_codec(
            search_space,
            {
                "fixed_float": 1.0,
                "active_float": 0.5,
                "fixed_int": 3,
                "fixed_mode": "only",
            },
        )

        self.assertEqual(keys, ["active_float"])
        self.assertEqual(lower, [0.0])
        self.assertEqual(upper, [2.0])
        self.assertEqual(
            encode(
                {
                    "fixed_float": 1.0,
                    "active_float": 1.5,
                    "fixed_int": 3,
                    "fixed_mode": "only",
                }
            ).tolist(),
            [1.5],
        )
        self.assertEqual(
            decode([1.25]),
            {
                "fixed_float": 1.0,
                "active_float": 1.25,
                "fixed_int": 3,
                "fixed_mode": "only",
            },
        )

    def test_cma_codec_represents_all_fixed_space_without_bounds(self) -> None:
        keys, lower, upper, x0, encode, decode = build_codec(
            {
                "width": ("float", 1.0, 1.0),
                "depth": ("int", 4, 4),
                "mode": ("categorical", ["only"]),
            },
            {"width": 1.0, "depth": 4, "mode": "only"},
        )

        self.assertEqual((keys, lower, upper, x0), ([], [], [], []))
        self.assertEqual(encode({}).tolist(), [])
        self.assertEqual(
            decode([]),
            {"width": 1.0, "depth": 4, "mode": "only"},
        )

    def test_cma_all_fixed_space_closes_without_constructing_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = Path(tmp) / "train.py"
            report_path = Path(tmp) / "tune_report.json"
            fixed_params = {"width": 1.0, "depth": 4, "mode": "only"}
            candidate_path.write_text(
                "PARAM_SCHEMA = {\n"
                "    'width': 'float',\n"
                "    'depth': 'int',\n"
                "    'mode': ('categorical', ['only']),\n"
                "}\n"
                "SEARCH_SPACE = {\n"
                "    'width': ('float', 1.0, 1.0),\n"
                "    'depth': ('int', 4, 4),\n"
                "    'mode': ('categorical', ['only']),\n"
                "}\n"
                "BASE_PARAMS = {'width': 1.0, 'depth': 4, 'mode': 'only'}\n"
                "def make_model(params):\n"
                "    return params\n"
            )
            (candidate_path.parent / "prepare.py").write_text(
                "def evaluate_config(make_model, params):\n"
                "    return 0.5\n"
            )
            write_tune_report(
                report_path,
                {
                    "phase_a": {
                        "status": "ok",
                        "candidate_code_revision": _candidate_execution_revision(
                            candidate_path
                        ),
                        "search_space": {
                            "width": ["float", 1.0, 1.0],
                            "depth": ["int", 4, 4],
                            "mode": ["categorical", ["only"]],
                        },
                        "warm_start_configs": [
                            {"params": fixed_params, "score": 0.5}
                        ],
                        "best_warm_params": fixed_params,
                        "best_warm_score": 0.5,
                        "deferred_configs": [
                            {"params": fixed_params}
                        ],
                    },
                    "phase_c": {
                        "stages": [
                            {"method": "bo", "status": "rejected", "trials": []}
                        ]
                    },
                },
            )
            train_module = mock.Mock(
                BASE_PARAMS=fixed_params,
                SEARCH_SPACE={
                    "width": ("float", 1.0, 1.0),
                    "depth": ("int", 4, 4),
                    "mode": ("categorical", ["only"]),
                },
                make_model=object(),
            )
            fake_cma = mock.Mock()
            fake_cma.CMAEvolutionStrategy.side_effect = AssertionError(
                "optimizer must not be constructed"
            )

            with mock.patch.dict(sys.modules, {"cma": fake_cma}), mock.patch(
                "cmaes_search.load_candidate_modules",
                return_value=(train_module, object()),
            ), mock.patch(
                "cmaes_search.resolve_score_fn", return_value=object()
            ), mock.patch(
                "cmaes_search.resolve_preflight_fn", return_value=None
            ), mock.patch(
                "cmaes_search.write_json"
            ) as write_result, mock.patch.object(
                sys,
                "argv",
                [
                    "cmaes_search.py",
                    "--candidate-path",
                    str(candidate_path),
                    "--tune-report-json",
                    str(report_path),
                ],
            ):
                self.assertEqual(cmaes_main(), 0)

            fake_cma.CMAEvolutionStrategy.assert_not_called()
            result = write_result.call_args.args[0]
            self.assertEqual(result["status"], "no_search_needed")
            self.assertTrue(result["fixed_search_space"])
            self.assertEqual(result["best_params"], fixed_params)
            self.assertEqual(result["deferred_skipped_already_seen"], 1)
            report = json.loads(report_path.read_text())
            stage = report["phase_c"]["stages"][-1]
            self.assertEqual(stage["status"], "no_search_needed")
            self.assertEqual(stage["early_stop_reason"], "fixed_search_space")

    def test_cma_duplicate_proposal_reuses_score_without_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = Path(tmp) / "train.py"
            report_path = Path(tmp) / "tune_report.json"
            params = {"x": 1, "y": 1, "z": 1}
            candidate_path.write_text(
                "PARAM_SCHEMA = {'x': 'int', 'y': 'int', 'z': 'int'}\n"
                "SEARCH_SPACE = {\n"
                "    'x': ('int', 1, 2),\n"
                "    'y': ('int', 1, 2),\n"
                "    'z': ('int', 1, 2),\n"
                "}\n"
                "BASE_PARAMS = {'x': 1, 'y': 1, 'z': 1}\n"
                "def make_model(params):\n"
                "    return params\n"
            )
            (candidate_path.parent / "prepare.py").write_text(
                "def evaluate_config(make_model, params):\n"
                "    return 0.5\n"
            )
            write_tune_report(
                report_path,
                {
                    "phase_a": {
                        "status": "ok",
                        "candidate_code_revision": _candidate_execution_revision(
                            candidate_path
                        ),
                        "search_space": {
                            "x": ["int", 1, 2],
                            "y": ["int", 1, 2],
                            "z": ["int", 1, 2],
                        },
                        "warm_start_configs": [
                            {"params": params, "score": 0.5}
                        ],
                        "best_warm_params": params,
                        "best_warm_score": 0.5,
                        "deferred_configs": [],
                    },
                    "phase_c": {
                        "stages": [
                            {"method": "bo", "status": "rejected", "trials": []}
                        ]
                    },
                },
            )
            strategy = mock.Mock()
            strategy.stop.return_value = False
            strategy.ask.return_value = [[1.0, 1.0, 1.0]]
            fake_cma = mock.Mock()
            fake_cma.CMAEvolutionStrategy.return_value = strategy

            with mock.patch.dict(sys.modules, {"cma": fake_cma}), mock.patch(
                "cmaes_search.timed_eval"
            ) as timed_eval_mock, mock.patch(
                "cmaes_search.write_json"
            ) as write_result, mock.patch.object(
                sys,
                "argv",
                [
                    "cmaes_search.py",
                    "--candidate-path",
                    str(candidate_path),
                    "--tune-report-json",
                    str(report_path),
                    "--max-evals",
                    "1",
                    "--popsize",
                    "1",
                ],
            ):
                self.assertEqual(cmaes_main(), 0)

            timed_eval_mock.assert_not_called()
            result = write_result.call_args.args[0]
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["duplicates_skipped"], 1)
            self.assertEqual(result["duplicate_scores_reused"], 1)
            self.assertEqual(result["best_score"], 0.5)

    def test_cma_rejected_proposal_leaves_full_objective_allowance(self) -> None:
        """A preflight-rejected re-warm proposal must not displace a search
        trial: preflight reserves no budget slot and never reaches score_fn, so
        `--max-evals` objective attempts must still be available after it."""
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = Path(tmp) / "train.py"
            report_path = Path(tmp) / "tune_report.json"
            candidate_path.write_text(
                "PARAM_SCHEMA = {'x': 'int', 'y': 'int', 'z': 'int'}\n"
                "SEARCH_SPACE = {\n"
                "    'x': ('int', 1, 4),\n"
                "    'y': ('int', 1, 4),\n"
                "    'z': ('int', 1, 4),\n"
                "}\n"
                "BASE_PARAMS = {'x': 1, 'y': 1, 'z': 1}\n"
                "def make_model(params):\n"
                "    return params\n"
            )
            (candidate_path.parent / "prepare.py").write_text(
                "def evaluate_config(make_model, params):\n"
                "    return 0.5\n"
                "def preflight_config(make_model, params):\n"
                "    return {'status': 'ok'}\n"
            )
            write_tune_report(
                report_path,
                {
                    "phase_a": {
                        "status": "ok",
                        "candidate_code_revision": _candidate_execution_revision(
                            candidate_path
                        ),
                        "search_space": {
                            "x": ["int", 1, 4],
                            "y": ["int", 1, 4],
                            "z": ["int", 1, 4],
                        },
                        "warm_start_configs": [
                            {"params": {"x": 1, "y": 1, "z": 1}, "score": 0.5}
                        ],
                        "best_warm_params": {"x": 1, "y": 1, "z": 1},
                        "best_warm_score": 0.5,
                        "deferred_configs": [],
                    },
                    # One admitted re-warm proposal, which preflight will reject.
                    "phase_c": {
                        "stages": [
                            {"method": "bo", "status": "rejected", "trials": []}
                        ],
                        "pending_proposals": [{"x": 4, "y": 4, "z": 4}],
                        "pending_proposals_bout_index": 0,
                    },
                },
            )

            rejected = {"x": 4, "y": 4, "z": 4}

            def preflight(params, *args, **kwargs):
                if {k: params[k] for k in rejected} == rejected:
                    raise ValueError("proposal is infeasible")
                return {"status": "ok"}

            strategy = mock.Mock()
            strategy.stop.return_value = False
            # Distinct fresh points, so neither is skipped as a duplicate of the
            # warm incumbent.
            strategy.ask.side_effect = [[[2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0]]]
            fake_cma = mock.Mock()
            fake_cma.CMAEvolutionStrategy.return_value = strategy

            with mock.patch.dict(sys.modules, {"cma": fake_cma}), mock.patch(
                "cmaes_search.resolve_preflight_fn", return_value=object()
            ), mock.patch(
                "cmaes_search.clamp_search_space_to_preflight",
                side_effect=lambda space, *a, **k: space,
            ), mock.patch(
                "cmaes_search.timed_preflight", side_effect=preflight
            ), mock.patch(
                "cmaes_search.timed_eval", return_value=0.25
            ) as timed_eval_mock, mock.patch(
                "cmaes_search.write_json"
            ) as write_result, mock.patch.object(
                sys,
                "argv",
                [
                    "cmaes_search.py",
                    "--candidate-path",
                    str(candidate_path),
                    "--tune-report-json",
                    str(report_path),
                    "--max-evals",
                    "2",
                    "--popsize",
                    "1",
                ],
            ):
                self.assertEqual(cmaes_main(), 0)

            result = write_result.call_args.args[0]
            self.assertEqual(result["preflight_rejections"], 1)
            # The rejection is recorded as feasibility evidence but charged no
            # objective slot, so both --max-evals attempts remain spendable.
            self.assertEqual(timed_eval_mock.call_count, 2)
            self.assertEqual(result["trials_attempted"], 2)

    def test_bo_duplicate_proposal_reuses_score_without_evaluation(self) -> None:
        class Trial:
            def __init__(
                self,
                params,
                *,
                value=None,
                user_attrs=None,
                system_attrs=None,
            ):
                self.params = dict(params)
                self.value = value
                self.user_attrs = dict(user_attrs or {})
                self.system_attrs = dict(system_attrs or {})

            def set_user_attr(self, key, value):
                self.user_attrs[key] = value

            def suggest_int(self, key, low, high):
                return self.params[key]

            def suggest_float(self, key, low, high, *, log=False):
                return self.params[key]

            def suggest_categorical(self, key, options):
                return self.params[key]

        class Study:
            def __init__(self, proposal):
                self.proposal = proposal
                self.trials = []
                self.stopped = False

            def add_trial(self, trial):
                self.trials.append(trial)

            def enqueue_trial(self, params, *, skip_if_exists=False):
                raise AssertionError("fixture has no deferred configs")

            def stop(self):
                self.stopped = True

            def optimize(
                self,
                objective,
                *,
                n_trials,
                timeout,
                show_progress_bar,
                callbacks,
                catch,
            ):
                for _ in range(n_trials):
                    trial = Trial(self.proposal)
                    trial.value = objective(trial)
                    self.trials.append(trial)
                    for callback in callbacks:
                        callback(self, trial)
                    if self.stopped:
                        break

        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = Path(tmp) / "train.py"
            report_path = Path(tmp) / "tune_report.json"
            params = {"x": 1, "y": 1, "z": 1}
            candidate_path.write_text(
                "PARAM_SCHEMA = {'x': 'int', 'y': 'int', 'z': 'int'}\n"
                "SEARCH_SPACE = {\n"
                "    'x': ('int', 1, 2),\n"
                "    'y': ('int', 1, 2),\n"
                "    'z': ('int', 1, 2),\n"
                "}\n"
                "BASE_PARAMS = {'x': 1, 'y': 1, 'z': 1}\n"
                "def make_model(params):\n"
                "    return params\n"
            )
            (candidate_path.parent / "prepare.py").write_text(
                "def evaluate_config(make_model, params):\n"
                "    return 0.5\n"
            )
            write_tune_report(
                report_path,
                {
                    "phase_a": {
                        "status": "ok",
                        "candidate_code_revision": _candidate_execution_revision(
                            candidate_path
                        ),
                        "search_space": {
                            "x": ["int", 1, 2],
                            "y": ["int", 1, 2],
                            "z": ["int", 1, 2],
                        },
                        "warm_start_configs": [
                            {"params": params, "score": 0.5}
                        ],
                        "best_warm_params": params,
                        "best_warm_score": 0.5,
                        "deferred_configs": [],
                    }
                },
            )
            study = Study(params)

            def create_trial(**kwargs):
                return Trial(
                    kwargs["params"],
                    value=kwargs.get("value"),
                    user_attrs=kwargs.get("user_attrs"),
                    system_attrs=kwargs.get("system_attrs"),
                )

            distribution = lambda *args, **kwargs: object()
            fake_optuna = types.SimpleNamespace(
                samplers=types.SimpleNamespace(TPESampler=distribution),
                distributions=types.SimpleNamespace(
                    FloatDistribution=distribution,
                    IntDistribution=distribution,
                    CategoricalDistribution=distribution,
                ),
                trial=types.SimpleNamespace(create_trial=create_trial),
                logging=types.SimpleNamespace(
                    WARNING=30,
                    set_verbosity=lambda level: None,
                ),
                create_study=lambda **kwargs: study,
            )

            with mock.patch.dict(
                sys.modules,
                {"optuna": fake_optuna},
            ), mock.patch(
                "bo_search.timed_eval"
            ) as timed_eval_mock, mock.patch(
                "bo_search.write_json"
            ) as write_result, mock.patch.object(
                sys,
                "argv",
                [
                    "bo_search.py",
                    "--candidate-path",
                    str(candidate_path),
                    "--tune-report-json",
                    str(report_path),
                    "--n-trials",
                    "1",
                ],
            ):
                self.assertEqual(bo_main(), 0)

            timed_eval_mock.assert_not_called()
            result = write_result.call_args.args[0]
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["duplicates_skipped"], 1)
            self.assertEqual(result["best_score"], 0.5)
            finalized = finalizable_tuning_result(
                json.loads(report_path.read_text())
            )
            self.assertEqual(finalized["best_score"], 0.5)

    def test_bo_preflight_rejection_is_an_optuna_constraint(self) -> None:
        trial = _FakeTrial()
        self.assertEqual(_infeasible_constraints(trial), (0.0,))
        self.assertFalse(_is_infeasible(trial))

        _set_feasibility(trial, feasible=False)
        self.assertEqual(_infeasible_constraints(trial), (1.0,))
        self.assertTrue(_is_infeasible(trial))

        _set_feasibility(trial, feasible=True)
        self.assertEqual(_infeasible_constraints(trial), (0.0,))
        self.assertFalse(_is_infeasible(trial))

    def test_same_failure_is_stable_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "tune_report.json"
            report_path.write_text("{}")
            kwargs = {
                "report_path": report_path,
                "candidate_path": Path(tmp) / "train.py",
                "phase": "phase_a",
                "method": "warmstart",
                "params": {"depth": 0},
                "error": ValueError("depth must be positive"),
                "traceback_text": TRACEBACK,
            }

            first = record_failure(**kwargs)
            artifact = report_path.parent / first["failure_ref"]["artifact"]
            original_bytes = artifact.read_bytes()
            second = record_failure(**kwargs)

            self.assertEqual(first, second)
            self.assertEqual(artifact.read_bytes(), original_bytes)
            self.assertEqual(len(list((report_path.parent / "_failures").glob("*.json"))), 1)
            self.assertRegex(first["failure_ref"]["failure_id"], r"^fail-[0-9a-f]{16}$")
            self.assertEqual(first["failure_ref"]["schema_version"], 2)
            self.assertRegex(first["failure_ref"]["sha256"], r"^sha256:[0-9a-f]{64}$")
            self.assertNotIn("content_sha256", first["failure_receipt"])

    def test_config_infeasible_error_classification(self) -> None:
        self.assertTrue(is_config_infeasible_error(TimeoutError("timed out")))
        self.assertTrue(
            is_config_infeasible_error(RuntimeError("CUDA out of memory. Tried to allocate"))
        )
        self.assertFalse(is_config_infeasible_error(ValueError("depth must be positive")))
        self.assertFalse(
            is_config_infeasible_error(RuntimeError("subprocess exited with code 2"))
        )

    def test_prior_infeasible_trials_reader_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "tune_report.json"
            write_tune_report(
                report_path,
                {
                    "phase_a": {"warm_start_configs": [{"params": {"x": 1}, "score": 1.0}]},
                    "phase_c": {
                        "stages": [
                            {
                                "method": "bo",
                                "trials": [
                                    {"params": {"x": 2}, "score": 0.9},
                                    {"params": {"x": 3}, "score": None, "status": "failed",
                                     "config_infeasible": True},
                                    {"params": {"x": 4}, "score": None, "status": "failed"},
                                    {"params": {"x": 5}, "score": None,
                                     "status": "preflight_rejected"},
                                ],
                            }
                        ]
                    },
                },
            )
            infeasible = read_prior_infeasible_trials(report_path)
            self.assertEqual([t["params"]["x"] for t in infeasible], [3, 5])
            # Scored priors still exclude every scoreless trial.
            self.assertEqual([t["params"]["x"] for t in read_prior_trials(report_path)], [1, 2])

    def test_restart_reinjects_crashes_as_constrained_trials(self) -> None:
        # The reviewer's repro: one success + one crash in the report; a fresh
        # study must see BOTH — the crash as a constrained-infeasible point.
        study = _FakeStudy([])
        dists = {"x": object()}
        n_ok, _ = _inject_prior_trials(
            study, [{"params": {"x": 1}, "score": 0.5}], dists, lambda **kw: kw, {}
        )
        n_bad, rejections = _inject_infeasible_trials(
            study,
            [{"params": {"x": 2}, "score": None, "status": "failed",
              "config_infeasible": True}],
            dists,
            lambda **kw: kw,
            1.0,
        )
        self.assertEqual((n_ok, n_bad), (1, 1))
        self.assertEqual(rejections, [])
        crash_trial = study.added[1]
        self.assertEqual(crash_trial["value"], 1.0)  # penalty value, not a score
        self.assertEqual(crash_trial["user_attrs"], {INFEASIBLE_ATTR: [1.0]})
        self.assertEqual(crash_trial["system_attrs"], {"constraints": (1.0,)})

    def test_restart_does_not_enqueue_a_known_infeasible_deferred_config(self) -> None:
        study = _FakeStudy([])
        dists = {"x": object()}
        n_bad, rejections = _inject_infeasible_trials(
            study,
            [
                {
                    "params": {"x": 2},
                    "score": None,
                    "status": "failed",
                    "config_infeasible": True,
                }
            ],
            dists,
            lambda **kwargs: kwargs,
            1.0,
        )
        n_enqueued = _enqueue_unique_deferred(
            study,
            [{"x": 2}],
            {"x": ("int", 1, 3)},
            dists,
        )

        self.assertEqual(n_bad, 1)
        self.assertEqual(rejections, [])
        self.assertEqual(n_enqueued, 0)
        self.assertEqual(study.enqueued, [])

    def test_infeasible_injection_rejects_param_mismatch(self) -> None:
        study = _FakeStudy([])
        n_bad, rejections = _inject_infeasible_trials(
            study,
            [{"params": {"wrong_key": 1}, "score": None, "status": "failed",
              "config_infeasible": True}],
            {"x": object()},
            lambda **kw: kw,
            1.0,
        )
        self.assertEqual(n_bad, 0)
        self.assertEqual(rejections[0]["reason"], "parameter_set_mismatch")

    def test_trial_fields_are_compact_and_full_traceback_is_retrievable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "tune_report.json"
            report_path.write_text("{}")
            failure = record_failure(
                report_path=report_path,
                candidate_path=Path("/tmp/run/candidates/007/train.py"),
                phase="phase_a",
                method="warmstart",
                params={"depth": 0},
                error=ValueError("depth must be positive"),
                traceback_text=TRACEBACK,
            )

            self.assertNotIn("error_traceback", failure)
            self.assertEqual(failure["failure_receipt"]["exception"], "ValueError: depth must be positive")
            self.assertEqual(
                [frame["traceback_line"] for frame in failure["failure_receipt"]["frames"]],
                [2, 4],
            )
            self.assertEqual(failure["failure_receipt"]["retained_traceback_lines"], 5)
            self.assertEqual(failure["failure_receipt"]["omitted_traceback_lines"], 1)
            self.assertEqual(
                render_failure(report_path, failure["failure_ref"]["failure_id"], view="full"),
                TRACEBACK,
            )

    def test_non_finite_scores_are_not_successful_trials(self) -> None:
        report = {
            "phase_a": {
                "warm_start_configs": [
                    {"params": {"depth": 1}, "score": float("inf")},
                    {"params": {"depth": 2}, "score": float("nan")},
                    {"params": {"depth": 3}, "score": 0.4},
                ],
                "best_warm_score": float("inf"),
                "trials_attempted": 3,
            },
            "phase_c": {
                "stages": [
                    {
                        "method": "bo",
                        "trials": [
                            {"params": {"depth": 4}, "score": float("-inf")},
                        ],
                    }
                ]
            },
        }

        self.assertFalse(is_finite_score(float("inf")))
        self.assertFalse(is_finite_score(float("nan")))
        self.assertEqual(select_best(report)["best_score"], 0.4)
        self.assertIsNone(summarize(report)["best_warm_score"])
        self.assertEqual(summarize(report)["trials_completed"], 1)
        self.assertEqual(summarize(report)["trials_attempted"], 4)
        self.assertIsNone(
            select_candidate(
                {
                    "records": [
                        {
                            "run_id": "000",
                            "status": "keep",
                            "best_warm_score": float("inf"),
                            "tune": False,
                        }
                    ]
                },
                n_min=1,
                top_percentile=0,
            )["run_id"]
        )

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "tune_report.json"
            report_path.write_text(json.dumps(report))
            self.assertEqual(
                read_prior_trials(report_path),
                [{"params": {"depth": 3}, "score": 0.4}],
            )

    def test_inherited_control_is_counted_but_cannot_become_the_best(self) -> None:
        report = {
            "phase_a": {
                "status": "ok",
                "warm_start_configs": [
                    {
                        "params": {"depth": 1},
                        "score": 0.1,
                        "role": "inherited_control",
                    },
                    {"params": {"depth": 2}, "score": 0.4},
                ],
                "best_warm_params": {"depth": 2},
                "best_warm_score": 0.4,
                "trials_attempted": 2,
            },
            "phase_c": {
                "stages": [
                    {
                        "method": "bo",
                        "trials": [
                            {"params": {"depth": 1}, "score": 0.05},
                            {"params": {"depth": 3}, "score": 0.3},
                        ],
                    }
                ]
            },
        }

        self.assertEqual(select_best(report)["best_score"], 0.3)
        self.assertEqual(summarize(report)["trials_completed"], 4)

    def test_candidate_selection_can_tune_ancestor_after_terminal_unbound_child(
        self,
    ) -> None:
        selected = select_candidate(
            {
                "records": [
                    {
                        "run_id": "000",
                        "status": "keep",
                        "best_warm_score": 0.1,
                        "tune": False,
                        "source_run_ids": [],
                    },
                    {
                        "run_id": "001",
                        "status": "crash",
                        "best_warm_score": 0.2,
                        "tune": False,
                        "source_run_ids": ["000"],
                        "parameter_transfer": None,
                    },
                ]
            },
            n_min=1,
            top_percentile=0,
        )

        self.assertEqual(selected["run_id"], "000")

    def test_candidate_selection_skips_parent_with_unresolved_primary_child(
        self,
    ) -> None:
        selected = select_candidate(
            {
                "records": [
                    {
                        "run_id": "000",
                        "status": "keep",
                        "best_warm_score": 0.1,
                        "tune": False,
                        "source_run_ids": [],
                    },
                    {
                        "run_id": "001",
                        "status": "pending",
                        "tune": False,
                        "source_run_ids": ["000"],
                        "parameter_transfer": None,
                    },
                    {
                        "run_id": "002",
                        "status": "keep",
                        "best_warm_score": 0.2,
                        "tune": False,
                        "source_run_ids": [],
                    },
                ]
            },
            n_min=2,
            top_percentile=0,
        )

        self.assertEqual(selected["run_id"], "002")

    def test_candidate_selection_exposes_and_obeys_deep_tune_allocation(self) -> None:
        ledger = {
            "records": [
                {
                    "run_id": "000",
                    "status": "keep",
                    "best_warm_score": 0.1,
                    "tune": False,
                    "source_run_ids": [],
                }
            ]
        }
        allocation = {
            "remaining": 7,
            "deep_tune": {
                "remaining": 5,
                "total_cap": 40,
                "per_candidate_cap": 3,
                "time_limit_seconds": 600,
                "per_candidate": [{"run_id": "000", "evals": 1}],
            },
        }

        selected = select_candidate(
            ledger,
            n_min=1,
            top_percentile=0,
            budget_allocation=allocation,
        )

        self.assertEqual(selected["run_id"], "000")
        self.assertEqual(
            selected["budget_allocation"]["trial_cap"],
            2,
        )

        allocation["deep_tune"]["remaining"] = 0
        blocked = select_candidate(
            ledger,
            n_min=1,
            top_percentile=0,
            budget_allocation=allocation,
        )
        self.assertIsNone(blocked["run_id"])
        self.assertEqual(blocked["reason"], "deep_tune_budget_exhausted")

    def test_attempt_summary_preserves_warm_retries_and_phase_c_failures(self) -> None:
        report = {
            "phase_a": {
                "warm_start_configs": [{"params": {"depth": 2}, "score": 0.4}],
                "trials_attempted": 3,
            },
            "phase_c": {
                "stages": [{
                    "method": "bo",
                    "trials": [
                        {"params": {"depth": 3}, "score": None, "status": "failed"},
                        {"params": {"depth": 4}, "score": 0.3},
                    ],
                }],
            },
        }

        summary = summarize(report)
        self.assertEqual(summary["trials_completed"], 2)
        self.assertEqual(summary["trials_attempted"], 5)

    def test_preflight_rejections_are_not_objective_attempts(self) -> None:
        report = {
            "preflight": {
                "attempts": [
                    {"params": {"depth": 2}, "status": "ok"},
                    {"params": {"depth": 4}, "status": "failed"},
                ]
            },
            "phase_a": {
                "warm_start_configs": [{"params": {"depth": 2}, "score": 0.4}],
                "trials_attempted": 1,
            },
            "phase_c": {
                "stages": [{
                    "method": "grid",
                    "trials": [
                        {
                            "params": {"depth": 4},
                            "score": None,
                            "status": "preflight_rejected",
                        },
                        {"params": {"depth": 3}, "score": 0.3},
                    ],
                }],
            },
        }

        summary = summarize(report)
        self.assertEqual(summary["trials_completed"], 2)
        self.assertEqual(summary["trials_attempted"], 2)
        self.assertEqual(summary["preflight_attempts"], 2)
        self.assertEqual(summary["preflight_failures"], 1)
        self.assertEqual(summary["feasibility_rejections"], 1)

    def test_timed_eval_rejects_non_finite_in_process_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite score"):
            timed_eval(
                lambda make_model, params: float("inf"),
                _plain_make_model,
                {},
                Path("/tmp/no-framework-config/candidate.py"),
            )

    @mock.patch("_common.read_runtime_limit", return_value=5)
    @mock.patch("_common.subprocess.Popen")
    def test_timed_eval_surfaces_child_process_error(self, popen, _read_limit) -> None:
        process = popen.return_value
        process.communicate.return_value = ("training output", "ValueError: child failed")
        process.returncode = 1

        with self.assertRaisesRegex(RuntimeError, "child failed"):
            timed_eval(object(), _plain_make_model, {}, Path("/tmp/candidate.py"))

    @mock.patch("_common.read_runtime_limit", return_value=5)
    @mock.patch("_common.subprocess.Popen")
    @mock.patch("_common.os.killpg")
    @mock.patch("_common.os.getpgid", return_value=1234)
    def test_timed_eval_surfaces_timeout(
        self, _getpgid, _killpg, popen, _read_limit
    ) -> None:
        process = popen.return_value
        process.pid = 1234
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="eval", timeout=5),
            ("", ""),
        ]

        with self.assertRaisesRegex(TimeoutError, "per_runtime_limit=5s"):
            timed_eval(object(), _plain_make_model, {}, Path("/tmp/candidate.py"))


if __name__ == "__main__":
    unittest.main()
