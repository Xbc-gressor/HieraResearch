"""Upstream 502/503 classification and coordinator auto-backoff."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hieraresearch.upstream as upstream_module  # noqa: E402
from hieraresearch.artifacts import atomic_write_json  # noqa: E402
from hieraresearch.coordinator import ExperimentCoordinator, RunControls  # noqa: E402
from hieraresearch.llm import (  # noqa: E402
    InferenceContractError,
    InferenceError,
    InferenceRequestError,
)
from hieraresearch.models import (  # noqa: E402
    CoordinatorPhase,
    CoordinatorState,
    RunIdentity,
)
from hieraresearch.upstream import (  # noqa: E402
    UpstreamBackoffPolicy,
    backoff_sleep_seconds,
    decide_upstream_recovery,
    is_retryable_upstream_failure,
    last_error_is_upstream_transport,
)


def _exc_502() -> InferenceError:
    return InferenceError(
        "Claude Messages request failed: Error code: 502 - "
        "{'error': {'message': 'Upstream service temporarily unavailable', "
        "'type': 'upstream_error'}, 'type': 'error'}"
    )


def _exc_503_accounts() -> InferenceError:
    return InferenceError(
        "Claude Messages request failed: Error code: 503 - "
        "{'error': {'message': 'No available accounts: no available accounts', "
        "'type': 'api_error'}, 'type': 'error'}"
    )


class ClassifierTests(unittest.TestCase):
    def test_status_embedded_502_and_503_are_upstream(self) -> None:
        self.assertTrue(is_retryable_upstream_failure(_exc_502()))
        self.assertTrue(is_retryable_upstream_failure(_exc_503_accounts()))

    def test_transport_retry_limit_with_upstream_cause_is_retryable(self) -> None:
        wrapped = InferenceError(
            "tuning-values transport retry limit reached: " + str(_exc_502())
        )
        self.assertTrue(is_retryable_upstream_failure(wrapped))

    def test_transport_retry_limit_without_upstream_cause_is_not_retryable(self) -> None:
        wrapped = InferenceError(
            "candidate writer transport retry limit reached: unknown transport failure"
        )
        self.assertFalse(is_retryable_upstream_failure(wrapped))
        # Exhausted local budget alone must never drive coordinator sleep.
        bare = InferenceError("transport retry limit reached")
        self.assertFalse(is_retryable_upstream_failure(bare))

    def test_contract_and_request_errors_are_not_upstream(self) -> None:
        self.assertFalse(
            is_retryable_upstream_failure(
                InferenceContractError("Claude Messages stopped with 'max_tokens'")
            )
        )
        self.assertFalse(
            is_retryable_upstream_failure(
                InferenceRequestError(
                    "Claude Messages rejected the request contract: bad schema"
                )
            )
        )

    def test_unknown_inference_error_is_not_silently_retried(self) -> None:
        self.assertFalse(
            is_retryable_upstream_failure(
                InferenceError("model returned an unusable candidate idea")
            )
        )

    def test_status_code_attribute_on_cause(self) -> None:
        class _SdkError(Exception):
            def __init__(self) -> None:
                super().__init__("boom")
                self.status_code = 529

        exc = InferenceError("wrapper")
        exc.__cause__ = _SdkError()
        self.assertTrue(is_retryable_upstream_failure(exc))

    def test_last_error_helper_matches_persisted_strings(self) -> None:
        self.assertTrue(last_error_is_upstream_transport(str(_exc_502())))
        self.assertFalse(last_error_is_upstream_transport("syntax error near line 1"))
        self.assertFalse(last_error_is_upstream_transport(None))


class PolicyTests(unittest.TestCase):
    def test_backoff_grows_then_caps(self) -> None:
        policy = UpstreamBackoffPolicy(
            base_backoff_seconds=30.0,
            max_single_backoff_seconds=100.0,
            backoff_multiplier=2.0,
        )
        # Full jitter draws uniform(0, cap); pin the draw to the cap so the
        # streak growth and per-attempt cap accounting stay exact.
        with mock.patch.object(upstream_module, "random") as jitter:
            jitter.uniform.side_effect = lambda low, high: high
            self.assertEqual(backoff_sleep_seconds(1, policy), 30.0)
            self.assertEqual(backoff_sleep_seconds(2, policy), 60.0)
            self.assertEqual(backoff_sleep_seconds(3, policy), 100.0)
            jitter.uniform.assert_has_calls(
                [
                    mock.call(0.0, 30.0),
                    mock.call(0.0, 60.0),
                    mock.call(0.0, 100.0),
                ]
            )

    def test_backoff_sleep_is_jittered_within_cap(self) -> None:
        policy = UpstreamBackoffPolicy(
            base_backoff_seconds=30.0,
            max_single_backoff_seconds=100.0,
            backoff_multiplier=2.0,
        )
        for streak, cap in ((1, 30.0), (2, 60.0), (3, 100.0)):
            samples = [backoff_sleep_seconds(streak, policy) for _ in range(60)]
            self.assertTrue(all(0.0 <= sample <= cap for sample in samples))
            # The draw must actually spread, not collapse to the cap.
            self.assertGreater(len(set(samples)), 1)

    def test_streak_exhaustion_blocks(self) -> None:
        policy = UpstreamBackoffPolicy(
            max_streak=2,
            max_backoff_total_seconds=10_000.0,
            base_backoff_seconds=1.0,
            max_single_backoff_seconds=1.0,
        )
        first = decide_upstream_recovery(
            current_streak=0,
            current_backoff_total_seconds=0.0,
            error=_exc_502(),
            policy=policy,
        )
        self.assertEqual(first.action, "backoff")
        self.assertEqual(first.streak, 1)
        second = decide_upstream_recovery(
            current_streak=1,
            current_backoff_total_seconds=first.backoff_total_seconds,
            error=_exc_502(),
            policy=policy,
        )
        self.assertEqual(second.action, "backoff")
        third = decide_upstream_recovery(
            current_streak=2,
            current_backoff_total_seconds=second.backoff_total_seconds,
            error=_exc_502(),
            policy=policy,
        )
        self.assertEqual(third.action, "block")
        self.assertIn("upstream_failure_streak_exhausted", third.reason)

    def test_wall_clock_cap_blocks_before_sleep(self) -> None:
        policy = UpstreamBackoffPolicy(
            max_streak=20,
            max_backoff_total_seconds=50.0,
            base_backoff_seconds=40.0,
            max_single_backoff_seconds=40.0,
        )
        # Pin the jittered draw to the cap: the block decision must be made
        # against the largest admissible sleep.
        with mock.patch.object(upstream_module, "random") as jitter:
            jitter.uniform.return_value = 40.0
            decision = decide_upstream_recovery(
                current_streak=0,
                current_backoff_total_seconds=20.0,
                error=_exc_503_accounts(),
                policy=policy,
            )
        self.assertEqual(decision.action, "block")
        self.assertIn("upstream_backoff_wall_clock_exhausted", decision.reason)
        self.assertEqual(decision.sleep_seconds, 0.0)


class StateParseTests(unittest.TestCase):
    def test_absent_upstream_fields_default(self) -> None:
        state = CoordinatorState.from_dict(
            {
                "schema_version": 1,
                "task_name": "toy",
                "tag": "t",
                "phase": "running",
            }
        )
        self.assertEqual(state.upstream_failure_streak, 0)
        self.assertEqual(state.upstream_backoff_total_seconds, 0.0)

    def test_malformed_upstream_fields_are_rejected(self) -> None:
        base = {
            "schema_version": 1,
            "task_name": "toy",
            "tag": "t",
            "phase": "running",
        }
        with self.assertRaises(ValueError):
            CoordinatorState.from_dict({**base, "upstream_failure_streak": -5})
        with self.assertRaises(ValueError):
            CoordinatorState.from_dict({**base, "upstream_failure_streak": True})
        with self.assertRaises(ValueError):
            CoordinatorState.from_dict({**base, "upstream_failure_streak": "3"})
        with self.assertRaises(ValueError):
            CoordinatorState.from_dict(
                {**base, "upstream_backoff_total_seconds": -1.0}
            )
        with self.assertRaises(ValueError):
            CoordinatorState.from_dict(
                {**base, "upstream_backoff_total_seconds": float("nan")}
            )
        with self.assertRaises(ValueError):
            CoordinatorState.from_dict(
                {**base, "upstream_backoff_total_seconds": "1.5"}
            )
        with self.assertRaises(ValueError):
            CoordinatorState.from_dict({**base, "upstream_last_error": 12})


class CoordinatorUpstreamTests(unittest.TestCase):
    def _coordinator(
        self, tmp: str, *, sleeps: list[float], **control_kwargs: object
    ) -> ExperimentCoordinator:
        identity = RunIdentity(Path(tmp), "toy", "upstream-recovery")
        identity.run_dir.mkdir(parents=True)
        (identity.run_dir / ".orchestrator").mkdir(parents=True)
        controls = RunControls(
            sync_environment=False,
            prepare_task=False,
            upstream_max_streak=int(control_kwargs.get("max_streak", 3)),
            upstream_max_backoff_total_seconds=float(
                control_kwargs.get("max_total", 10_000.0)
            ),
            upstream_base_backoff_seconds=float(control_kwargs.get("base", 2.0)),
            upstream_max_single_backoff_seconds=float(
                control_kwargs.get("max_single", 2.0)
            ),
        )

        class _Unused:
            pass

        coordinator = ExperimentCoordinator(
            identity,
            toolchain=_Unused(),  # type: ignore[arg-type]
            models=_Unused(),  # type: ignore[arg-type]
            controls=controls,
            sleep=sleeps.append,
        )
        coordinator.state = CoordinatorState(task_name="toy", tag="upstream-recovery")
        return coordinator

    def test_recover_backs_off_then_blocks_on_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sleeps: list[float] = []
            coordinator = self._coordinator(tmp, sleeps=sleeps, max_streak=2)
            assert coordinator.state is not None

            self.assertTrue(coordinator._recover_from_upstream(_exc_502()))
            self.assertEqual(coordinator.state.phase, CoordinatorPhase.RUNNING)
            self.assertEqual(coordinator.state.upstream_failure_streak, 1)
            self.assertEqual(len(sleeps), 1)
            self.assertTrue(0.0 <= sleeps[0] <= 2.0)

            self.assertTrue(coordinator._recover_from_upstream(_exc_503_accounts()))
            self.assertEqual(coordinator.state.upstream_failure_streak, 2)
            self.assertEqual(len(sleeps), 2)
            self.assertTrue(0.0 <= sleeps[1] <= 2.0)

            self.assertTrue(coordinator._recover_from_upstream(_exc_502()))
            self.assertEqual(coordinator.state.phase, CoordinatorPhase.BLOCKED)
            self.assertIn(
                "upstream_failure_streak_exhausted",
                coordinator.state.stop_condition or "",
            )
            self.assertEqual(len(sleeps), 2)

            state_path = coordinator.identity.run_dir / ".orchestrator" / "state.json"
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["phase"], "blocked")
            self.assertEqual(persisted["upstream_failure_streak"], 3)

    def test_non_upstream_inference_is_not_absorbed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sleeps: list[float] = []
            coordinator = self._coordinator(tmp, sleeps=sleeps)
            self.assertFalse(
                coordinator._recover_from_upstream(
                    InferenceError("model returned an unusable candidate idea")
                )
            )
            self.assertEqual(sleeps, [])
            self.assertEqual(coordinator.state.phase, CoordinatorPhase.RUNNING)

    def test_model_success_clears_streak_noop_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sleeps: list[float] = []
            coordinator = self._coordinator(tmp, sleeps=sleeps)
            assert coordinator.state is not None
            coordinator.state.upstream_failure_streak = 4
            coordinator.state.upstream_backoff_total_seconds = 120.0
            coordinator.state.upstream_last_error = "old"

            before = coordinator._completed_invocation_count()
            coordinator._maybe_clear_upstream_after_model_success(before)
            self.assertEqual(coordinator.state.upstream_failure_streak, 4)

            inv = (
                coordinator.identity.run_dir
                / ".orchestrator"
                / "invocations"
                / "fake-success"
            )
            inv.mkdir(parents=True)
            atomic_write_json(inv / "receipt.json", {"status": "completed"})
            coordinator._maybe_clear_upstream_after_model_success(before)
            self.assertEqual(coordinator.state.upstream_failure_streak, 0)
            self.assertEqual(coordinator.state.upstream_backoff_total_seconds, 0.0)
            self.assertIsNone(coordinator.state.upstream_last_error)

    def test_resume_blocked_resets_upstream_counters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sleeps: list[float] = []
            coordinator = self._coordinator(tmp, sleeps=sleeps)
            assert coordinator.state is not None
            coordinator.state.phase = CoordinatorPhase.BLOCKED
            coordinator.state.stop_condition = "upstream_failure_streak_exhausted"
            coordinator.state.upstream_failure_streak = 9
            coordinator.state.upstream_backoff_total_seconds = 999.0
            coordinator.controls = RunControls(
                sync_environment=False,
                prepare_task=False,
                resume_blocked=True,
                upstream_max_streak=3,
                upstream_base_backoff_seconds=2.0,
                upstream_max_single_backoff_seconds=2.0,
            )
            coordinator._upstream_policy = coordinator._policy_from_controls(
                coordinator.controls
            )

            class _LedgerPhase:
                def set_phase(self, *args, **kwargs):
                    del args, kwargs

            coordinator.toolchain = _LedgerPhase()  # type: ignore[assignment]
            coordinator._resume_state_if_authorized()
            self.assertEqual(coordinator.state.phase, CoordinatorPhase.RUNNING)
            self.assertIsNone(coordinator.state.stop_condition)
            self.assertEqual(coordinator.state.upstream_failure_streak, 0)
            self.assertEqual(coordinator.state.upstream_backoff_total_seconds, 0.0)

    def test_status_exposes_upstream_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sleeps: list[float] = []
            coordinator = self._coordinator(tmp, sleeps=sleeps)
            assert coordinator.state is not None
            coordinator.state.upstream_failure_streak = 2
            coordinator.state.upstream_last_decision = "upstream_backoff:2.0s"
            status = coordinator.status()
            self.assertEqual(status["upstream_failure_streak"], 2)
            self.assertEqual(status["upstream_last_decision"], "upstream_backoff:2.0s")


if __name__ == "__main__":
    unittest.main()
