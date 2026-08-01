"""Deterministic population promotion and Phase-C worker orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import RunIdentity
from .toolchain import Toolchain


@dataclass(frozen=True)
class DeepTuneOutcome:
    tuned_run_id: str | None
    ledger_updated: bool
    reason: str


class DeepTuner:
    def __init__(
        self,
        identity: RunIdentity,
        toolchain: Toolchain,
        task_config: dict[str, Any],
    ):
        self.identity = identity
        self.toolchain = toolchain
        self.task_config = task_config

    def run(self) -> DeepTuneOutcome:
        selection = self.toolchain.select_tuning_candidate(self.identity.run_dir)
        run_id = selection.get("run_id")
        reason = str(selection.get("reason") or "no eligible candidate")
        if run_id is None:
            return DeepTuneOutcome(None, False, reason)
        if not isinstance(run_id, str) or not run_id.isdigit():
            raise ValueError(f"invalid selected tuning run id: {run_id!r}")
        allocation = selection.get("budget_allocation")
        trial_cap = allocation.get("trial_cap") if isinstance(allocation, dict) else None
        if not isinstance(trial_cap, int) or isinstance(trial_cap, bool) or trial_cap <= 0:
            return DeepTuneOutcome(None, False, "no Phase-C objective allocation remains")

        candidate_dir = self.identity.run_dir / "candidates" / run_id
        candidate_path = candidate_dir / "train.py"
        report_path = candidate_dir / "tune_report.json"
        methods_run: list[str] = []
        for stage in range(4):
            action = self.toolchain.phase_c_action(candidate_path, report_path)
            kind = action.get("action")
            if kind == "stop":
                return DeepTuneOutcome(None, False, str(action.get("reason") or "phase-c stop"))
            if kind == "finalize":
                self.toolchain.finalize_tuning(
                    self.identity.run_dir,
                    run_id,
                    candidate_path,
                    report_path,
                )
                return DeepTuneOutcome(run_id, True, reason)
            if kind != "run":
                raise ValueError(f"unknown phase-c action: {kind!r}")
            method = action.get("method")
            if method not in {"grid", "bo", "cmaes"}:
                raise ValueError(f"invalid phase-c method: {method!r}")
            if method in methods_run:
                raise ValueError(f"phase-c action repeated method without progress: {method}")
            methods_run.append(method)
            log_path = (
                self.identity.run_dir
                / ".orchestrator"
                / "workers"
                / f"{run_id}-phase-c-{stage}-{method}.log"
            )
            result = self.toolchain.run_tuner(
                method,
                candidate_path,
                report_path,
                trial_cap,
                self.task_config,
                log_path,
            )
            if not result.ok:
                # The report and objective reservation log are authoritative.
                # A later round can resume this nonterminal stage.
                return DeepTuneOutcome(None, False, "phase_c_interrupted")
        raise ValueError("Phase-C method chain exceeded its deterministic bound")
