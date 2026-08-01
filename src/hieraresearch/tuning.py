"""Deterministic population promotion and Phase-C worker orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import ArtifactError, paths_revision
from .models import DeepTuneOutcome, DeepTuneSelection, RunIdentity
from .toolchain import Toolchain


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

    def select(self) -> DeepTuneSelection:
        before = paths_revision(self._selection_input_paths())
        value = self.toolchain.select_tuning_candidate(self.identity.run_dir)
        if paths_revision(self._selection_input_paths()) != before:
            raise ArtifactError("deep-tune selection inputs changed during selection")
        return DeepTuneSelection.from_tool_result(value, input_revision=before)

    def run(self, selection: DeepTuneSelection) -> DeepTuneOutcome:
        run_id = selection.run_id
        if run_id is None:
            return DeepTuneOutcome(None, False, selection.reason)
        trial_cap = selection.trial_cap
        if trial_cap is None:  # pragma: no cover - validated by DeepTuneSelection
            raise ValueError("reserved deep-tune candidate has no trial cap")

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
                receipt = self.toolchain.finalize_tuning(
                    self.identity.run_dir,
                    run_id,
                    candidate_path,
                    report_path,
                )
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("status") != "ok"
                    or receipt.get("run_id") != run_id
                    or receipt.get("ledger_updated") is not True
                ):
                    raise ArtifactError(
                        f"invalid deep-tune finalization receipt for {run_id}: {receipt!r}"
                    )
                record = self.toolchain.ledger_record(
                    self.identity.run_dir,
                    run_id,
                )
                if (
                    not isinstance(record, dict)
                    or record.get("tune") is not True
                    or record.get("final_best_score")
                    != receipt.get("final_best_score")
                ):
                    raise ArtifactError(
                        f"deep-tune finalization did not commit ledger record {run_id}"
                    )
                return DeepTuneOutcome(
                    run_id,
                    True,
                    str(action.get("reason") or "phase_c_finalized"),
                )
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

    def _selection_input_paths(self) -> tuple[Path, ...]:
        run_dir = self.identity.run_dir
        return (
            run_dir / "ledger.json",
            run_dir / "framework_cfg.json",
            run_dir / "evaluation_attempts.jsonl",
            *sorted((run_dir / "candidates").glob("*/tune_report.json")),
        )
