#!/usr/bin/env python3
"""Tuning-efficacy diagnostic for one run directory.

Answers the question run artifacts previously hid: did Phase C actually run
the tuner it claims, and were keep/discard decisions made on like-for-like
scores? Motivated by run 0730-ds-ex100-1, where every "bo" stage ended below
TPE's startup regime (zero model-driven draws) and hypotheses were rejected at
screening while incumbents carried 10+ evaluations of tuning.

Read-only. Usage:

    python tools/tuning_efficacy.py <run_dir>

Per candidate it prints warm/final scores, Phase-C method/trials, how many BO
draws were model-driven vs random fallback (receipt fields when present,
reconstructed otherwise), and flags:

- ``bo_never_engaged`` — a "bo" stage with scored trials but zero model-driven
  draws: the stage was random search wearing a BO label.
- ``screening_discard`` — a non-crash discard with no scored Phase-C trial:
  the rejection measures the hypothesis at one parameter point. Under the
  ledger's evidence grading it cannot drive contradiction-grade findings.

Exit status is always 0; this is a report, not a gate.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

DEFAULT_N_STARTUP = 10  # bo_search.TPESampler n_startup_trials


def _finite(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _bo_draw_split(stage: dict) -> tuple[int | None, int | None]:
    """(model_driven, random_fallback) draws for one bo stage.

    Prefer the receipt fields written by bo_search; for legacy reports
    reconstruct the split from priors + ordered trials. The reconstruction
    counts every recorded trial as a draw (deferred warm configs included),
    which leaves the model-driven total exact whenever it matters: those
    early positions were pre-startup in every legacy run seen so far.
    """
    if stage.get("method") != "bo":
        return None, None
    if "model_driven_trials" in stage:
        return (
            int(stage.get("model_driven_trials") or 0),
            int(stage.get("random_fallback_trials") or 0),
        )
    n_startup = int(stage.get("n_startup_trials") or DEFAULT_N_STARTUP)
    completes = int(stage.get("prior_trials_injected") or 0) + int(
        stage.get("infeasible_priors_injected") or 0
    )
    model_driven = 0
    random_fallback = 0
    for trial in stage.get("trials", []):
        if not isinstance(trial, dict):
            continue
        if completes >= n_startup:
            model_driven += 1
        else:
            random_fallback += 1
        if _finite(trial.get("score")) is not None:
            completes += 1
    return model_driven, random_fallback


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] in {"-h", "--help"}:
        print(__doc__.strip())
        return 0 if len(argv) == 2 else 2
    run_dir = Path(argv[1])
    ledger_path = run_dir / "ledger.json"
    if not ledger_path.is_file():
        print(f"no ledger.json under {run_dir}", file=sys.stderr)
        return 2

    ledger = json.loads(ledger_path.read_text())
    records = {
        str(record.get("run_id")): record
        for record in ledger.get("records", [])
        if isinstance(record, dict)
    }

    attempts_path = run_dir / "evaluation_attempts.jsonl"
    phase_counts: dict[str, int] = {}
    if attempts_path.is_file():
        for line in attempts_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase = row.get("phase")
            if isinstance(phase, str):
                phase_counts[phase] = phase_counts.get(phase, 0) + 1

    flags: list[str] = []
    header = (
        f"{'run':>4} {'status':<10} {'depth':<9} {'warm':>8} {'final':>8} "
        f"{'gain':>8} {'pc_method':<9} {'trials':>6} {'md/fb':>9}  notes"
    )
    print(header)
    print("-" * len(header))
    for report_path in sorted((run_dir / "candidates").glob("*/tune_report.json")):
        run_id = report_path.parent.name
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            continue
        record = records.get(run_id, {})
        phase_a = report.get("phase_a", {})
        warm = _finite(phase_a.get("best_warm_score"))
        final = _finite(report.get("final_best_score"))
        if final is None:
            # Screening-only reports carry no closing score; the ledger record
            # holds it once the run is recorded.
            final = _finite(record.get("final_best_score"))
        gain = (warm - final) if warm is not None and final is not None else None
        status = str(record.get("status", "?"))
        stages = report.get("phase_c", {}).get("stages", [])
        scored_trials = sum(
            1
            for stage in stages
            for trial in stage.get("trials", [])
            if isinstance(trial, dict) and _finite(trial.get("score")) is not None
        )
        depth = record.get("evaluation_depth")
        if depth not in {"tuned", "screening"}:
            depth = "tuned" if scored_trials > 0 else "screening"
        methods = ",".join(str(stage.get("method")) for stage in stages) or "-"
        trials_total = sum(len(stage.get("trials", [])) for stage in stages)
        md_fb_parts = []
        notes = []
        for stage in stages:
            md, fb = _bo_draw_split(stage)
            if md is not None:
                md_fb_parts.append(f"{md}/{fb}")
                if md == 0 and any(
                    _finite(t.get("score")) is not None
                    for t in stage.get("trials", [])
                    if isinstance(t, dict)
                ):
                    notes.append("bo_never_engaged")
                    flags.append(f"{run_id}: bo_never_engaged")
        if (
            status == "discard"
            and scored_trials == 0
            and record.get("status") in {"keep", "discard"}
        ):
            notes.append("screening_discard")
            flags.append(f"{run_id}: screening_discard")
        print(
            f"{run_id:>4} {status:<10} {depth:<9} "
            f"{('%.4f' % warm) if warm is not None else '-':>8} "
            f"{('%.4f' % final) if final is not None else '-':>8} "
            f"{('%+.4f' % gain) if gain is not None else '-':>8} "
            f"{methods:<9} {trials_total:>6} {('|'.join(md_fb_parts) or '-'):>9}  "
            f"{','.join(notes)}"
        )

    print()
    if phase_counts:
        total = sum(phase_counts.values())
        parts = ", ".join(f"{k}={v}" for k, v in sorted(phase_counts.items()))
        print(f"admitted evaluations: {total} ({parts})")
    if flags:
        print("flags:")
        for flag in flags:
            print(f"  {flag}")
    else:
        print("flags: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
