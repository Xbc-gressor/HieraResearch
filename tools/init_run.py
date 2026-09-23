#!/usr/bin/env python3
"""Initialize a new autoresearch run directory.

Creates the run directory structure and copies the framework_cfg.json template
if it exists, so the user has a local editable config with all framework
hyperparameters documented inline.

Usage:
    python tools/init_run.py <task_name> <tag>
      [--dimension-strategy <strategy>]
      [--llm-intelligence-score <score>]
      [--semantic-policy <policy>]
      [--scheduler-policy <policy>]
      [--inner-tuner-policy <policy>]
      [--k-warm <count>]
      [--k-eval <count>]
      [--max-evaluations <count>]
      [--time-budget <seconds> | --deadline <epoch-seconds>]
      [--final-reserve <seconds>]
      [--round-new-candidates N] [--round-rewrite-bouts r]
      [--round-tune-bouts t] [--round-seconds Q] [--round-noise-margin E]
      [--round-rewrite-top-k K]
      [--timeout <seconds> | --no-eval-timeout]

Example:
    python tools/init_run.py tabular-model-search exp-20260630 \
      --dimension-strategy llm_induced \
      --llm-intelligence-score 70 \
      --k-warm 5 \
      --k-eval 3 \
      --max-evaluations 200 \
      --timeout 60
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

from semantic_space import DEFAULT_DIMENSION_STRATEGY, DIMENSION_STRATEGIES
from validate_tasks import parse_task_toml
from run_cfg import read_framework_cfg


SEMANTIC_ARTIFACTS = ("dimension_catalog.json", "background.md", "ledger.json")
SEMANTIC_POLICIES = (
    "coverage",
    "coverage_experience",
    "coverage_attempt",
    "coverage_carrier_attempt",
    "gain",
    "gain_uncertainty",
    "gain_uncertainty_nocost",
    "judged_slate",
)
SCHEDULER_POLICIES = (
    "legacy",
    "legacy_wide",
    "v3_2",
    "anchor_challenger_v1",
    "anchor_transfer_challenger_v1",
    "round_v1",
)
#: ``round`` section knobs settable through ``--round-<key>``.
ROUND_OPTIONS = (
    "new_candidates",
    "rewrite_bouts",
    "tune_bouts",
    "round_seconds",
    "noise_margin",
    "rewrite_top_k",
)
INNER_POLICIES = (
    "deferred-random8-hebo10-spsa10-v1",
    "localtr8-hebo10-spsa10-v1",
    "localtr8-hebo10-hebo10-v1",
    "selfrank8-hebo10-hebo10",
    "mixup24-turbo20-v1",
    "hebo24-turbo20-v1",
    "hebo24-hebo20",
    "hebo24-transfer10-hebo10",
    "baseline-hebo-full-v1",
    "legacy",
)
# Which inner-benchmark arm proposes/ranks configs inside HEBO bouts; pairs
# with any inner policy (tuners.inner_policy.PROPOSER_ARMS).
PROPOSER_ARMS = ("pool_hebo_mace", "explicit_e3u2")

# Defaults for newly initialized experiment runs. Existing runs that omit
# these keys keep their historical runtime fallbacks; init_run never rewrites
# an existing run merely because the defaults changed.
DEFAULT_SEMANTIC_POLICY = "judged_slate"
DEFAULT_SCHEDULER_POLICY = "round_v1"
DEFAULT_INNER_POLICY = "hebo24-hebo20"
DEFAULT_PROPOSER_ARM = "explicit_e3u2"
DEFAULT_MAX_EVALUATIONS = 200
#: The transfer pair stays selectable as an explicit comparison arm.
TRANSFER_SCHEDULER_POLICY = "anchor_transfer_challenger_v1"
TRANSFER_INNER_POLICY = "hebo24-transfer10-hebo10"


def _read_framework_config(path: Path) -> dict:
    return read_framework_cfg(path)


def _task_runtime_limit(repo_root: Path, task_name: str) -> float | None:
    """Read the maintained task's default per-evaluation wall-clock limit."""
    task_toml = repo_root / "tasks" / task_name / "task.toml"
    if not task_toml.is_file():
        return None
    config = parse_task_toml(task_toml)
    run = config.get("run")
    if run is None:
        return None
    if not isinstance(run, dict):
        raise ValueError(f"{task_toml}: [run] must be a table")
    value = run.get("timeout_seconds")
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(
            f"{task_toml}: run.timeout_seconds must be a positive finite number"
        )
    return float(value)


def initialize_run(
    repo_root: Path,
    task_name: str,
    tag: str,
    *,
    dimension_strategy: str | None = None,
    llm_intelligence_score: float | None = None,
    semantic_policy: str | None = None,
    scheduler_policy: str | None = None,
    inner_policy: str | None = None,
    proposer_arm: str | None = None,
    k_warm: int | None = None,
    k_eval: int | None = None,
    max_evaluations: int | None = None,
    per_runtime_limit: float | None = None,
    no_eval_timeout: bool = False,
    space_expansion: bool | None = None,
    time_budget_seconds: float | None = None,
    deadline: float | None = None,
    final_reserve_seconds: float | None = None,
    round_options: dict | None = None,
    session_concurrency: int | None = None,
    rewrite_concurrency: int | None = None,
) -> Path:
    if no_eval_timeout and per_runtime_limit is not None:
        raise ValueError("--no-eval-timeout and --timeout are mutually exclusive")
    repo_root = Path(repo_root).resolve()
    round_options = {
        key: value for key, value in (round_options or {}).items()
        if value is not None
    }
    run_dir = repo_root / "runs" / task_name / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir.relative_to(repo_root)}")

    template = repo_root / "tasks" / "framework_cfg.example.json"
    target = run_dir / "framework_cfg.json"
    target_existed = target.exists()
    if template.exists():
        if not target.exists():
            shutil.copy2(template, target)
            print(f"Copied framework config template to {target.relative_to(repo_root)}")
            print("  → Edit this file to override framework behavior for this run.")
        else:
            print(
                "Framework config already exists at "
                f"{target.relative_to(repo_root)}, skipping copy."
            )
    elif not target.exists():
        print(
            f"Template {template.relative_to(repo_root)} not found, "
            "skipping framework_cfg.json copy."
        )
        print("  → Framework will use code defaults.")

    # Make the active policies explicit in every new run artifact. This keeps
    # the normal launch path useful without sacrificing A/B provenance: old
    # arms remain selectable through CLI flags and the resolved values are
    # persisted in framework_cfg.json.
    if not target_existed:
        if semantic_policy is None:
            semantic_policy = DEFAULT_SEMANTIC_POLICY
        if scheduler_policy is None:
            scheduler_policy = (
                TRANSFER_SCHEDULER_POLICY
                if inner_policy == TRANSFER_INNER_POLICY
                else DEFAULT_SCHEDULER_POLICY
            )
        if inner_policy is None:
            inner_policy = (
                TRANSFER_INNER_POLICY
                if scheduler_policy == TRANSFER_SCHEDULER_POLICY
                else DEFAULT_INNER_POLICY
            )
        if proposer_arm is None:
            proposer_arm = DEFAULT_PROPOSER_ARM

    # New runs inherit a task-appropriate limit instead of blindly retaining
    # the generic template's 60 seconds. Existing run-local choices remain
    # untouched, and an explicit --timeout still wins.
    if per_runtime_limit is None and not target_existed and not no_eval_timeout:
        per_runtime_limit = _task_runtime_limit(repo_root, task_name)

    if (
        dimension_strategy is not None
        and dimension_strategy not in DIMENSION_STRATEGIES
    ):
        raise ValueError(
            f"dimension strategy must be one of {sorted(DIMENSION_STRATEGIES)}"
        )
    if (
        llm_intelligence_score is not None
        and (
            isinstance(llm_intelligence_score, bool)
            or not isinstance(llm_intelligence_score, (int, float))
            or not math.isfinite(float(llm_intelligence_score))
            or not 0 <= float(llm_intelligence_score) <= 100
        )
    ):
        raise ValueError(
            "llm intelligence score must be a finite number in [0, 100]"
        )
    if semantic_policy is not None and semantic_policy not in SEMANTIC_POLICIES:
        raise ValueError(
            f"semantic policy must be one of {list(SEMANTIC_POLICIES)}"
        )
    if scheduler_policy is not None and scheduler_policy not in SCHEDULER_POLICIES:
        raise ValueError(
            f"scheduler policy must be one of {list(SCHEDULER_POLICIES)}"
        )
    if inner_policy is not None and inner_policy not in INNER_POLICIES:
        raise ValueError(
            f"inner tuner policy must be one of {list(INNER_POLICIES)}"
        )
    if proposer_arm is not None and proposer_arm not in PROPOSER_ARMS:
        raise ValueError(
            f"proposer arm must be one of {list(PROPOSER_ARMS)}"
        )
    if (
        k_eval is not None
        and (
            not isinstance(k_eval, int)
            or isinstance(k_eval, bool)
            or k_eval < 2
        )
    ):
        raise ValueError(
            "k_eval must be an integer of at least 2 so a non-fresh candidate "
            "evaluates its inherited control plus at least one alternative"
        )
    if (
        k_warm is not None
        and (
            not isinstance(k_warm, int)
            or isinstance(k_warm, bool)
            or k_warm < 2
        )
    ):
        raise ValueError(
            "k_warm must be an integer of at least 2 so at least one row "
            "beyond the control is proposed"
        )
    if (
        max_evaluations is not None
        and (
            not isinstance(max_evaluations, int)
            or isinstance(max_evaluations, bool)
            or max_evaluations <= 0
        )
    ):
        raise ValueError("max_evaluations must be a positive integer")
    if (
        per_runtime_limit is not None
        and (
            isinstance(per_runtime_limit, bool)
            or not isinstance(per_runtime_limit, (int, float))
            or not math.isfinite(per_runtime_limit)
            or per_runtime_limit <= 0
        )
    ):
        raise ValueError("timeout must be a positive number of seconds")
    if time_budget_seconds is not None and deadline is not None:
        raise ValueError("pass either --time-budget or --deadline, not both")
    for label, value in (("time budget", time_budget_seconds), ("deadline", deadline)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{label} must be a positive number of seconds")
    if final_reserve_seconds is not None and (
        isinstance(final_reserve_seconds, bool)
        or not isinstance(final_reserve_seconds, (int, float))
        or not math.isfinite(final_reserve_seconds)
        or final_reserve_seconds < 0
    ):
        raise ValueError("final reserve must be a non-negative number of seconds")
    unknown_round = sorted(set(round_options) - set(ROUND_OPTIONS))
    if unknown_round:
        raise ValueError(f"unknown round options: {unknown_round}")
    existing_deadline = None
    if target_existed and target.exists():
        existing_cfg = _read_framework_config(target)
        raw_deadline = existing_cfg.get("deadline")
        if isinstance(raw_deadline, (int, float)) and not isinstance(
            raw_deadline, bool
        ) and math.isfinite(raw_deadline):
            existing_deadline = float(raw_deadline)
    if existing_deadline is not None:
        if deadline is not None and deadline != existing_deadline:
            raise ValueError(
                "cannot change deadline after it is written; pass the "
                "existing --deadline or omit --time-budget/--deadline"
            )
        # Frozen: resume must not rewrite now+budget over the original
        # wall-clock stop. --time-budget on an already-deadlined run is a
        # no-op for the timestamp.
        time_budget_seconds = None
        deadline = None
    elif time_budget_seconds is not None:
        deadline = time.time() + float(time_budget_seconds)

    if (
        dimension_strategy is None
        and llm_intelligence_score is None
        and semantic_policy is None
        and scheduler_policy is None
        and inner_policy is None
        and k_warm is None
        and k_eval is None
        and max_evaluations is None
        and per_runtime_limit is None
        and not no_eval_timeout
        and space_expansion is None
        and deadline is None
        and final_reserve_seconds is None
        and not round_options
    ):
        from task_contract import stage_task_contract
        stage_task_contract(repo_root, run_dir, task_name)
        return run_dir

    config = _read_framework_config(target) if target.exists() else {}
    updates: list[str] = []
    current_mode = config.get("evaluation_timeout_mode", "fixed")
    requested_mode = "run_budget" if no_eval_timeout else ("fixed" if per_runtime_limit is not None else current_mode)
    if target_existed and "evaluation_timeout_mode" in config and (
        requested_mode != current_mode or
        (per_runtime_limit is not None and per_runtime_limit != config.get("per_runtime_limit"))
    ):
        raise ValueError("cannot change evaluation timeout policy on resume; start a new run")
    if requested_mode == "run_budget":
        if deadline is None and config.get("deadline") is None:
            raise ValueError("--no-eval-timeout requires --time-budget or --deadline")
        config["per_runtime_limit"] = None
        config["preflight_runtime_limit"] = None
    config["evaluation_timeout_mode"] = requested_mode
    updates.append(f"evaluation_timeout_mode={requested_mode}")

    effective_tuner = config.get("tuner", {})
    effective_tuner = effective_tuner if isinstance(effective_tuner, dict) else {}
    effective_scheduler = scheduler_policy or effective_tuner.get(
        "scheduler_policy", DEFAULT_SCHEDULER_POLICY
    )
    # The transfer pair is mutual (run_cfg._validate_anchor_transfer_challenger
    # enforces the same pairing whenever the frozen config is read). Check the
    # scheduler side first: a defaulted inner policy must not hide behind the
    # 24+20 pairing error below.
    if scheduler_policy == "anchor_transfer_challenger_v1":
        effective_inner = inner_policy or effective_tuner.get(
            "inner_policy", DEFAULT_INNER_POLICY
        )
        if effective_inner != "hebo24-transfer10-hebo10":
            raise ValueError(
                "anchor_transfer_challenger_v1 requires inner tuner policy "
                "hebo24-transfer10-hebo10"
            )
    if (
        inner_policy in (
            "mixup24-turbo20-v1",
            "hebo24-turbo20-v1",
            "hebo24-hebo20",
        )
        and effective_scheduler not in ("anchor_challenger_v1", "round_v1")
    ):
        raise ValueError(
            f"{inner_policy} requires scheduler_policy "
            "anchor_challenger_v1 or round_v1"
        )
    if scheduler_policy == "round_v1":
        effective_inner = inner_policy or effective_tuner.get(
            "inner_policy", DEFAULT_INNER_POLICY
        )
        if effective_inner in (TRANSFER_INNER_POLICY, "baseline-hebo-full-v1"):
            raise ValueError(
                f"round_v1 cannot pair with inner tuner policy {effective_inner}"
            )
    if (
        inner_policy == "hebo24-transfer10-hebo10"
        and effective_scheduler != "anchor_transfer_challenger_v1"
    ):
        raise ValueError(
            "hebo24-transfer10-hebo10 requires scheduler_policy "
            "anchor_transfer_challenger_v1"
        )
    if inner_policy == "baseline-hebo-full-v1":
        if effective_scheduler != "legacy":
            raise ValueError(
                "baseline-hebo-full-v1 requires scheduler_policy legacy: "
                "the baseline-tune loop runs one full-budget bout with no "
                "scheduler"
            )
        if max_evaluations is None and config.get("max_evaluations") is None:
            raise ValueError(
                "baseline-hebo-full-v1 requires max_evaluations: the single "
                "HEBO bout spans the whole run budget"
            )

    # Frozen-value guard. It protects a choice this run already recorded, so it
    # only applies once framework_cfg.json exists. A run dir pre-seeded from
    # outside (a frozen background, a copied catalog) has made no such choice:
    # the values in a freshly copied template are defaults, not commitments,
    # and treating them as frozen would reject every non-default flag.
    existing_artifacts = (
        [name for name in SEMANTIC_ARTIFACTS if (run_dir / name).exists()]
        if target_existed
        else []
    )

    # Complete-bout schedulers allocate a finite run-global budget and are
    # invalid without one. round_v1 is bounded by wall clock; the evaluation
    # cap is only its fallback when no time budget is given.
    if (
        scheduler_policy in ("v3_2", "anchor_challenger_v1",
                             "anchor_transfer_challenger_v1")
        and max_evaluations is None
        and config.get("max_evaluations") is None
    ):
        max_evaluations = DEFAULT_MAX_EVALUATIONS
    if (
        scheduler_policy == "round_v1"
        and max_evaluations is None
        and config.get("max_evaluations") is None
        and deadline is None
        and config.get("deadline") is None
    ):
        max_evaluations = DEFAULT_MAX_EVALUATIONS
        print(
            "round_v1 without --time-budget/--deadline: falling back to "
            f"max_evaluations={DEFAULT_MAX_EVALUATIONS}"
        )

    if dimension_strategy is not None:
        section = config.get("space_initialization", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: space_initialization must be an object")
        current = section.get("dimension_strategy", DEFAULT_DIMENSION_STRATEGY)
        if current not in DIMENSION_STRATEGIES:
            raise ValueError(
                f"{target}: space_initialization.dimension_strategy must be one of "
                f"{sorted(DIMENSION_STRATEGIES)}"
            )
        if current != dimension_strategy and existing_artifacts:
            raise ValueError(
                "cannot change dimension strategy after semantic artifacts exist: "
                + ", ".join(existing_artifacts)
            )
        if current != dimension_strategy or not target.exists():
            config["space_initialization"] = {
                **section,
                "dimension_strategy": dimension_strategy,
            }
            updates.append(f"dimension_strategy={dimension_strategy}")
        else:
            print(f"Dimension strategy already set to {dimension_strategy}.")

    if llm_intelligence_score is not None:
        section = config.get("semantic_search", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: semantic_search must be an object")
        current = section.get("llm_intelligence_score")
        if current is not None and (
            isinstance(current, bool)
            or not isinstance(current, (int, float))
            or not math.isfinite(float(current))
            or not 0 <= float(current) <= 100
        ):
            raise ValueError(
                f"{target}: semantic_search.llm_intelligence_score must be "
                "a finite number in [0, 100]"
            )
        normalized_score: int | float = llm_intelligence_score
        if (
            isinstance(normalized_score, float)
            and normalized_score.is_integer()
        ):
            normalized_score = int(normalized_score)
        if (
            (current is None or float(current) != float(normalized_score))
            and existing_artifacts
        ):
            raise ValueError(
                "cannot change llm intelligence score after semantic artifacts "
                "exist: "
                + ", ".join(existing_artifacts)
            )
        if current is None or float(current) != float(normalized_score):
            config["semantic_search"] = {
                **section,
                "llm_intelligence_score": normalized_score,
            }
            updates.append(f"llm_intelligence_score={normalized_score}")
        else:
            print(
                "LLM intelligence score already set to "
                f"{normalized_score}."
            )

    if semantic_policy is not None:
        section = config.get("semantic_search", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: semantic_search must be an object")
        current = section.get("policy")
        if current != semantic_policy and existing_artifacts:
            raise ValueError(
                "cannot change semantic policy after semantic artifacts exist: "
                + ", ".join(existing_artifacts)
            )
        if current != semantic_policy:
            config["semantic_search"] = {
                **section,
                "policy": semantic_policy,
            }
            updates.append(f"semantic_policy={semantic_policy}")
        else:
            print(f"Semantic policy already set to {semantic_policy}.")

    if scheduler_policy is not None:
        section = config.get("tuner", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: tuner must be an object")
        current = section.get("scheduler_policy")
        if current != scheduler_policy and existing_artifacts:
            raise ValueError(
                "cannot change scheduler policy after run artifacts exist: "
                + ", ".join(existing_artifacts)
            )
        if current != scheduler_policy:
            config["tuner"] = {
                **section,
                "scheduler_policy": scheduler_policy,
            }
            updates.append(f"scheduler_policy={scheduler_policy}")
        else:
            print(f"Scheduler policy already set to {scheduler_policy}.")

    if inner_policy is not None:
        section = config.get("tuner", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: tuner must be an object")
        current = section.get("inner_policy")
        if current != inner_policy and existing_artifacts:
            raise ValueError(
                "cannot change inner tuner policy after run artifacts exist: "
                + ", ".join(existing_artifacts)
            )
        if current != inner_policy:
            config["tuner"] = {
                **section,
                "inner_policy": inner_policy,
            }
            updates.append(f"inner_policy={inner_policy}")
        else:
            print(f"Inner tuner policy already set to {inner_policy}.")
        if inner_policy in (
            "mixup24-turbo20-v1",
            "hebo24-turbo20-v1",
            "hebo24-hebo20",
            "hebo24-transfer10-hebo10",
        ):
            section = config.get("tuner", {})
            current_cap = int(section.get("deep_tune_per_candidate_cap", 40))
            if current_cap < 44:
                config["tuner"] = {
                    **section,
                    "deep_tune_per_candidate_cap": 44,
                }
                updates.append("deep_tune_per_candidate_cap=44")
        if inner_policy == "baseline-hebo-full-v1":
            # The single bout spends the whole run budget on one candidate;
            # the per-candidate cap must not truncate it. Re-checked on every
            # init call so a raised max_evaluations on resume raises the cap.
            effective_max = (
                max_evaluations
                if max_evaluations is not None
                else config.get("max_evaluations")
            )
            section = config.get("tuner", {})
            current_cap = int(section.get("deep_tune_per_candidate_cap", 40))
            if effective_max is not None and current_cap < int(effective_max):
                config["tuner"] = {
                    **section,
                    "deep_tune_per_candidate_cap": int(effective_max),
                }
                updates.append(
                    f"deep_tune_per_candidate_cap={int(effective_max)}"
                )

    if proposer_arm is not None:
        section = config.get("tuner", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: tuner must be an object")
        current = section.get("proposer_arm")
        if current != proposer_arm and existing_artifacts:
            raise ValueError(
                "cannot change proposer arm after run artifacts exist: "
                + ", ".join(existing_artifacts)
            )
        if current != proposer_arm:
            config["tuner"] = {**section, "proposer_arm": proposer_arm}
            updates.append(f"proposer_arm={proposer_arm}")
        else:
            print(f"Proposer arm already set to {proposer_arm}.")

    if k_warm is not None:
        section = config.get("tuner", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: tuner must be an object")
        current = section.get("K")
        # Frozen per run for the same reason as K_eval: K fixes how many warm
        # configs the writer proposes per candidate, so K - K_eval is the
        # deferred-config count every promoted candidate's FIRST bout inherits.
        # Changing it mid-run would make earlier and later candidates'
        # screening and first-bout composition incomparable.
        if current != k_warm and existing_artifacts:
            raise ValueError(
                "cannot change K after run artifacts exist: "
                + ", ".join(existing_artifacts)
            )
        if current != k_warm:
            config["tuner"] = {**section, "K": k_warm}
            updates.append(f"K={k_warm}")
        else:
            print(f"K already set to {k_warm}.")

    if k_eval is not None:
        section = config.get("tuner", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: tuner must be an object")
        current = section.get("K_eval")
        # Frozen per run: K_eval is the per-candidate screening cost that the
        # scheduler's resource contract, got_select's admission cap, and every
        # recorded arrival episode are denominated in. Changing it mid-run
        # would make earlier and later screening costs incomparable.
        if current != k_eval and existing_artifacts:
            raise ValueError(
                "cannot change K_eval after run artifacts exist: "
                + ", ".join(existing_artifacts)
            )
        if current != k_eval:
            config["tuner"] = {**section, "K_eval": k_eval}
            updates.append(f"K_eval={k_eval}")
        else:
            print(f"K_eval already set to {k_eval}.")

    if max_evaluations is not None:
        config["max_evaluations"] = max_evaluations
        updates.append(f"max_evaluations={max_evaluations}")

    if deadline is not None:
        config["deadline"] = float(deadline)
        updates.append(f"deadline={float(deadline):.0f}")

    if final_reserve_seconds is not None:
        config["final_reserve_seconds"] = float(final_reserve_seconds)
        updates.append(f"final_reserve_seconds={float(final_reserve_seconds)}")

    if round_options:
        section = config.get("round", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: round must be an object")
        config["round"] = {**section, **round_options}
        updates.extend(f"round.{key}={value}" for key, value in
                       sorted(round_options.items()))

    if session_concurrency is not None:
        if isinstance(session_concurrency, bool) or session_concurrency < 1:
            raise ValueError("--session-concurrency must be a positive integer")
        section = config.get("pipeline", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: pipeline must be an object")
        # Mutable on resume: it changes only how much LLM work overlaps a GPU
        # evaluation, never what is evaluated or how it is scored.
        config["pipeline"] = {**section, "session_concurrency": session_concurrency}
        updates.append(f"pipeline.session_concurrency={session_concurrency}")

    if rewrite_concurrency is not None:
        if isinstance(rewrite_concurrency, bool) or rewrite_concurrency < 1:
            raise ValueError("--rewrite-concurrency must be a positive integer")
        section = config.get("pipeline", {})
        if not isinstance(section, dict):
            raise ValueError(f"{target}: pipeline must be an object")
        # Mutable on resume for the same reason: it only changes how many
        # rewrite climbs overlap, never what is evaluated or scored.
        config["pipeline"] = {**section, "rewrite_concurrency": rewrite_concurrency}
        updates.append(f"pipeline.rewrite_concurrency={rewrite_concurrency}")

    if per_runtime_limit is not None:
        normalized_limit: int | float = per_runtime_limit
        if isinstance(normalized_limit, float) and normalized_limit.is_integer():
            normalized_limit = int(normalized_limit)
        config["per_runtime_limit"] = normalized_limit
        updates.append(f"per_runtime_limit={normalized_limit}")

    if space_expansion is not None:
        config['space_expansion'] = {**config.get('space_expansion', {}), 'enabled': space_expansion}
        updates.append(f"space_expansion.enabled={space_expansion}")
    if config.get('space_expansion', {}).get('enabled'):
        if config.get('deadline') is None or config.get('semantic_search', {}).get('policy') != 'judged_slate':
            raise ValueError("space expansion requires a deadline and judged_slate")

    if updates:
        target.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        print(
            f"Set {', '.join(updates)} in "
            f"{target.relative_to(repo_root)}"
        )
    from task_contract import stage_task_contract
    stage_task_contract(repo_root, run_dir, task_name)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name", help="Task name (e.g., tabular-model-search)")
    parser.add_argument("tag", help="Run tag (e.g., exp-20260630)")
    parser.add_argument(
        "--dimension-strategy",
        choices=sorted(DIMENSION_STRATEGIES),
        help="run search-space initialization strategy (default: catalog_subset)",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        help="global experiment evaluation budget (must be positive)",
    )
    parser.add_argument(
        "--llm-intelligence-score",
        type=float,
        metavar="SCORE",
        help=(
            "LLM intelligence score used by semantic prediction calibration "
            "(finite number in [0, 100])"
        ),
    )
    parser.add_argument(
        "--semantic-policy",
        choices=SEMANTIC_POLICIES,
        help=(
            "semantic point policy; new runs default to "
            f"{DEFAULT_SEMANTIC_POLICY}"
        ),
    )
    parser.add_argument(
        "--scheduler-policy",
        choices=SCHEDULER_POLICIES,
        help=(
            "tuner scheduler policy; new runs default to "
            f"{DEFAULT_SCHEDULER_POLICY}"
        ),
    )
    parser.add_argument(
        "--inner-tuner-policy",
        dest="inner_policy",
        choices=INNER_POLICIES,
        help=(
            "regime-conditioned inner-tuner policy; new runs default to "
            f"{DEFAULT_INNER_POLICY}"
        ),
    )
    parser.add_argument(
        "--proposer-arm",
        dest="proposer_arm",
        choices=PROPOSER_ARMS,
        help=(
            "inner-benchmark arm that proposes and ranks configs inside HEBO "
            f"bouts; new runs default to {DEFAULT_PROPOSER_ARM}"
        ),
    )
    parser.add_argument(
        "--k-warm",
        dest="k_warm",
        type=int,
        metavar="COUNT",
        help=(
            "how many warm configs candidate-writer proposes "
            "per candidate (minimum 2; template default 5). K - K_eval of "
            "them are DEFERRED to the promoted candidate's first tuning "
            "bout. Frozen once run artifacts exist"
        ),
    )
    parser.add_argument(
        "--k-eval",
        dest="k_eval",
        type=int,
        metavar="COUNT",
        help=(
            "how many of the K proposed warm configs are evaluated at "
            "step 0+1 (minimum 2; template default 3). Frozen once run "
            "artifacts exist"
        ),
    )
    timeout_group = parser.add_mutually_exclusive_group()
    parser.add_argument("--space-expansion", action=argparse.BooleanOptionalAction,
                        default=None, help="review stalled semantic space and reserve one probe seat")
    timeout_group.add_argument("--no-eval-timeout", action="store_true",
                               help="no per-evaluation cap; requires a run deadline")
    timeout_group.add_argument(
        "--timeout",
        "--per-runtime-limit",
        dest="per_runtime_limit",
        type=float,
        metavar="SECONDS",
        help="hard wall-clock limit for each evaluation (must be positive)",
    )
    parser.add_argument(
        "--time-budget",
        dest="time_budget_seconds",
        type=float,
        metavar="SECONDS",
        help="run-level wall-clock budget from now; persisted as an absolute "
             "deadline in framework_cfg.json",
    )
    parser.add_argument(
        "--deadline",
        type=float,
        metavar="EPOCH_SECONDS",
        help="absolute run deadline (Unix epoch seconds)",
    )
    parser.add_argument(
        "--final-reserve",
        dest="final_reserve_seconds",
        type=float,
        metavar="SECONDS",
        help="tail of the time budget kept free for finalization/export",
    )
    parser.add_argument("--round-new-candidates", type=int, metavar="N",
                        help="round_v1: new finite candidates per optimization round")
    parser.add_argument("--round-rewrite-bouts", type=int, metavar="R",
                        help="round_v1: rewrite climbs per optimization round "
                             "(one climb hillclimbs one candidate until it "
                             "stalls, caps out, or the round quota ends)")
    parser.add_argument("--round-tune-bouts", type=int, metavar="T",
                        help="round_v1: tune bouts per optimization round")
    parser.add_argument("--round-seconds", type=float, metavar="Q",
                        help="round_v1: wall-clock quota of one optimization round")
    parser.add_argument("--round-noise-margin", type=float, metavar="E",
                        help="round_v1: rewrite keep/revert noise margin (>= 0)")
    parser.add_argument("--round-rewrite-top-k", type=int, metavar="K",
                        help="round_v1: rewrite eligibility is the top-K by score")
    parser.add_argument("--session-concurrency", type=int, metavar="S",
                        help="seats of one generation implemented at once on "
                             "the driver's session channel (1 = serial)")
    parser.add_argument("--rewrite-concurrency", type=int, metavar="R",
                        help="round_v1: rewrite climbs of one optimization "
                             "round run at once over different candidates "
                             "(1 = serial)")
    args = parser.parse_args()
    try:
        initialize_run(
            Path.cwd(),
            args.task_name,
            args.tag,
            dimension_strategy=args.dimension_strategy,
            llm_intelligence_score=args.llm_intelligence_score,
            semantic_policy=args.semantic_policy,
            scheduler_policy=args.scheduler_policy,
            inner_policy=args.inner_policy,
            proposer_arm=args.proposer_arm,
            k_warm=args.k_warm,
            k_eval=args.k_eval,
            max_evaluations=args.max_evaluations,
            per_runtime_limit=args.per_runtime_limit,
            no_eval_timeout=args.no_eval_timeout,
            space_expansion=args.space_expansion,
            time_budget_seconds=args.time_budget_seconds,
            deadline=args.deadline,
            final_reserve_seconds=args.final_reserve_seconds,
            round_options={
                "new_candidates": args.round_new_candidates,
                "rewrite_bouts": args.round_rewrite_bouts,
                "tune_bouts": args.round_tune_bouts,
                "round_seconds": args.round_seconds,
                "noise_margin": args.round_noise_margin,
                "rewrite_top_k": args.round_rewrite_top_k,
            },
            session_concurrency=args.session_concurrency,
            rewrite_concurrency=args.rewrite_concurrency,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
