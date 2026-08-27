"""Shared reading of the run-local `framework_cfg.json`.

That file is the single source of the evaluation budget, the per-evaluation
wall-clock limit, and per-run meta-parameter overrides, and it is designed to
be hand-edited (`init_run.py` invites the user to edit it). A file that exists
but cannot be parsed is therefore a hard error, not "unconfigured": the guards
enforcing budget and timeout must fail fast instead of silently dropping the
limits they exist to enforce. A missing file remains a legitimate "no
overrides configured" state and yields the caller's default.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class RunConfigError(ValueError):
    """framework_cfg.json exists but cannot be read, parsed, or validated."""


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _validate_optional_positive_int(
    config: dict,
    key: str,
    path: Path,
    *,
    label: str | None = None,
) -> None:
    if key not in config or config[key] is None:
        return
    value = config[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RunConfigError(
            f"{path}: {label or key} must be a positive integer or null"
        )


def _validate_positive_int_override(
    config: dict,
    key: str,
    path: Path,
    *,
    label: str | None = None,
) -> None:
    """Validate an integer override whose explicit null is not meaningful."""
    if key not in config:
        return
    value = config[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RunConfigError(f"{path}: {label or key} must be a positive integer")


def _validate_optional_positive_number(config: dict, key: str, path: Path) -> None:
    if key not in config or config[key] is None:
        return
    value = config[key]
    if (
        not _is_finite_number(value)
        or value <= 0
    ):
        raise RunConfigError(f"{path}: {key} must be a positive finite number or null")


def _validate_positive_number_override(config: dict, key: str, path: Path) -> None:
    """Validate a numeric override whose explicit null is not meaningful."""
    if key not in config:
        return
    value = config[key]
    if not _is_finite_number(value) or value <= 0:
        raise RunConfigError(f"{path}: tuner.{key} must be a positive finite number")


def _validate_tuner_config(tuner: dict, path: Path) -> None:
    """Validate every tuner override consumed by deterministic Python code."""
    # These consumers treat null as "use the derived/default value".
    for key in ("K", "K_eval", "n_min", "bo_patience"):
        _validate_optional_positive_int(
            tuner,
            key,
            path,
            label=f"tuner.{key}",
        )
    if (
        tuner.get("K") is not None
        and int(tuner["K"]) < 2
    ):
        raise RunConfigError(
            f"{path}: tuner.K must be at least 2 so a candidate proposes "
            "one row beyond its control"
        )
    if (
        tuner.get("K_eval") is not None
        and int(tuner["K_eval"]) < 2
    ):
        raise RunConfigError(
            f"{path}: tuner.K_eval must be at least 2 so a non-fresh "
            "candidate evaluates its inherited control plus an alternative"
        )

    # These consumers call int(value) whenever the key is present, so an
    # explicit null is invalid rather than equivalent to omission.
    for key in (
        "bo_n_trials",
        "bo_patience_cap",
        "bo_patience_floor",
        "deep_tune_per_candidate_cap",
        "bout_trials",
        "tuned_threshold",
        "rewarm_proposals",
    ):
        _validate_positive_int_override(
            tuner,
            key,
            path,
            label=f"tuner.{key}",
        )

    if "top_percentile" in tuner:
        value = tuner["top_percentile"]
        if (
            not _is_finite_number(value)
            or not 0 <= float(value) < 100
        ):
            raise RunConfigError(
                f"{path}: tuner.top_percentile must be a finite number "
                "in [0, 100)"
            )

    if "scheduler_policy" in tuner:
        # Scheduler v3.2 is an isolated policy arm; `legacy` and `legacy_wide`
        # are the percentile/alternation gate, differing only in how widely a
        # first-bout non-responder may be re-admitted. A typo here must fail
        # loudly rather than silently run the arm the experiment compares
        # against.
        value = tuner["scheduler_policy"]
        if value not in (
            "legacy",
            "legacy_wide",
            "v3_2",
            "anchor_challenger_v1",
            "anchor_transfer_challenger_v1",
        ):
            raise RunConfigError(
                f"{path}: tuner.scheduler_policy must be 'legacy', "
                "'legacy_wide', 'v3_2', 'anchor_challenger_v1', or "
                "'anchor_transfer_challenger_v1'"
            )

    if "inner_policy" in tuner:
        # Keep this list in lockstep with tuners.inner_policy.KNOWN_POLICY_IDS.
        # run_cfg is imported from stdlib-only helpers; do not import the
        # tuner package here.
        known = (
            "deferred-random8-hebo10-spsa10-v1",
            "localtr8-hebo10-spsa10-v1",
            "localtr8-hebo10-hebo10-v1",
            "selfrank8-hebo10-hebo10",
            "mixup24-turbo20-v1",
            "hebo24-turbo20-v1",
            "hebo24-hebo20",
            # The transfer policies pair only with each other (checked below
            # and in _validate_anchor_transfer_challenger).
            "hebo24-transfer10-hebo10",
            "baseline-hebo-full-v1",
            "legacy",
        )
        value = tuner["inner_policy"]
        if value not in known:
            allowed = " or ".join(repr(item) for item in known)
            raise RunConfigError(
                f"{path}: tuner.inner_policy must be {allowed}"
            )
        if (
            value in (
                "mixup24-turbo20-v1",
                "hebo24-turbo20-v1",
                "hebo24-hebo20",
            )
            and tuner.get("scheduler_policy", "v3_2")
            != "anchor_challenger_v1"
        ):
            raise RunConfigError(
                f"{path}: tuner.inner_policy {value!r} requires "
                "tuner.scheduler_policy 'anchor_challenger_v1'"
            )
        if (
            value == "hebo24-transfer10-hebo10"
            and tuner.get("scheduler_policy", "v3_2")
            != "anchor_transfer_challenger_v1"
        ):
            raise RunConfigError(
                f"{path}: tuner.inner_policy {value!r} requires "
                "tuner.scheduler_policy 'anchor_transfer_challenger_v1'"
            )
        if (
            value == "baseline-hebo-full-v1"
            and tuner.get("scheduler_policy", "v3_2") != "legacy"
        ):
            raise RunConfigError(
                f"{path}: tuner.inner_policy {value!r} is the "
                "baseline-tune loop's single full-budget bout and requires "
                "tuner.scheduler_policy 'legacy' (no scheduler)"
            )

    for key in ("scheduler_scenarios", "max_bouts_per_candidate"):
        _validate_positive_int_override(tuner, key, path, label=f"tuner.{key}")

    if tuner.get("deep_tune_budget_fraction") is not None:
        # null (or the key omitted) = no run-level Phase-C share, the default.
        value = tuner["deep_tune_budget_fraction"]
        if (
            not _is_finite_number(value)
            or not 0 <= float(value) <= 1
        ):
            raise RunConfigError(
                f"{path}: tuner.deep_tune_budget_fraction must be null or a "
                "finite number in [0, 1]"
            )

    _validate_positive_number_override(
        tuner,
        "deep_tune_time_limit_seconds",
        path,
    )

    # The adaptive rule in bo_search is
    # min(cap, max(floor, round(1.5 * n_dims))). Check the effective pair,
    # including its code defaults, whenever fixed patience is not selected.
    if tuner.get("bo_patience") is None:
        cap = tuner.get("bo_patience_cap", 20)
        floor = tuner.get("bo_patience_floor", 12)
        if floor > cap:
            raise RunConfigError(
                f"{path}: tuner.bo_patience_floor must be less than or equal "
                "to tuner.bo_patience_cap"
            )


def _validate_scheduler_v3_2(config: dict, tuner: dict, path: Path) -> None:
    """Reject configs whose real admission layer cannot execute a v3.2 decision.

    v3.2's whole resource contract is "a bout is admitted at full `B` or not
    at all", and the scheduler decides against a `remaining_budget` it reads
    from the same attempt log `reserve_evaluation` enforces. Three ways a
    run config can break that agreement, all silently:

    * **no `max_evaluations`.** The scheduler allocates a finite budget
      between TUNE and DEFER. With no cap there is nothing to allocate and
      `load_state` cannot even build a state.
    * **an explicit `deep_tune_budget_fraction`.** It is a second, invisible
      ceiling on Phase C that the scheduler does not model: decisions keep
      returning full TUNE while `reserve_evaluation` starts refusing the
      reservations, so a chosen bout dies partway and the realized
      transition stops matching the simulated one.
    * **`deep_tune_per_candidate_cap` below the policy-aware lifetime
      cost.** The legacy per-candidate attempt cap is what v3.2's bout
      cap replaces. Left smaller, it truncates a bout the scheduler
      admitted at full `B`. The frozen regime-conditioned policy is
      ``8+10+10+10=38``, not ``10*4=40``.

    Rejecting here rather than at decide time is deliberate: the failure is
    a property of the run's configuration, so it should stop the run before
    it spends its first evaluation.
    """
    if config.get("max_evaluations") is None:
        raise RunConfigError(
            f"{path}: tuner.scheduler_policy 'v3_2' requires max_evaluations "
            "(the scheduler allocates a bounded budget between TUNE and DEFER)"
        )
    if tuner.get("deep_tune_budget_fraction") is not None:
        raise RunConfigError(
            f"{path}: tuner.deep_tune_budget_fraction is incompatible with "
            "tuner.scheduler_policy 'v3_2' — the bout contract "
            "(B x MAX_BOUTS_PER_CANDIDATE) is the only Phase-C ceiling under "
            "v3.2; set it to null"
        )
    from scheduler.contract import (
        B_FIRST,
        BOUT_TRIALS,
        MAX_BOUTS_PER_CANDIDATE,
        ResourceContract,
    )

    bout_trials = int(tuner.get("bout_trials", BOUT_TRIALS))
    max_bouts = int(tuner.get("max_bouts_per_candidate", MAX_BOUTS_PER_CANDIDATE))
    # Same first-bout cost session.contract_for uses: every
    # regime-conditioned inner policy charges B_FIRST; only the explicit
    # legacy inner policy does not.
    legacy_inner = str(tuner.get("inner_policy", "")) == "legacy"
    contract = ResourceContract(
        bout_trials=bout_trials,
        max_bouts=max_bouts,
        first_bout_trials=bout_trials if legacy_inner else B_FIRST,
    )
    required = contract.lifetime_cost()
    per_candidate_cap = int(tuner.get("deep_tune_per_candidate_cap", 40))
    if per_candidate_cap < required:
        later_bouts = max(0, max_bouts - 1)
        raise RunConfigError(
            f"{path}: tuner.deep_tune_per_candidate_cap ({per_candidate_cap}) "
            f"is below the v3.2 bout contract "
            f"({contract.first_bout_trials} + {bout_trials} x {later_bouts} "
            f"= {required}); it would truncate a bout the scheduler "
            "admitted at full B"
        )


def _validate_anchor_challenger(config: dict, tuner: dict, path: Path) -> None:
    """Validate the deterministic two-INITIAL / two-DEEP tournament."""
    max_evaluations = config.get("max_evaluations")
    if max_evaluations is None:
        raise RunConfigError(
            f"{path}: tuner.scheduler_policy 'anchor_challenger_v1' requires "
            "max_evaluations"
        )
    if tuner.get("deep_tune_budget_fraction") is not None:
        raise RunConfigError(
            f"{path}: tuner.deep_tune_budget_fraction is incompatible with "
            "tuner.scheduler_policy 'anchor_challenger_v1'; its hard "
            "tournament reserve is the only Phase-C ceiling"
        )

    from tuners.inner_policy import POLICY_ID, expected_bout_trials

    legacy_bout_trials = int(tuner.get("bout_trials", 10))
    inner_policy_id = str(tuner.get("inner_policy", POLICY_ID))
    schedule = tuple(
        expected_bout_trials(inner_policy_id, index, legacy_bout_trials)
        for index in range(3)
    )
    # The second DEEP segment is index 2 when it stays with a responder, but
    # index 1 when zero gain switches to the other initialized candidate.
    tournament_total = 2 * schedule[0] + schedule[1] + max(schedule[1:])
    if int(max_evaluations) < tournament_total:
        raise RunConfigError(
            f"{path}: max_evaluations ({max_evaluations}) is below the full "
            f"anchor/challenger tournament reserve ({tournament_total})"
        )
    candidate_lifetime = sum(schedule)
    per_candidate_cap = int(
        tuner.get("deep_tune_per_candidate_cap", max(40, candidate_lifetime))
    )
    if per_candidate_cap < candidate_lifetime:
        raise RunConfigError(
            f"{path}: tuner.deep_tune_per_candidate_cap ({per_candidate_cap}) "
            f"is below one candidate's three-bout schedule "
            f"({schedule[0]} + {schedule[1]} + {schedule[2]} = "
            f"{candidate_lifetime})"
        )


def _validate_anchor_transfer_challenger(config: dict, tuner: dict, path: Path) -> None:
    """Validate the anchor + donor-transfer challenger pair (design §2).

    The scheduler reserves 10-eval segments the inner tuner must actually
    price (and vice versa), so the two transfer policies pair only with each
    other. The reserve is one ordinary INITIAL plus two post-anchor segments
    (24 + 10 + 10 = 44 under ``hebo24-transfer10-hebo10``); the per-candidate
    cap must keep the full ordinary three-bout lifetime.
    """
    if str(tuner.get("inner_policy", "")) != "hebo24-transfer10-hebo10":
        raise RunConfigError(
            f"{path}: tuner.scheduler_policy 'anchor_transfer_challenger_v1' "
            "requires tuner.inner_policy 'hebo24-transfer10-hebo10' (the "
            "transfer policies pair only with each other)"
        )
    max_evaluations = config.get("max_evaluations")
    if max_evaluations is None:
        raise RunConfigError(
            f"{path}: tuner.scheduler_policy 'anchor_transfer_challenger_v1' "
            "requires max_evaluations"
        )
    if tuner.get("deep_tune_budget_fraction") is not None:
        raise RunConfigError(
            f"{path}: tuner.deep_tune_budget_fraction is incompatible with "
            "tuner.scheduler_policy 'anchor_transfer_challenger_v1'; its hard "
            "transfer tournament reserve is the only Phase-C ceiling"
        )
    if (
        tuner.get("K_eval") is not None
        and int(tuner["K_eval"]) < 3
    ):
        raise RunConfigError(
            f"{path}: tuner.K_eval must be at least 3 under "
            "'anchor_transfer_challenger_v1'; the mandatory lineage and "
            "donor roles need three screening slots"
        )

    from tuners.inner_policy import expected_bout_trials

    legacy_bout_trials = int(tuner.get("bout_trials", 10))
    schedule = tuple(
        expected_bout_trials("hebo24-transfer10-hebo10", index, legacy_bout_trials)
        for index in range(3)
    )
    # The post-anchor segments are the challenger's TRANSFERRED bout or an
    # anchor DEEP continuation; both price at the 10-eval later-bout cost.
    tournament_total = schedule[0] + 2 * max(schedule[1:])
    if int(max_evaluations) < tournament_total:
        raise RunConfigError(
            f"{path}: max_evaluations ({max_evaluations}) is below the full "
            f"anchor/transfer-challenger tournament reserve ({tournament_total})"
        )
    candidate_lifetime = sum(schedule)
    per_candidate_cap = int(
        tuner.get("deep_tune_per_candidate_cap", max(40, candidate_lifetime))
    )
    if per_candidate_cap < candidate_lifetime:
        raise RunConfigError(
            f"{path}: tuner.deep_tune_per_candidate_cap ({per_candidate_cap}) "
            f"is below one candidate's three-bout schedule "
            f"({schedule[0]} + {schedule[1]} + {schedule[2]} = "
            f"{candidate_lifetime})"
        )


def _validate_judged_slate_config(judged_slate: dict, path: Path) -> None:
    """Validate the judged-slate listwise-judge arm's pool configuration."""
    unknown = sorted(set(judged_slate) - {"pool_size"})
    if unknown:
        raise RunConfigError(f"{path}: unknown judged_slate keys {unknown}")
    if "pool_size" not in judged_slate or judged_slate["pool_size"] is None:
        return
    value = judged_slate["pool_size"]
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 3 <= value <= 12
    ):
        raise RunConfigError(
            f"{path}: judged_slate.pool_size must be an integer in [3, 12]"
        )


def _validate_framework_cfg(config: dict, path: Path) -> None:
    """Validate the hard-limit fields shared by deterministic consumers."""
    _validate_optional_positive_int(config, "max_evaluations", path)
    _validate_optional_positive_number(config, "per_runtime_limit", path)
    _validate_optional_positive_number(config, "preflight_runtime_limit", path)

    judged_slate = config.get("judged_slate")
    if judged_slate is not None:
        if not isinstance(judged_slate, dict):
            raise RunConfigError(f"{path}: judged_slate must be an object")
        _validate_judged_slate_config(judged_slate, path)

    tuner = config.get("tuner")
    if tuner is None:
        return
    if not isinstance(tuner, dict):
        raise RunConfigError(f"{path}: tuner must be an object")
    _validate_tuner_config(tuner, path)
    if tuner.get("scheduler_policy") == "v3_2":
        _validate_scheduler_v3_2(config, tuner, path)
    elif tuner.get("scheduler_policy") == "anchor_challenger_v1":
        _validate_anchor_challenger(config, tuner, path)
    elif tuner.get("scheduler_policy") == "anchor_transfer_challenger_v1":
        _validate_anchor_transfer_challenger(config, tuner, path)


def read_framework_cfg(path: Any) -> dict:
    """Parse one framework_cfg.json into a dict.

    Raises RunConfigError when the file is unreadable, is not valid JSON, or
    does not contain a JSON object.
    """
    path = Path(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunConfigError(f"cannot read framework config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunConfigError(f"{path}: framework config must be an object")
    _validate_framework_cfg(value, path)
    return value


def find_framework_cfg(ref_path: Any) -> Path | None:
    """Nearest ancestor (inclusive) of ref_path holding a framework_cfg.json."""
    p = Path(ref_path).resolve()
    for anc in (p, *p.parents):
        cfg = anc / "framework_cfg.json"
        if cfg.is_file():
            return cfg
    return None


def load_run_cfg(ref_path: Any, section: str) -> dict:
    """One section from the nearest framework_cfg.json ({} when none exists)."""
    cfg = find_framework_cfg(ref_path)
    if cfg is None:
        return {}
    return dict(read_framework_cfg(cfg).get(section, {}))
