"""Regime-conditioned inner-tuner policy (scheduler v3.2 design §2.1).

The frozen production contract::

    inner_tuner_policy_id = deferred-random8-hebo10-spsa10-v1

    B_FIRST = 8, B_CONTINUE = 10, B_DEEP = 10, MAX_BOUTS_PER_CANDIDATE = 4

    0 completed bouts   -> FIRST
    1 completed bout    -> CONTINUE
    2-3 completed bouts -> DEEP

    FIRST    | deferred-warm configs first; remaining slots are explicit
             | Optuna RandomSampler draws over the production distributions
             | (NOT TPESampler(n_startup_trials=8): warm/infeasible priors and
             | deferred trials can push TPE past its startup threshold).
    CONTINUE | prompt-v2 HEBO arm (PLAN §6.4 ``pool_hebo_mace``): LLM pool
             | proposer (POOL=5, noise-range notes + heterogeneity
             | requirement) ranked by official HEBO MACE; one executed
             | config per step. No TPE/grid/cmaes fallback labeled HEBO.
    DEEP     | two-sided SPSA, 5 complete perturbation pairs (10 evals).
             | Only candidates with at least one non-degenerate continuous
             | (float) dimension have a DEEP action; others are simply not
             | eligible once they reach the DEEP regime — never a silent
             | TPE/grid bout still labeled DEEP/SPSA.

Under the new policy a bout's deferred-warm backlog occupies slots INSIDE
``B_q`` (design §2 rule 4); the legacy policy kept them as extra trials on
top of the bout budget.

The switch is per-run: ``framework_cfg.json`` ``tuner.inner_policy`` —
``deferred-random8-hebo10-spsa10-v1`` (default) or ``legacy`` (the
pre-policy uniform behavior: every bout runs the CONTINUE rule at
``tuner.bout_trials``). Stdlib-only at module level so both ``tune_tools``
and ``_common`` can import it without cycles.
"""

from __future__ import annotations

POLICY_ID = "deferred-random8-hebo10-spsa10-v1"
LEGACY_POLICY_ID = "legacy"

FIRST = "FIRST"
CONTINUE = "CONTINUE"
DEEP = "DEEP"

B_FIRST = 8
B_CONTINUE = 10
B_DEEP = 10
MAX_BOUTS_PER_CANDIDATE = 4

_REGIME_BOUT_SIZES = {FIRST: B_FIRST, CONTINUE: B_CONTINUE, DEEP: B_DEEP}


def load_policy_id(ref_path) -> str:
    """``tuner.inner_policy`` from the run's framework_cfg (default: the
    frozen policy)."""
    from _common import load_run_cfg  # function-level: _common imports us too

    raw = load_run_cfg(ref_path, "tuner").get("inner_policy", POLICY_ID)
    policy_id = str(raw)
    if policy_id not in (POLICY_ID, LEGACY_POLICY_ID):
        raise ValueError(
            f"tuner.inner_policy must be {POLICY_ID!r} or {LEGACY_POLICY_ID!r}; "
            f"got {raw!r}"
        )
    return policy_id


def regime_for_bout_index(bout_index: int) -> str:
    """Regime of the bout ABOUT TO RUN (0-based)."""
    if bout_index <= 0:
        return FIRST
    if bout_index == 1:
        return CONTINUE
    return DEEP


def bout_size(regime: str) -> int:
    return _REGIME_BOUT_SIZES[regime]


def expected_bout_trials(policy_id: str, bout_index: int, legacy_bout_trials: int) -> int:
    """The bout's full trial budget under the policy.

    The legacy policy sizes every bout by the run's ``tuner.bout_trials``.
    """
    if policy_id == LEGACY_POLICY_ID:
        return int(legacy_bout_trials)
    return bout_size(regime_for_bout_index(bout_index))


def method_chain_for_bout(policy_id: str, bout_index: int, search_space: dict) -> list:
    """The deterministic method chain for one bout.

    FIRST -> ["bo"] (driven with an explicit RandomSampler; see
    :func:`bo_sampler_for_bout`). CONTINUE -> ["hebo"] (prompt-v2 LLM pool
    + official HEBO MACE; no TPE/cmaes fallback). DEEP -> ["spsa"].
    """
    from tune_tools import select_method  # function-level: tune_tools imports us

    selected = select_method(len(search_space))
    legacy = [selected["method"], *selected["fallback"]]
    if policy_id != POLICY_ID:
        return legacy
    regime = regime_for_bout_index(bout_index)
    if regime == FIRST:
        return ["bo"]
    if regime == DEEP:
        return ["spsa"]
    return ["hebo"]


def bo_sampler_for_bout(policy_id: str, bout_index: int) -> str:
    """Which Optuna sampler drives a ``bo`` stage: FIRST bouts are explicit
    random draws; every other bo stage keeps multivariate TPE."""
    if policy_id == POLICY_ID and regime_for_bout_index(bout_index) == FIRST:
        return "random"
    return "tpe"


def deferred_occupy_bout_slots(policy_id: str) -> bool:
    """Whether deferred-warm configs consume slots inside the bout budget
    (new policy, design §2 rule 4) instead of extending it (legacy)."""
    return policy_id == POLICY_ID


def rewarm_allowed(policy_id: str, bout_index: int) -> bool:
    """LLM re-warm proposals exist only for CONTINUE bouts under the
    *legacy* policy. FIRST never had them; a DEEP bout must form complete
    SPSA pairs (a proposal displacing one leg would break the pair); the
    HEBO CONTINUE arm generates its own pool, so Phase-R proposals would
    only displace that protocol.
    """
    if policy_id != POLICY_ID:
        return bout_index >= 1
    return False


def has_movable_continuous(search_space: dict) -> bool:
    """At least one non-degenerate float dimension (SPSA's eligibility rule).

    Integer and categorical dimensions never move under SPSA, and a
    degenerate float cannot move either (every perturbation projects back
    to the single legal value).
    """
    for entry in search_space.values():
        try:
            if entry[0] == "float" and float(entry[1]) < float(entry[2]):
                return True
        except (TypeError, ValueError, IndexError):
            continue
    return False


def load_movable_continuous_flags(run_dir, ledger: dict) -> dict:
    """Per-candidate has_movable_continuous, read from each candidate's
    frozen ``phase_a.search_space`` (written by warmstart_eval; the ledger
    itself carries no search-space facts).

    Both selection layers (legacy select-candidate and scheduler v3.2
    state) share this reader so they cannot disagree about who has a DEEP
    action. A candidate whose report is missing or unparseable is absent
    from the result — callers default that to True rather than strip a
    DEEP action on a read failure.
    """
    import json
    from pathlib import Path

    flags: dict[str, bool] = {}
    for record in ledger.get("records", []):
        run_id = str(record.get("run_id"))
        report_path = Path(run_dir) / "candidates" / run_id / "tune_report.json"
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        space = report.get("phase_a", {}).get("search_space")
        if isinstance(space, dict) and space:
            flags[run_id] = has_movable_continuous(space)
    return flags
