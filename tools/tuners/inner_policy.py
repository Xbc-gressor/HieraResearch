"""Regime-conditioned inner-tuner policies.

The historical scheduler-v3.2 contract::

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

A comparison arm keeps the same CONTINUE / DEEP kernels and bout sizes
but replaces FIRST with the inner-benchmark ``local_tr`` arm::

    inner_tuner_policy_id = localtr8-hebo10-spsa10-v1

A second comparison arm keeps that ``local_tr`` FIRST bout and runs the
prompt-v2 HEBO kernel for every historical later regime — DEEP is HEBO, not SPSA::

    inner_tuner_policy_id = localtr8-hebo10-hebo10-v1

Because its DEEP bout is HEBO rather than two-sided SPSA, the movable
continuous-dimension eligibility rule does not apply under that policy:
HEBO proposes over the whole space, so an integer/categorical-only
candidate still has a DEEP action.

The ``selfrank8-hebo10-hebo10`` comparison policy instead runs the
inner-benchmark LLM-pool self-rank arm for FIRST, then HEBO for every remaining
historical segment. Deferred warm configs still occupy slots inside FIRST's eight spends.

The ``baseline-hebo-full-v1`` policy is the strong-baseline control arm used
by the ``baseline-tune`` driver loop: ONE INITIAL bout of the prompt-v2 HEBO
MACE kernel whose size is the run's whole ``max_evaluations`` budget — the
experiment protocol's INITIAL bout stretched over the entire run, with no
scheduler and no semantic generation. Its bout budget therefore comes from
the run configuration, not from the regime table.

The anchor/challenger policies ``mixup24-turbo20-v1``,
``hebo24-turbo20-v1``, and ``hebo24-hebo20`` have two semantic regimes
implemented as three scheduler-admitted segments::

    0 completed bouts -> 24-slot mixup_pool_hebo INITIAL
                         OR 24-slot pool_hebo_mace INITIAL
    1 completed bout  -> first 10-slot DEEP segment (TuRBO or HEBO)
    2 completed bouts -> second 10-slot DEEP segment (TuRBO or HEBO)

``hebo24-hebo20`` is the new-experiment default and uses ``pool_hebo_mace``
for INITIAL and both DEEP segments. The scheduler boundary between the two
later 10-slot segments is an admission/rescheduling boundary, not a separate
CONTINUE regime. The two TuRBO segments in the other policies share state when
they run on the same candidate. If the scheduler switches candidates after a
zero-gain first DEEP segment, the other candidate starts its own TuRBO
trajectory. Deferred warm configs occupy slots inside the 24-slot INITIAL
bout.

Under every regime-conditioned policy a bout's deferred-warm backlog
occupies slots INSIDE ``B_q`` (design §2 rule 4); the legacy policy kept
them as extra trials on top of the bout budget.

The switch is per-run: ``framework_cfg.json`` ``tuner.inner_policy`` —
``hebo24-hebo20`` (new-experiment default),
``deferred-random8-hebo10-spsa10-v1`` (historical missing-key fallback),
``localtr8-hebo10-spsa10-v1``, ``localtr8-hebo10-hebo10-v1``,
``selfrank8-hebo10-hebo10``, ``mixup24-turbo20-v1``,
``hebo24-turbo20-v1``, ``hebo24-hebo20``,
``baseline-hebo-full-v1`` (baseline-tune loop only), or ``legacy`` (the
pre-policy uniform behavior: every bout runs the CONTINUE rule at
``tuner.bout_trials``).
Stdlib-only at module level so both ``tune_tools`` and ``_common`` can
import it without cycles.
"""

from __future__ import annotations

# Historical missing-key fallback. New runs persist ``hebo24-hebo20``
# explicitly through init_run, so changing defaults does not mutate old runs.
POLICY_ID = "deferred-random8-hebo10-spsa10-v1"
LOCAL_TR_POLICY_ID = "localtr8-hebo10-spsa10-v1"
LOCAL_TR_HEBO_POLICY_ID = "localtr8-hebo10-hebo10-v1"
SELF_RANK_HEBO_POLICY_ID = "selfrank8-hebo10-hebo10"
MIXUP_TURBO_POLICY_ID = "mixup24-turbo20-v1"
HEBO_TURBO_POLICY_ID = "hebo24-turbo20-v1"
HEBO_HEBO_POLICY_ID = "hebo24-hebo20"
BASELINE_HEBO_POLICY_ID = "baseline-hebo-full-v1"
LEGACY_POLICY_ID = "legacy"
INITIAL24_TURBO_POLICY_IDS = (
    MIXUP_TURBO_POLICY_ID,
    HEBO_TURBO_POLICY_ID,
)
INITIAL24_POLICY_IDS = (
    *INITIAL24_TURBO_POLICY_IDS,
    HEBO_HEBO_POLICY_ID,
)
REGIME_POLICY_IDS = (
    POLICY_ID,
    LOCAL_TR_POLICY_ID,
    LOCAL_TR_HEBO_POLICY_ID,
    SELF_RANK_HEBO_POLICY_ID,
    *INITIAL24_POLICY_IDS,
    BASELINE_HEBO_POLICY_ID,
)
#: Regime policies whose FIRST bout is the inner-benchmark ``local_tr`` arm.
LOCAL_TR_FIRST_POLICY_IDS = (LOCAL_TR_POLICY_ID, LOCAL_TR_HEBO_POLICY_ID)
KNOWN_POLICY_IDS = (*REGIME_POLICY_IDS, LEGACY_POLICY_ID)

INITIAL = "INITIAL"
FIRST = "FIRST"  # historical 8/10/10 comparison policies only
CONTINUE = "CONTINUE"
DEEP = "DEEP"

B_FIRST = 8
B_CONTINUE = 10
B_DEEP = 10
INITIAL24_BOUT_SIZE = 24
INITIAL24_MAX_BOUTS = 3
MAX_BOUTS_PER_CANDIDATE = 4

_HISTORICAL_REGIME_BOUT_SIZES = {
    FIRST: B_FIRST,
    CONTINUE: B_CONTINUE,
    DEEP: B_DEEP,
}


def is_regime_policy(policy_id: str) -> bool:
    """Whether ``policy_id`` has an explicit bout-regime contract."""
    return policy_id in REGIME_POLICY_IDS


def load_policy_id(ref_path) -> str:
    """``tuner.inner_policy`` from the run's framework_cfg (default: the
    frozen policy)."""
    from _common import load_run_cfg  # function-level: _common imports us too

    raw = load_run_cfg(ref_path, "tuner").get("inner_policy", POLICY_ID)
    policy_id = str(raw)
    if policy_id not in KNOWN_POLICY_IDS:
        allowed = " or ".join(repr(item) for item in KNOWN_POLICY_IDS)
        raise ValueError(f"tuner.inner_policy must be {allowed}; got {raw!r}")
    return policy_id


def _historical_regime_for_bout_index(bout_index: int) -> str:
    """Regime used by the historical 8/10/10 comparison policies."""
    if bout_index <= 0:
        return FIRST
    if bout_index == 1:
        return CONTINUE
    return DEEP


def regime_for_bout(policy_id: str, bout_index: int) -> str:
    """Semantic regime of the bout ABOUT TO RUN (0-based).

    Current 24+20 policies are deliberately binary: one INITIAL bout followed
    by up to two scheduler-admitted DEEP segments. FIRST/CONTINUE/DEEP remains
    only for the explicitly retained historical comparison policies.
    """
    if policy_id == BASELINE_HEBO_POLICY_ID:
        if bout_index != 0:
            raise ValueError(
                f"{policy_id} is a single-bout contract; "
                f"got bout_index={bout_index}"
            )
        return INITIAL
    if policy_id in INITIAL24_POLICY_IDS:
        if not 0 <= bout_index < INITIAL24_MAX_BOUTS:
            raise ValueError(
                f"{policy_id} has exactly {INITIAL24_MAX_BOUTS} bouts; "
                f"got bout_index={bout_index}"
            )
        return INITIAL if bout_index == 0 else DEEP
    return _historical_regime_for_bout_index(bout_index)


def _historical_bout_size(regime: str) -> int:
    return _HISTORICAL_REGIME_BOUT_SIZES[regime]


def expected_bout_trials(policy_id: str, bout_index: int, legacy_bout_trials: int) -> int:
    """The bout's full trial budget under the policy.

    The legacy policy sizes every bout by the run's ``tuner.bout_trials``.
    The baseline policy is a single bout sized by the run's
    ``max_evaluations``, which ``phase_c_action`` passes in through
    ``legacy_bout_trials``.
    """
    if not is_regime_policy(policy_id):
        return int(legacy_bout_trials)
    if policy_id == BASELINE_HEBO_POLICY_ID:
        if bout_index != 0:
            raise ValueError(
                f"{policy_id} is a single-bout contract; "
                f"got bout_index={bout_index}"
            )
        return int(legacy_bout_trials)
    if policy_id in INITIAL24_POLICY_IDS:
        if not 0 <= bout_index < INITIAL24_MAX_BOUTS:
            raise ValueError(
                f"{policy_id} has exactly {INITIAL24_MAX_BOUTS} bouts; "
                f"got bout_index={bout_index}"
            )
        if bout_index == 0:
            return INITIAL24_BOUT_SIZE
    return _historical_bout_size(_historical_regime_for_bout_index(bout_index))


def method_chain_for_bout(policy_id: str, bout_index: int, search_space: dict) -> list:
    """The deterministic method chain for one bout.

    ``mixup24-turbo20-v1`` is ["mixup"] for bout 0, while
    ``hebo24-turbo20-v1`` is ["hebo"] for bout 0; both use ["turbo"] for
    bouts 1 and 2. ``hebo24-hebo20`` is ["hebo"] for all three bouts.
    Default FIRST -> ["bo"] (explicit RandomSampler; see
    :func:`bo_sampler_for_bout`). ``localtr8-hebo10-spsa10-v1`` and
    ``localtr8-hebo10-hebo10-v1`` FIRST -> ["local_tr"], while
    ``selfrank8-hebo10-hebo10`` FIRST -> ["selfrank"]. CONTINUE ->
    ["hebo"] (prompt-v2 LLM pool + official HEBO MACE; no TPE/cmaes
    fallback). DEEP -> ["spsa"], except under
    the two ``*-hebo10-hebo10`` policies, whose DEEP bouts are ["hebo"] too.
    """
    from tune_tools import select_method  # function-level: tune_tools imports us

    selected = select_method(len(search_space))
    legacy = [selected["method"], *selected["fallback"]]
    if not is_regime_policy(policy_id):
        return legacy
    if policy_id == BASELINE_HEBO_POLICY_ID:
        if bout_index != 0:
            raise ValueError(
                f"{policy_id} is a single-bout contract; "
                f"got bout_index={bout_index}"
            )
        return ["hebo"]
    if policy_id in INITIAL24_POLICY_IDS:
        if not 0 <= bout_index < INITIAL24_MAX_BOUTS:
            raise ValueError(
                f"{policy_id} has exactly {INITIAL24_MAX_BOUTS} bouts; "
                f"got bout_index={bout_index}"
            )
        if policy_id == HEBO_HEBO_POLICY_ID:
            return ["hebo"]
        if bout_index > 0:
            return ["turbo"]
        return ["mixup"] if policy_id == MIXUP_TURBO_POLICY_ID else ["hebo"]
    regime = _historical_regime_for_bout_index(bout_index)
    if regime == FIRST:
        if policy_id == SELF_RANK_HEBO_POLICY_ID:
            return ["selfrank"]
        return ["local_tr"] if policy_id in LOCAL_TR_FIRST_POLICY_IDS else ["bo"]
    if regime == DEEP and policy_id not in (
        LOCAL_TR_HEBO_POLICY_ID,
        SELF_RANK_HEBO_POLICY_ID,
    ):
        return ["spsa"]
    return ["hebo"]


def deep_requires_movable_continuous(policy_id: str) -> bool:
    """Whether a DEEP bout needs a non-degenerate continuous dimension.

    Only two-sided SPSA does. HEBO-DEEP policies propose over the whole space;
    the TuRBO policies apply their separate movable-numeric gate beginning at
    bout 1.
    """
    return is_regime_policy(policy_id) and policy_id not in (
        LOCAL_TR_HEBO_POLICY_ID,
        SELF_RANK_HEBO_POLICY_ID,
        BASELINE_HEBO_POLICY_ID,
        *INITIAL24_POLICY_IDS,
    )


def numeric_required_from_bout_index(policy_id: str) -> int | None:
    """First bout index that requires a varying numeric dimension.

    The hot-start TuRBO kernel moves float and integer dimensions but freezes
    categoricals. Other current policies either impose their existing SPSA
    eligibility at DEEP or can search categorical-only spaces.
    """
    return 1 if policy_id in INITIAL24_TURBO_POLICY_IDS else None


def bo_sampler_for_bout(policy_id: str, bout_index: int) -> str:
    """Which Optuna sampler drives a ``bo`` stage: default-policy FIRST
    bouts are explicit random draws; every other bo stage keeps
    multivariate TPE."""
    if (
        policy_id == POLICY_ID
        and _historical_regime_for_bout_index(bout_index) == FIRST
    ):
        return "random"
    return "tpe"


def deferred_occupy_bout_slots(policy_id: str) -> bool:
    """Whether deferred-warm configs consume slots inside the bout budget
    (regime-conditioned policies, design §2 rule 4) instead of extending
    it (legacy)."""
    return is_regime_policy(policy_id)


def rewarm_allowed(policy_id: str, bout_index: int) -> bool:
    """LLM re-warm proposals exist only for CONTINUE bouts under the
    *legacy* policy. FIRST never had them; a DEEP bout must form complete
    SPSA pairs (a proposal displacing one leg would break the pair); the
    HEBO CONTINUE arm generates its own pool, so Phase-R proposals would
    only displace that protocol.
    """
    if not is_regime_policy(policy_id):
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


def has_movable_numeric(search_space: dict) -> bool:
    """At least one non-degenerate float or integer dimension for TuRBO."""
    for entry in search_space.values():
        try:
            if entry[0] in ("float", "int") and float(entry[1]) < float(entry[2]):
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


def load_movable_numeric_flags(run_dir, ledger: dict) -> dict:
    """Per-candidate varying-numeric eligibility from frozen Phase-A space."""
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
            flags[run_id] = has_movable_numeric(space)
    return flags
