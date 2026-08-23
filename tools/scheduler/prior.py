"""Frozen design prior for the historical v3.2 comparison scheduler.

The current ``anchor_challenger_v1`` scheduler has no predictive prior.

This is an offline, versioned table — not live cross-run transfer. Every
run starts from the same records; a class switches to current-run exact
support only after that class has enough of its own observations. Mixing
live evidence across concurrent arms is still off.

Provenance (`PRIOR_ID`):

* FIRST gains: cell-eb `first/current` (production-like random kernel).
* LATER gains: cell-v2 prompt-v2 HEBO, ci+cn mixed — the distribution
  production CONTINUE actually sees. DEEP has no own sample and is not
  a separate predictive class.
* Arrival episodes: a small set of 2-candidate generations from complete
  production runs, including typical worse-than-incumbent warms, the
  occasional improving arrival, and a failed pair.

Numbers only. `evidence.py` builds the typed records so this module does
not import the models that consume it.
"""

from __future__ import annotations

PRIOR_ID = "scheduler-prior-first-later-v1"

# cell-eb cells-inner-v1 first/current, raw D = init_incumbent - final_best.
# Cost is B_FIRST=8 even though those cells ran a 10-eval budget: the
# production FIRST bout admits 8.
FIRST_GAINS: tuple[tuple[float, str], ...] = (
    (0.060897770646297644, "cell-eb first/current 0806-sn-pt125-2-005-b0"),
    (0.0014102172462431284, "cell-eb first/current 0807-exp125-1-006-b0"),
    (0.06257006900465356, "cell-eb first/current 0807-sn-pt125-0-001-b0"),
    (0.022303015014161875, "cell-eb first/current 0808-exp125-ds-1-000-b0"),
)

# cell-v2 cells-prompt-v2 pool_hebo_mace, ci+cn mixed.
LATER_GAINS: tuple[tuple[float, str], ...] = (
    (0.01610308802498084, "cell-v2 hebo ci 0806-sn-pt125-2-005-b1"),
    (0.034303822856618815, "cell-v2 hebo ci 0807-sn-pt125-0-004-b1"),
    (0.007899805163282592, "cell-v2 hebo cn 0805-v4f-pr125-1-000-b1"),
    (0.0004383456254950513, "cell-v2 hebo cn 0807-exp125-1-017-b1"),
)

# Representative 2-candidate episodes. Gaps are warm - pre-episode global
# best (eq. 7): positive = worse, None = spent slots with no usable score.
ARRIVAL_GAPS: tuple[tuple[float | None, ...], ...] = (
    (-0.0016, 0.0063),
    (-0.0360, 0.1200),
    (0.0627, 0.0613),
    (0.0311, 0.0149),
    (-0.0063, 0.0403),
    (0.0359, 0.0010),
    (0.0899, 0.0036),
    (0.0449, 0.0167),
    (0.0042, 0.0666),
    (None, None),
)
