"""Scheduler v3.2: budget allocation between TUNE(i) and DEFER.

The package is an isolated policy arm. New runs persist
``tuner.scheduler_policy = "v3_2"`` during initialization; ``legacy`` remains
an explicit CLI-selectable comparison arm. Historical runs that omit the key
retain the legacy percentile/alternation fallback in
``tools/tuners/tune_tools.py`` rather than changing behavior on resume.

Layering follows the design's central principle — *mechanical state exact,
statistical model coarse*:

``contract``   frozen resource contract (B, MAX_BOUTS), admission and
               eligibility predicates shared by policy, simulator, and the
               real execution layer.
``state``      exact mechanical state of one decision, plus its immutable
               snapshot encoding.
``evidence``   current-run empirical evidence derived from the snapshot
               chain: FIRST/LATER tuning transitions and ordered arrival
               episodes, and the plug-in models over them.
``rollout``    full-remaining-budget paired rollout under common random
               numbers keyed by future-event identity.
``policy``     the decision itself: coverage gate, rollout comparison,
               frozen tie rule.
``store``      append-only artifact store for snapshots, decisions, and
               realized outcomes.
"""
