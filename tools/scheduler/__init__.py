"""Scheduler v3.2: budget allocation between TUNE(i) and DEFER.

The package is an isolated policy arm. Nothing here runs unless a run's
``framework_cfg.json`` sets ``tuner.scheduler_policy = "v3_2"``; the legacy
percentile/alternation gate in ``tools/tuners/tune_tools.py`` stays the
default so scheduler experiments compare against an unchanged baseline.

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
