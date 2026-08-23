"""Run-global scheduler policies.

The package keeps each policy isolated. New experiment runs persist the
deterministic ``tuner.scheduler_policy = "anchor_challenger_v1"`` during
initialization; ``v3_2`` and ``legacy`` / ``legacy_wide`` remain explicit
CLI-selectable comparison arms (the legacy pair differ
only in how widely a first-bout non-responder is re-admitted). Historical
runs that omit the key retain the legacy percentile/alternation fallback in
``tools/tuners/tune_tools.py`` rather than changing behavior on resume.

Layering follows the design's central principle — *mechanical state exact,
statistical model coarse*:

``contract``   frozen resource contract (B, MAX_BOUTS), admission and
               eligibility predicates shared by policy, simulator, and the
               real execution layer.
``state``      exact mechanical state of one decision, plus its immutable
               snapshot encoding.
``evidence``   FIRST/LATER tuning transitions and ordered arrival
               episodes: frozen design prior until a class has enough
               current-run records of its own.
``rollout``    full-remaining-budget paired rollout under common random
               numbers keyed by future-event identity.
``policy``     the decision itself: seed-set defer while reserved fresh
               roots are still arriving, then rollout comparison and
               frozen tie rule. No sample-seeking coverage.
``store``      append-only artifact store for snapshots, decisions, and
               realized outcomes.
"""
