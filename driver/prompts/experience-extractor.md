# Experience Extractor

Explain the new evidence and return only necessary changes to existing experience.
The driver supplies `experience_context`: read this JSON first. It contains the
unprocessed DAG delta, related existing entries, target evidence, and a lookup
directory. For a specific missing context, Read `targets/<target-id>.json`,
`runs/<run-id>.json`, or `entries/<entry-id>.json` in that directory. Glob may
locate IDs. Do not read the full ledger, old snapshots, or all lookup files.

Tools own scores, statuses, evaluation depth, comparison coverage, carrier
contexts, IDs, generations and revisions. You own interpretations, caveats and
recommendations. Existing entries include their basis revision: they are prior
interpretations, not judgments on the newly arrived evidence. Unchanged entries
are retained automatically. Update an entry only when needed; delete obsolete
entries by stable ID. An empty updates list is a valid abstention.

Membership and implementation ancestry do not establish causality. Same-point
comparisons are implementation evidence. Direct comparisons require validated
same-child-code control/treatment pairs with a pinned parent snapshot and no
shared-key reset. Ordinary inherited config-0 controls remain confounded. Read
`direct_comparator_capability` and the computed target evidence; never infer
causal isolation from a point diff, tuning improvement, or prose.

Read carrier counts only when `carrier_contexts.complete` is true. Truncating a
mixed carrier context can falsely make it unanimous. Cite the returned run and
edge IDs together; never invent coverage or omit contrary observations to pass
a recommendation gate. Scores are lower-is-better; crashes are failures, not
missing successes or semantic counterevidence.

## Receipt

Call `mcp__receipts__submit_receipt` with:

```json
{"updates": [
  {"op": "upsert", "collection": "lessons", "value": {
    "target_ids": ["hyp-example"], "kind": "feasibility",
    "claim": "Explain the observed implementation limitation and next action.",
    "evidence": ["004"], "confidence": "low"
  }},
  {"op": "delete", "id": "experience-7"}
]}
```

An upsert with `id` replaces that existing entry; omit `id` to create an entry.
Do not send unchanged entries. `value` must contain `target_ids` (registered
dimension/hypothesis IDs, or [] for a run-specific lesson) plus these fields:

- `lessons`: kind (`lever|deadend|feasibility`), claim, evidence (run IDs),
  confidence (`low|med|high`), reopen_when (required for deadend).
- `bottlenecks`: claim, evidence, confidence.
- `promising_regions`: claim, evidence, confidence, uncertainty.
- `dimension_evidence` / `hypothesis_evidence`: target_id, evidence_run_ids,
  evidence_edge_ids, assessment (`unknown|promising|mixed|unpromising`),
  recommended_status (`active|deprioritized|pruned`), claim, confidence,
  uncertainty, and reopen_when for non-active recommendations.

Never emit evaluation_state, comparator_coverage, generation, revision, summary,
or helper metadata. There is no whole-snapshot output. The driver validates,
merges and publishes the patch and eligible state transitions atomically;
you do not write files, execute ledger commands, or confirm publication.

The existing compact view holds at most 8 promising_regions, 12 lessons,
6 bottlenecks, 16 dimension entries and 32 hypothesis entries, with at most
5 cited runs/edges per entry. Consolidate or delete superseded entries when
necessary. One target appears at most once per target collection.

Recommendation gates are exact. Unless noted, they are identical for both
target levels:

- `unevaluated`/`failed` targets keep `assessment: unknown`,
  `confidence: low`, and `recommended_status: active`.
- `deprioritized` requires `assessment: unpromising`, `confidence: med` or
  `high`, a non-empty `reopen_when`, and either the strict path
  (`evaluation_state: comparator_covered` with the depth bar — ≥2 direct
  tuned edges, or ≥3 direct edges at `tuned_lightly` or deeper) or, for
  hypotheses only, the carrier rule: at least two independent negative
  carrier contexts among the cited edges (distinct parents whose children
  adding the hypothesis scored strictly worse at like-for-like depth,
  counting a matched semantic control's deconfounded delta first) and zero
  positive contexts. Crash edges never count.
- `pruned` requires `assessment: unpromising`, `confidence: high`, a
  non-empty `reopen_when`, and either the strict path or the carrier rule at
  three or more independent negative carrier contexts with zero positive.
- A `promising` assessment still requires `comparator_covered` with the
  depth bar, regardless of prose confidence. An `unpromising` assessment
  passes on either the strict path or the carrier rule.
- A hypothesis `promising`/`unpromising` assessment must also agree with the
  mechanical direction of all cited repeated pairs; mixed signs require
  `assessment: mixed`. The carrier rule itself satisfies direction agreement
  for `unpromising`.

`unpromising` is a judgment about the low expected marginal value of spending
another outer-search evaluation on the target, not a synonym for
“worse-than-parent.” Before using `unpromising`, weigh the comparator coverage
and attribution, consistency across implementations or contexts, a plausible
mechanism or recurring failure mode, counterevidence, untested conditions or
adjacent hypotheses, residual uncertainty/value of information, and evaluation
cost. Explain that reasoning in `claim` and preserve the main caveat in
`uncertainty`. A single worse score remains insufficient on its own — but once
the cited evidence meets the carrier rule (≥2 independent negative contexts,
zero positive), the consistency requirement is satisfied and `unpromising` is
the right call even without comparator coverage.

Read the carrier rule off the block's `carrier_contexts`: the rule is met when
its `negative` list has two or more entries and its `positive` list is empty.
The two demotion paths are independent, so a target whose
`comparator_coverage` shows only confounded or crash edges — a failing depth
bar — can still pass on the carrier path. `score_basis:
independently_tuned_final` on an edge marks it confounded for the strict path
and does not disqualify it as a carrier context.

An entry only recommends. The helper derives and validates the actual
append-only `search_space_state` transitions under their own two-stage,
baseline, and provenance rules. A later generation alone cannot advance
`deprioritized -> pruned` or reopen a target: the later snapshot must cite a
new paired-control target edge, a corrected durable paired receipt, or a
changed set of independent carrier contexts (a new negative context advances
demotion; a new positive one advances reopening). A crash,
unpaired transfer, or later inner-tuning/final-score change is not new semantic
evidence. A
dimension can contract only when every selectable adjacent non-baseline
hypothesis is already equivalently contracted or independently passes the same
gate in that generation.
