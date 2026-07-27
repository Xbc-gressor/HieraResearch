---
paths:
  - "runs/**/ledger.json"
---

# ledger.json Rules

`ledger.json` is the immutable observation history for one run. The contract
keeps five concepts separate:

- `background.md` defines the frozen semantic search space;
- `source_run_ids` contains only numeric DAG parents;
- `semantic_point` attributes a complete candidate to that space;
- `policy_receipt` records the replaceable point-selection inputs and config;
- scores, crashes, tuning metadata, and logs are observations.

Point membership is attribution, not proof that a hypothesis caused a score.
The candidate remains a complete concrete solution and may contain
implementation details not determined by its point. Two implementations may
therefore occupy the same point.

## Never hand-edit

Only `tools/ledger.py` writes the ledger. In particular:

```bash
python tools/ledger.py add-record \
  --ledger <run_dir>/ledger.json --run-id <id> \
  --op <fresh|improve|crossover> --source-run-ids <numeric-parents> \
  --background <run_dir>/background.md \
  --semantic-point <point.json> --policy-receipt <receipt.json> \
  --idea '<complete solution>' --change '<process description>' \
  --candidate-name-hint '<name>'
```

`add-record` validates the exact background revision, every assignment,
conditions, exclusions, ancestry, and policy receipt before writing. A missing
mapping is not recoverable through `record-run`; create the validated record
first.

Other mutations remain:

- `set-tuning` for tuning metadata (`--mark-tuned` only in deep tuning);
- `record-run` for the lower-is-better score and keep/discard/crash state;
- `set-experience` for a complete validated derived snapshot;
- `set-phase` and `loop-state` for run control and the derived brief.

## Shape

```json
{
  "task": "<task>",
  "tag": "<tag>",
  "metric": "<metric>",
  "search_space": {
    "space_id": "<run-local stable id>",
    "space_revision": "sha256:<exact background digest>",
    "catalog": {
      "id": "<resolved catalog id>",
      "revision": "sha256:<catalog digest>"
    },
    "dimension_ids": ["dim-..."]
  },
  "dag_revision": 0,
  "records": [],
  "experience": {}
}
```

Every record has all fields (unavailable tuning/result fields are `null`):

| field | meaning |
|---|---|
| `run_id` | zero-padded candidate id |
| `kind` | always `optimization` |
| `op` | structural graph action: `fresh`, `improve`, or `crossover` |
| `source_run_ids` | numeric parents only: 0 / 1 / 2 for the three ops |
| `semantic_point` | complete mapping over all selected dimensions, including explicit conditional inactivity |
| `policy_receipt` | policy name/config, action, proposal-set digest, selected point, separate gain/uncertainty/cost/coverage components, evidence, and ranking |
| `idea` | self-contained complete solution, not merely a list of hypotheses |
| `change` | implementation process relative to parents; it may be non-empty even when the point is unchanged |
| `candidate_name`, `description`, `metric` | display metadata |
| `tune` | whether decoupled deep tuning ran |
| `status` | `pending`, `keep`, `discard`, or `crash` |
| `best_warm_score`, `final_best_score` | inner-HPO and final candidate observations |
| `n_dims`, `warm_start_K`, `warm_percentile` | tuning metadata, unrelated to semantic dimensions |
| `phase_b_decision`, `phase_c_method`, `trials_completed`, `trials_attempted`, `elapsed_seconds`, `applied` | tuning process metadata; completed counts finite scores, attempted counts every config→score call and owns the run budget |
| `dag_revision` | helper-owned graph-change cursor |

The semantic point contains the exact `space_revision`, a stable content-based
`point_id`, and one assignment for every selected dimension in registry order.
An active dimension selects exactly one local `hyp-*`; a conditionally inactive
dimension says `state: inactive` and cites its unsatisfied activation relations.

The policy receipt is derived and replaceable. `predicted_gain`, `uncertainty`,
`cost`, and deterministic `coverage` remain separate fields. These rubric
scores are not calibrated posteriors and are never copied into observations.
The outer graph policy remains in `got_select`; the receipt concerns only the
semantic point chosen after that graph action.

## Score and crash semantics

Scores are always lower-is-better. `keep` means a finite
`final_best_score` is strictly lower than the best previous keep. A non-finite
or missing result is a `crash` with `+inf`, never a missing success or ordinary
discard.

## Experience boundary

Raw records are the durable history. `experience` (schema 3) is a bounded
regenerated belief snapshot: `summary`, `promising_regions`, `lessons`,
`bottlenecks`, plus two bounded target collections, `dimension_evidence`
(at most 16 entries) and `hypothesis_evidence` (at most 32 entries), each
target appearing at most once per collection. Target evidence is replaceable
derived belief, never durable evidence: the validator recomputes its
comparator counts and evaluation state from the cited ids, and it never
mutates the frozen registry. Cited `evidence_edge_ids` name persisted
semantic receipts; they are attribution evidence, and comparator coverage
does not prove causality.

Each target entry carries `target_id`, `evaluation_state`, `assessment`,
`recommended_status`, `claim`, `evidence_run_ids` (0–5 unique terminal
target-related runs), `evidence_edge_ids` (0–5 unique persisted
target-touching edges), `comparator_coverage`, `confidence`, `uncertainty`,
and optional `reopen_when`. `evaluation_state` is mechanical: `unevaluated`
(no cited terminal runs or edges), `failed` (cited evidence is crash-only),
`observed` (a non-crash observation but fewer than two direct non-crash
edges), or `comparator_covered` (at least two direct non-crash edges).
`assessment` is `unknown`, `promising`, `mixed`, or `unpromising`;
`recommended_status` is `active`, `deprioritized`, or `pruned`; `confidence`
is `low`, `med`, or `high`.

Recommendation gates are exact and identical for both levels:

- `unevaluated`/`failed` targets keep `assessment: unknown`,
  `confidence: low`, and `recommended_status: active`; a crash alone never
  contradicts a semantic element.
- `deprioritized` requires `assessment: unpromising`, `confidence: med` or
  `high`, `evaluation_state: comparator_covered`, at least two direct
  non-crash edges, and a non-empty `reopen_when`.
- `pruned` requires `assessment: unpromising`, `confidence: high`,
  `evaluation_state: comparator_covered`, at least two direct non-crash
  edges, and a non-empty `reopen_when`.
- High-confidence `promising` or `unpromising` claims require
  `comparator_covered`.

`unpromising` means that another outer-search evaluation has low expected
marginal value after considering attribution, consistency, mechanism,
counterevidence, untested variants, residual uncertainty/value of information,
and cost. A worse final score or score delta alone is never sufficient; when
attribution is weak or relevant variants remain, use `mixed` and keep the
target active.

Belief recommendations stay separate from runtime eligibility: an entry only
recommends. Actual `deprioritized`/`pruned` transitions are append-only
`search_space_state` decisions with their own two-stage, baseline, and
provenance rules. Second-stage pruning or reopening also requires a later
snapshot with changed evidence for that target, not merely a newer generation
or an unrelated DAG update. Dimension contraction requires every selectable
adjacent non-baseline hypothesis to be equivalently contracted or independently
gate-qualified in the same generation. The experience extractor may use mechanically rendered
point coverage, parent diffs, and bounded target evidence as context, but it
must not rewrite the space or present membership as causal support.

Runtime `deprioritized` is a real semantic-admission budget lane. Policy
receipt schema 3 records the one-based selection index, configured
`deprioritized_budget_interval`, scheduled and selected lanes, fallback, and
pre-lane rank. Every Nth admission is reserved for the deprioritized lane
(default 5, or 20%); other admissions are active-lane only. A lane may be
crossed only when it is empty, and that deterministic fallback must be
recorded.

Validate before storing a snapshot:

```bash
python tools/background_contract.py validate-experience \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json \
  --experience <experience.json>
python tools/ledger.py set-experience \
  --ledger <run_dir>/ledger.json --background <run_dir>/background.md \
  --from-json <experience.json>
```

`ledger.dag_revision` advances only when a result becomes graph-visible or an
existing result changes. Incremental DAG rendering and bounded Top/Bottom
anchors remain unchanged.
