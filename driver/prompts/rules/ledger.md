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

- `set-tuning` for Phase-A tuning metadata;
- `finalize_tuning.py` for a completed tuning-bout close: it validates
  terminal Phase C for the current bout, applies the global best (warm plus
  every Phase-C trial across all bouts), and writes score/status/tuning
  metadata (`tuning_bouts`, `last_bout_improved`, graded `evaluation_depth`)
  together. Under the current policy, a finalized candidate stays eligible
  for another DEEP segment.
  `set-tuning --mark-tuned` is disabled, so there is no second close path;
- `record-run` for the lower-is-better score and keep/discard/crash state;
- `set-experience` for a complete validated derived snapshot;
- `set-phase` and `loop-state` for run control and the derived brief.

Each parameter-transfer receipt captures its exact parent record in the
append-only `lineage_snapshots` collection. A parent with an in-flight,
unbound child (or a scored child with a broken binding) cannot change; a
no-binding terminal `crash`/`unevaluated` child has no revision to preserve.
After a valid binding is captured, later tuning creates a new current revision
while old children remain bound to their historical one.

## Shape

```json
{
  "task": "<task>",
  "tag": "<tag>",
  "metric": "<metric>",
  "direct_comparator_capability": {
    "schema_version": 1,
    "status": "unavailable",
    "reason": "no_production_same_child_code_control_treatment_evaluator"
  },
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
| `kind` | always `optimization`; a provided baseline is an ordinary first `fresh` root, not a separate record species |
| `op` | structural graph action: `fresh`, `improve`, or `crossover` |
| `source_run_ids` | numeric parents only: 0 / 1 / 2 for the three ops |
| `semantic_point` | complete mapping over all selected dimensions, including explicit conditional inactivity |
| `policy_receipt` | policy name/config, action, proposal-set digest, selected point, separate prior/experience-adjusted gain and uncertainty plus cost/coverage components; under a carrier policy the deterministic carrier prior and per-hypothesis context counts; under an attempt policy the non-positive `attempt_prior` and its `attempts` outcome partition, base rate, and contributing run ids; plus experience receipt, evidence, and ranking |
| `route_provenance` | planned route provenance when the route arm is on: the exact memory rows shown, the route sketches, their preference order, and the chosen route. Planned only — it never claims what the writer actually built. A candidate nobody planned (the task-provided baseline) records `status: not_applicable` with a reason. `null` when the arm is off |
| `parameter_transfer` | for non-fresh candidates, the full primary-parent incumbent projection, semantic-control qualification, mandatory warm config-0 pointer, and scored control rows |
| `applied_incumbent` | exact applied params/schema plus candidate/report hashes represented by this record's score; descendants bind to this durable snapshot |
| `idea` | self-contained complete solution, not merely a list of hypotheses |
| `change` | implementation process relative to parents; it may be non-empty even when the point is unchanged |
| `candidate_name`, `description`, `metric` | display metadata |
| `tune` | derived bool: set true at every bout close (the candidate completed at least one tuning bout); independent of `evaluation_depth`, which is computed separately from cumulative Phase-C attempts — a zero-objective-attempt bout closes with `tune: true` and depth `screening` |
| `tuning_bouts` | completed progressive-tuning bouts; 0 for screening-only or legacy untuned records, 1 for legacy one-shot-tuned records |
| `last_bout_improved` | whether the last bout produced a trial strictly better than its pre-bout incumbent; null when unknown/never tuned |
| `status` | `pending`, `keep`, `discard`, `crash`, or evidence-neutral terminal `unevaluated` |
| `unevaluated_receipt` | helper-owned exhausted-budget/zero-attempt proof; present only for `unevaluated` |
| `best_warm_score`, `final_best_score` | inner-HPO and final candidate observations |
| `n_dims`, `warm_start_K`, `warm_percentile` | tuning metadata, unrelated to semantic dimensions |
| `phase_b_decision`, `phase_c_method`, `trials_completed`, `trials_attempted`, `elapsed_seconds`, `applied` | objective/tuning metadata; completed counts finite scores and attempted counts every admitted config→score call |
| `preflight_attempts`, `preflight_failures`, `feasibility_rejections` | no-score engineering checks; auditable but excluded from the objective-call budget |
| `dag_revision` | helper-owned graph-change cursor |

The semantic point contains the exact `space_revision`, a stable content-based
`point_id`, and one assignment for every selected dimension in registry order.
An active dimension selects exactly one local `hyp-*`; a conditionally inactive
dimension says `state: inactive` and cites its unsatisfied activation relations.

The policy receipt is derived and replaceable. Schema 6 keeps `prior_gain`,
`experience_gain_adjustment`, final `predicted_gain`, `prior_uncertainty`,
`experience_uncertainty_adjustment`, final `uncertainty`, `cost`, and
deterministic `coverage` separate. It also records the configured
`llm_intelligence_score` and derived `llm_judgment_weight`; this heuristic
prior scales the complete LLM-authored acquisition term, never the raw
prediction or deterministic coverage, and is not a calibrated probability.
The first schema-6/7 admission freezes the configured score for the run.
`direct_comparator_capability` is the production gate for the paired branch.
It is currently `unavailable`: no runtime evaluator may stamp a same-child-code
semantic control/treatment pair, so direct coverage, mechanical gain direction,
and comparator-backed uncertainty sharpening cannot fire. Synthetic fixtures
exercise that downstream contract without claiming the runtime produces it.
Its compact `experience` receipt pins the
generation/revision and exact helper-derived target conditioning, including
the proposal relation, comparator coverage, evidence ids, acquisition role,
and mechanical gain direction. Its run/edge citations equal the complete union
of the named target receipts. Generic experience prose is display-only and
never enters this receipt. Exact-zero abstention is always legal. Confounded
evidence can only preserve or raise uncertainty; signed gain requires at least
two directionally consistent same-child-code control/treatment pairs. A
parameter-inheritance control alone is uncertainty-only. Historical
schema-2/3/4/5 receipts remain readable but cannot become direct
retroactively.

## Score and crash semantics

Scores are always lower-is-better. In a non-fresh screen, inherited config 0 is
a parameter-continuity observation: it consumes and records an objective call
and participates normally in `best_warm_params`, `best_warm_score`,
`final_best_score`, and `BASE_PARAMS` selection. Its role affects attribution,
not optimization eligibility; by itself it remains uncertainty-only semantic
evidence. `K_eval` is at least 2 so the inherited point is screened against at
least one alternative. `keep` means a finite
`final_best_score` is strictly lower than the best previous keep; a calibrated
noise floor still requires independent replicate evidence. A non-finite or
missing result is a `crash` with `+inf`, never a missing success or ordinary
discard. If the strict cap is exhausted before a pending candidate owns an
objective attempt, `ledger.py resolve-unevaluated` stores `status:
unevaluated`, no score, and an accounting receipt. This resolves
lifecycle only; it creates no semantic observation.

`evaluation_attempts.jsonl` is the helper-owned append-only reservation log for
the strict run cap. A slot is appended immediately before entering `score_fn`;
failed score calls still consume it, while task-owned preflight never does.
`ledger.py brief/evaluations` reconciles the log with backward-readable
per-record aggregates. Never hand-edit or truncate the reservation log.

## Experience boundary

Raw records are the durable history. `experience` (schema 4; schema 3 stays readable) is a bounded
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
`observed` (a non-crash observation but fewer than two direct
edges at `tuned_lightly` or deeper), or `comparator_covered` (at least two
direct edges at `tuned_lightly` or deeper). A direct
edge is not merely a one-dimension final-vs-final or inherited-parameter
comparison: it requires a validated same-child-code control/treatment pair
whose configs differ only in the declared semantic switch, the pinned parent
snapshot, and no shared-key reset.
`evaluation_depth` is graded by cumulative Phase-C objective attempts:
`screening` (0), `tuned_lightly` (1 to `tuner.tuned_threshold`−1, default 19),
or `tuned` (≥ threshold, default 20). A direct edge is **tuned** when the
child record's `evaluation_depth` is `tuned` and **lightly tuned** at
`tuned_lightly`; screening-depth and legacy direct edges remain evidence but
cannot drive contradiction gates. Contradiction-grade transitions require at
least two direct tuned edges or at least three direct edges at
`tuned_lightly` or deeper.
Ordinary schema-2 transfers declare the
semantic pair `unverified`, so they remain confounded. Legacy,
multi-dimension, reset-bearing, unpaired, or independently tuned comparisons
are also confounded.
`assessment` is `unknown`, `promising`, `mixed`, or `unpromising`;
`recommended_status` is `active`, `deprioritized`, or `pruned`; `confidence`
is `low`, `med`, or `high`.

Recommendation gates are exact. Unless noted, they are identical for both
levels:

- `unevaluated`/`failed` targets keep `assessment: unknown`,
  `confidence: low`, and `recommended_status: active`; a crash alone never
  contradicts a semantic element.
- `deprioritized` requires `assessment: unpromising`, `confidence: med` or
  `high`, a non-empty `reopen_when`, and either the strict path
  (`evaluation_state: comparator_covered` with at least two direct tuned
  edges or at least three direct edges at `tuned_lightly` or deeper) or, for
  hypotheses only, the carrier rule: at least two independent negative
  carrier contexts among the cited edges (distinct parents whose children
  adding the hypothesis scored strictly worse at like-for-like depth,
  counting a matched semantic control's deconfounded delta first) and zero
  positive contexts. Crash edges never count.
- `pruned` requires `assessment: unpromising`, `confidence: high`, a
  non-empty `reopen_when`, and either the strict path or the carrier rule at
  three or more independent negative carrier contexts with zero positive.
- A `promising` claim still requires `comparator_covered` with the depth
  bar. An `unpromising` claim passes on either the strict path or the
  carrier rule.

`unpromising` means that another outer-search evaluation has low expected
marginal value after considering attribution, consistency, mechanism,
counterevidence, untested variants, residual uncertainty/value of information,
and cost. A single worse score or delta remains insufficient on its own — but
once the cited evidence meets the carrier rule (≥2 independent negative
contexts, zero positive), consistency is established and `unpromising` is
appropriate even without comparator coverage.

Belief recommendations stay separate from runtime eligibility: an entry only
recommends. Actual `deprioritized`/`pruned` transitions are append-only
`search_space_state` decisions with their own two-stage, baseline, and
provenance rules. Second-stage pruning or reopening also requires a later
snapshot with changed evidence for that target — a new paired-control edge, a
corrected paired receipt, or a changed set of independent carrier contexts —
not merely a newer generation or an unrelated DAG update. Dimension contraction
requires every selectable adjacent non-baseline hypothesis to be equivalently
contracted or independently gate-qualified in the same generation. The
experience extractor may use mechanically rendered
point coverage, parent diffs, and bounded target evidence as context, but it
must not rewrite the space or present membership as causal support.

Runtime `deprioritized` content stays eligible but is penalized in selection:
policy receipt schema 7 records the deterministic carrier prior per selected
point (`components.experience_prior` and per-hypothesis context counts), and
repeated negative carrier contexts push a point down the acquisition ranking.
Pruned content is excluded from proposals outright.

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
anchors remain unchanged. At every quiescent completed non-empty round, a
positive terminal DAG delta must be processed by `set-experience` before the
next semantic admission or final completion. A belief no-op advances only the
helper-owned DAG cursor; it does not increment `generation`.
