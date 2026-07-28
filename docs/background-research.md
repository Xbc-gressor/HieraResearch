# Background as a semantic search space (P2)

HieraResearch uses `background.md` to define the frozen outer semantic search
space for one run. It is no longer a flat list of end-to-end directions.

The contract deliberately separates:

- **space definition** — catalog dimensions, task hypotheses, conditions, and
  exclusions in `background.md`;
- **candidate attribution** — a complete `semantic_point` in each ledger
  record, plus helper-derived per-parent `semantic_edges` receipts;
- **ancestry** — numeric parent ids in `source_run_ids`;
- **observations** — scores, crashes, tuning metadata, and logs in raw records;
- **derived belief** — the bounded regenerated `ledger.experience` snapshot;
- **selection policy** — a replaceable `policy_receipt` for each chosen point;
- **runtime eligibility** — the append-only `ledger.search_space_state`
  decision overlay that filters future proposals without touching the frozen
  registry.

A candidate remains a complete concrete solution. Its point says which
hypotheses it instantiates; it does not claim that those hypotheses fully
determine the code or caused the score. Distinct implementations can occupy the
same point.

`docs/search-space.md` states this mechanism formally: a point is an
equivalence class (fiber) of implementations under the attribution map, and
freezing the run's resolved dimensions is selecting a subspace that must still
contain the optimum.

## Dimension strategies

`contracts/semantic-dimensions-v1.json` is the default source of dimension
identities. Its content-addressed receipt is printed by:

```bash
python tools/background_contract.py catalog
```

Deterministic tools resolve the source through
`framework_cfg.json.space_initialization.dimension_strategy`. The default
`catalog_subset` strategy loads the built-in catalog. `llm_induced` instead
has the background researcher create a validated
`<run_dir>/dimension_catalog.json` from the task contract before literature
retrieval. Its instructions live in `docs/dimension-induction.md` and are loaded
only for that strategy. An explicit `--catalog` path overrides the catalog
source, not the configured selection semantics.

With `catalog_subset`, the catalog is broader than a task and background
research selects a task-relevant subset. With `llm_induced`, the run-local
catalog is already the final task-specific dimension set, so the registry must
use every catalog dimension exactly once and in catalog order. Under either
strategy, every serialized run dimension must come from the resolved catalog
and repeat its stable id, definition, ownership boundary, and provenance
exactly, then add a task-specific selection reason, evidence receipts, mode,
status, baseline, and hypotheses. A gap in the resolved catalog is recorded
explicitly rather than routed into a catch-all dimension.

For `catalog_subset`, the built-in catalog applies the following ownership
table. The induced strategy follows the task-first procedure in its on-demand
guide without using this table as a candidate catalog:

| interface changed | owner |
|---|---|
| target/task decomposition | `dim-task-formulation` |
| eligible example/source membership | `dim-data-curation` |
| canonical model input for one example | `dim-input-representation` |
| derived or synthetic training examples/views | `dim-data-augmentation` |
| labels, preferences, rewards, or teacher signals | `dim-supervision` |
| learnable function topology/family | `dim-model-architecture` |
| starting parameters or trainable parameter subset | `dim-initialization-adaptation` |
| scalar training criterion | `dim-learning-objective` |
| one parameter update | `dim-optimization` |
| sequencing of data/objectives/trainability over time | `dim-training-protocol` |
| training-only evidence used to stop or choose | `dim-validation-selection` |
| aggregation of independently usable predictors | `dim-ensemble` |
| invocation of fixed predictors for raw predictions | `dim-inference` |
| raw prediction to metric-facing artifact/decision | `dim-output-postprocessing` |
| semantics-preserving compute/memory execution | `dim-resource-execution` |

Scalar learning rates, depths, batch sizes, mixture ratios, and similar settings
remain in inner HPO unless the hypothesis concerns a qualitatively different
mechanism or regime. HieraResearch graph/acquisition policy is not a candidate
dimension.

## Run-local hierarchy

The fenced `## Search space registry` JSON object has `schema_version: 3` and
`kind: semantic_search_space`. Its top-level fields are:

- stable run-local `space_id`;
- exact resolved catalog receipt;
- non-empty selected `dimensions`;
- explicit scoped `relations`;
- structured negative `guidance`;
- inspected evidence `sources`.

Every registered dimension keeps `status: active` in the frozen registry and has:

- `mode: searchable` or `baseline_only`;
- one explicit `kind: baseline` hypothesis;
- `baseline_hypothesis_id` pointing to that local baseline;
- zero or more task-specific, globally unique stable `hyp-*` values.

An optional intervention uses an identity/no-intervention baseline. Omission
means the dimension is not applicable; it does not mean its baseline was chosen.
A task-fixed material choice remains visible as a one-value `baseline_only`
dimension.

Each hypothesis preserves:

- stable id, title, claim, `status: active`, and provenance receipts;
- `kind: baseline`, `evidence_prior`, or `scope_probe`;
- claim boundary and exact five-facet scope;
- required local comparisons and reopening condition;
- literature credibility and rationale;
- testable lower-is-better expectation;
- typed source evidence links.

The frozen registry never authors pruning state. `active`, directly
`deprioritized`, and directly `excluded` eligibility are derived from typed
external guidance without deleting the registered element. Evidence-preserving
run-time pruning is implemented in the append-only `search_space_state`
overlay described below; it never appears inside the registry.

## Conditions and exclusions

Relations have stable `rel-*` ids, `status: active`, provenance, and evidence.
Three exact forms are available:

- `activates`: a `when` choice scope and `target_dimension_id`. A target with
  incoming activation relations is active when any source scope matches;
- `requires`: a `when` choice scope and a `then` allowed-choice scope;
- `excludes`: at least two choice scopes in `members`; selecting all matched
  scopes is invalid.

A choice scope is:

```json
{
  "dimension_id": "dim-...",
  "hypothesis_ids": ["hyp-..."]
}
```

Activation relations are acyclic. Conditions and exclusions apply only to the
named choices; a negative combination never bans an adjacent mechanism.

## Complete candidate points

Every new ledger record stores a content-addressed point against the exact
background revision:

```json
{
  "schema_version": 1,
  "space_id": "<background space id>",
  "space_revision": "sha256:<complete registry digest>",
  "point_id": "point-<content digest>",
  "assignments": [
    {
      "dimension_id": "dim-data-curation",
      "state": "selected",
      "hypothesis_id": "hyp-data-baseline"
    },
    {
      "dimension_id": "dim-ensemble",
      "state": "inactive",
      "activation_relation_ids": ["rel-ensemble-activation"]
    }
  ]
}
```

Assignments contain every selected dimension exactly once in registry order.
Every active dimension selects one local hypothesis. Conditional inactivity is
explicit and lists the unsatisfied incoming activation relations; a missing
field cannot mean inactive.

`source_run_ids` now contains only numeric parents: zero for `fresh`, one for
`improve`, and two distinct parents for `crossover`. Hypothesis ids never appear
there. Every non-fresh record also persists one mechanical receipt per numeric
parent in `semantic_edges` (schema 1): the helper derives the exact point
difference between the persisted parent and child points — `hypothesis_changed`,
`dimension_activated`, or `dimension_deactivated` per changed assignment, with a
`change_class` of `same_point`, `single_dimension`, or `multi_dimension` that is
an attribution-strength category, not a causal claim. Models never author these
receipts, and validation rebuilds them and requires exact persisted equality.
`background_contract.py lineage` can reconstruct the same point differences
from parents while warning that they are attribution, not causal edges.

The first ledger write preserves a top-level `search_space` receipt. Later
candidate additions, experience replacements, and preflight checks fail if the
background, catalog, mapping, ancestry, or policy receipt no longer matches.
Records without a point are rejected; pre-P1 runs are not migrated.

## Search behavior and exploration/exploitation

Structural and semantic selection are separate layers:

```text
got_select.py
  -> op + numeric parents
semantic_search.py propose
  -> bounded valid points for that action
replaceable acquisition policy
  -> one point + policy receipt
idea-generator
  -> complete concrete solution at that point
ledger.py add-record
  -> ancestry + attribution + policy receipt (still no observation)
evaluation
  -> score/crash and tuning observations
```

Proposal neighborhoods are deterministic:

- `fresh` covers under-tested baselines/interventions and bounded pairs;
- `improve` includes same-point reimplementation and one-hop neighbors;
- `crossover` includes valid parent recombinations and bounded neighbors.

Hypotheses are reusable and may participate in many points. There is no
“consumed direction” state.

Four replaceable policies are implemented:

1. `coverage` — deterministic exploration by inverse hypothesis coverage and
   exact-point novelty; no LLM score is required.
2. `gain` — predicted gain plus a small coverage term minus predicted cost.
3. `gain_uncertainty` — predicted gain plus an explicit uncertainty bonus and
   coverage, minus predicted cost.
4. `gain_uncertainty_nocost` — like `gain_uncertainty` but with no cost
   prediction at all, for settings where pre-implementation cost estimates
   are noise and only waste tokens.

For model-scored policies, each proposal receives separate `[0,1]` background
priors for gain and uncertainty, signed experience adjustments, final
`predicted_gain`/`uncertainty`, and `cost`, plus evidence strings
(`gain_uncertainty_nocost` omits `cost`). `semantic_search.py gain-context`
pins the exact bounded experience generation. Prediction schema 2 must cite
terminal runs or semantic edges carried by that experience, and a nonempty
experience must change gain or uncertainty. The deterministic helper verifies
that each final number equals its prior plus the signed adjustment.

They remain auditable rubric estimates, not calibrated Bayesian posteriors.
Policy receipt schema 4 preserves the prior, adjustment, final score,
experience revision/run citations, `coverage`, weights, proposal-set digest,
action, ranking, and evidence. These values stay in `policy_receipt`; they do
not become observations or beliefs. Historical schema-2/3 receipts remain
readable.

Runtime-deprioritized proposals occupy a separate, deterministic
semantic-admission budget lane. `deprioritized_budget_interval: N` reserves
every Nth one-based outer admission for that lane (default `N=5`, or 20%);
ordinary slots select only active proposals. Acquisition scores rank within
the scheduled lane and cannot buy a deprioritized point an active slot. When
the scheduled lane is empty, the other lane fills the slot and policy receipt
schema 4 records the selection index, interval, scheduled/selected lanes,
fallback reason, and the selected point's pre-lane acquisition rank.

Run-local configuration lives under `framework_cfg.json.semantic_search`.
`gain_uncertainty_nocost` is the code and copied-template default; `coverage` (fully
deterministic, no LLM scores) is an explicit opt-in for ablations, bootstrap
runs, or prediction-failure fallback, for example:

```json
{
  "semantic_search": {
    "policy": "coverage"
  }
}
```

Changing the acquisition policy does not change registry or history semantics.
`got_select` still owns outer graph exploration and parent selection; inner HPO
still tunes numeric parameters inside the chosen semantic point.

## Evidence, belief, and pruning

The implemented P2 loop closes over the frozen registry without mutating it:

```text
background schema 3 (immutable S)
  -> semantic point selection at search_space_state revision r
  -> candidate + persistent semantic edge receipts
  -> score/crash observations
  -> bounded experience schema 3 (replaceable belief)
  -> deterministic validated decision transition
  -> append-only search_space_state revision r+1
  -> next proposal set filters/orders against r+1
```

Every proposal set and policy receipt carries the
`search_space_state_revision` it was built against; `select` and `add-record`
reject stale revisions. Historical points remain valid at the revision where
they were selected: later pruning never rewrites a record, a point, an
observation, or a prior decision, and reopening appends a new transition.

Three state families stay distinct. External guidance is `active`,
`deprioritized`, or `excluded`. Runtime control is `active`, `deprioritized`,
or `pruned`. Belief coverage is `unevaluated`, `failed`, `observed`, or
`comparator_covered`. A crash is `+inf`, distinct from an unevaluated target,
and cannot by itself contradict or prune a semantic element. Automated pruning
is two-stage (`active -> deprioritized`, then `deprioritized -> pruned` in a
later experience generation with changed evidence for the same target), and
every recommendation is gated on mechanically recomputed evidence:
deprioritization and pruning both require `comparator_covered` with at least
two direct non-crash edges; deprioritization requires med/high confidence and
pruning requires high confidence. A later generation or unrelated DAG update
alone cannot complete the second stage or reopen a target: a new cited target
edge or changed observation on a cited target edge is required.

`unpromising` means the expected marginal value of another outer-search
evaluation is low after considering attribution, consistency across
implementations or contexts, a plausible mechanism or recurring failure mode,
counterevidence, untested conditions and adjacent hypotheses, residual
uncertainty/value of information, and cost. A worse child score or score delta
alone is never sufficient; weakly attributed or incomplete evidence remains
`mixed` and active.

Transitions never touch baselines, `baseline_only` dimensions, or externally
`excluded` hypotheses. A dimension may be deprioritized only when every
selectable non-baseline hypothesis in it is externally excluded, already
runtime-deprioritized/pruned, or independently deprioritize/prune-recommended
in the same generation. Dimension pruning requires the corresponding stronger
pruned state or recommendation, so evidence against one hypothesis cannot ban
adjacent mechanisms. A runtime-pruned dimension keeps its explicit baseline eligible:
new proposals pin the dimension to `baseline_hypothesis_id` instead of
changing point arity or the frozen `space_revision`, because the overlay
restricts which points may be proposed next — it does not redefine the space
their receipts are attributed to. `ledger.dag_revision` tracks graph-visible
score/status changes only; `search_space_state.revision` independently counts
append-only decisions.

A “dimension added/removed” edge in the roadmap is represented inside a fixed
run as the conditional receipt operations `dimension_activated` /
`dimension_deactivated`: the dimension was always registered, and the edge
records a change in its point-level activity. Changing registry membership
itself is P4 expansion and remains deferred.

The experience extractor never reconstructs comparator coverage from the
Top/Bottom graph window. Its authoritative bounded source is:

```bash
python tools/background_contract.py target-evidence \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json \
  --max-dimensions 16 --max-hypotheses 32 --max-edges-per-target 5
```

For each target it returns the exact cited `evidence_edge_ids`, per-edge
score/status observations, the mechanical `evaluation_state`, and the cited
`comparator_coverage`, disclosing bounded-view loss through
`available_comparator_coverage` and `omitted_edge_counts`. Repeated
`--target-id` selects exact known targets for a smaller follow-up view.

Admission is strictly round-serial: experience extraction and
`apply-space-state` run only at quiescent round boundaries, and
propose -> select -> `add-record` completes before any candidate
implementation starts. No extractor overlaps an in-flight candidate action;
concurrent admission is deferred until it has an explicit revision contract.

## Evidence-aware background research

The background-researcher runtime prompts keep only the mission, boundaries,
and phase order always loaded; they load the operational details on demand
from `docs/agent-resources/background-researcher/` (`retrieval.md` before the
first retrieval action, `evidence-registry.md` before registry distillation,
and `background-template.md` before writing the artifact).

The retrieval path starts from the task contract and resolved registry. For each
searchable dimension, research asks what evidence is needed to propose or
compare hypotheses and to establish relevant relations, consolidating shared
needs into many-to-many queries. It may add cross-cutting evidence **angles**—
such as problem-class baselines, failure modes, or counterevidence—that are not
claims about a particular search-space dimension. Those angles remain retrieval
intent rather than becoming new dimensions. The resulting plan is run under an
explicitly selected condition:

- `frozen`: a pinned local JSON corpus for reproducible, network-disabled work;
- `deepxiv`: optional open-world scholarly retrieval;
- `jina`: explicit live-web fallback/ablation, never an invisible default;
- runtime-native web tools: fallback only, with successful content recorded
  through `search_backends.py record-visit`.

The run-local `background_retrieval.json` uses retrieval schema 3. In addition
to backend failures, canonical deduplication, balanced selections, visits,
content depth, budgets, versions, and hashes, each query records
`target_dimension_ids` and `evidence_roles`. Queries are not dimensions: the
alignment is many-to-many, and any target ids come from the resolved registry.
Semantic roles are `hypothesis`, `baseline`, `failure_mode`, `counterevidence`,
and `relation`. `hypothesis` and `relation` queries make claims inside the
search space and must name at least one target; `baseline`, `failure_mode`,
and `counterevidence` questions about the problem class as a whole may name
none. A separate `inner_hpo_prior` query has no dimension targets and cannot
be mixed with semantic roles.

Every `searchable` dimension needs at least one grounding query or a unique
`coverage_exemption` with a rationale. `baseline_only` dimensions need no query
coverage, and novelty-only queries do not satisfy grounding coverage. The query
count is governed by evidence need and context budget, not a fixed number.

Structured planning is passed to the adapter before any backend call:

```bash
python tools/search_backends.py search \
  --manifest <run_dir>/background_retrieval.json --lane grounding \
  --query-spec '{"text":"<bounded evidence question>","target_dimension_ids":["<exact-resolved-dim-id>"],"evidence_roles":["hypothesis","counterevidence"]}' \
  --query-spec '{"text":"<problem-class comparator question>","target_dimension_ids":[],"evidence_roles":["baseline"]}' \
  --query-spec '{"text":"<numeric prior question>","target_dimension_ids":[],"evidence_roles":["inner_hpo_prior"]}' \
  --coverage-exemption '{"dimension_id":"<exact-uncovered-dim-id>","rationale":"<why applicable evidence is unavailable>"}' \
  --frozen-corpus <pinned-corpus.json>
```

The exemption argument is omitted when queries cover every searchable
dimension. Grounding has a 6000-token reading lane; novelty is a separate
2048-token lane. A novelty-only visit cannot support a registry claim.

All registry sources must have a successful substantive grounding visit
(`section`, `preview`, `full_text`, or exact fetched `page`). DeepXiv
`auto` visits record `head` as triage and then fetch up to three
query-relevant body sections, falling back to a preview only when necessary;
head/brief metadata alone never qualifies. Final background validation also
joins query targets and exemptions to the registry, rejecting unknown ids and
uncovered searchable dimensions. Search snippets, generated summaries, and
unvisited URLs are insufficient. Frozen and live conditions cannot be mixed in
one main evidence condition. Older retrieval manifests are rejected rather
than migrated silently.

## Credibility, scope, and negative guidance

Hypotheses keep the existing literature credibility label:

- `unverified`, `preliminary`, `corroborated`, `replicated`, or `contested`.

This is an external evidence stamp, not a truth score. A replicated method can
fail locally; a preliminary hypothesis can work. Run-status belief about
dimensions and hypotheses lives in the schema-3 experience snapshot and the
`search_space_state` overlay, never in this registry stamp.

Sources, hypotheses, and guidance share five exact-tag scope facets:

- `model_families`, `data_regimes`, `metrics`, `interventions`, and
  `evaluation_protocols`.

`background_contract.py` derives claim-to-hypothesis transfer as `direct`,
`partial`, `mismatch`, or `unknown`. Only `direct` guidance changes eligibility.
`caution` annotates; external `deprioritize` assigns a directly matched
hypothesis to the same limited budget lane as runtime deprioritization;
`exclude` removes it from proposal generation while preserving its identity
and receipt.

Unverified or contested negatives may only caution. Binding guidance needs
directly scoped, non-withdrawn primary empirical evidence. Exclusion additionally
requires corroborated/replicated evidence, two canonical independent direct
support sources, and directly scoped independent reproduction. Every binding
negative retains an out-of-scope `scope_probe` hypothesis so adjacent mechanisms
remain representable.

Every Markdown Pitfall or Deprioritize bullet begins with a registered `g-*`,
`task-constraint`, or `operational` marker. Free-text negative prose is
contract-invalid and never a selection input.

## Human and machine views

Before the JSON registry, `background.md` contains:

- `## Dimension coverage` — dimension, mode, explicit baseline, hypotheses,
  and selection reason;
- `## Dimensions` — every selected `dim-*` and `hyp-*` in the same hierarchy;
- `## Relations` — every `rel-*` in readable form;
- typed Pitfalls/Deprioritize sections.

The validator requires every machine id to appear in the human view. Routine
agents use the bounded renderer instead of ingesting the full registry:

```bash
python tools/background_contract.py render \
  --background <run_dir>/background.md \
  --ledger <run_dir>/ledger.json --max-hypotheses 6
```

## Commands

```bash
python tools/search_backends.py validate \
  --manifest <run_dir>/background_retrieval.json

python tools/background_contract.py catalog \
  --path <run_dir>/dimension_catalog.json  # llm_induced only

python tools/background_contract.py validate \
  --background <run_dir>/background.md \
  --retrieval-manifest <run_dir>/background_retrieval.json

python tools/background_contract.py preflight \
  --background <run_dir>/background.md [--ledger <run_dir>/ledger.json]

python tools/background_contract.py validate-point \
  --background <run_dir>/background.md --point <point.json>

python tools/background_contract.py lineage \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json \
  --compact --limit 8

python tools/background_contract.py target-evidence \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json \
  [--target-id <exact-id> ...]

python tools/background_contract.py validate-experience \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json \
  --experience <experience.json>

python tools/semantic_search.py propose \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json \
  --op <op> --parents <ids> --output <proposals.json>

python tools/semantic_search.py gain-context \
  --proposals <proposals.json> --ledger <run_dir>/ledger.json \
  --output <gain-context.json>

python tools/semantic_search.py select \
  --proposals <proposals.json> [--predictions <predictions.json>] \
  --ledger <run_dir>/ledger.json \
  --point-output <point.json> --receipt-output <policy.json>

python tools/ledger.py set-experience \
  --ledger <run_dir>/ledger.json --background <run_dir>/background.md \
  --from-json <experience.json>

python tools/ledger.py apply-space-state \
  --ledger <run_dir>/ledger.json --background <run_dir>/background.md

python tools/validate_background.py
```

Legacy `## Direction registry` files and schema 1/2 data fail with an explicit
error. The contract intentionally provides no migration path for local
disposable runs.

## Deferred boundary

Evidence-preserving pruning, two-level dimension/hypothesis belief extraction,
and persistent semantic DAG edge receipts are implemented above. Still
deferred: intermediate-log bottleneck retrieval (P3), dynamic space expansion
including registry membership changes (P4), and convergence/regret claims.
Concurrent admission is deferred until it has an explicit revision contract.
