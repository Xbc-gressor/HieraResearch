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
has the background researcher draft and validate a
`<run_dir>/dimension_catalog.json` from the task contract; the draft may
interleave with literature retrieval and be revised until the background
freezes. Its instructions live in `docs/dimension-induction.md` and are loaded
only for that strategy. An explicit catalog path is a setup/validation input;
once the background is frozen, high-frequency runtime commands such as
`background_contract.py render` and `semantic_search.py propose` consume that
frozen space and do not accept a catalog override.

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

If the task declares a provided candidate entrypoint, settle the dimension set
before inspecting its implementation under `llm_induced`, then make every local
baseline hypothesis describe that supplied candidate's actual choice on the
dimension. This makes the complete all-baselines point a faithful attribution
for the concrete control without allowing one implementation to determine the
space decomposition. Its scalar defaults remain inner-HPO settings.

That attribution is enforced through `<run_dir>/baseline_mechanisms.json`, a
schema-1 `baseline_mechanism_inventory` written alongside `background.md`: per
resolved dimension, the mechanism tags the entrypoint actually applies plus
`<file>:<line>` citations, and the entrypoint's sha256. Two checks then run in
`background_contract.py`:

- **Baseline completeness** (inventory supplied): a dimension's baseline
  `scope.interventions` must contain every mechanism its inventory lists.
- **Alternative disjointness** (always): a non-baseline hypothesis's
  `scope.interventions` must not intersect any baseline's, in its own dimension
  or another.

The second check is the operative one; the first exists so an omitted baseline
mechanism cannot hide a collision. A hypothesis proposing a mechanism the
control already applies is not a contrast: candidates attributed to it
re-implement the baseline, so their observations measure implementation noise
while the ledger records a clean single-dimension edge, and the derived belief
argues about a difference that does not exist. A variant that changes only the
*placement* or *degree* of a baseline mechanism is a distinct mechanism tag and
a distinct claim, not presence-versus-absence.

Each hypothesis preserves:

- stable id, title, claim, `status: active`, and provenance receipts;
- `kind: baseline`, `evidence_prior`, or `scope_probe`;
- claim boundary and exact five-facet scope;
- required local comparisons and reopening condition;
- literature credibility and rationale;
- testable lower-is-better expectation;
- typed source evidence links.

The frozen registry never authors pruning state. Typed external guidance
derives `active` or `deprioritized` standing without deleting the registered
element. Evidence-preserving run-time pruning is implemented in the
append-only `search_space_state` overlay described below; it never appears
inside the registry.

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
candidate additions and experience replacements fail if the mapping, ancestry,
state revision, experience revision, or policy receipt no longer matches.
Records without a point are rejected; pre-P1 runs are not migrated. Full
background, catalog, retrieval, and human-view validation belongs to setup and
the process-level resume boundary rather than each candidate-selection round.

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

Five replaceable policies are implemented:

1. `coverage_experience` — the default: deterministic coverage plus the
   carrier prior, a per-hypothesis count of independent ledger contexts where
   adding the hypothesis made its parent strictly worse (penalty) or better
   (smaller bonus). No LLM score is required.
2. `coverage` — deterministic exploration by inverse hypothesis coverage and
   exact-point novelty; no LLM score is required.
3. `gain` — predicted gain plus a small coverage term minus predicted cost.
4. `gain_uncertainty` — predicted gain plus an explicit uncertainty bonus and
   coverage, minus predicted cost.
5. `gain_uncertainty_nocost` — like `gain_uncertainty` but with no cost
   prediction at all, for settings where pre-implementation cost estimates
   are noise and only waste tokens.

The last three are currently dormant.

For model-scored policies, each proposal receives separate `[0,1]` background
priors for gain and uncertainty, signed experience adjustments, final
`predicted_gain`/`uncertainty`, and `cost`, plus evidence strings
(`gain_uncertainty_nocost` omits `cost`). `semantic_search.py gain-context`
pins the exact bounded experience generation while excluding generic prose,
raw scores, and signed legacy deltas. Prediction schema 3 cites exact
proposal-relevant structured targets, with run/edge ids derived as the complete
union of those target receipts rather than cherry-picked independently.
Exact-zero abstention is always valid;
weak/confounded evidence can only preserve or raise uncertainty, while
nonzero gain must follow repeated same-child-code paired-control direction.
Inherited parent-parameter controls without a verified semantic pair are
confounded and may affect uncertainty only. Production ledgers currently
declare `direct_comparator_capability.status: unavailable`, so the paired
branch cannot fire until a deterministic evaluator replaces that gate. The
deterministic helper verifies both arithmetic and evidence qualification.

They remain auditable rubric estimates, not calibrated Bayesian posteriors.
The run-local `llm_intelligence_score` is a fixed `[0,100]` heuristic
reliability prior: `score / 100` scales the complete LLM-authored
gain/uncertainty/cost contribution while deterministic coverage stays
unscaled. Raw forecasts remain unchanged. A score of `100` preserves legacy
selection exactly; `0` leaves only the configured coverage term even though
forecasts are still collected. It is not normalized against a changing
leaderboard and must not be described as a calibrated probability. The first
schema-6/7 admission freezes it for the run; selection and ledger validation
reject later changes.

Policy receipt schema 7 preserves the prior, adjustment, final score,
experience revision, exact target/proposal relation, comparator coverage,
evidence ids, acquisition role/direction, `coverage`, configured intelligence
score, applied judgment weight, other weights, proposal-set digest, action,
ranking, and evidence. Under `coverage_experience` it instead records the
deterministic `experience_prior` and per-hypothesis carrier context counts.
These values stay in `policy_receipt`; they do not
become observations or beliefs. Historical schema-2/3/4/5/6 receipts remain
readable.

Runtime-deprioritized content stays eligible but is penalized in selection:
the carrier prior subtracts from a point's acquisition score for every
independent negative carrier context its non-baseline hypotheses carry, so
repeated disasters push a point down the ranking while a later positive
context can lift it again. The schema-7 receipt records the prior, the
per-hypothesis counts, and the selection index; the legacy lane fields are
null with `fallback: lanes_removed`.

Run-local configuration lives under `framework_cfg.json.semantic_search`.
`coverage_experience` (fully deterministic, no LLM scores) is the code and
copied-template default. Model-scored policies remain explicit opt-ins, for
example:

```json
{
  "semantic_search": {
    "policy": "gain_uncertainty_nocost"
  }
}
```

Changing the acquisition policy does not change registry or history semantics.
`got_select` still owns outer graph exploration and parent selection; inner HPO
still tunes numeric parameters inside the chosen semantic point.

## Evidence, belief, and pruning

The implemented P2 loop closes over the frozen registry without mutating it:

```text
background schema 3 (frozen when the stage ends)
  -> semantic point selection at search_space_state revision r
  -> candidate + persistent semantic edge receipts
  -> score/crash observations
  -> bounded experience schema 4 (replaceable belief)
  -> deterministic validated decision transition
  -> append-only search_space_state revision r+1
  -> next proposal set filters/orders against r+1
```

The registry is revised freely within the background stage and freezes at its
end; expanding registry membership mid-run is a planned extension (P4) and
remains deferred.

Every proposal set and policy receipt carries the
`search_space_state_revision` it was built against; `select` and `add-record`
reject stale revisions. Historical points remain valid at the revision where
they were selected: later pruning never rewrites a record, a point, an
observation, or a prior decision, and reopening appends a new transition.

Three state families stay distinct. External guidance is `active` or
`deprioritized`. Runtime control is `active`, `deprioritized`, or `pruned`. Belief coverage is `unevaluated`, `failed`, `observed`, or
`comparator_covered`. A crash is `+inf`, distinct from an unevaluated target,
and cannot by itself contradict or prune a semantic element. Automated pruning
is two-stage (`active -> deprioritized`, then `deprioritized -> pruned` in a
later experience generation with changed evidence for the same target), and
every recommendation is gated on mechanically recomputed evidence:
deprioritization and pruning both require `comparator_covered` with at least
two direct tuned edges (matched comparators whose child was deep-tuned);
deprioritization requires med/high confidence,
pruning requires high confidence, and a hypothesis direction must agree across
all cited pairs. A later generation, crash, unpaired transfer, or unrelated
DAG update alone cannot complete the second stage or reopen a target: a new or
corrected cited paired observation is required.

`unpromising` means the expected marginal value of another outer-search
evaluation is low after considering attribution, consistency across
implementations or contexts, a plausible mechanism or recurring failure mode,
counterevidence, untested conditions and adjacent hypotheses, residual
uncertainty/value of information, and cost. A worse child score or score delta
alone is never sufficient; weakly attributed or incomplete evidence remains
`mixed` and active.

Transitions never touch baselines or `baseline_only` dimensions. A dimension
may be deprioritized only when every selectable non-baseline hypothesis in it
is already runtime-deprioritized/pruned or independently
deprioritize/prune-recommended in the same generation. Dimension pruning requires the corresponding stronger
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

View schema 2 adds the run's explicit direct-comparator capability receipt.
For each target it returns the exact cited `evidence_edge_ids`, per-edge
score/status observations, the mechanical `evaluation_state`, and the cited
`comparator_coverage`, disclosing bounded-view loss through
`available_comparator_coverage` and `omitted_edge_counts`. Repeated
`--target-id` selects exact known targets for a smaller follow-up view.

“Direct” has a strict control meaning. A non-fresh candidate inherits the
first parent's exact code snapshot, projects its pinned applied incumbent onto
the child's compatible parameter schema, and must evaluate that projection at
warm config 0. That row proves tuning continuity, not semantic isolation, and
its normal receipt is therefore `semantic_control.status: unverified`. It also
cannot become `best_warm_params`, `best_warm_score`, `final_best_score`, or
`BASE_PARAMS`, including through a Phase-C duplicate; it remains only an
observation and budget event, and non-fresh screening requires `K_eval >= 2`. A
single-dimension edge is direct only when the receipt additionally contains a
validated same-child-code control/treatment pair whose configs differ exactly
in the declared semantic switch and have no shared-key reset. Its semantic
delta is treatment minus control; the child's later tuned improvement is
stored separately. Legacy final-vs-final, multi-dimension, reset-bearing,
unpaired, and independently tuned comparisons remain confounded and cannot
authorize signed gain or directional belief. The current runtime has no
deterministic paired evaluator, so report-authored `paired` receipts are
rejected and production observations remain `unverified`. The ledger and this
view expose a helper-owned `direct_comparator_capability: unavailable` receipt;
the paired contract is downstream semantics for a future helper-owned
evaluator, not a prose escape hatch today.

Admission is strictly round-serial: after every completed non-empty round,
experience extraction and `apply-space-state` run at the next quiescent round
boundary before another semantic admission, and propose -> select ->
`add-record` completes before any candidate implementation starts. No extractor
overlaps an in-flight candidate action; concurrent admission is deferred until
it has an explicit revision contract.

## Evidence-aware background research

The background-researcher runtime prompts keep only the mission, boundaries,
and the loop shape always loaded; they load the operational details on demand
from `docs/agent-resources/background-researcher/` (`retrieval.md` before the
first retrieval action, `evidence-registry.md` before registry distillation,
and `background-template.md` before writing the artifact).

The retrieval path alternates between drafting the space and searching. Each
round poses a few bounded questions through the adapter, reads the result
cards, visits the hits worth reading, and revises dimensions, hypotheses, and
follow-up questions from what landed. Recipe-shaped, bottleneck-shaped, and
community-source questions are first class. Exploratory rounds without
dimension targets are fine: targets are intent records, and per-dimension
coverage is a `status`-dashboard signal the researcher manages, not a gate.
Cross-cutting evidence **angles**—such as problem-class baselines, failure
modes, or counterevidence—remain retrieval intent rather than becoming new
dimensions. The query count is governed by evidence need and context budget,
not a fixed number.

Backends are selected explicitly per round:

- `frozen`: a pinned local JSON corpus (`--frozen-corpus`) for reproducible,
  network-disabled work; the adapter permits no live backend alongside it;
- `deepxiv`: open-world scholarly retrieval;
- `jina-search`: live web search (requires `JINA_API_KEY`). Web visits read
  through the jina reader with a direct fallback (`--visit-backend
  auto|jina-read|direct`).

The run-local `background_retrieval.json` uses retrieval schema 4 and is
append-only: each `search` call appends a round of queries, three-state
backend calls (`success`/`empty`/`failed`), and results deduplicated by
canonical work; visits append globally, with retained content stored under
`retrieval/`. There are no lanes, reading budgets, balanced selections, or
coverage exemptions. Each query records optional `target_dimension_ids` and
`evidence_roles` (`hypothesis`, `baseline`, `failure_mode`, `counterevidence`,
`relation`).

```bash
python tools/search_backends.py search \
  --manifest <run_dir>/background_retrieval.json \
  --query-spec '{"text":"<bounded evidence question>","target_dimension_ids":["<exact-resolved-dim-id>"],"evidence_roles":["hypothesis","counterevidence"]}' \
  --query-spec '{"text":"<problem-class comparator question>","target_dimension_ids":[],"evidence_roles":["baseline"]}' \
  --backend deepxiv
```

A registry source needs a tool-recorded receipt: a search hit or a successful
visit in the manifest. From the record the contract derives each source's
`verification_status` (`snippet_only`, `preview`, `section`, `full_text`),
surfaced in `search_backends.py status`. The status dashboard's unvisited
high-rank hits also reach downstream consumers: `background_contract.py
render --retrieval-manifest` lists them as `unexplored_leads`. DeepXiv `auto`
visits record `head`
as triage and then fetch up to three query-relevant body sections, falling
back to a preview only when necessary. Before the background freezes, a
synchronous audit spot-checks citations against the recorded source content
and returns unfaithful ones to the researcher for repair or removal. Older
retrieval manifests are rejected rather than migrated silently.

## Credibility, scope, and negative guidance

Hypotheses keep the existing literature credibility label:

- `unverified`, `preliminary`, `corroborated`, `replicated`, or `contested`.

This is an external evidence stamp, not a truth score. A replicated method can
fail locally; a preliminary hypothesis can work. Run-status belief about
dimensions and hypotheses lives in the schema-4 experience snapshot and the
`search_space_state` overlay, never in this registry stamp.

Sources, hypotheses, and guidance share five exact-tag scope facets:

- `model_families`, `data_regimes`, `metrics`, `interventions`, and
  `evaluation_protocols`.

`background_contract.py` derives claim-to-hypothesis transfer as `direct`,
`partial`, `mismatch`, or `unknown`. Only `direct` guidance carries selection
weight. `caution` annotates; external `deprioritize` marks a directly matched
hypothesis deprioritized — like runtime deprioritization it stays eligible
but earns at least a one-context carrier-prior penalty in selection. Removing
a hypothesis from consideration is a runtime decision the `search_space_state`
overlay makes from scored evidence; background guidance cannot remove
anything.

Unverified or contested negatives may only caution. Binding guidance needs
directly scoped, non-withdrawn primary empirical evidence. Every binding
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
  --retrieval-manifest <run_dir>/background_retrieval.json \
  [--baseline-mechanisms <run_dir>/baseline_mechanisms.json]  # provided entrypoint

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
