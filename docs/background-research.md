# Background as a semantic search space (P1)

HieraResearch uses `background.md` to define the frozen outer semantic search
space for one run. It is no longer a flat list of end-to-end directions.

The contract deliberately separates:

- **space definition** — catalog dimensions, task hypotheses, conditions, and
  exclusions in `background.md`;
- **candidate attribution** — a complete `semantic_point` in each ledger record;
- **ancestry** — numeric parent ids in `source_run_ids`;
- **observations** — scores, crashes, tuning metadata, and logs in raw records;
- **derived belief** — the bounded regenerated `ledger.experience` snapshot;
- **selection policy** — a replaceable `policy_receipt` for each chosen point.

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

Every selected dimension is `active` in P1 and has:

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
- claim boundary and exact five-axis scope;
- required local comparisons and reopening condition;
- literature credibility and rationale;
- testable lower-is-better expectation;
- typed source evidence links.

P1 does not author pruned hypotheses or dimensions. `active`, directly
`deprioritized`, and directly `excluded` eligibility are derived from typed
external guidance without deleting the registered element. Evidence-preserving
run-time pruning is P2.

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
there. `background_contract.py lineage` can reconstruct point differences from
parents mechanically while warning that they are attribution, not causal edges.
Persistent semantic DAG receipts are P2.

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

Three replaceable policies are implemented:

1. `coverage` — deterministic exploration by inverse hypothesis coverage and
   exact-point novelty; no LLM score is required.
2. `gain` — predicted gain plus a small coverage term minus predicted cost.
3. `gain_uncertainty` — predicted gain plus an explicit uncertainty bonus and
   coverage, minus predicted cost.

For model-scored policies, each proposal receives separate `[0,1]`
`predicted_gain`, `uncertainty`, and `cost` rubric inputs plus evidence strings.
They are auditable estimates, not calibrated Bayesian posteriors. The selected
record preserves all four components (`coverage` included), weights, proposal
set digest, action, ranking, and evidence. These values stay in
`policy_receipt`; they do not become observations or beliefs.

Run-local configuration lives under `framework_cfg.json.semantic_search`.
`gain_uncertainty` is the code and copied-template default; `coverage` (fully
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

## Evidence-aware background research

The retrieval path remains evidence-aware. Research first decomposes the task
into independent questions, then uses an explicitly selected condition:

- `frozen`: a pinned local JSON corpus for reproducible, network-disabled work;
- `deepxiv`: optional open-world scholarly retrieval;
- `jina`: explicit live-web fallback/ablation, never an invisible default;
- runtime-native web tools: fallback only, with successful content recorded
  through `search_backends.py record-visit`.

The run-local `background_retrieval.json` records queries, backend failures,
canonical deduplication, balanced selections, visits, content depth, budgets,
versions, and hashes. Grounding has a 6000-token reading lane; novelty is a
separate 2048-token lane. A novelty-only visit cannot support a registry claim.

All registry sources must have a successful grounding visit. Search snippets,
generated summaries, and unvisited URLs are insufficient. Frozen and live
conditions cannot be mixed in one main evidence condition.

## Credibility, scope, and negative guidance

Hypotheses keep the existing literature credibility axis:

- `unverified`, `preliminary`, `corroborated`, `replicated`, or `contested`.

This is an external evidence stamp, not a truth score. A replicated method can
fail locally; a preliminary hypothesis can work. P1 does not create the P2
dimension/hypothesis run-status belief layer.

Sources, hypotheses, and guidance share five exact-tag axes:

- `model_families`, `data_regimes`, `metrics`, `interventions`, and
  `evaluation_protocols`.

`background_contract.py` derives claim-to-hypothesis transfer as `direct`,
`partial`, `mismatch`, or `unknown`. Only `direct` guidance changes eligibility.
`caution` annotates; `deprioritize` orders a directly matched hypothesis after
active ones; `exclude` removes it from proposal generation while preserving its
identity and receipt.

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

python tools/semantic_search.py propose \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json \
  --op <op> --parents <ids> --output <proposals.json>

python tools/semantic_search.py select \
  --proposals <proposals.json> [--predictions <predictions.json>] \
  --ledger <run_dir>/ledger.json \
  --point-output <point.json> --receipt-output <policy.json>

python tools/validate_background.py
```

Legacy `## Direction registry` files and schema 1/2 data fail with an explicit
error. P1 intentionally provides no migration path for local disposable runs.

## P1 boundary

This implementation does not add evidence-preserving pruning, two-level
dimension/hypothesis belief extraction, persistent semantic DAG edge receipts,
intermediate-log bottleneck retrieval, dynamic space expansion, or convergence
claims. Those remain ordered P2–P4 work.
