# Dimension-Aware Query Planning Design

## Goal

Make background-research query planning auditable against the resolved semantic
dimensions without treating retrieval queries as search-space dimensions.
Preserve hyperparameter research as a separate inner-HPO evidence activity.

## Boundaries

- The resolved dimension catalog and registry remain the source of search-space
  identity. Retrieval never creates, merges, or renames dimensions.
- Query-to-dimension alignment is many-to-many.
- Every `searchable` registry dimension must be covered by at least one
  grounding query or by an explicit evidence-free exemption with a rationale.
- `baseline_only` dimensions do not require literature-query coverage.
- Inner-HPO evidence remains outside the semantic search space. This change
  records its retrieval intent but does not add it to semantic points or change
  the tuner's context contract.
- Existing run artifacts are local and disposable. Retrieval manifest schema 1
  is rejected rather than migrated silently.

## Retrieval Manifest Contract

`background_retrieval.json` advances to schema version 2. Each query has:

```json
{
  "id": "q-01",
  "text": "Which regularized tree families are robust on small tabular data?",
  "lane": "grounding",
  "target_dimension_ids": ["dim-model-architecture"],
  "evidence_roles": ["hypothesis"]
}
```

The allowed evidence roles are:

- `hypothesis` — candidate mechanisms or regimes;
- `baseline` — strong or task-standard comparators;
- `failure_mode` — resource, data, or statistical failure conditions;
- `counterevidence` — negative results, replications, contradictions, or
  boundary evidence;
- `relation` — activation, requirement, compatibility, or exclusion evidence;
- `inner_hpo_prior` — numeric ranges or practitioner priors owned by inner HPO.

Semantic evidence queries must name one or more unique `dim-*` targets.
`inner_hpo_prior` is mutually exclusive with all semantic evidence roles and
must have no dimension targets. This forces a separate query instead of
smuggling numeric HPO into a semantic dimension.

The manifest also contains:

```json
"coverage_exemptions": [
  {
    "dimension_id": "dim-example",
    "rationale": "The frozen corpus has no applicable evidence; retain task-derived choices without a literature prior."
  }
]
```

Exemptions are unique by dimension, require a non-empty rationale, and cannot
duplicate a dimension targeted by a query.

## Command Interface

`tools/search_backends.py search` replaces plain repeated `--query` arguments
with repeated structured `--query-spec` JSON objects. The tool assigns stable
`q-NN` ids and the selected lane. Optional repeated `--coverage-exemption` JSON
objects record deliberate evidence gaps. Both forms are validated before any
backend call.

Example:

```bash
python tools/search_backends.py search \
  --manifest <run_dir>/background_retrieval.json \
  --lane grounding \
  --query-spec '{"text":"Which model families are robust here?","target_dimension_ids":["dim-model-architecture"],"evidence_roles":["hypothesis","baseline"]}' \
  --query-spec '{"text":"Which learning-rate ranges are commonly stable?","target_dimension_ids":[],"evidence_roles":["inner_hpo_prior"]}' \
  --frozen-corpus <pinned-corpus.json>
```

The query count is a context-budget concern, not a hard contract. Prompts ask
for the smallest non-duplicative set that covers the resolved searchable
dimensions and cross-cutting evidence needs.

## Cross-Artifact Validation

`search_backends.validate_manifest` validates schema shape, roles, target-id
syntax, inner-HPO separation, exemptions, and existing retrieval integrity.

`background_contract.validate_registry`, when given a retrieval manifest,
joins query targets to the completed registry and rejects:

- query targets absent from the registry;
- exemptions absent from the registry;
- searchable dimensions with neither a grounding query nor an exemption.

Semantic novelty-only queries do not satisfy background grounding coverage.

## Agent and Documentation Changes

Both Claude and OpenCode background-researcher prompts will:

- use resolved dimensions as the query coverage frame;
- state explicitly that queries are not dimensions;
- require query metadata and explain many-to-many alignment;
- keep inner-HPO queries separate;
- remove the fixed 3–6 requirement;
- use the structured CLI examples; and
- rely on final background validation for coverage enforcement.

`docs/background-research.md` will document the same contract. Prompt mirrors
retain only their runtime-specific tool-name differences.

## Testing

Network-free checks will cover:

- valid semantic and inner-HPO query specs;
- rejection of missing targets, unknown roles, and mixed HPO/semantic roles;
- coverage exemption validation;
- background rejection of unknown targets and uncovered searchable dimensions;
- successful many-to-many coverage; and
- synchronized required wording in both runtime prompts.
