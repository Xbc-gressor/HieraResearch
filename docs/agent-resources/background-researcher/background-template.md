# Output template for `<run_dir>/background.md`

Read this resource immediately before writing `<run_dir>/background.md`. It is
the exact output template: a human-view skeleton followed by the fenced
schema-3 search-space registry. Fill every `<...>` placeholder; do not treat
this file as a prose explanation of the schema.

````markdown
# Background — <task_name>

## Task framing
<one or two lines: what is optimized (lower is better), the data shape, the dependency constraint>

## Retrieval
<backends used and rounds run; for a frozen corpus: corpus id + cutoff + SHA-256;
include backend failures and coverage limitations>

## Dimension coverage
| dimension | mode | explicit baseline | hypotheses | why selected |
|---|---|---|---|---|
| `dim-...` | searchable / baseline_only | `hyp-...` | `hyp-...`, ... | ... |

## Dimensions
### `dim-...`
- `hyp-...` — <baseline title and task provenance>
- `hyp-...` — <literature hypothesis and credibility>

## Relations
- `rel-...` — <activates / requires / excludes in readable form>

## Pitfalls
- `task-constraint` — <task constraint; cite TASK.md rather than literature>
- `operational` — <non-literature runtime or implementation pitfall>

## Deprioritize
- `g-01` — <literature-derived deprioritization; exact scope lives in registry>

## Search space registry
```json
{
  "schema_version": 3,
  "kind": "semantic_search_space",
  "space_id": "<stable task+run search-space id>",
  "catalog": {
    "id": "<resolved catalog id>",
    "revision": "<exact value from the resolved catalog command>"
  },
  "dimensions": [
    {
      "id": "dim-...",
      "definition": "<copy catalog definition exactly>",
      "boundary": "<copy catalog boundary exactly>",
      "catalog_provenance": "<copy catalog provenance exactly>",
      "selection_reason": "<task-specific reason>",
      "evidence": [{"kind": "task_contract | literature | agent_synthesis", "ref": "<receipt>"}],
      "mode": "searchable | baseline_only",
      "status": "active",
      "baseline_hypothesis_id": "hyp-...",
      "hypotheses": [
        {
          "id": "hyp-<globally-unique-stable-slug>",
          "title": "<short choice name>",
          "claim": "<specific attribution hypothesis, not a universal truth>",
          "kind": "baseline | evidence_prior | scope_probe",
          "status": "active",
          "provenance": [
            {"kind": "task_contract | literature | agent_synthesis", "ref": "<path or source id>"}
          ],
          "probe_for": ["<g-NN; only for scope_probe, otherwise omit>"],
          "claim_scope": "<actual population, mechanism, metric, and setting>",
          "scope": {
            "model_families": ["<lowercase_tag>"],
            "data_regimes": ["<lowercase_tag>"],
            "metrics": ["<lowercase_tag>"],
            "interventions": ["<lowercase_tag>"],
            "evaluation_protocols": ["<lowercase_tag>"]
          },
          "required_comparisons": ["<matched local comparison>"],
          "reopen_when": "<new scope, implementation, or evidence>",
          "literature_credibility": "unverified | preliminary | corroborated | replicated | contested",
          "credibility_rationale": "<why>",
          "testable_expectation": "<lower-is-better task observation>",
          "evidence": [{"source_id": "src-01", "role": "supports | contradicts | context"}]
        }
      ]
    }
  ],
  "relations": [
    {
      "id": "rel-<stable-slug>",
      "type": "activates",
      "status": "active",
      "provenance": [{"kind": "task_contract | literature | agent_synthesis", "ref": "<receipt>"}],
      "evidence": [],
      "when": {"dimension_id": "dim-...", "hypothesis_ids": ["hyp-..."]},
      "target_dimension_id": "dim-..."
    }
  ],
  "guidance": [
    {
      "id": "g-01",
      "section": "pitfall | deprioritize",
      "effect": "caution | deprioritize",
      "claim": "<negative finding stated only within the typed scope>",
      "scope": {
        "model_families": ["xgboost"],
        "data_regimes": ["binary_risk_assessment"],
        "metrics": ["f1"],
        "interventions": ["global_sampling_to_balance"],
        "evaluation_protocols": ["cross_validation"]
      },
      "literature_credibility": "unverified | preliminary | corroborated | replicated | contested",
      "credibility_rationale": "<strength inside this exact scope>",
      "reopen_when": "<which scope facet or local evidence reopens it>",
      "evidence": [{"source_id": "src-01", "role": "supports | contradicts | context"}]
    }
  ],
  "sources": [
    {
      "id": "src-01",
      "type": "paper | official_code | official_docs | benchmark | dataset | first_party_report | web_lead",
      "title": "<source title>",
      "url": "https://...",
      "publication_status": "preprint_only | peer_reviewed | published_status_unknown | withdrawn_or_retracted | not_applicable",
      "validation_status": "claim_only | artifact_available | independently_reproduced | not_assessed",
      "studied_scope": {
        "model_families": ["<lowercase_tag>"],
        "data_regimes": ["<lowercase_tag>"],
        "metrics": ["<lowercase_tag>"],
        "interventions": ["<lowercase_tag>"],
        "evaluation_protocols": ["<lowercase_tag>"]
      }
    }
  ]
}
```

## Coverage and unresolved evidence
- <missing source, disputed claim, unavailable backend, evidence gap, or catalog coverage gap>
````
