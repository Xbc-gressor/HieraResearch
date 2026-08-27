# Task-first dimension induction

Read this resource only when
`framework_cfg.json.space_initialization.dimension_strategy` is
`llm_induced`. It defines how to create the run-local
`dimension_catalog.json` before literature retrieval begins.

## Inputs and independence

Use only the task's declared candidate decision surface:

- `tasks/<task>/TASK.md`, especially the evaluation contract and constraints;
- `tasks/<task>/task.toml`;
- candidate-visible interfaces in `tasks/<task>/prepare.py`.

When the invocation context supplies an explicit `task_packet`, that bounded
projection replaces the three installed-task inputs above. Use only the packet
and candidate-visible supporting paths it names, excluding any declared
provided-baseline implementation until the catalog is final; the resulting
catalog provenance must cite the packet and this document.

Do not consult `contracts/semantic-dimensions-v1.json` or literature results
while inducing dimensions. Do not inspect hidden evaluation data, infer a blind
task's identity, or treat the fixed evaluator and HieraResearch's own search
policy as candidate choices. A provided baseline may inform the later baseline
hypotheses, but it must not determine the decomposition.

## Induction procedure

1. List the qualitatively different decisions a legal candidate can make from
   task input through metric-facing output.
2. Group decisions by the interface whose output they directly change. One
   dimension owns one cohesive class of candidate decisions.
3. Apply a counterfactual check: two legal candidates should be able to differ
   in this dimension while the remaining dimensions stay conceptually fixed.
4. Merge dimensions that own the same interface or cannot be varied
   independently. Split a dimension whose choices alter materially different
   interfaces.
5. Remove items that are:
   - one named method or implementation;
   - an entire end-to-end pipeline;
   - a scalar setting handled by inner HPO;
   - evaluator behavior, benchmark rules, or framework acquisition policy;
   - a catch-all for otherwise unclassified ideas.
6. Check coverage against several plausible legal candidate families from the
   task contract. Each material choice must have one clear owner without a
   miscellaneous dimension.
7. Prefer 4–10 dimensions. This is a soft design budget, not a validation rule.
   Use fewer for a genuinely narrow task; exceed ten only when the ownership and
   counterfactual checks show that merging would erase a material distinction.

Definitions state what a dimension owns. Boundaries name the nearest plausible
choices it does not own. Use stable mechanism-level `dim-<slug>` ids rather
than method, model-brand, dataset, or run names. Conditional dependence between
dimensions belongs in the later registry relations; it is not a reason to merge
their ownership.

The resulting catalog is the final dimension set for the run. The background
registry must use every dimension exactly once and in catalog order; there is no
second subset-selection pass.

## Artifact contract

Write `<run_dir>/dimension_catalog.json` directly:

```json
{
  "schema_version": 1,
  "catalog_id": "llm-induced/<task_name>",
  "provenance": "Induced from tasks/<task_name>/TASK.md, task.toml, and prepare.py using docs/dimension-induction.md",
  "dimensions": [
    {
      "id": "dim-<stable-slug>",
      "definition": "<the cohesive candidate decision this dimension owns>",
      "boundary": "<the nearest candidate decisions this dimension does not own>"
    }
  ]
}
```

Do not persist scratch lists, alternative catalogs, rationales, or a compiled
form. Validate the final artifact mechanically:

```bash
python tools/background_contract.py catalog \
  --path <run_dir>/dimension_catalog.json
```

Correct catalog contract errors before retrieving literature. If a valid
catalog cannot be produced, report setup as blocked; do not fall back to the
built-in catalog.
