# Background Researcher Progressive Disclosure Design

**Status:** Approved for planning
**Date:** 2026-07-22

## Context

The Claude `background-researcher` prompt is 520 lines, and its OpenCode mirror
contains the same operational contract. The length is concentrated in two
blocks: progressive retrieval instructions and a complete `background.md`
output example. These details are necessary, but presenting all of them as
always-on role instructions weakens the salience of the agent's mission,
boundaries, stage order, and evidence invariants.

This change applies progressive disclosure to the existing contract. It does
not relax validation, change the search-space schema, or alter background
research behavior.

## Goals

- Keep the role, hard boundaries, workflow order, and cross-phase invariants in
  the agent prompt.
- Load specialized retrieval, evidence-registry, and artifact-format details at
  the phase that first needs them.
- Give Claude and OpenCode one shared source for each extracted instruction.
- Preserve every current semantic requirement and validation command.
- Keep resource routing explicit and one level deep from each agent prompt.

## Non-goals

- Changing retrieval schema 2 or semantic search-space schema 3.
- Adding a scaffold command or moving validation rules out of deterministic
  tools.
- Turning the dedicated role agent into a capability skill.
- Refactoring unrelated architecture documentation.
- Reducing research quality or source-verification requirements.

## Chosen Design

### Always-on agent prompt

Both runtime prompts retain:

1. Role, inputs, outputs, and prohibited actions.
2. Task scoping, lower-is-better framing, dependency constraints, and fixed
   evaluation surfaces.
3. Dimension-strategy resolution, catalog ownership, baseline requirements,
   and the freeze point.
4. Query planning, dimension/role attribution, coverage requirements, and the
   separation between semantic hypotheses and inner-HPO priors.
5. A compact evidence invariant: inspect primary material, record exact studied
   scope, do not promote snippets or summaries to evidence, and keep negative
   guidance scoped and reversible.
6. Phase gates that name the exact shared resource to read before retrieval,
   registry distillation, and artifact writing.
7. Final validation commands, return receipt, and concise hard boundaries.

The main prompt should remain a complete navigation and control-flow document.
An agent must be able to identify what to do next and which resource to load
without reading any resource prematurely.

### Shared Level 3 resources

Create direct resources under:

```text
docs/agent-resources/background-researcher/
├── retrieval.md
├── evidence-registry.md
└── background-template.md
```

`retrieval.md` is read immediately before the first retrieval action. It owns:

- frozen versus explicitly open-world backend selection;
- `search`, `visit`, and `record-visit` command forms;
- manifest receipts, lane budgets, progressive reading, deduplication, and
  fallback behavior;
- source-quality checks needed while reading;
- successful grounding-visit requirements.

`evidence-registry.md` is read after retrieval and before registry distillation.
It owns:

- the exact five-facet scope contract and conservative matching semantics;
- literature credibility labels and binding-negative thresholds;
- hypothesis kinds, required comparisons, reopening conditions, and scope
  probes;
- exact relation payloads and structured guidance effects.

`background-template.md` is read immediately before writing `background.md`.
It owns the complete human-view and fenced schema-3 JSON skeleton currently
embedded in the prompts. It is a template, not a second prose explanation of
the schema.

All three files are linked directly from both agent prompts. They do not link to
each other as required reading. Any resource exceeding 100 lines starts with a
short contents list so partial previews still expose its scope.

### Content movement and deduplication

Content is moved rather than paraphrased wherever possible so the first change
is behavior-preserving. A short invariant may remain in the agent prompt when
it affects more than one phase, but the detailed rule has one canonical home.

`docs/background-research.md` remains the architectural description for human
maintainers. It may link to the runtime resources, but runtime prompts do not
load that complete architecture document. Existing overlapping operational
examples should not be expanded during this change.

The Claude and OpenCode prompts keep their runtime-specific frontmatter and
tool declarations. Their shared body and resource routing remain synchronized.

## Phase Flow

```text
scope task
  -> resolve/freeze dimensions
  -> plan dimension-aware evidence questions
  -> read retrieval.md
  -> retrieve and inspect evidence
  -> read evidence-registry.md
  -> distill hypotheses, guidance, and relations
  -> read background-template.md
  -> write artifacts
  -> validate until clean
  -> return compact receipt
```

If a resource cannot be read, the agent stops before the affected phase and
reports the missing path. It must not reconstruct the omitted contract from
memory.

## Validation

Implementation verification will include:

1. Confirming both runtime prompts reference all resources with identical phase
   semantics.
2. Confirming every requirement removed from a prompt exists in exactly one
   shared resource or remains enforced by an existing deterministic validator.
3. Running `python tools/validate_background.py` and
   `python tools/validate_search_backends.py`.
4. Inspecting the resolved OpenCode agent with
   `opencode debug agent background-researcher`.
5. Comparing prompt/resource line counts and checking that the main Claude
   prompt is approximately 180–230 lines without using line count as a hard
   correctness criterion.
6. Reviewing diffs for the existing uncommitted dimension-aware query-planning
   work so no user changes are overwritten or reverted.

No heavyweight task preparation, network retrieval, or autonomous experiment
is required for this instruction-only refactor.

## Acceptance Criteria

- The always-loaded agent body foregrounds mission, boundaries, phase order,
  and evidence invariants.
- Retrieval mechanics, evidence/registry details, and the output template are
  loaded only at their named stage.
- Claude and OpenCode use the same three shared resources.
- Current dimension-aware query planning and scope-facet semantics are
  preserved.
- Existing deterministic validation passes.
- No run artifacts, task files, ledger state, or unrelated user changes are
  modified.
