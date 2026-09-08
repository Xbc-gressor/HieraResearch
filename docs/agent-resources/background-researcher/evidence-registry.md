# Evidence and registry contract for the background researcher

Read this resource after retrieval and before registry distillation. It
defines how studied scope is recorded, the five-facet scope contract and its
conservative matching, literature credibility labels, hypothesis kinds and
probe requirements, relation payloads, and structured guidance effects.

Contents: studied scope · five-facet scope contract · conservative matching ·
literature credibility · hypothesis kinds and scope probes · relations ·
structured guidance · source receipts · hypothesis preservation.

## Studied scope

Record the **studied scope**, not just the conclusion sentence: the problem or
input regime, candidate solution family, mechanism under study, objective or
metric, comparator, resource conditions, and evaluation protocol. Use the
task's own vocabulary. Evidence from one mechanism and regime does not
transfer automatically to a neighboring mechanism, a different constraint
regime, or a different comparison protocol. Split claims whenever those
boundaries change; never generalize from the settings a source happened to
study to the whole problem class.

## Five-facet scope contract

Encode the mechanically matched part of source, guidance, and hypothesis scope
with the same five exact-tag **facets**: `model_families`, `data_regimes`,
`metrics`, `interventions`, and `evaluation_protocols`. These stable field
names are an evidence-transfer contract, not search-space dimensions or an
assumption that every task is estimator-shaped. Interpret them through the
task contract: `model_families` identifies the relevant candidate solution or
algorithm family, and `data_regimes` identifies the relevant problem or input
regime. The other facets likewise record the task's own objective, mechanism,
and comparison protocol. Use specific lowercase tags and do not broaden a tag
beyond what the source studied. `background_contract.py` derives scope match
mechanically; you do not self-assign `applicability` or `scope_match`.

## Conservative matching

Matching is conservative. Guidance is direct only when its scope contains the
hypothesis on every facet. Any disjoint facet is a mismatch; overlap without
full containment is partial. Only direct guidance may weigh against a
hypothesis. This deliberately makes a false broad claim fail open (the
hypothesis remains explorable) instead of silently blocking work.

## Literature credibility

Assign one claim-level **literature credibility** label:

- `unverified` — only a lead, abstract-level claim, or source of unclear provenance;
- `preliminary` — direct primary evidence, but single-source, unreviewed, or
  methodologically limited;
- `corroborated` — multiple independent primary sources or unusually strong
  artifact-backed evidence agree;
- `replicated` — independently reproduced under meaningfully comparable conditions;
- `contested` — credible evidence materially disagrees.

The label is a compact evidence stamp, not a truth value. Explain it in
`credibility_rationale`. `unverified` and `contested` negative evidence may
only produce `caution`; it cannot deprioritize. A binding negative
item needs a directly scoped, non-withdrawn primary empirical source (paper,
benchmark, or first-party empirical report). Do not duplicate one canonical
work under several source ids to simulate corroboration.

## Hypothesis kinds and scope probes

Every non-baseline hypothesis must name the matched comparisons required to
test it here and a concrete reopening condition. Use
`kind: evidence_prior` for a positive prior and `kind: scope_probe` for a
credible alternative or boundary case that the evidence does not settle. A
probe's `probe_for` lists the negative `g-*` guidance id(s) whose boundary it
tests. Every `deprioritize` guidance item needs at least
one out-of-scope probe hypothesis. The probe stays in the normal search space;
the contract does not force an arbitrary bootstrap slot.

## Relations

Relation payloads are exact: `activates` has `when` plus
`target_dimension_id`; `requires` has `when` plus `then`; each choice scope is
`{"dimension_id": ..., "hypothesis_ids": [...]}`; `excludes` has a `members`
list of at least two such scopes. Activation relations must be acyclic. A
conditional point is inactive only through its declared incoming activation
relations, never through an omitted assignment. Every relation has provenance,
status, and evidence receipts.

## Structured guidance

Register machine-readable **guidance** for every literature-derived Pitfall or
Deprioritize claim. `caution` annotates only; `deprioritize` moves a directly
matched hypothesis behind active hypotheses — it stays eligible and earns a
selection penalty. There is no exclusion effect: removing a hypothesis from
consideration is a runtime decision made later from scored evidence.
`unverified`/`contested` findings remain cautions. A scope mismatch never
changes standing. Guidance does not delete hypotheses.

## Source receipts

Every registry source must come from your CLI retrieval record: a search hit
or a successful visit in `background_retrieval.json`. From that record each
source derives a verification tier — `snippet_only`, `preview`, `section`, or
`full_text` — reflecting how much of it was actually read. Cite only what the
recorded content carries: before the space freezes, citations are
spot-checked against the receipts, and a claim the record does not support
comes back for repair or removal.

## Hypothesis preservation

Do not omit a plausible legal hypothesis because of literature guidance. Give
it a stable `hyp-*` identity and let the typed matcher derive `active` or
`deprioritized`. A negative empirical result is scoped to its
actual mechanism and setting. Preserve a plausible nearby mechanism outside
that scope as a `scope_probe` rather than silently removing it.
