# The semantic search space, formally

This is the mathematical view of the P2 mechanism implemented by
`tools/semantic_space.py`, `tools/background_contract.py`,
`tools/semantic_evidence.py`, `tools/search_space_state.py`, and
`tools/semantic_search.py`. It restates what the code enforces; it adds no
requirement of its own. Terms in backticks name the exact contract fields.

## Spaces and maps

- Let `X` be the set of concrete candidates: complete runnable
  implementations for the task — a candidate's `train.py` together with its
  numeric configuration (`PARAM_SCHEMA`, `BASE_PARAMS`, everything inner HPO
  may change).
- The resolved dimension catalog fixes the dimension index set `D` (ordered
  `dim-*` ids, content-addressed by a catalog `revision`). By default it is
  `contracts/semantic-dimensions-v1.json`; `llm_induced` resolves a validated
  run-local catalog instead.
- Background research freezes the run dimensions `A`: `catalog_subset` selects
  `A ⊆ D`, while `llm_induced` defines a task-specific catalog and uses `A = D`.
  The registry of `kind: "semantic_search_space"` gives each dimension local
  `hyp-*` hypotheses containing one distinguished `baseline_hypothesis_id`,
  and declares `rel-*` relations (`activates`, `requires`, `excludes`). The
  frozen space is content-addressed by `space_revision`.
- The run's semantic space `S` is the set of assignment vectors that
  `validate_point` accepts: one entry per selected dimension, in registry
  order, each either `selected` (naming a local active hypothesis) or
  `inactive` (naming the unsatisfied `activates` relations). Because of the
  relations, `S` is a constrained subset of the product `∏_{d ∈ A} H_d` — a
  dependent product, not a free Cartesian one.
- The attribution map `π : X → S` sends a candidate to its ledger record's
  `semantic_point`. Two candidates are equivalent, `x ~ y`, iff their points
  have the same `point_id`.

## A point is an equivalence class

The outer loop never manipulates elements of `X` directly; it manipulates the
quotient `X/~`. A `point_id` names the whole fiber
`π⁻¹(s) = { x ∈ X : π(x) = s }`: the class of candidates that share the same
mechanism-level choices and differ only in inner-loop-owned detail (learning
rate, depth, batch size, the concrete code realizing the mechanisms). Like
cosets of a factor group, the fibers partition `X`, and the outer loop's
moves are moves between classes, never between individual implementations.
(No group operation is assumed, so the precise object is a quotient set
rather than a factor group.)

The score `f : X → ℝ ∪ {+inf}` (the task's `score_fn`; lower is better, a
crash is `+inf`) does not descend to the quotient — members of one fiber
score differently. The outer loop's induced objective is

    F(s) = inf { f(x) : x ∈ π⁻¹(s) },

the best score achievable at that semantic point. The inner loop
(`tunable-contract-extractor` warm-start plus `tuner-orchestrator` deep
tuning) searches within one fiber and approximates `F(s)`; every scored
candidate `x` yields only an upper bound `f(x) ≥ F(π(x))`. This is why the
contract insists that point membership is attribution, not causal evidence
of value: ledger observations are noisy one-sided bounds on `F`, and any
belief about `F` (acquisition `predicted_gain`, `uncertainty`) is a derived,
replaceable view over the ledger — never part of the registry or the
observation history.

## Dimension resolution is subspace selection

Under `catalog_subset`, freezing `A` pins every built-in dimension outside `A` —
either inapplicable to the task or fixed by a task constraint. Under
`llm_induced`, task-first decomposition defines `D` directly and the registry
uses the whole catalog. In either case, the run searches a frozen slice of the
larger space of possible candidate mechanisms, and the
two-level decomposition

    min_{x} f(x) = min_{s ∈ S} F(s)

recovers the true optimum only if the slice contains one — that is, if some
optimal solution agrees with the pinned values on every unselected
dimension. The registry rules in `docs/background-research.md` are the
operational sufficient condition: select every dimension holding a legal
material choice, and represent a constraint-fixed material choice as a
visible `mode: "baseline_only"` dimension rather than dropping it. The
residual risk — a material choice no catalog dimension owns — is a genuine
coverage gap: the frozen subspace may exclude the optimum and no within-run move
can repair that run's catalog.

## Search moves in the quotient

`got_select` owns the structural action (`fresh`, `improve`, `crossover`)
and its numeric parents. `semantic_search.py propose` then enumerates a
bounded, deterministic set of valid points for that action, filtered and
ordered against the ledger's current `search_space_state` revision:

- `fresh` completes the all-baselines point and single/pairwise deviations
  (`complete_point` fills unspecified coordinates with baselines and resolves
  activation to a fixpoint);
- `improve` re-selects one coordinate of the parent's point at a time;
- `crossover` recombines the parents' differing coordinates, masked and
  bounded.

Every enumerated point passes `validate_point` — admissibility under
`requires`/`excludes` is a deterministic check, not a policy judgment.
`semantic_search.py select` then applies a replaceable acquisition policy
(`coverage`, `gain`, `gain_uncertainty`, `gain_uncertainty_nocost`) over
fibers. For model-scored policies, `gain-context` pins the bounded experience
generation and its cited terminal/semantic-edge observations. Prediction schema 2 separates
the background/mechanism prior from signed experience adjustments for both
gain and uncertainty; the helper verifies the arithmetic and requires a
snapshot carrying cited evidence to affect at least one final number by 0.01
or more. A valid snapshot whose bounded collections cite no run or edge stays
revision-pinned but is prediction-empty: citations and adjustments are zero.
Policy receipt schema 4 persists those inputs separately from `coverage` and
`cost`, so the history update is auditable rather than implied by evidence prose. Before
acquisition, proposals are partitioned into active and deprioritized budget
lanes. With interval `N` (default 5), every
Nth one-based semantic admission selects within the deprioritized lane and
all other admissions select within the active lane; acquisition scores never
move a point across lanes. If a scheduled lane is empty, the other lane fills
the slot and the schema-4 receipt records the fallback and both lanes.
Hypotheses are coordinates, not consumable resources: one `hyp-*` may
participate in many points.

Finally, a ledger record carries two different maps with different
codomains: `source_run_ids` (ancestry — which concrete candidates informed
generation, valued in `X`) and `semantic_point` (attribution — `π(x)`,
valued in `S`). `point_diff` reconstructs the semantic difference between a
child and its parents mechanically; it makes no causal claim about scores.

## Persisted edges are attribution deltas

Every non-fresh record persists one `semantic_edges` receipt per numeric
parent. A receipt is the mechanical fiber difference `π(parent) → π(child)`:
`hypothesis_changed` when both coordinates name hypotheses,
`dimension_activated` when the child selects a previously inactive dimension,
`dimension_deactivated` for the reverse. `change_class` counts the changed
assignments (`same_point`, `single_dimension`, `multi_dimension`) — an
attribution-strength category, not a causal claim about the score delta.

A “dimension added/removed” edge in the roadmap's target vocabulary is, inside
one fixed run, exactly these conditional operations: registry membership never
changes within a run, so a dimension can only move between `inactive` and
`selected` at the point level. Changing registry membership itself is P4 space
expansion and remains deferred.

## Runtime eligibility is a family of selectable subsets

The frozen registry fixes `S` once and for all. The append-only
`search_space_state` overlay defines, at each revision `r`, a selectable
subset `S_r ⊆ S`:

- a runtime-`pruned` hypothesis drops out of every point in `S_r` but keeps
  its id, records, and receipts in `S`;
- a runtime-`pruned` dimension is pinned to its explicit
  `baseline_hypothesis_id`, so `S_r` restricts the dimension's coordinate to
  the baseline value;
- `deprioritized` content stays in `S_r` but receives only its configured
  semantic-admission budget (default one of every five slots);
  externally `excluded` content was never in any `S_r`.

Pruning a dimension pins its baseline rather than changing point arity or
`space_revision` because arity and the frozen revision are properties of `S`,
not of the selection policy: historical points keep validating against the
same `S` at their recorded state revisions, and the overlay only narrows which
fibers the next proposal may choose. Reopening appends a new decision instead
of mutating a flag, so `S_r` is a pure function of the whole decision log.

The loop at revision `r`:

    select point s ∈ S_r -> candidate x with π(x) = s and persisted edge
    receipts -> observations f(x) (a crash is +inf) -> bounded schema-3
    belief regenerated over the ledger -> deterministic validated decision
    transition -> revision r+1 -> the next proposal filters/orders over
    S_{r+1}.

`ledger.dag_revision` counts graph-visible score/status changes only;
`search_space_state.revision` counts append-only decisions. The two cursors
are independent by construction.

## Bounded evidence and round-serial admission

The extractor's belief inputs come from `background_contract.py
target-evidence`, a deterministic scan over the persisted receipts that
returns, per target, exact cited edge ids, per-edge score/status observations,
the mechanical `evaluation_state`, and cited comparator counts — never a
reconstruction from the bounded Top/Bottom graph window. Belief claims are
replaceable interpretations validated against those receipts; raw records are
the durable history.

Admission is strictly round-serial: extraction and state application happen
only at quiescent round boundaries, and propose → select → `add-record`
completes before candidate implementation begins, so no decision transition
can invalidate an in-flight selection. Concurrent admission awaits an explicit
revision contract.
