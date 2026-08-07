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
(`tunable-contract-extractor` warm-start plus `tuner-orchestrator`
progressive tuning bouts) searches within one fiber and approximates `F(s)`
incrementally; every scored candidate `x` yields only an upper bound
`f(x) ≥ F(π(x))`, and each tuning bout tightens that candidate's bound. The
graded `evaluation_depth` (screening / tuned_lightly / tuned) records how
tight the bound is, which is what lets lightly-tuned candidates count as
intermediate semantic evidence rather than only at the screening/tuned
extremes. This is why the
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
(`coverage_experience` by default — deterministic coverage plus the carrier
prior — plus `coverage`, `gain`, `gain_uncertainty`,
`gain_uncertainty_nocost`) over
fibers. For model-scored policies, `gain-context` pins the bounded experience
generation but exposes no generic prose, raw scores, or signed legacy deltas.
Prediction schema 3 separates the background/mechanism prior from signed
experience adjustments for gain and uncertainty. Exact-zero abstention is
always valid. Weak/confounded evidence can only preserve or raise uncertainty;
signed gain requires proposal-relevant, repeated, directionally consistent
same-child-code semantic control/treatment pairs. An inherited
parent-parameter control without that pair is uncertainty-only.
Production ledgers currently carry
`direct_comparator_capability.status: unavailable`, so this signed branch is
explicitly dormant rather than inferred to exist from its downstream schema.
`semantic_search.llm_intelligence_score` supplies a fixed pre-run heuristic
reliability prior `w = score / 100`. For model-scored policies, `w` multiplies
the complete LLM-authored gain/uncertainty/cost term; deterministic coverage
is added without scaling, and raw forecasts are preserved. Thus `100` is exact
legacy behavior and `0` leaves only the configured coverage term even though
forecasts are still collected. The score is neither a calibrated probability
nor normalized to a changing leaderboard. The first schema-6/7 admission freezes
it for the run; selection and ledger validation reject later changes.

Policy receipt schema 7 persists the exact
helper-derived target, proposal relation, comparator coverage, evidence ids,
acquisition role, gain direction, configured score, and applied weight
separately from `coverage` and `cost` — or, under `coverage_experience`, the
deterministic `experience_prior` and per-hypothesis carrier context counts;
the cited run/edge ids must be the complete union of the named target receipts,
not a model-selected subset. Deprioritized
content stays eligible but is penalized by the carrier prior rather than
lane-scheduled: every independent negative carrier context subtracts from a
point's acquisition score, and the schema-7 receipt records the prior, the
per-hypothesis counts, and the selection index (legacy lane fields are null,
`fallback: lanes_removed`).
Hypotheses are coordinates, not consumable resources: one `hyp-*` may
participate in many points.

When a task declares a provided candidate entrypoint, it is admitted before
ordinary acquisition as the first root at `complete_point(registry)`. The copy
keeps a content receipt, its supplied default configuration is evaluated once,
and the observation consumes the normal objective budget. It remains an
ordinary `kind: optimization`, `op: fresh` record—there is no second seed
species or scoring path. The only special policy action is deterministic:
`baseline-only` proposal generation plus a one-point deterministic receipt prevents
an acquisition prior from replacing the control. Tasks without a provided
entrypoint retain the normal `fresh` bootstrap.

Finally, a ledger record carries two different maps with different
codomains: `source_run_ids` (ancestry — which concrete candidates informed
generation, valued in `X`) and `semantic_point` (attribution — `π(x)`,
valued in `S`). `point_diff` reconstructs the semantic difference between a
child and its parents mechanically; it makes no causal claim about scores.

Non-fresh candidates also carry a `parameter_transfer` observation contract.
The candidate begins from the first (primary) parent's exact code snapshot.
The tuner pins only an applied Phase-A incumbent or a finalized-and-applied
Phase-C incumbent, projects every compatible shared parameter exactly, records
copied/reset/new/dropped keys, and evaluates that projection as mandatory warm
config 0. Config 0 is retained as a fidelity observation and objective-budget
event but excluded from `best_warm_params`, `best_warm_score`,
`final_best_score`, and `BASE_PARAMS`, even if Phase C later duplicates its
exact parameter vector; non-fresh screens therefore require at least one
additional evaluated row (`K_eval >= 2`). The parent record also
stores its exact applied params/schema and
candidate/report hashes, so a child binds to durable parent state. The
ledger captures that exact parent record in an append-only
`lineage_snapshots` receipt. A parent cannot change while a primary child is
still in flight or a scored child's binding is invalid. A terminal
`crash`/`unevaluated` child with no transfer has no parent revision to preserve.
Otherwise the parent may be tuned after the child's transfer is captured: old
children continue to validate against their historical revision, while future
children inherit the new applied incumbent. The
inherited control preserves tuning quality but cannot isolate arbitrary child
code changes: production schema-2 receipts stamp
`semantic_control.status: unverified`. A single-dimension edge becomes direct
only with a validated same-child-code control/treatment pair whose configs
differ in exactly the declared semantic switch and have no shared-key reset.
Its semantic delta is treatment minus control; later child tuning is a separate
tuning delta. Legacy, unpaired, and final-vs-final edges remain confounded.
Deep-tuning closure has one path, `finalize_tuning.py`, which renders and
validates the exact prospective candidate/report/ledger state before changing
any durable target. Before its first write the wrapper persists a small recovery
journal: a retry restores the prior candidate/report bytes when the ledger did
not commit, or keeps the consistent forward state when the ledger commit landed
but later bookkeeping failed. The old `set-tuning --mark-tuned` path is disabled.
Report-authored paired semantic controls are rejected because no deterministic
paired evaluator exists yet. The ledger and bounded target-evidence view expose
that fact through the helper-owned `direct_comparator_capability` receipt;
runtime evidence therefore abstains rather than manufacturing a signed
semantic delta.

At a strict budget boundary, `ledger.py resolve-unevaluated` resolves a pending
candidate with zero objective attempts. It proves global exhaustion and zero
candidate attempts, stores a hashed `unevaluated_receipt`, and advances the
lifecycle DAG cursor without creating score evidence. Final completion still
waits for the resulting per-round experience refresh.

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
    receipts -> observations f(x) (a crash is +inf) -> bounded schema-4
    belief regenerated over the ledger -> deterministic validated decision
    transition -> revision r+1 -> the next proposal filters/orders over
    S_{r+1}.

`ledger.dag_revision` counts graph-visible score/status changes only;
`search_space_state.revision` counts append-only decisions. The two cursors
are independent by construction.

## Bounded evidence and round-serial admission

The extractor's belief inputs come from `background_contract.py
target-evidence` (view schema 2), a deterministic scan over the persisted receipts that
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
