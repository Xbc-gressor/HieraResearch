# The gap to the hillclimb baseline

Diagnosis of why the search-space loop lost to `autoresearch-hillclimb`, and
what has been done about it. The workspace `CLAUDE.md` Current Phase section
summarizes this file; the ranked detail lives here.

**This file dates from 2026-07 and records what was true when written.** Every
status line below is a claim about code, so check the code before treating an
item as open — `git -C HieraResearch log`, the named helper, and the named test
outrank this document.

## Reference evidence

| run | budget | result |
|---|---|---|
| `~/Hidden/remoteZ/runs/autoresearch-baseline/0729-ds-ex100-0` | `max_evaluations=100`, ~15 h wall-clock | best **1.0432** val_bpb — entirely from BO-tuning the provided baseline; 0 of 16 generated candidates survived |
| `~/Hidden/remoteA` (hillclimb) | 50 evaluations | scored **better** |

The win condition is beating the hillclimb run's final score at matched
evaluation budget — not improving an internal metric.

## Ranked findings

### 1. Candidate fidelity collapse (root cause)

Generated candidates were from-scratch rewrites of `train.py` from an idea
text; children inherited neither the parent's implementation nor, in effect,
its tuned parameters. Run 013 carried 000's tuned config 0 yet scored 1.224 vs
1.0432, because the rewrite dropped `torch.compile` and reset tuned defaults.
Rewrite noise (~0.1–0.7 bpb) swamped semantic effects (~0.01–0.05).

**Status: mitigated, unvalidated.** Non-fresh candidates start from a
byte-for-byte helper-pinned copy of the primary parent's `train.py` and edit it
in place — `candidate-writer` write mode 2,
`implementation_source.kind: primary_parent_snapshot`,
`validate_parameter_transfer_binding`. Only `fresh` candidates are written from
scratch. No run has yet tested this.

### 2. No budget governance — largest remaining design gap

The screening/tuning split was emergent: 34 screening evals across 17
candidates, then 66 BO evals on 3 of them. The deep-tune gate
(`tune_tools.py select_candidate`) is budget-blind and ranks by warm score.
There is no wall-clock guard at round or run level; one tuning round ran 3.4 h.

**Status: mitigated, unvalidated.** Phase C now has a reservation-enforced
run share (`deep_tune_budget_fraction`), a per-candidate objective cap, and a
cumulative candidate wall-clock cap. `select-candidate` emits the remaining
allocation and every objective call still passes the atomic global guard.
The defaults reserve at least 60% of a bounded run for screening, but no fresh
matched-budget run has yet shown that the 40%/20-trial/3600-second defaults beat
hillclimb.

### 3. BO deployment defects

The algorithm itself produced the run's only gains. Two corrections to the
original analysis: constrained TPE and the `infeasible_value` penalty were
committed **before** the run (`262f312`) and active during it, so failures were
*not* invisible to the surrogate; and patience *did* reset on resume.

The ~18 repeated OOM proposals happened anyway because failed trials were not
re-injected on resume (`read_prior_trials` keeps only finite-score trials),
each fresh study burned 10 random startup trials, and the agent
killed/restarted the study ad hoc.

**Status: fixed, unvalidated in a fresh run.** `prior_patience_state` persists
best *and* streak across restarts, and failed trials increment the counter
(`_common.py`). `read_prior_infeasible_trials` separately restores persisted
config-infeasible crashes and preflight rejections as constrained observations
before deferred configs are queued, so a known-crashed deferred point is not
retried on restart.

### 4. Belief epistemics inverted

The ungated free-text `experience.summary` overclaimed ("strongest negative
signal" without a comparator edge) and became signed acquisition adjustments.
The snapshot went stale — generation 2 at run 006, with a prompt-only refresh
cadence.

**Status: mitigated.** Generic prose is display-only and never an acquisition
input; an empty `summary` is a valid abstention; belief payloads are preserved
byte-for-byte on no-op deltas, so `generation` increments only on real change.
Direct comparators require a validated same-child-code control/treatment pair
with a pinned parent snapshot — final-vs-final, reset-bearing, unpaired, and
independently tuned comparisons stay confounded. `promising`/`unpromising`
requires mechanical `comparator_covered` with ≥2 direct non-crash edges.
Refresh is triggered by the deterministic `experience_refresh_required` brief
flag. `semantic_search.llm_intelligence_score` can down-weight LLM-authored
forecasts. The paired evaluator itself is not implemented: production ledgers
now expose `direct_comparator_capability.status: unavailable`, and that gate
keeps the downstream direct-coverage/gain-direction branch dormant instead of
presenting fixture-only behavior as a runtime capability.

### 5. Screening statistic reified

`final_best_score = best-of-K_eval` warm configs was non-commensurable across
candidates: `K_eval ∈ {1,2}`, K agent-chosen, single seed, no noise model.
Keep/discard compared untuned warm scores against a bar lowered by tuning,
which made the later discards predetermined. Same-point duplicates put
implementation spread at ~0.5–0.7 bpb — far above believed effects — and
nothing computed it.

**Status: partially addressed.** Under byte-copy inheritance, config 0 still
evaluates the parent's applied parameters on child code as a fidelity
observation, but it is no longer eligible for `best_warm_params`,
`best_warm_score`, `final_best_score`, or `BASE_PARAMS`. Non-fresh screening
therefore requires `K_eval ≥ 2`, leaving at least one selectable row beyond the
control; randomized warm screening landed pre-run in `8c94676`. **Still
missing:** a noise model (replicate seeds, same-point variance estimation).
The best selectable single-seed warm row is still reified as
`final_best_score`, so sub-noise improvements outside the excluded control can
still become `keep`.

### 6. Over-engineering pattern

Rigor went to the components nearest to working — screening statistics,
deferral accounting, where K/K_eval changed no verdict in the run — while the
broken ones (fidelity, budget policy) were left to prompt discipline.

**Status: standing warning.** The mitigation pass redirects rigor correctly;
keep applying it to budget governance next.

## Immediate next steps

1. Land and commit the in-tree mitigations.
2. Validate them in a fresh run at matched budget.

Do not pull a deferred mechanism (P3/P4 — bottleneck retrieval, space
expansion, convergence/regret study) forward by silently inventing contracts
that its prerequisites have not established.
