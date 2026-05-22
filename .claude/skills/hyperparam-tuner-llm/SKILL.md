---
name: hyperparam-tuner-llm
description: Phase A warm-start method for the `tuner-orchestrator` agent. Reads the candidate's `SEARCH_SPACE`, dataset characteristics from `prepare.py`, and prior ledger entries; proposes K = 5 deliberately diverse hyperparameter configurations to seed the subsequent real-search Phase C method (grid / bo / cmaes). The orchestrator hands these 5 configs to `tools/tuners/warmstart_eval.py` for evaluation on the test split; the resulting trial set is then visible to BO via prior injection, to CMA-ES as the source of its initial mean, and to the Phase B percentile decision.
metadata:
  short-description: Propose K=5 diverse warm-start configs (Phase A)
---

# Hyperparam Tuner — LLM Warm-start

Methodology for proposing **5 diverse warm-start configurations** that the
`tuner-orchestrator` agent will evaluate in Phase A. Stays in the main
Claude's context. Pure reasoning — no trials are run by this skill; the
evaluation happens in `warmstart_eval.py` after the orchestrator collects
your 5 configs.

The output is the **seed set** for everything that follows in this
candidate's tuning:

- One of the 5 (the highest scorer) becomes `best_warm_score`, used by
  Phase B's cross-idea percentile decision.
- All 5 are injected into Optuna as completed trials when Phase C picks
  `bo`, accelerating TPE convergence.
- The highest scorer is used as the CMA-ES initial mean (x0) when Phase C
  picks `cmaes`.

Diversity therefore matters more than raw quality: 5 clustered configs
waste the seed budget; 5 well-spread ones make the downstream search
sample-efficient.

## Inputs To Read

1. The candidate's `train.py` — find these symbols (the candidate-writer
   contract guarantees them):
   - `BASE_PARAMS: dict` — current defaults (one of your 5 should be near
     this, possibly with minor adjustments).
   - `SEARCH_SPACE: dict` — declared ranges, one entry per tunable key.
   - `make_model(dataset, params)` — model factory.
2. `tasks/<task>/prepare.py` — dataset characteristics only:
   sample counts, feature counts, class counts, class balance, any
   preprocessing applied before `train.py` sees data.
3. `runs/<task>/<tag>/results.tsv` — entries from the same model family:
   what `BASE_PARAMS` neighborhoods have been tried, what scored well.

Skip everything else. Do not read other candidates' `train.py` in detail.

## Diversity Strategy

Aim for the 5 configs to cover meaningfully different regions of
`SEARCH_SPACE`. Concretely, pick one config along each of these directions
(adjust to fit the SEARCH_SPACE you actually see):

1. **Near BASE_PARAMS** — a small perturbation of the candidate-writer's
   defaults. Safety anchor.
2. **Capacity-up** — higher complexity / less regularization (deeper
   trees, more estimators, weaker shrinkage, larger hidden units).
3. **Capacity-down** — lower complexity / more regularization (shallower,
   fewer estimators, stronger shrinkage / L1 / L2, larger min_samples_leaf).
4. **Learning-rate extreme** — for boosted / iterative models, push lr
   toward one log-scale extreme that the search space allows.
5. **Categorical pivot** — if SEARCH_SPACE has categorical keys, vary the
   most impactful one; otherwise pick another continuous direction not
   covered by 1–4 (e.g., a different subsample / colsample ratio).

When the SEARCH_SPACE has only 2–3 keys, the 5 directions overlap; pick
configurations spread maximally apart in the lower-dimensional space
instead of forcing 5 distinct themes.

## Hard Rules

1. **Stay strictly inside `SEARCH_SPACE`.** Every value must be within
   declared bounds (or in the categorical option list). The orchestrator
   sanity-checks before passing to warmstart_eval.py; out-of-bound configs
   are rejected.
2. **Every key in `BASE_PARAMS` must appear in every config.** No missing
   keys, no extra keys. Use BASE_PARAMS's value as the fallback when a
   direction doesn't sensibly perturb a particular key.
3. **No duplicates.** All 5 configs must differ from each other and from
   BASE_PARAMS in at least one key.
4. **No silent new hyperparameter keys** that `make_model` does not
   consume.

## Output Format

Return exactly this block — no extra prose, no markdown around it:

```text
recommended_configs:
  - {<key>: <value>, <key>: <value>, ...}    # near-baseline
  - {<key>: <value>, <key>: <value>, ...}    # capacity-up
  - {<key>: <value>, <key>: <value>, ...}    # capacity-down
  - {<key>: <value>, <key>: <value>, ...}    # lr-extreme (or another direction)
  - {<key>: <value>, <key>: <value>, ...}    # categorical-pivot (or another direction)
rationale: <2–4 sentences: which signals (ledger / dataset / search space)
            shaped the diversity choice>
risks:     <one short line; "none notable" allowed>
confidence: <high | medium | low>
```

The orchestrator will serialize `recommended_configs` to JSON as a list of
5 dicts and pass that to `warmstart_eval.py`. Make sure each dict literal
is valid Python (no trailing commas inside, all keys quoted as strings or
bare identifiers consistently).

`confidence` reflects how informed the proposal is:
- `high` — multiple ledger entries for similar model families with clear
  gradients informing the directions.
- `medium` — some signal from ledger or dataset shape, but not strong.
- `low` — no relevant ledger history, blind diversity sampling.

## Boundaries

- **Do not write code.** You produce a config list; warmstart_eval.py
  evaluates it.
- **Do not call `test_accuracy()` or any evaluation function** from this
  skill's context. Evaluation is the script's job.
- **Do not edit `train.py`, `prepare.py`, or any file.** The orchestrator
  decides what gets applied to `BASE_PARAMS` after Phase C.
- **Do not skip diversity to chase one direction.** If you suspect one
  direction is clearly best, encode that as 2 of the 5 configs with
  different parameter combinations, not 5 copies.
