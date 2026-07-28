# ES Optimization Design

A CPU-friendly **black-box optimizer design** task — deliberately NOT an
estimator-search task. The candidate is a stateful, iterative optimization
algorithm (evolution strategy, differential evolution, CMA-ES, hybrids…)
that must minimize unknown continuous functions through a budgeted number of
`evaluate()` calls. It exists to diversify the task suite's contract shape: no
`x_train/y_train`, no `.fit/.predict`, no train/test split, no accuracy — the
feedback signal is the best function value found within a fixed evaluation
budget.

## Goal

Minimize `mean_log10_1p_best_fitness`: the mean of `log10(1 + best_fitness)`
over a fixed suite of 36 problems (4 benchmark functions × dims {2, 10, 30} ×
3 fixed seeds). Every benchmark's global minimum is 0, so 0 means every problem
solved near-exactly; blind random search scores ~1–2. **Lower is better.**

Why it has headroom: the four families fail differently — rastrigin/ackley are
highly multimodal, rosenbrock is a narrow curved valley, and schwefel is
deceptive (its optimum sits at ~±421 per coordinate, far from the origin, so
naive origin-centered initialization stalls). A plain fixed-σ ES plateaus;
step-size/covariance adaptation, restarts, and budget-aware population
schedules are what move the score — so the outer idea search **and** the inner
numeric tuning both have room to matter.

## Evaluation Contract

Authoritative description of how a candidate must construct, optimize, score,
and report. `task.toml` holds the machine-readable config (`[evaluation].score_fn`,
`[result]` metric, `[constraints]`); this section holds the prose contract. When
they disagree, `task.toml` wins for values it declares.

There is **one global `config → score` function** and **no separate official
run**: a candidate is never executed as `python train.py`. Its score is
produced where `make_model` is evaluated against that function by the tuner
scripts.

- **Construct**: `train.py` exposes `make_model(problem, params)` returning a
  configured **optimizer object** with a `run() -> float` method (the best
  fitness found), plus the tuner contract (`PARAM_SCHEMA`, `SEARCH_SPACE`,
  `BASE_PARAMS`) written by `tunable-contract-extractor`.
- **Optimize**: the optimizer may probe the function **only** through
  `problem.evaluate(x)` (`x`: a sequence of length `problem.dim`), at most
  `problem.budget` times (`budget = 2000 × dim`); track `problem.evals_used`
  and plan batch sizes so it is never exceeded. All randomness must derive
  from `problem.seed` (e.g. `np.random.default_rng(problem.seed)`) so scores
  are reproducible. Readable problem attributes: `name`, `dim`, `seed`,
  `bounds` (a `(lo, hi)` pair), `budget`, `evals_used`.
- **Score**: `evaluation.score_fn` (`prepare.evaluate_config(make_model, params)`)
  is the ONE evaluation surface — it runs the optimizer on all 36 problems and
  returns the mean `log10(1 + best_fitness)` (lower is better). Its return
  value **is** the candidate's `final_best_score`.

Rules:

- Respect the evaluation budget. Exceeding it raises `RuntimeError` from
  `problem.evaluate` and the candidate crashes.
- Derive all randomness from `problem.seed`; do not reseed from the clock or
  global entropy.
- Do not exploit analytic knowledge of the benchmarks' global optima (e.g.
  returning the known optimum coordinates, or hardcoding initialization at the
  published optimum of a named function). The optimizer must find good points
  through `evaluate()` alone. Adapting *strategy* to `problem.name`/`dim`
  (algorithm selection, per-family hyperparameter branches) is allowed;
  hardcoding answers is not.
- Do not catch broad exceptions to fabricate a score. If a candidate cannot
  build/optimize/return a finite fitness, let it fail so the run is recorded
  as `crash`.
- One candidate strategy per `train.py`; do not enumerate competing candidates.

## Files

- `train.py`: no task-root baseline; `candidate-writer` writes each complete
  candidate at its validated `background.md` semantic point (`fresh` from
  scratch, or informed by numeric parents for `improve`/`crossover`).
- `prepare.py`: fixed benchmark suite, budget accounting, and the single
  `evaluate_config` scoring function. Readonly during normal experiments.
- `pyproject.toml`: task-local uv env. CPU-friendly dependency additions allowed.
- `task.toml`: machine-readable run/result contract.

## Search Space

The intended search is continuous black-box optimizer design on CPU. Fruitful
directions include:

- **(μ+λ)/(μ,λ) evolution strategies** with rank-based selection.
- **Step-size control**: 1/5th success rule, self-adaptive σ, cumulative
  step-size adaptation (CSA).
- **Covariance adaptation**: full CMA-ES, or separable/diagonal CMA for dim 30.
- **Restarts** with growing population or budget schedules (BIPOP/IPOP-like),
  and how the total budget is split across restarts.
- **Differential evolution / particle swarm** variants and their mutation,
  crossover, and inertia schedules.
- **Hybrids**: global ES/DE exploration + local pattern-search/Nelder-Mead
  refinement near the end of the budget.
- **Initialization** over the full `bounds` box (origin-centered starts fail on
  schwefel), and per-family strategy selection via `problem.name`/`dim`.

Numeric knobs the inner tuner can then exploit: population size, initial σ,
restart multiplier, selection pressure, crossover/inertia coefficients, and
the budget fraction reserved for local refinement.

Avoid: pure random search and fixed-σ (1+1)-ES without adaptation — they
plateau far from the optima, especially at dim 30.

Comparison rules: lower `mean_log10_1p_best_fitness` is better; compare
candidates across runs; prefer simpler algorithms when effectively tied.

## Run

There is **no `python train.py` run**: a candidate is scored only where the
tuner scripts call `evaluate_config`. To evaluate a candidate by hand:

```bash
uv --directory tasks/es-optimization-design sync
# Requires an existing <run_id> ledger record; also derives _candidate_brief.json.
python tools/new_candidate.py es-optimization-design <tag> <run_id> --skip-entrypoint
# after candidate-writer + tunable-contract-extractor produce train.py + _warm_configs.json:
# (--project selects the task env without chdir, so the repo-relative paths below resolve)
uv --project tasks/es-optimization-design run python tools/tuners/warmstart_eval.py \
  --candidate-path   runs/es-optimization-design/<tag>/candidates/<run_id>/train.py \
  --configs-json     runs/es-optimization-design/<tag>/candidates/<run_id>/_warm_configs.json \
  --tune-report-json runs/es-optimization-design/<tag>/candidates/<run_id>/tune_report.json
```

Normally the experiment loop drives this through its agents; see `program.md`.

## Scoring And Recording

There is no run-log summary. The tuner scripts call
`prepare.evaluate_config(make_model, params)` and the score is written straight
to `runs/es-optimization-design/<tag>/ledger.json` via `tools/ledger.py`
(`tunable-contract-extractor` records `final_best_score` = `best_warm_score`;
`tuner-orchestrator`, if it selects the candidate, lowers it with the tuned
best). A completed run is `keep` only if its `final_best_score` strictly
improves over the best previous kept value, otherwise `discard`; an unrunnable
candidate is `crash` (`+inf`). `record-run` also regenerates `loop_state.md`.
