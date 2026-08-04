# Autoresearch Smoke

Pipeline smoke fixture for the HieraResearch framework. This is a shrunk copy
of `autoresearch-baseline` that exists to exercise the full experiment loop
(coordinator, candidate generation, preflight, warm-start eval, tuner, ledger)
cheaply on a small GPU — one evaluation is ~60 s of training plus
startup/compile/eval on a 24 GB card.

**Scores from this task are not comparable to `autoresearch-baseline`.** Do not
use them as evidence about model or policy quality; use them only as evidence
that the pipeline ran truthfully end to end.

## Deltas versus autoresearch-baseline

Everything not listed here is identical to the baseline task.

- `prepare.py`: `TIME_BUDGET = 60` (was 300), `EVAL_TOKENS = 4 * 524288`
  (was 40 units). Same `MAX_SEQ_LEN = 2048`, same data/tokenizer cache
  (`~/.cache/autoresearch`, shared with the baseline task).
- `train.py`: `DEFAULT_PARAMS` uses `depth = 4`, `device_batch_size = 48`
  (was 8 / 128) so the default config peaks well under 24 GB VRAM.
- `task.toml`: `timeout_seconds = 600` per config (was 900).

## Evaluation Contract

Same single config→score contract as the baseline task; `task.toml` holds the
machine-readable values and wins on any disagreement.

- **Construct**: `train.py` exposes `make_model(env, params)` returning a
  trainer object with `run() -> float`, plus `PARAM_SCHEMA`, `SEARCH_SPACE`,
  `BASE_PARAMS` for the tuner. The trainer exposes `preflight() -> dict`,
  which runs exactly one real-shape training step and never returns a score.
- **Train**: only via `env.make_dataloader(tokenizer, B, T, "train")`, for at
  most `env.train_budget_seconds` (60 s) after warmup, excluding startup and
  compilation.
- **Score**: `prepare.evaluate_config(make_model, params)` is the one
  evaluation surface; its returned `val_bpb` (lower is better) is the
  candidate's `final_best_score`.
- **Preflight**: `prepare.preflight_config(make_model, params)` runs in an
  isolated subprocess before any config enters `evaluate_config`; it must not
  touch validation data or emit a score, and does not consume the objective
  budget.

Rules (unchanged from baseline):

- Keep `prepare.py`, the tokenizer, and the evaluation metric fixed.
- Respect `env.train_budget_seconds`; derive all randomness from `env.seed`.
- Express effective batch size via `device_batch_size` and `grad_accum_steps`;
  never expose `total_batch_size` as a sampled coordinate.
- Do not catch broad exceptions to fabricate a score; let broken candidates
  crash so the run records `crash`.
- Keep `preflight()` behaviorally aligned with `run()`'s construction and
  first step; it may return resource telemetry but never a validation value.
- Keep the `__main__` driver
  `evaluate_config(make_model, DEFAULT_PARAMS)` structurally unchanged.
- One candidate strategy per `train.py`.

## Run

```bash
uv --directory tasks/autoresearch-smoke sync
uv --directory tasks/autoresearch-smoke run python prepare.py
uv --directory tasks/autoresearch-smoke run python train.py
```

Under the experiment loop, use the same tools as the baseline task with task
name `autoresearch-smoke`; run artifacts land under
`runs/autoresearch-smoke/<tag>/`. For a smoke run, also lower `tuner.K` /
`tuner.K_eval` and `max_evaluations` in the run's `framework_cfg.json` — the
defaults spend several full evaluations per candidate.

## Output Format

Same summary as the baseline task (`val_bpb:`, `peak_vram_mb:`, etc.); the
declared metric is `val_bpb`, lower is better. Expect peak VRAM well under
24 GB for `DEFAULT_PARAMS`; report it in the smoke run's receipts.
