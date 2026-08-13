# Autoresearch DDP

This is the multi-GPU (DDP) variant of `autoresearch-baseline`. It gives an
agent a small LLM pretraining setup and asks it to improve validation bits per
byte by editing `train.py` while keeping `prepare.py` and the evaluation metric
fixed. One evaluation trains across **`prepare.WORLD_SIZE` GPUs (default 4)**
with PyTorch DDP inside a fixed **75 s** training budget.

This variant is the **wall-clock compression** of the single-GPU task: the
seed's per-rank batch aligns the total batch (and hence per-step gradient
semantics and step count) with the single-GPU 300 s run, so 4 ranks reproduce
that run's trajectory in ~1/4 the wall-clock (75 s assumes ideal scaling;
calibrate the constant against the target machine). Scores are comparable
only at equal world size and equal budget; do not rank them against
single-GPU scores without a measured scaling anchor. The world size and the
budget are fixed task constants, not tunable coordinates.

## Goal

Minimize `val_bpb`. Lower is better.

The training script runs for a fixed 75-second training budget (wall-clock,
synchronized across ranks), excluding startup and compilation. VRAM is a soft
constraint per device: some increase is acceptable for a meaningful `val_bpb`
gain, but it should not blow up dramatically.

## Evaluation Contract

Authoritative description of how a candidate must construct, train, score, and
report. `task.toml` holds the machine-readable config (`[evaluation].score_fn`,
`[result]` metric, `[constraints]`); this section holds the prose contract. When
they disagree, `task.toml` wins for values it declares.

There is **one global `config → score` function** and **no separate official
run**: under the experiment loop a candidate is never executed as
`python train.py`. Its score is produced where `make_model` is evaluated
against that function by the tuner scripts.

**Process model.** With `WORLD_SIZE > 1`, `evaluate_config` and the two
no-score probes internally launch a torchrun process group
(`python -m torch.distributed.run --standalone --nproc_per_node=WORLD_SIZE`),
one worker process per GPU, each bound to `env.rank` / `env.world_size`. With
`WORLD_SIZE == 1` everything runs in-process, byte-identical to the
single-GPU task. The env overrides `AUTORESEARCH_DDP_WORLD_SIZE` and
`AUTORESEARCH_DDP_TORCHRUN` exist **only** for single-GPU regression of the
machinery; experiments always run at the declared world size. Accounting is
unchanged: one `score_fn` call consumes one objective slot regardless of world
size, and a non-zero worker-group exit is recorded as a crash.

- **Construct**: `train.py` exposes `make_model(env, params)` returning a
  configured **trainer object** with a `run() -> float` method, plus the tuner
  contract (`PARAM_SCHEMA`, `SEARCH_SPACE`, `BASE_PARAMS`) written by
  `tunable-contract-extractor`. The provided task-root `train.py` already
  carries `make_model` + `PARAM_SCHEMA`, and its `DEFAULT_PARAMS` hold the
  original hyperparameter values with the micro-batch re-based for the DDP
  envelope (`device_batch_size=64, grad_accum_steps=1`, total batch aligned
  with the single-GPU baseline's 524288 tokens/step). The trainer also
  exposes
  **`preflight() -> dict`**, which constructs the real model/optimizer and runs
  exactly one real-shape training step, but never invokes validation or returns
  an objective score.
- **Distributed semantics**: the trainer initializes NCCL when
  `env.world_size > 1` (`prepare.train`-side helper `_maybe_init_distributed`
  in the seed), DDP-wraps the model, and selects its device with
  `torch.cuda.set_device(env.rank)`. Model init uses `env.seed` identically on
  every rank, so ranks construct identical weights.
- **Train**: `env` is the task-defined `prepare.PretrainEnv`. The trainer
  trains only via `env.make_dataloader(tokenizer, B, T, "train")`, for at most
  `env.train_budget_seconds` (75 s) of training time — counted after warmup
  steps, excluding startup and compilation, exactly as the original script
  accounted for it. The bound dataloader automatically serves each rank a
  deterministic disjoint shard of the document stream (document-batch index ≡
  rank mod world size), so ranks never train on the same rows. The time-budget
  stop decision must be synchronized across ranks (the seed all-reduces a done
  flag each step) — a rank that breaks early or late would hang the others in
  the next gradient allreduce. Readable env attributes: `tokenizer`,
  `make_dataloader`, `evaluate_bpb`, `max_seq_len`, `train_budget_seconds`,
  `vocab_size`, `device`, `seed`, plus the DDP additions `rank`, `world_size`.
- **Score**: `evaluation.score_fn`
  (`prepare.evaluate_config(make_model, params)`) is the ONE evaluation
  surface — it runs one full budgeted DDP training run and returns the
  post-training `val_bpb`. The fixed `env.evaluate_bpb` metric runs **on every
  rank** with the unwrapped model (a DDP forward here would add pointless
  synchronization): each rank scores its deterministic disjoint shard of the
  pinned validation set, the fixed surface all-reduces the two metric sums
  (total nats, total bytes) before the division, and every rank returns the
  same full-set value. The metric definition — summed nats over summed bytes
  across the full pinned validation set — is unchanged; that value **is** the
  candidate's `final_best_score`.
- **Preflight**: before each proposed config may enter `evaluate_config`, the
  framework tuner or standalone hillclimb runner calls the fixed
  `prepare.preflight_config(make_model, params)`. Under DDP it runs as a
  worker group exactly like the score path; it calls the trainer's
  `preflight()` only, must not call `evaluate_bpb`, and the fixed wrapper
  max-reduces the reported `peak_vram_mb` across ranks. A failure creates a
  feasibility receipt and is repaired/rejected before `score_fn`, so it is
  reported separately from the objective-call budget.
- **Resource probe**: the framework additionally runs
  `prepare.resource_probe_config(make_model, params)` — the same no-score
  contract, but with the dataloader's `T` pinned to `env.max_seq_len` — to
  measure the worst-case training-shape memory envelope (max across ranks).
  Both probes are no-score and consume no objective budget.

Rules:

- Keep `prepare.py`, the tokenizer, and the evaluation metric fixed.
- Respect `env.train_budget_seconds`; derive all randomness from `env.seed`.
- Express effective batch size with the independent tuner coordinates
  `device_batch_size` and `grad_accum_steps`; derive `total_batch_size` as
  `device_batch_size * env.max_seq_len * grad_accum_steps * env.world_size`.
  `device_batch_size` is always per-rank. Do not expose `total_batch_size` as
  an independently sampled coordinate, and do not expose the world size as a
  tunable coordinate.
- VRAM is a soft constraint per device; do not blow it up dramatically.
- Do not catch broad exceptions to fabricate a score. If a candidate cannot
  build/train/return a finite `val_bpb`, let it fail so the run is recorded
  as `crash`. Any rank raising fails the whole worker group — that is the
  intended honest-failure path.
- Keep `preflight()` behaviorally aligned with the construction and first
  training step used by `run()` (including the DDP wrap); it may return
  resource telemetry such as peak VRAM, but never `val_bpb` or another
  validation-derived value. The model must be constructible at the full
  `env.max_seq_len` even when `run()` starts below it, because the framework
  probes that shape for its memory envelope.
- Keep the standalone
  `if __name__ == "__main__": evaluate_config(make_model, DEFAULT_PARAMS)`
  driver structurally unchanged. Candidate preflight reads `DEFAULT_PARAMS`, so
  standalone hyperparameter changes must edit that mapping rather than pass a
  second ad-hoc params dict only from `__main__`. Structural model/trainer
  changes remain in the code reached by `make_model`, where both paths see them.
- One candidate strategy per `train.py`; do not enumerate competing candidates.

Evaluation cost: one config eval is one full budgeted training run (~75 s of
training plus startup/compilation and the sharded final eval) across all
ranks. Startup grows with world size (each rank pays its own torch.compile
and kernel load, largely in parallel); this task's 600 s
`run.timeout_seconds` becomes the per-config `per_runtime_limit`.
`init_run.py --timeout` may override it. Consider lowering `tuner.K` /
`tuner.K_eval` in `framework_cfg.json` — the defaults cost 3 full training
runs per candidate at step 0+1.

The run-level `tools/preflight_env.py` check and candidate preflight calls do
not consume `max_evaluations` because they never enter `score_fn`. Their
attempts, failures, feasibility rejections, and runtime remain auditable
separately. Every admitted `score_fn` call—including one that later crashes—
atomically consumes one objective slot.

## Files

- `prepare.py`: fixed data prep, tokenizer, sharded dataloader, the
  `PretrainEnv` task environment, the fixed `evaluate_bpb` metric, the single
  `evaluate_config` scoring function, and the torchrun orchestration. Readonly
  during experiments.
- `train.py`: model, optimizer, hyperparameters, DDP wiring, and training loop
  behind `make_model(env, params)`; also directly runnable (see below).
- `pyproject.toml`: this task's uv environment.
- `uv.lock`: this task's locked dependency resolution.

During autonomous experiments this task uses candidate directories. Because the
task-root `train.py` is declared by `[seed].provided`, the experiment admits its
unchanged run-local copy first at the all-baselines semantic point. Its exact
`DEFAULT_PARAMS` are the sole step-0+1 baseline evaluation; the normal decoupled
tuner may later tune that same point. Otherwise `candidate-writer` writes each
candidate's `train.py` (a `fresh` candidate from scratch, or informed by parent
candidates for `improve`/`crossover`). Only run-local candidate files are edited.

## Run

Standalone: `python train.py` passes `make_model` + `DEFAULT_PARAMS` through
the same fixed `evaluate_config` score surface (launching the torchrun group
when `WORLD_SIZE > 1`) and prints the parseable summary (rank 0 only).
`autoresearch-hillclimb` first runs `tools/preflight_candidate.py` against the
working copy, atomically reserves one objective slot, and only then invokes this
entrypoint:

```bash
uv --directory tasks/autoresearch-ddp sync
uv --directory tasks/autoresearch-ddp run python prepare.py
uv --directory tasks/autoresearch-ddp run python train.py
```

Under manual mode, read the printed summary directly. Under hillclimb, redirect
it to the run log, validate the required result patterns, and append exactly one
row per reserved objective attempt to `results.tsv`; do not create a framework
ledger or invoke the legacy `parse_result.py --ledger` path.

Under the experiment loop there is **no `python train.py` run**: a candidate is
scored only where the tuner scripts call `evaluate_config`:

```bash
# Run 000 uses --provided-baseline after deterministic all-baselines admission.
python tools/new_candidate.py autoresearch-ddp <tag> 000 --provided-baseline
# Later records derive _candidate_brief.json without copying the entrypoint.
python tools/new_candidate.py autoresearch-ddp <tag> <run_id> --skip-entrypoint
# after candidate-writer + tunable-contract-extractor produce train.py + _warm_configs.json:
# (--project selects the task env without chdir, so the repo-relative paths below resolve)
uv --project tasks/autoresearch-ddp run python tools/tuners/warmstart_eval.py \
  --candidate-path   runs/autoresearch-ddp/<tag>/candidates/<run_id>/train.py \
  --configs-json     runs/autoresearch-ddp/<tag>/candidates/<run_id>/_warm_configs.json \
  --tune-report-json runs/autoresearch-ddp/<tag>/candidates/<run_id>/tune_report.json
```

When using the repository-level run layout, write logs and results under:

```text
runs/autoresearch-ddp/<tag>/
```

## Output Format

`Trainer.run()` prints a summary like this in both modes (rank 0 only):

```text
---
val_bpb:          0.997900
training_seconds: 75.1
total_seconds:    201.9
peak_vram_mb:     12400.5
mfu_percent:      38.20
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
world_size:       4
```

The key metric is `val_bpb`, and lower is better. `peak_vram_mb` is the max
across ranks; `mfu_percent`, `total_tokens_M`, and the effective batch account
for all ranks (the H100 peak-FLOPS constant is multiplied by the world size).
In standalone mode a completed run should include both `val_bpb:` and
`peak_vram_mb:` in the run log; the generic task parser is declared in
`task.toml`.

Under the experiment loop there is no log parse: `tunable-contract-extractor`
records `final_best_score` = `best_warm_score` straight into the candidate's
`ledger.json` record via `tools/ledger.py` (`record-run` + `set-tuning`); the
decoupled `tuner-orchestrator`, if it selects the candidate, lowers
`final_best_score` with the tuned best. One JSON record per run; see
`driver/prompts/rules/ledger.md`.
