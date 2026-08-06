# Single-GPU Evaluation Lock Design

## Problem

An experiment round may admit multiple semantic candidates. Although
`warmstart_eval.py` evaluates one candidate's selected warm configurations
sequentially, the run coordinator can start more than one
`tunable-contract-extractor` at once. Each extractor then launches its own
detached evaluator, so two candidates can execute GPU probes or objective
evaluations concurrently on the repository's single GPU.

Run `0806-sn-pt125-2` demonstrated the failure: evaluation reservations for
candidates `005` and `006` interleaved, both warm-start jobs reported the same
743.3-second elapsed interval, and candidate `005` failed with only 3.14 GiB
free on an 79.25 GiB GPU while another evaluation occupied approximately the
other half. Concurrent execution can both cause OOM and invalidate successful
scores by reducing fixed-time training throughput.

The coordinator prompt already says to process generated actions in order, but
prompt-only serialization is not a correctness boundary. The evaluation-budget
lock serializes reservations, not GPU use, and the existing Phase-C lock is
candidate-local.

## Requirements

- At most one HieraResearch GPU probe or objective evaluation may execute on
  the host at a time.
- Warm-start, BO, CMA-ES, grid search, correctness preflight, and resource probe
  must share the same admission boundary.
- Lock contention must wait. It must not produce a crash, blocked run, failed
  preflight, consumed objective slot, or tuner timeout.
- An objective slot is reserved only after the caller owns the GPU lease.
- The per-evaluation and preflight runtime limits begin only after admission;
  queue time is not execution time.
- Process exit and forced termination must release the lease without manual
  cleanup or stale-lock recovery.
- If an evaluator parent exits while its evaluation child remains alive, the
  child must retain the lease until it exits.
- The change must not alter score calculation, budget limits, candidate
  selection, or persisted tuning contracts.

## Chosen Design

Add a host-wide, per-user POSIX `flock` lease in `tools/tuners/_common.py`, the
shared boundary already used by all framework tuner backends.

The lock file has one stable path in the system temporary directory, scoped by
the current numeric user id. It is intentionally shared across repository
checkouts, run tags, candidates, and tuner methods on the host. The project is
currently single-GPU, so no device-selection abstraction is introduced.

`timed_preflight()` acquires the lease before starting `_preflight_one.py` and
holds it until that subprocess exits. This covers both correctness probes and
resource probes.

`timed_eval()` acquires the same lease before calling `reserve_evaluation()`.
It holds the lease through either the in-process score path or the complete
`_eval_one.py` subprocess lifetime. Consequently, waiting neither burns an
objective slot nor consumes the configured execution timeout.

On POSIX subprocess paths, the lock file descriptor is explicitly inherited by
the child. If the evaluator parent is terminated while the child continues,
the child's open descriptor keeps the `flock` active. Normal child completion
or process-group termination closes the final descriptor and releases the lock
automatically.

Lock acquisition is blocking and has no application-level timeout. A living
evaluation is normal queue contention, not a blocked experiment. No lock-wait
exception or ledger state is introduced.

The experiment-agent contract will continue to require candidates to be
processed in order. This reduces idle queued extractor processes, but safety
depends only on the deterministic evaluator lease.

## Alternatives Rejected

### Prompt-only serialization

Restating sequential orchestration is insufficient because the observed run
already violated that instruction. It also cannot coordinate independent run
sessions.

### Force generation batch size to one

Setting `B=1` reduces one source of overlap but changes the search policy and
does not prevent independent runs from sharing the GPU.

### Candidate-local or run-local locks

These prevent duplicate work inside one candidate or tag but allow exactly the
cross-candidate and cross-run contention that caused the incident.

### Central scheduler service

A daemon or queue would add lifecycle and recovery complexity without benefit
for a single GPU. Kernel-managed advisory locking supplies the required
serialization and crash cleanup directly.

## Failure Semantics

- A caller waiting for the lease remains alive and continues after the current
  holder exits.
- A holder that exits normally, raises, times out, or is killed releases its
  descriptor automatically.
- Lock-file contents carry no state and are never interpreted, so an old empty
  file is harmless.
- Failure to open or use the lock is a framework-environment error, not an
  objective attempt, because it occurs before reservation. This is distinct
  from ordinary contention, which only waits.
- Non-POSIX platforms fail before GPU execution rather than silently running
  concurrently. The supported experiment runtime is POSIX.

## Tests

Tests use real subprocesses and a temporary lock path override so they exercise
kernel lock behavior without touching the host's production lease.

1. Two independent processes enter the evaluation boundary; their GPU critical
   sections must not overlap.
2. While process A holds the lease, process B must not append an objective
   reservation. After A exits, B must reserve and complete normally.
3. A preflight holder and an objective caller must serialize on the same lease.
4. Terminating the evaluator parent while its child remains alive must not let
   a second caller enter until the child exits.
5. Existing targeted evaluation, preflight, tuner, and full unit suites must
   remain green.

## Scope

This change fixes framework GPU admission only. It does not reinterpret the
scores already recorded by affected runs, add multi-GPU scheduling, change
outer-loop concurrency, or modify benchmark/task code.
