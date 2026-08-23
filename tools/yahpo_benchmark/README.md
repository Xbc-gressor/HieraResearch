# Standalone YAHPO pilot

This runner compares `hebo_only`, `hebo_mace_llm_pool`, and
`llambo_modern_batched` from the same five counted random initial evaluations.
It does not use the production scheduler or inner-benchmark checkpoints.

Obtain the official `yahpo_data` v1.0.2 directory separately, then run the
discarded two-task smoke:

```bash
uv run python -m tools.yahpo_benchmark smoke \
  --data-path /path/to/yahpo_data \
  --output-dir runs/yahpo-smoke
```

If `runs/yahpo-smoke/summary.json` projects no more than the $125 hard cap,
run the eight-task pilot:

```bash
uv run python -m tools.yahpo_benchmark pilot \
  --data-path /path/to/yahpo_data \
  --smoke-summary runs/yahpo-smoke/summary.json \
  --output-dir runs/yahpo-pilot
```

Both output directories must be empty. Results are written incrementally to
`cells/<task>/<optimizer>/result.json`; `summary.json` contains endpoint
normalized regret, BO-phase AUC, paired comparisons, calls, tokens, and USD.
