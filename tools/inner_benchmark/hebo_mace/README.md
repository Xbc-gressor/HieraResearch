# Isolated HEBO MACE ranker env (PLAN §6.4)

Official HEBO (pinned commit `ee6112d39d1a9e9703fecaf9057193e1ec9dae72`,
`HEBO/` subdirectory of https://github.com/huawei-noah/HEBO) plus CPU-only
torch, isolated from the repo-root env because HEBO's dependency set
(torch / gpytorch / GPy / pymoo==0.6.0 / catboost, numpy<1.25 → CPython 3.11)
conflicts with it.

Setup (run once, or after pulling a new pin):

```bash
cd tools/inner_benchmark/hebo_mace
env UV_CACHE_DIR=/tmp/undo-uv-cache uv lock   # resolve + write uv.lock
env UV_CACHE_DIR=/tmp/undo-uv-cache uv sync   # install into .venv/
```

The arm (`arms/pool_hebo_mace.py`) calls per ranking step:

```bash
uv --project tools/inner_benchmark/hebo_mace run --no-sync python rank.py
```

`rank.py` reads one JSON object on stdin (`search_space` / `history` /
`pool` / `seed`) and writes one JSON object on stdout (`values`, one
larger-is-better 3-vector `[-lcb, log EI, log PI]` per pool member — see the
module docstring for the sign convention and the HEBO API path). Any failure
prints `{"error": ...}` and exits nonzero; the arm maps every failure mode to
`ArmError` (fail-fast per PLAN §6.4 — no invented "approximate HEBO").
