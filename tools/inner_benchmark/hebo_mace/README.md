# HEBO MACE ranker (PLAN §6.4)

Official HEBO (pinned commit `ee6112d39d1a9e9703fecaf9057193e1ec9dae72`,
`HEBO/` subdirectory of https://github.com/huawei-noah/HEBO) plus CPU-only
torch. The dependencies are part of the repository-root lockfile; HEBO's
`numpy<1.25` constraint pins that environment to CPython 3.11.

Setup from the repository root:

```bash
uv sync --frozen
```

The arm (`arms/pool_hebo_mace.py`) calls per ranking step:

```bash
<root .venv python> tools/inner_benchmark/hebo_mace/rank.py
```

`rank.py` reads one JSON object on stdin (`search_space` / `history` /
`pool` / `seed`) and writes one JSON object on stdout (`values`, one
larger-is-better 3-vector `[-lcb, log EI, log PI]` per pool member — see the
module docstring for the sign convention and the HEBO API path). Any failure
prints `{"error": ...}` and exits nonzero; the arm maps every failure mode to
`ArmError` (fail-fast per PLAN §6.4 — no invented "approximate HEBO").
