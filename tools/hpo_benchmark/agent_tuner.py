"""Agent-in-the-loop HPO tuner — one ROUND per invocation (semi-manual).

The "agent" is the orchestrating Claude (no API key in env → not a background
script). Each round: (1) evaluate any agent-proposed configs, (2) run M TPE trials
seeded by ALL history so far, (3) persist state. The orchestrator reads the state
between rounds and writes the next proposals — injecting domain reasoning + escaping
BO myopia, all within the candidate's existing leaf-HP SEARCH_SPACE (no structure).

Compared head-to-head (equal total eval budget) against pure TPE from bench.py.

Round (ABS path, task env):
  uv --directory tasks/hard-interactions run python <ROOT>/tools/hpo_benchmark/agent_tuner.py \
      --candidate M4-xgb --m 6 [--propose proposals.json]
State + curve persist at runs/hard-interactions/hpo-bench/agent/<candidate>.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "hpo_benchmark"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import optuna  # noqa: E402
from bench import (  # noqa: E402
    load_candidate, _dist, _suggest, _cast_warm,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
AGENTDIR = ROOT / "runs" / "hard-interactions" / "hpo-bench" / "agent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--m", type=int, default=6, help="TPE trials this round")
    ap.add_argument("--propose", type=Path, help="JSON list of agent-proposed configs")
    args = ap.parse_args()
    AGENTDIR.mkdir(parents=True, exist_ok=True)
    statef = AGENTDIR / f"{args.candidate}.json"

    make_model, evaluate, space, warm, warm_best = load_candidate(args.candidate)
    dists = {k: _dist(v) for k, v in space.items()}

    if statef.exists():
        state = json.loads(statef.read_text())
    else:
        state = {"candidate": args.candidate, "n_dims": len(space),
                 "warm_best": warm_best,
                 "history": [{"params": p, "score": s, "src": "warm"} for p, s in warm]}

    # (1) evaluate agent proposals
    if args.propose and args.propose.exists():
        for raw in json.loads(args.propose.read_text()):
            cp = _cast_warm(raw, space)
            if cp is None:
                print(f"  skip proposal (missing keys): {raw}")
                continue
            try:
                sc = float(evaluate(make_model, cp))
            except Exception as exc:
                print(f"  proposal failed {type(exc).__name__}: {exc}")
                continue
            state["history"].append({"params": cp, "score": sc, "src": "agent"})
            print(f"  agent config → {sc:.4f}")

    # (2) M TPE trials seeded by all history
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=len(state["history"])))
    for h in state["history"]:
        cp = _cast_warm(h["params"], space)
        if cp is not None and isinstance(h.get("score"), (int, float)):
            study.add_trial(optuna.trial.create_trial(
                params=cp, distributions=dists, value=float(h["score"])))
    t0 = time.time()
    study.optimize(lambda t: float(evaluate(make_model, _suggest(t, space))),
                   n_trials=args.m, gc_after_trial=True)
    for t in study.trials[len(study.trials) - args.m:]:
        if t.value is not None:
            state["history"].append({"params": t.params, "score": t.value, "src": "tpe"})

    statef.write_text(json.dumps(state, indent=2))
    scores = [h["score"] for h in state["history"] if isinstance(h.get("score"), (int, float))]
    best = min(scores)
    by_agent = [h["score"] for h in state["history"] if h["src"] == "agent"]
    print(f"\n{args.candidate}: rounds-total={len(state['history'])} evals | "
          f"warm_best={warm_best:.4f} → best={best:.4f} "
          f"({'+'+format(warm_best-best,'.4f') if best<warm_best else '~0'}) "
          f"[{time.time()-t0:.0f}s this round]")
    if by_agent:
        print(f"  agent-proposed best so far: {min(by_agent):.4f} "
              f"({sum(1 for s in by_agent if s < warm_best)}/{len(by_agent)} beat warm)")
    # show recent history tail for the orchestrator to reason over
    print("  recent (src/score):", [(h['src'], round(h['score'], 4))
                                     for h in state['history'][-8:] if isinstance(h.get('score'), (int, float))])


if __name__ == "__main__":
    main()
