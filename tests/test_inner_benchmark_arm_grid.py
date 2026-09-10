from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402
import space as space_mod  # noqa: E402
from arms import grid  # noqa: E402
from ib_support import (  # noqa: E402
    evaluation_events,
    fake_eval_by_params,
    hrow,
    ok_preflight,
    write_checkpoint,
)

SPOOKY_SOURCE = '''
PARAM_SCHEMA = {"alpha": ("float", "log"), "max_ngram": "int"}
BASE_PARAMS = {"alpha": 1.0, "max_ngram": 3}
SEARCH_SPACE = {"alpha": ("float", 0.01, 2.0, "log"), "max_ngram": ("int", 1, 3)}
def make_model(params):
    return params
'''


def _score(params):
    return abs(math.log10(params["alpha"]) + 1) + 0.1 * params["max_ngram"]


def test_grid_covers_budget_then_refines(tmp_path) -> None:
    probe = write_checkpoint(tmp_path, name="probe", train_source=SPOOKY_SOURCE)
    contract = space_mod.read_contract(probe / "candidate" / "train.py")
    alpha = next(d for d in contract.dimensions if d.name == "alpha")
    # B=24 over {1,2,3} -> 8 log levels. Pre-execute the max_ngram=1 row of
    # the level-0 grid so only 16 fresh combos remain and refinement must
    # supply the other 8.
    history = [
        hrow({"alpha": a, "max_ngram": 1}, 0.9 + i * 0.001)
        for i, a in enumerate(grid._numeric_levels(alpha, 8))
    ]
    ckpt = write_checkpoint(
        tmp_path, name="spooky", train_source=SPOOKY_SOURCE, history=history,
        incumbent_params={"alpha": 1.0, "max_ngram": 3}, incumbent_score=0.7,
    )
    result = runner.run_cell(
        arm=grid.ARM, checkpoint_dir=ckpt, out_dir=tmp_path / "out", seed=1,
        budget=24, eval_fn=fake_eval_by_params(_score), preflight_fn=ok_preflight,
    )
    events = evaluation_events(tmp_path / "out")
    assert result["status"] == "ok" and result["evaluations"] == 24
    levels = [event["arm_state"]["grid_level"] for event in events]
    assert levels == [0] * 16 + [1] * 8
    configs = {(event["proposal"]["alpha"], event["proposal"]["max_ngram"]) for event in events}
    assert len(configs) == 24
    assert all(0.01 <= a <= 2.0 and n in (1, 2, 3) for a, n in configs)
