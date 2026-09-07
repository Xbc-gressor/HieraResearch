from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import donor_history  # noqa: E402
import runner  # noqa: E402
import space  # noqa: E402
from arms import rgpe_hebo  # noqa: E402
from ib_support import (  # noqa: E402
    cfg,
    evaluation_events,
    fake_eval_from,
    hrow,
    ok_preflight,
    write_checkpoint,
)

# Donor: `depth` shared by name, `step` aligns to `lr` by canonical role and is
# coupled with `epochs` (which the recipient lacks) -> the whole group must
# drop; `mode` shared by name; `width` is donor-only.
DONOR_SOURCE = '''
PARAM_SCHEMA = {
    "depth": "int",
    "step": ("float", "log"),
    "epochs": "int",
    "width": "int",
    "mode": ("categorical", ["fast", "slow", "odd"]),
}
SEARCH_SPACE = {
    "depth": ("int", 1, 16),
    "step": ("float", 0.0001, 0.1, "log"),
    "epochs": ("int", 1, 100),
    "width": ("int", 8, 64),
    "mode": ("categorical", ["fast", "slow", "odd"]),
}
BASE_PARAMS = {"depth": 4, "step": 0.001, "epochs": 10, "width": 16, "mode": "fast"}
def make_model(params):
    return dict(params)
'''

RECIPIENT_ANN = {
    "depth": {"canonical": "max_tree_depth", "couples_with": []},
    "lr": {"canonical": "learning_rate", "couples_with": []},
    "dropout": {"canonical": "dropout", "couples_with": []},
    "mode": {"canonical": "mode", "couples_with": []},
}
DONOR_ANN = {
    "depth": {"canonical": "max_tree_depth", "couples_with": []},
    "step": {"canonical": "learning_rate", "couples_with": ["epochs"]},
    "epochs": {"canonical": "n_rounds", "couples_with": ["step"]},
    "width": {"canonical": "width", "couples_with": []},
    "mode": {"canonical": "mode", "couples_with": []},
}


def load_suggest_rgpe():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "suggest_rgpe", ROOT / "tools" / "inner_benchmark" / "hebo_mace" / "suggest_rgpe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def donor_rows(n=8):
    rows = []
    for i in range(n):
        params = {"depth": 1 + 2 * i, "step": 0.001 * (i + 1), "epochs": 5 + i,
                  "width": 16, "mode": ["fast", "slow", "odd"][i % 3]}
        rows.append((params, 100.0 - i))
    return rows


def test_base_task_alignment_projection_and_standardization(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)
    recipient = space.read_contract(ckpt / "candidate" / "train.py")
    donor_path = tmp_path / "donor.py"
    donor_path.write_text(DONOR_SOURCE)
    donor = space.read_contract(donor_path)

    align = donor_history.align_dimensions(recipient, donor, RECIPIENT_ANN, DONOR_ANN)
    # depth by name, mode by name; step->lr aligned by role but its donor group
    # (step, epochs) cannot land entirely -> dropped.
    assert align == {"depth": "depth", "mode": "mode"}

    tasks = donor_history.build_base_tasks(
        recipient,
        [{"name": "d", "contract": donor, "rows": donor_rows(), "annotation": DONOR_ANN}],
        recipient_ann=RECIPIENT_ANN,
    )
    assert len(tasks) == 1
    task = tasks[0]
    assert task["shared_dims"] == ["depth", "mode"]
    # rows with mode="odd" (not a recipient option) are dropped: 8 rows, i%3==2 -> 2 dropped
    assert len(task["points"]) == 6
    # depth clipped into the recipient's [1, 8]
    assert all(1 <= p["params"]["depth"] <= 8 for p in task["points"])
    assert set(task["points"][0]["params"]) == {"depth", "mode"}
    ys = [p["y_std"] for p in task["points"]]
    assert abs(sum(ys)) < 1e-9
    assert abs(sum(v * v for v in ys) / len(ys) - 1.0) < 1e-9

    # Too few distinct points -> not modelled.
    assert donor_history.build_base_tasks(
        recipient,
        [{"name": "d", "contract": donor, "rows": donor_rows(3), "annotation": DONOR_ANN}],
        recipient_ann=RECIPIENT_ANN,
    ) == []


def fake_suggest_fn(calls):
    def suggest(*, search_space, history, seed, scramble_seed, quasi_index,
                initial_suggest_extra, base_tasks, rgpe_horizon, base_cache_key):
        calls.append({"history": history, "base_tasks": base_tasks,
                      "rgpe_horizon": rgpe_horizon, "base_cache_key": base_cache_key})
        n = len(calls)
        params = cfg(depth=(n % 8) + 1, lr=0.002 + 0.0001 * n, dropout=0.3 + 0.01 * n)
        if len(history) < 3 and base_tasks:
            return {"suggestion": params, "mode": "quasi", "quasi_consumed": 1}
        return {
            "suggestion": params, "mode": "surrogate", "quasi_consumed": 0,
            "front_size": 11,
            "rgpe": {"weights": {"d": 0.4, "__target__": 0.6}, "dropped": [],
                     "n_target": len(history), "horizon": rgpe_horizon},
        }
    return suggest


def test_arm_binds_base_tasks_and_records_weights(tmp_path) -> None:
    ckpt = write_checkpoint(
        tmp_path,
        history=[hrow(cfg(depth=1, lr=0.0002, dropout=0.05, mode="slow"), 95.0)],
    )
    base_tasks = [{"name": "d", "shared_dims": ["depth"],
                   "points": [{"params": {"depth": i}, "y_std": 0.1 * i} for i in range(6)]}]
    calls = []
    result = runner.run_cell(
        arm=rgpe_hebo.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=3,
        budget=4,
        eval_fn=fake_eval_from([("ok", 90.0 - i) for i in range(4)]),
        preflight_fn=ok_preflight,
        extras={"hebo_suggest_fn": fake_suggest_fn(calls), "base_tasks": base_tasks},
    )
    assert result["status"] == "ok"
    assert result["evaluations"] == 4
    # H defaults to initial finite-unique history (2) + budget (4).
    assert all(call["rgpe_horizon"] == 6 for call in calls)
    assert all(call["base_tasks"] == base_tasks for call in calls)
    assert calls[0]["base_cache_key"] == "ckpt:3"
    events = evaluation_events(tmp_path / "out")
    assert [e["source"] for e in events] == ["hebo_quasi"] + ["hebo_suggest"] * 3
    assert "rgpe" not in events[0]["arm_state"]
    assert events[1]["arm_state"]["rgpe"]["weights"] == {"d": 0.4, "__target__": 0.6}
    assert all(e["arm_state"]["n_base_tasks"] == 1 for e in events)


def test_rgpe_suggest_real_hebo_smoke() -> None:
    pytest.importorskip("hebo", reason="needs the pinned HEBO environment")
    mod = load_suggest_rgpe()

    search_space = {
        "depth": ["int", 1, 8],
        "lr": ["float", 0.0001, 0.1, "log"],
        "mode": ["categorical", ["fast", "slow"]],
    }
    history = [
        {"params": {"depth": 4, "lr": 0.001, "mode": "fast"}, "score": 100.0},
        {"params": {"depth": 1, "lr": 0.0002, "mode": "slow"}, "score": 95.0},
        {"params": {"depth": 2, "lr": 0.0003, "mode": "slow"}, "score": 96.0},
        {"params": {"depth": 5, "lr": 0.0005, "mode": "slow"}, "score": 92.0},
    ]
    # A base task that ranks the target history perfectly (y decreasing in depth)
    # and one that is pure noise on a single shared dim.
    good = {"name": "good", "shared_dims": ["depth", "lr"],
            "points": [{"params": {"depth": d, "lr": 0.0001 * (d + 1)}, "y_std": 1.5 - 0.4 * d}
                       for d in range(1, 8)]}
    bad = {"name": "bad", "shared_dims": ["mode"],
           "points": [{"params": {"mode": ["fast", "slow"][i % 2]}, "y_std": (-1) ** (i // 2) * 0.7}
                      for i in range(6)]}
    payload = {
        "search_space": search_space, "history": history, "seed": 11,
        "scramble_seed": 5, "quasi_index": 4, "initial_suggest_extra": [],
        "base_tasks": [good, bad], "rgpe_horizon": 20, "rgpe_bootstrap": 200,
        "base_cache_key": "smoke",
    }
    result = mod.compute(payload)
    assert result["mode"] == "surrogate"
    weights = result["rgpe"]["weights"]
    assert set(weights) == {"good", "bad", "__target__"}
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    s = result["suggestion"]
    assert 1 <= s["depth"] <= 8 and 0.0001 <= s["lr"] <= 0.1 and s["mode"] in ("fast", "slow")
    # Below the RGPE warmup the seam stays in quasi mode even with base tasks.
    short = dict(payload, history=history[:2])
    assert mod.compute(short)["mode"] == "quasi"


def test_sane_gp_bounds_degenerate_surrogate() -> None:
    """Starting the surrogate at RGPE_WARMUP reaches a HEBO fit the official
    ``1 + num_paras`` warmup never does: a mean spiked far outside the observed
    y band, the variance at the noise floor, and outright NaN on a fraction of
    calls. MACE's standardized improvement then diverged and cost the whole cell.
    """
    torch = pytest.importorskip("torch")
    mod = load_suggest_rgpe()

    class Spiked:
        noise = 0.001

        def predict(self, xc, xe):
            return (torch.tensor([[-352.0], [0.5], [float("nan")]]),
                    torch.tensor([[1e-9], [1.6], [float("nan")]]))

    y = torch.tensor([[-1.2], [0.3], [1.5]])  # span 2.7
    guarded = mod._SaneGP(Spiked(), y)
    mu, var = guarded.predict(None, None)

    assert torch.isfinite(mu).all() and torch.isfinite(var).all()
    assert [float(v) for v in mu.flatten()] == pytest.approx([-3.9, 0.5, 4.2])
    floor = (0.01 * 2.7) ** 2
    assert [float(v) for v in var.flatten()] == pytest.approx([floor, 1.6, floor])
    assert guarded.noise == 0.001
