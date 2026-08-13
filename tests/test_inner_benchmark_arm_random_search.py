from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402
from arms import current, random_search  # noqa: E402
from ib_support import (  # noqa: E402
    evaluation_events,
    fake_eval_from,
    ok_preflight,
    write_checkpoint,
)


def reject_preflight(candidate_path, params, *, preflight_fn, per_runtime_limit, python_cmd=None):
    import objective

    return objective.PreflightOutcome(status="rejected", detail="always infeasible")


def exploding_session_factory(role_name, first_extras=None):
    raise AssertionError("random_search must never create an LLM session")


def test_first_bout_full_budget_without_llm(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)
    out = tmp_path / "out"

    result = runner.run_cell(
        arm=random_search.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=5,
        budget=6,
        eval_fn=fake_eval_from([("ok", 90.0 - index) for index in range(6)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": exploding_session_factory},
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 6
    events = evaluation_events(out)
    assert [event["source"] for event in events] == ["random"] * 6
    assert result["llm_calls"] == 0
    assert result["llm_input_tokens"] == 0


def test_same_seed_replays_proposal_sequence(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)

    def run(out_dir, seed):
        return runner.run_cell(
            arm=random_search.ARM,
            checkpoint_dir=ckpt,
            out_dir=out_dir,
            seed=seed,
            budget=5,
            eval_fn=fake_eval_from([("ok", 90.0 - index) for index in range(5)]),
            preflight_fn=ok_preflight,
        )

    out_a, out_b, out_c = (tmp_path / name for name in ("a", "b", "c"))
    assert run(out_a, seed=7)["status"] == "ok"
    assert run(out_b, seed=7)["status"] == "ok"
    assert run(out_c, seed=8)["status"] == "ok"
    proposals = [
        [event["proposal"] for event in evaluation_events(out)]
        for out in (out_a, out_b, out_c)
    ]
    encode = lambda rows: json.dumps(rows, sort_keys=True, default=str)
    assert encode(proposals[0]) == encode(proposals[1])  # reproducible per seed
    assert encode(proposals[0]) != encode(proposals[2])  # seed actually matters


def test_aligns_with_current_startup_phase(tmp_path) -> None:
    # First-bout checkpoint: 1 injected prior (the incumbent), so Current's
    # TPE stays in its RandomSampler startup fallback for all 7 proposals
    # (COMPLETE trials at ask time: 1..7 < n_startup_trials=8). TPESampler's
    # fallback is literally RandomSampler(seed=seed) consuming the same RNG
    # stream, so the sequences must align draw for draw.
    ckpt = write_checkpoint(tmp_path)

    def run(arm, out_dir):
        return runner.run_cell(
            arm=arm,
            checkpoint_dir=ckpt,
            out_dir=out_dir,
            seed=7,
            budget=7,
            eval_fn=fake_eval_from([("ok", 90.0 - index) for index in range(7)]),
            preflight_fn=ok_preflight,
        )

    out_current, out_random = tmp_path / "current", tmp_path / "random"
    assert run(current.ARM, out_current)["status"] == "ok"
    assert run(random_search.ARM, out_random)["status"] == "ok"
    encode = lambda rows: json.dumps(rows, sort_keys=True, default=str)
    proposals_current = [event["proposal"] for event in evaluation_events(out_current)]
    proposals_random = [event["proposal"] for event in evaluation_events(out_random)]
    assert encode(proposals_current) == encode(proposals_random)


def test_task_preflight_rejections_cost_no_budget_until_tripwire(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)

    result = runner.run_cell(
        arm=random_search.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=1,
        budget=4,
        eval_fn=fake_eval_from([]),  # asserts if the objective ever starts
        preflight_fn=reject_preflight,
    )

    # Rejections consume no budget; the arm keeps proposing fresh configs
    # until the RUNNER's tripwire ends the cell.
    assert result["status"] == "arm_error"
    assert "5 consecutive preflight rejections" in result["reason"]
    assert result["evaluations"] == 0
    assert result["counts"]["task_preflight_rejected"] == 5
