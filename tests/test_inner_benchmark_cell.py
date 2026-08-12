from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(ROOT))

import cell  # noqa: E402
import llm  # noqa: E402
from driver.roles import InvocationContext  # noqa: E402
from driver.session import FakeSessionRunner  # noqa: E402
from ib_support import cfg, write_checkpoint  # noqa: E402


def pool_receipt(step):
    """A valid POOL=5 receipt; lr offsets keep every step's members distinct."""
    return {
        "configs": [
            cfg(
                depth=1 + index,
                lr=0.001 + 0.0001 * index + step * 0.00001,
                dropout=0.1,
                mode="slow",
            )
            for index in range(5)
        ],
        "order": [0, 1, 2, 3, 4],
        "rationale": f"step {step}",
    }


def test_execute_cell_end_to_end_with_fake_sessions(tmp_path) -> None:
    # Default task (project None): the eval/preflight subprocesses use the
    # root interpreter. Pointing task.project at a real second project env is
    # NOT this test's job — the partition is pinned by the runner/objective
    # tests, and uv's nonexistent-project fallback differs across uv releases.
    ckpt = write_checkpoint(tmp_path)
    out = tmp_path / "out"
    fake = FakeSessionRunner([{"receipt": pool_receipt(step)} for step in range(3)])

    result = cell.execute_cell(
        arm_name="llm_pool_self_rank",
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=11,
        model="test-model",
        budget=3,
        machine="testbox",
        session_runner=fake,
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 3
    assert result["llm_calls"] == 3
    # Manifest: pinned model record (§5.3) + machine record (§八).
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model"]["model"] == "test-model"
    assert manifest["model"]["decoding"] is None
    assert manifest["model"]["sdk_version"] != "not-installed"
    assert manifest["extra"]["machine"]["label"] == "testbox"
    assert manifest["extra"]["machine"]["hostname"]
    assert manifest["arm"] == "llm_pool_self_rank"
    # Invocation context: task name falls back to "unknown" (no project),
    # tag identifies the cell.
    _, ctx = fake.calls[0]
    assert ctx.task == "unknown"
    assert ctx.tag == "ckpt--llm_pool_self_rank--s11"
    # One run_dir per cell (D16): driver artifacts gather under <out>/llm.
    assert ctx.run_dir == out / "llm"
    receipts = [
        path
        for path in (out / "llm" / "receipts").glob("*.json")
        if not path.name.endswith(".session.json")
    ]
    assert len(receipts) == 3
    # The arm-state pool persistence survives the real wiring (PLAN §6.4).
    events = [
        json.loads(line)
        for line in (out / "events.jsonl").read_text().splitlines()
    ]
    evaluations = [event for event in events if event["kind"] == "evaluation"]
    assert len(evaluations[0]["arm_state"]["pool_configs"]) == 5


def test_execute_cell_real_objective_path(tmp_path) -> None:
    """The default wiring runs the REAL objective subprocess (task.project
    None -> the cell's own interpreter, the toy task is stdlib-only)."""
    ckpt = write_checkpoint(tmp_path)
    out = tmp_path / "out"

    result = cell.execute_cell(
        arm_name="current",  # first regime: no rewarm, zero LLM calls
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=3,
        model="test-model",
        budget=2,
        session_runner=FakeSessionRunner([]),
    )

    assert result["status"] == "ok"
    assert result["evaluations"] == 2
    assert result["llm_calls"] == 0
    # The toy score_fn returns float(depth): real finite scores came back.
    assert result["best_score"] is not None


def _hook_call(hook, tool_name, file_path=None):
    input_data = {"tool_name": tool_name, "tool_input": {}}
    if file_path is not None:
        input_data["tool_input"]["file_path"] = file_path
    return anyio.run(hook, input_data, "tool-1", None)


def test_working_copy_hook_confinement(tmp_path) -> None:
    allowed = tmp_path / "work" / "train.py"
    allowed.parent.mkdir()
    allowed.write_text("BASE_PARAMS = {}\n")
    hook = cell._working_copy_hook(str(allowed))

    # Exact working copy: allowed (empty dict = no opinion).
    assert _hook_call(hook, "Edit", str(allowed)) == {}
    assert _hook_call(hook, "Read", str(allowed)) == {}
    # Dot-dot noise resolving to the same file is still the same file.
    assert _hook_call(hook, "Edit", str(allowed.parent / ".." / "work" / "train.py")) == {}
    # Anything else — frozen checkpoints included — is denied.
    verdict = _hook_call(hook, "Edit", str(tmp_path / "ckpt" / "candidate" / "train.py"))
    assert verdict["hookSpecificOutput"]["permissionDecision"] == "deny"
    verdict = _hook_call(hook, "Read", "/etc/hostname")
    assert verdict["hookSpecificOutput"]["permissionDecision"] == "deny"
    # Non-file tools pass through (the base capability hook owns those).
    assert _hook_call(hook, "Bash") == {}
    # Missing confinement path fails closed.
    closed = cell._working_copy_hook(None)
    verdict = _hook_call(closed, "Read", str(allowed))
    assert verdict["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_editor_options_carry_the_confinement_hook(tmp_path) -> None:
    events = cell.EventsLog(tmp_path / "llm")
    runner = cell.BenchSessionRunner(model="test-model", events=events)
    ctx = InvocationContext(
        task="toy-task",
        tag="toy-tag",
        run_dir=tmp_path / "llm",
        invocation_id=1,
        extra={llm.WORKING_COPY_KEY: str(tmp_path / "work" / "train.py")},
    )

    editor_options = runner._build_options(
        llm.BENCH_ROLES["bench-hillclimb-editor"], ctx, server=None
    )
    # Base capability hook + the confinement hook, in that order.
    assert len(editor_options.hooks["PreToolUse"]) == 2

    proposer_options = runner._build_options(
        llm.BENCH_ROLES["bench-pool-proposer"], ctx, server=None
    )
    assert len(proposer_options.hooks["PreToolUse"]) == 1
