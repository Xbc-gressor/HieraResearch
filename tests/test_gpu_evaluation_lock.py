from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import evaluation_budget  # noqa: E402
from _common import timed_eval, timed_preflight  # noqa: E402


def _make_run(root: Path, tag: str) -> Path:
    run_dir = root / "runs" / "unit" / tag
    candidate = run_dir / "candidates" / "001" / "train.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# candidate\n")
    (run_dir / "framework_cfg.json").write_text(
        json.dumps({"max_evaluations": 1})
    )
    (run_dir / "ledger.json").write_text(json.dumps({"records": []}))
    return candidate


def _eval_worker(candidate: str, entered, release, result) -> None:
    def score(_make_model, _params):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release evaluation")
        return 0.5

    try:
        value = timed_eval(
            score,
            lambda *_args, **_kwargs: None,
            {},
            Path(candidate),
            phase="phase_a",
            method="warmstart",
        )
        result.put(("ok", value))
    except BaseException as exc:  # pragma: no cover - reported to the parent
        result.put(("error", repr(exc)))


def _subprocess_eval_worker(candidate: str) -> None:
    from _common import load_candidate_modules, resolve_score_fn

    train_module, prepare_module = load_candidate_modules(
        Path(candidate), required_symbols=("make_model",)
    )
    evaluate = resolve_score_fn(prepare_module, Path(candidate))
    timed_eval(
        evaluate,
        train_module.make_model,
        {},
        Path(candidate),
        phase="phase_a",
        method="warmstart",
    )


def _preflight_worker(candidate: str, result) -> None:
    try:
        value = timed_preflight({}, Path(candidate))
        result.put(("preflight", value))
    except BaseException as exc:  # pragma: no cover - reported to the parent
        result.put(("error", repr(exc)))


def _stop(processes, releases) -> None:
    for release in releases:
        release.set()
    for process in processes:
        process.join(3)
        if process.is_alive():
            process.terminate()
            process.join(3)


def test_concurrent_objectives_wait_before_reserving_budget(tmp_path) -> None:
    """Removing the host lease must let B reserve and enter while A is active."""
    ctx = multiprocessing.get_context("fork")
    candidate_a = _make_run(tmp_path, "a")
    candidate_b = _make_run(tmp_path, "b")
    entered_a, entered_b = ctx.Event(), ctx.Event()
    release_a, release_b = ctx.Event(), ctx.Event()
    result = ctx.Queue()
    process_a = ctx.Process(
        target=_eval_worker,
        args=(str(candidate_a), entered_a, release_a, result),
    )
    process_b = ctx.Process(
        target=_eval_worker,
        args=(str(candidate_b), entered_b, release_b, result),
    )

    process_a.start()
    assert entered_a.wait(2), "first objective never entered"
    process_b.start()
    try:
        time.sleep(0.25)
        assert not entered_b.is_set(), "second objective overlapped the first"
        attempt_log_b = candidate_b.parents[2] / evaluation_budget.ATTEMPT_LOG
        assert not attempt_log_b.exists(), "waiting objective consumed its budget slot"

        release_a.set()
        assert entered_b.wait(2), "second objective did not continue after lease release"
        release_b.set()
    finally:
        _stop((process_a, process_b), (release_a, release_b))

    assert process_a.exitcode == 0
    assert process_b.exitcode == 0
    assert sorted(result.get(timeout=1) for _ in range(2)) == [
        ("ok", 0.5),
        ("ok", 0.5),
    ]


def test_evaluation_child_retains_lease_after_parent_is_terminated(
    tmp_path, monkeypatch
) -> None:
    """Dropping fd inheritance must let B enter while A's orphan child runs."""
    ctx = multiprocessing.get_context("fork")
    candidate_a = _make_run(tmp_path, "orphan-a")
    candidate_b = _make_run(tmp_path, "orphan-b")
    run_dir_a = candidate_a.parents[2]
    (run_dir_a / "framework_cfg.json").write_text(
        json.dumps({"max_evaluations": 1, "per_runtime_limit": 5})
    )
    (candidate_a.parent / "prepare.py").write_text(
        """
import os
from pathlib import Path
import time

def evaluate_config(make_model, params):
    Path(os.environ["GPU_LOCK_TEST_CHILD_STARTED"]).write_text(str(os.getpid()))
    time.sleep(0.8)
    return 0.5
""".lstrip()
    )
    candidate_a.write_text(
        """
PARAM_SCHEMA = {}
BASE_PARAMS = {}
SEARCH_SPACE = {}

def make_model(*_args, **_kwargs):
    return None
""".lstrip()
    )
    child_started = tmp_path / "evaluation-child.pid"
    monkeypatch.setenv("GPU_LOCK_TEST_CHILD_STARTED", str(child_started))
    process_a = ctx.Process(
        target=_subprocess_eval_worker,
        args=(str(candidate_a),),
    )
    entered_b, release_b = ctx.Event(), ctx.Event()
    result = ctx.Queue()
    process_b = ctx.Process(
        target=_eval_worker,
        args=(str(candidate_b), entered_b, release_b, result),
    )

    process_a.start()
    deadline = time.monotonic() + 3
    while not child_started.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child_started.exists(), "evaluation subprocess never started"
    child_pid = int(child_started.read_text())
    process_a.terminate()
    process_a.join(2)
    assert process_a.exitcode is not None

    process_b.start()
    try:
        time.sleep(0.25)
        assert not entered_b.is_set(), "orphan evaluation child lost the GPU lease"
        assert entered_b.wait(2), "lease was not released when orphan child exited"
        release_b.set()
    finally:
        _stop((process_b,), (release_b,))
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            pass
        else:
            try:
                os.kill(child_pid, 9)
            except ProcessLookupError:
                pass

    assert process_b.exitcode == 0
    assert result.get(timeout=1) == ("ok", 0.5)


def test_preflight_and_objective_share_one_gpu_lease(tmp_path, monkeypatch) -> None:
    """Removing the preflight lease must let an objective overlap its probe."""
    ctx = multiprocessing.get_context("fork")
    run_dir = tmp_path / "runs" / "autoresearch-baseline" / "probe"
    candidate_probe = run_dir / "candidates" / "001" / "train.py"
    candidate_probe.parent.mkdir(parents=True)
    (run_dir / "framework_cfg.json").write_text(
        json.dumps({"max_evaluations": 1, "preflight_runtime_limit": 5})
    )
    (run_dir / "ledger.json").write_text(json.dumps({"records": []}))
    (candidate_probe.parent / "prepare.py").write_text(
        """
import os
from pathlib import Path
import time

def preflight_config(make_model, params):
    Path(os.environ["GPU_LOCK_TEST_PREFLIGHT_STARTED"]).write_text("started")
    release = Path(os.environ["GPU_LOCK_TEST_PREFLIGHT_RELEASE"])
    deadline = time.monotonic() + 4
    while not release.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return {"status": "ok"}

def evaluate_config(make_model, params):
    raise AssertionError("not used")
""".lstrip()
    )
    candidate_probe.write_text(
        """
PARAM_SCHEMA = {}
BASE_PARAMS = {}
SEARCH_SPACE = {}

def make_model(*_args, **_kwargs):
    return None
""".lstrip()
    )
    candidate_eval = _make_run(tmp_path, "after-probe")
    preflight_started = tmp_path / "preflight.started"
    preflight_release = tmp_path / "preflight.release"
    monkeypatch.setenv("GPU_LOCK_TEST_PREFLIGHT_STARTED", str(preflight_started))
    monkeypatch.setenv("GPU_LOCK_TEST_PREFLIGHT_RELEASE", str(preflight_release))
    result = ctx.Queue()
    process_probe = ctx.Process(
        target=_preflight_worker,
        args=(str(candidate_probe), result),
    )
    entered_eval, release_eval = ctx.Event(), ctx.Event()
    process_eval = ctx.Process(
        target=_eval_worker,
        args=(str(candidate_eval), entered_eval, release_eval, result),
    )

    process_probe.start()
    deadline = time.monotonic() + 3
    while not preflight_started.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert preflight_started.exists(), "preflight subprocess never entered"
    process_eval.start()
    try:
        time.sleep(0.25)
        assert not entered_eval.is_set(), "objective overlapped the GPU preflight"
        preflight_release.write_text("release")
        assert entered_eval.wait(2), "objective did not continue after preflight"
        release_eval.set()
    finally:
        _stop(
            (process_probe, process_eval),
            (release_eval,),
        )

    assert process_probe.exitcode == 0
    assert process_eval.exitcode == 0
    values = sorted(result.get(timeout=1) for _ in range(2))
    assert values == [("ok", 0.5), ("preflight", {"status": "ok"})]
