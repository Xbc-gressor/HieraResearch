from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402
from arms import llm_pool_self_rank  # noqa: E402
from ib_support import (  # noqa: E402
    BASE,
    cfg,
    evaluation_events,
    fake_eval_from,
    hrow,
    ok_preflight,
    write_checkpoint,
)


class FakeSession:
    """Scripted BoutSession stand-in: receipts (or exceptions) in call order."""

    def __init__(self, script):
        self._script = list(script)
        self.asks = []

    def ask(self, extra=None):
        self.asks.append(extra)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def totals(self):
        return {
            "llm_calls": len(self.asks),
            "llm_input_tokens": 10 * len(self.asks),
            "llm_output_tokens": 0,
        }


def fake_session_factory(script, created=None):
    def factory(role_name, first_extras=None):
        if created is not None:
            created.append((role_name, first_extras))
        return FakeSession(script)

    return factory


def pool_receipt(depths, *, order=None, mode="slow"):
    """A valid POOL=5 receipt over the toy contract, distinct by depth."""
    configs = [
        cfg(depth=depth, lr=0.001 + 0.0001 * index, dropout=0.1, mode=mode)
        for index, depth in enumerate(depths)
    ]
    return {
        "configs": configs,
        "order": order if order is not None else [0, 1, 2, 3, 4],
        "rationale": "scripted pool",
    }


def test_executes_proposer_rank1_and_accounts_llm(tmp_path) -> None:
    ckpt = write_checkpoint(tmp_path)
    out = tmp_path / "out"
    created = []
    # Rank-1 is pool member 2 (order[0] == 2) every step; lr distinguishes
    # both pool members and steps so nothing duplicates executed history.
    script = [
        {
            "configs": [
                cfg(
                    depth=1 + offset,
                    lr=0.0001 * (1 + step) + 0.00001 * offset,
                    dropout=0.1,
                    mode="slow",
                )
                for offset in range(5)
            ],
            "order": [2, 0, 1, 3, 4],
            "rationale": "scripted pool",
        }
        for step in range(10)
    ]
    scores = [90.0 - 5.0 * index for index in range(10)]

    result = runner.run_cell(
        arm=llm_pool_self_rank.ARM,
        checkpoint_dir=ckpt,
        out_dir=out,
        seed=11,
        eval_fn=fake_eval_from([("ok", score) for score in scores]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory(script, created)},
    )

    assert result["status"] == "ok"
    assert result["llm_calls"] == 10
    assert result["llm_input_tokens"] == 100
    assert result["internal_duplicate_count"] == 0
    assert result["beat_initial_incumbent"] is True
    # Rank-1 (order[0] == 2) executed every step: lr 0.0001*(1+step)+0.00002.
    proposals = [event["proposal"] for event in evaluation_events(out)]
    assert [p["lr"] for p in proposals] == pytest.approx(
        [0.0001 * (1 + step) + 0.00002 for step in range(10)]
    )
    assert all(p["depth"] == 3 for p in proposals)
    # One bout session, created with the §四 first-message blocks.
    assert len(created) == 1
    role_name, first_extras = created[0]
    assert role_name == "bench-pool-proposer"
    assert {
        "search_space",
        "candidate",
        "incumbent",
        "history",
        "protocol",
        "budget",
    } <= set(first_extras)


def test_duplicate_members_filtered_before_rank1(tmp_path) -> None:
    # Rank-1 (order[0]==0) duplicates the incumbent; the arm must execute the
    # next non-duplicate member (rank 2) and count the internal duplicate.
    incumbent = BASE
    dup = dict(incumbent)
    fresh = cfg(depth=5, lr=0.005, dropout=0.2, mode="slow")
    others = [
        cfg(depth=6, lr=0.006, dropout=0.2, mode="slow"),
        cfg(depth=7, lr=0.007, dropout=0.2, mode="slow"),
        cfg(depth=8, lr=0.008, dropout=0.2, mode="slow"),
    ]
    receipt = {
        "configs": [dup, fresh, *others],
        "order": [0, 1, 2, 3, 4],
        "rationale": "dup at rank1",
    }
    ckpt = write_checkpoint(tmp_path)

    result = runner.run_cell(
        arm=llm_pool_self_rank.ARM,
        checkpoint_dir=ckpt,
        out_dir=tmp_path / "out",
        seed=1,
        budget=1,
        eval_fn=fake_eval_from([("ok", 50.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([receipt])},
    )

    assert result["status"] == "ok"
    assert result["best_config"] == fresh
    assert result["internal_duplicate_count"] == 1


def test_invalid_receipt_retries_then_succeeds(tmp_path) -> None:
    bad = {"configs": [cfg()], "order": [0], "rationale": "too few"}
    good = pool_receipt([1, 2, 3, 4, 5])
    session = FakeSession([bad, good])

    result = runner.run_cell(
        arm=llm_pool_self_rank.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        budget=1,
        eval_fn=fake_eval_from([("ok", 50.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": lambda role, first_extras=None: session},
    )

    assert result["status"] == "ok"
    assert result["llm_calls"] == 2  # the retry burned a second call
    # The retry rides on a correction message naming the problem.
    assert "correction" in session.asks[1]["outcome"]
    assert "exactly 5" in session.asks[1]["outcome"]


def test_three_consecutive_failures_is_arm_error(tmp_path) -> None:
    bad = {"configs": [cfg()], "order": [0], "rationale": "too few"}

    result = runner.run_cell(
        arm=llm_pool_self_rank.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        eval_fn=fake_eval_from([]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([bad, bad, bad])},
    )

    assert result["status"] == "arm_error"
    assert "3 consecutive" in result["reason"]
    assert result["llm_calls"] == 3


def test_all_duplicate_pool_regenerates(tmp_path) -> None:
    all_dup = {
        "configs": [cfg(depth=1), cfg(depth=2), cfg(depth=3), cfg(depth=4), cfg(depth=5)],
        "order": [0, 1, 2, 3, 4],
        "rationale": "all dup",
    }
    # Every member duplicates executed history (checkpoint history + incumbent).
    history = [hrow(cfg(depth=depth), 90.0 + depth) for depth in (1, 2, 3, 4)]
    good = pool_receipt([6, 7, 8, 2, 3])
    good["configs"][3] = cfg(depth=6, lr=0.002, dropout=0.2, mode="slow")
    good["configs"][4] = cfg(depth=7, lr=0.002, dropout=0.2, mode="slow")

    result = runner.run_cell(
        arm=llm_pool_self_rank.ARM,
        checkpoint_dir=write_checkpoint(
            tmp_path,
            incumbent_params=cfg(depth=5),
            history=history,
        ),
        out_dir=tmp_path / "out",
        seed=1,
        budget=1,
        eval_fn=fake_eval_from([("ok", 50.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([all_dup, good])},
    )

    assert result["status"] == "ok"
    assert result["internal_duplicate_count"] == 5
    assert result["llm_calls"] == 2


def test_full_pool_and_order_persisted_in_events(tmp_path) -> None:
    """PLAN §6.4: the complete pool, the proposer's original order, and the
    duplicate-filter mask are persisted per evaluation, so every rank
    position maps back to a concrete config post-hoc."""
    dup = dict(BASE)  # duplicates the incumbent
    fresh = [
        cfg(depth=5, lr=0.005, dropout=0.2, mode="slow"),
        cfg(depth=6, lr=0.006, dropout=0.2, mode="slow"),
        cfg(depth=7, lr=0.007, dropout=0.2, mode="slow"),
        cfg(depth=8, lr=0.008, dropout=0.2, mode="slow"),
    ]
    receipt = {
        "configs": [dup, *fresh],
        "order": [2, 0, 1, 3, 4],
        "rationale": "dup inside",
    }

    result = runner.run_cell(
        arm=llm_pool_self_rank.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        budget=1,
        eval_fn=fake_eval_from([("ok", 50.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": fake_session_factory([receipt])},
    )

    assert result["status"] == "ok"
    state = evaluation_events(tmp_path / "out")[0]["arm_state"]
    # Ranked pre-filter pool: raw_configs[k] == configs[pool_order[k]].
    assert state["pool_order"] == [2, 0, 1, 3, 4]
    assert state["pool_configs"] == [
        receipt["configs"][index] for index in [2, 0, 1, 3, 4]
    ]
    # The incumbent duplicate sat at rank 2 (order[1] == 0) and was dropped.
    assert state["pool_duplicate_mask"] == [False, True, False, False, False]
    # The executed config is the first unmasked member of the ranked pool.
    assert result["best_config"] == fresh[1]


def test_failed_ask_preserves_pending_outcome(tmp_path) -> None:
    """A failed invocation never entered the transcript: the pending
    authoritative outcome must survive it and ride the retry ask."""
    good_1 = pool_receipt([1, 2, 3, 4, 5])
    good_2 = pool_receipt([6, 7, 8, 2, 3])  # depth <= 8 (space max), no dups
    session = FakeSession([good_1, RuntimeError("transient"), good_2])

    result = runner.run_cell(
        arm=llm_pool_self_rank.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),
        out_dir=tmp_path / "out",
        seed=1,
        budget=2,
        eval_fn=fake_eval_from([("ok", 90.0), ("ok", 80.0)]),
        preflight_fn=ok_preflight,
        extras={"session_factory": lambda role, first_extras=None: session},
    )

    assert result["status"] == "ok"
    assert result["llm_calls"] == 3  # one burned on the transient failure
    # The failed ask carried the step-1 outcome; the retry re-delivers it.
    assert "outcome" in session.asks[1]
    assert session.asks[2].get("outcome") == session.asks[1]["outcome"]


def test_improvement_verdict_uses_pre_evaluation_incumbent(tmp_path) -> None:
    """Regression: the outcome verdict must compare against the incumbent the
    proposal had to beat. The runner advances ctx.state before the feedback
    returns, so reading the live state misreports every strict improvement
    as "did NOT improve" (50 < 50)."""
    session = FakeSession([pool_receipt([1, 2, 3, 4, 5]), pool_receipt([6, 7, 8, 2, 3])])

    result = runner.run_cell(
        arm=llm_pool_self_rank.ARM,
        checkpoint_dir=write_checkpoint(tmp_path),  # incumbent 100.0
        out_dir=tmp_path / "out",
        seed=1,
        budget=2,
        eval_fn=fake_eval_from([("ok", 90.0), ("ok", 80.0)]),  # both improve
        preflight_fn=ok_preflight,
        extras={"session_factory": lambda role, first_extras=None: session},
    )

    assert result["status"] == "ok"
    # 90.0 < 100.0 -> IMPROVED against the pre-evaluation incumbent.
    assert "IMPROVED" in session.asks[1]["outcome"]
    assert "current incumbent score = 100.0" in session.asks[1]["outcome"]
