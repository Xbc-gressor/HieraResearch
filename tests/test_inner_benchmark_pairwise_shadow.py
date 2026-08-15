"""Deterministic protocol checks for the GPU-free pairwise shadow runner."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))

import pairwise_shadow as shadow  # noqa: E402
from driver.session import FakeSessionRunner  # noqa: E402
from ib_support import cfg, write_checkpoint  # noqa: E402


def _verdict(duel: shadow.Duel, winner: int) -> shadow.DuelVerdict:
    return shadow.DuelVerdict(duel=duel, winner_index=winner, payload={})


def test_pool5_orientation_is_complete_and_position_balanced() -> None:
    duels = shadow.balanced_duels(5, random.Random(17))
    assert len(duels) == 10
    assert len({frozenset((duel.a_index, duel.b_index)) for duel in duels}) == 10
    assert Counter(duel.a_index for duel in duels) == Counter({i: 2 for i in range(5)})
    assert Counter(duel.b_index for duel in duels) == Counter({i: 2 for i in range(5)})
    assert all(0 in (duel.a_index, duel.b_index) for duel in duels[:4])


def test_rank1_sweep_exactly_stops_after_four_calls() -> None:
    duels = shadow.balanced_duels(5, random.Random(23))
    batches = []

    def run_batch(batch):
        batches.append(tuple(batch))
        return [_verdict(duel, 0) for duel in batch]

    result = shadow.run_exact_tournament(5, duels, run_batch, random.Random(1))
    assert [len(batch) for batch in batches] == [4]
    assert result["winner_index"] == 0
    assert result["early_stopped"] is True
    assert result["full_round_robin_completed"] is False
    assert result["observed_vote_counts"] == [4, 0, 0, 0, 0]


def test_rank1_loss_completes_round_robin_and_uses_seeded_tie() -> None:
    duels = shadow.balanced_duels(5, random.Random(29))
    batches = []

    def run_batch(batch):
        batches.append(tuple(batch))
        # In the regular orientation every candidate appears as A twice, so
        # always choosing A produces a five-way Copeland tie after all edges.
        return [_verdict(duel, duel.a_index) for duel in batch]

    result = shadow.run_exact_tournament(5, duels, run_batch, random.Random(7))
    assert [len(batch) for batch in batches] == [4, 6]
    assert result["early_stopped"] is False
    assert result["full_round_robin_completed"] is True
    assert result["full_vote_counts"] == [2, 2, 2, 2, 2]
    assert result["top_indexes"] == [0, 1, 2, 3, 4]
    assert result["tie_break"] == "seeded_uniform"
    assert result["winner_index"] == random.Random(7).choice([0, 1, 2, 3, 4])


def test_obtain_pool_reuses_production_pool_validation(tmp_path, monkeypatch) -> None:
    checkpoint_dir = write_checkpoint(tmp_path, name="frozen")
    item = shadow._load_frozen(checkpoint_dir)
    configs = [cfg(depth=depth) for depth in (1, 2, 3, 5, 6)]
    fake = FakeSessionRunner(
        [
            {
                "receipt": {
                    "configs": configs,
                    "order": [2, 0, 4, 1, 3],
                    "rationale": "ranked fixture",
                }
            }
        ]
    )
    monkeypatch.setattr(shadow, "SDKSessionRunner", lambda **kwargs: fake)

    pool, record = shadow._obtain_pool(
        item,
        replicate_dir=tmp_path / "replicate",
        model="test-model",
        budget=10,
        cli_path=None,
    )

    assert [config["depth"] for config in pool] == [3, 1, 6, 2, 5]
    assert record["self_rank_config"] == pool[0]
    assert record["executed_pool_proposer_ranks"] == [1, 2, 3, 4, 5]
    assert len(fake.calls) == 1


def test_discover_checkpoint_dirs_recurses_and_sorts(tmp_path) -> None:
    second = write_checkpoint(tmp_path / "nested", name="b")
    first = write_checkpoint(tmp_path, name="a")

    assert shadow.discover_checkpoint_dirs(None, [str(tmp_path)]) == sorted(
        [first.resolve(), second.resolve()], key=str
    )
