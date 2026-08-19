"""POOL=3 experimental variant of the LLM-pool + HEBO MACE arm.

This is deliberately a separate benchmark arm so the frozen POOL=5 arm and
the production inner-tuner policy remain unchanged.
"""

from __future__ import annotations

from arms.pool_hebo_mace import PoolHeboMace


class Pool3HeboMace(PoolHeboMace):
    name = "pool3_hebo_mace"
    pool_size = 3


ARM = Pool3HeboMace()
