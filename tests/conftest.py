"""Suite-wide test isolation for run-time override variables.

``TARGET_TIER`` is the per-run E2E A/B switch and is routinely exported in
shells that also run the suite; tests must stay closed against it (the
tier-specific tests set it explicitly via ``mock.patch.dict``).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_target_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TARGET_TIER", raising=False)
