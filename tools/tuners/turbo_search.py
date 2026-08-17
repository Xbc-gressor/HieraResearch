#!/usr/bin/env python3
"""Production 10-slot adapter for the stateful hot-start TuRBO arm."""

from __future__ import annotations

import hebo_search
from arms.turbo import ARM


hebo_search.METHOD = "turbo"
hebo_search.ARM = ARM


if __name__ == "__main__":
    raise SystemExit(hebo_search.main())
