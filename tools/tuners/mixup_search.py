#!/usr/bin/env python3
"""Production 24-slot INITIAL adapter for ``mixup_pool_hebo``."""

from __future__ import annotations

import hebo_search
from arms.mixup_pool_hebo import ARM


hebo_search.METHOD = "mixup"
hebo_search.ARM = ARM


if __name__ == "__main__":
    raise SystemExit(hebo_search.main())
