#!/usr/bin/env python3
"""Production LLM-pool self-rank bout.

This is the production adapter for the inner-benchmark
``llm_pool_self_rank`` arm.  The shared pool-search runner lives in
``hebo_search``; only the selector and persisted method identity differ.
"""

from __future__ import annotations

import hebo_search
from arms.llm_pool_self_rank import ARM


hebo_search.METHOD = "selfrank"
hebo_search.ARM = ARM


if __name__ == "__main__":
    raise SystemExit(hebo_search.main())
