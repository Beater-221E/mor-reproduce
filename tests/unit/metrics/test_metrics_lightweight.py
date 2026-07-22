"""Lightweight metrics import must not pull Transformers."""

from __future__ import annotations

import sys


def test_metrics_import_is_lightweight():
    # Ensure not already loaded from other tests in same process is hard;
    # check module dependency graph after import.
    before = {m for m in sys.modules if m.startswith("transformers")}
    from minionerec.evaluation import metrics as m

    after = {mod for mod in sys.modules if mod.startswith("transformers")}
    assert m.hit_rate_at_k(0, 1) == 1.0
    # Importing metrics alone should not newly import transformers
    assert after == before or "transformers" not in after
