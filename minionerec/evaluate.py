"""Deprecated entry — use ``mor evaluate --config ...``."""
from __future__ import annotations
import warnings

warnings.warn("Deprecated: use `mor evaluate --config ...`", DeprecationWarning, stacklevel=2)
from minionerec.evaluation.evaluator import main

if __name__ == "__main__":
    main()
