"""Deprecated entry — use ``mor train-rl --config ...``."""
from __future__ import annotations
import warnings

warnings.warn("Deprecated: use `mor train-rl --config ...`", DeprecationWarning, stacklevel=2)
from minionerec.training.rl import main

if __name__ == "__main__":
    main()
