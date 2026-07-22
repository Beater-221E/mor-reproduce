"""Deprecated entry — use ``mor train-sft --config ...``."""
from __future__ import annotations
import warnings

warnings.warn("Deprecated: use `mor train-sft --config ...`", DeprecationWarning, stacklevel=2)
from minionerec.training.sft import main

if __name__ == "__main__":
    main()
