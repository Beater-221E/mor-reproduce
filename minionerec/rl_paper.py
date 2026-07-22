"""Deprecated shim — paper RL uses unified trainer with algorithm.variant=paper."""
from __future__ import annotations
import warnings

warnings.warn(
    "minionerec.rl_paper is deprecated; use `mor train-rl` with algorithm.variant=paper",
    DeprecationWarning,
    stacklevel=2,
)
from minionerec.training.rl import main, run_rl, train_rl  # noqa: F401
