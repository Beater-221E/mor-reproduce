"""Deprecated shim — use ``minionerec.training.rl``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.rl_official is deprecated; use minionerec.training.rl", DeprecationWarning, stacklevel=2)
from minionerec.training.rl import *  # noqa: F401,F403
from minionerec.training.rl import main, run_rl, train_rl  # noqa: F401
