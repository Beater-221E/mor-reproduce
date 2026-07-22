"""Deprecated — use ``minionerec.training.rl_legacy_loop`` or ``training.rl``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.rl (legacy) is deprecated; prefer minionerec.training.rl", DeprecationWarning, stacklevel=2)
from minionerec.training.rl_legacy_loop import *  # noqa: F401,F403
