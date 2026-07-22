"""Deprecated shim — use ``minionerec.training.objectives``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.grpo_common is deprecated; use minionerec.training.objectives", DeprecationWarning, stacklevel=2)
from minionerec.training.objectives import *  # noqa: F401,F403
