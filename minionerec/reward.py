"""Deprecated — use ``minionerec.rewards.legacy`` (legacy RL only)."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.reward is deprecated; use minionerec.rewards.ranking or rewards.legacy", DeprecationWarning, stacklevel=2)
from minionerec.rewards.legacy import *  # noqa: F401,F403
