"""Deprecated shim — use ``minionerec.rewards.ranking``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.rewards_official is deprecated; use minionerec.rewards.ranking", DeprecationWarning, stacklevel=2)
from minionerec.rewards.ranking import *  # noqa: F401,F403
