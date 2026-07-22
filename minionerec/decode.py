"""Deprecated — use ``minionerec.generation.legacy_trie`` or ``generation.constraints.SidConstraint``."""
from __future__ import annotations
import warnings

warnings.warn(
    "minionerec.decode is deprecated; use minionerec.generation.constraints.SidConstraint",
    DeprecationWarning,
    stacklevel=2,
)
from minionerec.generation.legacy_trie import *  # noqa: F401,F403
