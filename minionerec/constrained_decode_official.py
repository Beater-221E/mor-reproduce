"""Deprecated shim — use ``minionerec.generation.constraints``."""
from __future__ import annotations
import warnings

warnings.warn(
    "minionerec.constrained_decode_official is deprecated; use minionerec.generation.constraints",
    DeprecationWarning,
    stacklevel=2,
)
from minionerec.generation.constraints import *  # noqa: F401,F403
