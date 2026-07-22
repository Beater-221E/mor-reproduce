"""Deprecated shim — use ``minionerec.runtime.distributed``."""
from __future__ import annotations
import warnings

warnings.warn(
    "minionerec.distributed_utils is deprecated; use minionerec.runtime.distributed",
    DeprecationWarning,
    stacklevel=2,
)
from minionerec.runtime.distributed import *  # noqa: F401,F403
