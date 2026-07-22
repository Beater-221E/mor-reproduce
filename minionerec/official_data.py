"""Deprecated shim — use ``minionerec.data.datasets``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.official_data is deprecated; use minionerec.data.datasets", DeprecationWarning, stacklevel=2)
from minionerec.data.datasets import *  # noqa: F401,F403
