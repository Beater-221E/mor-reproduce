"""Deprecated — use ``minionerec.data.amazon23``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.prep is deprecated; use minionerec.data.amazon23", DeprecationWarning, stacklevel=2)
from minionerec.data.amazon23 import *  # noqa: F401,F403
