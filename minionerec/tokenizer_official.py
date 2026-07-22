"""Deprecated shim — use ``minionerec.sid.tokenizer``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.tokenizer_official is deprecated; use minionerec.sid.tokenizer", DeprecationWarning, stacklevel=2)
from minionerec.sid.tokenizer import *  # noqa: F401,F403
