"""Deprecated — use ``minionerec.sid.embeddings``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.emb is deprecated; use minionerec.sid.embeddings", DeprecationWarning, stacklevel=2)
from minionerec.sid.embeddings import *  # noqa: F401,F403
