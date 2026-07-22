"""Deprecated — use ``minionerec.sid.rqvae``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.rqvae is deprecated; use minionerec.sid.rqvae", DeprecationWarning, stacklevel=2)
from minionerec.sid.rqvae import *  # noqa: F401,F403
