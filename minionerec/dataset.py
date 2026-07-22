"""Deprecated — use ``minionerec.data.legacy_datasets`` or ``data.datasets``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.dataset is deprecated; use minionerec.data.datasets", DeprecationWarning, stacklevel=2)
from minionerec.data.legacy_datasets import *  # noqa: F401,F403
