"""Deprecated shim — use ``minionerec.training.sft``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.sft_official is deprecated; use minionerec.training.sft", DeprecationWarning, stacklevel=2)
from minionerec.training.sft import *  # noqa: F401,F403
from minionerec.training.sft import main, run_sft, train_sft  # noqa: F401
