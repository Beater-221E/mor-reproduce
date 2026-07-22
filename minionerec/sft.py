"""Deprecated — use ``minionerec.training.sft_legacy_loop`` or ``training.sft``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.sft (legacy) is deprecated; prefer minionerec.training.sft", DeprecationWarning, stacklevel=2)
from minionerec.training.sft_legacy_loop import *  # noqa: F401,F403
