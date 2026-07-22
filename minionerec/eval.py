"""Deprecated — use ``minionerec.evaluation.legacy_evaluator`` or ``evaluation.evaluator``."""
from __future__ import annotations
import warnings

warnings.warn("minionerec.eval is deprecated; use minionerec.evaluation.evaluator", DeprecationWarning, stacklevel=2)
from minionerec.evaluation.legacy_evaluator import *  # noqa: F401,F403
