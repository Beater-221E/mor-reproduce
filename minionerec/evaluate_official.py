"""Deprecated shim — use ``minionerec.evaluation.evaluator``."""
from __future__ import annotations
import warnings

warnings.warn(
    "minionerec.evaluate_official is deprecated; use minionerec.evaluation.evaluator",
    DeprecationWarning,
    stacklevel=2,
)
from minionerec.evaluation.evaluator import *  # noqa: F401,F403
from minionerec.evaluation.metrics import compute_metrics, hr_at_k, ndcg_at_k  # noqa: F401
