"""Ranking metrics (HR / NDCG). Pure Python — no Transformers / CUDA / Dataset."""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_rate_at_k(rank: int | None, k: int) -> float:
    """``rank`` is 0-based; ``None`` means not found."""
    if rank is None:
        return 0.0
    return 1.0 if rank < k else 0.0


# Aliases kept for callers/tests
hr_at_k = hit_rate_at_k


def ndcg_at_k(rank: int | None, k: int) -> float:
    """NDCG contribution for a single example: 1/log2(rank+2) when rank < k."""
    if rank is None or rank >= k:
        return 0.0
    return 1.0 / math.log2(rank + 2)


def compute_ranking_metrics(
    predictions: list[list[str]],
    targets: Sequence[str],
    ks: Sequence[int] = (3, 5, 10),
) -> dict[str, float]:
    """Aggregate HR@K / NDCG@K over a batch (string strip matches official eval)."""
    metrics = {f"HR@{k}": 0.0 for k in ks}
    metrics.update({f"NDCG@{k}": 0.0 for k in ks})
    dup = 0
    n = len(targets)
    for preds, tgt in zip(predictions, targets, strict=False):
        tgt_n = tgt.strip('\n" ')
        cleaned = [p.strip('\n" ') for p in preds]
        if len(cleaned) != len(set(cleaned)):
            dup += 1
        rank = None
        for i, p in enumerate(cleaned):
            if p == tgt_n:
                rank = i
                break
        for k in ks:
            metrics[f"HR@{k}"] += hit_rate_at_k(rank, k)
            metrics[f"NDCG@{k}"] += ndcg_at_k(rank, k)
    for k in ks:
        metrics[f"HR@{k}"] /= max(1, n)
        metrics[f"NDCG@{k}"] /= max(1, n)
    metrics["duplicate_rate"] = dup / max(1, n)
    metrics["num_examples"] = n
    return metrics


# Backward-compatible name used by older tests/evaluator
compute_metrics = compute_ranking_metrics
