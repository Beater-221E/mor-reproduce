"""Hybrid rewards for GRPO (paper §3.4.2)."""

from __future__ import annotations

import math
from typing import Sequence

from minionerec.util import parse_sid


def rule(pred: str, gold: str) -> float:
    pred_c = parse_sid(pred)
    gold_c = parse_sid(gold)
    if pred_c is not None and gold_c is not None:
        return 1.0 if pred_c == gold_c else 0.0
    # fallback string match on SID substring
    return 1.0 if gold.strip() in pred else 0.0


def rank(preds: Sequence[str], gold: str) -> list[float]:
    """
    Rank-aware reward over a group of candidates.
    R_rank(e_k) = 0 if correct else normalized -1/log(rho+1).
    """
    raw = []
    for rank, pred in enumerate(preds, start=1):
        if rule(pred, gold) >= 1.0:
            raw.append(0.0)
        else:
            raw.append(-1.0 / math.log(rank + 1.0))
    denom = sum(raw)
    if abs(denom) < 1e-8:
        return [0.0 for _ in raw]
    return [-v / denom for v in raw]  # paper normalizes tilde / sum(tilde); tilde negative


def hybrid(preds: Sequence[str], gold: str) -> list[float]:
    rr = [rule(p, gold) for p in preds]
    rk = rank(preds, gold)
    return [a + b for a, b in zip(rr, rk)]


def make_fn(reward_type: str = "ranking"):
    def _fn(prompts, completions, answers, **kwargs):
        # TRL may pass completions as list[str] or list[list[dict]]
        flat = []
        for c in completions:
            if isinstance(c, str):
                flat.append(c)
            elif isinstance(c, list) and c and isinstance(c[0], dict):
                flat.append(c[0].get("content", ""))
            else:
                flat.append(str(c))

        # Group by prompt if needed; assume already grouped sequentially with G completions each
        rewards = []
        # answers length == num prompts; completions length == num prompts * G
        g = max(1, len(flat) // max(1, len(answers)))
        for i, gold in enumerate(answers):
            group = flat[i * g : (i + 1) * g]
            if reward_type == "rule":
                rewards.extend([rule(p, gold) for p in group])
            else:
                rewards.extend(hybrid(group, gold))
        return rewards

    return _fn
