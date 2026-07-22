"""
Official rule + ranking rewards.

Source: MiniOneRec-official @ 0c64b955 / rl.py
  - rule_reward
  - ndcg_rule_reward (used when reward_type=ranking together with rule)

Rank is generation-order within the group (0-based index into ndcg_rewards),
NOT beam score or model log-probability.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RewardResult:
    total: torch.Tensor
    rule: torch.Tensor
    ranking: torch.Tensor


def make_ndcg_rank_table(num_generations: int) -> list[float]:
    """
    Official:
      ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]
      ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]
    """
    raw = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
    s = sum(raw)
    return [-elm / s for elm in raw]


def strip_completion(text: str) -> str:
    return text.strip('\n" ')


def rule_reward(prompts: Sequence[str], completions: Sequence[str], targets: Sequence[str]) -> list[float]:
    """Exact match after strip('\\n\" '). Official rl.py:rule_reward."""
    rewards = []
    for completion, target in zip(completions, targets, strict=False):
        if strip_completion(completion) == strip_completion(target):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def ndcg_rule_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    targets: Sequence[str],
    num_generations: int,
) -> list[float]:
    """
    Official ranking reward (rl.py:ndcg_rule_reward):
      - correct completion -> 0.0
      - incorrect -> normalized -1/log2(rank+2) by generation order within group
      - if group has no correct answer -> all zeros
    Rank is 0-based position i % num_generations in the completion list order.
    """
    ndcg_rewards = make_ndcg_rank_table(num_generations)
    rewards: list[float] = []
    flag = False
    lis: list[float] = []
    for i, completion in enumerate(completions):
        if strip_completion(completion) == strip_completion(targets[i]):
            flag = True
            lis.append(0.0)
        else:
            lis.append(ndcg_rewards[i % num_generations])
        if (i + 1) % num_generations == 0:
            if flag:
                rewards.extend(lis)
            else:
                rewards.extend([0.0] * num_generations)
            flag = False
            lis = []
    return rewards


def hybrid_ranking_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    targets: Sequence[str],
    num_generations: int,
) -> tuple[list[float], list[float], list[float]]:
    """R = R_rule + R_rank. Returns (total, rule, rank)."""
    r_rule = rule_reward(prompts, completions, targets)
    r_rank = ndcg_rule_reward(prompts, completions, targets, num_generations)
    total = [a + b for a, b in zip(r_rule, r_rank, strict=True)]
    return total, r_rule, r_rank


def compute_ranking_reward(
    completions: Sequence[str],
    targets: Sequence[str],
    group_size: int,
    prompts: Sequence[str] | None = None,
) -> RewardResult:
    """Single public reward API used by the unified RL trainer."""
    prompts = prompts if prompts is not None else [""] * len(completions)
    total, rule, ranking = hybrid_ranking_reward(prompts, completions, targets, group_size)
    return RewardResult(
        total=torch.tensor(total, dtype=torch.float32),
        rule=torch.tensor(rule, dtype=torch.float32),
        ranking=torch.tensor(ranking, dtype=torch.float32),
    )


def group_advantages(rewards: Sequence[float], num_generations: int, eps: float = 1e-4) -> list[float]:
    """
    Official ReReTrainer:
      advantages = (rewards - mean_g) / (std_g + 1e-4)
    No adv_clip in official commit.
    """
    r = torch.tensor(list(rewards), dtype=torch.float32)
    g = r.view(-1, num_generations)
    mean = g.mean(dim=1).repeat_interleave(num_generations)
    std = g.std(dim=1).repeat_interleave(num_generations)
    adv = (r - mean) / (std + eps)
    return adv.tolist()


def build_reward_funcs(reward_type: str, prompt2history: dict, history2target: dict, num_generations: int):
    """Return callable(s) matching official rl.py reward wiring."""

    def _targets_from_prompts(prompts):
        history = [prompt2history[p] for p in prompts]
        return [history2target[h] for h in history]

    def _rule(prompts, completions, **kwargs):
        return rule_reward(prompts, completions, _targets_from_prompts(prompts))

    def _ndcg(prompts, completions, **kwargs):
        return ndcg_rule_reward(prompts, completions, _targets_from_prompts(prompts), num_generations)

    if reward_type == "rule":
        return _rule
    if reward_type == "ranking":
        return [_rule, _ndcg]
    if reward_type == "ranking_only":
        return _ndcg
    raise ValueError(f"Unsupported reward_type={reward_type} (CF/semantic disabled in this alignment)")
