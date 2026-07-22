"""
RL policy objectives.

official (ReReTrainer.compute_loss @ 0c64b955):
  ratio_t = exp(logπ_θ - logπ_θ.detach())  # == 1 always
  KL_t = exp(logπ_ref - logπ_θ) - (logπ_ref - logπ_θ) - 1
  L = mean_i[ mean_t( -ratio_t * A_i + β * KL_t ) ]

paper:
  w_t = exp(logπ_θ - logπ_old) with old snapshot from rollout
  L_policy = -mean_i[ mean_t min(w A, clip(w,1-ε,1+ε) A) ]
  L = L_policy + β L_KL

legacy: sequence-mean logπ × advantage (mor-reproduce historical path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    logprobs = F.log_softmax(logits, dim=-1)
    return logprobs.gather(dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def schulman_kl(ref_logps: torch.Tensor, logps: torch.Tensor) -> torch.Tensor:
    """Official Schulman approx KL (minionerec_trainer.py)."""
    return torch.exp(ref_logps - logps) - (ref_logps - logps) - 1


def official_source_policy_loss(
    per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    beta: float,
    dapo: bool = False,
    gspo: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Source: minionerec_trainer.py compute_loss (official commit).
    Note: exp(logps - logps.detach()) is identically 1 — no true old-policy ratio.
    """
    token_kl = schulman_kl(ref_per_token_logps, per_token_logps)
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * token_kl)

    if dapo:
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp_min(1)
    elif gspo:
        per_token_ratio = per_token_logps - per_token_logps.detach()
        s_score = torch.exp((per_token_ratio * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1))
        sequence_kl = (token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)
        loss = -(s_score * advantages - beta * sequence_kl).mean()
    else:
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)).mean()

    mean_kl = ((token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)).mean()
    metrics = {
        "kl_mean": float(mean_kl.detach().item()),
        "ratio_mean": "not_applicable",
        "ratio_min": "not_applicable",
        "ratio_max": "not_applicable",
        "clip_fraction": "not_applicable",
        "policy_loss": float(loss.detach().item()),
    }
    return loss, metrics


def paper_aligned_policy_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    beta: float,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """True old-policy GRPO/PPO-style clipped surrogate + KL."""
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)
    adv = advantages.unsqueeze(1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token_policy = -torch.min(unclipped, clipped)
    kl = schulman_kl(ref_per_token_logps, per_token_logps)
    per_token = per_token_policy + beta * kl
    loss = ((per_token * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)).mean()

    with torch.no_grad():
        mask = completion_mask.bool()
        r = ratio[mask]
        clip_frac = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps))[mask].float().mean()
        mean_kl = ((kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)).mean()
        metrics = {
            "kl_mean": float(mean_kl.item()),
            "ratio_mean": float(r.mean().item()) if r.numel() else 0.0,
            "ratio_min": float(r.min().item()) if r.numel() else 0.0,
            "ratio_max": float(r.max().item()) if r.numel() else 0.0,
            "clip_fraction": float(clip_frac.item()) if r.numel() else 0.0,
            "policy_loss": float(loss.detach().item()),
            "positive_advantage_frac": float((advantages > 0).float().mean().item()),
            "negative_advantage_frac": float((advantages < 0).float().mean().item()),
        }
    return loss, metrics


@dataclass
class RolloutBatch:
    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    old_log_probs: torch.Tensor | None
    reference_log_probs: torch.Tensor | None


class RLObjective(Protocol):
    def compute(
        self,
        policy_log_probs: torch.Tensor,
        batch: RolloutBatch,
    ) -> tuple[torch.Tensor, dict]: ...


class OfficialObjective:
    """Official MiniOneRec ReReTrainer loss (fake ratio ≡ 1)."""

    def __init__(self, beta: float = 1e-3, dapo: bool = False, gspo: bool = False):
        self.beta = beta
        self.dapo = dapo
        self.gspo = gspo

    def compute(self, policy_log_probs: torch.Tensor, batch: RolloutBatch) -> tuple[torch.Tensor, dict]:
        assert batch.reference_log_probs is not None
        return official_source_policy_loss(
            policy_log_probs,
            batch.reference_log_probs,
            batch.advantages,
            batch.completion_mask,
            beta=self.beta,
            dapo=self.dapo,
            gspo=self.gspo,
        )


class PaperGrpoObjective:
    """True old-policy clipped GRPO + Schulman KL."""

    def __init__(self, beta: float = 1e-3, clip_eps: float = 0.2):
        self.beta = beta
        self.clip_eps = clip_eps

    def compute(self, policy_log_probs: torch.Tensor, batch: RolloutBatch) -> tuple[torch.Tensor, dict]:
        assert batch.old_log_probs is not None and batch.reference_log_probs is not None
        return paper_aligned_policy_loss(
            policy_log_probs,
            batch.old_log_probs,
            batch.reference_log_probs,
            batch.advantages,
            batch.completion_mask,
            beta=self.beta,
            clip_eps=self.clip_eps,
        )


class LegacyObjective:
    """Sequence-level mean logπ × advantage (legacy mor-reproduce RL)."""

    def __init__(self, beta: float = 0.0):
        self.beta = beta

    def compute(self, policy_log_probs: torch.Tensor, batch: RolloutBatch) -> tuple[torch.Tensor, dict]:
        mask = batch.completion_mask
        seq_logp = (policy_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        loss = -(seq_logp * batch.advantages).mean()
        metrics = {
            "kl_mean": 0.0,
            "ratio_mean": "not_applicable",
            "ratio_min": "not_applicable",
            "ratio_max": "not_applicable",
            "clip_fraction": "not_applicable",
            "policy_loss": float(loss.detach().item()),
        }
        return loss, metrics


def build_objective(
    variant: str,
    *,
    beta: float = 1e-3,
    clip_eps: float = 0.2,
    dapo: bool = False,
    gspo: bool = False,
):
    from minionerec.config import _LEGACY_VARIANT, RLVariant

    v = _LEGACY_VARIANT.get(variant, variant)
    v = RLVariant(str(v))
    if v is RLVariant.OFFICIAL:
        return OfficialObjective(beta=beta, dapo=dapo, gspo=gspo)
    if v is RLVariant.PAPER:
        return PaperGrpoObjective(beta=beta, clip_eps=clip_eps)
    if v is RLVariant.LEGACY:
        return LegacyObjective(beta=beta)
    raise ValueError(f"Unknown RL variant: {variant}")
