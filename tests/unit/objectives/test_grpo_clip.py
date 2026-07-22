import torch
from minionerec.grpo_common import paper_aligned_policy_loss


def test_clip_activates_for_large_ratio():
    B, T = 2, 4
    old = torch.zeros(B, T)
    cur = torch.ones(B, T)  # ratio e
    ref = torch.zeros(B, T)
    adv = torch.ones(B)
    mask = torch.ones(B, T)
    _, m = paper_aligned_policy_loss(cur, old, ref, adv, mask, beta=0.0, clip_eps=0.2)
    assert m["clip_fraction"] == 1.0
