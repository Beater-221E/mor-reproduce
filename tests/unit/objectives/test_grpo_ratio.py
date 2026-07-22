import torch
from minionerec.grpo_common import paper_aligned_policy_loss, selective_log_softmax


def test_selective_log_softmax_shape():
    logits = torch.randn(2, 3, 10)
    idx = torch.randint(0, 10, (2, 3))
    lp = selective_log_softmax(logits, idx)
    assert lp.shape == (2, 3)


def test_grpo_ratio_and_clip():
    B, T = 4, 5
    old = torch.zeros(B, T)
    cur = torch.zeros(B, T)
    cur[:, :] = 0.5  # ratio = exp(0.5) ~ 1.65
    ref = torch.zeros(B, T)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.ones(B, T)
    loss, metrics = paper_aligned_policy_loss(cur, old, ref, adv, mask, beta=0.0, clip_eps=0.2)
    assert torch.isfinite(loss)
    assert metrics["clip_fraction"] > 0
    assert metrics["ratio_mean"] > 1.0
