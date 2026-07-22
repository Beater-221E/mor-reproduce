import torch
from minionerec.grpo_common import schulman_kl, official_source_policy_loss


def test_kl_nonnegative_when_same():
    x = torch.zeros(2, 3)
    assert torch.allclose(schulman_kl(x, x), torch.zeros_like(x))


def test_official_loss_finite():
    B, T = 4, 6
    logps = torch.randn(B, T, requires_grad=True)
    ref = torch.randn(B, T)
    adv = torch.randn(B)
    mask = torch.ones(B, T)
    loss, m = official_source_policy_loss(logps, ref, adv, mask, beta=0.001)
    assert torch.isfinite(loss)
    assert m["ratio_mean"] == "not_applicable"
    loss.backward()
