import torch


def test_old_policy_snapshot_detached():
    x = torch.randn(2, 3, requires_grad=True)
    old = x.detach().clone()
    y = (x * 2).sum()
    y.backward()
    assert old.grad is None
    assert not old.requires_grad
