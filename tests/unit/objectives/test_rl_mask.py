import torch


def test_completion_mask_cuts_after_eos():
    eos = 2
    completion_ids = torch.tensor([[5, 6, 2, 9, 9], [5, 2, 9, 9, 9]])
    is_eos = completion_ids == eos
    eos_idx = torch.full((2,), 5, dtype=torch.long)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    seq = torch.arange(5).expand(2, -1)
    mask = (seq <= eos_idx.unsqueeze(1)).int()
    assert mask[0].tolist() == [1, 1, 1, 0, 0]
    assert mask[1].tolist() == [1, 1, 0, 0, 0]
