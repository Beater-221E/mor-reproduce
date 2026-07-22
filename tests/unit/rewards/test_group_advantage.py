import math

from minionerec.rewards_official import group_advantages


def test_group_advantage_zero_mean_and_std0():
    # two groups of 2
    rewards = [1.0, 0.0, 5.0, 5.0]
    adv = group_advantages(rewards, num_generations=2, eps=1e-4)
    assert abs(sum(adv[:2])) < 1e-5
    # second group std~0 -> advantages ~0
    assert abs(adv[2]) < 1e-3 and abs(adv[3]) < 1e-3
    assert all(math.isfinite(a) for a in adv)
