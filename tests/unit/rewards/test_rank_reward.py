from minionerec.rewards_official import make_ndcg_rank_table, ndcg_rule_reward


def test_rank_table_matches_official_sign():
    # Official: values remain negative after normalization; sum == -1
    t = make_ndcg_rank_table(4)
    assert len(t) == 4
    assert abs(sum(t) + 1.0) < 1e-6
    assert all(x < 0 for x in t)


def test_ndcg_reward_with_and_without_hit():
    G = 4
    target = "<a_1><b_2><c_3>\n"
    comps = [target, "x", "y", "z"]
    targets = [target] * 4
    r = ndcg_rule_reward(["p"] * 4, comps, targets, G)
    assert r[0] == 0.0
    assert r[1] < 0 and r[2] < 0 and r[3] < 0
    comps2 = ["a", "b", "c", "d"]
    r2 = ndcg_rule_reward(["p"] * 4, comps2, targets, G)
    assert r2 == [0.0, 0.0, 0.0, 0.0]
