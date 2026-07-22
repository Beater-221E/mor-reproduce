from minionerec.rewards_official import rule_reward


def test_rule_exact_and_whitespace():
    gold = ["<a_1><b_2><c_3>\n"]
    assert rule_reward(["p"], ["<a_1><b_2><c_3>\n"], gold) == [1.0]
    assert rule_reward(["p"], ['  "<a_1><b_2><c_3>"  \n'], gold) == [1.0]
    assert rule_reward(["p"], ["<a_1><b_2><c_3>"], gold) == [1.0]
    assert rule_reward(["p"], ["<a_9><b_9><c_9>\n"], gold) == [0.0]
    assert rule_reward(["p"], ["some title"], gold) == [0.0]
    assert rule_reward(["p"], ["<a_1><b_2>\n"], gold) == [0.0]
    assert rule_reward(["p"], ["<a_1><b_2><c_3><d_0>\n"], gold) == [0.0]
