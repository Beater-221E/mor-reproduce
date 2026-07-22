from minionerec.util import format_sid, parse_sid, all_sid_tokens
from minionerec.reward import rule, hybrid
from minionerec.decode import SIDTrie


def test_sid():
    codes = [1, 2, 3]
    assert parse_sid(format_sid(codes)) == codes
    assert len(all_sid_tokens()) == 768


def test_reward():
    gold = format_sid([1, 2, 3])
    preds = [gold, format_sid([9, 9, 9])]
    assert rule(preds[0], gold) == 1.0
    assert hybrid(preds, gold)[0] > hybrid(preds, gold)[1]


def test_trie():
    t = SIDTrie()
    t.insert([10, 20, 30])
    assert t.allowed([]) == {10}
    assert t.allowed([10]) == {20}


if __name__ == "__main__":
    test_sid()
    test_reward()
    test_trie()
    print("ok")
