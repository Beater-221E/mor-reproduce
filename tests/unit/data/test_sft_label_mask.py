"""Unit tests for SFT label masking."""

from minionerec.official_data import Tokenizer


class DummyTok:
    def __init__(self):
        self.bos_token_id = None
        self.eos_token_id = 1

    def encode(self, s):
        return [ord(c) % 50 + 10 for c in s]

    def decode(self, ids):
        return "".join(chr((i - 10) % 50 + 97) for i in ids)


def test_response_only_labels_pattern():
    prompt = [10, 11, 12, 13]
    response = [20, 21, 22, 1]
    labels = [-100] * len(prompt) + response
    assert all(y == -100 for y in labels[: len(prompt)])
    assert all(y != -100 for y in labels[len(prompt) :])
    assert sum(1 for y in labels if y != -100) == len(response)


def test_tokenizer_wrapper_strips_eos():
    tok = Tokenizer(DummyTok())
    ids = tok.encode("ab", bos=False, eos=True)
    assert ids[-1] == 1
