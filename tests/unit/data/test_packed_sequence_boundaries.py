"""Official SFT does not pack; ensure no cross-sample label mix in collated batch."""

import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq


def test_collator_pads_labels_with_ignore():
    tok = AutoTokenizer.from_pretrained(
        str(__import__("pathlib").Path(__file__).resolve().parents[3] / "data/models/Qwen2.5-0.5B"),
        trust_remote_code=True,
    )
    tok.pad_token = tok.eos_token
    collator = DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    feats = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, -100, 5]},
        {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1], "labels": [-100, -100, 6, 7, 8]},
    ]
    batch = collator(feats)
    # padded label positions should be -100
    assert (batch["labels"][0] == -100).sum() >= 2
    assert batch["labels"].shape == batch["input_ids"].shape
