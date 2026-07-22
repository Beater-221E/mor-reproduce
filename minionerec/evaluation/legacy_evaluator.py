"""Offline evaluation: constrained beam search → HR@K / NDCG@K."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from minionerec.sid.codec import parse_sid
from minionerec.generation.legacy_trie import SIDConstraint, make_trie, load_sids


def ndcg_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def hit_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


@torch.no_grad()
def evaluate(
    model_path: Path,
    test_file: Path,
    sid_map_path: Path,
    output_path: Path,
    num_beams: int = 50,
    max_new_tokens: int = 16,
    batch_size: int = 4,
    device: str = "cuda:0",
    ks=(3, 5, 10),
) -> dict:
    sid_map = load_sids(sid_map_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    trie = make_trie(tokenizer, sid_map)
    processor = SIDConstraint(trie, tokenizer)

    rows = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # generative retrieval only
    rows = [r for r in rows if r.get("task", "generative_retrieval") == "generative_retrieval"]

    ranks = []
    invalid = 0
    predictions = []

    for i in tqdm(range(0, len(rows), batch_size), desc="eval"):
        batch = rows[i : i + batch_size]
        prompts = []
        golds = []
        for r in batch:
            msgs = r["messages"][:-1]
            use_chat = hasattr(tokenizer, "apply_chat_template") and getattr(
                tokenizer, "chat_template", None
            )
            if use_chat:
                prompts.append(
                    tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                )
            else:
                prompts.append(f"User: {msgs[0]['content']}\nAssistant:")
            golds.append(r.get("answer", r["messages"][-1]["content"]))

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        processor.set_prompt_lengths([int(x) for x in prompt_lens])

        outs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            logits_processor=[processor],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # outs: (batch * beams, seq)
        for b, gold in enumerate(golds):
            seqs = outs[b * num_beams : (b + 1) * num_beams]
            plen = int(prompt_lens[b])
            cand_sids = []
            for seq in seqs:
                gen = tokenizer.decode(seq[plen:], skip_special_tokens=False)
                codes = parse_sid(gen)
                if codes is None:
                    invalid += 1
                    cand_sids.append(None)
                else:
                    cand_sids.append(tuple(codes))
            gold_codes = parse_sid(gold)
            gold_t = tuple(gold_codes) if gold_codes else None
            rank = None
            for rnk, c in enumerate(cand_sids, start=1):
                if c is not None and gold_t is not None and c == gold_t:
                    rank = rnk
                    break
            ranks.append(rank)
            predictions.append(
                {
                    "gold": gold,
                    "rank": rank,
                    "top": [str(c) for c in cand_sids[:10]],
                }
            )

    metrics = {"n": len(ranks), "invalid_generations": invalid, "cc": invalid}
    for k in ks:
        metrics[f"HR@{k}"] = sum(hit_at_k(r, k) for r in ranks) / max(1, len(ranks))
        metrics[f"NDCG@{k}"] = sum(ndcg_at_k(r, k) for r in ranks) / max(1, len(ranks))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "predictions": predictions}, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--test_file", type=Path, required=True)
    parser.add_argument("--sid_map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_beams", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    evaluate(
        args.model_path,
        args.test_file,
        args.sid_map,
        args.output,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
