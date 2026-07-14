"""Self-contained full-parameter GRPO loop with constrained beam sampling.

This avoids brittle TRL API differences while matching paper §3.4:
group-relative advantages, KL to reference, hybrid rewards, constrained beams.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from minionerec.dataset import RLData
from minionerec.decode import SIDConstraint, make_trie, load_sids
from minionerec.model import save_tok
from minionerec.reward import hybrid, rule


def group_rewards(preds, gold, reward_type: str):
    if reward_type == "rule":
        return [rule(p, gold) for p in preds]
    return hybrid(preds, gold)


def collate_prompts(batch, tokenizer, cutoff=512):
    prompts = [b["prompt"] for b in batch]
    answers = [b["answer"] for b in batch]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=cutoff)
    return enc, answers, prompts


@torch.no_grad()
def generate_group(model, tokenizer, enc, processor, num_generations, max_new_tokens, beam_search, device):
    processor.set_prompt_lengths(enc["attention_mask"].sum(dim=1).tolist())
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    if beam_search:
        outs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_generations,
            num_return_sequences=num_generations,
            do_sample=False,
            logits_processor=[processor],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        outs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=num_generations,
            logits_processor=[processor],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return outs


def completion_logprobs(model, input_ids, attention_mask, prompt_len):
    """Sum token logprobs of completion tokens."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # mask prompt region (shifted)
    seq_len = labels.size(1)
    idx = torch.arange(seq_len, device=labels.device)
    mask = (idx >= (prompt_len - 1)) & (labels != -100)
    # also ignore pads
    mask = mask & (labels != model.config.pad_token_id if model.config.pad_token_id is not None else True)
    token_lp = token_lp * mask
    lengths = mask.sum(dim=1).clamp(min=1)
    return token_lp.sum(dim=1) / lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rl.yaml"))
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--sid_map", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_generations", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_generations = args.num_generations or int(cfg.get("num_generations", 16))
    model_dir = args.model_path / "final" if (args.model_path / "final").exists() else args.model_path
    tok_dir = args.model_path / "tokenizer" if (args.model_path / "tokenizer").exists() else model_dir

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    sid_map = load_sids(args.sid_map)
    trie = make_trie(tokenizer, sid_map)
    processor = SIDConstraint(trie, tokenizer)

    train_ds = RLData(args.train_file, tokenizer)
    if args.max_samples > 0:
        train_ds.rows = train_ds.rows[: args.max_samples]

    sampler = (
        torch.utils.data.distributed.DistributedSampler(train_ds)
        if distributed
        else None
    )
    per_device = int(cfg.get("per_device_train_batch_size", 1))
    loader = DataLoader(
        train_ds,
        batch_size=per_device,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda b: collate_prompts(b, tokenizer),
    )

    raw_model = model.module if hasattr(model, "module") else model
    optim = torch.optim.AdamW(raw_model.parameters(), lr=float(cfg["learning_rate"]))
    epochs = int(cfg["num_train_epochs"])
    total_steps = max(1, epochs * math.ceil(len(loader)))
    warmup = int(total_steps * float(cfg.get("warmup_ratio", 0.03)))
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)
    beta = float(cfg.get("beta", 0.001))
    max_new_tokens = int(cfg.get("max_completion_length", 128))
    beam_search = bool(cfg.get("beam_search", True))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.3))

    # grad accum to approach global batch of prompts
    global_batch = int(cfg.get("global_batch_size", 512))
    accum = max(1, global_batch // (per_device * world_size))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    log_history = []

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running = 0.0
        optim.zero_grad()
        for batch_idx, (enc, answers, prompts) in enumerate(tqdm(loader, disable=local_rank > 0)):
            enc_cpu = enc
            with torch.no_grad():
                outs = generate_group(
                    raw_model,
                    tokenizer,
                    enc_cpu,
                    processor,
                    num_generations=num_generations,
                    max_new_tokens=min(max_new_tokens, 8),
                    beam_search=beam_search,
                    device=device,
                )

            # decode completions per prompt
            prompt_lens = enc_cpu["attention_mask"].sum(dim=1).tolist()
            bsz = len(answers)
            loss_terms = []
            for b in range(bsz):
                plen = int(prompt_lens[b])
                group_ids = outs[b * num_generations : (b + 1) * num_generations]
                comps = [
                    tokenizer.decode(seq[plen:], skip_special_tokens=False) for seq in group_ids
                ]
                rewards = torch.tensor(
                    group_rewards(comps, answers[b], cfg.get("reward_type", "ranking")),
                    device=device,
                    dtype=torch.float32,
                )
                adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)

                # compute logprobs for each completion under policy and ref
                for g in range(num_generations):
                    seq = group_ids[g].unsqueeze(0).to(device)
                    attn = torch.ones_like(seq)
                    # pad token mask
                    if tokenizer.pad_token_id is not None:
                        attn = (seq != tokenizer.pad_token_id).long()
                    lp = completion_logprobs(raw_model, seq, attn, plen)
                    with torch.no_grad():
                        lp_ref = completion_logprobs(ref_model, seq, attn, plen)
                    # GRPO-style: advantage * logprob - beta * KL approx (lp - lp_ref)
                    kl = lp - lp_ref
                    loss_terms.append(-(adv[g] * lp) + beta * kl)

            if not loss_terms:
                continue
            loss = torch.stack(loss_terms).mean() / accum
            loss.backward()
            running += loss.item() * accum
            if (batch_idx + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1
                if local_rank <= 0 and step % 10 == 0:
                    entry = {"step": step, "epoch": epoch, "loss": running / 10}
                    log_history.append(entry)
                    print(entry)
                    running = 0.0

        # sync ref optionally
        if cfg.get("sync_ref_model", True) and local_rank <= 0:
            ref_model.load_state_dict(raw_model.state_dict())
            if distributed:
                # broadcast already same on rank0 copy; others reload next epoch from checkpoint if needed
                pass

    if local_rank <= 0:
        out_final = args.output_dir / "final"
        out_final.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(out_final)
        save_tok(tokenizer, out_final)
        with open(args.output_dir / "train_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "steps": step,
                    "num_generations": num_generations,
                    "world_size": world_size,
                    "accum": accum,
                },
                f,
                indent=2,
            )
        with open(args.output_dir / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump(log_history, f, indent=2)
        print(f"Saved RL model to {out_final}")

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
