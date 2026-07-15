"""Self-contained full-parameter GRPO loop with constrained beam sampling.

Memory-oriented for 4x V100-16GB:
- policy on GPU (fp16) + AdamW8bit
- reference: cpu | cuda (always on GPU) | cuda_pingpong (GPU forward, CPU storage)
- per-completion backward (do not retain G graphs)
- memory-efficient token logprob (no full-vocab log_softmax materialization)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
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


def collate_prompts(batch, tokenizer, cutoff=384):
    prompts = [b["prompt"] for b in batch]
    answers = [b["answer"] for b in batch]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=cutoff)
    return enc, answers, prompts


def build_optimizer(params, lr: float, optim_name: str):
    name = (optim_name or "adamw8bit").lower()
    if name in ("adamw8bit", "paged_adamw8bit", "8bit"):
        try:
            import bitsandbytes as bnb

            if name == "paged_adamw8bit" and hasattr(bnb.optim, "PagedAdamW8bit"):
                return bnb.optim.PagedAdamW8bit(params, lr=lr)
            return bnb.optim.AdamW8bit(params, lr=lr)
        except Exception as exc:  # noqa: BLE001
            print(f"[rl] bitsandbytes optimizer unavailable ({exc}); falling back to AdamW")
    return torch.optim.AdamW(params, lr=lr)


@torch.no_grad()
def generate_group(model, tokenizer, enc, processor, num_generations, max_new_tokens, beam_search, device):
    processor.set_prompt_lengths(enc["attention_mask"].sum(dim=1).tolist())
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_generations,
        logits_processor=[processor],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if beam_search:
        outs = model.generate(
            **gen_kwargs,
            num_beams=num_generations,
            do_sample=False,
        )
    else:
        outs = model.generate(
            **gen_kwargs,
            do_sample=True,
            temperature=1.0,
        )
    return outs


def completion_logprobs(model, input_ids, attention_mask, prompt_len, pad_token_id=None):
    """Mean completion token logprob without materializing full-vocab log_softmax."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1]
    labels = input_ids[:, 1:]
    # gather - logsumexp avoids an extra [B,S,V] softmax buffer
    selected = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    lse = torch.logsumexp(logits.float(), dim=-1)
    token_lp = selected.float() - lse

    seq_len = labels.size(1)
    idx = torch.arange(seq_len, device=labels.device)
    mask = idx >= (prompt_len - 1)
    if pad_token_id is not None:
        mask = mask & (labels != pad_token_id)
    token_lp = token_lp * mask
    lengths = mask.sum(dim=1).clamp(min=1)
    return (token_lp.sum(dim=1) / lengths).to(dtype=logits.dtype)


def resolve_ref_mode(ref_device_name: str) -> tuple[str, bool]:
    """Return (storage, pingpong_to_gpu). storage is 'cpu' or 'cuda'."""
    name = str(ref_device_name).lower()
    if name == "cpu":
        return "cpu", False
    if name in ("cuda", "gpu"):
        return "cuda", False
    if name in ("cuda_pingpong", "pingpong", "gpu_pingpong"):
        return "cpu", True
    raise ValueError(
        f"Unsupported ref_device={ref_device_name!r} "
        "(use cpu | cuda | cuda_pingpong)"
    )


@torch.no_grad()
def ref_completion_logprobs(
    ref_model,
    *,
    device: torch.device,
    pingpong: bool,
    input_ids,
    attention_mask,
    prompt_len: int,
    pad_token_id=None,
):
    """Reference logprob; optionally stage ref weights on GPU only for this forward."""
    if pingpong:
        ref_model.to(device, non_blocking=True)
    lp = completion_logprobs(
        ref_model, input_ids, attention_mask, prompt_len, pad_token_id=pad_token_id
    )
    if pingpong:
        ref_model.to("cpu", non_blocking=True)
        torch.cuda.empty_cache()
    return lp.to(device)


def prune_checkpoints(output_dir: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    ckpts = sorted(
        (p for p in output_dir.glob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    while len(ckpts) > save_total_limit:
        old = ckpts.pop(0)
        shutil.rmtree(old, ignore_errors=True)
        print(f"[rl] pruned old checkpoint {old.name}", flush=True)


def save_rl_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    raw_model,
    tokenizer,
    sched,
    log_history: list,
    cfg: dict,
    *,
    num_generations: int,
    world_size: int,
    accum: int,
    ref_device_name: str,
    ref_home: torch.device,
    ref_pingpong: bool,
    optim_name: str,
) -> Path:
    ckpt = output_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    raw_model.save_pretrained(ckpt)
    save_tok(tokenizer, ckpt)
    torch.save(sched.state_dict(), ckpt / "scheduler.pt")
    trainer_state = {
        "step": step,
        "epoch": epoch,
        "log_history": log_history,
        "num_generations": num_generations,
        "world_size": world_size,
        "accum": accum,
        "optim": optim_name,
        "ref_mode": ref_device_name,
        "ref_home": str(ref_home),
        "ref_pingpong": ref_pingpong,
        "config": cfg,
    }
    with open(ckpt / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)
    latest = output_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(ckpt.name)
    print(f"[rl] saved checkpoint-{step} -> {ckpt}", flush=True)
    prune_checkpoints(output_dir, int(cfg.get("save_total_limit", 3)))
    return ckpt


def append_metrics_jsonl(path: Path, entry: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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

    # Help fragmentation on long RL runs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    prompt_cutoff = int(cfg.get("prompt_cutoff", 384))
    ref_device_name = str(cfg.get("ref_device", "cuda_pingpong"))
    ref_storage, ref_pingpong = resolve_ref_mode(ref_device_name)
    optim_name = str(cfg.get("optim", "adamw8bit"))
    max_new_tokens_cap = int(cfg.get("max_new_tokens_cap", 8))

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)

    # ref storage: cpu (incl. pingpong) or cuda (always resident on GPU)
    ref_home = torch.device("cpu" if ref_storage == "cpu" else device)
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.float16
    ).to(ref_home)
    if ref_storage == "cuda":
        ref_model.config.use_cache = False

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
        collate_fn=lambda b: collate_prompts(b, tokenizer, cutoff=prompt_cutoff),
    )

    raw_model = model.module if hasattr(model, "module") else model
    optim = build_optimizer(raw_model.parameters(), lr=float(cfg["learning_rate"]), optim_name=optim_name)
    if local_rank <= 0:
        ref_mode = ref_device_name if not ref_pingpong else "cuda_pingpong"
        print(
            f"[rl] device={device} ref_mode={ref_mode} ref_home={ref_home} "
            f"optim={optim.__class__.__name__} G={num_generations} "
            f"prompt_cutoff={prompt_cutoff} checkpointing={cfg.get('gradient_checkpointing', True)}",
            flush=True,
        )

    epochs = int(cfg["num_train_epochs"])
    global_batch = int(cfg.get("global_batch_size", 512))
    accum = max(1, global_batch // (per_device * world_size))
    steps_per_epoch = max(1, math.ceil(len(loader) / accum))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup = int(total_steps * float(cfg.get("warmup_ratio", 0.03)))
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)
    beta = float(cfg.get("beta", 0.001))
    max_new_tokens = int(cfg.get("max_completion_length", 128))
    beam_search = bool(cfg.get("beam_search", True))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.3))
    logging_steps = int(cfg.get("logging_steps", 1))
    save_steps = int(cfg.get("save_steps", 100))
    metrics_jsonl = args.output_dir / "metrics.jsonl"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    log_history = []
    pad_id = tokenizer.pad_token_id
    micro_step = 0
    t_step = time.time()

    if local_rank <= 0:
        print(
            f"[rl] train start steps/epoch≈{steps_per_epoch} total_opt_steps≈{total_steps} "
            f"accum={accum} logging_steps={logging_steps} save_steps={save_steps}",
            flush=True,
        )

    # running aggregates between optimizer steps (for logging)
    run_loss = 0.0
    run_reward = 0.0
    run_kl = 0.0
    run_lp = 0.0
    run_n = 0
    run_micro_batches = 0

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        optim.zero_grad(set_to_none=True)
        for batch_idx, (enc, answers, prompts) in enumerate(tqdm(loader, disable=local_rank > 0)):
            micro_step += 1
            # Generation: enable KV cache, no grad
            raw_model.eval()
            raw_model.config.use_cache = True
            with torch.no_grad():
                outs = generate_group(
                    raw_model,
                    tokenizer,
                    enc,
                    processor,
                    num_generations=num_generations,
                    max_new_tokens=min(max_new_tokens, max_new_tokens_cap),
                    beam_search=beam_search,
                    device=device,
                )
            raw_model.config.use_cache = False
            raw_model.train()
            torch.cuda.empty_cache()

            prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
            bsz = len(answers)
            n_terms = max(1, bsz * num_generations)
            batch_loss_value = 0.0
            batch_reward_value = 0.0
            sync_now = ((batch_idx + 1) % accum == 0)

            term_i = 0
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
                batch_reward_value += float(rewards.mean().item())

                # Micro-backward: one completion graph at a time
                for g in range(num_generations):
                    term_i += 1
                    seq = group_ids[g].unsqueeze(0).to(device)
                    attn = torch.ones_like(seq)
                    if pad_id is not None:
                        attn = (seq != pad_id).long()

                    # Ref first (pingpong: ref on GPU only during this forward).
                    with torch.no_grad():
                        if ref_pingpong or ref_storage == "cuda":
                            lp_ref = ref_completion_logprobs(
                                ref_model,
                                device=device,
                                pingpong=ref_pingpong,
                                input_ids=seq,
                                attention_mask=attn,
                                prompt_len=plen,
                                pad_token_id=pad_id,
                            )
                        else:
                            lp_ref = ref_completion_logprobs(
                                ref_model,
                                device=device,
                                pingpong=False,
                                input_ids=seq.cpu(),
                                attention_mask=attn.cpu(),
                                prompt_len=plen,
                                pad_token_id=pad_id,
                            )

                    lp = completion_logprobs(raw_model, seq, attn, plen, pad_token_id=pad_id)
                    kl = lp.float() - lp_ref.float()
                    loss = (-(adv[g] * lp.float()) + beta * kl) / (accum * n_terms)

                    last_term = term_i == n_terms
                    sync_ctx = (
                        nullcontext()
                        if (not distributed or (sync_now and last_term))
                        else model.no_sync()
                    )
                    with sync_ctx:
                        loss.backward()
                    term_loss = float(loss.detach().item()) * accum * n_terms
                    batch_loss_value += term_loss
                    run_kl += float(kl.detach().item())
                    run_lp += float(lp.detach().item())
                    run_n += 1

                    del lp, lp_ref, kl, loss, seq, attn

            del outs
            torch.cuda.empty_cache()
            run_loss += batch_loss_value / n_terms
            run_reward += batch_reward_value / max(1, bsz)
            run_micro_batches += 1

            if sync_now:
                grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)
                if hasattr(grad_norm, "item"):
                    grad_norm = float(grad_norm.item())
                else:
                    grad_norm = float(grad_norm)
                optim.step()
                lr = float(optim.param_groups[0]["lr"])
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                now = time.time()
                sec = now - t_step
                t_step = now
                remain = max(0, total_steps - step)
                eta_h = remain * sec / 3600.0 if sec == sec else float("nan")
                epoch_f = epoch + (batch_idx + 1) / max(1, len(loader))

                if local_rank <= 0 and (step % logging_steps == 0 or step == 1):
                    denom = max(1, run_n)
                    micro_denom = max(1, run_micro_batches)
                    entry = {
                        "step": step,
                        "micro_step": micro_step,
                        "epoch": epoch_f,
                        "loss": run_loss / micro_denom,
                        "reward_mean": run_reward / micro_denom,
                        "kl_mean": run_kl / denom,
                        "lp_mean": run_lp / denom,
                        "grad_norm": grad_norm,
                        "learning_rate": lr,
                        "sec_per_step": sec,
                        "accum": accum,
                    }
                    log_history.append(entry)
                    append_metrics_jsonl(metrics_jsonl, entry)
                    print(
                        f"[rl] step={step}/{total_steps} micro={micro_step} "
                        f"epoch={epoch_f:.4f} {sec:.1f}s/step eta≈{eta_h:.1f}h "
                        f"loss={entry['loss']:.4f} reward_mean={entry['reward_mean']:.4f} "
                        f"kl_mean={entry['kl_mean']:.4f} lp_mean={entry['lp_mean']:.4f} "
                        f"grad_norm={grad_norm:.4f} learning_rate={lr:.6g}",
                        flush=True,
                    )
                    print(
                        f"[rl] metrics step={step} epoch={epoch_f:.4f} "
                        + " ".join(f"{k}={entry[k]:.6g}" if isinstance(entry[k], float) else f"{k}={entry[k]}" for k in (
                            "loss", "reward_mean", "kl_mean", "lp_mean", "grad_norm", "learning_rate"
                        )),
                        flush=True,
                    )
                    run_loss = 0.0
                    run_reward = 0.0
                    run_kl = 0.0
                    run_lp = 0.0
                    run_n = 0
                    run_micro_batches = 0

                if local_rank <= 0 and save_steps > 0 and step % save_steps == 0:
                    save_rl_checkpoint(
                        args.output_dir,
                        step,
                        epoch,
                        raw_model,
                        tokenizer,
                        sched,
                        log_history,
                        cfg,
                        num_generations=num_generations,
                        world_size=world_size,
                        accum=accum,
                        ref_device_name=ref_device_name,
                        ref_home=ref_home,
                        ref_pingpong=ref_pingpong,
                        optim_name=optim_name,
                    )

        # sync ref optionally (CPU-safe)
        if cfg.get("sync_ref_model", True) and local_rank <= 0:
            cpu_state = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
            ref_model.load_state_dict(cpu_state, strict=True)
            del cpu_state

    if local_rank <= 0:
        if step > 0 and save_steps > 0 and step % save_steps != 0:
            save_rl_checkpoint(
                args.output_dir,
                step,
                epochs - 1,
                raw_model,
                tokenizer,
                sched,
                log_history,
                cfg,
                num_generations=num_generations,
                world_size=world_size,
                accum=accum,
                ref_device_name=ref_device_name,
                ref_home=ref_home,
                ref_pingpong=ref_pingpong,
                optim_name=optim_name,
            )
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
                    "optim": optim.__class__.__name__,
                    "ref_mode": ref_device_name,
                    "ref_home": str(ref_home),
                    "ref_pingpong": ref_pingpong,
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
