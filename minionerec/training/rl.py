"""
Official-source RL (GRPO-like) trainer.

Source: MiniOneRec-official @ 0c64b955
  rl.py + rl.sh + minionerec_trainer.ReReTrainer

Loss is official_source (fake ratio==1), NOT paper old-policy GRPO.
Legacy minionerec/rl.py retained as legacy_grpo_like.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset
from torch.utils.data import ConcatDataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from transformers import AutoTokenizer

from minionerec.generation.constraints import (
    ConstrainedLogitsProcessor,
    build_sid_hash_dict,
    load_semantic_ids_from_info,
    make_prefix_fn,
)
from minionerec.runtime.distributed import (
    all_reduce_mean_scalar,
    all_reduce_sum_scalar,
    barrier,
    cleanup_distributed,
    is_main_process,
    print_rank0,
    resolve_effective_batch,
    setup_distributed,
    unwrap_model,
)
from minionerec.training.objectives import RolloutBatch, build_objective, selective_log_softmax
from minionerec.data.datasets import RLSeqTitle2SidDataset, RLTitle2SidDataset, SidDataset
from minionerec.rewards.ranking import build_reward_funcs, group_advantages
from minionerec.runtime.paths import project_root, resolve_path
from minionerec.data.adapters import ensure_official_paths
from minionerec.config import _LEGACY_VARIANT


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _is_v100() -> bool:
    return torch.cuda.is_available() and "v100" in torch.cuda.get_device_name(0).lower()


def _get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    logits = logits[:, -logits_to_keep:]
    ids = input_ids[:, -logits_to_keep:]
    return selective_log_softmax(logits, ids)


class RepeatRandomSampler(torch.utils.data.Sampler):
    """Official RepeatRandomSampler (minionerec_trainer.py)."""

    def __init__(self, data_source, repeat_count: int, seed: int | None = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


def build_rl_datasets(paths, cfg, seed: int):
    category = cfg.get("category_prompt") or "industrial and scientific items"
    sample = int(cfg.get("max_train_samples", -1))
    train_datasets = []
    d1 = SidDataset(str(paths.train_csv), sample=sample, seed=seed, category=category)
    train_datasets.append(d1)
    d2 = RLTitle2SidDataset(
        item_file=str(paths.item_json),
        index_file=str(paths.index_json),
        sample=sample,
        seed=seed,
        category=category,
    )
    train_datasets.append(d2)
    seq_sample = 10000 if sample < 0 else min(10000, sample)
    if cfg.get("rl_seq_title_sample") is not None:
        seq_sample = int(cfg["rl_seq_title_sample"])
    d3 = RLSeqTitle2SidDataset(str(paths.train_csv), sample=seq_sample, seed=seed, category=category)
    train_datasets.append(d3)
    # RLSid2Title / RLSidhis2Title commented out in official rl.py — do not enable.
    train_data = ConcatDataset(train_datasets)
    eval_sample = int(cfg.get("max_eval_samples", -1))
    eval_data = SidDataset(str(paths.valid_csv), sample=eval_sample, seed=seed, category=category)

    prompt2history: dict = {}
    history2target: dict = {}
    for ds in train_datasets:
        prompt2history.update(ds.prompt2history)
        history2target.update(ds.history2target)
    prompt2history.update(eval_data.prompt2history)
    history2target.update(eval_data.history2target)

    train_dataset = Dataset.from_dict({k: [elm[k] for elm in train_data] for k in train_data[0].keys()})
    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = Dataset.from_dict({k: [elm[k] for elm in eval_data] for k in eval_data[0].keys()})
    eval_dataset = eval_dataset.shuffle(seed=seed)
    return train_dataset, eval_dataset, prompt2history, history2target, train_datasets


@torch.no_grad()
def constrained_generate(
    model,
    tokenizer,
    prompt_ids,
    prompt_mask,
    *,
    hash_dict,
    base_model: str,
    num_beams: int,
    beam_search: bool,
    temperature: float,
    max_new_tokens: int,
):
    prefix_fn = make_prefix_fn(hash_dict)
    ccc = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=prefix_fn,
        num_beams=num_beams if beam_search else 1,
        base_model=base_model,
        eos_token_id=tokenizer.eos_token_id,
    )
    ccc.reset()
    # Official orders Temperature then Constrained; for pure beam (do_sample=False)
    # Temperature can destabilize scores — skip when beam_search.
    if beam_search:
        lp = LogitsProcessorList([ccc])
        gen_cfg = GenerationConfig(
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        lp = LogitsProcessorList([TemperatureLogitsWarper(temperature=temperature), ccc])
        gen_cfg = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        generation_config=gen_cfg,
        logits_processor=lp,
    )


def run_rl(cfg: dict[str, Any]) -> Path:
    rank, local_rank, world_size, device, distributed = setup_distributed()
    root = project_root()
    os.chdir(root)
    seed = int(cfg.get("seed", 42))
    set_seed(seed + rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)

    impl = cfg.get("implementation_target", "official_source")
    assert impl in ("official_source", "paper_aligned", "legacy_grpo_like", "official", "paper", "legacy")
    objective = build_objective(
        impl,
        beta=float(cfg.get("beta", 1e-3)),
        clip_eps=float(cfg.get("clip_eps", 0.2)),
        dapo=bool(cfg.get("dapo", False)),
        gspo=bool(cfg.get("gspo", False)),
    )

    out_dir = resolve_path(cfg["output_dir"], root)
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    paths = ensure_official_paths(
        {
            **cfg,
            "processed_data_root": str(resolve_path(cfg["processed_data_root"], root)),
            "official_format_root": str(resolve_path(cfg.get("official_format_root", "data/official_format"), root)),
        }
    )

    model_path = str(resolve_path(cfg["model_name_or_path"], root))
    # V100: fp16 policy/ref quickly NaNs with new SID tokens; use fp32 for correctness.
    if _is_v100():
        dtype = torch.float32
        print_rank0("[rl] V100 detected: using float32 for policy/ref")
    else:
        dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Reference model (frozen) — not wrapped in DDP
    ref_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    train_dataset, eval_dataset, prompt2history, history2target, _ = build_rl_datasets(paths, cfg, seed)

    G = int(cfg.get("num_generations", 16))
    if cfg.get("smoke_g4"):
        assert G == 4, "smoke_g4 config must set num_generations=4"
        print_rank0("[rl] WARNING: smoke_g4 — not a formal MiniOneRec alignment run")

    reward_type = cfg.get("reward_type", "ranking")
    reward_funcs = build_reward_funcs(reward_type, prompt2history, history2target, G)
    if not isinstance(reward_funcs, list):
        reward_funcs = [reward_funcs]

    semantic_ids = load_semantic_ids_from_info(str(paths.info_txt))
    hash_dict = build_sid_hash_dict(tokenizer, semantic_ids, base_model=model_path)

    # Official rl.sh: train_batch_size 64, grad_accum 2, 8 GPUs -> effective prompt batch
    train_bs = int(cfg.get("train_batch_size", 2))  # per-device prompt batch
    accum = int(cfg.get("gradient_accumulation_steps", 8))
    lr = float(cfg.get("learning_rate", 1e-5))
    beta = float(cfg.get("beta", 1e-3))
    epochs = int(cfg.get("num_train_epochs", 1))
    max_steps = cfg.get("max_steps")
    beam_search = bool(cfg.get("beam_search", True))
    temperature = float(cfg.get("temperature", 1.0))
    max_new_tokens = int(cfg.get("max_completion_length", 128))
    clip_eps = float(cfg.get("clip_eps", 0.2))
    updates_per_rollout = int(cfg.get("updates_per_rollout", 1))
    eff_batch = resolve_effective_batch(train_bs, accum, world_size)

    prompts = list(train_dataset["prompt"])
    if cfg.get("max_train_samples") and int(cfg["max_train_samples"]) > 0:
        prompts = prompts[: int(cfg["max_train_samples"])]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=False)
    # Global steps estimate: each rank processes ~len/world_size prompts per epoch
    prompts_per_rank = max(1, math.ceil(len(prompts) / world_size))
    steps_per_epoch = max(1, math.ceil(prompts_per_rank / max(1, train_bs)))
    total_steps = steps_per_epoch * epochs
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))
    warmup = int(0.03 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    metrics_f = None
    if is_main_process():
        metrics_path = out_dir / "metrics.jsonl"
        metrics_f = open(metrics_path, "w", encoding="utf-8")
        resolved = {
            **cfg,
            "implementation_target": impl,
            "num_generations": G,
            "beta": beta,
            "beam_search": beam_search,
            "train_batch_size_per_device": train_bs,
            "gradient_accumulation_steps": accum,
            "world_size": world_size,
            "effective_prompt_batch": eff_batch,
            "dtype": str(dtype),
            "model_path": model_path,
            "n_train_prompts": len(prompts),
            "official_paths": paths.to_dict(),
        }
        (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        tokenizer.save_pretrained(out_dir / "tokenizer")
        print_rank0(
            f"[rl] world_size={world_size} per_device_bs={train_bs} accum={accum} "
            f"effective_prompt_batch={eff_batch} n_prompts={len(prompts)}"
        )

    model.train()
    global_step = 0
    best_reward = -1e9
    best_dir = out_dir / "best_checkpoint"
    final_dir = out_dir / "final_checkpoint"

    def save_ckpt(path: Path):
        if not is_main_process():
            return
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        unwrap_model(model).save_pretrained(path)
        tokenizer.save_pretrained(path)

    # Shard prompts across ranks (deterministic shuffle with base seed)
    order = list(range(len(prompts)))
    random.Random(seed).shuffle(order)
    order = order[rank::world_size]

    invalid_total = 0
    valid_total = 0
    raw_model = unwrap_model(model)

    for epoch in range(epochs):
        for start in range(0, len(order), train_bs):
            if max_steps is not None and global_step >= int(max_steps):
                break
            batch_idx = order[start : start + train_bs]
            batch_prompts = [prompts[i] for i in batch_idx]

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(cfg.get("max_prompt_length", 512)),
                add_special_tokens=False,
            )
            prompt_ids = enc["input_ids"].to(device)
            prompt_mask = enc["attention_mask"].to(device)

            raw_model.eval()
            with torch.no_grad():
                if beam_search:
                    gen = constrained_generate(
                        raw_model,
                        tokenizer,
                        prompt_ids,
                        prompt_mask,
                        hash_dict=hash_dict,
                        base_model=model_path,
                        num_beams=G,
                        beam_search=True,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                    )
                    prompt_len = prompt_ids.size(1)
                    prompt_ids_exp = prompt_ids.repeat_interleave(G, dim=0)
                    prompt_mask_exp = prompt_mask.repeat_interleave(G, dim=0)
                    completion_ids = gen[:, prompt_len:]
                    prompts_exp = [p for p in batch_prompts for _ in range(G)]
                else:
                    comps = []
                    for _ in range(G):
                        g = constrained_generate(
                            raw_model,
                            tokenizer,
                            prompt_ids,
                            prompt_mask,
                            hash_dict=hash_dict,
                            base_model=model_path,
                            num_beams=1,
                            beam_search=False,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                        )
                        comps.append(g[:, prompt_ids.size(1) :])
                    max_c = max(c.size(1) for c in comps)
                    padded = []
                    for c in comps:
                        if c.size(1) < max_c:
                            pad = torch.full(
                                (c.size(0), max_c - c.size(1)),
                                tokenizer.pad_token_id,
                                device=device,
                                dtype=c.dtype,
                            )
                            c = torch.cat([c, pad], dim=1)
                        padded.append(c)
                    B = prompt_ids.size(0)
                    stacked = torch.stack(padded, dim=1)  # (B, G, C)
                    completion_ids = stacked.reshape(B * G, -1)
                    prompt_ids_exp = prompt_ids.repeat_interleave(G, dim=0)
                    prompt_mask_exp = prompt_mask.repeat_interleave(G, dim=0)
                    prompts_exp = [p for p in batch_prompts for _ in range(G)]

            completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            known = {s.strip("\n\" ") for s in semantic_ids}
            valid_flags = []
            for t in completions_text:
                ok = t.strip("\n\" ") in known
                valid_flags.append(ok)
            n_valid = sum(valid_flags)
            n_invalid = len(valid_flags) - n_valid
            valid_total += n_valid
            invalid_total += n_invalid
            valid_rate = n_valid / max(1, len(valid_flags))
            invalid_rate = 1.0 - valid_rate

            rewards_mat = []
            for fn in reward_funcs:
                rewards_mat.append(fn(prompts=prompts_exp, completions=completions_text))
            rewards = torch.tensor(rewards_mat, dtype=torch.float32, device=device).sum(dim=0)
            advantages = torch.tensor(
                group_advantages(rewards.tolist(), G, eps=1e-4), dtype=torch.float32, device=device
            )

            is_eos = completion_ids == tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            seq_idx = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

            prompt_completion_ids = torch.cat([prompt_ids_exp, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask_exp, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            with torch.no_grad():
                ref_logps = _get_per_token_logps(ref_model, prompt_completion_ids, attention_mask, logits_to_keep)
                needs_old = _LEGACY_VARIANT.get(impl, impl) in ("paper", "paper_aligned")
                if needs_old:
                    old_logps = _get_per_token_logps(
                        raw_model, prompt_completion_ids, attention_mask, logits_to_keep
                    ).detach()
                else:
                    old_logps = None

            rollout = RolloutBatch(
                prompt_ids=prompt_ids_exp,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                rewards=rewards,
                advantages=advantages,
                old_log_probs=old_logps,
                reference_log_probs=ref_logps,
            )

            model.train()
            loss = None
            loss_metrics: dict[str, Any] = {}
            for _u in range(updates_per_rollout):
                logps = _get_per_token_logps(model, prompt_completion_ids, attention_mask, logits_to_keep)
                loss, loss_metrics = objective.compute(logps, rollout)
                if not torch.isfinite(loss):
                    print_rank0(f"[rl] non-finite loss at step={global_step}; skipping update")
                    optimizer.zero_grad(set_to_none=True)
                    break
                (loss / accum).backward()

            if (global_step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("max_grad_norm", 0.3)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Aggregate metrics across ranks for logging
            reward_mean = all_reduce_mean_scalar(float(rewards.mean().item()), device)
            invalid_rate_g = all_reduce_mean_scalar(invalid_rate, device)
            valid_rate_g = all_reduce_mean_scalar(valid_rate, device)
            loss_val = float(loss.detach().item()) if loss is not None else float("nan")
            loss_val = all_reduce_mean_scalar(loss_val, device)

            dup_rates = []
            for i in range(0, len(completions_text), G):
                group = completions_text[i : i + G]
                dup_rates.append(1.0 - len(set(group)) / G)

            row = {
                "step": global_step,
                "epoch": epoch,
                "world_size": world_size,
                "loss": loss_val,
                "policy_loss": loss_metrics.get("policy_loss"),
                "kl_loss": loss_metrics.get("kl_mean"),
                "kl_mean": loss_metrics.get("kl_mean"),
                "beta": beta,
                "reward_total_mean": reward_mean,
                "reward_std": float(rewards.std().item()) if rewards.numel() > 1 else 0.0,
                "advantage_mean": float(advantages.mean().item()),
                "advantage_std": float(advantages.std().item()) if advantages.numel() > 1 else 0.0,
                "advantage_min": float(advantages.min().item()),
                "advantage_max": float(advantages.max().item()),
                "ratio_mean": loss_metrics.get("ratio_mean"),
                "ratio_min": loss_metrics.get("ratio_min"),
                "ratio_max": loss_metrics.get("ratio_max"),
                "clip_fraction": loss_metrics.get("clip_fraction"),
                "completion_length": float(completion_mask.sum(1).float().mean().item()),
                "valid_completion_rate": valid_rate_g,
                "invalid_completion_rate": invalid_rate_g,
                "duplicate_rate": float(np.mean(dup_rates)),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "peak_gpu_memory": (
                    float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0
                ),
            }
            if len(reward_funcs) >= 2:
                r0 = torch.tensor(reward_funcs[0](prompts=prompts_exp, completions=completions_text))
                r1 = torch.tensor(reward_funcs[1](prompts=prompts_exp, completions=completions_text))
                row["reward_rule_mean"] = float(r0.mean())
                row["reward_rank_mean"] = float(r1.mean())

            if is_main_process() and metrics_f is not None:
                metrics_f.write(json.dumps(row) + "\n")
                metrics_f.flush()
                if global_step % int(cfg.get("logging_steps", 1)) == 0:
                    print_rank0(
                        f"[rl] step={global_step} loss={row['loss']:.4f} reward={row['reward_total_mean']:.4f} "
                        f"invalid={invalid_rate_g:.4f} kl={row['kl_mean']} world_size={world_size}"
                    )

            if is_main_process() and row["reward_total_mean"] > best_reward:
                best_reward = row["reward_total_mean"]
                save_ckpt(best_dir)

            if invalid_rate_g > 0 and cfg.get("fail_on_invalid", True) and global_step >= 2:
                print_rank0("[rl] ERROR invalid_completion_rate > 0; dumping samples")
                if is_main_process():
                    for t in completions_text[:20]:
                        print("  FAIL:", repr(t))
                if cfg.get("strict_invalid_abort", False):
                    cleanup_distributed()
                    raise RuntimeError("invalid_completion_rate > 0")

            global_step += 1
            if max_steps is not None and global_step >= int(max_steps):
                break
        if max_steps is not None and global_step >= int(max_steps):
            break

    if metrics_f is not None:
        metrics_f.close()

    barrier()
    save_ckpt(final_dir)
    if is_main_process() and not best_dir.exists():
        save_ckpt(best_dir)

    valid_total_g = int(all_reduce_sum_scalar(float(valid_total), device))
    invalid_total_g = int(all_reduce_sum_scalar(float(invalid_total), device))
    summary = {
        "best_reward": best_reward if is_main_process() else None,
        "steps": global_step,
        "world_size": world_size,
        "valid_completion_rate": valid_total_g / max(1, valid_total_g + invalid_total_g),
        "invalid_completion_rate": invalid_total_g / max(1, valid_total_g + invalid_total_g),
        "best_checkpoint": str(best_dir),
        "final_checkpoint": str(final_dir),
        "implementation_target": impl,
    }
    if is_main_process():
        _save_json(out_dir / "train_metrics.json", summary)
        print_rank0("[rl] done: " + json.dumps(summary, indent=2))
        if summary["invalid_completion_rate"] > 0:
            print_rank0(
                "[rl] WARNING: invalid completions observed; do not treat as valid RL experiment if rate != 0"
            )

    cleanup_distributed()
    return best_dir


@dataclass
class TrainingResult:
    best_checkpoint: Path
    final_checkpoint: Path | None = None
    metrics: dict[str, Any] | None = None


def train_rl(config: "RLConfig | dict[str, Any]") -> TrainingResult:
    """Public RL entry. Accepts typed ``RLConfig`` or legacy flat dict."""
    from minionerec.config import RLConfig, dump_resolved

    if isinstance(config, RLConfig):
        cfg = config.to_legacy_dict()
        out = resolve_path(config.runtime.output_dir, project_root())
        dump_resolved(config, out / "config.resolved.yaml")
    else:
        cfg = config
    best = run_rl(cfg)
    final = Path(cfg["output_dir"]) / "final_checkpoint"
    metrics_path = Path(cfg["output_dir"]) / "train_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    return TrainingResult(best_checkpoint=Path(best), final_checkpoint=final if final.exists() else None, metrics=metrics)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_rl(cfg)


if __name__ == "__main__":
    main()
