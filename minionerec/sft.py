"""Full-parameter multi-task SFT entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import warnings
from functools import partial
from pathlib import Path

import torch
import yaml
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.utils import logging as hf_logging

from minionerec.dataset import SFTData, collate
from minionerec.model import load_model, load_tokenizer, save_tok


def quiet_logs() -> None:
    """Silence noisy library logs; keep our [sft] lines."""
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"
    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_default_handler()
        hf_logging.enable_propagation()
    except Exception:
        pass
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.ERROR)
    for name in (
        "transformers",
        "transformers.modeling_utils",
        "transformers.configuration_utils",
        "transformers.tokenization_utils_base",
        "transformers.generation",
        "transformers.trainer",
        "transformers.training_args",
        "accelerate",
        "accelerate.accelerator",
        "accelerate.utils",
        "deepspeed",
        "deepspeed.utils",
        "torch",
        "torch.distributed",
        "torch.distributed.elastic",
        "torch.distributed.elastic.multiprocessing.redirects",
    ):
        log = logging.getLogger(name)
        log.setLevel(logging.ERROR)
        log.propagate = False



def resolve_micro_batch(size_key: str, cfg: dict, n_gpu: int, global_batch: int) -> tuple[int, int]:
    micro = int(cfg["micro_batch_size"][size_key])
    accum = max(1, global_batch // (micro * n_gpu))
    return micro, accum


def materialize_ds_config(src: str | Path, micro: int, accum: int, n_gpu: int, out_dir: Path) -> str:
    """Write DeepSpeed json with explicit batch knobs to avoid HF/DS mismatch."""
    with open(src, encoding="utf-8") as f:
        ds = json.load(f)
    ds["train_micro_batch_size_per_gpu"] = micro
    ds["gradient_accumulation_steps"] = accum
    ds["train_batch_size"] = micro * accum * n_gpu
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ds_sft.runtime.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ds, f, indent=2)
    return str(path)


class MicroStepProgressCallback(TrainerCallback):
    """Log inside an optimizer step so progress is visible with large grad_accum."""

    def __init__(self, every: int = 8):
        self.every = every
        self.micro = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.micro = 0

    def on_step_begin(self, args, state, control, **kwargs):
        # optimizer step boundary
        self.micro = 0

    def on_substep_end(self, args, state, control, **kwargs):
        # called after each grad-accum micro-batch (transformers>=4.36)
        self.micro += 1
        if state.is_world_process_zero and self.micro % self.every == 0:
            print(
                f"[sft] opt_step={state.global_step} micro={self.micro}/{args.gradient_accumulation_steps}",
                flush=True,
            )


class SFTTrainer(Trainer):
    """Trainer with optional rare micro-step logs (rank0 only)."""

    def __init__(self, *args, log_every_micro: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_every_micro = log_every_micro
        self._micro_in_step = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._micro_in_step += 1
        if num_items_in_batch is None:
            loss = super().training_step(model, inputs)
        else:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if (
            self._log_every_micro > 0
            and self.is_world_process_zero()
            and self._micro_in_step % self._log_every_micro == 0
        ):
            loss_val = loss.item() if hasattr(loss, "item") else float(loss)
            print(
                f"[sft] step={self.state.global_step} "
                f"micro={self._micro_in_step}/{self.args.gradient_accumulation_steps} "
                f"loss={loss_val:.4f}",
                flush=True,
            )
        if self._micro_in_step >= self.args.gradient_accumulation_steps:
            self._micro_in_step = 0
        return loss


def main():
    quiet_logs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    parser.add_argument("--model_size", type=str, required=True, choices=["0.5B", "1.5B", "3B"])
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--eval_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--deepspeed", type=str, default="configs/ds_sft.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    model_name = cfg["model_map"][args.model_size]
    n_gpu = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    micro, accum = resolve_micro_batch(
        args.model_size, cfg, n_gpu=n_gpu, global_batch=int(cfg["global_batch_size"])
    )

    # Keep Accelerate / DeepSpeed accum in sync
    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(accum)

    tokenizer, n_added = load_tokenizer(model_name)
    # save tokenizer early so cache workers can reload from disk
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tok_dir = args.output_dir / "tokenizer"
    save_tok(tokenizer, tok_dir)

    # Build/load token cache BEFORE CUDA model init (avoid fork-after-CUDA).
    use_cache = bool(cfg.get("token_cache", True))
    pack = bool(cfg.get("pack_sequences", True))
    train_ds = SFTData(
        args.train_file,
        tokenizer,
        cutoff_len=cfg["cutoff_len"],
        cache_dir=args.train_file.parent / ".tok_cache",
        tokenizer_dir=tok_dir,
        use_cache=use_cache,
        pack=pack,
    )
    eval_ds = SFTData(
        args.eval_file,
        tokenizer,
        cutoff_len=cfg["cutoff_len"],
        cache_dir=args.eval_file.parent / ".tok_cache",
        tokenizer_dir=tok_dir,
        use_cache=use_cache,
        pack=False,  # keep eval unpacked for cleaner metrics
    )

    model = load_model(model_name, tokenizer, torch_dtype=torch.float16)
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (micro * n_gpu * accum)))
    eval_steps = max(1, int(steps_per_epoch * float(cfg.get("eval_every_epochs", 0.5))))

    ds_json = None
    if os.environ.get("SKIP_DEEPSPEED", "0") != "1":
        ds_json = materialize_ds_config(args.deepspeed, micro, accum, n_gpu, args.output_dir)

    num_workers = int(cfg.get("dataloader_num_workers", 2 if use_cache else 0))
    if not use_cache:
        num_workers = 0

    is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if is_main:
        print(
            f"[sft] size={args.model_size} n_gpu={n_gpu} micro={micro} "
            f"grad_accum={accum} global_batch={micro * n_gpu * accum} "
            f"train={len(train_ds)} eval={len(eval_ds)} "
            f"steps/epoch≈{steps_per_epoch} workers={num_workers}",
            flush=True,
        )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=max(1, micro),
        gradient_accumulation_steps=accum,
        num_train_epochs=float(cfg["max_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_steps=int(cfg.get("warmup_steps", 20)),
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=int(cfg.get("logging_steps", 10)),
        logging_first_step=True,
        log_level="error",
        log_level_replica="error",
        disable_tqdm=not is_main,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        deepspeed=ds_json,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=num_workers > 0,
        report_to=["none"],
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    data_collator = partial(collate, pad_token_id=tokenizer.pad_token_id)
    log_every = int(cfg.get("log_every_micro", 0))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        log_every_micro=log_every,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=int(cfg.get("early_stop_patience", 3))))

    if is_main:
        print("[sft] train start", flush=True)
    train_result = trainer.train()
    if is_main:
        print("[sft] train done", flush=True)
    trainer.save_model(str(args.output_dir / "final"))
    save_tok(tokenizer, args.output_dir / "final")

    metrics = train_result.metrics
    metrics["n_added_tokens"] = n_added
    metrics["micro_batch"] = micro
    metrics["grad_accum"] = accum
    metrics["world_size"] = n_gpu
    with open(args.output_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(args.output_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    if is_main:
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    quiet_logs()
    main()
