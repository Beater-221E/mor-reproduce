"""Full-parameter multi-task SFT entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import warnings
from functools import partial
from pathlib import Path

import torch
import yaml
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.utils import logging as hf_logging

from minionerec.data.legacy_datasets import SFTData, collate
from minionerec.model import load_model, load_tokenizer, load_tokenizer_from_dir, save_tok
from minionerec.runtime.paths import prepare_save_dir, project_root, resolve_path


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


class MetricsLogCallback(TrainerCallback):
    """Print loss / lr / timing as plain lines (survives tee; tqdm+silenced HF logs hide them)."""

    def __init__(self):
        self._t_step = None
        self._last_logs: dict = {}

    def on_train_begin(self, args, state, control, **kwargs):
        self._t_step = time.time()
        if state.is_world_process_zero:
            print(
                f"[sft] logging every {args.logging_steps} steps; "
                f"at_step={state.global_step}/{state.max_steps}",
                flush=True,
            )

    def on_step_end(self, args, state, control, **kwargs):
        # All ranks must agree; force log so loss is emitted after resume too.
        control.should_log = True
        if state.is_world_process_zero:
            now = time.time()
            sec = now - self._t_step if self._t_step is not None else float("nan")
            self._t_step = now
            remain = max(0, (state.max_steps or 0) - state.global_step)
            eta_h = remain * sec / 3600.0 if sec == sec else float("nan")
            extra = ""
            if self._last_logs:
                parts = []
                for k in ("loss", "grad_norm", "learning_rate", "eval_loss"):
                    if k in self._last_logs:
                        v = self._last_logs[k]
                        parts.append(
                            f"{k}={float(v):.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
                        )
                if parts:
                    extra = " " + " ".join(parts)
            print(
                f"[sft] step={state.global_step}/{state.max_steps} "
                f"epoch={float(state.epoch or 0):.4f} {sec:.1f}s/step eta≈{eta_h:.1f}h{extra}",
                flush=True,
            )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        # Cache for the next step line; SFTTrainer.log already prints metrics.
        self._last_logs = {k: v for k, v in logs.items() if k != "epoch"}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_world_process_zero and metrics:
            bits = [f"{k}={float(v):.6g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in metrics.items()]
            print(f"[sft] eval step={state.global_step} " + " ".join(bits), flush=True)


class SFTTrainer(Trainer):
    """Trainer with forced metric prints + optional micro-step logs."""

    def __init__(self, *args, log_every_micro: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_every_micro = log_every_micro
        self._micro_in_step = 0

    def log(self, logs: dict, start_time=None) -> None:
        # HF may silence its own logger (we set log_level=error); always print rank0.
        if self.is_world_process_zero() and logs:
            bits = []
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    bits.append(f"{k}={float(v):.6g}")
                else:
                    bits.append(f"{k}={v}")
            print(
                f"[sft] metrics step={self.state.global_step} "
                f"epoch={float(self.state.epoch or 0):.4f} " + " ".join(bits),
                flush=True,
            )
        if start_time is None:
            return super().log(logs)
        return super().log(logs, start_time)

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
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help='Resume path, or "auto" for latest checkpoint under output_dir',
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    root = project_root(Path(args.config))
    model_name = str(resolve_path(cfg["model_map"][args.model_size], base=root))
    n_gpu = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    micro, accum = resolve_micro_batch(
        args.model_size, cfg, n_gpu=n_gpu, global_batch=int(cfg["global_batch_size"])
    )

    # Keep Accelerate / DeepSpeed accum in sync
    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(accum)

    tokenizer, n_added = load_tokenizer(model_name)
    # Prefer saved tokenizer when present (resume / consistent special tokens).
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tok_dir = args.output_dir / "tokenizer"
    if (tok_dir / "tokenizer_config.json").exists() or (tok_dir / "tokenizer.json").exists():
        tokenizer = load_tokenizer_from_dir(tok_dir)
        n_added = 0
    else:
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
        disable_tqdm=True,  # resume makes tqdm ETA wrong; use StepTimingCallback
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
    trainer.add_callback(MetricsLogCallback())
    early_stop_patience = int(cfg.get("early_stop_patience", 3))
    if early_stop_patience > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=early_stop_patience))
    elif is_main:
        print("[sft] early stopping disabled; will run full max_epochs", flush=True)

    resume_ckpt = None
    if args.resume:
        if args.resume.lower() in {"1", "true", "auto", "yes"}:
            resume_ckpt = True  # HF picks latest checkpoint-* under output_dir
        else:
            resume_ckpt = args.resume
        if is_main:
            print(f"[sft] resume_from_checkpoint={resume_ckpt}", flush=True)

    if is_main:
        print("[sft] train start", flush=True)
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    if is_main:
        print("[sft] train done", flush=True)

    # Clear stale symlink/dir first: a leftover final -> checkpoint-N link (or a
    # broken one after prune) makes mkdir(exist_ok=True) raise FileExistsError.
    # With load_best_model_at_end=True this writes the best in-memory weights.
    final_dir = args.output_dir / "final"
    if is_main:
        prepare_save_dir(final_dir)
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
        print(f"[sft] saving final -> {final_dir} (best_ckpt={best_ckpt})", flush=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    trainer.save_model(str(final_dir))
    save_tok(tokenizer, final_dir)
    if is_main:
        print(f"[sft] saved final model to {final_dir}", flush=True)

    metrics = train_result.metrics
    metrics["n_added_tokens"] = n_added
    metrics["micro_batch"] = micro
    metrics["grad_accum"] = accum
    metrics["world_size"] = n_gpu
    if is_main:
        with open(args.output_dir / "train_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(args.output_dir / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    quiet_logs()
    main()
