"""
Official-source SFT trainer for MiniOneRec alignment.

Source commit: 0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed
  sft.py + sft.sh

Keeps Amazon23/SID fixed; uses official task construction via official_data.py.
Legacy minionerec/sft.py is retained as baseline.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from minionerec.runtime.distributed import (
    barrier,
    cleanup_distributed,
    is_main_process,
    print_rank0,
    resolve_effective_batch,
    resolve_sft_accum,
    setup_distributed,
)
from minionerec.data.datasets import FusionSeqRecDataset, SidItemFeatDataset, SidSFTDataset
from minionerec.sid.tokenizer import (
    extend_tokenizer_with_sid,
    load_base_tokenizer,
    load_model_for_sft,
    tokenizer_roundtrip_test,
)
from minionerec.runtime.paths import project_root, resolve_path
from minionerec.data.adapters import ensure_official_paths
from minionerec.sid.map_io import load_sid_map
from minionerec.data.validation import assert_ready_for_training, run_data_validation, run_sid_validation


def _is_v100() -> bool:
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0).lower()
    return "v100" in name


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_task_examples(out: Path, datasets: list, tokenizer, n: int = 3) -> None:
    lines = ["# SFT task examples (decoded)\n"]
    for ds in datasets:
        name = type(ds).__name__
        lines.append(f"\n## {name}\n")
        for i in range(min(n, len(ds))):
            item = ds[i]
            ids = item["input_ids"]
            labels = item["labels"]
            prompt_ids = [t for t, y in zip(ids, labels) if y == -100]
            resp_ids = [t for t, y in zip(ids, labels) if y != -100]
            lines.append(f"### example {i}\n")
            lines.append("**prompt:**\n```\n" + tokenizer.decode(prompt_ids) + "\n```\n")
            lines.append("**response:**\n```\n" + tokenizer.decode(resp_ids) + "\n```\n")
            # leakage checks
            prompt_txt = tokenizer.decode(prompt_ids)
            resp_txt = tokenizer.decode(resp_ids).strip()
            if resp_txt and resp_txt in prompt_txt and name != "FusionSeqRecDataset":
                # sid2title has SID in prompt by design; title2sid must not leak
                if item.get("task") == "title2sid" and resp_txt.replace("\n", "") in prompt_txt:
                    lines.append("**LEAKAGE WARNING**\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def _dataset_stats(datasets: list, tokenizer) -> dict:
    stats = []
    total = sum(len(d) for d in datasets)
    for ds in datasets:
        prompt_lens = []
        resp_lens = []
        trunc = 0
        for i in range(len(ds)):
            item = ds[i]
            labels = item["labels"]
            prompt_lens.append(sum(1 for y in labels if y == -100))
            resp_lens.append(sum(1 for y in labels if y != -100))
            if len(item["input_ids"]) >= getattr(ds, "max_len", 512):
                trunc += 1
        stats.append(
            {
                "task_name": type(ds).__name__,
                "number_of_examples": len(ds),
                "fraction_of_total": len(ds) / max(1, total),
                "average_prompt_tokens": float(np.mean(prompt_lens)) if prompt_lens else 0,
                "average_response_tokens": float(np.mean(resp_lens)) if resp_lens else 0,
                "truncation_rate": trunc / max(1, len(ds)),
            }
        )
    return {"tasks": stats, "total_examples": total}


def run_sft(cfg: dict[str, Any]) -> Path:
    root = project_root()
    os.chdir(root)
    rank, local_rank, world_size, device, distributed = setup_distributed()
    set_seed(int(cfg.get("seed", 42)) + rank)
    random.seed(int(cfg.get("seed", 42)) + rank)
    np.random.seed(int(cfg.get("seed", 42)) + rank)

    impl = cfg.get("implementation_target", "official_source")
    assert impl in ("official_source", "paper_aligned")

    out_dir = resolve_path(cfg["output_dir"], root)
    artifacts = resolve_path(cfg.get("artifacts_dir", "artifacts"), root)
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)
    barrier()

    print_rank0(
        f"[sft] distributed={distributed} world_size={world_size} "
        f"local_rank={local_rank} device={device}"
    )

    # Data validation + official-format export on rank0 only
    if is_main_process():
        data_report = run_data_validation(
            resolve_path(cfg["processed_data_root"], root),
            cfg["dataset"],
            out_json=artifacts / "data_validation.json",
            allow_sid_collision=bool(cfg.get("allow_sid_collision", False)),
        )
        paths = ensure_official_paths(
            {
                **cfg,
                "processed_data_root": str(resolve_path(cfg["processed_data_root"], root)),
                "official_format_root": str(resolve_path(cfg.get("official_format_root", "data/official_format"), root)),
                "force_export_official_format": bool(cfg.get("force_export_official_format", False)),
            }
        )
        _save_json(artifacts / "_export_done.json", {"ok": True, "paths": paths.to_dict()})
    barrier()
    # Non-rank0 loads paths after export
    paths = ensure_official_paths(
        {
            **cfg,
            "processed_data_root": str(resolve_path(cfg["processed_data_root"], root)),
            "official_format_root": str(resolve_path(cfg.get("official_format_root", "data/official_format"), root)),
            "force_export_official_format": False,
        }
    )
    if is_main_process():
        data_report = json.loads((artifacts / "data_validation.json").read_text(encoding="utf-8"))
    else:
        data_report = {"hard_fail": False, "hard_fail_reasons": [], "missing_sid": 0, "train_test_leakage": 0}

    model_path = str(resolve_path(cfg["model_name_or_path"], root))
    # Prefer Instruct; if weights missing, require explicit diagnostic override.
    safetensors = Path(model_path) / "model.safetensors"
    index_file = Path(model_path) / "model.safetensors.index.json"
    if not safetensors.exists() and not index_file.exists():
        alt = cfg.get("diagnostic_model_name_or_path")
        if alt and cfg.get("model_variant") == "qwen2.5-0.5b-base-diagnostic":
            model_path = str(resolve_path(alt, root))
            print_rank0(f"[sft] WARNING: using diagnostic base model {model_path}")
        else:
            raise FileNotFoundError(
                f"Model weights not found under {model_path}. "
                f"Set model_variant=qwen2.5-0.5b-base-diagnostic and diagnostic_model_name_or_path to use Base."
            )

    tokenizer = load_base_tokenizer(model_path)
    sid_map = load_sid_map(resolve_path(cfg["processed_data_root"], root) / cfg["dataset"] / "sid" / "sid_map.json")
    tokenizer, new_tokens, n_added = extend_tokenizer_with_sid(
        tokenizer,
        sid_map=sid_map,
        add_all_codebook=bool(cfg.get("add_all_codebook_tokens", False)),
    )
    rt = tokenizer_roundtrip_test(tokenizer, [v["sid"] for v in sid_map.values()])
    if is_main_process():
        sid_report = run_sid_validation(
            resolve_path(cfg["processed_data_root"], root),
            cfg["dataset"],
            tokenizer=tokenizer,
            out_json=artifacts / "sid_validation.json",
        )
        sid_report["tokenizer_roundtrip_failure"] = rt["n_failed"]
        sid_report["roundtrip_failures"] = rt["failures"]
        _save_json(artifacts / "sid_validation.json", sid_report)
    barrier()
    sid_report = json.loads((artifacts / "sid_validation.json").read_text(encoding="utf-8"))
    assert_ready_for_training(data_report, sid_report)

    v100 = _is_v100()
    prec = cfg.get("precision", {}) or {}
    preferred = prec.get("preferred", "bf16")
    v100_fallback = prec.get("fallback_for_v100", "fp32")
    if v100:
        use_fp16 = v100_fallback == "fp16"
        use_bf16 = False
        use_fp32 = v100_fallback in ("fp32", "float32") or not use_fp16
        if use_fp32:
            use_fp16 = False
            print_rank0("[sft] V100 detected: using fp32 compute (SID embedding stability)")
    else:
        use_fp16 = preferred == "fp16"
        use_bf16 = preferred == "bf16"
        use_fp32 = preferred == "fp32"

    print_rank0(f"[sft] loading model from {model_path} (may take 1–3 min, little log until done)…")
    model, tok_info = load_model_for_sft(
        model_path,
        tokenizer,
        freeze_llm=bool(cfg.get("freeze_llm", False)),
        v100_fp16=use_fp16,
    )
    tok_info["number_of_sid_tokens_added"] = n_added
    tok_info["n_sid_token_strings"] = len(new_tokens)
    if is_main_process():
        _save_json(out_dir / "tokenizer_info.json", tok_info)
    print_rank0("[sft] tokenizer/model: " + json.dumps(tok_info))

    cutoff = int(cfg.get("cutoff_len", 512))
    sample = int(cfg.get("max_train_samples", -1))
    category = cfg.get("category_prompt") or (
        "industrial and scientific items" if cfg["dataset"] == "Industrial_and_Scientific" else cfg["dataset"]
    )
    seed = int(cfg.get("seed", 42))

    train_datasets = []
    enabled = cfg.get("sft_tasks") or ["sid_sft", "sid_item_feat", "fusion_seq"]
    if "sid_sft" in enabled:
        d1 = SidSFTDataset(str(paths.train_csv), tokenizer, max_len=cutoff, sample=sample, seed=seed, category=category)
        train_datasets.append(d1)
    if "sid_item_feat" in enabled:
        d2 = SidItemFeatDataset(
            str(paths.item_json), str(paths.index_json), tokenizer, max_len=cutoff, sample=sample, seed=seed, category=category
        )
        train_datasets.append(d2)
    if "fusion_seq" in enabled:
        d3 = FusionSeqRecDataset(
            str(paths.train_csv),
            str(paths.item_json),
            str(paths.index_json),
            tokenizer,
            max_len=cutoff,
            sample=sample,
            seed=seed,
            category=category,
        )
        train_datasets.append(d3)
    if not train_datasets:
        raise ValueError(f"No SFT tasks enabled: {enabled}")
    train_data = ConcatDataset(train_datasets)
    val_sample = int(cfg.get("max_eval_samples", -1))
    val_data = SidSFTDataset(str(paths.valid_csv), tokenizer, max_len=cutoff, sample=val_sample, seed=seed, category=category)

    if is_main_process():
        _write_task_examples(artifacts / "sft_task_examples.md", train_datasets, tokenizer)
        stats = _dataset_stats(train_datasets, tokenizer)
        _save_json(artifacts / "sft_dataset_statistics.json", stats)
        print_rank0("[sft] dataset stats: " + json.dumps(stats, indent=2))
    barrier()

    # optional fraction
    train_frac = float(cfg.get("train_fraction", 1.0))
    eval_frac = float(cfg.get("eval_fraction", 1.0))

    hf_train = HFDataset.from_dict({k: [v[k] for v in train_data] for k in ["input_ids", "attention_mask", "labels"]})
    hf_train = hf_train.shuffle(seed=42)
    if train_frac < 1.0:
        hf_train = hf_train.select(range(max(1, int(train_frac * len(hf_train)))))
    hf_val = HFDataset.from_dict({k: [v[k] for v in val_data] for k in ["input_ids", "attention_mask", "labels"]})
    hf_val = hf_val.shuffle(seed=42)
    if eval_frac < 1.0:
        hf_val = hf_val.select(range(max(1, int(eval_frac * len(hf_val)))))

    # Official: accum = global_batch // (micro * world_size); B_eff = micro * world_size * accum
    global_batch = int(cfg.get("batch_size", 1024))
    micro = int(cfg.get("micro_batch_size", 8 if world_size >= 1 else 4))
    accum = resolve_sft_accum(global_batch, micro, world_size)
    effective = resolve_effective_batch(micro, accum, world_size)
    print_rank0(
        f"[sft] effective_batch={effective} (= micro {micro} x world_size {world_size} x accum {accum}); "
        f"target_global_batch={global_batch} distributed={distributed}"
    )

    early_cfg = cfg.get("early_stopping_patience", {})
    if isinstance(early_cfg, dict):
        patience = int(early_cfg.get(impl, early_cfg.get("official_source", 3)))
    else:
        patience = int(early_cfg)

    callbacks = []
    if cfg.get("early_stopping", True) and patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=micro,
        gradient_accumulation_steps=accum,
        warmup_steps=int(cfg.get("warmup_steps", 20)),
        num_train_epochs=float(cfg.get("num_train_epochs", 10)),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=int(cfg.get("logging_steps", 1)),
        optim=cfg.get("optimizer", "adamw_torch"),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        eval_strategy="steps",
        eval_steps=float(cfg.get("eval_steps", 0.05)),
        save_strategy="steps",
        save_steps=float(cfg.get("eval_steps", 0.05)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(cfg.get("greater_is_better", False)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
        report_to=[],
        seed=int(cfg.get("seed", 42)),
        data_seed=int(cfg.get("data_seed", cfg.get("seed", 42))),
        remove_unused_columns=False,
        max_steps=int(cfg["max_steps"]) if cfg.get("max_steps") is not None else -1,
        ddp_find_unused_parameters=False,
        ddp_timeout=int(cfg.get("ddp_timeout", 1800)),
    )

    trainer = Trainer(
        model=model,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        args=args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callbacks,
    )
    model.config.use_cache = False

    # save resolved config + env (rank0)
    resolved = {
        **cfg,
        "effective_batch_size": effective,
        "micro_batch_size_resolved": micro,
        "gradient_accumulation_steps": accum,
        "world_size": world_size,
        "distributed": distributed,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "early_stopping_patience_resolved": patience,
        "implementation_target": impl,
        "official_paths": paths.to_dict(),
        "tokenizer_info": tok_info,
        "model_path_resolved": model_path,
        "conda_python": os.environ.get("PYTHON", ""),
    }
    if is_main_process():
        _save_json(out_dir / "config_resolved.json", resolved)
        (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        if (artifacts / "environment.json").exists():
            shutil.copy(artifacts / "environment.json", out_dir / "environment.json")
        tok_dir = out_dir / "tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(tok_dir)
    barrier()

    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))

    best_dir = out_dir / "best_checkpoint"
    final_dir = out_dir / "final_checkpoint"
    if is_main_process():
        for d in (best_dir, final_dir):
            if d.exists():
                shutil.rmtree(d)
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(best_dir)
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(final_dir)

        metrics = trainer.state.log_history
        _save_json(out_dir / "train_metrics.json", metrics)
        with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as f:
            for row in metrics:
                f.write(json.dumps(row) + "\n")

        best_loss = None
        best_step = None
        final_eval = None
        for row in metrics:
            if "eval_loss" in row:
                final_eval = row["eval_loss"]
                if best_loss is None or row["eval_loss"] < best_loss:
                    best_loss = row["eval_loss"]
                    best_step = row.get("step")
        summary = {
            "best_eval_loss": best_loss,
            "best_step": best_step,
            "final_eval_loss": final_eval,
            "best_checkpoint": str(best_dir),
            "final_checkpoint": str(final_dir),
            "effective_batch_size": effective,
            "world_size": world_size,
        }
        _save_json(out_dir / "sft_summary.json", summary)
        print_rank0("[sft] done: " + json.dumps(summary, indent=2))
    barrier()
    cleanup_distributed()
    return best_dir


@dataclass
class TrainingResult:
    best_checkpoint: Path
    final_checkpoint: Path | None = None
    metrics: dict[str, Any] | None = None


def train_sft(config: Any) -> TrainingResult:
    """Public SFT entry. Accepts typed ``SFTConfig`` or legacy flat dict."""
    from minionerec.config import SFTConfig, dump_resolved
    from minionerec.runtime.paths import project_root, resolve_path

    if isinstance(config, SFTConfig):
        cfg = config.to_legacy_dict()
        out = resolve_path(config.runtime.output_dir, project_root())
        out.mkdir(parents=True, exist_ok=True)
        dump_resolved(config, out / "config.resolved.yaml")
    else:
        cfg = config
    best = run_sft(cfg)
    final = Path(cfg["output_dir"]) / "final_checkpoint"
    summary_path = Path(cfg["output_dir"]) / "sft_summary.json"
    metrics = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return TrainingResult(best_checkpoint=Path(best), final_checkpoint=final if final.exists() else None, metrics=metrics)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_sft(cfg)


if __name__ == "__main__":
    main()
