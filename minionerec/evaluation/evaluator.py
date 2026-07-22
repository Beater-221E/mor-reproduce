"""
Official-aligned evaluation (HR/NDCG @ 3/5/10) with constrained beam search.

Source: MiniOneRec-official @ 0c64b955 evaluate.py + calc.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList

from minionerec.generation.constraints import (
    ConstrainedLogitsProcessor,
    build_sid_hash_dict,
    load_semantic_ids_from_info,
    make_prefix_fn,
)
from minionerec.data.datasets import EvalSidDataset
from minionerec.runtime.paths import project_root, resolve_path
from minionerec.data.adapters import ensure_official_paths
from minionerec.evaluation.metrics import compute_metrics, hr_at_k, ndcg_at_k


@torch.no_grad()
def evaluate_checkpoint(cfg: dict[str, Any]) -> dict[str, Any]:
    root = project_root()
    model_path = str(resolve_path(cfg["model_name_or_path"], root))
    out_dir = resolve_path(cfg.get("output_dir", "experiments/eval"), root)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = ensure_official_paths(
        {
            **cfg,
            "processed_data_root": str(resolve_path(cfg["processed_data_root"], root)),
            "official_format_root": str(resolve_path(cfg.get("official_format_root", "data/official_format"), root)),
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (torch.cuda.is_available() and "v100" in torch.cuda.get_device_name(0).lower()) else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()

    semantic_ids = load_semantic_ids_from_info(str(paths.info_txt))
    known = {s.strip("\n\" ") for s in semantic_ids}
    hash_dict = build_sid_hash_dict(tokenizer, semantic_ids, base_model=model_path)
    prefix_fn = make_prefix_fn(hash_dict)

    category = cfg.get("category_prompt") or "industrial and scientific items"
    max_samples = int(cfg.get("max_eval_samples", cfg.get("max_samples", -1)))
    ds = EvalSidDataset(
        str(paths.test_csv),
        tokenizer,
        max_len=int(cfg.get("cutoff_len", 512)),
        sample=max_samples,
        test=True,
        seed=int(cfg.get("seed", 42)),
        category=category,
    )
    meta = ds.get_all()

    num_beams = int(cfg.get("num_beams", 50))
    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    batch_size = int(cfg.get("batch_size", 4))
    gen_cfg = GenerationConfig(
        num_beams=num_beams,
        num_return_sequences=num_beams,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        length_penalty=float(cfg.get("length_penalty", 0.0)),
    )

    predictions: list[list[str]] = []
    targets: list[str] = []
    invalid = 0
    lengths = []
    t0 = time.time()

    for start in range(0, len(ds), batch_size):
        batch = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        # left pad
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attn = []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append([tokenizer.pad_token_id] * pad + x["input_ids"])
            attn.append([0] * pad + x["attention_mask"])
        input_ids_t = torch.tensor(input_ids, device=device)
        attn_t = torch.tensor(attn, device=device)

        ccc = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=prefix_fn,
            num_beams=num_beams,
            base_model=model_path,
            eos_token_id=tokenizer.eos_token_id,
        )
        ccc.reset()
        lp = LogitsProcessorList([ccc])
        gen = model.generate(
            input_ids_t,
            attention_mask=attn_t,
            generation_config=gen_cfg,
            logits_processor=lp,
        )
        prompt_len = input_ids_t.size(1)
        # reshape (B*beams, seq)
        texts = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
        B = len(batch)
        for b in range(B):
            group = texts[b * num_beams : (b + 1) * num_beams]
            cleaned = []
            for t in group:
                # official splits Response:\n if present
                t2 = t.split("Response:\n")[-1] if "Response:\n" in t else t
                cleaned.append(t2)
                if t2.strip("\n\" ") not in known:
                    invalid += 1
                lengths.append(len(tokenizer.encode(t2, add_special_tokens=False)))
            predictions.append(cleaned)
            targets.append(meta[start + b]["output"])
        if (start // batch_size) % 10 == 0:
            print(f"[eval] {start}/{len(ds)}")

    runtime = time.time() - t0
    metrics = compute_metrics(predictions, targets)
    metrics["invalid_completion_rate"] = invalid / max(1, len(predictions) * num_beams)
    metrics["average_generation_length"] = sum(lengths) / max(1, len(lengths))
    metrics["evaluation_runtime"] = runtime
    metrics["num_beams"] = num_beams
    metrics["model_name_or_path"] = model_path
    metrics["checkpoint_label"] = cfg.get("checkpoint_label", Path(model_path).name)

    results_path = out_dir / "eval_metrics.json"
    results_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    gens_path = out_dir / "generations.json"
    gens_path.write_text(
        json.dumps(
            [
                {"predict": p, "output": t}
                for p, t in zip(predictions, targets)
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[eval]", json.dumps(metrics, indent=2))
    return metrics


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    evaluate_checkpoint(cfg)


if __name__ == "__main__":
    main()
