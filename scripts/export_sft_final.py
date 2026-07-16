#!/usr/bin/env python3
"""Re-export SFT final/ using prepare_save_dir (same path as new sft.py save)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from minionerec.model import load_tokenizer_from_dir, save_tok
from minionerec.util import prepare_save_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("checkpoints/sft_Industrial_and_Scientific_1.5B"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Checkpoint to export (default: best from trainer_state, else latest)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    src = args.source
    if src is None:
        # Prefer best_model_checkpoint recorded in any remaining trainer_state.
        best = None
        for ckpt in sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
            state_path = ckpt / "trainer_state.json"
            if not state_path.exists():
                continue
            state = json.loads(state_path.read_text(encoding="utf-8"))
            cand = state.get("best_model_checkpoint")
            if cand and Path(cand).is_dir():
                best = Path(cand)
        if best is None:
            ckpts = sorted(
                (p for p in run_dir.glob("checkpoint-*") if (p / "model.safetensors").exists()),
                key=lambda p: int(p.name.split("-")[-1]),
            )
            if not ckpts:
                raise SystemExit(f"No checkpoint with model.safetensors under {run_dir}")
            best = ckpts[-1]
        src = best
    src = src.resolve()
    if not (src / "model.safetensors").exists():
        raise SystemExit(f"Missing model.safetensors in {src}")

    tok_dir = run_dir / "tokenizer"
    if not tok_dir.exists():
        raise SystemExit(f"Missing tokenizer dir: {tok_dir}")

    final_dir = prepare_save_dir(run_dir / "final")
    print(f"[export] source={src}", flush=True)
    print(f"[export] cleared and created {final_dir}", flush=True)

    print("[export] loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(src),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    print(f"[export] saving model -> {final_dir}", flush=True)
    model.save_pretrained(final_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[export] saving tokenizer from {tok_dir}", flush=True)
    tokenizer = load_tokenizer_from_dir(tok_dir)
    save_tok(tokenizer, final_dir)

    meta = {
        "source_checkpoint": str(src),
        "reason": "re-export final via prepare_save_dir after broken symlink",
    }
    (final_dir / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[export] done: {final_dir}", flush=True)
    for p in sorted(final_dir.iterdir()):
        size = p.stat().st_size if p.is_file() else 0
        print(f"  {p.name}\t{size}", flush=True)


if __name__ == "__main__":
    main()
