"""Checkpoint save/load helpers (rank-0)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from minionerec.runtime.distributed import is_main_process, unwrap_model


class CheckpointManager:
    def __init__(self, output_dir: Path | str, tokenizer: Any | None = None):
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, name: str = "final_checkpoint") -> Path | None:
        if not is_main_process():
            return None
        path = self.output_dir / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        unwrap_model(model).save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
        return path
