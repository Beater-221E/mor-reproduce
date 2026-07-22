"""JSON / JSONL artifact writers (rank-0 safe)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from minionerec.runtime.distributed import is_main_process


class ArtifactWriter:
    """Atomic JSON/JSONL writes; no-ops on non-main ranks when ``rank0_only``."""

    def __init__(self, root: Path | str, *, rank0_only: bool = True):
        self.root = Path(root)
        self.rank0_only = rank0_only
        if self._active():
            self.root.mkdir(parents=True, exist_ok=True)

    def _active(self) -> bool:
        return (not self.rank0_only) or is_main_process()

    def write_json(self, relative: str | Path, obj: Any, *, indent: int = 2) -> Path | None:
        if not self._active():
            return None
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
        _atomic_write_text(path, payload)
        return path

    def append_jsonl(self, relative: str | Path, row: dict[str, Any]) -> Path | None:
        if not self._active():
            return None
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def save_json(path: Path | str, obj: Any) -> Path:
    """Convenience wrapper used by trainers."""
    path = Path(path)
    writer = ArtifactWriter(path.parent, rank0_only=False)
    writer.write_json(path.name, obj)
    return path
