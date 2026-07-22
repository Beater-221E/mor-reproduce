"""Filesystem helpers and path resolution (no ML deps)."""

from __future__ import annotations

import shutil
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Resolve repo root from a path under the project (or cwd)."""
    p = (start or Path.cwd()).resolve()
    if p.is_file():
        p = p.parent
    for cand in (p, *p.parents):
        if (cand / "minionerec").is_dir() and (cand / "configs").is_dir():
            return cand
    return Path.cwd().resolve()


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve relative paths against project root; leave absolute paths unchanged."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (base or project_root()) / p


def prepare_save_dir(path: Path) -> Path:
    """Make ``path`` a fresh directory, removing any prior file/symlink/dir."""
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
