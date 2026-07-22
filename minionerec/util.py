"""Deprecated compatibility shim — prefer ``constants``, ``sid.codec``, ``runtime.paths``."""

from __future__ import annotations

import warnings

warnings.warn(
    "minionerec.util is deprecated; import from minionerec.constants / sid.codec / runtime.paths",
    DeprecationWarning,
    stacklevel=2,
)

from minionerec.constants import (  # noqa: E402,F401
    CODEBOOK_SIZE,
    DATASETS,
    NUM_CODEBOOK_LAYERS,
    SID_LAYER_PREFIXES,
)
from minionerec.runtime.paths import prepare_save_dir, project_root, resolve_path  # noqa: E402,F401
from minionerec.sid.codec import all_sid_tokens, format_sid, parse_sid, sid_token  # noqa: E402,F401
