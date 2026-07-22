"""Deprecated — use ``minionerec.data.tasks``."""

from __future__ import annotations

import warnings

warnings.warn("minionerec.tasks is deprecated; use minionerec.data.tasks", DeprecationWarning, stacklevel=2)
from minionerec.data.tasks import *  # noqa: F401,F403
from minionerec.data.tasks import main  # noqa: F401

if __name__ == "__main__":
    main()
