"""Environment snapshot for reproducibility artifacts."""

from __future__ import annotations

import platform
import sys
from typing import Any


def collect_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        info["pytorch"] = torch.__version__
        info["cuda_compiled"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:  # pragma: no cover
        info["pytorch_error"] = repr(e)
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception as e:  # pragma: no cover
        info["transformers_error"] = repr(e)
    return info
