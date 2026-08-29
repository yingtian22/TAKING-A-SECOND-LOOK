from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def collect_environment_info() -> dict[str, Any]:
    """Collect Python, platform, and key library versions for audit reports."""
    info: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "current_working_directory": str(Path.cwd()),
        "conda_env_name": os.environ.get("CONDA_DEFAULT_ENV"),
    }

    try:
        import numpy as np

        info["numpy_version"] = np.__version__
    except ImportError:
        info["numpy_version"] = None

    try:
        import pandas as pd

        info["pandas_version"] = pd.__version__
    except ImportError:
        info["pandas_version"] = None

    try:
        import scipy

        info["scipy_version"] = scipy.__version__
    except ImportError:
        info["scipy_version"] = None

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_cuda_is_available"] = bool(torch.cuda.is_available())
        info["gpu_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            info["gpu_name"] = None
    except ImportError:
        info["torch_version"] = None
        info["torch_cuda_version"] = None
        info["torch_cuda_is_available"] = False
        info["gpu_count"] = 0
        info["gpu_name"] = None

    return info
