from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from secondlook.utils.logging import get_logger

logger = get_logger(__name__)


def _format_bytes(n: int) -> str:
    if n < 1024**2:
        return f"{n / 1024:.1f} KiB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MiB"
    return f"{n / 1024**3:.2f} GiB"


def _nan_ratio(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.isnan(arr).sum()) / float(arr.size)


@dataclass
class SICSplit:
    inputs: np.ndarray
    targets: np.ndarray
    input_dates: np.ndarray | None
    target_dates: np.ndarray | None
    metadata: dict[str, Any]


def load_npz_split(npz_path: Path, max_samples: int | None = None) -> SICSplit:
    """
    Load a minimal SIC train/val/test NPZ split.

    Expected arrays: inputs [N,T,H,W], targets [N,T,H,W].
    Optional date keys: dates, input_dates, target_dates, target_starts.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ split not found: {npz_path}")

    size_bytes = npz_path.stat().st_size
    logger.info("Loading %s (%s)", npz_path, _format_bytes(size_bytes))

    with np.load(npz_path, allow_pickle=True) as z:
        keys = list(z.files)
        if "inputs" not in keys or "targets" not in keys:
            raise KeyError(f"{npz_path} must contain 'inputs' and 'targets'; keys={keys}")

        inputs = np.asarray(z["inputs"], dtype=np.float32)
        targets = np.asarray(z["targets"], dtype=np.float32)

        input_dates = np.asarray(z["input_dates"]) if "input_dates" in keys else None
        target_dates = np.asarray(z["target_dates"]) if "target_dates" in keys else None

        metadata: dict[str, Any] = {
            "npz_path": str(npz_path.resolve()),
            "file_size_bytes": size_bytes,
            "keys": keys,
        }

        if "dates" in keys:
            metadata["dates"] = z["dates"]
        if "target_starts" in keys:
            metadata["target_starts"] = z["target_starts"]

    if max_samples is not None:
        inputs = inputs[:max_samples]
        targets = targets[:max_samples]
        if input_dates is not None:
            input_dates = input_dates[:max_samples]
        if target_dates is not None:
            target_dates = target_dates[:max_samples]

    for name, arr in ("inputs", inputs), ("targets", targets):
        ratio = _nan_ratio(arr)
        logger.info("%s shape=%s dtype=%s nan_ratio=%.6f", name, arr.shape, arr.dtype, ratio)
        if arr.ndim != 4:
            raise ValueError(f"{name} expected ndim=4 [N,T,H,W], got shape={arr.shape}")
        n, t, h, w = arr.shape
        if t != 7:
            logger.warning("%s time dimension is %d, expected 7", name, t)
        if (h, w) != (448, 304):
            logger.warning("%s spatial shape is (%d,%d), expected (448,304)", name, h, w)

    return SICSplit(
        inputs=inputs,
        targets=targets,
        input_dates=input_dates,
        target_dates=target_dates,
        metadata=metadata,
    )
