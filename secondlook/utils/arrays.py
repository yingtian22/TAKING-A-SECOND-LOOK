from __future__ import annotations

import numpy as np


def fill_nan(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace non-finite values with *fill*."""
    out = np.array(arr, dtype=np.float32, copy=True)
    out[~np.isfinite(out)] = fill
    return out


def build_valid_mask(
    inputs: np.ndarray,
    targets: np.ndarray,
    ocean_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Valid pixels for loss/metrics: finite input & target, optional ocean mask."""
    valid = np.isfinite(inputs) & np.isfinite(targets)
    if ocean_mask is not None:
        om = ocean_mask.astype(bool)
        if om.ndim == 2:
            valid = valid & om[None, None, :, :]
        elif om.shape == valid.shape:
            valid = valid & om
    return valid.astype(np.float32)
