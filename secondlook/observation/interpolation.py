from __future__ import annotations

import numpy as np
from scipy import ndimage


def _as_batch(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    if arr.ndim == 2:
        return arr[None], True
    return arr, False


def nearest_fill(
    values: np.ndarray,
    mask: np.ndarray,
    ocean_mask: np.ndarray | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Nearest-neighbor propagate values from mask=True locations.

    Supports [H,W] or [N,H,W]. Unobserved pixels receive the nearest observed value.
    """
    vals, squeeze = _as_batch(np.asarray(values, dtype=np.float32))
    m, _ = _as_batch(mask.astype(bool))

    n, h, w = vals.shape
    out = np.full((n, h, w), fill_value, dtype=np.float32)
    for i in range(n):
        mi = m[i]
        if not mi.any():
            continue
        _, (iy, ix) = ndimage.distance_transform_edt(~mi, return_indices=True)
        out[i] = vals[i][iy, ix].astype(np.float32)

    if ocean_mask is not None:
        om = ocean_mask.astype(bool)
        if om.ndim == 2:
            out = np.where(om[None], out, fill_value)
        else:
            out = np.where(om, out, fill_value)

    return out[0] if squeeze else out


def distance_to_mask(mask: np.ndarray, ocean_mask: np.ndarray | None = None) -> np.ndarray:
    """Distance from each pixel to the nearest True pixel in mask."""
    m, squeeze = _as_batch(mask.astype(bool))
    n, h, w = m.shape
    out = np.zeros((n, h, w), dtype=np.float32)
    for i in range(n):
        out[i] = ndimage.distance_transform_edt(~m[i]).astype(np.float32)
    if ocean_mask is not None:
        om = ocean_mask.astype(bool)
        if om.ndim == 2:
            out = np.where(om[None], out, np.inf)
        else:
            out = np.where(om, out, np.inf)
    return out[0] if squeeze else out
