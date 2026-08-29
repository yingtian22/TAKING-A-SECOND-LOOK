from __future__ import annotations

import numpy as np

ICE_THRESHOLD = 0.15
MIZ_LOW = 0.15
MIZ_HIGH = 0.80


def finite_valid_mask(
    pred: np.ndarray,
    target: np.ndarray,
    extra_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask of pixels that are finite in pred, target, and optional extra mask."""
    valid = np.isfinite(pred) & np.isfinite(target)
    if extra_mask is not None:
        valid = valid & extra_mask.astype(bool)
    return valid


def _broadcast_ocean_mask(ocean_mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if ocean_mask is None:
        return None
    om = ocean_mask.astype(bool)
    if om.shape == shape[-2:]:
        prefix = (1,) * (len(shape) - 2)
        return np.broadcast_to(om.reshape(*prefix, *om.shape), shape)
    if om.shape == shape:
        return om
    return None


def _apply_mask(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return values[np.isfinite(values)]
    m = mask.astype(bool)
    return values[m & np.isfinite(values)]


def rmse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Root mean squared error over valid pixels."""
    diff = pred - target
    vals = _apply_mask(diff, mask)
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(vals**2)))


def mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean absolute error over valid pixels."""
    diff = np.abs(pred - target)
    vals = _apply_mask(diff, mask)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def bias(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean signed error (pred - target) over valid pixels."""
    diff = pred - target
    vals = _apply_mask(diff, mask)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def edge_mask(field: np.ndarray, ocean_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Edge mask from gradient magnitude on *field*.

    Threshold = 85th percentile of gradient magnitude within valid ocean/finite region.
    Returns bool mask with same spatial shape as the last two dims of *field*.
    """
    if field.ndim == 2:
        sample = field
    elif field.ndim == 3:
        sample = np.nanmean(field, axis=0)
    else:
        raise ValueError(f"edge_mask expects [H,W] or [N,H,W], got {field.shape}")

    gy, gx = np.gradient(np.nan_to_num(sample, nan=0.0))
    grad_mag = np.sqrt(gx**2 + gy**2)

    valid = np.isfinite(sample)
    om = ocean_mask.astype(bool) if ocean_mask is not None and ocean_mask.ndim == 2 else None
    if om is not None:
        valid = valid & om

    if not np.any(valid):
        return np.zeros(sample.shape, dtype=bool)

    threshold = float(np.percentile(grad_mag[valid], 85))
    edge = grad_mag >= threshold
    if om is not None:
        edge = edge & om
    return edge & valid


def miz_mask(field: np.ndarray, ocean_mask: np.ndarray | None = None) -> np.ndarray:
    """Marginal ice zone: 0.15 <= SIC <= 0.80, optionally restricted to ocean."""
    m = (field >= MIZ_LOW) & (field <= MIZ_HIGH) & np.isfinite(field)
    om = _broadcast_ocean_mask(ocean_mask, field.shape)
    if om is not None:
        m = m & om
    return m


def sie_error(
    pred: np.ndarray,
    target: np.ndarray,
    ocean_mask: np.ndarray | None = None,
) -> float:
    """
    Normalized sea-ice extent error.

    Extent approximated by count(SIC >= 0.15). Returns mean over samples of
    abs(extent_pred - extent_target) / n_valid_ocean_pixels.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

    if pred.ndim == 2:
        pred = pred[None]
        target = target[None]

    errors: list[float] = []
    for i in range(pred.shape[0]):
        p = pred[i]
        t = target[i]
        valid = np.isfinite(p) & np.isfinite(t)
        om = _broadcast_ocean_mask(ocean_mask, p.shape)
        if om is not None:
            valid = valid & om
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        extent_p = int(np.sum((p >= ICE_THRESHOLD) & valid))
        extent_t = int(np.sum((t >= ICE_THRESHOLD) & valid))
        errors.append(abs(extent_p - extent_t) / n_valid)

    if not errors:
        return float("nan")
    return float(np.mean(errors))


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    ocean_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute standard sea-ice metrics for [N,H,W] or [H,W] arrays.

    All metrics use finite valid pixels; ocean_mask further restricts evaluation.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    if pred.ndim == 2:
        pred = pred[None]
        target = target[None]

    om2d = ocean_mask
    if om2d is not None and om2d.ndim > 2:
        om2d = om2d[0]

    valid = finite_valid_mask(pred, target, om2d)
    edge2d = edge_mask(target, ocean_mask=om2d)
    edge = valid & edge2d[np.newaxis, :, :]
    miz = miz_mask(target, ocean_mask=om2d) & valid

    return {
        "RMSE": rmse(pred, target, valid),
        "MAE": mae(pred, target, valid),
        "Bias": bias(pred, target, valid),
        "Edge_RMSE": rmse(pred, target, edge),
        "MIZ_RMSE": rmse(pred, target, miz),
        "SIE_error": sie_error(pred, target, ocean_mask=om2d),
    }
