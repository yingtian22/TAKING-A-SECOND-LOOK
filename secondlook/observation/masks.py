from __future__ import annotations

from typing import Literal

import numpy as np

from secondlook.metrics.seaice_metrics import edge_mask, miz_mask

MaskType = Literal["random", "edge", "stripe", "coarse"]


def _sample_fraction(valid: np.ndarray, obs_fraction: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(valid.shape, dtype=bool)
    idx = np.flatnonzero(valid.ravel())
    if idx.size == 0:
        return out
    k = max(1, int(round(obs_fraction * idx.size)))
    k = min(k, idx.size)
    chosen = rng.choice(idx, size=k, replace=False)
    flat = np.zeros(valid.size, dtype=bool)
    flat[chosen] = True
    return flat.reshape(valid.shape)


def _random_mask(
    valid: np.ndarray,
    obs_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if valid.ndim == 2:
        return _sample_fraction(valid, obs_fraction, rng)
    out = np.zeros(valid.shape, dtype=bool)
    for i in range(valid.shape[0]):
        out[i] = _sample_fraction(valid[i], obs_fraction, rng)
    return out


def _edge_mask_from_prior(
    valid: np.ndarray,
    prior: np.ndarray,
    obs_fraction: float,
    rng: np.random.Generator,
    *,
    edge_bias_ratio: float = 0.7,
) -> np.ndarray:
    if valid.ndim == 2:
        valid = valid[None]
        prior = prior[None]
    n, h, w = valid.shape
    out = np.zeros((n, h, w), dtype=bool)
    for i in range(n):
        vm = valid[i]
        edge = edge_mask(prior[i], ocean_mask=vm)
        miz = miz_mask(prior[i], ocean_mask=vm)
        priority = (edge | miz) & vm
        other = vm & ~priority
        n_total = max(1, int(round(obs_fraction * vm.sum())))
        n_pri = min(int(round(edge_bias_ratio * n_total)), int(priority.sum()))
        n_other = min(n_total - n_pri, int(other.sum()))
        chosen: list[int] = []
        if n_pri > 0:
            pidx = np.flatnonzero(priority.ravel())
            chosen.extend(rng.choice(pidx, size=n_pri, replace=False).tolist())
        if n_other > 0:
            oidx = np.flatnonzero(other.ravel())
            chosen.extend(rng.choice(oidx, size=n_other, replace=False).tolist())
        if not chosen and vm.any():
            idx = np.flatnonzero(vm.ravel())
            chosen = rng.choice(idx, size=min(n_total, idx.size), replace=False).tolist()
        flat = np.zeros(h * w, dtype=bool)
        flat[chosen] = True
        out[i] = flat.reshape(h, w)
    return out[0] if n == 1 else out


def _stripe_mask(
    valid: np.ndarray,
    obs_fraction: float,
    rng: np.random.Generator,
    *,
    stripe_width: int = 8,
    stripe_gap: int = 24,
) -> np.ndarray:
    if valid.ndim == 2:
        valid = valid[None]
    n, h, w = valid.shape
    xx = np.arange(w)[None, None, :]
    period = stripe_width + stripe_gap
    stripe = (xx % period) < stripe_width
    base = np.broadcast_to(stripe, (n, h, w)).copy()
    out = base & valid
    for i in range(n):
        vm = valid[i]
        target_n = max(1, int(round(obs_fraction * vm.sum())))
        cur = int(out[i].sum())
        if cur > target_n * 1.2:
            idx = np.flatnonzero(out[i].ravel())
            keep = rng.choice(idx, size=target_n, replace=False)
            thin = np.zeros(h * w, dtype=bool)
            thin[keep] = True
            out[i] = thin.reshape(h, w)
        elif cur < target_n * 0.8:
            extra = vm & ~out[i]
            need = min(target_n - cur, int(extra.sum()))
            if need > 0:
                add = np.flatnonzero(extra.ravel())
                pick = rng.choice(add, size=int(need), replace=False)
                flat = out[i].ravel()
                flat[pick] = True
                out[i] = flat.reshape(h, w)
    return out[0] if n == 1 else out


def _coarse_stride_for_fraction(obs_fraction: float, h: int, w: int) -> int:
    for stride in range(4, max(h, w)):
        grid = np.zeros((h, w), dtype=bool)
        grid[::stride, ::stride] = True
        frac = grid.sum() / (h * w)
        if frac <= obs_fraction * 1.2:
            return stride
    return 16


def _coarse_mask(valid: np.ndarray, obs_fraction: float) -> np.ndarray:
    if valid.ndim == 2:
        valid = valid[None]
    n, h, w = valid.shape
    stride = _coarse_stride_for_fraction(obs_fraction, h, w)
    grid = np.zeros((h, w), dtype=bool)
    grid[::stride, ::stride] = True
    out = np.broadcast_to(grid, (n, h, w)).copy() & valid
    return out[0] if n == 1 else out


def generate_obs_mask(
    shape: tuple[int, int] | tuple[int, int, int],
    mask_type: MaskType,
    obs_fraction: float,
    rng: np.random.Generator,
    *,
    prior: np.ndarray | None = None,
    ocean_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate toy observation mask without using target lead-day fields.

    Edge masks use *prior* (or C_t) only — never target C_{t+7}.
    """
    if len(shape) == 2:
        h, w = shape
        valid = np.ones((h, w), dtype=bool)
    else:
        _, h, w = shape
        valid = np.ones((shape[0], h, w), dtype=bool)

    if ocean_mask is not None:
        om = ocean_mask.astype(bool)
        if om.ndim == 2:
            valid = valid & om[None, :, :] if valid.ndim == 3 else valid & om
        else:
            valid = valid & om

    if mask_type == "random":
        m = _random_mask(valid, obs_fraction, rng)
    elif mask_type == "edge":
        if prior is None:
            raise ValueError("prior required for edge mask")
        m = _edge_mask_from_prior(valid, prior, obs_fraction, rng)
    elif mask_type == "stripe":
        m = _stripe_mask(valid, obs_fraction, rng)
    elif mask_type == "coarse":
        m = _coarse_mask(valid, obs_fraction)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    return m & valid
