from __future__ import annotations

import numpy as np

from secondlook.observation.interpolation import distance_to_mask, nearest_fill


def _as_batch(*arrays: np.ndarray) -> tuple[list[np.ndarray], bool]:
    squeeze = arrays[0].ndim == 2
    out = []
    for arr in arrays:
        out.append(arr[None] if arr.ndim == 2 else arr)
    return out, squeeze


def _squeeze(arr: np.ndarray, squeeze: bool) -> np.ndarray:
    return arr[0] if squeeze else arr


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def original_prior(prior_target: np.ndarray) -> np.ndarray:
    return _clip01(np.asarray(prior_target, dtype=np.float32))


def sparse_replace(
    prior_target: np.ndarray,
    sparse_obs: np.ndarray,
    obs_mask: np.ndarray,
) -> np.ndarray:
    arrs, squeeze = _as_batch(prior_target, sparse_obs, obs_mask)
    prior, sparse, mask = arrs
    out = prior.copy()
    out[mask] = sparse[mask]
    return _squeeze(_clip01(out), squeeze)


def sparse_interp(
    sparse_obs: np.ndarray,
    obs_mask: np.ndarray,
    ocean_mask: np.ndarray | None = None,
) -> np.ndarray:
    arrs, squeeze = _as_batch(sparse_obs, obs_mask)
    sparse, mask = arrs
    filled = nearest_fill(sparse, mask, ocean_mask=ocean_mask)
    return _squeeze(_clip01(filled), squeeze)


def _residual_field(
    prior_obs_day: np.ndarray,
    sparse_obs: np.ndarray,
    obs_mask: np.ndarray,
    ocean_mask: np.ndarray | None,
) -> np.ndarray:
    arrs, squeeze = _as_batch(prior_obs_day, sparse_obs, obs_mask)
    p_obs, sparse, mask = arrs
    n = p_obs.shape[0]
    out = np.zeros_like(p_obs, dtype=np.float32)
    for i in range(n):
        mi = mask[i].astype(bool)
        residual_obs = np.zeros_like(p_obs[i], dtype=np.float32)
        residual_obs[mi] = sparse[i][mi] - p_obs[i][mi]
        out[i] = nearest_fill(residual_obs, mi, ocean_mask=ocean_mask)
    return _squeeze(out, squeeze)


def residual_interp(
    prior_target: np.ndarray,
    prior_obs_day: np.ndarray,
    sparse_obs: np.ndarray,
    obs_mask: np.ndarray,
    ocean_mask: np.ndarray | None = None,
) -> np.ndarray:
    residual_full = _residual_field(prior_obs_day, sparse_obs, obs_mask, ocean_mask)
    arrs, squeeze = _as_batch(prior_target)
    p_tgt = arrs[0]
    if squeeze:
        return _clip01(p_tgt + residual_full)
    if residual_full.ndim == 2:
        residual_full = residual_full[None]
    return _clip01(p_tgt + residual_full)


def nudging(
    prior_target: np.ndarray,
    prior_obs_day: np.ndarray,
    sparse_obs: np.ndarray,
    obs_mask: np.ndarray,
    alpha: float,
    ocean_mask: np.ndarray | None = None,
) -> np.ndarray:
    residual_full = _residual_field(prior_obs_day, sparse_obs, obs_mask, ocean_mask)
    arrs, squeeze = _as_batch(prior_target)
    p_tgt = arrs[0]
    if squeeze:
        return _clip01(p_tgt + alpha * residual_full)
    if residual_full.ndim == 2:
        residual_full = residual_full[None]
    return _clip01(p_tgt + alpha * residual_full)


def local_oi(
    prior_target: np.ndarray,
    prior_obs_day: np.ndarray,
    sparse_obs: np.ndarray,
    obs_mask: np.ndarray,
    length_scale: float,
    ocean_mask: np.ndarray | None = None,
) -> np.ndarray:
    arrs, squeeze = _as_batch(prior_target, prior_obs_day, sparse_obs, obs_mask)
    p_tgt, p_obs, sparse, mask = arrs
    n = p_tgt.shape[0]
    out = np.zeros_like(p_tgt, dtype=np.float32)
    ls = max(float(length_scale), 1e-6)
    for i in range(n):
        mi = mask[i].astype(bool)
        residual_obs = np.zeros_like(p_tgt[i], dtype=np.float32)
        residual_obs[mi] = sparse[i][mi] - p_obs[i][mi]
        residual_full = nearest_fill(residual_obs, mi, ocean_mask=ocean_mask)
        dist = distance_to_mask(mi, ocean_mask=ocean_mask)
        gain = np.exp(-dist / ls).astype(np.float32)
        out[i] = _clip01(p_tgt[i] + gain * residual_full)
    return _squeeze(out, squeeze)
