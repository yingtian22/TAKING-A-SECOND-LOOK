from __future__ import annotations

import numpy as np


def apply_observation_noise(
    obs_full: np.ndarray,
    obs_mask: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build sparse observation field with optional Gaussian noise on masked pixels.

    Values outside obs_mask are zero; baselines must only use masked locations.
    """
    obs = np.asarray(obs_full, dtype=np.float32)
    mask = obs_mask.astype(bool)
    squeeze = False
    if obs.ndim == 2:
        obs = obs[None]
        mask = mask[None]
        squeeze = True

    out = np.zeros_like(obs, dtype=np.float32)
    for i in range(obs.shape[0]):
        o = obs[i].copy()
        m = mask[i]
        if noise_std > 0 and m.any():
            o[m] = o[m] + rng.normal(0.0, noise_std, size=int(m.sum())).astype(np.float32)
        o = np.clip(o, 0.0, 1.0)
        out[i] = np.where(m, o, 0.0)

    return out[0] if squeeze else out
