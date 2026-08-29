"""Stage 8B: state-dependent adaptive propagation (adaptive correlation-length
correction).

Only the scalar length L0 of the frozen `local_oi` scheme is generalized to a
learned per-pixel field L_theta(x). The nearest-observation association, the
innovation, the exponential kernel family, ocean handling and clipping are all
identical to `local_oi`.

    L_theta(x) = L0 * exp(beta * tanh(s_theta(x))),  beta = ln(2)
    gain(x)    = exp(-d(x) / L_theta(x))
    C(x)       = clip01( C_prior_target(x) + gain(x) * e(nn(x)) )

The numpy side (EDT distance, nearest-neighbor innovation) is computed once
per sample and is constant w.r.t. network parameters; the torch side is
elementwise and fully differentiable w.r.t. the raw scale field.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage

from secondlook.observation.interpolation import distance_to_mask, nearest_fill

BETA = math.log(2.0)
CONFIDENCE_LENGTH_SCALE = 10.0  # same definition as V1's confidence channel

FEATURE_NAMES = [
    "prior_target",
    "prior_obs_day",
    "sparse_obs",
    "mask",
    "fixed_local_correction",
    "fixed_base_delta",
    "confidence",
    "gradient_magnitude",
]


def gradient_magnitude(prior_target: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of prior target-day SIC (no target access)."""
    gx = ndimage.sobel(prior_target, axis=1, mode="nearest") / 8.0
    gy = ndimage.sobel(prior_target, axis=0, mode="nearest") / 8.0
    return np.sqrt(gx**2 + gy**2).astype(np.float32)


def build_scale_features(
    *,
    prior_target: np.ndarray,
    prior_obs_day: np.ndarray,
    sparse_obs: np.ndarray,
    obs_mask: np.ndarray,
    fixed_local_correction: np.ndarray,
    dist: np.ndarray,
    ocean_mask: np.ndarray | None = None,
    confidence_length_scale: float = CONFIDENCE_LENGTH_SCALE,
) -> np.ndarray:
    """Stack the 8 Stage-8B input channels. All inputs [H,W] float32.

    Only prediction-time-available fields are used; the target day SIC never
    enters feature construction.
    """
    sparse_canvas = np.where(obs_mask, sparse_obs, 0.0).astype(np.float32)
    base_delta = (fixed_local_correction - prior_target).astype(np.float32)
    scale = max(float(confidence_length_scale), 1e-6)
    confidence = np.exp(-dist / scale).astype(np.float32)
    grad_mag = gradient_magnitude(prior_target)
    if ocean_mask is not None:
        grad_mag = np.where(ocean_mask.astype(bool), grad_mag, 0.0).astype(np.float32)
    feats = np.stack(
        [
            prior_target,
            prior_obs_day,
            sparse_canvas,
            obs_mask.astype(np.float32),
            fixed_local_correction,
            base_delta,
            confidence,
            grad_mag,
        ],
        axis=0,
    )
    return np.nan_to_num(feats, nan=0.0).astype(np.float32)


def nearest_innovation_fields(
    *,
    prior_obs_day: np.ndarray,
    sparse_obs: np.ndarray,
    obs_mask: np.ndarray,
    ocean_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dist, innovation_nn), identical to the frozen local_oi path.

    dist: EDT distance to nearest observed pixel (inf on land if ocean_mask).
    innovation_nn: nearest observed pixel's innovation (nearest_fill).
    """
    mi = obs_mask.astype(bool)
    residual_obs = np.zeros_like(prior_obs_day, dtype=np.float32)
    residual_obs[mi] = sparse_obs[mi] - prior_obs_day[mi]
    innovation_nn = nearest_fill(residual_obs, mi, ocean_mask=ocean_mask)
    dist = distance_to_mask(mi, ocean_mask=ocean_mask)
    return dist.astype(np.float32), innovation_nn.astype(np.float32)


def bounded_length_field(
    raw_scale: torch.Tensor,
    base_length: float,
) -> torch.Tensor:
    """L = L0 * exp(beta * tanh(raw_scale)) in [0.5*L0, 2*L0]."""
    return float(base_length) * torch.exp(BETA * torch.tanh(raw_scale))


def adaptive_nearest_innovation_propagation(
    *,
    prior_target: torch.Tensor,
    dist: torch.Tensor,
    innovation_nn: torch.Tensor,
    length_field: torch.Tensor,
    ocean_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable correction. All tensors [N,1,H,W] (or broadcastable).

    Returns (corrected, gain). Land pixels (ocean_mask == False) get gain 0.
    dist may contain inf on land. The inf is replaced by a large finite value
    BEFORE the division: exp(-inf/L) = 0 is fine in forward, but the backward
    grad of exp(-d/L) w.r.t. L is gain * d / L^2, and 0 * inf = NaN. With
    d = 1e4 the gain underflows to exactly 0 and the gradient is 0 * finite.
    """
    dist_safe = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, 1e4))
    gain = torch.exp(-dist_safe / length_field)
    if ocean_mask is not None:
        gain = gain * ocean_mask
    corrected = torch.clamp(prior_target + gain * innovation_nn, 0.0, 1.0)
    return corrected, gain
