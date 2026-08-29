"""Stage 8A: state-dependent anisotropic generalization of the project's
`local_oi` correction (nearest-neighbor innovation propagation with
exponential distance decay).

Minimal equivalent generalization: the isotropic distance-to-nearest-
observation is replaced by an anisotropic (Mahalanobis-type) distance to the
SAME nearest observation, so that r = 1 degenerates exactly to `local_oi`.

Orientation is derived analytically from the prior target-day SIC gradient
(ice-edge tangent). No target SIC is ever used. No learning happens here.

Math (per pixel x, nearest observed pixel nn(x), displacement s = x - nn(x)):

    Sigma(x) = R(phi) diag(l_par^2, l_perp^2) R(phi)^T
    l_par = L * sqrt(r_eff),  l_perp = L / sqrt(r_eff)   (area-preserving)
    r_eff(x) = 1 + a(x) * (r - 1)                        (flat-region gating)
    d_A^2 = s^T Sigma(x)^-1 s = u^2/(L^2 r_eff) + v^2 r_eff / L^2
    gain  = exp(-d_A)            (same exponential kernel as local_oi)
    out   = clip01(prior_target + gain * innovation(nn(x)))

with (u, v) the displacement components in the (tangent, normal) frame and
a(x) = |grad C| / (|grad C| + G0) a smooth anisotropy-strength gating.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from secondlook.observation.interpolation import nearest_fill

# Fixed gating constant (SIC per pixel). NOT tuned on any data: chosen once as
# a small fraction of the SIC range so that a(x) ~ 0.5 at |grad C| = 0.05.
DEFAULT_G0 = 0.05

_EPS = 1e-12


def _as_batch(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        return a[None], True
    return a, False


def compute_sic_gradient(
    prior_target: np.ndarray,
    ocean_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradients of the prior target-day SIC. Supports [H,W] or [N,H,W].

    Returns (gx, gy) in SIC / pixel. Land pixels (ocean_mask == False) are
    zeroed so that the orientation field carries no signal there.
    """
    p, squeeze = _as_batch(prior_target)
    gx = np.zeros_like(p)
    gy = np.zeros_like(p)
    for i in range(p.shape[0]):
        gx[i] = ndimage.sobel(p[i], axis=1, mode="nearest") / 8.0
        gy[i] = ndimage.sobel(p[i], axis=0, mode="nearest") / 8.0
    if ocean_mask is not None:
        om = ocean_mask.astype(bool)
        if om.ndim == 2:
            om = om[None]
        gx = np.where(om, gx, 0.0)
        gy = np.where(om, gy, 0.0)
    if squeeze:
        return gx[0], gy[0]
    return gx, gy


def compute_orientation_field(
    gx: np.ndarray,
    gy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tangent orientation of the ice edge from the SIC gradient.

    The gradient is normal to the edge: n = g / |g|; tangent t = (-n_y, n_x).
    Returns (cos_t, sin_t, grad_mag) where (cos_t, sin_t) is the unit tangent.
    Where |g| ~ 0 the tangent is set to (1, 0); it is multiplied by a(x) = 0
    downstream, so the exact value is irrelevant there.
    """
    grad_mag = np.sqrt(gx**2 + gy**2)
    # max() instead of "+ eps": with "+ eps" the stored tangent is deflated by
    # ~(2*eps/|g|) when |g| is tiny, breaking unit norm and hence the exact
    # r = 1 isotropic degeneration. max() keeps the norm exactly 1 whenever
    # |g| > eps, and the flat fallback sets a canonical tangent below eps.
    inv = 1.0 / np.maximum(grad_mag, _EPS)
    n_x, n_y = gx * inv, gy * inv
    cos_t, sin_t = -n_y, n_x
    zero = grad_mag < _EPS
    cos_t = np.where(zero, 1.0, cos_t).astype(np.float32)
    sin_t = np.where(zero, 0.0, sin_t).astype(np.float32)
    return cos_t.astype(np.float32), sin_t, grad_mag.astype(np.float32)


def compute_anisotropy_strength(
    grad_mag: np.ndarray,
    g0: float = DEFAULT_G0,
) -> np.ndarray:
    """Smooth gating a(x) in [0, 1]: a = |g| / (|g| + g0).

    Flat regions (|g| -> 0) give a -> 0, i.e. r_eff -> 1 (isotropic fallback).
    Strong gradients give a -> 1, i.e. r_eff -> r.
    """
    g0 = max(float(g0), _EPS)
    a = grad_mag / (grad_mag + g0)
    return a.astype(np.float32)


def build_anisotropic_metric(
    prior_target: np.ndarray,
    length_scale: float,
    anisotropy_ratio: float,
    ocean_mask: np.ndarray | None = None,
    g0: float = DEFAULT_G0,
) -> dict[str, np.ndarray]:
    """Precompute the per-pixel anisotropic metric fields.

    Returns dict with cos_t, sin_t (unit tangent), r_eff (effective ratio),
    grad_mag, strength a(x), and scalar L. Array shapes match prior_target
    ([H,W] or [N,H,W]).
    """
    gx, gy = compute_sic_gradient(prior_target, ocean_mask=ocean_mask)
    cos_t, sin_t, grad_mag = compute_orientation_field(gx, gy)
    a = compute_anisotropy_strength(grad_mag, g0=g0)
    r_eff = 1.0 + a * (float(anisotropy_ratio) - 1.0)
    return {
        "cos_t": cos_t,
        "sin_t": sin_t,
        "grad_mag": grad_mag,
        "strength": a,
        "r_eff": r_eff.astype(np.float32),
        "length_scale": np.float32(max(float(length_scale), 1e-6)),
    }


def anisotropic_oi_correction(
    prior_target: np.ndarray,
    prior_obs_day: np.ndarray,
    sparse_obs: np.ndarray,
    obs_mask: np.ndarray,
    length_scale: float,
    anisotropy_ratio: float = 1.0,
    ocean_mask: np.ndarray | None = None,
    g0: float = DEFAULT_G0,
) -> np.ndarray:
    """Anisotropic generalization of `local_oi`. Supports [H,W] or [N,H,W].

    anisotropy_ratio = 1.0 degenerates EXACTLY to `local_oi` (up to float32
    rounding): r_eff = 1 everywhere makes d_A equal to the isotropic distance.
    """
    p_tgt, squeeze = _as_batch(prior_target)
    p_obs, _ = _as_batch(prior_obs_day)
    sparse, _ = _as_batch(sparse_obs)
    mask, _ = _as_batch(obs_mask)
    n, h, w = p_tgt.shape

    metric = build_anisotropic_metric(
        p_tgt, length_scale, anisotropy_ratio, ocean_mask=ocean_mask, g0=g0
    )
    cos_t = metric["cos_t"]
    sin_t = metric["sin_t"]
    r_eff = metric["r_eff"]
    ls = float(metric["length_scale"])
    if cos_t.ndim == 2:  # single 2D metric shared across batch
        cos_t = cos_t[None]
        sin_t = sin_t[None]
        r_eff = r_eff[None]

    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    out = np.zeros_like(p_tgt)
    for i in range(n):
        mi = mask[i].astype(bool)
        if not mi.any():
            out[i] = np.clip(p_tgt[i], 0.0, 1.0)
            continue
        residual_obs = np.zeros_like(p_tgt[i])
        residual_obs[mi] = sparse[i][mi] - p_obs[i][mi]
        residual_full = nearest_fill(residual_obs, mi, ocean_mask=ocean_mask)

        _, (iy, ix) = ndimage.distance_transform_edt(~mi, return_indices=True)
        dx = xx - ix.astype(np.float32)
        dy = yy - iy.astype(np.float32)

        ct = cos_t[i] if cos_t.shape[0] > 1 else cos_t[0]
        st = sin_t[i] if sin_t.shape[0] > 1 else sin_t[0]
        r_i = r_eff[i] if r_eff.shape[0] > 1 else r_eff[0]

        # float64 for the distance computation: keeps the r = 1 degeneration
        # to local_oi exact to ~1e-7 even for large displacements (float32
        # unit tangents satisfy ct^2 + st^2 = 1 only to ~1e-7, which otherwise
        # accumulates to ~1e-5 in the gain at distance ~30 px).
        u = dx.astype(np.float64) * ct.astype(np.float64) \
            + dy.astype(np.float64) * st.astype(np.float64)
        v = dy.astype(np.float64) * ct.astype(np.float64) \
            - dx.astype(np.float64) * st.astype(np.float64)
        r64 = r_i.astype(np.float64)
        d_a = np.sqrt(u * u / r64 + v * v * r64) / ls
        gain = np.exp(-d_a).astype(np.float32)

        if ocean_mask is not None:
            om = ocean_mask.astype(bool)
            if om.ndim == 3:
                om = om[i]
            gain = np.where(om, gain, 0.0)

        out[i] = np.clip(p_tgt[i] + gain * residual_full, 0.0, 1.0).astype(np.float32)

    return out[0] if squeeze else out
