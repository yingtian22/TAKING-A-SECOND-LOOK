"""Localized Optimal Interpolation for frozen second-look SIC correction.

This is textbook local OI, NOT Fixed Propagation (`local_oi`) and NOT DAPPER
`OptInterp`. Innovation is formed at observation time and applied to the
frozen target-day prior. No dynamical forecast is rerun.

For each analysis pixel x, only observations with Euclidean grid distance
<= r_loc are used. The local system is

    (B_oo + R) alpha = r
    delta(x)         = B_xo alpha

with isotropic Gaussian B_ij = sigma_b^2 exp(-d(i,j)^2 / (2 L^2)).
If more than max_local_obs neighbors fall inside r_loc, the nearest
max_local_obs (ties broken by (row, col)) are kept. Observation order is
then sorted by (row, col) so the solve is permutation-invariant.

Two solver modes:
  exact_local  — per-pixel NumPy solve (reference; for tests / small grids)
  cached_local — cKDTree neighbor query + batched Cholesky (validation/runtime)
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

SolverMode = Literal["exact_local", "cached_local"]

_CLIP_MIN = 0.0
_CLIP_MAX = 1.0
_SOLVE_DTYPE = np.float64
_INITIAL_JITTER = 1e-10
_JITTER_LADDER = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
_CHUNK = 4096
_EMPTY_DIAGNOSTICS = {
    "n_target_pixels": 0,
    "n_ocean_pixels": 0,
    "n_observations": 0,
    "n_empty_neighborhood": 0,
    "n_solves": 0,
    "n_chol_ok": 0,
    "n_jitter_bump": 0,
    "n_severe_fallback": 0,
    "max_jitter_used": 0.0,
    "max_local_obs_used": 0,
    "median_local_obs": 0.0,
    "mean_local_obs": 0.0,
    "p95_local_obs": 0.0,
    "max_local_matrix_size": 0,
    "median_local_matrix_size": 0.0,
    "mean_log10_cond": float("nan"),
    "max_log10_cond": float("nan"),
    "n_cond_sampled": 0,
    "singular_solve_count": 0,
    "cholesky_failure_count": 0,
    "fallback_count": 0,
    "query_ms": 0.0,
    "assemble_ms": 0.0,
    "solve_ms": 0.0,
    "reconstruct_ms": 0.0,
}


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, _CLIP_MIN, _CLIP_MAX).astype(np.float32)


def r_variance(sigma_obs: float, epsilon_r: float) -> float:
    """Diagonal R entry: max(sigma_obs^2, epsilon_R). Never zero."""
    return float(max(float(sigma_obs) ** 2, float(epsilon_r)))


def _as_2d(*arrays: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for a in arrays:
        arr = np.asarray(a)
        if arr.ndim != 2:
            raise ValueError(f"localized OI expects 2-D fields, got shape {arr.shape}")
        out.append(arr)
    return out


def _sort_obs(ys: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((xs, ys))
    return ys[order], xs[order]


def _gaussian_b(d2: np.ndarray, length_scale: float, sigma_b2: float) -> np.ndarray:
    inv = 1.0 / (2.0 * float(length_scale) ** 2)
    return sigma_b2 * np.exp(-np.asarray(d2, dtype=_SOLVE_DTYPE) * inv)


def _chol_solve(A: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve A x = rhs for SPD A via Cholesky. A is (k,k), rhs (k,) or (n,k)."""
    chol = np.linalg.cholesky(A)
    if rhs.ndim == 1:
        z = np.linalg.solve(chol, rhs)
        return np.linalg.solve(chol.T, z)
    z = np.linalg.solve(chol, rhs.T)
    return np.linalg.solve(chol.T, z).T


def _chol_solve_batch(A: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Batched SPD solve. A: (n,k,k), rhs: (n,k) -> (n,k)."""
    chol = np.linalg.cholesky(A)
    z = np.linalg.solve(chol, rhs[..., None])
    return np.linalg.solve(np.swapaxes(chol, -1, -2), z)[..., 0]


def _select_local(
    d2: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    r_loc2: float,
    max_local_obs: int,
) -> np.ndarray:
    """Indices of local observations; nearest-max_local_obs then (row,col) sort."""
    local = np.flatnonzero(d2 <= r_loc2)
    if local.size == 0:
        return local
    if local.size > max_local_obs:
        # lex: distance, then row, then col
        keys_d = d2[local]
        keys_y = ys[local]
        keys_x = xs[local]
        take = np.lexsort((keys_x, keys_y, keys_d))[:max_local_obs]
        local = local[take]
    ly = ys[local]
    lx = xs[local]
    reorder = np.lexsort((lx, ly))
    return local[reorder]


def _solve_local_system(
    Boo: np.ndarray,
    r_loc_vec: np.ndarray,
    r_var: float,
    jitter0: float,
    stats: dict[str, Any],
) -> tuple[np.ndarray, float]:
    """Cholesky with jitter ladder; lstsq is a counted severe fallback."""
    k = int(Boo.shape[0])
    eye = np.eye(k, dtype=_SOLVE_DTYPE)
    last_err: Exception | None = None
    bumped = False
    for jit in _JITTER_LADDER:
        if jit < jitter0:
            continue
        A = Boo + (r_var + jit) * eye
        try:
            alpha = _chol_solve(A, r_loc_vec)
            if jit > jitter0 + 1e-30:
                bumped = True
            stats["n_chol_ok"] += 1
            if bumped:
                stats["n_jitter_bump"] += 1
            stats["max_jitter_used"] = max(float(stats["max_jitter_used"]), float(jit))
            return alpha, float(jit)
        except np.linalg.LinAlgError as exc:
            last_err = exc
            stats["cholesky_failure_count"] += 1
            continue
    stats["n_severe_fallback"] += 1
    stats["fallback_count"] += 1
    stats["singular_solve_count"] += 1
    A = Boo + (r_var + _JITTER_LADDER[-1]) * eye
    alpha = np.linalg.lstsq(A, r_loc_vec, rcond=None)[0]
    stats["max_jitter_used"] = max(float(stats["max_jitter_used"]), float(_JITTER_LADDER[-1]))
    if last_err is not None:
        stats["last_lin_alg_error"] = str(last_err)
    return alpha.astype(_SOLVE_DTYPE), float(_JITTER_LADDER[-1])


def _exact_local_delta(
    prior_obs: np.ndarray,
    sparse: np.ndarray,
    obs_mask: np.ndarray,
    ocean_mask: np.ndarray | None,
    length_scale: float,
    r_loc: float,
    sigma_b: float,
    r_var: float,
    max_local_obs: int,
    collect_cond: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    import time

    h, w = prior_obs.shape
    stats = {k: (0 if isinstance(v, int) else v) for k, v in _EMPTY_DIAGNOSTICS.items()}
    stats["n_target_pixels"] = int(h * w)
    t0 = time.perf_counter()

    ys, xs = np.where(np.asarray(obs_mask, dtype=bool))
    ys, xs = _sort_obs(ys.astype(np.int32), xs.astype(np.int32))
    innov = (np.asarray(sparse, dtype=_SOLVE_DTYPE)[ys, xs]
             - np.asarray(prior_obs, dtype=_SOLVE_DTYPE)[ys, xs])
    n_obs = int(ys.size)
    stats["n_observations"] = n_obs
    delta = np.zeros((h, w), dtype=_SOLVE_DTYPE)
    if n_obs == 0:
        stats["query_ms"] = (time.perf_counter() - t0) * 1000.0
        return delta, stats

    om = None if ocean_mask is None else np.asarray(ocean_mask, dtype=bool)
    tgt_ys, tgt_xs = np.where(om) if om is not None else np.indices((h, w)).reshape(2, -1)
    tgt_ys = np.asarray(tgt_ys, dtype=np.int32).ravel()
    tgt_xs = np.asarray(tgt_xs, dtype=np.int32).ravel()
    stats["n_ocean_pixels"] = int(tgt_ys.size)
    stats["query_ms"] = (time.perf_counter() - t0) * 1000.0

    sigma_b2 = float(sigma_b) ** 2
    r_loc2 = float(r_loc) ** 2
    local_counts: list[int] = []
    cond_logs: list[float] = []
    t_asm = t_solve = t_rec = 0.0

    for y, x in zip(tgt_ys.tolist(), tgt_xs.tolist()):
        t1 = time.perf_counter()
        d2 = (ys.astype(np.int64) - int(y)) ** 2 + (xs.astype(np.int64) - int(x)) ** 2
        local = _select_local(d2.astype(_SOLVE_DTYPE), ys, xs, r_loc2, max_local_obs)
        t_asm += time.perf_counter() - t1
        if local.size == 0:
            stats["n_empty_neighborhood"] += 1
            local_counts.append(0)
            continue
        k = int(local.size)
        local_counts.append(k)
        ly = ys[local]
        lx = xs[local]
        t1 = time.perf_counter()
        dy = ly.astype(_SOLVE_DTYPE)[:, None] - ly.astype(_SOLVE_DTYPE)[None, :]
        dx = lx.astype(_SOLVE_DTYPE)[:, None] - lx.astype(_SOLVE_DTYPE)[None, :]
        Boo = _gaussian_b(dy * dy + dx * dx, length_scale, sigma_b2)
        rvec = innov[local]
        t_asm += time.perf_counter() - t1
        t1 = time.perf_counter()
        alpha, _jit = _solve_local_system(Boo, rvec, r_var, _INITIAL_JITTER, stats)
        t_solve += time.perf_counter() - t1
        stats["n_solves"] += 1
        if collect_cond and k >= 1:
            A = Boo + (r_var + max(_INITIAL_JITTER, _jit)) * np.eye(k)
            try:
                c = float(np.linalg.cond(A))
                if np.isfinite(c) and c > 0:
                    cond_logs.append(float(np.log10(c)))
            except np.linalg.LinAlgError:
                pass
        t1 = time.perf_counter()
        d2x = (ly.astype(_SOLVE_DTYPE) - y) ** 2 + (lx.astype(_SOLVE_DTYPE) - x) ** 2
        bxo = _gaussian_b(d2x, length_scale, sigma_b2)
        delta[y, x] = float(bxo @ alpha)
        t_rec += time.perf_counter() - t1

    _finalize_count_stats(stats, local_counts, cond_logs)
    stats["assemble_ms"] = t_asm * 1000.0
    stats["solve_ms"] = t_solve * 1000.0
    stats["reconstruct_ms"] = t_rec * 1000.0
    return delta, stats


def _finalize_count_stats(
    stats: dict[str, Any],
    local_counts: list[int],
    cond_logs: list[float],
) -> None:
    arr = np.asarray(local_counts, dtype=np.int32) if local_counts else np.zeros(0, dtype=np.int32)
    used = arr[arr > 0]
    stats["max_local_obs_used"] = int(arr.max()) if arr.size else 0
    stats["median_local_obs"] = float(np.median(arr)) if arr.size else 0.0
    stats["mean_local_obs"] = float(np.mean(arr)) if arr.size else 0.0
    stats["p95_local_obs"] = float(np.percentile(arr, 95)) if arr.size else 0.0
    stats["max_local_matrix_size"] = int(used.max()) if used.size else 0
    stats["median_local_matrix_size"] = float(np.median(used)) if used.size else 0.0
    if cond_logs:
        cl = np.asarray(cond_logs, dtype=np.float64)
        stats["mean_log10_cond"] = float(np.mean(cl))
        stats["max_log10_cond"] = float(np.max(cl))
        stats["n_cond_sampled"] = int(cl.size)


@njit
def _chol_lower(A, L, n):
    """In-place Cholesky A -> L (lower). Returns 0 on success, 1 on failure."""
    for i in range(n):
        s = A[i, i]
        for k in range(i):
            s -= L[i, k] * L[i, k]
        if s <= 1e-30:
            return 1
        L[i, i] = s ** 0.5
        inv = 1.0 / L[i, i]
        for j in range(i + 1, n):
            t = A[j, i]
            for k in range(i):
                t -= L[j, k] * L[i, k]
            L[j, i] = t * inv
    return 0


@njit
def _chol_solve_vec(L, b, x, z, n):
    """Solve L L^T x = b using workspace z."""
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= L[i, k] * z[k]
        z[i] = s / L[i, i]
    for i in range(n - 1, -1, -1):
        s = z[i]
        for k in range(i + 1, n):
            s -= L[k, i] * x[k]
        x[i] = s / L[i, i]


@njit
def _lstsq_small(A, b, x, n):
    """Normal-equation fallback for tiny systems."""
    ata = np.zeros((n, n), dtype=np.float64)
    atb = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for k in range(n):
            s += A[k, i] * b[k]
        atb[i] = s
        for j in range(i, n):
            v = 0.0
            for k in range(n):
                v += A[k, i] * A[k, j]
            ata[i, j] = v
            ata[j, i] = v
        ata[i, i] += 1e-8
    L = np.zeros((n, n), dtype=np.float64)
    if _chol_lower(ata, L, n) != 0:
        for i in range(n):
            x[i] = 0.0
        return 1
    z = np.zeros(n, dtype=np.float64)
    _chol_solve_vec(L, atb, x, z, n)
    return 0


@njit(parallel=True)
def _numba_local_oi_kernel(
    tgt_y,
    tgt_x,
    obs_y,
    obs_x,
    innov,
    nn_idx,
    nn_valid,
    length_scale,
    sigma_b2,
    r_var,
    width,
    max_k,
    delta_out,
    k_count_out,
    jitter_out,
    fail_out,
    bump_out,
):
    """Per-pixel localized OI. nn_idx/nn_valid from KDTree (k-nearest within r_loc)."""
    n_tgt = tgt_y.shape[0]
    inv2l2 = 1.0 / (2.0 * length_scale * length_scale)
    jitters = np.array([1e-10, 1e-8, 1e-6, 1e-4, 1e-2], dtype=np.float64)
    for p in prange(n_tgt):
        tmp_idx = np.empty(max_k, dtype=np.int32)
        kk = 0
        for j in range(max_k):
            if nn_valid[p, j]:
                tmp_idx[kk] = nn_idx[p, j]
                kk += 1
        k_count_out[p] = kk
        if kk == 0:
            delta_out[p] = 0.0
            jitter_out[p] = 0.0
            fail_out[p] = 0
            bump_out[p] = 0
            continue
        keys = np.empty(kk, dtype=np.int64)
        for j in range(kk):
            ii = tmp_idx[j]
            keys[j] = np.int64(obs_y[ii]) * np.int64(width + 1) + np.int64(obs_x[ii])
        for a in range(kk - 1):
            mloc = a
            for b in range(a + 1, kk):
                if keys[b] < keys[mloc]:
                    mloc = b
            if mloc != a:
                keys[a], keys[mloc] = keys[mloc], keys[a]
                tmp_idx[a], tmp_idx[mloc] = tmp_idx[mloc], tmp_idx[a]
        ly = np.empty(kk, dtype=np.float64)
        lx = np.empty(kk, dtype=np.float64)
        rvec = np.empty(kk, dtype=np.float64)
        for j in range(kk):
            ii = tmp_idx[j]
            ly[j] = obs_y[ii]
            lx[j] = obs_x[ii]
            rvec[j] = innov[ii]
        Boo = np.empty((kk, kk), dtype=np.float64)
        for i in range(kk):
            for j in range(i, kk):
                d2 = (ly[i] - ly[j]) * (ly[i] - ly[j]) + (lx[i] - lx[j]) * (lx[i] - lx[j])
                v = sigma_b2 * np.exp(-d2 * inv2l2)
                Boo[i, j] = v
                Boo[j, i] = v
        A = np.empty((kk, kk), dtype=np.float64)
        Lmat = np.empty((kk, kk), dtype=np.float64)
        alpha = np.empty(kk, dtype=np.float64)
        z = np.empty(kk, dtype=np.float64)
        ok = 0
        used_jit = jitters[0]
        bumped = 0
        for t in range(jitters.shape[0]):
            jit = jitters[t]
            for i in range(kk):
                for j in range(kk):
                    A[i, j] = Boo[i, j]
                A[i, i] = Boo[i, i] + r_var + jit
            for i in range(kk):
                for j in range(kk):
                    Lmat[i, j] = 0.0
            if _chol_lower(A, Lmat, kk) == 0:
                _chol_solve_vec(Lmat, rvec, alpha, z, kk)
                ok = 1
                used_jit = jit
                if t > 0:
                    bumped = 1
                break
        if ok == 0:
            for i in range(kk):
                for j in range(kk):
                    A[i, j] = Boo[i, j]
                A[i, i] = Boo[i, i] + r_var + jitters[jitters.shape[0] - 1]
            _lstsq_small(A, rvec, alpha, kk)
            used_jit = jitters[jitters.shape[0] - 1]
            fail_out[p] = 1
        else:
            fail_out[p] = 0
        jitter_out[p] = used_jit
        bump_out[p] = bumped
        py = float(tgt_y[p])
        px = float(tgt_x[p])
        acc = 0.0
        for j in range(kk):
            d2 = (py - ly[j]) * (py - ly[j]) + (px - lx[j]) * (px - lx[j])
            acc += (sigma_b2 * np.exp(-d2 * inv2l2)) * alpha[j]
        delta_out[p] = acc


def _neighbor_query(
    obs_xy: np.ndarray,
    tgt_xy: np.ndarray,
    r_loc: float,
    max_local_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx, dist) with shape (n_tgt, max_local_obs); inf => missing."""
    n_obs = int(obs_xy.shape[0])
    k = int(min(max_local_obs, n_obs))
    tree = cKDTree(obs_xy)
    dist, idx = tree.query(
        tgt_xy, k=k, distance_upper_bound=float(r_loc) + 1e-12, workers=-1
    )
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    return idx.astype(np.int32), dist.astype(_SOLVE_DTYPE)


def _cached_local_delta(
    prior_obs: np.ndarray,
    sparse: np.ndarray,
    obs_mask: np.ndarray,
    ocean_mask: np.ndarray | None,
    length_scale: float,
    r_loc: float,
    sigma_b: float,
    r_var: float,
    max_local_obs: int,
    collect_cond: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    import time

    h, w = prior_obs.shape
    stats = {k: (0 if isinstance(v, int) else v) for k, v in _EMPTY_DIAGNOSTICS.items()}
    stats["n_target_pixels"] = int(h * w)
    om = np.ones((h, w), dtype=bool) if ocean_mask is None else np.asarray(ocean_mask, dtype=bool)
    tgt_ys, tgt_xs = np.where(om)
    tgt_ys = np.ascontiguousarray(tgt_ys.astype(np.int32))
    tgt_xs = np.ascontiguousarray(tgt_xs.astype(np.int32))
    stats["n_ocean_pixels"] = int(tgt_ys.size)

    ys, xs = np.where(np.asarray(obs_mask, dtype=bool))
    ys, xs = _sort_obs(ys.astype(np.int32), xs.astype(np.int32))
    ys = np.ascontiguousarray(ys)
    xs = np.ascontiguousarray(xs)
    n_obs = int(ys.size)
    stats["n_observations"] = n_obs
    delta = np.zeros((h, w), dtype=_SOLVE_DTYPE)
    if n_obs == 0 or tgt_ys.size == 0:
        stats["n_empty_neighborhood"] = int(tgt_ys.size)
        return delta, stats

    innov = np.ascontiguousarray(
        np.asarray(sparse, dtype=_SOLVE_DTYPE)[ys, xs]
        - np.asarray(prior_obs, dtype=_SOLVE_DTYPE)[ys, xs]
    )
    obs_xy = np.column_stack((ys.astype(_SOLVE_DTYPE), xs.astype(_SOLVE_DTYPE)))
    tgt_xy = np.column_stack((tgt_ys.astype(_SOLVE_DTYPE), tgt_xs.astype(_SOLVE_DTYPE)))

    t0 = time.perf_counter()
    nn_idx, nn_dist = _neighbor_query(obs_xy, tgt_xy, r_loc, max_local_obs)
    stats["query_ms"] = (time.perf_counter() - t0) * 1000.0
    valid = np.isfinite(nn_dist) & (nn_dist <= float(r_loc) + 1e-12)
    valid &= nn_idx < n_obs
    nn_idx = np.ascontiguousarray(nn_idx)
    valid = np.ascontiguousarray(valid)

    n_tgt = int(tgt_ys.size)
    k_max = int(nn_idx.shape[1])
    delta_out = np.zeros(n_tgt, dtype=np.float64)
    k_count_out = np.zeros(n_tgt, dtype=np.int32)
    jitter_out = np.zeros(n_tgt, dtype=np.float64)
    fail_out = np.zeros(n_tgt, dtype=np.int32)
    bump_out = np.zeros(n_tgt, dtype=np.int32)

    t1 = time.perf_counter()
    _numba_local_oi_kernel(
        tgt_ys, tgt_xs, ys, xs, innov, nn_idx, valid,
        float(length_scale), float(sigma_b) ** 2, float(r_var),
        int(w), int(k_max),
        delta_out, k_count_out, jitter_out, fail_out, bump_out,
    )
    solve_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    delta[tgt_ys, tgt_xs] = delta_out
    rec_ms = (time.perf_counter() - t2) * 1000.0

    local_counts = k_count_out.tolist()
    stats["n_empty_neighborhood"] = int((k_count_out == 0).sum())
    stats["n_solves"] = int((k_count_out > 0).sum())
    stats["n_chol_ok"] = int(((k_count_out > 0) & (fail_out == 0)).sum())
    stats["n_jitter_bump"] = int(bump_out.sum())
    stats["n_severe_fallback"] = int(fail_out.sum())
    stats["fallback_count"] = int(fail_out.sum())
    stats["singular_solve_count"] = int(fail_out.sum())
    stats["cholesky_failure_count"] = int(fail_out.sum())
    stats["max_jitter_used"] = float(jitter_out.max()) if n_tgt else 0.0
    stats["assemble_ms"] = 0.0
    stats["solve_ms"] = solve_ms
    stats["reconstruct_ms"] = rec_ms

    cond_logs: list[float] = []
    if collect_cond and stats["n_solves"] > 0:
        pick = np.linspace(0, n_tgt - 1, num=min(64, n_tgt), dtype=int)
        sigma_b2 = float(sigma_b) ** 2
        for pi in pick.tolist():
            kk = int(k_count_out[pi])
            if kk < 1:
                continue
            idxs = nn_idx[pi, valid[pi]]
            if idxs.size != kk:
                idxs = nn_idx[pi, :kk]
            ly = ys[idxs].astype(_SOLVE_DTYPE)
            lx = xs[idxs].astype(_SOLVE_DTYPE)
            dy = ly[:, None] - ly[None, :]
            dx = lx[:, None] - lx[None, :]
            Boo = _gaussian_b(dy * dy + dx * dx, length_scale, sigma_b2)
            jit = max(_INITIAL_JITTER, float(jitter_out[pi]))
            A = Boo + (r_var + jit) * np.eye(kk)
            try:
                c = float(np.linalg.cond(A))
                if np.isfinite(c) and c > 0:
                    cond_logs.append(float(np.log10(max(c, 1.0))))
            except np.linalg.LinAlgError:
                pass

    _finalize_count_stats(stats, local_counts, cond_logs)
    return delta, stats



def _batch_chol_with_fallback(
    Boo: np.ndarray,
    rhs: np.ndarray,
    r_var: float,
) -> tuple[np.ndarray, float, int, int, int]:
    """Try increasing jitter on the whole chunk; per-row lstsq for leftovers."""
    n, k, _ = Boo.shape
    eye = np.eye(k, dtype=_SOLVE_DTYPE)[None, :, :]
    n_bump = 0
    n_fail = 0
    used_jit = _INITIAL_JITTER
    alpha = np.empty((n, k), dtype=_SOLVE_DTYPE)
    remaining = np.ones(n, dtype=bool)

    for i_jit, jit in enumerate(_JITTER_LADDER):
        if not remaining.any():
            break
        idx = np.flatnonzero(remaining)
        A = Boo[idx] + (r_var + jit) * eye
        try:
            sol = _chol_solve_batch(A, rhs[idx])
            alpha[idx] = sol
            remaining[idx] = False
            used_jit = max(used_jit, float(jit))
            if i_jit > 0:
                n_bump += int(idx.size)
        except np.linalg.LinAlgError:
            # identify which rows failed by trying smaller groups
            if idx.size == 1:
                continue
            mid = idx.size // 2
            for part in (idx[:mid], idx[mid:]):
                A2 = Boo[part] + (r_var + jit) * eye
                try:
                    sol = _chol_solve_batch(A2, rhs[part])
                    alpha[part] = sol
                    remaining[part] = False
                    used_jit = max(used_jit, float(jit))
                    if i_jit > 0:
                        n_bump += int(part.size)
                except np.linalg.LinAlgError:
                    continue

    if remaining.any():
        idx = np.flatnonzero(remaining)
        n_fail = int(idx.size)
        jit = float(_JITTER_LADDER[-1])
        used_jit = max(used_jit, jit)
        eye_k = np.eye(k, dtype=_SOLVE_DTYPE)
        for i in idx.tolist():
            A = Boo[i] + (r_var + jit) * eye_k
            alpha[i] = np.linalg.lstsq(A, rhs[i], rcond=None)[0]
            remaining[i] = False

    n_ok = n - n_fail
    return alpha, used_jit, n_ok, n_bump, n_fail


def localized_oi_correct(
    forecast_obs_time: np.ndarray,
    forecast_target: np.ndarray,
    observations: np.ndarray,
    obs_mask: np.ndarray,
    sigma_obs: float,
    length_scale: float,
    localization_radius: float,
    sigma_background: float,
    r_floor: float,
    ocean_mask: np.ndarray | None = None,
    max_local_obs: int = 32,
    solver_mode: SolverMode = "cached_local",
    collect_cond: bool = False,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply localized OI increment to a frozen target-day forecast.

    Returns (corrected_target float32 in [0,1], diagnostics).
    Rejects any attempt to pass target-day ground truth.
    """
    forbidden = {
        "target", "targets", "x_t_true", "x_T_true", "ground_truth",
        "target_gt", "future_obs", "target_error", "lead_target",
    }
    bad = forbidden.intersection(kwargs)
    if bad:
        raise RuntimeError(f"localized OI forbids target leakage kwargs: {sorted(bad)}")

    fo, ft, obs, mask = _as_2d(
        forecast_obs_time, forecast_target, observations, obs_mask
    )
    if ocean_mask is not None:
        ocean_mask = np.asarray(ocean_mask, dtype=bool)
        if ocean_mask.shape != fo.shape:
            raise ValueError("ocean_mask shape mismatch")

    if float(length_scale) <= 0:
        raise ValueError("length_scale must be > 0")
    if float(localization_radius) <= 0:
        raise ValueError("localization_radius must be > 0")
    if int(max_local_obs) < 1:
        raise ValueError("max_local_obs must be >= 1")

    r_var = r_variance(sigma_obs, r_floor)
    args = dict(
        prior_obs=fo,
        sparse=obs,
        obs_mask=mask,
        ocean_mask=ocean_mask,
        length_scale=float(length_scale),
        r_loc=float(localization_radius),
        sigma_b=float(sigma_background),
        r_var=r_var,
        max_local_obs=int(max_local_obs),
        collect_cond=bool(collect_cond),
    )
    if solver_mode == "exact_local":
        delta, stats = _exact_local_delta(**args)
    elif solver_mode == "cached_local":
        delta, stats = _cached_local_delta(**args)
    else:
        raise ValueError(f"unknown solver_mode {solver_mode!r}")

    corrected = _clip01(np.asarray(ft, dtype=np.float64) + delta)
    stats = dict(stats)
    stats.update({
        "solver_mode": solver_mode,
        "length_scale": float(length_scale),
        "localization_radius": float(localization_radius),
        "sigma_background": float(sigma_background),
        "sigma_obs": float(sigma_obs),
        "r_floor": float(r_floor),
        "r_variance": float(r_var),
        "max_local_obs": int(max_local_obs),
        "severe_fallback_rate": (
            float(stats["n_severe_fallback"]) / max(int(stats["n_solves"]), 1)
        ),
    })
    return corrected, stats


def merge_diagnostics(acc: dict[str, Any] | None, stats: dict[str, Any]) -> dict[str, Any]:
    """Accumulate per-case diagnostics (sums / maxima)."""
    if acc is None:
        acc = {
            "n_cases": 0,
            "n_solves": 0,
            "n_chol_ok": 0,
            "n_jitter_bump": 0,
            "n_severe_fallback": 0,
            "n_empty_neighborhood": 0,
            "n_ocean_pixels": 0,
            "n_observations": 0,
            "singular_solve_count": 0,
            "cholesky_failure_count": 0,
            "fallback_count": 0,
            "max_jitter_used": 0.0,
            "max_local_obs_used": 0,
            "max_local_matrix_size": 0,
            "sum_median_local_obs": 0.0,
            "sum_mean_local_obs": 0.0,
            "cond_logs_mean_sum": 0.0,
            "cond_logs_max": float("-inf"),
            "n_cond_cases": 0,
            "query_ms_sum": 0.0,
            "assemble_ms_sum": 0.0,
            "solve_ms_sum": 0.0,
            "reconstruct_ms_sum": 0.0,
        }
    acc["n_cases"] += 1
    for key in (
        "n_solves", "n_chol_ok", "n_jitter_bump", "n_severe_fallback",
        "n_empty_neighborhood", "n_ocean_pixels", "n_observations",
        "singular_solve_count", "cholesky_failure_count", "fallback_count",
    ):
        acc[key] += int(stats.get(key, 0))
    acc["max_jitter_used"] = max(float(acc["max_jitter_used"]), float(stats.get("max_jitter_used", 0)))
    acc["max_local_obs_used"] = max(int(acc["max_local_obs_used"]), int(stats.get("max_local_obs_used", 0)))
    acc["max_local_matrix_size"] = max(
        int(acc["max_local_matrix_size"]), int(stats.get("max_local_matrix_size", 0))
    )
    acc["sum_median_local_obs"] += float(stats.get("median_local_obs", 0))
    acc["sum_mean_local_obs"] += float(stats.get("mean_local_obs", 0))
    mlc = stats.get("mean_log10_cond", float("nan"))
    if mlc is not None and np.isfinite(mlc):
        acc["cond_logs_mean_sum"] += float(mlc)
        acc["n_cond_cases"] += 1
        acc["cond_logs_max"] = max(float(acc["cond_logs_max"]), float(stats.get("max_log10_cond", mlc)))
    for key in ("query_ms", "assemble_ms", "solve_ms", "reconstruct_ms"):
        acc[f"{key}_sum"] += float(stats.get(key, 0.0))
    return acc


def finalize_merged_diagnostics(acc: dict[str, Any]) -> dict[str, Any]:
    n = max(int(acc.get("n_cases", 0)), 1)
    n_solves = max(int(acc.get("n_solves", 0)), 1)
    out = dict(acc)
    out["severe_fallback_rate"] = float(acc.get("n_severe_fallback", 0)) / n_solves
    out["jitter_bump_rate"] = float(acc.get("n_jitter_bump", 0)) / n_solves
    out["mean_median_local_obs"] = float(acc.get("sum_median_local_obs", 0)) / n
    out["mean_mean_local_obs"] = float(acc.get("sum_mean_local_obs", 0)) / n
    nc = int(acc.get("n_cond_cases", 0))
    out["mean_log10_cond"] = (
        float(acc["cond_logs_mean_sum"]) / nc if nc else float("nan")
    )
    if not np.isfinite(out.get("cond_logs_max", np.nan)):
        out["max_log10_cond"] = float("nan")
    else:
        out["max_log10_cond"] = float(acc["cond_logs_max"])
    out["mean_query_ms"] = float(acc.get("query_ms_sum", 0)) / n
    out["mean_assemble_ms"] = float(acc.get("assemble_ms_sum", 0)) / n
    out["mean_solve_ms"] = float(acc.get("solve_ms_sum", 0)) / n
    out["mean_reconstruct_ms"] = float(acc.get("reconstruct_ms_sum", 0)) / n
    return out


def _toy_single_obs_delta(
    h: int,
    w: int,
    oy: int,
    ox: int,
    innov: float,
    length_scale: float,
    sigma_b: float,
    r_var: float,
) -> np.ndarray:
    """Closed-form local OI increment for a single observation."""
    sigma_b2 = float(sigma_b) ** 2
    alpha = float(innov) / (sigma_b2 + r_var)
    yy, xx = np.ogrid[:h, :w]
    d2 = (yy - oy) ** 2 + (xx - ox) ** 2
    bxo = _gaussian_b(d2, length_scale, sigma_b2)
    return bxo * alpha


def run_unit_tests(rtol: float = 1e-6) -> list[dict[str, Any]]:
    """Stage 20B correctness tests. Returns one record per test."""
    results: list[dict[str, Any]] = []

    def _record(name: str, passed: bool, detail: str) -> None:
        results.append({"test": name, "passed": bool(passed), "detail": detail})

    h = w = 9
    ocean = np.ones((h, w), dtype=bool)
    prior_obs = np.full((h, w), 0.4, dtype=np.float32)
    prior_tgt = np.full((h, w), 0.5, dtype=np.float32)
    L, r_loc, sb, rf = 2.0, 8.0, 0.10, 1e-5
    common = dict(
        length_scale=L,
        localization_radius=r_loc,
        sigma_background=sb,
        r_floor=rf,
        ocean_mask=ocean,
        max_local_obs=32,
    )

    # Test 1 — zero innovation
    mask = np.zeros((h, w), dtype=bool)
    mask[2, 3] = True
    mask[5, 6] = True
    obs = prior_obs.copy()
    obs[mask] = prior_obs[mask]
    for mode in ("exact_local", "cached_local"):
        out, _ = localized_oi_correct(
            prior_obs, prior_tgt, obs, mask, sigma_obs=0.03, solver_mode=mode, **common
        )
        err = float(np.max(np.abs(out - prior_tgt)))
        _record(
            f"T1_zero_innovation_{mode}",
            err < 1e-7,
            f"max|X_OI - X_T^f|={err:.3e}",
        )

    # Test 2 — single observation vs analytic Kalman update
    mask = np.zeros((h, w), dtype=bool)
    mask[4, 4] = True
    innov = 0.20
    obs = prior_obs.copy()
    obs[4, 4] = prior_obs[4, 4] + innov
    sigma_obs = 0.03
    r_var = r_variance(sigma_obs, rf)
    analytic = _toy_single_obs_delta(h, w, 4, 4, innov, L, sb, r_var)
    expected = _clip01(prior_tgt.astype(np.float64) + analytic)
    for mode in ("exact_local", "cached_local"):
        out, _ = localized_oi_correct(
            prior_obs, prior_tgt, obs, mask, sigma_obs=sigma_obs,
            solver_mode=mode, **common
        )
        err = float(np.max(np.abs(out.astype(np.float64) - expected.astype(np.float64))))
        _record(
            f"T2_single_observation_{mode}",
            err < 1e-6,
            f"max abs vs analytic={err:.3e}",
        )

    # Test 3 — noise → 0: observed pixel increment ≈ innov * sb2/(sb2+eps)
    mask = np.zeros((h, w), dtype=bool)
    mask[4, 4] = True
    obs = prior_obs.copy()
    obs[4, 4] = prior_obs[4, 4] + innov
    out, _ = localized_oi_correct(
        prior_obs, prior_tgt, obs, mask, sigma_obs=0.0,
        solver_mode="exact_local", **common
    )
    delta_at_obs = float(out[4, 4] - prior_tgt[4, 4])
    sb2 = sb ** 2
    expected_gain = sb2 / (sb2 + rf)
    _record(
        "T3_near_zero_noise_response",
        abs(delta_at_obs - innov * expected_gain) < 5e-3 and delta_at_obs > 0.15,
        f"delta_at_obs={delta_at_obs:.6f} expected≈{innov * expected_gain:.6f}",
    )

    # Test 4 — sign consistency (single positive innovation)
    out, _ = localized_oi_correct(
        prior_obs, prior_tgt, obs, mask, sigma_obs=0.03,
        solver_mode="exact_local", **common
    )
    delta = out.astype(np.float64) - prior_tgt.astype(np.float64)
    _record(
        "T4_sign_consistency",
        bool(np.all(delta >= -1e-12) and np.max(delta) > 0),
        f"min_delta={float(delta.min()):.3e} max_delta={float(delta.max()):.3e}",
    )

    # Test 5 — permutation invariance
    mask = np.zeros((h, w), dtype=bool)
    mask[1, 2] = True
    mask[3, 7] = True
    mask[6, 1] = True
    obs = prior_obs.copy()
    obs[1, 2] = prior_obs[1, 2] + 0.11
    obs[3, 7] = prior_obs[3, 7] - 0.07
    obs[6, 1] = prior_obs[6, 1] + 0.04
    out_a, _ = localized_oi_correct(
        prior_obs, prior_tgt, obs, mask, sigma_obs=0.03,
        solver_mode="exact_local", **common
    )
    # shuffling stored observation values in a copied mask-equivalent field
    # (same locations) must match; construct by writing in reverse visit order
    obs_b = np.zeros_like(obs)
    obs_b[mask] = obs[mask]
    out_b, _ = localized_oi_correct(
        prior_obs, prior_tgt, obs_b, mask, sigma_obs=0.03,
        solver_mode="cached_local", **common
    )
    err = float(np.max(np.abs(out_a - out_b)))
    _record("T5_permutation_invariance", err < 1e-6, f"exact vs cached (same mask) maxabs={err:.3e}")

    # extra: physically permute by transposing coordinate interpretation — rebuild
    # observations in a different numpy where order
    ys, xs = np.where(mask)
    order = np.arange(ys.size)[::-1]
    mask_perm = np.zeros_like(mask)
    obs_perm = prior_obs.copy()
    for y, x in zip(ys[order], xs[order]):
        mask_perm[y, x] = True
        obs_perm[y, x] = obs[y, x]
    out_p, _ = localized_oi_correct(
        prior_obs, prior_tgt, obs_perm, mask_perm, sigma_obs=0.03,
        solver_mode="exact_local", **common
    )
    errp = float(np.max(np.abs(out_a - out_p)))
    _record("T5b_observation_order", errp < 1e-6, f"reversed where-order maxabs={errp:.3e}")

    # Test 6 — no target leakage
    leaked = False
    try:
        localized_oi_correct(
            prior_obs, prior_tgt, obs, mask, sigma_obs=0.03,
            solver_mode="exact_local", target=prior_tgt, **common
        )
    except RuntimeError as exc:
        leaked = "leakage" in str(exc).lower() or "forbid" in str(exc).lower()
    _record("T6_no_target_leakage", leaked, "kwargs target rejected" if leaked else "DID NOT REJECT")

    # Test 7 — determinism
    outs = []
    for _ in range(3):
        o, _s = localized_oi_correct(
            prior_obs, prior_tgt, obs, mask, sigma_obs=0.03,
            solver_mode="cached_local", **common
        )
        outs.append(o)
    dmax = max(float(np.max(np.abs(outs[0] - outs[i]))) for i in range(1, 3))
    _record("T7_determinism", dmax == 0.0, f"repeat maxabs={dmax:.3e}")

    # Test 8 — clipping
    prior_hi = np.full((h, w), 0.98, dtype=np.float32)
    obs_hi = prior_obs.copy()
    obs_hi[4, 4] = 1.0
    mask_hi = np.zeros((h, w), dtype=bool)
    mask_hi[4, 4] = True
    out, _ = localized_oi_correct(
        prior_obs, prior_hi, obs_hi, mask_hi, sigma_obs=0.0,
        solver_mode="exact_local", **common
    )
    _record(
        "T8_clipping",
        bool(out.min() >= 0.0 - 1e-12 and out.max() <= 1.0 + 1e-12),
        f"range=[{float(out.min()):.6f}, {float(out.max()):.6f}]",
    )

    # exact vs cached on a multi-obs toy (no truncation)
    err_ec = float(np.max(np.abs(out_a - out_b)))
    _record(
        "T_exact_vs_cached_toy",
        err_ec < 1e-6,
        f"maxabs={err_ec:.3e} (also T5)",
    )
    return results

