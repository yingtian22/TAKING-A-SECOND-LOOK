from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from secondlook.baselines.observation_baselines import local_oi, residual_interp
from secondlook.observation.interpolation import distance_to_mask
from secondlook.observation.masks import MaskType, generate_obs_mask
from secondlook.observation.perturbation import apply_observation_noise
from secondlook.utils.arrays import fill_nan

PriorMode = Literal["mixed", "persistence", "direct_unet"]
BaseType = Literal["residual_interp", "local_oi"]


def _load_split_array(npz_path: str, key: str, cache_dir: Path | None) -> np.ndarray:
    """
    Load one array from an NPZ split.

    Compressed (ZIP_DEFLATED) NPZ members cannot be memory-mapped; numpy would
    decompress the whole array into RAM on every access. When *cache_dir* is
    given, extract the member once to an uncompressed .npy cache and mmap it.
    """
    with zipfile.ZipFile(npz_path) as zf:
        compress_type = zf.getinfo(f"{key}.npy").compress_type
    if compress_type == zipfile.ZIP_STORED:
        archive = np.load(npz_path, mmap_mode="r", allow_pickle=True)
        return archive[key]
    if cache_dir is None:
        archive = np.load(npz_path, allow_pickle=True)
        return np.asarray(archive[key], dtype=np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{Path(npz_path).stem}_{key}.npy"
    if cache_path.is_file():
        try:
            return np.load(cache_path, mmap_mode="r")
        except ValueError:
            cache_path.unlink(missing_ok=True)
    archive = np.load(npz_path, allow_pickle=True)
    arr = np.asarray(archive[key], dtype=np.float32)
    np.save(cache_path, arr)
    del arr
    return np.load(cache_path, mmap_mode="r")


def _ensure_hw(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    return out


def _compute_base_correction(
    base_type: BaseType,
    *,
    prior_target: np.ndarray,
    prior_obs_day: np.ndarray,
    sparse_obs: np.ndarray,
    obs_mask: np.ndarray,
    ocean_mask: np.ndarray | None,
    oi_length_scale: float,
) -> np.ndarray:
    if base_type == "residual_interp":
        out = residual_interp(
            prior_target,
            prior_obs_day,
            sparse_obs,
            obs_mask,
            ocean_mask=ocean_mask,
        )
    else:
        out = local_oi(
            prior_target,
            prior_obs_day,
            sparse_obs,
            obs_mask,
            length_scale=oi_length_scale,
            ocean_mask=ocean_mask,
        )
    return _ensure_hw(out)


FEATURE_CHANNEL_NAMES = [
    "prior_target",
    "base_correction",
    "base_delta",
    "prior_obs_day",
    "sparse_obs",
    "mask",
    "residual_obs",
    "confidence",
]


def _build_features(
    *,
    prior_target: np.ndarray,
    base_correction: np.ndarray,
    prior_obs_day: np.ndarray,
    sparse_obs: np.ndarray,
    obs_mask: np.ndarray,
    ocean_mask: np.ndarray | None,
    confidence_length_scale: float,
    disabled_feature_channels: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    prior_target = _ensure_hw(prior_target)
    base_correction = _ensure_hw(base_correction)
    prior_obs_day = _ensure_hw(prior_obs_day)
    sparse_obs = _ensure_hw(sparse_obs)
    obs_mask = _ensure_hw(obs_mask).astype(bool)
    base_delta = base_correction - prior_target
    sparse_canvas = np.where(obs_mask, sparse_obs, 0.0).astype(np.float32)
    residual_canvas = np.zeros_like(prior_obs_day, dtype=np.float32)
    residual_canvas[obs_mask] = sparse_obs[obs_mask] - prior_obs_day[obs_mask]
    dist = distance_to_mask(obs_mask, ocean_mask=ocean_mask)
    scale = max(float(confidence_length_scale), 1e-6)
    confidence = np.exp(-dist / scale).astype(np.float32)

    feats = np.stack(
        [
            prior_target,
            base_correction,
            base_delta,
            prior_obs_day,
            sparse_canvas,
            obs_mask.astype(np.float32),
            residual_canvas,
            confidence,
        ],
        axis=0,
    )
    feats = fill_nan(feats, 0.0).astype(np.float32)
    if disabled_feature_channels:
        for name in disabled_feature_channels:
            if name not in FEATURE_CHANNEL_NAMES:
                raise ValueError(
                    f"Unknown feature channel '{name}'. "
                    f"Valid: {FEATURE_CHANNEL_NAMES}"
                )
            feats[FEATURE_CHANNEL_NAMES.index(name)] = 0.0
    return feats


class CorrectionSampleDataset(Dataset):
    """
    Random correction-training cases from a real split.

    Each __getitem__ draws a sample index and observation setting, then builds
    base correction + 8-channel features without using target lead day in inputs.
    """

    def __init__(
        self,
        *,
        split: str,
        npz_path: str,
        num_samples: int,
        lead_day: int,
        base_type: BaseType,
        prior_mode: PriorMode,
        settings: dict[str, Any],
        ocean_mask: np.ndarray | None,
        oi_length_scales: dict[str, float],
        direct_unet_preds: np.ndarray | None = None,
        seed: int = 42,
        epoch: int = 0,
        cache_dir: str | Path | None = None,
        disabled_feature_channels: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.split = split
        self.lead_day = int(lead_day)
        self.base_type = base_type
        self.prior_mode = prior_mode
        self.num_samples = int(num_samples)
        self.ocean_mask = ocean_mask
        self.oi_length_scales = oi_length_scales
        self.confidence_length_scale = float(settings.get("confidence_length_scale", 10.0))
        self.obs_days = [int(d) for d in settings["obs_days"] if int(d) < self.lead_day]
        self.obs_fractions = [float(f) for f in settings["obs_fractions"]]
        self.mask_types: list[MaskType] = list(settings["mask_types"])
        self.noise_stds = [float(n) for n in settings["noise_stds"]]
        self.base_seed = int(seed)
        self.epoch = int(epoch)
        self.disabled_feature_channels = list(disabled_feature_channels or [])

        cache = Path(cache_dir) if cache_dir is not None else None
        self.inputs = _load_split_array(npz_path, "inputs", cache)
        self.targets = _load_split_array(npz_path, "targets", cache)
        self.n_real = int(self.inputs.shape[0])

        if prior_mode in ("mixed", "direct_unet"):
            if direct_unet_preds is None:
                raise ValueError("direct_unet_preds required for prior_mode direct_unet/mixed")
            if int(direct_unet_preds.shape[0]) != self.n_real:
                raise ValueError(
                    f"direct_unet_preds N={direct_unet_preds.shape[0]} != split N={self.n_real}"
                )
            self.direct_unet_preds = direct_unet_preds
        else:
            self.direct_unet_preds = None

        if prior_mode == "mixed":
            self.prior_types = ["persistence", "direct_unet"]
        elif prior_mode == "persistence":
            self.prior_types = ["persistence"]
        else:
            self.prior_types = ["direct_unet"]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.base_seed + self.epoch * 1_000_003 + idx * 9973)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = self._rng(idx)
        sample_idx = int(rng.integers(0, self.n_real))
        prior_type = str(rng.choice(self.prior_types))
        obs_day = int(rng.choice(self.obs_days))
        obs_fraction = float(rng.choice(self.obs_fractions))
        mask_type: MaskType = str(rng.choice(self.mask_types))  # type: ignore[assignment]
        noise_std = float(rng.choice(self.noise_stds))

        inp = fill_nan(np.asarray(self.inputs[sample_idx], dtype=np.float32))
        tgt_seq = fill_nan(np.asarray(self.targets[sample_idx], dtype=np.float32))
        target = tgt_seq[self.lead_day - 1]
        obs_full = tgt_seq[obs_day - 1]

        if prior_type == "persistence":
            prior_target = inp[-1].copy()
            prior_obs_day = inp[-1].copy()
        else:
            assert self.direct_unet_preds is not None
            prior_seq = fill_nan(np.asarray(self.direct_unet_preds[sample_idx], dtype=np.float32))
            prior_target = prior_seq[self.lead_day - 1]
            prior_obs_day = prior_seq[obs_day - 1]

        edge_prior = prior_target if rng.random() < 0.5 else inp[-1]
        obs_mask = generate_obs_mask(
            target.shape,
            mask_type,
            obs_fraction,
            rng,
            prior=edge_prior if mask_type == "edge" else None,
            ocean_mask=self.ocean_mask,
        )

        sparse_obs = apply_observation_noise(obs_full, obs_mask, noise_std, rng)
        oi_ls = float(self.oi_length_scales[prior_type])
        base_corr = _compute_base_correction(
            self.base_type,
            prior_target=prior_target,
            prior_obs_day=prior_obs_day,
            sparse_obs=sparse_obs,
            obs_mask=obs_mask,
            ocean_mask=self.ocean_mask,
            oi_length_scale=oi_ls,
        )

        features = _build_features(
            prior_target=prior_target,
            base_correction=base_corr,
            prior_obs_day=prior_obs_day,
            sparse_obs=sparse_obs,
            obs_mask=obs_mask,
            ocean_mask=self.ocean_mask,
            confidence_length_scale=self.confidence_length_scale,
            disabled_feature_channels=self.disabled_feature_channels,
        )

        valid = np.isfinite(target)
        if self.ocean_mask is not None:
            valid = valid & self.ocean_mask.astype(bool)

        return {
            "features": torch.from_numpy(features),
            "base_correction": torch.from_numpy(base_corr[None].astype(np.float32)),
            "target": torch.from_numpy(target[None].astype(np.float32)),
            "valid_mask": torch.from_numpy(valid[None]),
            "metadata": {
                "sample_idx": sample_idx,
                "prior_type": prior_type,
                "base_type": self.base_type,
                "obs_day": obs_day,
                "obs_fraction": obs_fraction,
                "mask_type": mask_type,
                "noise_std": noise_std,
            },
        }


def collate_correction_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "base_correction": torch.stack([b["base_correction"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch]),
        "metadata": [b["metadata"] for b in batch],
    }
