"""Stage 8B dataset: identical sampling distribution to Stage 3's
CorrectionSampleDataset (same RNG call sequence), but returns the fields
needed by adaptive propagation training:

- the 8 Stage-8B input channels (see adaptive_propagation.FEATURE_NAMES)
- dist / innovation_nn for the differentiable correction path
- the prior-specific base length L0 (8 persistence / 5 direct_unet)
- target and valid mask (target is used ONLY in the loss)

V1 code is not modified; this subclass only adds a new __getitem__.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from secondlook.baselines.adaptive_propagation import (
    build_scale_features,
    nearest_innovation_fields,
)
from secondlook.baselines.observation_baselines import local_oi
from secondlook.datasets.correction_dataset import CorrectionSampleDataset
from secondlook.observation.masks import generate_obs_mask
from secondlook.observation.perturbation import apply_observation_noise
from secondlook.utils.arrays import fill_nan


class AdaptiveScaleDataset(CorrectionSampleDataset):
    """Same case sampling as CorrectionSampleDataset; Stage-8B outputs."""

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # --- identical RNG consumption order as the parent class ---
        rng = self._rng(idx)
        sample_idx = int(rng.integers(0, self.n_real))
        prior_type = str(rng.choice(self.prior_types))
        obs_day = int(rng.choice(self.obs_days))
        obs_fraction = float(rng.choice(self.obs_fractions))
        mask_type = str(rng.choice(self.mask_types))
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
        # --- end of the identical sampling path ---

        l0 = float(self.oi_length_scales[prior_type])
        fixed_corr = local_oi(
            prior_target, prior_obs_day, sparse_obs, obs_mask, l0,
            ocean_mask=self.ocean_mask,
        )
        dist, innovation_nn = nearest_innovation_fields(
            prior_obs_day=prior_obs_day,
            sparse_obs=sparse_obs,
            obs_mask=obs_mask,
            ocean_mask=self.ocean_mask,
        )
        features = build_scale_features(
            prior_target=prior_target,
            prior_obs_day=prior_obs_day,
            sparse_obs=sparse_obs,
            obs_mask=obs_mask,
            fixed_local_correction=fixed_corr,
            dist=np.where(np.isfinite(dist), dist, 1e6).astype(np.float32),
            ocean_mask=self.ocean_mask,
            confidence_length_scale=self.confidence_length_scale,
        )

        valid = np.isfinite(target)
        if self.ocean_mask is not None:
            valid = valid & self.ocean_mask.astype(bool)

        return {
            "features": torch.from_numpy(features),
            "prior_target": torch.from_numpy(prior_target[None].astype(np.float32)),
            "dist": torch.from_numpy(dist[None].astype(np.float32)),
            "innovation_nn": torch.from_numpy(innovation_nn[None].astype(np.float32)),
            "fixed_local_correction": torch.from_numpy(fixed_corr[None].astype(np.float32)),
            "base_length": torch.tensor(l0, dtype=torch.float32),
            "target": torch.from_numpy(target[None].astype(np.float32)),
            "valid_mask": torch.from_numpy(valid[None]),
            "metadata": {
                "sample_idx": sample_idx,
                "prior_type": prior_type,
                "obs_day": obs_day,
                "obs_fraction": obs_fraction,
                "mask_type": mask_type,
                "noise_std": noise_std,
            },
        }


def collate_adaptive_scale_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "prior_target": torch.stack([b["prior_target"] for b in batch]),
        "dist": torch.stack([b["dist"] for b in batch]),
        "innovation_nn": torch.stack([b["innovation_nn"] for b in batch]),
        "fixed_local_correction": torch.stack([b["fixed_local_correction"] for b in batch]),
        "base_length": torch.stack([b["base_length"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch]),
        "metadata": [b["metadata"] for b in batch],
    }
