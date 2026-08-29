"""Stage 8E dataset: identical sampling distribution to Stage 3's
CorrectionSampleDataset (same RNG call sequence), returning the components
needed to build (a) the frozen ScaleNet's 8 input channels and (b) the
V1-style 8 DeltaUNet feature slots in which ONLY slots 1 (base_correction)
and 2 (base_delta) are replaced by the adaptive propagation base.

The frozen ScaleNet forward and the adaptive propagation are performed in the
training/eval script (on device, under torch.no_grad); this dataset only
prepares numpy-side fields. Target is used ONLY in the loss/metrics.

V1 code is not modified.
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


class AdaptiveGuidedDataset(CorrectionSampleDataset):
    """Same case sampling as CorrectionSampleDataset; Stage-8E outputs."""

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
        ).astype(np.float32)
        dist, innovation_nn = nearest_innovation_fields(
            prior_obs_day=prior_obs_day,
            sparse_obs=sparse_obs,
            obs_mask=obs_mask,
            ocean_mask=self.ocean_mask,
        )
        dist_safe = np.where(np.isfinite(dist), dist, 1e4).astype(np.float32)
        scale_features = build_scale_features(
            prior_target=prior_target,
            prior_obs_day=prior_obs_day,
            sparse_obs=sparse_obs,
            obs_mask=obs_mask,
            fixed_local_correction=fixed_corr,
            dist=dist_safe,
            ocean_mask=self.ocean_mask,
            confidence_length_scale=self.confidence_length_scale,
        )

        # V1-style slot components (slots 0,3,4,5,6,7 identical to V1)
        sparse_canvas = np.where(obs_mask, sparse_obs, 0.0).astype(np.float32)
        residual_canvas = np.zeros_like(prior_obs_day, dtype=np.float32)
        mi = obs_mask.astype(bool)
        residual_canvas[mi] = sparse_obs[mi] - prior_obs_day[mi]
        confidence = np.exp(-dist_safe / max(self.confidence_length_scale, 1e-6)).astype(np.float32)

        valid = np.isfinite(target)
        if self.ocean_mask is not None:
            valid = valid & self.ocean_mask.astype(bool)

        def _t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(a[None].astype(np.float32))

        return {
            "scale_features": torch.from_numpy(scale_features),
            "prior_target": _t(prior_target),
            "prior_obs_day": _t(prior_obs_day),
            "sparse_canvas": _t(sparse_canvas),
            "obs_mask": _t(obs_mask.astype(np.float32)),
            "residual_canvas": _t(residual_canvas),
            "confidence": _t(confidence),
            "dist": _t(dist),
            "innovation_nn": _t(innovation_nn),
            "fixed_local_correction": _t(fixed_corr),
            "base_length": torch.tensor(l0, dtype=torch.float32),
            "target": _t(target),
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


def collate_adaptive_guided_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("scale_features", "prior_target", "prior_obs_day", "sparse_canvas",
            "obs_mask", "residual_canvas", "confidence", "dist", "innovation_nn",
            "fixed_local_correction", "base_length", "target", "valid_mask")
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["metadata"] = [b["metadata"] for b in batch]
    return out
