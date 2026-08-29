"""Stage 8E: Adaptive-Propagation-Guided DeltaUNet integration helpers.

Pipeline:
    prior -> frozen AdaptiveScaleNet -> adaptive propagation base
          -> V1 DeltaUNet (8 feature slots; only slots 1/2 use the adaptive
             base) -> final correction

The ScaleNet is frozen (eval, requires_grad=False, no optimizer). The
DeltaUNet is byte-identical to V1 (architecture, residual_scale=0.25, seed-42
initialization). No joint fine-tuning, no anisotropy, no bound changes.
"""

from __future__ import annotations

import torch
from torch import nn

from secondlook.baselines.adaptive_propagation import (
    adaptive_nearest_innovation_propagation,
    bounded_length_field,
)


@torch.no_grad()
def adaptive_base_from_batch(
    scale_net: nn.Module,
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen ScaleNet -> L_theta -> adaptive propagation base.

    Returns (adaptive_base [N,1,H,W], length_field [N,1,H,W]).
    """
    scale_net.eval()
    raw = scale_net(batch["scale_features"].to(device))
    L0 = batch["base_length"].to(device).view(-1, 1, 1, 1)
    L_field = bounded_length_field(raw, 1.0) * L0
    adaptive_base, _ = adaptive_nearest_innovation_propagation(
        prior_target=batch["prior_target"].to(device),
        dist=batch["dist"].to(device),
        innovation_nn=batch["innovation_nn"].to(device),
        length_field=L_field,
    )
    return adaptive_base, L_field


def build_delta_features(
    batch: dict,
    adaptive_base: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """V1's 8 feature slots; ONLY slots 1 (base_correction) and 2 (base_delta)
    use the adaptive base. Slots 0,3,4,5,6,7 are identical to V1.
    """
    prior_target = batch["prior_target"].to(device)
    feats = torch.cat(
        [
            prior_target,                       # 0 prior_target
            adaptive_base,                      # 1 base_correction (ADAPTIVE)
            adaptive_base - prior_target,       # 2 base_delta (ADAPTIVE)
            batch["prior_obs_day"].to(device),  # 3 prior_obs_day
            batch["sparse_canvas"].to(device),  # 4 sparse_obs
            batch["obs_mask"].to(device),       # 5 mask
            batch["residual_canvas"].to(device),  # 6 residual_obs
            batch["confidence"].to(device),     # 7 confidence
        ],
        dim=1,
    )
    return torch.nan_to_num(feats, nan=0.0)
