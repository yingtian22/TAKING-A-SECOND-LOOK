from __future__ import annotations

import torch


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean L1 over masked valid pixels."""
    diff = torch.abs(pred - target)
    if mask is None:
        valid = torch.isfinite(pred) & torch.isfinite(target)
        mask = valid.float()
    diff = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error over masked valid pixels."""
    diff = (pred - target) ** 2
    if mask is None:
        valid = torch.isfinite(pred) & torch.isfinite(target)
        mask = valid.float()
    diff = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom
