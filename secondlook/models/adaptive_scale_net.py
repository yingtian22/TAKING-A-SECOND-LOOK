"""Stage 8B: AdaptiveScaleNet — lightweight per-pixel raw scale field predictor.

The network predicts s_theta(x); the bounded length field is
L(x) = L0 * exp(ln(2) * tanh(s_theta(x)))  in  [0.5 * L0, 2.0 * L0].

The final 1x1 head is zero-initialized so that at initialization
s_theta == 0, L == L0, and the adaptive correction degenerates exactly to the
frozen `local_oi` baseline. Training therefore learns local *deviations* from
the existing fixed propagation, not a new assimilation algorithm.

Deliberately NOT included in the input: residual_obs (proven redundant in the
Stage 7 ablation, RESIDUAL_OBS_REDUNDANT_ON_TEST) and any gradient orientation
channels (Stage 8A found no positive signal for fixed edge-tangent anisotropy).
"""

from __future__ import annotations

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class AdaptiveScaleNet(nn.Module):
    """8-channel input -> single raw scale map. ~77k parameters."""

    def __init__(self, in_channels: int = 8, hidden_channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            _ResidualBlock(hidden_channels, dilation=1),
            _ResidualBlock(hidden_channels, dilation=2),
            _ResidualBlock(hidden_channels, dilation=4),
            _ResidualBlock(hidden_channels, dilation=8),
        )
        self.head = nn.Conv2d(hidden_channels, 1, 1)
        # Zero-init: raw scale == 0 everywhere => L == L0 (exact local_oi).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        return self.head(h)
