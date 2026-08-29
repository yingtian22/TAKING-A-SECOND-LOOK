from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from secondlook.models.direct_unet import ConvBlock, DownBlock, UpBlock


class DeltaUNet(nn.Module):
    """Lightweight U-Net predicting delta correction on top of base correction."""

    def __init__(
        self,
        *,
        in_channels: int = 8,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        residual_scale: float = 0.25,
        norm: str = "group",
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.model_config = {
            "model_type": "DeltaUNet",
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_channels": base_channels,
            "depth": depth,
            "residual_scale": self.residual_scale,
            "norm": norm,
        }

        ch = base_channels
        self.stem = ConvBlock(in_channels, ch, norm=norm)
        self.downs = nn.ModuleList()
        enc_chs = [ch]
        cur = ch
        for _ in range(depth):
            nxt = cur * 2
            self.downs.append(DownBlock(cur, nxt, norm=norm))
            cur = nxt
            enc_chs.append(cur)

        self.bottleneck = ConvBlock(cur, cur * 2, norm=norm)
        dec_in = cur * 2

        self.ups = nn.ModuleList()
        for skip_ch in reversed(enc_chs):
            out_ch = skip_ch
            self.ups.append(UpBlock(dec_in, skip_ch, out_ch, norm=norm))
            dec_in = out_ch

        self.head = nn.Conv2d(dec_in, out_channels, 1)

    def forward(
        self,
        features: torch.Tensor,
        base_correction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        features: [B, C, H, W]
        base_correction: [B, 1, H, W]
        returns (corrected, raw_delta)
        """
        skips: list[torch.Tensor] = []
        h = self.stem(features)
        skips.append(h)
        for down in self.downs:
            h, pooled = down(h)
            skips.append(h)
            h = pooled
        h = self.bottleneck(h)
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip)
        raw = self.head(h)
        delta = self.residual_scale * torch.tanh(raw)
        corrected = torch.clamp(base_correction + delta, 0.0, 1.0)
        return corrected, delta
