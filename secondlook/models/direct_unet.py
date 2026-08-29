from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, norm: str = "group") -> None:
        super().__init__()
        if norm == "batch":
            norm_layer = lambda c: nn.BatchNorm2d(c)  # noqa: E731
        else:
            norm_layer = lambda c: nn.GroupNorm(min(8, c), c)  # noqa: E731

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm_layer(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            norm_layer(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, norm: str = "group") -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, norm=norm)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        return h, self.pool(h)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, *, norm: str = "group") -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DirectForecastUNet(nn.Module):
    """Open-loop 7-day SIC forecast from 7-day input history."""

    def __init__(
        self,
        *,
        in_channels: int = 7,
        out_channels: int = 7,
        base_channels: int = 48,
        depth: int = 3,
        norm: str = "group",
    ) -> None:
        super().__init__()
        self.model_config = {
            "model_type": "DirectForecastUNet",
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_channels": base_channels,
            "depth": depth,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T_in, H, W] -> pred_seq: [B, T_out, H, W] in [0, 1]."""
        skips: list[torch.Tensor] = []
        h = self.stem(x)
        skips.append(h)
        for down in self.downs:
            h, pooled = down(h)
            skips.append(h)
            h = pooled
        h = self.bottleneck(h)
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip)
        return torch.sigmoid(self.head(h))
