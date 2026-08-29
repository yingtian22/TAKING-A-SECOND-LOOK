from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from secondlook.metrics.seaice_metrics import compute_metrics, rmse
from secondlook.models.delta_unet import DeltaUNet
from secondlook.training.losses import masked_l1_loss, masked_mse_loss


@torch.no_grad()
def validate_correction(
    model: DeltaUNet,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    ocean_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Validate corrected vs base correction on random correction cases."""
    model.eval()
    base_sse, corr_sse, count = 0.0, 0.0, 0.0
    edge_base_sse, edge_corr_sse, edge_count = 0.0, 0.0, 0.0
    miz_base_sse, miz_corr_sse, miz_count = 0.0, 0.0, 0.0

    for batch in loader:
        features = batch["features"].to(device)
        base = batch["base_correction"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid_mask"].to(device).float()

        with autocast(enabled=amp):
            corrected, _ = model(features, base)

        base_np = base.float().cpu().numpy()[:, 0]
        corr_np = corrected.float().cpu().numpy()[:, 0]
        tgt_np = target.cpu().numpy()[:, 0]
        valid_np = valid.cpu().numpy()[:, 0].astype(bool)

        for i in range(base_np.shape[0]):
            vm = valid_np[i]
            if not vm.any():
                continue
            bd = base_np[i][vm] - tgt_np[i][vm]
            cd = corr_np[i][vm] - tgt_np[i][vm]
            base_sse += float(np.sum(bd**2))
            corr_sse += float(np.sum(cd**2))
            count += float(vm.sum())

            md_base = compute_metrics(
                np.where(vm, base_np[i], np.nan),
                np.where(vm, tgt_np[i], np.nan),
                ocean_mask=ocean_mask,
            )
            md_corr = compute_metrics(
                np.where(vm, corr_np[i], np.nan),
                np.where(vm, tgt_np[i], np.nan),
                ocean_mask=ocean_mask,
            )
            if np.isfinite(md_base["Edge_RMSE"]):
                edge_base_sse += md_base["Edge_RMSE"] ** 2
                edge_corr_sse += md_corr["Edge_RMSE"] ** 2
                edge_count += 1.0
            if np.isfinite(md_base["MIZ_RMSE"]):
                miz_base_sse += md_base["MIZ_RMSE"] ** 2
                miz_corr_sse += md_corr["MIZ_RMSE"] ** 2
                miz_count += 1.0

    val_base_rmse = float(np.sqrt(base_sse / max(count, 1.0)))
    val_corr_rmse = float(np.sqrt(corr_sse / max(count, 1.0)))
    val_edge = float(np.sqrt(edge_corr_sse / max(edge_count, 1.0)))
    val_miz = float(np.sqrt(miz_corr_sse / max(miz_count, 1.0)))
    improvement = (val_base_rmse - val_corr_rmse) / val_base_rmse * 100.0 if val_base_rmse > 0 else 0.0

    return {
        "val_base_RMSE": val_base_rmse,
        "val_RMSE": val_corr_rmse,
        "val_Edge_RMSE": val_edge,
        "val_MIZ_RMSE": val_miz,
        "val_improvement_vs_base_pct": improvement,
    }


def train_one_epoch(
    model: DeltaUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool,
    scaler: GradScaler,
    l1_weight: float = 1.0,
    mse_weight: float = 0.2,
) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        features = batch["features"].to(device)
        base = batch["base_correction"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid_mask"].to(device).float()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            corrected, _ = model(features, base)
            loss = l1_weight * masked_l1_loss(corrected, target, valid)
            if mse_weight > 0:
                loss = loss + mse_weight * masked_mse_loss(corrected, target, valid)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss encountered")

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


class CorrectionTrainer:
    """Trainer for base-correction-guided DeltaUNet."""

    def __init__(
        self,
        model: DeltaUNet,
        device: torch.device,
        *,
        amp: bool = True,
        l1_weight: float = 1.0,
        mse_weight: float = 0.2,
    ) -> None:
        self.model = model
        self.device = device
        self.amp = amp
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.scaler = GradScaler(enabled=amp)

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        return train_one_epoch(
            self.model,
            loader,
            optimizer,
            self.device,
            amp=self.amp,
            scaler=self.scaler,
            l1_weight=self.l1_weight,
            mse_weight=self.mse_weight,
        )

    def validate(
        self,
        loader: DataLoader,
        *,
        ocean_mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        return validate_correction(
            self.model,
            loader,
            self.device,
            amp=self.amp,
            ocean_mask=ocean_mask,
        )
