from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from secondlook.metrics.seaice_metrics import compute_metrics, rmse
from secondlook.models.direct_unet import DirectForecastUNet
from secondlook.training.losses import masked_l1_loss, masked_mse_loss
from secondlook.utils.arrays import build_valid_mask, fill_nan


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_model(
    model: DirectForecastUNet,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    n_leads: int | None = None
    sse = None
    count = None
    edge_sse = 0.0
    edge_count = 0.0
    miz_sse = 0.0
    miz_count = 0.0

    for batch in loader:
        x = batch["inputs"].to(device)
        y = batch["targets"].to(device)
        valid = batch["valid_mask"].to(device)
        with autocast(enabled=amp):
            pred = model(x)
        pred_np = pred.float().cpu().numpy()
        tgt_np = y.cpu().numpy()
        valid_np = valid.cpu().numpy().astype(bool)

        if n_leads is None:
            n_leads = pred_np.shape[1]
            sse = np.zeros(n_leads, dtype=np.float64)
            count = np.zeros(n_leads, dtype=np.float64)

        for li in range(n_leads):
            vm = valid_np[:, li]
            diff = pred_np[:, li] - tgt_np[:, li]
            sse[li] += float(np.sum(diff[vm] ** 2))
            count[li] += float(vm.sum())

        md7 = compute_metrics(
            np.where(valid_np[:, -1], pred_np[:, -1], np.nan),
            np.where(valid_np[:, -1], tgt_np[:, -1], np.nan),
        )
        edge_rmse = md7["Edge_RMSE"]
        miz_rmse = md7["MIZ_RMSE"]
        if np.isfinite(edge_rmse):
            n_batch = pred_np.shape[0]
            edge_sse += edge_rmse**2 * n_batch
            edge_count += n_batch
        if np.isfinite(miz_rmse):
            n_batch = pred_np.shape[0]
            miz_sse += miz_rmse**2 * n_batch
            miz_count += n_batch

    assert n_leads is not None and sse is not None and count is not None
    lead_metrics: dict[str, float] = {}
    for li in range(n_leads):
        d = li + 1
        lead_metrics[f"RMSE_lead{d}"] = float(np.sqrt(sse[li] / max(count[li], 1.0)))

    mean_rmse = float(np.mean([lead_metrics[f"RMSE_lead{i+1}"] for i in range(n_leads)]))
    val_edge = float(np.sqrt(edge_sse / max(edge_count, 1.0)))
    val_miz = float(np.sqrt(miz_sse / max(miz_count, 1.0)))
    return {
        **lead_metrics,
        "val_RMSE_mean": mean_rmse,
        "val_RMSE_lead7": lead_metrics["RMSE_lead7"],
        "val_Edge_RMSE_lead7": val_edge,
        "val_MIZ_RMSE_lead7": val_miz,
    }


def train_one_epoch(
    model: DirectForecastUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool,
    scaler: GradScaler,
    mse_weight: float = 0.2,
) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x = batch["inputs"].to(device)
        y = batch["targets"].to(device)
        valid = batch["valid_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            pred = model(x)
            loss = masked_l1_loss(pred, y, valid)
            if mse_weight > 0:
                loss = loss + mse_weight * masked_mse_loss(pred, y, valid)
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


class OpenLoopTrainer:
    """Minimal trainer wrapper for DirectForecastUNet."""

    def __init__(
        self,
        model: DirectForecastUNet,
        device: torch.device,
        *,
        amp: bool = True,
        mse_weight: float = 0.2,
    ) -> None:
        self.model = model
        self.device = device
        self.amp = amp
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
            mse_weight=self.mse_weight,
        )

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        return evaluate_model(self.model, loader, self.device, amp=self.amp)


class MmapOpenLoopDataset(Dataset):
    """Memory-efficient open-loop dataset using mmap NPZ arrays."""
    def __init__(
        self,
        npz_path: str,
        *,
        ocean_mask: np.ndarray | None = None,
        max_samples: int | None = None,
    ) -> None:
        archive = np.load(npz_path, mmap_mode="r", allow_pickle=True)
        self.inputs = archive["inputs"]
        self.targets = archive["targets"]
        self.n = int(self.inputs.shape[0])
        if max_samples is not None:
            self.n = min(self.n, int(max_samples))
        self.ocean_mask = ocean_mask

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        x = fill_nan(np.asarray(self.inputs[idx], dtype=np.float32))
        y = fill_nan(np.asarray(self.targets[idx], dtype=np.float32))
        valid = build_valid_mask(x[None], y[None], ocean_mask=self.ocean_mask)[0]
        return {
            "inputs": torch.from_numpy(x),
            "targets": torch.from_numpy(y),
            "valid_mask": torch.from_numpy(valid),
        }


class InMemoryOpenLoopDataset(Dataset):
    """In-memory open-loop dataset for small splits."""

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        *,
        ocean_mask: np.ndarray | None = None,
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        self.ocean_mask = ocean_mask

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        x = fill_nan(self.inputs[idx])
        y = fill_nan(self.targets[idx])
        valid = build_valid_mask(x[None], y[None], ocean_mask=self.ocean_mask)[0]
        return {
            "inputs": torch.from_numpy(x),
            "targets": torch.from_numpy(y),
            "valid_mask": torch.from_numpy(valid),
        }


def collate_openloop_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "inputs": torch.stack([b["inputs"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch]),
    }
