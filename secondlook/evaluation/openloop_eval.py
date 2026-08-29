from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from secondlook.data.sic_loader import SICSplit, load_npz_split
from secondlook.metrics.seaice_metrics import compute_metrics
from secondlook.utils.arrays import fill_nan


def persistence_forecast(inputs: np.ndarray) -> np.ndarray:
    """
    Persistence prior: repeat last input day for all 7 future lead days.

    inputs: [N, 7, H, W] -> predictions: [N, 7, H, W]
    """
    last = np.asarray(inputs[:, -1], dtype=np.float32)
    n_leads = int(inputs.shape[1])
    return np.repeat(last[:, np.newaxis, :, :], n_leads, axis=1)


def evaluate_predictions_by_lead(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    ocean_mask: np.ndarray | None = None,
    lead_days: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Compute metrics for each lead day."""
    n_leads = predictions.shape[1]
    if lead_days is None:
        lead_days = list(range(1, n_leads + 1))

    rows: list[dict[str, Any]] = []
    for lead in lead_days:
        idx = lead - 1
        pred = fill_nan(predictions[:, idx])
        tgt = fill_nan(targets[:, idx])
        metrics = compute_metrics(pred, tgt, ocean_mask=ocean_mask)
        rows.append({"Lead": lead, **metrics})
    return rows


def save_prediction_npz(
    path: str | Path,
    *,
    predictions: np.ndarray,
    split: str,
    model_name: str,
    target_dates: np.ndarray | None = None,
) -> Path:
    """Save open-loop prior predictions to NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "predictions": predictions.astype(np.float32),
        "model_name": np.array(model_name),
        "split": np.array(split),
        "lead_days": np.array(list(range(1, predictions.shape[1] + 1)), dtype=np.int32),
    }
    if target_dates is not None:
        payload["target_dates"] = target_dates
    np.savez_compressed(path, **payload)
    return path.resolve()


def load_prediction_npz(path: str | Path) -> dict[str, Any]:
    """Load saved prior predictions."""
    with np.load(path, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    if "predictions" in out:
        out["predictions"] = np.asarray(out["predictions"], dtype=np.float32)
    if "model_name" in out:
        out["model_name"] = str(np.asarray(out["model_name"]).item())
    if "split" in out:
        out["split"] = str(np.asarray(out["split"]).item())
    return out


def load_split_targets(
    processed_dir: str | Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load targets and optional target_dates for a split."""
    fname = {"train": "train.npz", "val": "val.npz", "test": "test.npz"}.get(split, f"{split}.npz")
    data = load_npz_split(Path(processed_dir) / fname)
    return data.targets, data.target_dates
