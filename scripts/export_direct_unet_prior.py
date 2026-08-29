"""Export the frozen Direct U-Net open-loop prior on the test split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from secondlook.data.sic_loader import load_npz_split
from secondlook.evaluation.openloop_eval import save_prediction_npz
from secondlook.models.direct_unet import DirectForecastUNet
from secondlook.training.trainer import InMemoryOpenLoopDataset, collate_openloop_batch
from scripts.protocol import load_paths


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "checkpoints" / "direct_unet" / "best.pt")
    p.add_argument("--output", type=Path,
                   default=ROOT / "outputs" / "direct_unet" / "test_predictions_direct_unet.npz")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    paths = load_paths()
    test_npz = paths["processed_dir"] / "test.npz"
    if not test_npz.is_file():
        raise FileNotFoundError(f"Missing {test_npz}. See README data section.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = DirectForecastUNet(
        in_channels=int(cfg.get("in_channels", 7)),
        out_channels=int(cfg.get("out_channels", 7)),
        base_channels=int(cfg.get("base_channels", 48)),
        depth=int(cfg.get("depth", 3)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = load_npz_split(test_npz)
    loader = DataLoader(
        InMemoryOpenLoopDataset(data.inputs, data.targets),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_openloop_batch,
    )
    chunks = []
    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(device)
            pred = model(x)
            chunks.append(pred.float().cpu().numpy())
    pred = np.concatenate(chunks, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_prediction_npz(args.output, predictions=pred, split="test",
                        model_name="direct_unet")
    print(f"Wrote {args.output}  shape={pred.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
