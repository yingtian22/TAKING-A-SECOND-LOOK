"""One-shot: extract 4 test samples into examples/demo_cases.npz.

Reads the original processed test NPZ and Direct U-Net prior from the
development machine. Not needed after examples/demo_cases.npz exists.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

IDX = np.array([0, 90, 180, 270], dtype=np.int32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test-npz", type=Path,
                   default=Path(r"E:\CODE\CryoCast-Diff\data2\processed\g02202_v5_2014_2020_7to7\test.npz"))
    p.add_argument("--du-prior", type=Path,
                   default=Path(r"E:\CODE\SeaIceSecondLook\outputs\stage1_openloop\direct_unet\test_predictions_direct_unet.npz"))
    p.add_argument("--ocean-mask", type=Path,
                   default=Path(r"E:\CODE\CryoCast-Diff\data2\masks\g02202_v5\ocean_mask.npy"))
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "examples" / "demo_cases.npz")
    args = p.parse_args()

    test = np.load(args.test_npz)
    du = np.load(args.du_prior)
    pred_key = "predictions" if "predictions" in du.files else du.files[0]
    ocean = np.load(args.ocean_mask)
    payload = {
        "inputs": np.asarray(test["inputs"][IDX], dtype=np.float32),
        "targets": np.asarray(test["targets"][IDX], dtype=np.float32),
        "du_prior": np.asarray(du[pred_key][IDX], dtype=np.float32),
        "ocean_mask": ocean.astype(np.uint8),
        "sample_indices": IDX,
    }
    if "input_dates" in test.files:
        payload["input_dates"] = test["input_dates"][IDX]
    if "target_dates" in test.files:
        payload["target_dates"] = test["target_dates"][IDX]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"Wrote {args.out}  samples={list(IDX)}  size={args.out.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
