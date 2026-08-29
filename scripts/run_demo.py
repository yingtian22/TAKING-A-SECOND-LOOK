"""Run ECHO on the bundled demo cases (no full dataset required)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from secondlook.metrics.seaice_metrics import compute_metrics
from scripts.eval_methods import (
    load_delta_net,
    load_scale_net,
    predict_echo_delta,
    predict_echo_scale,
    predict_fixed,
    predict_prior,
)
from scripts.protocol import CKPT_DELTA, CKPT_SCALE, build_setting_cases, load_demo_arrays

DEMO_SETTINGS = (
    ("persistence", 5, 0.10, "random", 0.0),
    ("direct_unet", 5, 0.10, "edge", 0.03),
)


def main() -> int:
    res = load_demo_arrays()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale = load_scale_net(CKPT_SCALE[42], device)
    delta = load_delta_net(CKPT_DELTA[42], device)
    ocean = res["ocean_mask"]
    rows = []
    for prior, day, frac, geom, noise in DEMO_SETTINGS:
        sc = build_setting_cases(res, prior, day, frac, geom, noise)
        preds = {
            "Original Prior": predict_prior(sc),
            "Fixed Propagation": predict_fixed(sc),
            "ECHO-Scale": predict_echo_scale(sc, scale, device),
            "ECHO-Delta": predict_echo_delta(sc, delta, device),
        }
        for name, pred in preds.items():
            m = compute_metrics(pred, sc["target_lead"], ocean_mask=ocean)
            rows.append({
                "prior": prior, "obs_day": day, "obs_fraction": frac,
                "mask_type": geom, "noise_std": noise, "method": name,
                "n": res["n_samples"], **m,
            })
            print(f"{name:22s}  {prior:12s}  RMSE={m['RMSE']:.5f}")
    out = ROOT / "outputs" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "demo_metrics.csv", index=False)
    print(f"\nWrote {out / 'demo_metrics.csv'}")
    print("This is a smoke demo on 4 bundled samples, not the 96-setting paper table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
