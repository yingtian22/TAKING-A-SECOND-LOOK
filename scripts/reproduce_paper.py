"""Reproduce the paper's 96-setting SETTING_MACRO_AVERAGE for core methods.

Requires the processed test NPZ and the Direct U-Net prior
(see README). Default: Prior / Residual / Nudging / Fixed / ECHO-Scale /
ECHO-Delta. ECHO uses training seed 42 unless --all-seeds.

This is the same frozen protocol as the paper (eval seed 42, 96 x 360).
Full 96-setting evaluation is slow (hours). Use --quick for a 2-setting check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
    predict_nudging,
    predict_prior,
    predict_residual,
)
from scripts.protocol import (
    CKPT_DELTA,
    CKPT_SCALE,
    PRIORS,
    build_setting_cases,
    iter_settings,
    load_test_arrays,
)

PAPER = {
    "Original Prior": 0.081614,
    "Residual Interpolation": 0.071722,
    "Nudging": 0.069505,
    "Fixed Propagation": 0.068966,
    "ECHO-Scale": 0.067332,   # seed 42
    "ECHO-Delta": 0.062201,   # seed 42
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="2 settings x 8 samples (sanity check, not paper table)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   choices=[42, 43, 44])
    p.add_argument("--methods", nargs="+",
                   default=["prior", "residual", "nudging", "fixed", "scale", "delta"])
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "outputs" / "paper_eval")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    max_samples = 8 if args.quick else args.max_samples
    res = load_test_arrays(max_samples=max_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  n={res['n_samples']}  quick={args.quick}")

    scale_nets = {s: load_scale_net(CKPT_SCALE[s], device)
                  for s in args.seeds if "scale" in args.methods}
    delta_nets = {s: load_delta_net(CKPT_DELTA[s], device)
                  for s in args.seeds if "delta" in args.methods}

    settings = list(iter_settings())
    if args.quick:
        settings = [settings[0], settings[16]]  # one random, one later setting

    rows = []
    si = 0
    for prior in PRIORS:
        for obs_day, frac, geom, noise in settings:
            sc = build_setting_cases(res, prior, obs_day, frac, geom, noise)
            tgt, ocean = sc["target_lead"], sc["ocean_mask"]
            jobs = []
            if "prior" in args.methods:
                jobs.append(("Original Prior", None, predict_prior(sc)))
            if "residual" in args.methods:
                jobs.append(("Residual Interpolation", None, predict_residual(sc)))
            if "nudging" in args.methods:
                jobs.append(("Nudging", None, predict_nudging(sc)))
            if "fixed" in args.methods:
                jobs.append(("Fixed Propagation", None, predict_fixed(sc)))
            for seed, net in scale_nets.items():
                jobs.append(("ECHO-Scale", seed, predict_echo_scale(sc, net, device)))
            for seed, net in delta_nets.items():
                jobs.append(("ECHO-Delta", seed, predict_echo_delta(sc, net, device)))
            for name, seed, pred in jobs:
                m = compute_metrics(pred, tgt, ocean_mask=ocean)
                rows.append({
                    "setting_index": si, "method": name, "model_seed": seed,
                    "prior": prior, "obs_day": obs_day, "obs_fraction": frac,
                    "mask_type": geom, "noise_std": noise,
                    "num_samples": res["n_samples"], **m,
                })
            si += 1
            print(f"[{si}] {prior} d{obs_day} f{frac} {geom} n{noise}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "setting_results.csv", index=False)
    summary = (df.groupby(["method", "model_seed"], dropna=False)["RMSE"]
               .mean().reset_index())
    summary.to_csv(args.output_dir / "macro_average.csv", index=False)
    print("\nSETTING_MACRO_AVERAGE RMSE")
    print(summary.to_string(index=False))
    if not args.quick and res["n_samples"] == 360 and set(args.seeds) == {42}:
        print("\nVs paper seed-42 reference:")
        for _, r in summary.iterrows():
            name = r["method"]
            if name in PAPER:
                got, exp = float(r["RMSE"]), PAPER[name]
                ok = abs(got - exp) < 5e-5
                print(f"  {name:24s}  got={got:.6f}  paper={exp:.6f}  "
                      f"{'OK' if ok else 'CHECK'}")
    meta = {"quick": args.quick, "n_samples": res["n_samples"],
            "n_settings_per_prior": len(settings), "seconds": time.time() - t0,
            "device": str(device), "seeds": args.seeds}
    (args.output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
