# ECHO: Taking a Second Look

Code and checkpoints for

> **Taking a Second Look: Correcting Sea Ice Forecasts with Sparse Observations**

ECHO revises a **frozen** 7-day sea-ice concentration (SIC) forecast using **sparse intermediate observations**. Two variants:

- **ECHO-Scale** — learns a bounded per-pixel propagation length (76.9K params)
- **ECHO-Delta** — learns a bounded residual around Fixed Propagation (7.85M params)

This folder is a **standalone reproduction pack**. It is copied from the research tree; the original project was not modified.

## What you can run

| Command | Data needed | What it matches |
|---|---|---|
| `python scripts/run_demo.py` | **Nothing extra** (bundled `examples/`) | Smoke test on 4 samples |
| `python scripts/reproduce_paper.py --quick` | `test.npz` + Direct U-Net prior | 2 settings × 8 samples |
| `python scripts/reproduce_paper.py` | full test split + prior | Paper **seed-42** 96-setting table for Prior / Residual / Nudging / Fixed / ECHO |

Paper Table 1 reports ECHO as the **mean of training seeds 42/43/44**. Seed-42 paired numbers are the official W/T/L protocol.

Canonical Table 1 numbers (do not mix the two aggregations):

| Method | RMSE |
|---|---:|
| Original Prior | 0.08161 |
| Residual Interpolation | 0.07172 |
| Nudging | 0.06950 |
| Fixed Propagation | 0.06897 |
| Localized OI | 0.06825 |
| EnKF-PertObs | 0.07989 |
| Adapted 4DVarNet | 0.06909 |
| **ECHO-Scale** (3-seed) | **0.06735 ± 0.00003** |
| **ECHO-Delta** (3-seed) | **0.06249 ± 0.00028** |

Seed-42 reference: ECHO-Scale **0.067332**, ECHO-Delta **0.062201**.  
See `paper_results/table1_canonical.csv`.

EnKF-PertObs and adapted 4DVarNet are **not** re-run in this pack (task-adapted baselines; see the paper). Localized OI is omitted from the default script (slow CPU local Kalman). The included methods are enough to reproduce the ECHO vs Fixed / Residual / Nudging claims.

## Environment

```powershell
conda create -n echo-seaice python=3.10
conda activate echo-seaice
pip install -r requirements.txt
```

`torch` should match your CUDA build. The paper numbers were produced with PyTorch 2.12 + CUDA 12.6 on an RTX 4090. CPU works for the demo.

```powershell
cd E:\CODE\ECHO-SeaIce
python scripts/verify_checkpoints.py
python scripts/run_demo.py
```

Expected demo behavior: ECHO-Delta RMSE &lt; ECHO-Scale RMSE &lt; Fixed &lt; Original Prior on the bundled cases (small-N, not the paper average).

## Data (not in this repo)

### Product

[NOAA/NSIDC Sea Ice Concentration CDR, G02202 Version 5](https://nsidc.org/data/g02202/versions/5), northern hemisphere daily 25 km (`sic_psn25_*_F17_v05r00.nc`). DOI: [10.7265/rjzb-pf78](https://doi.org/10.7265/rjzb-pf78).

Download daily files for **2014–2020** from NSIDC (Earthdata login).

### Processed NPZ the code expects

Sliding 7→7 windows on the 448×304 polar-stereographic grid, year split:

| Split | Years | Samples | File |
|---|---|---:|---|
| train | 2014–2018 | 1819 | `train.npz` |
| val | 2019 | 365 | `val.npz` |
| test | 2020 | 360 | `test.npz` |

Each NPZ must contain:

- `inputs`  `float32` `[N, 7, 448, 304]` — past 7 days SIC in `[0, 1]` (NaN on land)
- `targets` `float32` `[N, 7, 448, 304]` — next 7 days SIC
- optional: `input_dates`, `target_dates`

**Put the files here:**

```text
data/processed/g02202_v5_2014_2020_7to7/train.npz
data/processed/g02202_v5_2014_2020_7to7/val.npz
data/processed/g02202_v5_2014_2020_7to7/test.npz
```

`data/masks/g02202_v5/ocean_mask.npy` and `metadata.json` are already included.

If your processed tree lives elsewhere:

```powershell
$env:SECONDLOOK_DATA = "D:\path\to\data"
$env:SECONDLOOK_PROCESSED = "D:\path\to\data\processed\g02202_v5_2014_2020_7to7"
```

### Direct U-Net prior (required for the full paper eval)

The Direct U-Net prior is **not** the raw dataset. After `test.npz` is in place:

```powershell
python scripts/export_direct_unet_prior.py
```

This writes `outputs/direct_unet/test_predictions_direct_unet.npz` using `checkpoints/direct_unet/best.pt`.

Alternatively, point to an existing prior:

```powershell
$env:SECONDLOOK_DU_PRIOR = "E:\CODE\SeaIceSecondLook\outputs\stage1_openloop\direct_unet\test_predictions_direct_unet.npz"
```

## Reproduce the paper (full test)

```powershell
# sanity (minutes)
python scripts/reproduce_paper.py --quick

# seed-42 formal 96 x 360 (hours on GPU)
python scripts/reproduce_paper.py --seeds 42

# three-seed ECHO means (matches Table 1 ECHO rows; much longer)
python scripts/reproduce_paper.py --seeds 42 43 44 --methods scale delta
```

Outputs: `outputs/paper_eval/setting_results.csv`, `macro_average.csv`.

Protocol (frozen):

- lead 7; obs days 3/5; fractions 5/10/30%; geometries random/edge/stripe/coarse; σ ∈ {0, 0.03}
- priors: persistence and Direct U-Net
- eval setting seed 42
- aggregation: unweighted mean of 96 setting RMSEs
- Fixed \(L_0\): persistence 8, Direct U-Net 5 (validation-selected)
- Nudging α = 0.75

A run with 360 test samples and seed 42 should match the seed-42 RMSE column to about \(5\times10^{-5}\). Tiny float differences across GPU/CPU are possible; setting-level W/T/L vs Fixed should stay 96/0/0.

## Checkpoints

| File | Paper name | Training seed |
|---|---|---|
| `checkpoints/echo_scale/seed{42,43,44}.pt` | ECHO-Scale | 42 / 43 / 44 |
| `checkpoints/echo_delta/seed{42,43,44}.pt` | ECHO-Delta | 42 / 43 / 44 |
| `checkpoints/direct_unet/best.pt` | Direct U-Net prior | 42 |

SHA256 of the Scale checkpoints is checked against the formal Stage 15B freeze (`python scripts/verify_checkpoints.py`).

## Demo examples

`examples/demo_cases.npz` holds **4 test samples** (indices 0, 90, 180, 270) plus the ocean mask and the matching Direct U-Net prior frames. It is enough to run `run_demo.py` without downloading G02202.

## Layout

```text
secondlook/          library (models, baselines, metrics, masks)
scripts/             demo + paper eval
checkpoints/         frozen weights
examples/            4-sample smoke pack
paper_results/       canonical Table 1
data/                mask + metadata; NPZ not included
```

## License / data terms

Code is released for research reproduction. G02202 is an NSIDC product — follow [NSIDC data use policy](https://nsidc.org/data/user-resources/data-use-and-copyright). Do not redistribute the daily NetCDF or the processed NPZ if your license does not allow it.
