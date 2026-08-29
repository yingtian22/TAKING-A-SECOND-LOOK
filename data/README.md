# Data layout

Do **not** commit `train.npz` / `val.npz` / `test.npz`. Place them here after download/processing:

```text
data/
  masks/g02202_v5/ocean_mask.npy          # already in this repo
  processed/g02202_v5_2014_2020_7to7/
    metadata.json                         # already in this repo
    train.npz                             # 1819 samples  [N,7,448,304]
    val.npz                               # 365 samples
    test.npz                              # 360 samples   (required for paper eval)
```

See the top-level README for NSIDC G02202 download and NPZ format.
