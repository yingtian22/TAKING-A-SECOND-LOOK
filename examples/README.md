# Bundled demo cases

`demo_cases.npz` contains 4 official test-split samples (indices 0, 90, 180, 270):

- `inputs`, `targets`, `du_prior`: `[4, 7, 448, 304]`
- `ocean_mask`: `[448, 304]`
- `sample_indices`

This is only for `python scripts/run_demo.py`. It is **not** the 96-setting paper evaluation.
