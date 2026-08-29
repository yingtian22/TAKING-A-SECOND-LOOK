"""Frozen second-look evaluation protocol (paper / Stage 15B)."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secondlook.data.mask_resolver import resolve_ocean_mask
from secondlook.evaluation.openloop_eval import load_prediction_npz
from secondlook.observation.masks import generate_obs_mask
from secondlook.observation.perturbation import apply_observation_noise
from secondlook.utils.arrays import fill_nan

LEAD_DAY = 7
BASE_SEED = 42
L0 = {"persistence": 8.0, "direct_unet": 5.0}
NUDGING_ALPHA = 0.75
CONFIDENCE_LS = 10.0
PRIORS = ("persistence", "direct_unet")
OBS_DAYS = (3, 5)
OBS_FRACTIONS = (0.05, 0.10, 0.30)
MASK_TYPES = ("random", "edge", "stripe", "coarse")
NOISE_STDS = (0.0, 0.03)
CKPT_SCALE = {
    42: ROOT / "checkpoints" / "echo_scale" / "seed42.pt",
    43: ROOT / "checkpoints" / "echo_scale" / "seed43.pt",
    44: ROOT / "checkpoints" / "echo_scale" / "seed44.pt",
}
CKPT_DELTA = {
    42: ROOT / "checkpoints" / "echo_delta" / "seed42.pt",
    43: ROOT / "checkpoints" / "echo_delta" / "seed43.pt",
    44: ROOT / "checkpoints" / "echo_delta" / "seed44.pt",
}


def setting_seed(prior: str, obs_day: int, obs_fraction: float,
                 mask_type: str, noise_std: float) -> int:
    """Stage-4 / Stage-15B setting seed (argument order is frozen)."""
    parts = (prior, LEAD_DAY, obs_day, mask_type, obs_fraction, noise_std)
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return (BASE_SEED + int(h[:8], 16)) % (2**31)


def iter_settings():
    for obs_day in OBS_DAYS:
        for obs_fraction in OBS_FRACTIONS:
            for mask_type in MASK_TYPES:
                for noise_std in NOISE_STDS:
                    yield obs_day, obs_fraction, mask_type, noise_std


def load_paths() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))
    data_root = Path(os.environ.get("SECONDLOOK_DATA", ROOT / cfg["data_root"]))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    processed = Path(os.environ.get(
        "SECONDLOOK_PROCESSED", data_root / "processed" / "g02202_v5_2014_2020_7to7"))
    if not processed.is_absolute():
        processed = ROOT / processed if not processed.exists() else processed
    prior = Path(os.environ.get("SECONDLOOK_DU_PRIOR", ROOT / cfg["direct_unet_prior"]))
    return {"data_root": data_root, "processed_dir": processed, "du_prior": prior}


def load_test_arrays(max_samples: int | None = None) -> dict:
    paths = load_paths()
    ocean = resolve_ocean_mask(paths["data_root"], paths["processed_dir"],
                               target_shape=(448, 304))["mask"]
    arch = np.load(paths["processed_dir"] / "test.npz", mmap_mode="r")
    n = int(arch["inputs"].shape[0]) if max_samples is None else int(max_samples)
    inputs = fill_nan(np.asarray(arch["inputs"][:n], dtype=np.float32))
    targets = fill_nan(np.asarray(arch["targets"][:n], dtype=np.float32))
    if not paths["du_prior"].is_file():
        raise FileNotFoundError(
            f"Direct U-Net prior not found: {paths['du_prior']}\n"
            "Run: python scripts/export_direct_unet_prior.py")
    du = fill_nan(np.asarray(
        load_prediction_npz(paths["du_prior"])["predictions"][:n], dtype=np.float32))
    return {"ocean_mask": ocean, "inputs": inputs, "targets": targets,
            "du_prior": du, "n_samples": n}


def load_demo_arrays() -> dict:
    demo = ROOT / "examples" / "demo_cases.npz"
    z = np.load(demo)
    return {
        "ocean_mask": z["ocean_mask"].astype(bool),
        "inputs": fill_nan(z["inputs"].astype(np.float32)),
        "targets": fill_nan(z["targets"].astype(np.float32)),
        "du_prior": fill_nan(z["du_prior"].astype(np.float32)),
        "n_samples": int(z["inputs"].shape[0]),
        "sample_indices": z["sample_indices"],
    }


def build_setting_cases(res: dict, prior_type: str, obs_day: int,
                        obs_fraction: float, mask_type: str, noise_std: float):
    inputs, targets, du = res["inputs"], res["targets"], res["du_prior"]
    ocean_mask = res["ocean_mask"]
    ct = inputs[:, -1]
    if prior_type == "persistence":
        p_tgt, p_obs = ct, ct
    else:
        p_tgt = du[:, LEAD_DAY - 1]
        p_obs = du[:, obs_day - 1]
    obs_full = targets[:, obs_day - 1]
    rng = np.random.default_rng(
        setting_seed(prior_type, obs_day, obs_fraction, mask_type, noise_std))
    edge_prior = p_obs if mask_type == "edge" else None
    obs_mask = generate_obs_mask(obs_full.shape, mask_type, obs_fraction, rng,
                                 prior=edge_prior, ocean_mask=ocean_mask)
    sparse_obs = apply_observation_noise(obs_full, obs_mask, noise_std, rng)
    return {
        "prior": prior_type, "obs_day": obs_day, "obs_fraction": obs_fraction,
        "mask_type": mask_type, "noise_std": noise_std,
        "setting_seed": setting_seed(prior_type, obs_day, obs_fraction,
                                     mask_type, noise_std),
        "prior_target": p_tgt, "prior_obs_day": p_obs,
        "target_lead": targets[:, LEAD_DAY - 1],
        "obs_mask": obs_mask, "sparse_obs": sparse_obs,
        "ocean_mask": ocean_mask,
    }
