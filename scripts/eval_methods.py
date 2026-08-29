"""Inference for paper methods (Fixed / ECHO-Scale / ECHO-Delta / simple baselines)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secondlook.baselines.adaptive_propagation import (
    build_scale_features,
    nearest_innovation_fields,
)
from secondlook.baselines.observation_baselines import local_oi, nudging, residual_interp
from secondlook.datasets.correction_dataset import _build_features
from secondlook.models.adaptive_scale_net import AdaptiveScaleNet
from secondlook.models.delta_unet import DeltaUNet

from scripts.protocol import CONFIDENCE_LS, L0


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def predict_prior(sc: dict) -> np.ndarray:
    return clip01(sc["prior_target"])


def predict_residual(sc: dict) -> np.ndarray:
    return residual_interp(
        sc["prior_target"], sc["prior_obs_day"], sc["sparse_obs"],
        sc["obs_mask"], ocean_mask=sc["ocean_mask"])


def predict_nudging(sc: dict, alpha: float = 0.75) -> np.ndarray:
    return nudging(
        sc["prior_target"], sc["prior_obs_day"], sc["sparse_obs"],
        sc["obs_mask"], alpha=alpha, ocean_mask=sc["ocean_mask"])


def predict_fixed(sc: dict) -> np.ndarray:
    return local_oi(
        sc["prior_target"], sc["prior_obs_day"], sc["sparse_obs"],
        sc["obs_mask"], length_scale=L0[sc["prior"]],
        ocean_mask=sc["ocean_mask"])


def load_scale_net(ckpt_path: Path, device: torch.device) -> AdaptiveScaleNet:
    model = AdaptiveScaleNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_delta_net(ckpt_path: Path, device: torch.device) -> DeltaUNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = DeltaUNet(
        in_channels=int(cfg.get("in_channels", 8)),
        out_channels=int(cfg.get("out_channels", 1)),
        base_channels=int(cfg.get("base_channels", 32)),
        depth=int(cfg.get("depth", 3)),
        residual_scale=float(cfg.get("residual_scale", 0.25)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_echo_scale(sc: dict, model: AdaptiveScaleNet, device: torch.device) -> np.ndarray:
    ocean = sc["ocean_mask"]
    L0p = float(L0[sc["prior"]])
    n = sc["prior_target"].shape[0]
    out = np.empty_like(sc["prior_target"], dtype=np.float32)
    for i in range(n):
        pt, po = sc["prior_target"][i], sc["prior_obs_day"][i]
        so, om = sc["sparse_obs"][i], sc["obs_mask"][i]
        fixed = np.asarray(local_oi(pt, po, so, om, length_scale=L0p, ocean_mask=ocean),
                           dtype=np.float32)
        dist, innov = nearest_innovation_fields(
            prior_obs_day=po, sparse_obs=so, obs_mask=om, ocean_mask=ocean)
        dist_safe = np.where(np.isfinite(dist), dist, 1e4).astype(np.float32)
        feats = build_scale_features(
            prior_target=pt, prior_obs_day=po, sparse_obs=so, obs_mask=om,
            fixed_local_correction=fixed, dist=dist_safe,
            ocean_mask=ocean, confidence_length_scale=CONFIDENCE_LS)
        raw = model(torch.from_numpy(feats[None]).to(device))
        L_field = (L0p * torch.exp(np.log(2.0) * torch.tanh(raw)))[0, 0].cpu().numpy()
        out[i] = clip01(pt + np.exp(-dist_safe / L_field) * innov)
    return out


@torch.no_grad()
def predict_echo_delta(sc: dict, model: DeltaUNet, device: torch.device,
                       batch_size: int = 4) -> np.ndarray:
    ocean = sc["ocean_mask"]
    L0p = float(L0[sc["prior"]])
    base = predict_fixed(sc)
    n = sc["prior_target"].shape[0]
    out = np.empty_like(sc["prior_target"], dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feats = [
            _build_features(
                prior_target=sc["prior_target"][i],
                base_correction=base[i],
                prior_obs_day=sc["prior_obs_day"][i],
                sparse_obs=sc["sparse_obs"][i],
                obs_mask=sc["obs_mask"][i],
                ocean_mask=ocean,
                confidence_length_scale=CONFIDENCE_LS,
            )
            for i in range(start, end)
        ]
        feat_t = torch.from_numpy(np.stack(feats)).to(device)
        base_t = torch.from_numpy(base[start:end, None].astype(np.float32)).to(device)
        corrected, _ = model(feat_t, base_t)
        out[start:end] = corrected.float().cpu().numpy()[:, 0]
    return out
