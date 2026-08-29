from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from secondlook.utils.logging import get_logger

logger = get_logger(__name__)

_MASK_CANDIDATES = ("ocean_mask.npy", "valid_mask.npy", "mask.npy")


def _load_mask(path: Path, target_shape: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    mask = np.load(path)
    if tuple(mask.shape) != target_shape:
        logger.warning(
            "Mask %s has shape %s, expected %s; skipping",
            path,
            mask.shape,
            target_shape,
        )
        return None
    return mask


def _resolve_metadata_mask(
    processed_dir: Path,
    data_root: Path,
    target_shape: tuple[int, int],
) -> tuple[Path | None, np.ndarray | None]:
    meta_path = processed_dir / "metadata.json"
    if not meta_path.is_file():
        return None, None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse %s: %s", meta_path, exc)
        return None, None

    mask_rel = meta.get("mask_path")
    if not mask_rel:
        return None, None

    rel = str(mask_rel).replace("\\", "/")
    candidates = [
        processed_dir / rel,
        data_root / rel,
        data_root / rel.lstrip("data/"),
        data_root / rel.replace("data/masks/", "masks/", 1),
    ]
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        mask = _load_mask(cand, target_shape)
        if mask is not None:
            return cand.resolve(), mask
    return None, None


def _resolve_g02202_masks(
    data_root: Path,
    target_shape: tuple[int, int],
) -> tuple[Path | None, np.ndarray | None]:
    search_roots = [
        data_root / "data" / "masks",
        data_root / "masks",
    ]
    for root in search_roots:
        if not root.is_dir():
            continue
        for sub in sorted(root.glob("*g02202*")):
            if sub.is_dir():
                for fname in _MASK_CANDIDATES:
                    mask = _load_mask(sub / fname, target_shape)
                    if mask is not None:
                        return (sub / fname).resolve(), mask
        for fname in _MASK_CANDIDATES:
            mask = _load_mask(root / fname, target_shape)
            if mask is not None:
                return (root / fname).resolve(), mask
    return None, None


def resolve_ocean_mask(
    data_root: str | Path,
    processed_dir: str | Path,
    target_shape: tuple[int, int] = (448, 304),
) -> dict[str, Any]:
    """
    Locate an ocean/valid mask without raising if none is found.

    Returns a dict with keys: found, path, mask, valid_fraction, warning.
    """
    data_root = Path(data_root)
    processed_dir = Path(processed_dir)
    warning = "No ocean mask found; finite valid pixels will be used in later metrics."

    for fname in _MASK_CANDIDATES:
        cand = processed_dir / fname
        mask = _load_mask(cand, target_shape)
        if mask is not None:
            valid_fraction = float(np.mean(mask.astype(bool)))
            return {
                "found": True,
                "path": str(cand.resolve()),
                "mask": mask,
                "valid_fraction": valid_fraction,
                "warning": None,
            }

    path, mask = _resolve_metadata_mask(processed_dir, data_root, target_shape)
    if mask is not None and path is not None:
        valid_fraction = float(np.mean(mask.astype(bool)))
        return {
            "found": True,
            "path": str(path),
            "mask": mask,
            "valid_fraction": valid_fraction,
            "warning": None,
        }

    path, mask = _resolve_g02202_masks(data_root, target_shape)
    if mask is not None and path is not None:
        valid_fraction = float(np.mean(mask.astype(bool)))
        return {
            "found": True,
            "path": str(path),
            "mask": mask,
            "valid_fraction": valid_fraction,
            "warning": None,
        }

    logger.warning(warning)
    return {
        "found": False,
        "path": None,
        "mask": None,
        "valid_fraction": None,
        "warning": warning,
    }
