"""Raw depth image -> metres + valid mask (spec 01.2, R1-R2). All dataset encoding quirks live here."""

from __future__ import annotations

import numpy as np

TUM_SCALE = 5000.0     # 16-bit PNG, 5000 units = 1 m
SUN_SCALE = 1000.0     # millimetres after bit rotation


def sun_unrotate(raw: np.ndarray) -> np.ndarray:
    """SUN RGB-D stores depth with a 3-bit right rotation: value = (raw >> 3) | (raw << 13), as 16 bit."""
    r = raw.astype(np.uint32)
    return (((r >> 3) | (r << 13)) & 0xFFFF).astype(np.uint16)


def to_metres(raw: np.ndarray, scale: float, min_m: float = 0.1, max_m: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Convert integer depth to float32 metres; return (depth_m, valid). Invalid pixels are set to 0.0."""
    depth = raw.astype(np.float32) / np.float32(scale)
    valid = np.isfinite(depth) & (raw > 0) & (depth >= min_m) & (depth <= max_m)
    depth = np.where(valid, depth, np.float32(0.0)).astype(np.float32)
    return depth, valid


def decode_tum(raw: np.ndarray, min_m: float = 0.1, max_m: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    return to_metres(raw, TUM_SCALE, min_m, max_m)


def decode_sun(raw: np.ndarray, min_m: float = 0.1, max_m: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    return to_metres(sun_unrotate(raw), SUN_SCALE, min_m, max_m)


def mask_range(depth_m: np.ndarray, min_m: float, max_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Re-mask an already-metric depth map (e.g. float input that may contain NaN/inf)."""
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth_m) & (depth_m >= min_m) & (depth_m <= max_m)
    return np.where(valid, depth_m, 0.0).astype(np.float32), valid
