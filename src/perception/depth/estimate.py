"""Depth association: 2D box -> one object depth, a spread, and a trust status (spec 03.1).

Pixel convention: boxes are continuous [0, W] x [0, H] (spec 02.1); pixel column i covers [i, i + 1). The sampling
region of a box (x1, y1, x2, y2) after shrinking is columns floor(x1) .. ceil(x2) - 1 and rows floor(y1) .. ceil(y2) - 1,
clipped to the image. Only pixels with valid depth are used; missing depth is never filled or guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class AssocParams:
    shrink: float = 0.5
    estimator: str = "B"          # "A" median, "B" nearest significant mode
    bin_m: float = 0.05
    m_frac: float = 0.15
    n_min: int = 20
    tau_cov: float = 0.3
    tau_spread_m: float = 0.25

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **overrides) -> "AssocParams":
        a, s = cfg["association"], cfg["status"]
        p = dict(shrink=a["shrink"], estimator=a["estimator"], bin_m=a["bin_m"], m_frac=a["m_frac"],
                 n_min=s["n_min"], tau_cov=s["tau_cov"], tau_spread_m=s["tau_spread_m"])
        p.update(overrides)
        return cls(**p)


@dataclass
class DepthEstimate:
    z: float                       # metres; NaN when status == "no_depth"
    estimator: str                 # "A" or "B" (B with fallback=True actually used A)
    spread: float                  # 1.4826 * MAD of the valid depths (m); NaN when no valid pixels
    n_valid: int
    valid_frac: float              # n_valid / pixels in region (0 for an empty region)
    status: str                    # "valid" | "low_coverage" | "no_depth"
    fallback: bool = False         # estimator B found no significant bin and used the median
    region: tuple[int, int, int, int] = field(default=(0, 0, 0, 0))  # (c0, r0, c1, r1), half-open


def region_of(box, hw: tuple[int, int], shrink: float) -> tuple[int, int, int, int]:
    """Shrunken, clipped integer region (c0, r0, c1, r1), half-open. May be empty (c1 <= c0 or r1 <= r0)."""
    h, w = hw
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hx, hy = (x2 - x1) * shrink / 2, (y2 - y1) * shrink / 2
    c0 = max(0, math.floor(cx - hx))
    r0 = max(0, math.floor(cy - hy))
    c1 = min(w, math.ceil(cx + hx))
    r1 = min(h, math.ceil(cy + hy))
    return c0, r0, c1, r1


def robust_spread(z: np.ndarray, med: float | None = None) -> float:
    if not z.size:
        return math.nan
    return float(MAD_TO_SIGMA * np.median(np.abs(z - (np.median(z) if med is None else med))))


def nearest_mode(z: np.ndarray, bin_m: float, m_frac: float, med: float | None = None) -> tuple[float, bool]:
    """Estimator B. Returns (depth, fallback). Bins are fixed at multiples of bin_m (not data-dependent edges).
    `med` is the median of z if the caller already has it (used for the fallback)."""
    b = np.floor(z / bin_m).astype(np.int64)
    off = int(b.min())
    counts = np.bincount(b - off)                         # index ascending = nearest bin first
    sig = np.flatnonzero(counts >= m_frac * z.size)
    if sig.size == 0:
        return (float(np.median(z)) if med is None else med), True
    b0 = int(sig[0]) + off
    return float(np.median(z[np.abs(b - b0) <= 1])), False


def estimate_box(depth_m: np.ndarray, valid: np.ndarray, box, p: AssocParams) -> DepthEstimate:
    if p.estimator not in ("A", "B"):
        raise ValueError(f"unknown estimator {p.estimator!r}")
    reg = region_of(box, depth_m.shape, p.shrink)
    c0, r0, c1, r1 = reg
    n_px = max(0, c1 - c0) * max(0, r1 - r0)
    if n_px == 0:
        return DepthEstimate(math.nan, p.estimator, math.nan, 0, 0.0, "no_depth", False, reg)
    z = depth_m[r0:r1, c0:c1][valid[r0:r1, c0:c1]].astype(np.float64)
    n, frac = int(z.size), z.size / n_px
    if n < p.n_min:
        return DepthEstimate(math.nan, p.estimator, robust_spread(z), n, frac, "no_depth", False, reg)
    med = float(np.median(z))
    spread = robust_spread(z, med)
    est, fb = (med, False) if p.estimator == "A" else nearest_mode(z, p.bin_m, p.m_frac, med)
    status = "low_coverage" if (frac < p.tau_cov or spread > p.tau_spread_m) else "valid"
    return DepthEstimate(est, p.estimator, spread, n, frac, status, fb, reg)


def estimate_boxes(depth_m: np.ndarray, valid: np.ndarray, boxes, p: AssocParams) -> list[DepthEstimate]:
    """All boxes of one frame. boxes: (N, 4) xyxy in original-image pixels."""
    return [estimate_box(depth_m, valid, b, p) for b in np.asarray(boxes, np.float64).reshape(-1, 4)]
