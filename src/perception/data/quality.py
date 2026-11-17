"""Pure per-frame data-quality statistics (spec 01.4, T1)."""

from __future__ import annotations

import numpy as np

DEPTH_HIST_EDGES = np.round(np.arange(0.0, 10.0001, 0.1), 4)  # 0.1 m bins up to 10 m


def valid_fraction(valid: np.ndarray) -> float:
    v = np.asarray(valid, dtype=bool)
    return float(v.sum()) / float(v.size) if v.size else 0.0


def depth_percentiles(depth: np.ndarray, valid: np.ndarray, qs=(5, 50, 95)) -> list[float] | None:
    """Percentiles of depth over valid pixels only; None if no pixel is valid."""
    d = np.asarray(depth)[np.asarray(valid, dtype=bool)]
    if d.size == 0:
        return None
    return [float(x) for x in np.percentile(d, qs)]


def depth_histogram(depth: np.ndarray, valid: np.ndarray, edges: np.ndarray = DEPTH_HIST_EDGES) -> np.ndarray:
    """Integer counts of valid depth values per bin (values ≥ the last edge go in the last bin)."""
    d = np.asarray(depth)[np.asarray(valid, dtype=bool)]
    d = np.clip(d, edges[0], edges[-1] - 1e-9)
    return np.histogram(d, bins=edges)[0].astype(np.int64)


def differs_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels where two equally shaped arrays differ (used for depth vs depth_bfx)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float((a != b).mean())
