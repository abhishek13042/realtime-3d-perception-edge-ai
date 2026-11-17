"""Pinhole camera model, OpenCV convention: +X right, +Y down, +Z forward, metres (spec 01.2, R3)."""

from __future__ import annotations

import functools

import cv2
import numpy as np


def make_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Nx3 camera-frame points -> Nx2 pixel coordinates (u, v). Points must have Z > 0."""
    p = np.asarray(points, dtype=np.float64)
    z = p[:, 2]
    u = K[0, 0] * p[:, 0] / z + K[0, 2]
    v = K[1, 1] * p[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def unproject(uv: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Nx2 pixels + N planar depths -> Nx3 camera-frame points: X=(u-cx)Z/fx, Y=(v-cy)Z/fy."""
    uv = np.asarray(uv, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x = (uv[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


@functools.lru_cache(maxsize=8)
def _normalized_grid(key: tuple) -> np.ndarray:
    k9, d5, h, w = key
    K, dist = np.array(k9, np.float64).reshape(3, 3), np.array(d5, np.float64)
    v, u = np.mgrid[0:h, 0:w]
    uv = np.stack([u.ravel(), v.ravel()], 1).astype(np.float64).reshape(-1, 1, 2)
    xn = cv2.undistortPoints(uv, K, dist).reshape(h, w, 2)
    xn.setflags(write=False)
    return xn


def normalized_grid(K: np.ndarray, dist: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """HxWx2 undistorted normalised coordinates (x/z, y/z) of every pixel centre; cached per (K, dist, size)."""
    return _normalized_grid((tuple(np.asarray(K, float).ravel()), tuple(np.asarray(dist, float).ravel()),
                             int(hw[0]), int(hw[1])))


def unproject_pixels(rows: np.ndarray, cols: np.ndarray, z: np.ndarray, K: np.ndarray,
                     dist: np.ndarray | None = None, hw: tuple[int, int] | None = None) -> np.ndarray:
    """Integer pixel indices + planar depths -> Nx3 camera-frame points.
    With `dist` (OpenCV k1, k2, p1, p2, k3) the pixels are undistorted first (needs the image size `hw`)."""
    z = np.asarray(z, np.float64)
    if dist is None or not np.any(dist):
        return unproject(np.stack([cols, rows], axis=1), z, K)
    if hw is None:
        raise ValueError("hw is required when dist is given")
    xn = normalized_grid(K, dist, hw)[rows, cols]
    return np.c_[xn * z[:, None], z]


def unproject_depth_map(depth_m: np.ndarray, K: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """HxW depth -> Mx3 points for valid pixels, in row-major pixel order."""
    if valid is None:
        valid = depth_m > 0
    v, u = np.nonzero(valid)
    return unproject(np.stack([u, v], axis=1), depth_m[v, u], K)


def scale_intrinsics(K: np.ndarray, s: float) -> np.ndarray:
    """Intrinsics after resizing the image by factor s, using the pixel-center convention:
    f' = s*f, c' = s*(c + 0.5) - 0.5."""
    K2 = K.astype(np.float64).copy()
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] = s * (K[0, 2] + 0.5) - 0.5
    K2[1, 2] = s * (K[1, 2] + 0.5) - 0.5
    return K2
