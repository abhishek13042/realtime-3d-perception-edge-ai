"""Object 3D centres from a depth estimate (spec 03.2 R1-R3). OpenCV camera frame, metres.

Pixel coordinates passed to `unproject` are integer pixel indices: OpenCV intrinsics put the centre of pixel (u, v)
at (u, v), so pixel index u is the right coordinate (the box convention [i, i + 1) of spec 02.1 is only used to pick
which pixels are sampled).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from perception.depth.estimate import MAD_TO_SIGMA, DepthEstimate
from perception.geometry.camera import unproject_pixels

NAN3 = (math.nan, math.nan, math.nan)


@dataclass
class Object3D:
    centre: tuple[float, float, float]   # camera frame (m); NaN when no centre
    n_inliers: int
    spread3: tuple[float, float, float]  # per-axis 1.4826 * MAD of the inlier points (m)
    status: str                          # copied from the DepthEstimate


def inlier_mask(depth_m: np.ndarray, valid: np.ndarray, est: DepthEstimate, k: float,
                sigma_floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows and cols (absolute image indices) of region pixels within +-k*max(spread, floor) of est.z."""
    c0, r0, c1, r1 = est.region
    sub, vs = depth_m[r0:r1, c0:c1], valid[r0:r1, c0:c1]
    band = k * max(est.spread if math.isfinite(est.spread) else 0.0, sigma_floor)
    rr, cc = np.nonzero(vs & (np.abs(sub - est.z) <= band))
    return rr + r0, cc + c0


def object_centre(depth_m: np.ndarray, valid: np.ndarray, K: np.ndarray, est: DepthEstimate,
                  k: float = 2.0, sigma_floor: float = 0.02, dist: np.ndarray | None = None) -> Object3D:
    """`dist`: colour-camera distortion (Frame.dist); when given, inlier pixels are undistorted (spec 03.2 R6)."""
    if est.status == "no_depth" or not math.isfinite(est.z):
        return Object3D(NAN3, 0, NAN3, est.status)
    rr, cc = inlier_mask(depth_m, valid, est, k, sigma_floor)
    if rr.size == 0:  # cannot happen for a mode/median of the same pixels, but never guess
        return Object3D(NAN3, 0, NAN3, est.status)
    pts = unproject_pixels(rr, cc, depth_m[rr, cc], K, dist, depth_m.shape)
    med = np.median(pts, axis=0)
    mad = MAD_TO_SIGMA * np.median(np.abs(pts - med), axis=0)
    return Object3D(tuple(map(float, med)), int(rr.size), tuple(map(float, mad)), est.status)


def to_world(points_cam: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """Nx3 camera-frame points -> world frame: p_w = R p_c + t, with T_world_cam the 4x4 camera-to-world pose."""
    p = np.asarray(points_cam, np.float64).reshape(-1, 3)
    return p @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]


def to_camera(points_world: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    p = np.asarray(points_world, np.float64).reshape(-1, 3)
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    return (p - t) @ R
