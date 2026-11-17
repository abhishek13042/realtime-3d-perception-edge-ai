"""Point clouds (spec 03.2 R4): frame cloud, per-object cloud, numpy voxel downsampling, ASCII PLY I/O.
Core code never imports open3d; the viewer script uses it when available."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perception.depth.estimate import DepthEstimate
from perception.geometry.camera import unproject_pixels
from perception.geometry.object3d import inlier_mask


def frame_cloud(depth_m: np.ndarray, valid: np.ndarray, K: np.ndarray, rgb: np.ndarray | None = None,
                stride: int = 1, dist: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Valid pixels (every `stride`-th row and column) -> (N, 3) float32 points and (N, 3) uint8 colours."""
    m = np.zeros_like(valid)
    m[::stride, ::stride] = valid[::stride, ::stride]
    rr, cc = np.nonzero(m)
    pts = unproject_pixels(rr, cc, depth_m[rr, cc], K, dist, depth_m.shape).astype(np.float32)
    return pts, (rgb[rr, cc].astype(np.uint8) if rgb is not None else None)


def object_cloud(depth_m, valid, K, est: DepthEstimate, rgb=None, k: float = 2.0, sigma_floor: float = 0.02,
                 dist: np.ndarray | None = None):
    """The inlier pixels of one object (the same set used for its centre)."""
    if est.status == "no_depth":
        return np.zeros((0, 3), np.float32), (np.zeros((0, 3), np.uint8) if rgb is not None else None)
    rr, cc = inlier_mask(depth_m, valid, est, k, sigma_floor)
    pts = unproject_pixels(rr, cc, depth_m[rr, cc], K, dist, depth_m.shape).astype(np.float32)
    return pts, (rgb[rr, cc].astype(np.uint8) if rgb is not None else None)


def voxel_downsample(pts: np.ndarray, voxel: float, colors: np.ndarray | None = None):
    """One point per occupied voxel: the centroid of its points (and their mean colour)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts.astype(np.float32), (None if colors is None else np.zeros((0, 3), np.uint8))
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = np.zeros((counts.size, 3))
    np.add.at(out, inv, pts)
    out /= counts[:, None]
    col = None
    if colors is not None:
        col = np.zeros((counts.size, 3))
        np.add.at(col, inv, np.asarray(colors, np.float64))
        col = np.round(col / counts[:, None]).astype(np.uint8)
    return out.astype(np.float32), col


def write_ply(path: str | Path, pts: np.ndarray, colors: np.ndarray | None = None) -> None:
    pts = np.asarray(pts, np.float32).reshape(-1, 3)
    head = ["ply", "format ascii 1.0", f"element vertex {len(pts)}",
            "property float x", "property float y", "property float z"]
    if colors is not None:
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    head.append("end_header")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(head) + "\n")
        if colors is None:
            np.savetxt(f, pts, fmt="%.6f")
        else:
            data = np.hstack([pts.astype(object), np.asarray(colors, np.uint8).astype(object)])
            np.savetxt(f, data, fmt=["%.6f"] * 3 + ["%d"] * 3)


def read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Reader for the ASCII files written by write_ply (not a general PLY reader)."""
    with open(path, encoding="ascii") as f:
        props, n = [], 0
        for line in f:
            line = line.strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.loadtxt(f, ndmin=2) if n else np.zeros((0, len(props)))
    pts = data[:, :3].astype(np.float32)
    col = data[:, 3:6].astype(np.uint8) if len(props) >= 6 else None
    return pts, col
