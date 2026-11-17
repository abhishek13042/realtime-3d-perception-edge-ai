"""The single Frame contract every dataset reader returns (spec 01.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray            # uint8 HxWx3, RGB order
    depth_m: np.ndarray        # float32 HxW, planar Z in metres, 0.0 where invalid
    valid: np.ndarray          # bool HxW
    K: np.ndarray              # float64 3x3, intrinsics for this image size
    frame_id: str
    dataset: str
    scene_id: str
    T_world_cam: np.ndarray | None = None  # float64 4x4 camera-to-world
    timestamp: float | None = None
    dist: np.ndarray | None = None         # OpenCV (k1, k2, p1, p2, k3) of the colour camera; None = undistorted

    def __post_init__(self) -> None:
        if self.dist is not None:
            self.dist = np.asarray(self.dist, np.float64).reshape(-1)
            if self.dist.shape != (5,):
                raise ValueError(f"dist must have 5 coefficients (k1, k2, p1, p2, k3), got {self.dist.shape}")
        h, w = self.depth_m.shape[:2]
        if self.rgb.dtype != np.uint8 or self.rgb.shape != (h, w, 3):
            raise ValueError(f"rgb must be uint8 {(h, w, 3)}, got {self.rgb.dtype} {self.rgb.shape}")
        if self.depth_m.dtype != np.float32 or self.depth_m.ndim != 2:
            raise ValueError(f"depth_m must be float32 HxW, got {self.depth_m.dtype} {self.depth_m.shape}")
        if self.valid.dtype != np.bool_ or self.valid.shape != (h, w):
            raise ValueError(f"valid must be bool {(h, w)}, got {self.valid.dtype} {self.valid.shape}")
        if self.K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {self.K.shape}")
        if self.T_world_cam is not None and self.T_world_cam.shape != (4, 4):
            raise ValueError(f"T_world_cam must be 4x4, got {self.T_world_cam.shape}")
        if np.any(self.depth_m[~self.valid] != 0.0):
            raise ValueError("depth_m must be 0.0 wherever valid is False")

    @property
    def hw(self) -> tuple[int, int]:
        return self.depth_m.shape
