"""Frame transforms between SUN RGB-D's upright (gravity-aligned) frame and the OpenCV camera frame (spec 01.3).

SUN RGB-D toolbox (read3dPoints.m) builds upright points from camera points as
    p_upright = Rtilt · flip(p_cam),  flip(x, y, z) = (x, z, −y)
where p_cam is OpenCV (x right, y down, z forward) and the upright frame is x right, y forward, z up.
The inverse used here:
    p_cam = flip⁻¹(Rtiltᵀ · p_upright),  flip⁻¹(a, b, c) = (a, −c, b)
"""

from __future__ import annotations

import numpy as np

# flip⁻¹ as a matrix: rows give (a, −c, b)
_UNFLIP = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0]])


def upright_to_cam_matrix(Rtilt: np.ndarray) -> np.ndarray:
    """3×3 rotation R such that p_cam = R · p_upright."""
    return _UNFLIP @ np.asarray(Rtilt, dtype=np.float64).T


def upright_to_cam(points: np.ndarray, Rtilt: np.ndarray) -> np.ndarray:
    """Transform (N, 3) or (3,) upright points to the OpenCV camera frame."""
    p = np.asarray(points, dtype=np.float64)
    return p @ upright_to_cam_matrix(Rtilt).T


def cam_to_upright(points: np.ndarray, Rtilt: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    return p @ upright_to_cam_matrix(Rtilt)


def box_corners(centroid: np.ndarray, basis: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """8 corners (8, 3) of an oriented box. `basis` rows are the box axes, `coeffs` the half sizes
    (SUN RGB-D convention, toolbox get_corners_of_bb3d.m). Order: bottom face (z−) 0-3, top face (z+) 4-7,
    each face counter-clockwise from (−x, −y), so edges are (0-1, 1-2, 2-3, 3-0), (4-5, ...), (i, i+4)."""
    c = np.asarray(centroid, dtype=np.float64)
    B = np.asarray(basis, dtype=np.float64)
    h = np.asarray(coeffs, dtype=np.float64)
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
    return c + (signs * h) @ B


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
