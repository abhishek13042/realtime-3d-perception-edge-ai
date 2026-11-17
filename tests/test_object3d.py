"""Spec 03.2 tests: object centres, world mapping, point clouds, synthetic plane recovery."""

import math

import numpy as np
import pytest

from perception.depth.estimate import AssocParams, estimate_box
from perception.geometry.camera import make_K, project, unproject
from perception.geometry.object3d import object_centre, to_camera, to_world
from perception.geometry.pointcloud import frame_cloud, object_cloud, read_ply, voxel_downsample, write_ply

K = make_K(525.0, 525.0, 319.5, 239.5)
HW = (480, 640)
P = AssocParams(shrink=0.5, estimator="B", bin_m=0.05, m_frac=0.15, n_min=20, tau_cov=0.3, tau_spread_m=0.25)


def test_round_trip_below_1e6():
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-2, 2, (1000, 2)), rng.uniform(0.3, 8, 1000)]
    back = unproject(project(pts, K), pts[:, 2], K)
    assert np.abs(back - pts).max() < 1e-6


def test_tilted_plane_recovered():
    # Plane n . X = d with a 20 deg tilt about the camera x-axis, 2 m away.
    a = math.radians(20)
    n = np.array([0.0, math.sin(a), math.cos(a)])
    d = 2.0
    v, u = np.mgrid[0:HW[0], 0:HW[1]]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(HW)], -1)
    depth = (d / (rays @ n)).astype(np.float64)
    pts, _ = frame_cloud(depth, np.ones(HW, bool), K)
    pts = pts.astype(np.float64)
    c = pts.mean(0)
    n_hat = np.linalg.svd(pts - c, full_matrices=False)[2][-1]
    n_hat *= np.sign(n_hat @ n)
    assert math.degrees(math.acos(min(1.0, n_hat @ n))) < 0.1
    assert abs(n_hat @ c - d) < 1e-3


def _box_scene(X=0.3, Y=-0.2, Z=1.5, half=0.25, bg=4.0):
    """Fronto-parallel square object (side 2*half) centred at (X, Y, Z) in front of a far wall."""
    depth = np.full(HW, bg, np.float64)
    (u0, v0), (u1, v1) = project(np.array([[X - half, Y - half, Z], [X + half, Y + half, Z]]), K)
    depth[int(math.ceil(v0)):int(math.floor(v1)) + 1, int(math.ceil(u0)):int(math.floor(u1)) + 1] = Z
    # project() gives OpenCV coordinates (pixel centre at integer u); detector boxes use the pixel-edge
    # convention of spec 02.1 (pixel i covers [i, i + 1)), i.e. box coordinate = OpenCV coordinate + 0.5.
    return depth.astype(np.float32), (u0 + 0.5, v0 + 0.5, u1 + 0.5, v1 + 0.5)


def test_centre_of_synthetic_object():
    depth, box = _box_scene()
    valid = np.ones(HW, bool)
    est = estimate_box(depth, valid, box, P)
    obj = object_centre(depth, valid, K, est, k=2.0, sigma_floor=0.02)
    assert est.status == "valid" and obj.n_inliers == est.n_valid
    assert np.allclose(obj.centre, (0.3, -0.2, 1.5), atol=1e-3)


def test_inlier_band_excludes_background():
    depth, (u0, v0, u1, v1) = _box_scene()
    valid = np.ones(HW, bool)
    big = (u0 - 60, v0 - 60, u1 + 60, v1 + 60)       # box much larger than the object: background in region
    est = estimate_box(depth, valid, big, AssocParams(**{**P.__dict__, "shrink": 1.0}))
    assert est.z == pytest.approx(1.5, abs=0.01)
    obj = object_centre(depth, valid, K, est)
    assert obj.centre[2] == pytest.approx(1.5, abs=1e-6) and obj.n_inliers < est.n_valid
    pts, _ = object_cloud(depth, valid, K, est)
    assert np.allclose(pts[:, 2], 1.5)


def test_no_depth_gives_nan_centre():
    depth = np.zeros(HW, np.float32)
    valid = np.zeros(HW, bool)
    est = estimate_box(depth, valid, (100, 100, 200, 200), P)
    obj = object_centre(depth, valid, K, est)
    assert est.status == "no_depth" and all(math.isnan(c) for c in obj.centre) and obj.n_inliers == 0
    assert object_cloud(depth, valid, K, est)[0].shape == (0, 3)


def test_world_round_trip():
    a = math.radians(30)
    T = np.eye(4)
    T[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    T[:3, 3] = [1.0, -2.0, 0.5]
    p = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]])
    w = to_world(p, T)
    assert np.allclose(w[0], T[:3, 3] + T[:3, 2])           # optical axis point -> t + R[:, 2]
    assert np.allclose(to_camera(w, T), p)


def test_frame_cloud_valid_only_and_colours():
    rng = np.random.default_rng(1)
    depth = rng.uniform(0.5, 5, HW).astype(np.float32)
    valid = rng.random(HW) > 0.3
    depth[~valid] = 0
    rgb = rng.integers(0, 255, HW + (3,), dtype=np.uint8)
    pts, col = frame_cloud(depth, valid, K, rgb)
    assert len(pts) == valid.sum() and (pts[:, 2] > 0).all()
    rr, cc = np.nonzero(valid)
    assert np.array_equal(col, rgb[rr, cc])
    pts2, _ = frame_cloud(depth, valid, K, stride=2)
    assert len(pts2) == valid[::2, ::2].sum()


def test_voxel_downsample():
    pts = np.array([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [0.21, 0.0, 0.0]])
    col = np.array([[0, 0, 0], [100, 200, 50], [9, 9, 9]], np.uint8)
    out, oc = voxel_downsample(pts, 0.1, col)
    order = np.argsort(out[:, 0])
    assert np.allclose(out[order], [[0.02, 0.02, 0.02], [0.21, 0, 0]])
    assert np.array_equal(oc[order][0], [50, 100, 25])
    assert voxel_downsample(np.zeros((0, 3)), 0.1)[0].shape == (0, 3)


def test_ply_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(50, 3)).astype(np.float32)
    col = rng.integers(0, 255, (50, 3), dtype=np.uint8)
    write_ply(tmp_path / "a.ply", pts, col)
    p2, c2 = read_ply(tmp_path / "a.ply")
    assert np.allclose(p2, pts, atol=1e-6) and np.array_equal(c2, col)
    write_ply(tmp_path / "b.ply", pts)
    p3, c3 = read_ply(tmp_path / "b.ply")
    assert np.allclose(p3, pts, atol=1e-6) and c3 is None
