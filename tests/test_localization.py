"""Spec 03.3 tests: surface-vs-volume offset, projection linkage, bucket edges, split guard."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from perception.data.sunrgbd_annotations import Box2D, Box3D
from perception.depth.estimate import AssocParams, estimate_box
from perception.eval.localization import (bucket_of, greedy_match, half_extent_along, link_2d_3d, project_box,
                                          require_tuning_split, status_of, surface_point)
from perception.geometry.camera import make_K, unproject
from perception.geometry.object3d import object_centre

K = make_K(500.0, 500.0, 319.5, 239.5)
HW = (480, 640)
I3 = np.eye(3)


def upright_box(cx_cam, cz_cam, half_xyz, cls="chair"):
    """Axis-aligned box with Rtilt = I. Upright (x, y fwd, z up) maps to camera (x, -z, y), so a camera centre
    (cx, 0, cz) is upright (cx, cz, 0); coeffs are (half width, half depth along the view, half height)."""
    return Box3D(cls=cls, raw_cls=cls, centroid=[cx_cam, cz_cam, 0.0], basis=I3.tolist(), coeffs=list(half_xyz),
                 orientation=[1.0, 0.0, 0.0])


def test_oracle_centre_error_equals_surface_volume_offset():
    """A box straight ahead, front face at Z = 2.5 m, depth 1.0 m, in front of a wall at 6 m. The estimate sits on
    the visible face, so its error to the volume centre is the half depth (0.5 m), and ~0 to the surface point."""
    box = upright_box(0.0, 3.0, (0.4, 0.5, 0.3))
    centre = box.centroid_cam(I3)
    np.testing.assert_allclose(centre, [0.0, 0.0, 3.0], atol=1e-12)
    vv, uu = np.mgrid[0:HW[0], 0:HW[1]]
    xn, yn = (uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1]
    face = (np.abs(xn * 2.5) <= 0.4) & (np.abs(yn * 2.5) <= 0.3)
    depth = np.where(face, 2.5, 6.0).astype(np.float32)
    valid = np.ones(HW, bool)

    front = box.corners_cam(I3)
    front = front[np.isclose(front[:, 2], 2.5)]
    box2d = project_box(front, K, HW)  # the visible (modal) 2D box is the front face
    est = estimate_box(depth, valid, box2d, AssocParams(shrink=0.5, estimator="A"))
    assert est.status == "valid" and est.z == pytest.approx(2.5)
    c = np.array(object_centre(depth, valid, K, est, k=2.0, sigma_floor=0.02).centre)

    half = half_extent_along(box, I3, centre / np.linalg.norm(centre))
    assert half == pytest.approx(0.5)
    assert np.linalg.norm(c - centre) == pytest.approx(0.5, abs=0.005)
    assert c[2] - centre[2] == pytest.approx(-0.5, abs=1e-6)  # signed bias: estimate nearer than the volume centre
    assert np.linalg.norm(c - surface_point(centre, half)) < 0.005


def test_projected_cube_gives_expected_2d_box():
    cube = upright_box(0.0, 4.0, (0.5, 0.5, 0.5))
    got = project_box(cube.corners_cam(I3), K, HW)
    du = 0.5 / 3.5 * 500.0  # extremes come from the near face (Z = 3.5)
    expected = [319.5 - du + 0.5, 239.5 - du + 0.5, 319.5 + du + 0.5, 239.5 + du + 0.5]  # box convention: +0.5
    np.testing.assert_allclose(got, expected, atol=1e-9)
    # consistency with unproject: the box corners back-project onto the near-face corners
    uv = np.array([[got[0] - 0.5, got[1] - 0.5]])
    np.testing.assert_allclose(unproject(uv, np.array([3.5]), K)[0], [-0.5, -0.5, 3.5], atol=1e-9)


def test_projected_box_clipped_and_behind_camera():
    wide = upright_box(0.0, 1.0, (5.0, 0.2, 0.2))
    assert project_box(wide.corners_cam(I3), K, HW)[0::2] == [0.0, 640.0]
    behind = upright_box(0.0, -3.0, (0.5, 0.5, 0.5))
    assert project_box(behind.corners_cam(I3), K, HW) is None


def test_link_by_projection_ignores_annotation_order():
    left, right = upright_box(-1.0, 4.0, (0.3, 0.3, 0.3)), upright_box(1.0, 4.0, (0.3, 0.3, 0.3))
    pl, pr = project_box(left.corners_cam(I3), K, HW), project_box(right.corners_cam(I3), K, HW)
    shrink = lambda b: [b[0] + 3, b[1] + 3, b[2] - 3, b[3] - 3]  # visible boxes slightly inside the amodal ones
    ann = SimpleNamespace(Rtilt_np=I3, K_np=K, boxes3d=[left, right],
                          boxes2d=[Box2D("chair", "chair", shrink(pr), True), Box2D("chair", "chair", shrink(pl), True),
                                   Box2D("table", "table", shrink(pl), False)])
    pairs = link_2d_3d(ann, HW)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1), (1, 0)]  # 2D 0 is the right box -> 3D 1; table not linked
    assert all(v > 0.8 for _, _, v in pairs)


@pytest.mark.parametrize("z,name", [(0.0, "0-2"), (1.999, "0-2"), (2.0, "2-4"), (3.999, "2-4"), (4.0, ">4"),
                                    (9.0, ">4")])
def test_bucket_edges(z, name):
    assert bucket_of(z) == name


def test_status_of_matches_estimator_rule():
    rng = np.random.default_rng(0)
    depth = rng.uniform(1.0, 3.0, HW).astype(np.float32)
    valid = rng.random(HW) > 0.5
    p = AssocParams(shrink=0.5, estimator="A", n_min=20, tau_cov=0.6, tau_spread_m=0.5)
    boxes = [[100, 100, 110, 104], [100, 100, 300, 300], [0, 0, 640, 480], [10, 10, 12, 12]]
    ests = [estimate_box(depth, valid, b, p) for b in boxes]
    st = status_of(np.array([e.n_valid for e in ests]), np.array([e.valid_frac for e in ests]),
                   np.nan_to_num(np.array([e.spread for e in ests]), nan=np.inf), 20, 0.6, 0.5)
    assert list(st) == [e.status for e in ests]


def test_greedy_match_by_score_and_class():
    gt = [("chair", [0, 0, 10, 10]), ("chair", [20, 0, 30, 10]), ("tv", [0, 0, 10, 10])]
    boxes = [[0, 0, 10, 9], [0, 0, 10, 10], [20, 0, 30, 10], [0, 0, 10, 10]]
    m = greedy_match(boxes, [0.5, 0.9, 0.8, 0.7], ["chair", "chair", "chair", "tv"], gt)
    assert sorted(m) == [(1, 0), (2, 1), (3, 2)]  # the 0.5 chair loses GT 0 to the 0.9 chair


def test_split_guard_rejects_test():
    assert require_tuning_split("sunrgbd_val") == "sunrgbd_val"
    for bad in ("sunrgbd_test", "sunrgbd_train", "val"):
        with pytest.raises(ValueError):
            require_tuning_split(bad)
    path = Path(__file__).resolve().parents[1] / "scripts" / "tune_depth.py"
    spec = importlib.util.spec_from_file_location("tune_depth", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--split", "sunrgbd_test"]) == 2  # refused before any data is read
