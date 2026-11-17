import numpy as np

from perception.geometry.camera import make_K, project, scale_intrinsics, unproject, unproject_depth_map

K = make_K(525.0, 525.0, 319.5, 239.5)


def test_project_unproject_round_trip():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-2, 2, 1000), rng.uniform(-2, 2, 1000), rng.uniform(0.3, 8, 1000)])
    back = unproject(project(pts, K), pts[:, 2], K)
    assert np.max(np.abs(back - pts)) < 1e-9


def test_unproject_plane_at_2m():
    h, w = 480, 640
    depth = np.full((h, w), 2.0, dtype=np.float32)
    pts = unproject_depth_map(depth, K)
    assert pts.shape == (h * w, 3)
    assert np.allclose(pts[:, 2], 2.0)
    assert np.isclose(pts[:, 0].min(), (0 - 319.5) * 2 / 525)
    assert np.isclose(pts[:, 0].max(), (w - 1 - 319.5) * 2 / 525)


def test_unproject_respects_valid_mask():
    depth = np.ones((4, 4), dtype=np.float32)
    valid = np.zeros((4, 4), bool)
    valid[1, 2] = True
    pts = unproject_depth_map(depth, K, valid)
    assert pts.shape == (1, 3)


def test_scale_intrinsics_half_pixel_center():
    K2 = scale_intrinsics(K, 0.5)
    assert np.isclose(K2[0, 0], 262.5) and np.isclose(K2[1, 1], 262.5)
    # principal point at the exact image centre stays at the exact centre of the half-size image
    assert np.isclose(K2[0, 2], 159.5) and np.isclose(K2[1, 2], 119.5)


def test_axis_convention_opencv():
    # a point to the right and below the optical axis projects right of and below the principal point
    uv = project(np.array([[0.5, 0.5, 2.0]]), K)[0]
    assert uv[0] > K[0, 2] and uv[1] > K[1, 2]


# --- spec 03.2 R6: undistortion when back-projecting ---------------------------------------------------------------
FR1_DIST = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])


def test_unproject_pixels_zero_dist_equals_plain():
    from perception.geometry.camera import unproject_pixels
    rng = np.random.default_rng(1)
    rr, cc = rng.integers(0, 480, 500), rng.integers(0, 640, 500)
    z = rng.uniform(0.5, 5, 500)
    plain = unproject(np.stack([cc, rr], 1), z, K)
    assert np.allclose(unproject_pixels(rr, cc, z, K), plain)
    assert np.allclose(unproject_pixels(rr, cc, z, K, np.zeros(5), (480, 640)), plain)


def test_unproject_pixels_matches_cv2_and_reprojects():
    import cv2
    from perception.geometry.camera import unproject_pixels
    rng = np.random.default_rng(2)
    rr, cc = rng.integers(0, 480, 300), rng.integers(0, 640, 300)
    z = rng.uniform(0.5, 5, 300)
    pts = unproject_pixels(rr, cc, z, K, FR1_DIST, (480, 640))
    uv = np.stack([cc, rr], 1).astype(np.float64)
    xn = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, FR1_DIST).reshape(-1, 2)
    assert np.allclose(pts[:, :2], xn * z[:, None]) and np.allclose(pts[:, 2], z)
    # Re-projecting WITH distortion recovers the original pixels (within the iterative solver's tolerance).
    back, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, FR1_DIST)
    centre = np.hypot(cc - K[0, 2], rr - K[1, 2]) < 300  # cv2 iterations are least accurate in the far corners
    assert np.max(np.abs(back.reshape(-1, 2) - uv)[centre]) < 0.05
    # And distortion matters: the corner pixel moves by far more than a pixel's worth.
    assert not np.allclose(pts, unproject(uv, z, K), atol=1e-3)


def test_unproject_pixels_requires_size_with_dist():
    import pytest
    from perception.geometry.camera import unproject_pixels
    with pytest.raises(ValueError):
        unproject_pixels(np.array([0]), np.array([0]), np.array([1.0]), K, FR1_DIST)
