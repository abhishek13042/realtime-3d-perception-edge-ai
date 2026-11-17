import numpy as np
from scipy.spatial.transform import Rotation

from perception.geometry.transforms import box_corners, cam_to_upright, upright_to_cam, upright_to_cam_matrix


def test_forward_maps_to_z():
    np.testing.assert_allclose(upright_to_cam([0, 2, 0], np.eye(3)), [0, 0, 2], atol=1e-12)


def test_up_maps_to_negative_y():
    np.testing.assert_allclose(upright_to_cam([0, 0, 1], np.eye(3)), [0, -1, 0], atol=1e-12)


def test_is_rotation_and_invertible():
    rng = np.random.default_rng(0)
    Rtilt = Rotation.from_euler("xz", [8, -5], degrees=True).as_matrix()
    R = upright_to_cam_matrix(Rtilt)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12
    p = rng.normal(size=(50, 3))
    q = upright_to_cam(p, Rtilt)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), np.linalg.norm(p, axis=1), atol=1e-12)
    np.testing.assert_allclose(cam_to_upright(q, Rtilt), p, atol=1e-12)


def test_box_corners():
    basis = Rotation.from_euler("z", 30, degrees=True).as_matrix().T
    c = np.array([1.0, 3.0, -0.5])
    corners = box_corners(c, basis, [0.5, 1.0, 0.25])
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners.mean(axis=0), c, atol=1e-12)
    # edge lengths along each axis equal full sizes
    np.testing.assert_allclose(np.linalg.norm(corners[1] - corners[0]), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(corners[3] - corners[0]), 2.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(corners[4] - corners[0]), 0.5, atol=1e-12)
