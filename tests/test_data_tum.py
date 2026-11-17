import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from perception.data.frame import Frame
from perception.data.tum import PoseInterpolator, associate


def test_associate_nearest_and_drops_far():
    a = np.array([0.00, 0.033, 0.066, 1.0])
    b = np.array([0.005, 0.030, 0.070, 0.5])
    pairs = associate(a, b, max_dt=0.02)
    assert pairs == [(0, 0), (1, 1), (2, 2)]  # 1.0 has no partner within 20 ms


def test_associate_one_to_one():
    pairs = associate(np.array([0.0, 0.001]), np.array([0.0005]), max_dt=0.02)
    assert len(pairs) == 1


def test_pose_interpolation_midpoint():
    q0 = Rotation.identity().as_quat()
    q1 = Rotation.from_euler("z", 90, degrees=True).as_quat()
    interp = PoseInterpolator(np.array([0.0, 1.0]), np.array([[0, 0, 0], [2, 4, 6]]), np.array([q0, q1]))
    T = interp(0.5)
    assert np.allclose(T[:3, 3], [1, 2, 3])
    angle = Rotation.from_matrix(T[:3, :3]).magnitude()
    assert np.isclose(np.degrees(angle), 45.0)
    with pytest.raises(ValueError):
        interp(1.5)


def test_frame_rejects_bad_contract():
    good = dict(rgb=np.zeros((4, 5, 3), np.uint8), depth_m=np.zeros((4, 5), np.float32),
                valid=np.zeros((4, 5), bool), K=np.eye(3), frame_id="0", dataset="t", scene_id="s")
    Frame(**good)
    with pytest.raises(ValueError):
        Frame(**{**good, "depth_m": np.zeros((4, 5), np.float64)})
    bad_depth = np.ones((4, 5), np.float32)  # non-zero where invalid
    with pytest.raises(ValueError):
        Frame(**{**good, "depth_m": bad_depth})
    with pytest.raises(ValueError):
        Frame(**{**good, "dist": np.zeros(4)})
    assert Frame(**{**good, "dist": [0.1, 0, 0, 0, 0]}).dist.shape == (5,)
