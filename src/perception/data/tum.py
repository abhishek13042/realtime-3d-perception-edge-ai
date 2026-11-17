"""TUM RGB-D sequence reader (spec 01.2, R4). Lazy: indexing loads one frame."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from perception.data.frame import Frame
from perception.depth.decode import decode_tum
from perception.geometry.camera import make_K

# Calibrated colour intrinsics from the TUM dataset page. fr3 images are already undistorted.
TUM_INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}
ROS_DEFAULT = (525.0, 525.0, 319.5, 239.5)
# Colour-camera distortion (d0..d4 on the TUM "file formats / intrinsic camera calibration" page), in OpenCV order
# (k1, k2, p1, p2, k3). fr3 is published undistorted. Used by the spec 03.2 R6 measurement.
TUM_DISTORTION = {
    "freiburg1": (0.2624, -0.9531, -0.0054, 0.0026, 1.1633),
    "freiburg2": (0.2312, -0.7849, -0.0033, -0.0001, 0.9172),
    "freiburg3": (0.0, 0.0, 0.0, 0.0, 0.0),
}


def read_list(path: Path) -> list[tuple[float, list[str]]]:
    """Parse a TUM text list: '# comment' lines skipped, each row 'timestamp rest...'."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        rows.append((float(parts[0]), parts[1:]))
    return rows


def associate(t_a: np.ndarray, t_b: np.ndarray, max_dt: float) -> list[tuple[int, int]]:
    """Greedy one-to-one nearest-timestamp matching, closest pairs first, |dt| <= max_dt."""
    t_a, t_b = np.asarray(t_a), np.asarray(t_b)
    if len(t_a) == 0 or len(t_b) == 0:
        return []
    order = np.argsort(t_b)
    sb = t_b[order]
    cands = []
    for i, t in enumerate(t_a):
        k = np.searchsorted(sb, t)
        for j in (k - 1, k):
            if 0 <= j < len(sb) and abs(sb[j] - t) <= max_dt:
                cands.append((abs(sb[j] - t), i, int(order[j])))
    cands.sort()
    used_a, used_b, pairs = set(), set(), []
    for _, i, j in cands:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    return sorted(pairs)


class PoseInterpolator:
    """Camera-to-world pose at arbitrary time: linear translation, SLERP rotation."""

    def __init__(self, times: np.ndarray, t_xyz: np.ndarray, q_xyzw: np.ndarray):
        order = np.argsort(times)
        self.times = np.asarray(times, dtype=np.float64)[order]
        self.t = np.asarray(t_xyz, dtype=np.float64)[order]
        self.slerp = Slerp(self.times, Rotation.from_quat(np.asarray(q_xyzw, dtype=np.float64)[order]))

    def covers(self, t: float) -> bool:
        return self.times[0] <= t <= self.times[-1]

    def __call__(self, t: float) -> np.ndarray:
        if not self.covers(t):
            raise ValueError(f"t={t} outside ground-truth range")
        T = np.eye(4)
        T[:3, :3] = self.slerp([t]).as_matrix()[0]
        T[:3, 3] = [np.interp(t, self.times, self.t[:, k]) for k in range(3)]
        return T


class TUMSequence:
    def __init__(self, seq_dir: str | Path, max_dt: float = 0.02, intrinsics: str = "calibrated",
                 min_m: float = 0.1, max_m: float = 10.0, undistort: bool = False):
        self.dir = Path(seq_dir)
        self.name = self.dir.name
        self.min_m, self.max_m = min_m, max_m
        self.K = make_K(*self._intrinsics(intrinsics))
        # undistort=True attaches the colour-camera distortion to every Frame (spec 03.2 R6); geometry code then
        # undistorts pixel coordinates when back-projecting. Images themselves are never resampled.
        self.dist = None
        if undistort and intrinsics != "ros_default":
            d = next((np.array(v) for k, v in TUM_DISTORTION.items() if k in self.name), None)
            self.dist = d if d is not None and np.any(d) else None

        rgb = read_list(self.dir / "rgb.txt")
        dep = read_list(self.dir / "depth.txt")
        pairs = associate(np.array([t for t, _ in rgb]), np.array([t for t, _ in dep]), max_dt)

        gt_path = self.dir / "groundtruth.txt"
        self.poses = None
        if gt_path.exists():
            gt = read_list(gt_path)
            arr = np.array([[t, *map(float, r[:7])] for t, r in gt])
            self.poses = PoseInterpolator(arr[:, 0], arr[:, 1:4], arr[:, 4:8])

        self.items = []  # (timestamp, rgb_rel, depth_rel)
        for i, j in pairs:
            t = rgb[i][0]
            if self.poses is not None and not self.poses.covers(t):
                continue  # spec R4: drop frames outside the ground-truth time range
            self.items.append((t, rgb[i][1][0], dep[j][1][0]))
        self.n_rgb, self.n_depth, self.n_pairs = len(rgb), len(dep), len(pairs)

    @classmethod
    def from_config(cls, sequence: str, cfg: dict) -> "TUMSequence":
        """Construct from configs/dataset.yaml (data root, association, intrinsics, depth range, undistort)."""
        from perception.config import data_root
        return cls(data_root(cfg) / "tum" / sequence, max_dt=cfg["tum"]["max_dt_s"],
                   intrinsics=cfg["tum"]["intrinsics"], min_m=cfg["depth"]["min_m"], max_m=cfg["depth"]["max_m"],
                   undistort=bool(cfg["tum"].get("undistort", False)))

    def _intrinsics(self, mode: str) -> tuple[float, float, float, float]:
        if mode == "ros_default":
            return ROS_DEFAULT
        for prefix, k in TUM_INTRINSICS.items():
            if prefix in self.name:
                return k
        raise ValueError(f"cannot infer TUM camera from sequence name {self.name!r}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Frame:
        t, rgb_rel, dep_rel = self.items[idx]
        bgr = cv2.imread(str(self.dir / rgb_rel), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(self.dir / dep_rel), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None:
            raise IOError(f"failed to read frame {idx} of {self.name}")
        if raw.dtype != np.uint16:
            raise ValueError(f"expected 16-bit depth, got {raw.dtype}")
        depth, valid = decode_tum(raw, self.min_m, self.max_m)
        return Frame(
            rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            depth_m=depth,
            valid=valid,
            K=self.K.copy(),
            frame_id=f"{t:.6f}",
            dataset="tum",
            scene_id=self.name,
            T_world_cam=self.poses(t) if self.poses is not None else None,
            timestamp=t,
            dist=None if self.dist is None else self.dist.copy(),
        )
