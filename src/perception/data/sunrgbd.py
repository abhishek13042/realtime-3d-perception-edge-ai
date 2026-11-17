"""SUN RGB-D reader (spec 01.2, R5). The image list and per-image K come from SUNRGBDMeta3DBB_v2.mat.

Uses the raw `depth/` folder, not `depth_bfx/` (hole-filled), so that the valid-depth fraction is honest.
Lazy: indexing loads one frame. 3D boxes are parsed in spec 01.3 (perception.data.sunrgbd_annotations).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

from perception.data.frame import Frame
from perception.depth.decode import decode_sun

REMOTE_PREFIX = "/n/fs/sun3d/data/"


@dataclass(frozen=True)
class SunRecord:
    index: int            # 0-based position in the metadata (1-based image ids in the official splits = index + 1)
    sequence: str         # e.g. SUNRGBD/kv2/kinect2data/000002_..., also used as scene_id
    sensor: str           # kv1 | kv2 | realsense | xtion
    rgb_rel: str
    depth_rel: str
    K: np.ndarray
    Rtilt: np.ndarray
    raw: object = None    # the mat_struct, kept for annotation parsing


def _rel(path: str) -> str:
    return path[len(REMOTE_PREFIX):] if path.startswith(REMOTE_PREFIX) else path.lstrip("/")


def load_meta(meta_path: str | Path) -> list[SunRecord]:
    m = sio.loadmat(str(meta_path), squeeze_me=True, struct_as_record=False)["SUNRGBDMeta"]
    return [
        SunRecord(index=i, sequence=r.sequenceName, sensor=r.sensorType, rgb_rel=_rel(r.rgbpath),
                  depth_rel=_rel(r.depthpath), K=np.asarray(r.K, dtype=np.float64),
                  Rtilt=np.asarray(r.Rtilt, dtype=np.float64), raw=r)
        for i, r in enumerate(m)
    ]


class SunRGBD:
    def __init__(self, root: str | Path, records: list[SunRecord] | None = None,
                 min_m: float = 0.1, max_m: float = 10.0):
        self.root = Path(root)  # .../sunrgbd (contains SUNRGBD/ and the .mat files)
        self.records = records if records is not None else load_meta(self.root / "SUNRGBDMeta3DBB_v2.mat")
        self.min_m, self.max_m = min_m, max_m

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Frame:
        r = self.records[idx]
        bgr = cv2.imread(str(self.root / r.rgb_rel), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(self.root / r.depth_rel), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None:
            raise IOError(f"failed to read SUN record {r.index}: {r.rgb_rel}")
        if raw.dtype != np.uint16:
            raise ValueError(f"expected 16-bit depth, got {raw.dtype}")
        if raw.shape != bgr.shape[:2]:
            raise ValueError(f"record {r.index}: depth {raw.shape} != rgb {bgr.shape[:2]}")
        depth, valid = decode_sun(raw, self.min_m, self.max_m)
        return Frame(
            rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            depth_m=depth,
            valid=valid,
            K=r.K.copy(),
            frame_id=f"{r.index + 1:06d}",
            dataset="sunrgbd",
            scene_id=r.sequence,
        )
