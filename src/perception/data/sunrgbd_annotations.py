"""SUN RGB-D 2D/3D box annotations (spec 01.3, R1–R3, R7).

Source: SUNRGBDMeta3DBB_v2.mat (3D boxes, upright frame) and SUNRGBDMeta2DBB_v2.mat (2D boxes).
Both files list the same 10,335 images in the same order; the parser asserts this.

Parsed annotations are cached as compact JSON next to the data (R7), so later phases do not hold the
~10 MB mat structs (several hundred MB as Python objects) in RAM.

Conventions:
- image_id = f"{index + 1:06d}" (1-based position in the metadata, same as Frame.frame_id).
- 3D boxes keep the raw upright parameters (centroid, basis rows, half sizes `coeffs`); camera-frame
  corners are computed on demand with perception.geometry.transforms.
- 2D boxes are converted from MATLAB [x, y, w, h] (1-based) to 0-based [x1, y1, x2, y2] in pixels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

from perception.data.sunrgbd import _rel
from perception.geometry.transforms import box_corners, upright_to_cam

CACHE_VERSION = 1


def normalize_class(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


@dataclass
class Box3D:
    cls: str                 # normalized class name
    raw_cls: str
    centroid: list[float]    # upright frame (x right, y forward, z up), metres
    basis: list[list[float]] # 3×3, rows are box axes in the upright frame
    coeffs: list[float]      # half sizes along the basis rows, metres
    orientation: list[float]

    def corners_upright(self) -> np.ndarray:
        return box_corners(self.centroid, self.basis, self.coeffs)

    def corners_cam(self, Rtilt: np.ndarray) -> np.ndarray:
        return upright_to_cam(self.corners_upright(), Rtilt)

    def centroid_cam(self, Rtilt: np.ndarray) -> np.ndarray:
        return upright_to_cam(np.asarray(self.centroid), Rtilt)

    @property
    def size(self) -> np.ndarray:
        return 2.0 * np.asarray(self.coeffs)


@dataclass
class Box2D:
    cls: str
    raw_cls: str
    xyxy: list[float]        # 0-based pixels
    has3d: bool


@dataclass
class ImageAnn:
    image_id: str
    index: int
    sequence: str
    sensor: str
    rgb_rel: str
    depth_rel: str
    K: list[list[float]]
    Rtilt: list[list[float]]
    boxes3d: list[Box3D] = field(default_factory=list)
    boxes2d: list[Box2D] = field(default_factory=list)

    @property
    def Rtilt_np(self) -> np.ndarray:
        return np.asarray(self.Rtilt, dtype=np.float64)

    @property
    def K_np(self) -> np.ndarray:
        return np.asarray(self.K, dtype=np.float64)


def _as_list(x) -> list:
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return []
        return list(np.atleast_1d(x))
    return [x]


def _r(a, nd: int = 6):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def parse(meta3d_path: str | Path, meta2d_path: str | Path) -> list[ImageAnn]:
    m3 = sio.loadmat(str(meta3d_path), squeeze_me=True, struct_as_record=False)["SUNRGBDMeta"]
    m2 = sio.loadmat(str(meta2d_path), squeeze_me=True, struct_as_record=False)["SUNRGBDMeta2DBB"]
    if len(m3) != len(m2):
        raise ValueError(f"3D meta has {len(m3)} images, 2D meta has {len(m2)}")
    out: list[ImageAnn] = []
    for i, (r3, r2) in enumerate(zip(m3, m2)):
        if r3.sequenceName != r2.sequenceName:
            raise ValueError(f"index {i}: 3D/2D order mismatch {r3.sequenceName} vs {r2.sequenceName}")
        boxes3d = [
            Box3D(cls=normalize_class(b.classname), raw_cls=str(b.classname), centroid=_r(b.centroid),
                  basis=_r(b.basis), coeffs=_r(b.coeffs), orientation=_r(b.orientation))
            for b in _as_list(r3.groundtruth3DBB)
        ]
        boxes2d = []
        for b in _as_list(r2.groundtruth2DBB):
            x, y, w, h = (float(v) for v in np.asarray(b.gtBb2D).ravel()[:4])
            boxes2d.append(Box2D(cls=normalize_class(b.classname), raw_cls=str(b.classname),
                                 xyxy=[x - 1.0, y - 1.0, x - 1.0 + w, y - 1.0 + h], has3d=bool(b.has3dbox)))
        out.append(ImageAnn(image_id=f"{i + 1:06d}", index=i, sequence=str(r3.sequenceName),
                            sensor=str(r3.sensorType), rgb_rel=_rel(r3.rgbpath), depth_rel=_rel(r3.depthpath),
                            K=_r(r3.K), Rtilt=_r(r3.Rtilt), boxes3d=boxes3d, boxes2d=boxes2d))
    return out


def save_cache(anns: list[ImageAnn], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": CACHE_VERSION, "images": [asdict(a) for a in anns]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def load_cache(path: str | Path) -> list[ImageAnn]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != CACHE_VERSION:
        raise ValueError(f"cache version {payload.get('version')} != {CACHE_VERSION}; rebuild it")
    anns = []
    for d in payload["images"]:
        d["boxes3d"] = [Box3D(**b) for b in d["boxes3d"]]
        d["boxes2d"] = [Box2D(**b) for b in d["boxes2d"]]
        anns.append(ImageAnn(**d))
    return anns


def load_annotations(sun_root: str | Path, rebuild: bool = False) -> list[ImageAnn]:
    """Load from the JSON cache, building it from the .mat files on first use."""
    sun_root = Path(sun_root)
    cache = sun_root / "_cache" / f"annotations_v{CACHE_VERSION}.json"
    if cache.exists() and not rebuild:
        return load_cache(cache)
    anns = parse(sun_root / "SUNRGBDMeta3DBB_v2.mat", sun_root / "SUNRGBDMeta2DBB_v2.mat")
    save_cache(anns, cache)
    return anns
