"""End-to-end per-frame pipeline (spec 05.3 R1): Frame -> FrameResult (problem statement §4).

Every stage is an existing function; nothing here re-implements their logic:
- detect: `Detector.detect` (02.1), which reports its own preprocess / infer / postprocess times;
- track: `Tracker.update` (04.1; per-stride parameters from configs/tracker.yaml);
- depth: `estimate_box` (03.1, frozen configs/depth.yaml);
- geometry: `object_centre` with the frame's distortion, and `to_world` when a pose is known (03.2).

Stage order: detect -> track -> depth -> geometry. Tracking runs before depth only so the depth stage knows which
low-score boxes the tracker kept (ByteTrack's second stage). The tracker never sees depth, so the order does not
change any output (the 04.1 offline runner attaches 3D after tracking too).

Which boxes become objects:
- every detection with score >= conf_operational (02.3), and
- every box the tracker matched in this frame (this includes second-stage boxes with a lower score).
`track_id` is the tracker's ID once that track has been matched in >= min_track_len frames (tracker.yaml; counted
causally, up to and including this frame), else null. In single-image mode (`track=False`) it is always null.

Null rules (§4): `depth_m` is null when the status is `no_depth`; `depth_spread_m` and `center_cam_m` are null when
there is no depth (or the value is not finite). No NaN ever reaches the JSON.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from perception.data.frame import Frame
from perception.depth.estimate import AssocParams, estimate_box
from perception.detection.classes import COCO_NAMES
from perception.geometry.object3d import object_centre, to_world
from perception.tracking.bytetrack import Tracker, TrackerParams
from perception.tracking.track_history import Observation

STATUSES = ("valid", "low_coverage", "no_depth")
TIMING_KEYS = ("preprocess", "detect", "depth", "geometry", "track", "total")
OBJECT_KEYS = ("track_id", "class_name", "score", "bbox_xyxy", "depth_status", "depth_m", "depth_spread_m",
               "valid_depth_fraction", "center_cam_m")


def _num(v: float) -> float | None:
    return float(v) if v is not None and math.isfinite(v) else None


def _vec(v) -> tuple[float, float, float] | None:
    if v is None:
        return None
    v = tuple(float(x) for x in v)
    return v if all(math.isfinite(x) for x in v) else None


@dataclass
class ObjectResult:
    track_id: int | None              # confirmed track ID or None (§4)
    class_name: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    depth_status: str
    depth_m: float | None
    depth_spread_m: float | None
    valid_depth_fraction: float
    center_cam_m: tuple[float, float, float] | None
    # Internal (not part of the §4 JSON): used by the equivalence check and the reliability metrics.
    cls: int = -1
    det_index: int = -1
    raw_track_id: int | None = None   # tracker ID even before confirmation; None if not matched this frame
    n_inliers: int = 0
    centre_world: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"track_id": self.track_id, "class_name": self.class_name, "score": float(self.score),
                "bbox_xyxy": [float(x) for x in self.bbox_xyxy], "depth_status": self.depth_status,
                "depth_m": self.depth_m, "depth_spread_m": self.depth_spread_m,
                "valid_depth_fraction": float(self.valid_depth_fraction),
                "center_cam_m": None if self.center_cam_m is None else list(self.center_cam_m)}


@dataclass
class FrameResult:
    frame_id: str
    timestamp_s: float | None         # None for single images (SUN RGB-D has no timestamps)
    objects: list[ObjectResult]
    timing_ms: dict[str, float]
    # Internal: protocol §5 `detector` boundary (model forward only), the raw detection count, frame indices.
    detector_ms: float = 0.0
    n_dets: int = 0
    frame_index: int = 0
    seq_index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "timestamp_s": self.timestamp_s,
                "objects": [o.to_dict() for o in self.objects],
                "timing_ms": {k: round(float(self.timing_ms[k]), 4) for k in TIMING_KEYS}}

    def observations(self) -> list[Observation]:
        """Tracked objects as 04.1 Observations (same fields the offline runner writes), in det_index order."""
        ts = float(self.timestamp_s) if self.timestamp_s is not None else math.nan
        return [Observation(self.frame_index, self.seq_index, ts, o.raw_track_id, o.cls, o.det_index,
                            tuple(o.bbox_xyxy), o.score, o.depth_status, o.n_inliers, o.center_cam_m,
                            o.centre_world)
                for o in self.objects if o.raw_track_id is not None]


def validate_result(d: dict[str, Any]) -> None:
    """Raise ValueError when a serialised FrameResult breaks the §4 contract (schema and null rules)."""
    if set(d) != {"frame_id", "timestamp_s", "objects", "timing_ms"}:
        raise ValueError(f"top-level keys {sorted(d)}")
    if not isinstance(d["frame_id"], str):
        raise ValueError("frame_id must be a string")
    if d["timestamp_s"] is not None and not isinstance(d["timestamp_s"], (int, float)):
        raise ValueError("timestamp_s must be a number or null")
    if set(d["timing_ms"]) != set(TIMING_KEYS):
        raise ValueError(f"timing_ms keys {sorted(d['timing_ms'])}")
    if any(not isinstance(v, (int, float)) or v < 0 for v in d["timing_ms"].values()):
        raise ValueError("timing_ms values must be non-negative numbers")
    for o in d["objects"]:
        if set(o) != set(OBJECT_KEYS):
            raise ValueError(f"object keys {sorted(o)}")
        if o["track_id"] is not None and not isinstance(o["track_id"], int):
            raise ValueError("track_id must be an int or null")
        if not (0.0 <= o["score"] <= 1.0):
            raise ValueError(f"score {o['score']} outside [0, 1]")
        if len(o["bbox_xyxy"]) != 4 or o["bbox_xyxy"][2] < o["bbox_xyxy"][0] or o["bbox_xyxy"][3] < o["bbox_xyxy"][1]:
            raise ValueError(f"bad bbox {o['bbox_xyxy']}")
        if o["depth_status"] not in STATUSES:
            raise ValueError(f"depth_status {o['depth_status']!r}")
        if not (0.0 <= o["valid_depth_fraction"] <= 1.0):
            raise ValueError("valid_depth_fraction outside [0, 1]")
        if o["depth_status"] == "no_depth":
            if o["depth_m"] is not None or o["depth_spread_m"] is not None or o["center_cam_m"] is not None:
                raise ValueError("no_depth object must have null depth_m, depth_spread_m and center_cam_m")
        elif o["depth_m"] is None:
            raise ValueError(f"{o['depth_status']} object must have a depth_m")
        if o["center_cam_m"] is not None and len(o["center_cam_m"]) != 3:
            raise ValueError("center_cam_m must be [X, Y, Z]")


class Pipeline:
    """Frame -> FrameResult. One instance per stream; `reset()` before a new sequence."""

    def __init__(self, detector, depth_cfg: dict, tracker_cfg: dict, stride: int = 1, *,
                 conf_operational: float, track: bool = True, names=COCO_NAMES):
        self.detector = detector
        self.p = AssocParams.from_config(depth_cfg)
        self.k = float(depth_cfg["centre"]["k"])
        self.floor = float(depth_cfg["centre"]["sigma_floor_m"])
        self.params = TrackerParams.from_config(tracker_cfg, stride=stride)
        self.min_len = int(tracker_cfg["min_track_len"])
        self.conf_op = float(conf_operational)
        self.track = track
        self.names = names
        self.reset()

    def reset(self) -> None:
        self.tracker = Tracker(self.params) if self.track else None
        self._hits: dict[int, int] = {}
        self.frame_index = 0

    def process(self, frame: Frame, seq_index: int | None = None) -> FrameResult:
        t0 = time.perf_counter()
        det = self.detector.detect(frame.rgb)
        t1 = time.perf_counter()
        boxes = det.boxes.astype(np.float64)
        tracked = []
        if self.tracker is not None:
            tracked = self.tracker.update(boxes, det.scores, det.class_ids)
            for tb in tracked:
                self._hits[tb.track_id] = self._hits.get(tb.track_id, 0) + 1
        t2 = time.perf_counter()

        by_det = {tb.det_index: tb for tb in tracked}
        idx = sorted(set(np.flatnonzero(det.scores >= self.conf_op).tolist()) | set(by_det))
        # Tracked boxes use the tracker's copy (identical values) so the inputs match the offline runner exactly.
        items = [(i, by_det[i].box_xyxy if i in by_det else tuple(map(float, boxes[i]))) for i in idx]
        ests = [estimate_box(frame.depth_m, frame.valid, b, self.p) for _, b in items]
        t3 = time.perf_counter()
        o3s = [object_centre(frame.depth_m, frame.valid, frame.K, e, self.k, self.floor, frame.dist) for e in ests]
        worlds = []
        for o3 in o3s:
            cc = _vec(o3.centre)
            worlds.append(_vec(to_world(np.asarray(cc), frame.T_world_cam)[0])
                          if cc is not None and frame.T_world_cam is not None else None)
        t4 = time.perf_counter()

        objects = []
        for (i, b), e, o3, cw in zip(items, ests, o3s, worlds):
            tb = by_det.get(i)
            has_depth = e.status != "no_depth"
            confirmed = tb is not None and self._hits[tb.track_id] >= self.min_len
            objects.append(ObjectResult(
                track_id=tb.track_id if confirmed else None, class_name=self.names[int(det.class_ids[i])],
                score=float(det.scores[i]), bbox_xyxy=tuple(b), depth_status=e.status,
                depth_m=_num(e.z) if has_depth else None, depth_spread_m=_num(e.spread) if has_depth else None,
                valid_depth_fraction=float(e.valid_frac), center_cam_m=_vec(o3.centre) if has_depth else None,
                cls=int(det.class_ids[i]), det_index=int(i), raw_track_id=None if tb is None else tb.track_id,
                n_inliers=int(o3.n_inliers), centre_world=cw))
        t5 = time.perf_counter()

        tm = det.timings_ms
        timing = {"preprocess": tm["preprocess"], "detect": tm["infer"] + tm["postprocess"],
                  "track": (t2 - t1) * 1e3, "depth": (t3 - t2) * 1e3, "geometry": (t4 - t3) * 1e3,
                  "total": (t5 - t0) * 1e3}
        fi = self.frame_index
        self.frame_index += 1
        return FrameResult(frame.frame_id, None if frame.timestamp is None else float(frame.timestamp), objects,
                           timing, detector_ms=tm["infer"], n_dets=len(det), frame_index=fi,
                           seq_index=fi if seq_index is None else int(seq_index),
                           extra={"assemble_ms": (t5 - t4) * 1e3, "detect_total_ms": (t1 - t0) * 1e3})
