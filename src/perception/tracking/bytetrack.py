"""ByteTrack adapter (spec 04.1 R1-R3).

Wraps `ultralytics.trackers.byte_tracker.BYTETracker` (Ultralytics 8.4.208, AGPL-3.0) behind a plain-numpy interface:
`Tracker.update(boxes_xyxy, scores, classes) -> list[TrackedBox]`.

Decisions recorded here:
- **Only boxes matched in this frame are output (R1).** Coasting tracks (Kalman prediction only) are never returned,
  so no 3D is made from a prediction. Each output carries `det_index`, the row of the input detection it matched, and
  `box_xyxy` is that detection's box (the measurement), not the Kalman-filtered box.
- **Class-aware by default (R2).** Ultralytics ByteTrack ignores class when matching. With `class_aware` the adapter
  runs one BYTETracker per class. They share one ID counter, so IDs are unique across classes. Every class tracker
  is stepped on every frame (with no detections if the class is absent), so lost tracks age at the same rate. A tracker
  created mid-sequence starts at the current frame count, so its new tracks need one confirming frame like any other
  (Ultralytics activates new tracks immediately only on frame 1).
- **No hidden installs.** Ultralytics' matching module pip-installs `lap` when it is missing. `lap` is pinned in
  pyproject.toml, and it is imported here first so a missing package fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    import lap  # noqa: F401  # required by ultralytics.trackers.utils.matching; must not be auto-installed
except ImportError as e:  # pragma: no cover
    raise ImportError("perception.tracking needs `lap` (pinned in pyproject.toml, optional deps 'detection'); "
                      "install it explicitly instead of letting ultralytics pip-install it") from e

from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: E402


@dataclass(frozen=True)
class TrackerParams:
    track_high_thresh: float = 0.2411
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.2411
    track_buffer: int = 30           # processed frames
    match_thresh: float = 0.8
    fuse_score: bool = True
    class_aware: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any], stride: int | None = None, **overrides) -> "TrackerParams":
        """configs/tracker.yaml -> params. `per_stride[stride]` (written by 04.2 tuning) overrides the base values."""
        p: dict[str, Any] = dict(cfg["bytetrack"])
        p["class_aware"] = bool(cfg.get("class_aware", True))
        if stride is not None:
            p.update((cfg.get("per_stride") or {}).get(int(stride), {}) or {})
        p.update(overrides)
        names = {f.name for f in fields(cls)}
        unknown = set(p) - names
        if unknown:
            raise ValueError(f"unknown tracker parameters {sorted(unknown)}")
        p["track_buffer"] = int(p["track_buffer"])
        return cls(**p)

    def with_(self, **kw) -> "TrackerParams":
        return replace(self, **kw)


@dataclass(frozen=True)
class TrackedBox:
    track_id: int
    box_xyxy: tuple[float, float, float, float]
    score: float
    cls: int
    det_index: int


class _Dets:
    """The minimal Results-like object BYTETracker.update needs: conf, cls, xywh, boolean indexing and len()."""

    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    @classmethod
    def from_xyxy(cls, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> "_Dets":
        b = boxes.astype(np.float32, copy=False)
        xywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        return cls(xywh.reshape(-1, 4), scores.astype(np.float32, copy=False), classes.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, m) -> "_Dets":
        return _Dets(self.xywh[m], self.conf[m], self.cls[m])


class Tracker:
    """2D multi-object tracker. Call `update` once per processed frame, in order."""

    def __init__(self, params: TrackerParams | None = None):
        self.params = params or TrackerParams()
        self.reset()

    def reset(self) -> None:
        self.frame_count = 0
        self._ids = iter(range(1, 2**62))  # shared by all class trackers -> IDs unique across classes
        self._trackers: dict[int, BYTETracker] = {}

    def _args(self) -> SimpleNamespace:
        p = self.params
        return SimpleNamespace(track_high_thresh=p.track_high_thresh, track_low_thresh=p.track_low_thresh,
                               new_track_thresh=p.new_track_thresh, track_buffer=p.track_buffer,
                               match_thresh=p.match_thresh, fuse_score=p.fuse_score)

    def _new_tracker(self) -> BYTETracker:
        tr = BYTETracker(self._args())
        tr._ids = self._ids
        tr.frame_id = self.frame_count - 1  # update() increments it to the current frame count
        return tr

    def update(self, boxes_xyxy, scores, classes) -> list[TrackedBox]:
        boxes = np.asarray(boxes_xyxy, np.float64).reshape(-1, 4)
        scores = np.asarray(scores, np.float64).reshape(-1)
        classes = np.asarray(classes, np.int64).reshape(-1)
        if not (len(boxes) == len(scores) == len(classes)):
            raise ValueError(f"length mismatch: {len(boxes)} boxes, {len(scores)} scores, {len(classes)} classes")
        self.frame_count += 1

        # Detections at or below track_low_thresh are never used by ByteTrack; dropping them first is only faster.
        keep = np.flatnonzero(scores > self.params.track_low_thresh)
        groups: dict[int, np.ndarray]
        if self.params.class_aware:
            groups = {int(c): keep[classes[keep] == c] for c in np.unique(classes[keep])}
            for c in groups:
                if c not in self._trackers:
                    self._trackers[c] = self._new_tracker()
        else:
            if -1 not in self._trackers:
                self._trackers[-1] = self._new_tracker()
            groups = {-1: keep}

        out: list[TrackedBox] = []
        empty = np.empty(0, np.int64)
        for key in sorted(self._trackers):  # fixed order -> deterministic ID assignment (R7)
            tr = self._trackers[key]
            sub = groups.get(key, empty)
            if sub.size == 0 and not tr.tracked_stracks and not tr.lost_stracks:
                tr.frame_id += 1  # idle tracker: update() with no input and no tracks only advances the frame
                continue
            tr.update(_Dets.from_xyxy(boxes[sub], scores[sub], classes[sub]))
            for t in tr.tracked_stracks:
                if not (t.is_activated and t.frame_id == tr.frame_id):  # matched in this frame only (R1)
                    continue
                d = int(sub[int(t.idx)])
                out.append(TrackedBox(int(t.track_id), tuple(map(float, boxes[d])), float(scores[d]),
                                      int(classes[d]), d))
        out.sort(key=lambda tb: tb.det_index)
        return out

    @property
    def n_trackers(self) -> int:
        return len(self._trackers)
