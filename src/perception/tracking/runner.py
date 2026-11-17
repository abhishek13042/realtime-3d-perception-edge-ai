"""Stored detections -> tracks + 3D over a TUM sequence (spec 04.1 R4-R6). Used by run_tracking.py and 04.2.

The 3D fields of a box depend only on the frame and the box, never on the tracker settings, so `Centres3D` caches
them per (sequence index, detection index). The 04.2 tuning grid then pays the depth cost once per box.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from perception.depth.estimate import AssocParams
from perception.tracking.bytetrack import TrackedBox, Tracker, TrackerParams
from perception.tracking.track_history import Observation, attach_3d, frame_line


def align_records(seq, records: list[dict]) -> list[dict]:
    """Detection records in sequence order. Every sequence frame must have one (02.2 runs cover all frames)."""
    by_id = {r["frame_id"]: r for r in records}
    ids = [f"{t:.6f}" for t, _, _ in seq.items]
    missing = [f for f in ids if f not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} sequence frames have no detection record, e.g. {missing[:3]}")
    return [by_id[f] for f in ids]


class Centres3D:
    """Cached 3D attachment: (seq_index, TrackedBox) -> Observation fields from Phase 03 code (frozen depth.yaml)."""

    def __init__(self, seq, depth_cfg: dict):
        self.seq = seq
        self.p = AssocParams.from_config(depth_cfg)
        self.k = float(depth_cfg["centre"]["k"])
        self.floor = float(depth_cfg["centre"]["sigma_floor_m"])
        self._cache: dict[tuple[int, int], Observation] = {}
        self._frame = (None, None)
        self.ms = 0.0  # time spent on frame loading + 3D (not the tracker)

    def frame(self, seq_index: int):
        if self._frame[0] != seq_index:
            self._frame = (seq_index, self.seq[seq_index])
        return self._frame[1]

    def __call__(self, frame_index: int, seq_index: int, tracked: list[TrackedBox]) -> list[Observation]:
        t0 = time.perf_counter()
        todo = [tb for tb in tracked if (seq_index, tb.det_index) not in self._cache]
        if todo:
            fr = self.frame(seq_index)
            # Cache under a neutral track id; the caller's id is put back below.
            for tb, o in zip(todo, attach_3d(fr, todo, self.p, self.k, self.floor, 0, seq_index)):
                self._cache[(seq_index, tb.det_index)] = o
        out = []
        for tb in tracked:
            c = self._cache[(seq_index, tb.det_index)]
            out.append(Observation(frame_index, seq_index, c.timestamp, tb.track_id, tb.cls, tb.det_index,
                                   tb.box_xyxy, tb.score, c.status, c.n_inliers, c.centre_cam, c.centre_world))
        self.ms += (time.perf_counter() - t0) * 1e3
        return out


def track_sequence(records: list[dict], params: TrackerParams, stride: int,
                   attach: Callable[[int, int, list[TrackedBox]], list[Observation]]) -> tuple[list[dict], list[float]]:
    """Feed every `stride`-th record (sequence order) to a fresh Tracker. Returns the frame lines and the per-frame
    tracker-update times (ms, excluding 3D)."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    tracker = Tracker(params)
    lines, upd_ms = [], []
    for frame_index, seq_index in enumerate(range(0, len(records), stride)):
        r = records[seq_index]
        boxes = np.asarray(r["boxes"], np.float64).reshape(-1, 4)
        t0 = time.perf_counter()
        tracked = tracker.update(boxes, r["scores"], r["class_ids"])
        dt = (time.perf_counter() - t0) * 1e3
        upd_ms.append(dt)
        obs = attach(frame_index, seq_index, tracked)
        lines.append(frame_line(frame_index, seq_index, r["frame_id"], float(r["frame_id"]), len(boxes), obs,
                                {"tracker": dt}))
    return lines, upd_ms
