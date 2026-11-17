"""3D attachment and per-track history (spec 04.1 R4, R5).

Track file (JSONL, `benchmarks/tracking/<run>.jsonl`). Line 1 is `{"header": {...}}` with the result-hygiene fields
of evaluation protocol §7 (git commit and dirty flag, config hashes, the source detection run's hash, timestamp).
Every later line is one processed frame:
`{frame_index, frame_id, timestamp, n_dets, tracks: [{track_id, cls, det_index, box, score, status, n_inliers,
centre_cam, centre_world}], timings_ms}`. NaN centres are written as null. `centre_world` is null when the frame
has no pose or the centre is NaN.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from perception.depth.estimate import AssocParams, estimate_box
from perception.geometry.object3d import object_centre, to_world
from perception.tracking.bytetrack import TrackedBox


@dataclass(frozen=True)
class Observation:
    frame_index: int                 # index in the processed frame stream (0, 1, 2, ... at any stride)
    seq_index: int                   # index in the sequence (frame_index * stride)
    timestamp: float
    track_id: int
    cls: int
    det_index: int
    box: tuple[float, float, float, float]
    score: float
    status: str                      # "valid" | "low_coverage" | "no_depth"
    n_inliers: int
    centre_cam: tuple[float, float, float] | None
    centre_world: tuple[float, float, float] | None


def _finite3(v) -> tuple[float, float, float] | None:
    v = tuple(float(x) for x in v)
    return v if all(math.isfinite(x) for x in v) else None


def attach_3d(frame, tracked: Iterable[TrackedBox], p: AssocParams, k: float, sigma_floor: float,
              frame_index: int, seq_index: int) -> list[Observation]:
    """Spec 04.1 R4: estimate_box + object_centre on each tracked box; centre_world = T_world_cam * centre_cam."""
    obs = []
    for tb in tracked:
        est = estimate_box(frame.depth_m, frame.valid, tb.box_xyxy, p)
        o3 = object_centre(frame.depth_m, frame.valid, frame.K, est, k, sigma_floor, frame.dist)
        cc = _finite3(o3.centre)
        cw = None
        if cc is not None and frame.T_world_cam is not None:
            cw = _finite3(to_world(np.asarray(cc), frame.T_world_cam)[0])
        obs.append(Observation(frame_index, seq_index, float(frame.timestamp), tb.track_id, tb.cls, tb.det_index,
                               tb.box_xyxy, tb.score, o3.status, o3.n_inliers, cc, cw))
    return obs


@dataclass
class TrackHistory:
    """Per-track observation lists, in processing order."""

    tracks: dict[int, list[Observation]] = field(default_factory=dict)

    def add(self, observations: Iterable[Observation]) -> None:
        for o in observations:
            self.tracks.setdefault(o.track_id, []).append(o)

    def confirmed(self, min_len: int) -> dict[int, list[Observation]]:
        return {i: v for i, v in self.tracks.items() if len(v) >= min_len}

    @classmethod
    def from_frames(cls, frames: Iterable[dict]) -> "TrackHistory":
        h = cls()
        for fr in frames:
            h.add(observations_of(fr))
        return h


# ---- JSONL --------------------------------------------------------------------------------------------------------

def _r(v, nd=6):
    return None if v is None else [round(float(x), nd) for x in v]


def frame_line(frame_index: int, seq_index: int, frame_id: str, timestamp: float, n_dets: int,
               obs: list[Observation], timings_ms: dict[str, float]) -> dict[str, Any]:
    return {
        "frame_index": frame_index, "seq_index": seq_index, "frame_id": frame_id, "timestamp": timestamp,
        "n_dets": int(n_dets),
        "tracks": [{"track_id": o.track_id, "cls": o.cls, "det_index": o.det_index,
                    "box": [float(x) for x in o.box], "score": float(o.score), "status": o.status,
                    "n_inliers": o.n_inliers, "centre_cam": _r(o.centre_cam), "centre_world": _r(o.centre_world)}
                   for o in obs],
        "timings_ms": {k: round(v, 4) for k, v in timings_ms.items()},
    }


def observations_of(fr: dict) -> list[Observation]:
    t = lambda v: None if v is None else tuple(v)  # noqa: E731
    return [Observation(fr["frame_index"], fr["seq_index"], fr["timestamp"], d["track_id"], d["cls"], d["det_index"],
                        tuple(d["box"]), d["score"], d["status"], d["n_inliers"], t(d["centre_cam"]),
                        t(d["centre_world"])) for d in fr["tracks"]]


def write_tracks(path: str | Path, header: dict, lines: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"header": header}, separators=(",", ":")) + "\n")
        for ln in lines:
            f.write(json.dumps(ln, separators=(",", ":")) + "\n")


def read_tracks(path: str | Path) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as f:
        first = json.loads(f.readline())
        if "header" not in first:
            raise ValueError(f"{path}: missing header line")
        return first["header"], [json.loads(ln) for ln in f if ln.strip()]


# ---- provenance (protocol §7) -------------------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(repo: Path) -> dict[str, Any]:
    try:
        r = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=repo, capture_output=True, text=True,
                           timeout=10)
        commit = r.stdout.strip() if r.returncode == 0 else None  # None: repository has no commits yet
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True,
                                    timeout=30).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def run_header(repo: Path, configs: dict[str, Path], inputs: dict[str, Path], extra: dict | None = None) -> dict:
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None

    return {
        **git_state(repo),
        "config_sha256": {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in configs.items()},
        "input_sha256": {k: sha256_file(v) for k, v in inputs.items()},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {p: ver(p) for p in ("ultralytics", "lap", "numpy", "opencv-python")},
        "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
        **(extra or {}),
    }
