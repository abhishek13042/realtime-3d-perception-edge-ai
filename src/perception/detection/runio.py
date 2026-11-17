"""Detection run files and latency statistics (spec 02.2 R1, R3).

A run is a JSONL file. Line 1 is a header `{"header": {...provenance...}}`, and every later line is one frame:
`{frame_id, dataset, scene_id, sensor, image_hw, boxes, scores, class_ids, timings_ms}` (boxes in original-image px).
Lines are flushed one by one. A crash leaves at most one truncated last line, which `read_run` drops and
`completed_ids` ignores, so `--resume` redoes that frame.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from perception.detection.base import Detections


def provenance(cfg: dict, cfg_path: Path, detector, extra: dict | None = None) -> dict[str, Any]:
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None

    repo = cfg_path.resolve().parents[1]
    try:
        r = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=repo, capture_output=True, text=True,
                           timeout=10)
        commit = r.stdout.strip() if r.returncode == 0 else None  # None: repository has no commits yet
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True,
                                    timeout=30).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    import torch

    return {
        "weights": str(cfg["weights"]), "weights_sha256": detector.weights_sha256, "head": detector.head,
        "imgsz": detector.imgsz, "conf": detector.conf, "max_det": detector.max_det,
        "iou_nms": detector.iou if detector.head == "one2many_nms" else None,
        "threads": torch.get_num_threads(),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "git_commit": commit, "git_dirty": dirty,
        "versions": {p: ver(p) for p in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python")},
        "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
        **(extra or {}),
    }


def frame_record(det: Detections, frame_id: str, dataset: str, scene_id: str, sensor: str | None,
                 image_hw: tuple[int, int], timings_ms: dict[str, float]) -> dict[str, Any]:
    return {
        "frame_id": frame_id, "dataset": dataset, "scene_id": scene_id, "sensor": sensor,
        "image_hw": [int(image_hw[0]), int(image_hw[1])],
        # float32 -> repr of its float64 value keeps it exact on reload (02.2 AC2 compares exactly).
        "boxes": det.boxes.astype(np.float64).tolist(), "scores": det.scores.astype(np.float64).tolist(),
        "class_ids": det.class_ids.tolist(),
        "timings_ms": {k: round(v, 3) for k, v in timings_ms.items()},
    }


def record_to_detections(rec: dict) -> Detections:
    return Detections(np.asarray(rec["boxes"], np.float32).reshape(-1, 4), np.asarray(rec["scores"], np.float32),
                      np.asarray(rec["class_ids"], np.int64))


def _lines(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line after a crash


def read_run(path: str | Path) -> tuple[dict, list[dict]]:
    it = _lines(Path(path))
    first = next(it, None)
    if first is None or "header" not in first:
        raise ValueError(f"{path}: missing header line")
    return first["header"], list(it)


def completed_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    return {r["frame_id"] for r in _lines(path) if "frame_id" in r}


class RunWriter:
    """Appends frame lines. On a fresh file it writes the header. On resume it checks that the header matches."""

    CHECK = ("weights_sha256", "head", "imgsz", "conf", "max_det", "iou_nms", "config_sha256")

    def __init__(self, path: str | Path, header: dict, resume: bool):
        self.path = Path(path)
        if self.path.exists() and self.path.stat().st_size > 0:
            if not resume:
                raise FileExistsError(f"{self.path} exists; pass --resume to continue it or delete it")
            old, _ = read_run(self.path)
            diff = {k: (old.get(k), header.get(k)) for k in self.CHECK if old.get(k) != header.get(k)}
            if diff:
                raise ValueError(f"cannot resume {self.path}: settings differ {diff}")
            self._drop_truncated_tail()
            self.f = open(self.path, "a", encoding="utf-8")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(self.path, "w", encoding="utf-8")
            self._write({"header": header})

    def _drop_truncated_tail(self) -> None:
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            self.path.write_bytes(data[: data.rfind(b"\n") + 1])

    def _write(self, obj: dict) -> None:
        self.f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self.f.flush()

    def write(self, rec: dict) -> None:
        self._write(rec)

    def close(self) -> None:
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def latency_stats(values_ms) -> dict[str, float]:
    """Median / p95 / mean / min / max over a list of per-frame times (ms). p95 uses linear interpolation."""
    a = np.asarray(values_ms, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": round(float(np.median(a)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "mean": round(float(a.mean()), 3), "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}
