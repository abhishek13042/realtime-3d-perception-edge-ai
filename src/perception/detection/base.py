"""Detector interface (spec 02.1 R1, R3).

A backend implements only `_infer(chw_float_batch) -> (N, 6) array [x1, y1, x2, y2, score, class_id]` in
letterbox pixels. Everything else is shared, so the PyTorch and ONNX backends (Phase 06) differ only in the runtime:
- the letterbox,
- the conversion to float,
- the inverse mapping to original-image pixels,
- thresholds, max_det, and class filtering.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from perception.detection.classes import COCO_NAMES
from perception.detection.letterbox import letterbox, unletterbox_boxes


@dataclass
class Detections:
    """Detections for one image, in original-image pixels (image spans [0, W] x [0, H])."""

    boxes: np.ndarray      # (N, 4) float32 xyxy
    scores: np.ndarray     # (N,) float32, sorted descending
    class_ids: np.ndarray  # (N,) int64, COCO index 0-79
    timings_ms: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.scores)

    @property
    def class_names(self) -> list[str]:
        return [COCO_NAMES[i] for i in self.class_ids]

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))


class Detector(ABC):
    """Base class: RGB uint8 image in, Detections in original-image pixels out."""

    def __init__(self, imgsz: int = 640, conf: float = 0.25, max_det: int = 300,
                 classes: list[int] | None = None, pad_value: int = 114):
        self.imgsz = imgsz
        self.conf = conf
        self.max_det = max_det
        self.classes = None if classes is None else np.asarray(sorted(classes), dtype=np.int64)
        self.pad_value = pad_value

    def preprocess(self, rgb: np.ndarray):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected HxWx3 uint8 RGB, got {rgb.dtype} {rgb.shape}")
        canvas, info = letterbox(rgb, self.imgsz, self.pad_value)
        x = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255.0)
        return x, info

    @abstractmethod
    def _infer(self, x: np.ndarray) -> np.ndarray:
        """x: (1, 3, S, S) float32 in [0, 1], RGB. Returns (N, 6) [x1, y1, x2, y2, score, cls] in letterbox px."""

    def postprocess(self, raw: np.ndarray, info) -> Detections:
        raw = np.asarray(raw, dtype=np.float32).reshape(-1, 6)
        keep = raw[:, 4] >= self.conf
        if self.classes is not None:
            keep &= np.isin(raw[:, 5].astype(np.int64), self.classes)
        raw = raw[keep]
        order = np.argsort(-raw[:, 4], kind="stable")[: self.max_det]
        raw = raw[order]
        boxes = unletterbox_boxes(raw[:, :4], info).astype(np.float32)
        return Detections(boxes, raw[:, 4].copy(), raw[:, 5].astype(np.int64))

    def detect(self, rgb: np.ndarray) -> Detections:
        t0 = time.perf_counter()
        x, info = self.preprocess(rgb)
        t1 = time.perf_counter()
        raw = self._infer(x)
        t2 = time.perf_counter()
        det = self.postprocess(raw, info)
        t3 = time.perf_counter()
        det.timings_ms = {"preprocess": (t1 - t0) * 1e3, "infer": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}
        return det
