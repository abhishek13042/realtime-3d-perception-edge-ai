"""PyTorch backend for YOLO26 (spec 02.1 R4, R5, R7).

Raw output layout (inspected 2026-11-10, ultralytics 8.4.208, yolo26n.pt). The Detect head has two branches:
- one2one (`head: one2one`): NMS-free. With `end2end = True` the fused model returns (B, 300, 6)
  [x1, y1, x2, y2, score, cls] in letterbox px, picked by top-k over all anchor x class scores. Ultralytics uses this
  path when predict is called with nms=False.
- one2many (`head: one2many_nms`): returns (B, 84, 8400) [cx, cy, w, h, 80 class scores], which needs NMS. This is
  Ultralytics' default predict path (nms=None).
We default to one2one: it is YOLO26's design, it needs no NMS operator (simpler ONNX/edge export in Phase 06), and its
cost does not grow with the number of boxes. one2many_nms is kept for the ablation in spec 02.3.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from perception.detection.base import Detector


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TorchYOLODetector(Detector):
    def __init__(self, weights: str | Path, sha256: str | None = None, head: str = "one2one",
                 iou: float = 0.7, threads: int | None = None, **kw):
        super().__init__(**kw)
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"{weights} missing; see configs/detector.yaml for where it comes from")
        if sha256 is not None and (got := sha256_file(weights)) != sha256:
            raise ValueError(f"weights hash mismatch for {weights}: {got} != {sha256}")
        if head not in ("one2one", "one2many_nms"):
            raise ValueError(f"unknown head {head!r}")
        if threads:
            torch.set_num_threads(int(threads))
        from ultralytics import YOLO  # imported lazily: heavy

        model = YOLO(str(weights)).model
        model.end2end = head == "one2one"   # selects the branch; fuse() then drops the unused one
        self.model = model.fuse(verbose=False).eval()
        self.names = dict(model.names)
        self.head, self.iou = head, iou
        self.weights, self.weights_sha256 = weights, sha256_file(weights)

    @torch.inference_mode()
    def _infer(self, x: np.ndarray) -> np.ndarray:
        out = self.model(torch.from_numpy(x))
        y = out[0] if isinstance(out, (tuple, list)) else out
        if self.head == "one2one":
            return y[0].numpy()                      # (300, 6)
        return self._nms(y[0])                       # (84, 8400) -> (N, 6)

    def _nms(self, p: torch.Tensor) -> np.ndarray:
        import torchvision

        p = p.T                                      # (8400, 84)
        scores, cls = p[:, 4:].max(1)
        m = scores >= self.conf
        p, scores, cls = p[m], scores[m], cls[m]
        xy, wh = p[:, :2], p[:, 2:4]
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)
        keep = torchvision.ops.batched_nms(boxes, scores, cls, self.iou)[: self.max_det]
        return torch.cat([boxes[keep], scores[keep, None], cls[keep, None].float()], 1).numpy()


def build_detector(cfg: dict, **overrides) -> TorchYOLODetector:
    """Build the detector from configs/detector.yaml (optionally overriding e.g. conf)."""
    from perception.config import REPO_ROOT

    c = {**cfg, **overrides}
    return TorchYOLODetector(
        weights=REPO_ROOT / c["weights"], sha256=c.get("weights_sha256"), head=c["head"], iou=c.get("iou", 0.7),
        threads=c.get("threads"), imgsz=c["imgsz"], conf=c["conf"], max_det=c["max_det"],
        classes=c.get("classes"), pad_value=c.get("pad_value", 114),
    )
