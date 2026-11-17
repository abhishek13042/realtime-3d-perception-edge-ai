"""Letterbox resize and its exact inverse (spec 02.1 R2). Pure functions, no model dependency.

Box convention: continuous pixel coordinates in which the image spans [0, W] x [0, H]. This is the convention
YOLO-family models and pycocotools use. Phase 01 GT boxes use a 0-based xyxy convention that differs from it by at
most 0.5 px, which is negligible at the IoU >= 0.5 used for evaluation.

The geometry matches Ultralytics' LetterBox(auto=False, center=True) exactly:
- each side is resized to round(side * r);
- the padding is split as top = round(dh/2 - 0.1), left = round(dw/2 - 0.1);
- the inverse uses separate x/y gains (new_side / old_side).
So our pre/postprocessing is interchangeable with Ultralytics' own, and with the ONNX backend in Phase 06.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    src_hw: tuple[int, int]  # original (H, W)
    dst_hw: tuple[int, int]  # model input (H, W)
    gain_x: float            # resized_w / src_w
    gain_y: float            # resized_h / src_h
    pad_x: int               # left padding in px
    pad_y: int               # top padding in px


def letterbox_params(src_hw: tuple[int, int], size: int | tuple[int, int]) -> LetterboxInfo:
    """Compute the letterbox geometry for an image of shape src_hw into a size x size (or (H, W)) canvas."""
    dst_h, dst_w = (size, size) if isinstance(size, int) else size
    h, w = src_hw
    if h <= 0 or w <= 0:
        raise ValueError(f"empty image shape {src_hw}")
    r = min(dst_h / h, dst_w / w)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = (dst_w - new_w) / 2, (dst_h - new_h) / 2
    return LetterboxInfo((h, w), (dst_h, dst_w), new_w / w, new_h / h, round(dw - 0.1), round(dh - 0.1))


def letterbox(img: np.ndarray, size: int | tuple[int, int] = 640, pad_value: int = 114) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize img (H, W, C) keeping its aspect ratio, and centre it on a constant-colour canvas."""
    info = letterbox_params(img.shape[:2], size)
    new_w = round(info.src_hw[1] * info.gain_x)
    new_h = round(info.src_hw[0] * info.gain_y)
    resized = img if (new_h, new_w) == img.shape[:2] else cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((*info.dst_hw, img.shape[2]), pad_value, dtype=img.dtype)
    canvas[info.pad_y:info.pad_y + new_h, info.pad_x:info.pad_x + new_w] = resized
    return canvas, info


def letterbox_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map xyxy boxes from original-image px into letterbox px (the forward transform; used by tests)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]] * info.gain_x + info.pad_x
    b[:, [1, 3]] = b[:, [1, 3]] * info.gain_y + info.pad_y
    return b


def unletterbox_boxes(boxes: np.ndarray, info: LetterboxInfo, clip: bool = True) -> np.ndarray:
    """Map xyxy boxes from letterbox px back to original-image px, optionally clipping to [0, W] x [0, H]."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - info.pad_x) / info.gain_x
    b[:, [1, 3]] = (b[:, [1, 3]] - info.pad_y) / info.gain_y
    if clip:
        h, w = info.src_hw
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b
