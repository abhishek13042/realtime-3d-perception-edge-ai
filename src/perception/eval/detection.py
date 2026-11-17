"""2D detection evaluation on the COCO classes shared with SUN RGB-D (spec 02.3).

Conventions:
- GT boxes are SUN 2D boxes (0-based pixel xyxy) used as continuous coordinates and clipped to the image.
  Against our continuous [0, W] predictions this is a <= 0.5 px offset, which is negligible at IoU >= 0.5
  for any box larger than a few pixels.
- Only GT classes listed in configs/class_map.yaml are kept, after mapping to COCO names. Predictions are kept when
  their COCO class is a mapped target.
- COCO ids in the converted JSON are our 0-79 class index + 1, so 0 is never used.
- An "ignore" GT (iscrowd=1) is a region where a matching prediction counts as neither TP nor FP. It is used only
  in the desk -> dining-table sensitivity row (R5).
"""

from __future__ import annotations

import contextlib
import io
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from perception.detection.classes import COCO_ID, COCO_NAMES


# --- conversion ----------------------------------------------------------------------------------------------
@dataclass
class GTStats:
    kept: int = 0
    unmapped: int = 0
    degenerate: int = 0   # zero area after clipping, or fully outside the image
    ignore: int = 0


def build_coco_gt(anns, image_hw: dict[str, tuple[int, int]], sun_to_coco: dict[str, str],
                  ignore_as: dict[str, str] | None = None) -> tuple[dict, GTStats]:
    """anns: iterable of ImageAnn restricted to the images in `image_hw`.
    ignore_as: {sun_class: coco_name}. Boxes of these SUN classes become iscrowd=1 regions of that COCO class."""
    ignore_as = ignore_as or {}
    stats, images, annotations = GTStats(), [], []
    for a in anns:
        if a.image_id not in image_hw:
            continue
        h, w = image_hw[a.image_id]
        img_id = int(a.image_id)
        images.append({"id": img_id, "height": int(h), "width": int(w), "sensor": a.sensor})
        for b in a.boxes2d:
            crowd = 0
            if b.cls in sun_to_coco:
                coco = sun_to_coco[b.cls]
            elif b.cls in ignore_as:
                coco, crowd = ignore_as[b.cls], 1
            else:
                stats.unmapped += 1
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy)
            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                stats.degenerate += 1
                continue
            bw, bh = float(x2 - x1), float(y2 - y1)
            annotations.append({"id": len(annotations) + 1, "image_id": img_id, "category_id": COCO_ID[coco] + 1,
                                "bbox": [float(x1), float(y1), bw, bh], "area": bw * bh, "iscrowd": crowd})
            stats.ignore += crowd
            stats.kept += 1 - crowd
    cats = sorted({COCO_ID[c] for c in sun_to_coco.values()})
    return {"images": images, "annotations": annotations,
            "categories": [{"id": c + 1, "name": COCO_NAMES[c]} for c in cats]}, stats


def build_coco_dets(records, target_ids: set[int]) -> list[dict]:
    """records: frame dicts from a run JSONL (boxes in original px). Keeps only target COCO class indices."""
    out = []
    for r in records:
        img_id = int(r["frame_id"])
        for (x1, y1, x2, y2), s, c in zip(r["boxes"], r["scores"], r["class_ids"]):
            if c in target_ids:
                out.append({"image_id": img_id, "category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(s)})
    return out


# --- COCOeval wrapper ----------------------------------------------------------------------------------------
SUMMARY_KEYS = ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")


def coco_evaluate(gt: dict, dets: list[dict], img_ids: list[int] | None = None) -> dict:
    """Returns {summary, per_class: {name: {AP, AP50, n_gt}}}. If a metric is undefined (no GT), it is None."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = {k: list(v) for k, v in gt.items()}
        cg.createIndex()
        cd = cg.loadRes(dets) if dets else None
        if cd is None:  # pycocotools cannot load an empty result list
            cd = COCO()
            cd.dataset = {"images": gt["images"], "categories": gt["categories"], "annotations": []}
            cd.createIndex()
        ev = COCOeval(cg, cd, "bbox")
        if img_ids is not None:
            ev.params.imgIds = sorted(img_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    summary = {k: (None if v < 0 else round(float(v), 5)) for k, v in zip(SUMMARY_KEYS, ev.stats)}

    prec = ev.eval["precision"]  # [T(iou), R, K(cat), A(area), M(maxdet)]
    use = set(ev.params.imgIds)
    n_gt = defaultdict(int)
    for a in gt["annotations"]:
        if a["image_id"] in use and not a["iscrowd"]:
            n_gt[a["category_id"]] += 1
    per_class = {}
    for k, cat in enumerate(ev.params.catIds):
        p = prec[:, :, k, 0, -1]
        ap = p[p > -1].mean() if (p > -1).any() else None
        p50 = prec[0, :, k, 0, -1]
        ap50 = p50[p50 > -1].mean() if (p50 > -1).any() else None
        per_class[COCO_NAMES[cat - 1]] = {"AP": None if ap is None else round(float(ap), 5),
                                          "AP50": None if ap50 is None else round(float(ap50), 5),
                                          "n_gt": n_gt[cat]}
    return {"summary": summary, "per_class": per_class}


# --- precision / recall at a threshold, and the F1 threshold picker ------------------------------------------
def _iou(box, others):
    x1 = np.maximum(box[0], others[:, 0]); y1 = np.maximum(box[1], others[:, 1])  # noqa: E702
    x2 = np.minimum(box[2], others[:, 2]); y2 = np.minimum(box[3], others[:, 3])  # noqa: E702
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
    return inter / (area(box) + area(others) - inter + 1e-12)


def match_at_iou(gt: dict, dets: list[dict], iou_thr: float = 0.5) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy COCO-style matching per (image, class) in descending score order. Crowd GT is ignored here (strict).
    Because the matching is greedy by score, each prediction's TP flag is the same for every confidence threshold
    below its score. One pass therefore serves the whole threshold sweep.
    Returns (scores, is_tp) over all predictions, and the number of GT boxes."""
    g = defaultdict(list)
    for a in gt["annotations"]:
        if not a["iscrowd"]:
            x, y, w, h = a["bbox"]
            g[(a["image_id"], a["category_id"])].append([x, y, x + w, y + h])
    d = defaultdict(list)
    for p in dets:
        x, y, w, h = p["bbox"]
        d[(p["image_id"], p["category_id"])].append((p["score"], [x, y, x + w, y + h]))
    scores, tps = [], []
    for key, preds in d.items():
        gts = np.asarray(g.get(key, []), np.float64).reshape(-1, 4)
        used = np.zeros(len(gts), bool)
        for s, box in sorted(preds, key=lambda t: -t[0]):
            tp = False
            if len(gts):
                ious = _iou(np.asarray(box, np.float64), gts)
                ious[used] = -1
                j = int(np.argmax(ious))
                if ious[j] >= iou_thr:
                    used[j], tp = True, True
            scores.append(s)
            tps.append(tp)
    n_gt = sum(len(v) for v in g.values())
    return np.asarray(scores, np.float64), np.asarray(tps, bool), n_gt


def pr_curve(scores: np.ndarray, is_tp: np.ndarray, n_gt: int):
    """Precision, recall, and F1 at each distinct score used as the threshold (keep score >= t)."""
    order = np.argsort(-scores, kind="stable")
    s, tp = scores[order], is_tp[order].astype(np.int64)
    ctp, k = np.cumsum(tp), np.arange(1, len(s) + 1)
    last = np.r_[s[1:] != s[:-1], True]  # last index of each tied score, so ties enter together
    s, ctp, k = s[last], ctp[last], k[last]
    precision = ctp / k
    recall = ctp / max(n_gt, 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    return s, precision, recall, f1


def pick_f1_threshold(scores, is_tp, n_gt) -> dict:
    t, p, r, f1 = pr_curve(np.asarray(scores, np.float64), np.asarray(is_tp, bool), n_gt)
    if len(t) == 0:
        return {"threshold": None, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    i = int(np.argmax(f1))
    return {"threshold": float(t[i]), "f1": float(f1[i]), "precision": float(p[i]), "recall": float(r[i])}


def pr_at(scores, is_tp, n_gt, threshold: float) -> dict:
    keep = np.asarray(scores) >= threshold
    tp = int(np.asarray(is_tp)[keep].sum())
    n = int(keep.sum())
    p = tp / n if n else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"threshold": threshold, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": tp, "n_pred": n, "n_gt": n_gt}
