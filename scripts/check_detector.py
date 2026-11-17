"""Spec 02.1 T5 / AC2-AC4: parity against Ultralytics predict, determinism, and overlays.

- Parity: on N SUN val images, our detector and YOLO(...).predict are run at the same input size and threshold, for
  both heads (one2one vs predict(nms=False); one2many_nms vs predict(nms=None)). Detections are matched one-to-one
  per class by IoU (greedy, highest score first).
- Determinism: a second pass in the same process must be identical. The printed digest is compared against a
  separate process with --digest-only.
- Overlays: the first 5 images go to benchmarks/detection/check/.

Usage:  python -I scripts/check_detector.py [--n 20] [--conf 0.25] [--digest-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import data_root, load_config  # noqa: E402
from perception.data.sampling import sample_ids  # noqa: E402
from perception.data.sunrgbd_annotations import load_annotations  # noqa: E402
from perception.detection.classes import COCO_NAMES  # noqa: E402
from perception.detection.torch_backend import build_detector  # noqa: E402

OUT = REPO / "benchmarks" / "detection"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def match(ours, ref):
    """Greedy one-to-one match per class. Returns (pairs [(i, j, iou)], unmatched_ours, unmatched_ref)."""
    pairs, used = [], set()
    for i in np.argsort(-ours[1]):
        cand = [j for j in range(len(ref[1])) if j not in used and ref[2][j] == ours[2][i]]
        if not cand:
            continue
        ious = iou_matrix(ours[0][i:i + 1], ref[0][cand])[0]
        k = int(np.argmax(ious))
        if ious[k] > 0.5:
            pairs.append((int(i), cand[k], float(ious[k])))
            used.add(cand[k])
    mo = len(ours[1]) - len(pairs)
    return pairs, mo, len(ref[1]) - len(used)


def digest(dets) -> str:
    h = hashlib.sha256()
    for d in dets:
        for a in (d.boxes, d.scores, d.class_ids):
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def draw(bgr, det, color=(0, 255, 0)):
    img = bgr.copy()
    for b, s, c in zip(det.boxes, det.scores, det.class_ids):
        x1, y1, x2, y2 = (int(round(v)) for v in b)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{COCO_NAMES[c]} {s:.2f}", (x1 + 2, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--digest-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config("detector")
    sun_root = data_root(load_config("dataset")) / "sunrgbd"
    anns = {a.image_id: a for a in load_annotations(sun_root)}
    ids = sample_ids("sunrgbd_val", args.n, 0)
    images = []
    for i in ids:
        bgr = cv2.imread(str(sun_root / anns[i].rgb_rel), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"cannot read {anns[i].rgb_rel}")
        images.append(bgr)

    det = build_detector(cfg, conf=args.conf)
    run1 = [det.detect(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)) for b in images]
    d1 = digest(run1)
    if args.digest_only:
        print("digest", d1)
        return 0
    run2 = [det.detect(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)) for b in images]
    same_process_identical = d1 == digest(run2)

    from ultralytics import YOLO

    ref_model = YOLO(str(REPO / cfg["weights"]))
    parity = {}
    for head, nms in (("one2one", False), ("one2many_nms", None)):
        ours_det = run1 if head == cfg["head"] else None
        if ours_det is None:
            d = build_detector(cfg, conf=args.conf, head=head)
            ours_det = [d.detect(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)) for b in images]
        n_o = n_r = n_pair = 0
        worst_iou, worst_ds, per_image_count_equal = 1.0, 0.0, 0
        for bgr, o in zip(images, ours_det):
            r = ref_model.predict(bgr, imgsz=cfg["imgsz"], conf=args.conf, iou=cfg["iou"], max_det=cfg["max_det"],
                                  nms=nms, rect=False, device="cpu", verbose=False)[0].boxes
            ref = (r.xyxy.numpy(), r.conf.numpy(), r.cls.numpy().astype(np.int64))
            pairs, mo, mr = match((o.boxes, o.scores, o.class_ids), ref)
            n_o, n_r, n_pair = n_o + len(o), n_r + len(ref[1]), n_pair + len(pairs)
            per_image_count_equal += int(len(o) == len(ref[1]))
            for i, j, iou in pairs:
                worst_iou = min(worst_iou, iou)
                worst_ds = max(worst_ds, abs(float(o.scores[i]) - float(ref[1][j])))
        parity[head] = {"ours": n_o, "reference": n_r, "matched": n_pair, "images_same_count": per_image_count_equal,
                        "min_matched_iou": round(worst_iou, 6), "max_abs_score_diff": float(f"{worst_ds:.3g}")}
        print(head, parity[head])

    (OUT / "check").mkdir(parents=True, exist_ok=True)
    for i in range(min(5, len(images))):
        cv2.imwrite(str(OUT / "check" / f"{ids[i]}.jpg"), draw(images[i], run1[i]))

    t = np.array([[r.timings_ms[k] for k in ("preprocess", "infer", "postprocess")] for r in run2])
    result = {
        "spec": "02.1 T5", "n_images": len(ids), "image_ids": ids, "conf": args.conf, "imgsz": cfg["imgsz"],
        "head": cfg["head"], "weights_sha256": det.weights_sha256, "digest_run1": d1,
        "same_process_identical": same_process_identical, "parity": parity,
        "dev_timing_ms_median_preprocess_infer_postprocess": np.median(t, 0).round(2).tolist(),
        "note": "Timing is a smoke figure (20 images, no warm-up protocol); the latency benchmark is spec 02.2.",
    }
    (OUT / "detector_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("same_process_identical", "digest_run1",
                                             "dev_timing_ms_median_preprocess_infer_postprocess")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
