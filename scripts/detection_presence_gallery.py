"""Spec 02.2 R4/R5: measured object presence on TUM, and the 24-image detection gallery.

Reads the stored full runs (no re-inference), so it needs sun_test_full and tum_*_full plus conf_operational.
- R4: per TUM sequence, the share of frames with >= 1 detection at conf_operational, and per-class counts.
- R5: 4 overlays per SUN sensor (test split; seed 0) and 4 per TUM sequence (evenly spaced). Each shows
  predictions at conf_operational (green) and, for SUN, the GT 2D boxes of mapped classes (orange, dashed label).

Usage:  python -I scripts/detection_presence_gallery.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import data_root, load_config  # noqa: E402
from perception.data.rgb_sources import sun_items, tum_items  # noqa: E402
from perception.data.sunrgbd_annotations import load_annotations  # noqa: E402
from perception.detection.classes import COCO_NAMES  # noqa: E402
from perception.detection.runio import read_run  # noqa: E402

OUT = REPO / "benchmarks" / "detection"
TUM_RUNS = {"rgbd_dataset_freiburg1_desk": "tum_fr1_desk_full",
            "rgbd_dataset_freiburg3_long_office_household": "tum_fr3_long_office_full"}
GREEN, ORANGE = (60, 200, 60), (0, 140, 255)


def kept(rec, thr):
    s = np.asarray(rec["scores"])
    m = s >= thr
    return np.asarray(rec["boxes"]).reshape(-1, 4)[m], s[m], np.asarray(rec["class_ids"])[m]


def label(img, text, x, y, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y = max(y, h + 4)
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def overlay(path, rec, thr, gt=()):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    for cls, (x1, y1, x2, y2) in gt:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, ORANGE, 1)
        label(img, f"GT {cls}", p1[0], p2[1], ORANGE)
    boxes, scores, cls = kept(rec, thr)
    for b, s, c in zip(boxes, scores, cls):
        x1, y1, x2, y2 = (int(round(v)) for v in b)
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
        label(img, f"{COCO_NAMES[c]} {s:.2f}", x1, y1, GREEN)
    return img


def main() -> int:
    thr = load_config("detector")["conf_operational"]
    if thr is None:
        raise SystemExit("conf_operational not set; run scripts/eval_detection.py --write-config first")
    gal = OUT / "gallery"
    gal.mkdir(parents=True, exist_ok=True)
    presence = {"conf_operational": thr, "sequences": {}}

    for seq, run in TUM_RUNS.items():
        _, recs = read_run(OUT / f"{run}.jsonl")
        items = {it.frame_id: it for it in tum_items(seq)}
        if len(recs) != len(items):
            raise SystemExit(f"{run}: {len(recs)} records but {len(items)} frames")
        frames_with, det_count, frame_count = 0, Counter(), Counter()
        for r in recs:
            _, _, c = kept(r, thr)
            frames_with += len(c) > 0
            det_count.update(COCO_NAMES[i] for i in c)
            frame_count.update({COCO_NAMES[i] for i in c})
        presence["sequences"][seq] = {
            "n_frames": len(recs), "frames_with_detection": frames_with,
            "share_with_detection": round(frames_with / len(recs), 4),
            "per_class": {k: {"frames": frame_count[k], "detections": det_count[k]}
                          for k, _ in frame_count.most_common()},
        }
        print(f"{seq}: {frames_with}/{len(recs)} frames with >= 1 detection; top "
              f"{', '.join(f'{k} {v}' for k, v in frame_count.most_common(5))}")
        for j in np.linspace(0, len(recs) - 1, 4).round().astype(int):
            r = recs[j]
            cv2.imwrite(str(gal / f"tum_{run.split('_full')[0][4:]}_{r['frame_id']}.jpg"),
                        overlay(items[r["frame_id"]].path, r, thr))

    sun_to_coco = yaml.safe_load((REPO / "configs" / "class_map.yaml").read_text())["sun_to_coco"]
    anns = {a.image_id: a for a in load_annotations(data_root(load_config("dataset")) / "sunrgbd")}
    _, recs = read_run(OUT / "sun_test_full.jsonl")
    by_id = {r["frame_id"]: r for r in recs}
    items = {it.frame_id: it for it in sun_items("sunrgbd_test")}
    rng = np.random.default_rng(0)
    for sensor in ("kv1", "kv2", "realsense", "xtion"):
        ids = sorted(i for i, it in items.items() if it.sensor == sensor)
        for i in sorted(rng.choice(ids, 4, replace=False)):
            gt = [(sun_to_coco[b.cls], b.xyxy) for b in anns[i].boxes2d if b.cls in sun_to_coco]
            cv2.imwrite(str(gal / f"sun_{sensor}_{i}.jpg"), overlay(items[i].path, by_id[i], thr, gt))

    (OUT / "tum_presence.json").write_text(json.dumps(presence, indent=2), encoding="utf-8")
    print(f"wrote tum_presence.json and {len(list(gal.glob('*.jpg')))} gallery images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
