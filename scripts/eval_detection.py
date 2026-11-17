"""Spec 02.3: evaluate stored detection runs on the COCO classes shared with SUN RGB-D.

Step 1 (val): choose the operational threshold by max F1 @ IoU 0.5 on val only (R6).
Step 2 (test): COCO metrics overall, per class, per sensor, and per size; P/R at the val-chosen threshold; and the
desk-as-ignore sensitivity row (R5).
Writes benchmarks/detection/eval_<tag>.json. With --write-config, conf_operational is stored in configs/detector.yaml.

Usage:
  python -I scripts/eval_detection.py --val sun_val_full --test sun_test_full --tag baseline --write-config
  python -I scripts/eval_detection.py --val sun_val_nms --test sun_test_nms --tag nms_ablation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import data_root, load_config  # noqa: E402
from perception.data.sunrgbd_annotations import load_annotations  # noqa: E402
from perception.detection.classes import COCO_ID  # noqa: E402
from perception.detection.runio import read_run  # noqa: E402
from perception.eval.detection import (build_coco_dets, build_coco_gt, coco_evaluate, match_at_iou,  # noqa: E402
                                       pick_f1_threshold, pr_at, pr_curve)

OUT = REPO / "benchmarks" / "detection"
LOW_N = 50
SENSORS = ("kv1", "kv2", "realsense", "xtion")


def load_run(name):
    header, recs = read_run(OUT / f"{name}.jsonl")
    return header, recs, {r["frame_id"]: tuple(r["image_hw"]) for r in recs}


def check_complete(name, recs, split):
    ids = [s.strip() for s in (REPO / "configs" / "splits" / f"{split}.txt").read_text().split() if s.strip()]
    got = {r["frame_id"] for r in recs}
    missing = set(ids) - got
    if missing or len(recs) != len(ids):
        raise SystemExit(f"{name}: {len(recs)} records for {len(ids)} {split} ids ({len(missing)} missing)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--write-config", action="store_true")
    args = ap.parse_args()

    sun_to_coco = yaml.safe_load((REPO / "configs" / "class_map.yaml").read_text())["sun_to_coco"]
    targets = {COCO_ID[c] for c in sun_to_coco.values()}
    anns = load_annotations(data_root(load_config("dataset")) / "sunrgbd")

    # --- val: threshold ---
    vh, vrecs, vhw = load_run(args.val)
    check_complete(args.val, vrecs, "sunrgbd_val")
    vgt, vst = build_coco_gt(anns, vhw, sun_to_coco)
    vdets = build_coco_dets(vrecs, targets)
    vs, vtp, vn = match_at_iou(vgt, vdets, 0.5)
    best = pick_f1_threshold(vs, vtp, vn)
    t, p, r, f1 = pr_curve(vs, vtp, vn)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    val_curve = [pr_at(vs, vtp, vn, float(g)) for g in grid]
    val_coco = coco_evaluate(vgt, vdets)
    thr = round(best["threshold"], 4)
    print(f"val: {len(vrecs)} images, {vn} GT; best F1 {best['f1']:.3f} at conf {thr} "
          f"(P {best['precision']:.3f}, R {best['recall']:.3f}); val AP50 {val_coco['summary']['AP50']}")

    # --- test ---
    th, trecs, thw = load_run(args.test)
    check_complete(args.test, trecs, "sunrgbd_test")
    for k in ("weights_sha256", "head", "imgsz", "conf"):
        if vh.get(k) != th.get(k):
            raise SystemExit(f"val/test runs differ in {k}: {vh.get(k)} vs {th.get(k)}")
    tgt, tst = build_coco_gt(anns, thw, sun_to_coco)
    tdets = build_coco_dets(trecs, targets)
    overall = coco_evaluate(tgt, tdets)
    ts, ttp, tn = match_at_iou(tgt, tdets, 0.5)
    at_thr = pr_at(ts, ttp, tn, thr)
    for name, row in overall["per_class"].items():
        row["low_n"] = row["n_gt"] < LOW_N
    sensor_of = {i["id"]: i["sensor"] for i in tgt["images"]}
    per_sensor = {}
    for s in SENSORS:
        ids = [i for i, v in sensor_of.items() if v == s]
        res = coco_evaluate(tgt, tdets, img_ids=ids)
        sub = {"images": [i for i in tgt["images"] if i["id"] in set(ids)],
               "annotations": [a for a in tgt["annotations"] if a["image_id"] in set(ids)],
               "categories": tgt["categories"]}
        ss, stp, sn = match_at_iou(sub, [d for d in tdets if d["image_id"] in set(ids)], 0.5)
        per_sensor[s] = {"n_images": len(ids), "n_gt": sn, **{k: res["summary"][k] for k in ("mAP", "AP50")},
                         "at_operational": pr_at(ss, stp, sn, thr)}
    # R5: desk as an ignore region for dining table (strict stays the headline)
    igt, ist = build_coco_gt(anns, thw, sun_to_coco, ignore_as={"desk": "dining table"})
    ign = coco_evaluate(igt, tdets)
    sensitivity = {"strict": overall["per_class"].get("dining table"),
                   "desk_as_ignore": ign["per_class"].get("dining table"),
                   "n_desk_ignore_regions": ist.ignore,
                   "all_classes_mAP_strict": overall["summary"]["mAP"],
                   "all_classes_mAP_desk_ignore": ign["summary"]["mAP"]}

    result = {
        "spec": "02.3", "tag": args.tag, "label": "Measured: YOLO26-n pretrained (COCO), no fine-tuning, PyTorch CPU",
        "runs": {"val": args.val, "test": args.test}, "run_header": th,
        "classes": sorted(overall["per_class"]), "class_map": "configs/class_map.yaml",
        "eval_settings": {"pred_conf_floor": th["conf"], "coco_maxDets": 100, "pr_iou": 0.5, "low_n_below": LOW_N},
        "gt_conversion": {"val": vars(vst), "test": vars(tst)},
        "val": {"n_images": len(vrecs), "n_gt": vn, "best_f1": best, "conf_operational": thr,
                "coco_summary": val_coco["summary"], "pr_grid": val_curve,
                "curve_points": int(len(t))},
        "test": {"n_images": len(trecs), "n_gt": tn, "coco_summary": overall["summary"],
                 "at_operational": at_thr, "per_class": overall["per_class"], "per_sensor": per_sensor,
                 "sensitivity_desk_ignore": sensitivity},
    }
    out = OUT / f"eval_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    s = overall["summary"]
    print(f"test: mAP {s['mAP']}, AP50 {s['AP50']}, AR100 {s['AR100']}; at conf {thr}: "
          f"P {at_thr['precision']:.3f} R {at_thr['recall']:.3f} F1 {at_thr['f1']:.3f}")
    print(f"wrote {out.relative_to(REPO)}")

    if args.write_config:
        cfg_path = REPO / "configs" / "detector.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        new = re.sub(r"^conf_operational:.*$",
                     f"conf_operational: {thr}  # max F1 @ IoU 0.5 on SUN val (spec 02.3 R6; eval_{args.tag}.json)",
                     text, count=1, flags=re.M)
        cfg_path.write_text(new, encoding="utf-8")
        print(f"configs/detector.yaml: conf_operational = {thr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
