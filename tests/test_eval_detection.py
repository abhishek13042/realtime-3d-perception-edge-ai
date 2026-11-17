"""Spec 02.3 R7: check the evaluator before trusting it."""

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from perception.data.sunrgbd_annotations import Box2D, ImageAnn  # noqa: E402
from perception.eval.detection import (build_coco_dets, build_coco_gt, coco_evaluate, match_at_iou,  # noqa: E402
                                       pick_f1_threshold, pr_at)

MAP = {"chair": "chair", "sofa": "couch", "sofa_chair": "couch", "table": "dining table"}
HW = (480, 640)


def _ann(image_id, boxes, sensor="kv1"):
    return ImageAnn(image_id=image_id, index=int(image_id) - 1, sequence="s", sensor=sensor, rgb_rel="", depth_rel="",
                    K=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], Rtilt=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    boxes2d=[Box2D(c, c, list(map(float, b)), False) for c, b in boxes])


ANNS = [
    _ann("000001", [("chair", (10, 10, 210, 210)), ("sofa", (300, 100, 600, 400))]),
    _ann("000002", [("chair", (50, 50, 250, 300)), ("lamp", (0, 0, 100, 100))], sensor="kv2"),
]


def _gt(anns=ANNS, **kw):
    return build_coco_gt(anns, {a.image_id: HW for a in anns}, MAP, **kw)


def _as_records(gt, score=0.9):
    """GT turned into predictions (frame records like a run JSONL)."""
    recs = {}
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        r = recs.setdefault(a["image_id"], {"frame_id": f"{a['image_id']:06d}", "boxes": [], "scores": [],
                                            "class_ids": []})
        r["boxes"].append([x, y, x + w, y + h]); r["scores"].append(score); r["class_ids"].append(a["category_id"] - 1)  # noqa: E702
    return list(recs.values())


def test_conversion_mapping_and_unmapped():
    gt, st = _gt()
    names = {c["id"]: c["name"] for c in gt["categories"]}
    assert sorted(names[a["category_id"]] for a in gt["annotations"]) == ["chair", "chair", "couch"]
    assert st.kept == 3 and st.unmapped == 1  # lamp dropped
    # sofa and sofa_chair merge into the same category
    gt2, _ = _gt([_ann("000001", [("sofa", (0, 0, 50, 50)), ("sofa_chair", (100, 100, 150, 150))])])
    assert len({a["category_id"] for a in gt2["annotations"]}) == 1


def test_degenerate_and_clipped_boxes():
    gt, st = _gt([_ann("000001", [("chair", (700, 10, 800, 50)), ("chair", (-20, -20, 30, 30))])])
    assert st.degenerate == 1 and gt["annotations"][0]["bbox"] == [0.0, 0.0, 30.0, 30.0]


def test_perfect_predictions_ap_one():
    gt, _ = _gt()
    targets = {c["id"] - 1 for c in gt["categories"]}
    res = coco_evaluate(gt, build_coco_dets(_as_records(gt), targets))
    assert res["summary"]["mAP"] == pytest.approx(1.0) and res["summary"]["AP50"] == pytest.approx(1.0)
    assert res["per_class"]["chair"]["n_gt"] == 2 and res["per_class"]["chair"]["AP"] == pytest.approx(1.0)


def test_no_predictions_ap_zero():
    gt, _ = _gt()
    assert coco_evaluate(gt, [])["summary"]["mAP"] == 0.0


def test_toy_hand_computed_ap50():
    # 2 chair GT; 3 ranked predictions: TP (0.9), FP (0.8), TP (0.7).
    # Precision/recall points: (1, .5), (.5, .5), (2/3, 1). With COCO 101-point interpolation, recall 0..0.50
    # (51 points) has precision 1 and recall 0.51..1.0 (50 points) has precision 2/3.
    # AP50 = (51 + 50 * 2/3) / 101.
    gt, _ = _gt([_ann("000001", [("chair", (0, 0, 100, 100)), ("chair", (200, 200, 300, 300))])])
    recs = [{"frame_id": "000001", "boxes": [[0, 0, 100, 100], [400, 0, 470, 60], [200, 200, 300, 300]],
             "scores": [0.9, 0.8, 0.7], "class_ids": [56, 56, 56]}]
    res = coco_evaluate(gt, build_coco_dets(recs, {56}))
    assert res["summary"]["AP50"] == pytest.approx((51 + 50 * 2 / 3) / 101, abs=1e-4)
    s, tp, n = match_at_iou(gt, build_coco_dets(recs, {56}))
    assert n == 2 and dict(zip(s.tolist(), tp.tolist())) == {0.9: True, 0.8: False, 0.7: True}
    # threshold 0.75 keeps 0.9 (TP) and 0.8 (FP)
    assert pr_at(s, tp, n, 0.75) == pytest.approx({"threshold": 0.75, "precision": 0.5, "recall": 0.5,
                                                   "f1": 0.5, "tp": 1, "n_pred": 2, "n_gt": 2})
    assert pr_at(s, tp, n, 0.85)["precision"] == 1.0


def test_non_target_classes_dropped():
    gt, _ = _gt()
    recs = [{"frame_id": "000001", "boxes": [[0, 0, 10, 10]], "scores": [0.9], "class_ids": [0]}]  # person
    assert build_coco_dets(recs, {56, 57}) == []


def test_ignore_region_neither_tp_nor_fp():
    anns = [_ann("000001", [("chair", (0, 0, 100, 100)), ("desk", (300, 300, 500, 450))])]
    recs = [{"frame_id": "000001", "boxes": [[0, 0, 100, 100], [300, 300, 500, 450]], "scores": [0.9, 0.95],
             "class_ids": [56, 60]}]  # chair TP; "dining table" on the desk
    strict, st = build_coco_gt(anns, {"000001": HW}, {"chair": "chair", "table": "dining table"})
    ign, st2 = build_coco_gt(anns, {"000001": HW}, {"chair": "chair", "table": "dining table"},
                             ignore_as={"desk": "dining table"})
    assert st.unmapped == 1 and st2.ignore == 1 and st2.kept == 1
    dets = build_coco_dets(recs, {56, 60})
    # Strict: the desk prediction is an FP of a class with no GT. dining table AP is undefined, chair AP is 1.
    assert coco_evaluate(strict, dets)["per_class"]["dining table"]["AP"] is None
    res = coco_evaluate(ign, dets)
    assert res["per_class"]["chair"]["AP"] == pytest.approx(1.0) and res["per_class"]["dining table"]["n_gt"] == 0


def test_f1_picker_known_optimum():
    # 10 GT. Scores descending, with TP pattern: 6 TPs, then FP, FP, TP, FP...
    scores = np.array([.95, .9, .85, .8, .75, .7, .6, .5, .4, .3, .2])
    is_tp = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], bool)
    # Candidate F1: at 0.7 -> P 1, R .6 -> .75; at 0.4 -> P 7/9, R .7 -> .7368. Optimum is 0.7.
    best = pick_f1_threshold(scores, is_tp, 10)
    assert best["threshold"] == 0.7 and best["f1"] == pytest.approx(0.75)


def test_f1_picker_ties_enter_together():
    best = pick_f1_threshold(np.array([0.9, 0.5, 0.5]), np.array([1, 0, 1], bool), 2)
    assert best["threshold"] in (0.9, 0.5)
    assert best["f1"] == pytest.approx(max(2 * 1 * .5 / 1.5, 2 * (2 / 3) * 1 / (2 / 3 + 1)))
