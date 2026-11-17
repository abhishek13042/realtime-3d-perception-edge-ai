"""Spec 05.4 R1: rule-based failure-gallery selection from the 05.2 SUN test arrays (no by-eye picking).

Input: the dict of arrays in benchmarks/results/localization_sun_test_objects.npz (o_* = objects, b_* = output boxes).
Every rule sorts with a full tie-break (value, then image id, then box), keeps at most one item per image per group
(so one scene cannot fill a group), and returns plain dicts. Same input -> same items in the same order.

Groups:
- large_error: the 12 largest 3D errors among detected-mode objects with status `valid`;
- no_depth: 8 detected-mode objects with status `no_depth`, highest detector score first;
- missed: 8 linked GT objects with no detected-mode match (IoU 0.5, 05.2 matching), from the 4 most frequent classes
  of the oracle set, 2 per class, largest GT 2D box first;
- false_positive: 8 output boxes (score >= conf_operational) that match no GT box at all, highest score first.
"""

from __future__ import annotations

import numpy as np

GROUPS = {"large_error": 12, "no_depth": 8, "missed": 8, "false_positive": 8}
CATEGORIES = {
    "depth": ("background bleed", "occluder", "missing or invalid depth", "sensor edge"),
    "gt": ("GT-box problem",),
    "detection": ("localisation", "misclassification", "small object", "truncation"),
}
ALL_CATEGORIES = tuple(c for v in CATEGORIES.values() for c in v)


def _key(group: str, image_id: str, cls: str, box) -> str:
    return f"{group}:{image_id}:{cls}:{int(box[0])}_{int(box[1])}"


def _take(order, image_ids, n: int) -> list[int]:
    out, seen = [], set()
    for j in order:
        if image_ids[j] in seen:
            continue
        seen.add(image_ids[j])
        out.append(int(j))
        if len(out) == n:
            break
    return out


def _sort(primary: np.ndarray, image_ids: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Indices sorted by primary ascending, then image id, then box x1, y1 (np.lexsort: last key is primary)."""
    return np.lexsort((boxes[:, 1], boxes[:, 0], image_ids, primary))


def _obj_item(z: dict, group: str, j: int, **extra) -> dict:
    box = z["o_box"][j]
    err = float(np.linalg.norm(z["o_centre"][j] - z["o_gt"][j])) if np.all(np.isfinite(z["o_centre"][j])) else None
    return {"key": _key(group, str(z["o_image_id"][j]), str(z["o_raw_cls"][j]), box), "group": group,
            "image_id": str(z["o_image_id"][j]), "sensor": str(z["o_sensor"][j]), "class": str(z["o_cls"][j]),
            "raw_class": str(z["o_raw_cls"][j]), "box": [round(float(v), 1) for v in box],
            "score": None if not np.isfinite(z["o_score"][j]) else round(float(z["o_score"][j]), 3),
            "status": str(z["o_status"][j]), "gt_centre": [round(float(v), 3) for v in z["o_gt"][j]],
            "err3d_m": None if err is None else round(err, 3),
            "valid_frac": round(float(z["o_valid_frac"][j]), 3), "link_iou": round(float(z["o_link_iou"][j]), 3),
            **extra}


def select(z: dict) -> dict[str, list[dict]]:
    mode, iid = z["o_mode"], z["o_image_id"]
    det = mode == "detected"
    err = np.linalg.norm(z["o_centre"] - z["o_gt"], axis=1)
    out: dict[str, list[dict]] = {}

    idx = np.flatnonzero(det & (z["o_status"] == "valid") & np.isfinite(err))
    order = idx[_sort(-err[idx], iid[idx], z["o_box"][idx])]
    out["large_error"] = [_obj_item(z, "large_error", j) for j in _take(order, iid, GROUPS["large_error"])]

    idx = np.flatnonzero(det & (z["o_status"] == "no_depth"))
    order = idx[_sort(-z["o_score"][idx], iid[idx], z["o_box"][idx])]
    out["no_depth"] = [_obj_item(z, "no_depth", j) for j in _take(order, iid, GROUPS["no_depth"])]

    # Missed: oracle objects (GT box) whose (image, GT centre) has no detected-mode object.
    gt_key = lambda j: (str(iid[j]), tuple(np.round(z["o_gt"][j], 6)))  # noqa: E731
    found = {gt_key(j) for j in np.flatnonzero(det)}
    ora = np.flatnonzero(mode == "oracle")
    miss = np.array([j for j in ora if gt_key(j) not in found], dtype=np.int64)
    classes, counts = np.unique(z["o_cls"][ora], return_counts=True)
    top = [str(c) for c in classes[np.lexsort((classes, -counts))][:4]]
    area = (z["o_box"][:, 2] - z["o_box"][:, 0]) * (z["o_box"][:, 3] - z["o_box"][:, 1])
    items, used = [], set()
    per_class = GROUPS["missed"] // len(top) if top else 0
    for c in top:
        m = miss[z["o_cls"][miss] == c]
        m = np.array([j for j in m if iid[j] not in used], dtype=np.int64)
        order = m[_sort(-area[m], iid[m], z["o_box"][m])]
        for j in _take(order, iid, per_class):
            used.add(iid[j])
            items.append(_obj_item(z, "missed", j, gt_box_area_px=round(float(area[j]), 1)))
    out["missed"] = items

    bi = z["b_image_id"]
    idx = np.flatnonzero(~z["b_matched_any"])
    order = idx[_sort(-z["b_score"][idx], bi[idx], z["b_box"][idx])]
    out["false_positive"] = [
        {"key": _key("false_positive", str(bi[j]), str(z["b_cls"][j]), z["b_box"][j]), "group": "false_positive",
         "image_id": str(bi[j]), "sensor": str(z["b_sensor"][j]), "class": str(z["b_cls"][j]),
         "raw_class": str(z["b_cls"][j]), "box": [round(float(v), 1) for v in z["b_box"][j]],
         "score": round(float(z["b_score"][j]), 3), "status": str(z["b_status"][j])}
        for j in _take(order, bi, GROUPS["false_positive"])]
    return out


def category_counts(review: dict, items: dict[str, list[dict]]) -> dict:
    """Counts per category, per group and overall. Raises if an item is unreviewed or has an unknown category."""
    keys = [it["key"] for v in items.values() for it in v]
    missing = [k for k in keys if k not in review]
    if missing:
        raise ValueError(f"{len(missing)} gallery items have no review entry, e.g. {missing[:3]}")
    bad = {k: review[k]["category"] for k in keys if review[k]["category"] not in ALL_CATEGORIES}
    if bad:
        raise ValueError(f"unknown categories: {bad}")
    per_group = {g: {} for g in items}
    total: dict[str, int] = {}
    for g, v in items.items():
        for it in v:
            c = review[it["key"]]["category"]
            per_group[g][c] = per_group[g].get(c, 0) + 1
            total[c] = total.get(c, 0) + 1
    return {"per_group": per_group, "total": dict(sorted(total.items(), key=lambda kv: -kv[1]))}
