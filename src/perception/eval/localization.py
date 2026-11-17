"""3D localization evaluation on SUN RGB-D (spec 03.3; evaluation protocol section 3).

- GT centre = `Box3D.centroid_cam(Rtilt)`, a volume centre in the OpenCV camera frame.
- 2D <-> 3D linkage (R1): each 3D box is projected to the image (amodal xyxy) and linked to a 2D GT box of the same
  class by maximum total IoU (Hungarian, IoU > 0). The `has3d` order was checked first: on 20 val images it
  agreed with the best projection in 126/132 pairs, but only 85/132 reached the fixed IoU >= 0.5 check, because 2D boxes
  are visible extents and projected 3D boxes are amodal. By the R1 rule, projection linkage is therefore used.
- Matching of detections (R2): same COCO class, highest IoU >= 0.5 among still-unmatched GT, greedy by score.
- Distance buckets use the GT centre Z: [0, 2), [2, 4), [4, inf). So exactly 2.0 m is in "2-4" and exactly 4.0 m is
  in ">4".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from perception.geometry.camera import project
from perception.geometry.transforms import upright_to_cam_matrix

BUCKETS = (("0-2", 0.0, 2.0), ("2-4", 2.0, 4.0), (">4", 4.0, math.inf))
NEAR_Z = 0.05  # corners closer than this (or behind the camera) are dropped before projecting


def bucket_of(z: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= z < hi:
            return name
    raise ValueError(f"no bucket for z={z}")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def project_box(corners_cam: np.ndarray, K: np.ndarray, hw: tuple[int, int]) -> list[float] | None:
    """Amodal 2D box (xyxy, box convention: pixel i spans [i, i+1)) of a camera-frame 3D box, clipped to the image.
    OpenCV puts pixel centres at integer u, so the box edge is u + 0.5. None if nothing is in front of the camera."""
    c = np.asarray(corners_cam, np.float64)
    c = c[c[:, 2] > NEAR_Z]
    if len(c) == 0:
        return None
    uv = project(c, K) + 0.5
    h, w = hw
    box = [max(0.0, uv[:, 0].min()), max(0.0, uv[:, 1].min()), min(float(w), uv[:, 0].max()),
           min(float(h), uv[:, 1].max())]
    return box if box[2] > box[0] and box[3] > box[1] else None


def half_extent_along(box3d, Rtilt: np.ndarray, direction_cam: np.ndarray) -> float:
    """Half the extent of a GT box along a unit camera-frame direction (support function of the box)."""
    axes_cam = np.asarray(box3d.basis, np.float64) @ upright_to_cam_matrix(Rtilt).T  # rows: box axes, camera frame
    return float(np.sum(np.abs(axes_cam @ direction_cam) * np.asarray(box3d.coeffs, np.float64)))


@dataclass
class GTObject:
    image_id: str
    sensor: str
    cls: str                 # COCO name after class_map
    raw_cls: str
    box2d: list[float]       # visible-extent 2D GT box (xyxy)
    centre: np.ndarray       # GT volume centre, camera frame (m)
    half_depth: float        # half the GT box extent along the ray to its centre (m), for the R4 analysis
    link_iou: float          # IoU of the projected 3D box with the linked 2D box


def link_2d_3d(ann, hw: tuple[int, int]) -> list[tuple[int, int, float]]:
    """(index into boxes2d, index into boxes3d, IoU) pairs: same normalised class, Hungarian on IoU, IoU > 0."""
    R, K = ann.Rtilt_np, ann.K_np
    proj = [project_box(b.corners_cam(R), K, hw) for b in ann.boxes3d]
    pairs = []
    for cls in {b.cls for b in ann.boxes3d}:
        i2 = [i for i, b in enumerate(ann.boxes2d) if b.cls == cls]
        i3 = [j for j, b in enumerate(ann.boxes3d) if b.cls == cls and proj[j] is not None]
        if not i2 or not i3:
            continue
        M = np.array([[iou(ann.boxes2d[i].xyxy, proj[j]) for j in i3] for i in i2])
        r, c = linear_sum_assignment(-M)
        pairs += [(i2[a], i3[b], float(M[a, b])) for a, b in zip(r, c) if M[a, b] > 0]
    return sorted(pairs)


def gt_objects(ann, hw: tuple[int, int], sun_to_coco: dict[str, str]) -> tuple[list[GTObject], list[tuple[str, list]]]:
    """Linked GT objects of mapped classes, plus every mapped 2D GT box as (coco class, xyxy) for matching. Unlinked
    2D boxes can still absorb a detection, but they have no 3D centre, so they are never scored."""
    R = ann.Rtilt_np
    objs = []
    for i, j, v in link_2d_3d(ann, hw):
        b2, b3 = ann.boxes2d[i], ann.boxes3d[j]
        if b2.cls not in sun_to_coco:
            continue
        c = b3.centroid_cam(R).reshape(3)
        objs.append(GTObject(ann.image_id, ann.sensor, sun_to_coco[b2.cls], b2.raw_cls, list(b2.xyxy), c,
                             half_extent_along(b3, R, c / np.linalg.norm(c)), v))
    all2d = [(sun_to_coco[b.cls], list(b.xyxy)) for b in ann.boxes2d if b.cls in sun_to_coco]
    return objs, all2d


def greedy_match(boxes, scores, classes: list[str], all2d: list[tuple[str, list]],
                 iou_thr: float = 0.5) -> list[tuple[int, int]]:
    """Protocol §3: greedy by score; each detection takes the same-class unmatched GT 2D box with the highest
    IoU >= iou_thr. Returns (detection index, all2d index)."""
    used, out = set(), []
    for d in np.argsort(-np.asarray(scores), kind="stable"):
        best, bi = iou_thr, -1
        for g, (c, b) in enumerate(all2d):
            if g in used or c != classes[d]:
                continue
            v = iou(boxes[d], b)
            if v >= best:
                best, bi = v, g
        if bi >= 0:
            used.add(bi)
            out.append((int(d), bi))
    return out


def match_detections(boxes, scores, classes: list[str], gt_objs: list[GTObject], all2d: list[tuple[str, list]],
                     iou_thr: float = 0.5) -> list[tuple[int, int]]:
    """(detection index, gt_objs index) for matched detections whose GT 2D box is linked to a 3D box. Matching runs
    against all mapped GT 2D boxes, so an unlinked GT box still takes its detection (which is then not scored)."""
    obj_of = {(o.cls, tuple(o.box2d)): k for k, o in enumerate(gt_objs)}
    out = []
    for d, g in greedy_match(boxes, scores, classes, all2d, iou_thr):
        k = obj_of.get((all2d[g][0], tuple(all2d[g][1])))
        if k is not None:
            out.append((d, k))
    return out


def status_of(n_valid: np.ndarray, valid_frac: np.ndarray, spread: np.ndarray, n_min: int, tau_cov: float,
              tau_spread: float) -> np.ndarray:
    """Vectorised spec 03.1 status rule, so thresholds can be swept without re-estimating."""
    st = np.where((valid_frac < tau_cov) | (spread > tau_spread), "low_coverage", "valid").astype(object)
    st[np.asarray(n_valid) < n_min] = "no_depth"
    return st


def error_stats(err3d: np.ndarray, dz: np.ndarray) -> dict:
    """err3d: 3D centre error (m); dz: signed Z error (m), estimate minus GT."""
    if err3d.size == 0:
        return {"n": 0}
    return {"n": int(err3d.size),
            "centre_median_m": round(float(np.median(err3d)), 4), "centre_mean_m": round(float(err3d.mean()), 4),
            "centre_p90_m": round(float(np.percentile(err3d, 90)), 4),
            "z_mae_m": round(float(np.abs(dz).mean()), 4), "z_rmse_m": round(float(np.sqrt((dz ** 2).mean())), 4),
            "z_signed_mean_m": round(float(dz.mean()), 4)}


def breakdown(err3d, dz, gt_z, sensor, mask) -> dict:
    """Overall, per distance bucket, and per sensor, over the objects selected by `mask`."""
    err3d, dz, gt_z, sensor, mask = map(np.asarray, (err3d, dz, gt_z, sensor, mask))
    out = {"all": error_stats(err3d[mask], dz[mask]), "by_distance": {}, "by_sensor": {}}
    b = np.array([bucket_of(z) for z in gt_z]) if gt_z.size else np.zeros(0, str)
    for name, _, _ in BUCKETS:
        m = mask & (b == name)
        out["by_distance"][name] = error_stats(err3d[m], dz[m])
    for s in sorted(set(sensor.tolist())):
        m = mask & (sensor == s)
        out["by_sensor"][s] = error_stats(err3d[m], dz[m])
    return out


TUNING_SPLITS = ("sunrgbd_val",)


def require_tuning_split(split: str) -> str:
    """Spec 03.3 R8: parameters are chosen on val only; any other split (test above all) is refused."""
    if split not in TUNING_SPLITS:
        raise ValueError(f"depth tuning runs only on {TUNING_SPLITS}; refused split {split!r} (no test peeking)")
    return split


def surface_point(centre: np.ndarray, half_depth: float) -> np.ndarray:
    """R4 analysis only: the GT volume centre moved half the box depth toward the camera along its viewing ray."""
    c = np.asarray(centre, np.float64)
    return c - half_depth * c / np.linalg.norm(c)
