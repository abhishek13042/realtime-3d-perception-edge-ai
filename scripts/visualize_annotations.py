"""Visual + quantitative check of SUN RGB-D 3D boxes in the camera frame (spec 01.3, T5/T6, AC1/AC2).

Overlays: 2D GT boxes (green) and projected 3D wireframes (orange; bottom face darker) for N random test
images -> benchmarks/data_quality/annotations/.

Quantitative check (AC2): for each 3D box fully in front of the camera, project its corners, fill the convex
hull, and take the median valid depth inside it. Statistic: median over boxes of |median_depth − centre Z|,
compared with the median box Z extent. Run with the correct transform and with deliberately wrong ones
(negative controls) so a pass is meaningful.

Usage:  python -I scripts/visualize_annotations.py [--n 20] [--n-stats 600] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import data_root, load_config  # noqa: E402
from perception.data.splits import read_split  # noqa: E402
from perception.data.sunrgbd import SunRGBD, SunRecord  # noqa: E402
from perception.data.sunrgbd_annotations import ImageAnn, load_annotations  # noqa: E402
from perception.geometry.camera import project  # noqa: E402
from perception.geometry.transforms import BOX_EDGES, _UNFLIP  # noqa: E402

MIN_Z = 0.1

# name -> function(Rtilt) giving R with p_cam = R . p_upright
TRANSFORMS = {
    "correct (unflip . Rtilt^T)": lambda Rt: _UNFLIP @ Rt.T,
    "control: unflip . Rtilt (tilt applied the wrong way)": lambda Rt: _UNFLIP @ Rt,
    "control: unflip only (tilt ignored)": lambda Rt: _UNFLIP,
    "control: no axis swap (upright used as camera)": lambda Rt: Rt.T,
}


def _record(a: ImageAnn) -> SunRecord:
    return SunRecord(index=a.index, sequence=a.sequence, sensor=a.sensor, rgb_rel=a.rgb_rel,
                     depth_rel=a.depth_rel, K=a.K_np, Rtilt=a.Rtilt_np)


def box_stats(a: ImageAnn, depth: np.ndarray, valid: np.ndarray, R: np.ndarray) -> list[tuple[float, float]]:
    """Per box: (|median footprint depth − centre Z|, Z extent). Skips boxes partly behind the camera,
    off-image, or with < 20 valid depth pixels in the footprint."""
    H, W = depth.shape
    out = []
    for b in a.boxes3d:
        corners = b.corners_upright() @ R.T
        if corners[:, 2].min() <= MIN_Z:
            continue
        uv = project(corners, a.K_np)
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        sel = (mask > 0) & valid
        if sel.sum() < 20:
            continue
        cz = (np.asarray(b.centroid) @ R.T)[2]
        out.append((abs(float(np.median(depth[sel])) - cz), float(np.ptp(corners[:, 2]))))
    return out


def draw(a: ImageAnn, rgb: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for b in a.boxes2d:
        x1, y1, x2, y2 = (int(round(v)) for v in b.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 200, 60), 1)
    R = TRANSFORMS["correct (unflip . Rtilt^T)"](a.Rtilt_np)
    for b in a.boxes3d:
        corners = b.corners_upright() @ R.T
        if corners[:, 2].min() <= MIN_Z:
            continue
        uv = project(corners, a.K_np)
        for i, j in BOX_EDGES:
            col = (0, 90, 200) if (i < 4 and j < 4) else (0, 165, 255)
            cv2.line(img, tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[j]).astype(int)), col, 2, cv2.LINE_AA)
        top = uv[4:].min(axis=0)
        cv2.putText(img, b.cls, (int(top[0]), max(12, int(top[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-stats", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config("dataset")
    sun_root = data_root(cfg) / "sunrgbd"
    anns = {a.image_id: a for a in load_annotations(sun_root)}
    test = [i for i in read_split(REPO / "configs" / "splits" / "sunrgbd_test.txt") if anns[i].boxes3d]
    rng = np.random.default_rng(args.seed)
    stat_ids = rng.permutation(test)[: args.n_stats].tolist()
    vis_ids = stat_ids[: args.n]

    reader = SunRGBD(sun_root, records=[_record(anns[i]) for i in stat_ids],
                     min_m=cfg["depth"]["min_m"], max_m=cfg["depth"]["max_m"])
    out_dir = REPO / "benchmarks" / "data_quality" / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_t = {k: [] for k in TRANSFORMS}
    per_sensor: dict[str, list] = {}
    for k_img, iid in enumerate(stat_ids):
        a = anns[iid]
        f = reader[k_img]
        for name, fn in TRANSFORMS.items():
            s = box_stats(a, f.depth_m, f.valid, fn(a.Rtilt_np))
            per_t[name].extend(s)
            if name.startswith("correct"):
                per_sensor.setdefault(a.sensor, []).extend(s)
        if iid in vis_ids:
            cv2.imwrite(str(out_dir / f"sun_{iid}_{a.sensor}.png"), draw(a, f.rgb))

    def summarize(s):
        if not s:
            return {"boxes": 0}
        e = np.array(s)
        err_med, ext_med = float(np.median(e[:, 0])), float(np.median(e[:, 1]))
        return {"boxes": len(s), "median_abs_err_m": round(err_med, 3), "median_z_extent_m": round(ext_med, 3),
                "frac_err_lt_extent": round(float((e[:, 0] < e[:, 1]).mean()), 3), "pass": err_med < ext_med}

    report = {"images": len(stat_ids), "seed": args.seed, "split": "test",
              "transforms": {k: summarize(v) for k, v in per_t.items()},
              "correct_by_sensor": {k: summarize(v) for k, v in sorted(per_sensor.items())},
              "overlays": sorted(p.name for p in out_dir.glob("sun_*.png"))}
    (REPO / "benchmarks" / "data_quality" / "annotations_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for k, v in report["transforms"].items():
        print(f"{k:55s} {v}")
    for k, v in report["correct_by_sensor"].items():
        print(f"  {k:10s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
