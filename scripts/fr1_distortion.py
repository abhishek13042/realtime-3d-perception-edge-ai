"""Spec 03.2 R6 / AC4: effect of ignoring TUM fr1 lens distortion on object centres (no GT needed).

For every stored fr1 detection at conf >= conf_operational with a centre (status != no_depth), the centre is
computed twice from the SAME inlier pixels and depths:
  raw          - pixel (u, v) unprojected with K (current pipeline, distortion ignored);
  undistorted  - (u, v) passed through cv2.undistortPoints(K, dist) to normalised coordinates, then X = x_n * Z.
The 3D displacement |raw - undistorted| is reported overall and by image radius of the box centre
(distance from the principal point, px).

Decision rule (fixed in the spec before measuring): p90 displacement > 2 cm -> enable undistortion for fr1;
otherwise document it as a known small bias.

Usage:  python -I scripts/fr1_distortion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import load_config  # noqa: E402
from perception.data.tum import TUM_DISTORTION, TUMSequence  # noqa: E402
from perception.depth.estimate import AssocParams, estimate_boxes  # noqa: E402
from perception.detection.runio import read_run  # noqa: E402
from perception.geometry.camera import unproject  # noqa: E402
from perception.geometry.object3d import inlier_mask  # noqa: E402

SEQ = "rgbd_dataset_freiburg1_desk"
RUN = REPO / "benchmarks" / "detection" / "tum_fr1_desk_full.jsonl"
THRESH_M = 0.02
RADIUS_BINS = (0, 100, 200, 300, 1000)


def stats(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "median_cm": round(float(np.median(x)) * 100, 3),
            "p90_cm": round(float(np.percentile(x, 90)) * 100, 3), "max_cm": round(float(x.max()) * 100, 3)}


def main() -> int:
    dcfg, gcfg = load_config("dataset"), load_config("depth")
    conf_op = float(load_config("detector")["conf_operational"])
    p = AssocParams.from_config(gcfg)
    k, floor = float(gcfg["centre"]["k"]), float(gcfg["centre"]["sigma_floor_m"])
    seq = TUMSequence.from_config(SEQ, dcfg)
    K = seq.K
    dist = np.array(TUM_DISTORTION["freiburg1"], np.float64)
    index = {f"{t:.6f}": i for i, (t, _, _) in enumerate(seq.items)}
    _, recs = read_run(RUN)

    disp, rad, depth_z, n_frames = [], [], [], 0
    for r in recs:
        if r["frame_id"] not in index:
            continue
        b, s = np.asarray(r["boxes"], np.float64).reshape(-1, 4), np.asarray(r["scores"])
        b = b[s >= conf_op]
        if not len(b):
            continue
        fr = seq[index[r["frame_id"]]]
        n_frames += 1
        for box, est in zip(b, estimate_boxes(fr.depth_m, fr.valid, b, p)):
            if est.status == "no_depth":
                continue
            rr, cc = inlier_mask(fr.depth_m, fr.valid, est, k, floor)
            z = fr.depth_m[rr, cc].astype(np.float64)
            uv = np.stack([cc, rr], 1).astype(np.float64)
            raw = np.median(unproject(uv, z, K), axis=0)
            xn = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
            und = np.median(np.c_[xn * z[:, None], z], axis=0)
            disp.append(float(np.linalg.norm(raw - und)))
            # Box centre in OpenCV pixel coordinates (box convention minus 0.5) relative to the principal point.
            cx, cy = (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5
            rad.append(float(np.hypot(cx - K[0, 2], cy - K[1, 2])))
            depth_z.append(float(raw[2]))

    disp, rad, depth_z = np.array(disp), np.array(rad), np.array(depth_z)
    by_radius = {f"{lo}-{hi}px": stats(disp[(rad >= lo) & (rad < hi)]) for lo, hi in zip(RADIUS_BINS, RADIUS_BINS[1:])}
    rel = disp / depth_z
    overall = stats(disp)
    p90 = float(np.percentile(disp, 90))
    decision = "enable undistortion for fr1" if p90 > THRESH_M else "keep raw pixels; document as a known small bias"
    report = {
        "spec": "03.2 R6 / AC4", "label": "Measured (no GT): centre displacement raw vs undistorted pixels",
        "sequence": SEQ, "run": str(RUN.relative_to(REPO)).replace("\\", "/"), "conf_operational": conf_op,
        "K": K.tolist(), "dist_k1_k2_p1_p2_k3": dist.tolist(), "params": {**p.__dict__, "centre_k": k,
                                                                         "sigma_floor_m": floor},
        "n_frames": n_frames, "overall": overall, "by_box_centre_radius": by_radius,
        "relative_to_depth": {"median_pct": round(float(np.median(rel)) * 100, 3),
                              "p90_pct": round(float(np.percentile(rel, 90)) * 100, 3)},
        "rule": {"threshold_p90_cm": THRESH_M * 100, "p90_cm": round(p90 * 100, 3), "decision": decision},
    }
    out = REPO / "benchmarks" / "geometry" / "fr1_distortion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k_: report[k_] for k_ in ("n_frames", "overall", "by_box_centre_radius",
                                                 "relative_to_depth", "rule")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
