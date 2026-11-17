"""Spec 03.2 R5 / T6: point-cloud viewer. It shows the frame cloud, the detected object centres, and (for SUN) the GT 3D boxes.

  python -I scripts/view_pointcloud.py --sun 000123            # one SUN image id
  python -I scripts/view_pointcloud.py --tum rgbd_dataset_freiburg1_desk 300   # sequence + frame index
  python -I scripts/view_pointcloud.py --suggest-sun           # val ids at the near / mid / far depth terciles
  add --show to open an interactive open3d window (needs the `viz` extra)

Detections come from the stored Phase 02 runs (benchmarks/detection/*.jsonl) at conf >= conf_operational, so the
viewer never re-runs the detector. Centres are computed with configs/depth.yaml. TUM frames carry the lens
distortion when configs/dataset.yaml says so.

Outputs go to benchmarks/geometry/viewer/<name>.*:
  .png   matplotlib view: top-down (X-Z) and camera view (X-Y) side by side. It is always written.
  .ply   frame cloud (voxel 1 cm) and centres.
  .json  per object: class, score, status, centre. For SUN it also gives the nearest GT 3D box and the distance from
         the centre to that box (0 = inside). This is an aid for the by-eye check of AC3, not a metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import data_root, load_config  # noqa: E402
from perception.data.sunrgbd import SunRecord, SunRGBD  # noqa: E402
from perception.data.sunrgbd_annotations import ImageAnn, load_annotations  # noqa: E402
from perception.data.tum import TUMSequence  # noqa: E402
from perception.depth.estimate import AssocParams, estimate_boxes  # noqa: E402
from perception.detection.classes import COCO_NAMES  # noqa: E402
from perception.detection.runio import read_run  # noqa: E402
from perception.geometry.object3d import object_centre  # noqa: E402
from perception.geometry.pointcloud import frame_cloud, voxel_downsample, write_ply  # noqa: E402
from perception.geometry.transforms import cam_to_upright  # noqa: E402

OUT = REPO / "benchmarks" / "geometry" / "viewer"
RUNS = REPO / "benchmarks" / "detection"
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def find_detections(run_glob: str, frame_id: str, conf: float):
    for path in sorted(RUNS.glob(run_glob)):
        if "nms" in path.name or "repro" in path.name:
            continue
        _, recs = read_run(path)
        for r in recs:
            if r["frame_id"] == frame_id:
                b = np.asarray(r["boxes"], np.float64).reshape(-1, 4)
                s, c = np.asarray(r["scores"]), np.asarray(r["class_ids"], int)
                keep = s >= conf
                return b[keep], s[keep], c[keep], path.name
    return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), None


def box_edges_valid(corners: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(corners[i], corners[j]) for i, j in EDGES]


def distance_to_box(c_cam: np.ndarray, box, Rtilt: np.ndarray) -> float:
    """Euclidean distance (m) from a camera-frame point to an oriented GT box; 0 when inside."""
    p = cam_to_upright(c_cam.reshape(1, 3), Rtilt)[0]
    B = np.asarray(box.basis, np.float64)
    local = np.linalg.solve(B.T, p - np.asarray(box.centroid))  # corners = c + (signs*h) @ B
    return float(np.linalg.norm(np.maximum(np.abs(local) - np.asarray(box.coeffs), 0.0)))


def load_sun(image_id: str, dcfg: dict):
    root = data_root(dcfg) / "sunrgbd"
    ann: ImageAnn = next(a for a in load_annotations(root) if a.image_id == image_id)
    rec = SunRecord(index=ann.index, sequence=ann.sequence, sensor=ann.sensor, rgb_rel=ann.rgb_rel,
                    depth_rel=ann.depth_rel, K=ann.K_np, Rtilt=ann.Rtilt_np)
    fr = SunRGBD(root, records=[rec], min_m=dcfg["depth"]["min_m"], max_m=dcfg["depth"]["max_m"])[0]
    return fr, ann


def suggest_sun(dcfg: dict, conf: float) -> None:
    """Val images with >= 2 detections, grouped by the median depth of their valid pixels (terciles).
    Only val is used, so the choice never looks at test."""
    import cv2
    from perception.depth.decode import decode_sun
    root = data_root(dcfg) / "sunrgbd"
    anns = {a.image_id: a for a in load_annotations(root)}
    _, recs = read_run(RUNS / "sun_val_full.jsonl")
    rows = []
    for r in recs:
        if sum(s >= conf for s in r["scores"]) < 2 or not anns[r["frame_id"]].boxes3d:
            continue
        raw = cv2.imread(str(root / anns[r["frame_id"]].depth_rel), cv2.IMREAD_UNCHANGED)
        d, v = decode_sun(raw, dcfg["depth"]["min_m"], dcfg["depth"]["max_m"])
        if v.mean() > 0.5:
            rows.append((float(np.median(d[v])), r["frame_id"], anns[r["frame_id"]].sensor))
    rows.sort()
    n = len(rows)
    for name, lo, hi in (("near", 0, n // 3), ("mid", n // 3, 2 * n // 3), ("far", 2 * n // 3, n)):
        mid = rows[(lo + hi) // 2]
        print(f"{name}: {hi - lo} images, median-depth {rows[lo][0]:.2f}-{rows[hi - 1][0]:.2f} m; "
              f"middle example {mid[1]} ({mid[0]:.2f} m, {mid[2]})")


def render_png(path: Path, pts, cols, centres, labels, gt_boxes, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    views = (("top-down (X right, Z forward)", 0, 2, False), ("camera view (X right, Y down)", 0, 1, True))
    for ax, (name, a, b, flip) in zip(axes, views):
        ax.scatter(pts[:, a], pts[:, b], c=cols / 255.0, s=0.4, linewidths=0, rasterized=True)
        for corners, gl in gt_boxes:
            for p, q in box_edges_valid(corners):
                ax.plot([p[a], q[a]], [p[b], q[b]], color="tab:green", lw=1.0)
        for c, lab in zip(centres, labels):
            if np.all(np.isfinite(c)):
                ax.scatter([c[a]], [c[b]], s=90, c="red", marker="o", edgecolors="white", linewidths=1.2, zorder=5)
                ax.annotate(lab, (c[a], c[b]), xytext=(4, 4), textcoords="offset points", fontsize=8, color="red")
        # Limits from the 0.5-99.5 percentiles plus all centres, so a few stray far returns do not squash the scene.
        fin = centres[np.all(np.isfinite(centres), axis=1)]
        for dim, setter in ((a, ax.set_xlim), (b, ax.set_ylim)):
            lo, hi = np.percentile(pts[:, dim], [0.5, 99.5]) if len(pts) else (0.0, 1.0)
            if len(fin):
                lo, hi = min(lo, fin[:, dim].min()), max(hi, fin[:, dim].max())
            pad = 0.05 * (hi - lo) + 0.05
            setter(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)" if flip else "Z (m)")
        if flip:
            ax.invert_yaxis()
        ax.grid(alpha=0.3)
    fig.suptitle(title + "   (red = detected centres, green = GT 3D boxes)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def capture_open3d(path: Path, pts, cols, centres, gt_boxes) -> bool:
    """Offscreen open3d render from above and behind the camera; False if open3d is missing or has no GL context."""
    try:
        import open3d as o3d
        geoms = _o3d_geoms(pts, cols, centres, gt_boxes)
        vis = o3d.visualization.Visualizer()
        if not vis.create_window(visible=False, width=1280, height=900):
            return False
        for g_ in geoms:
            vis.add_geometry(g_)
        vc = vis.get_view_control()
        vc.set_lookat(np.median(pts, axis=0))
        vc.set_front([0.0, -0.6, -1.0])  # camera sits behind and above the RGB-D camera, looking forward and down
        vc.set_up([0.0, -1.0, 0.0])
        vc.set_zoom(0.55)
        vis.get_render_option().point_size = 2.0
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(path), do_render=True)
        vis.destroy_window()
        return path.exists()
    except Exception as exc:  # noqa: BLE001 - the matplotlib PNG is the required output; this one is optional
        print(f"open3d capture skipped: {exc}")
        return False


def _o3d_geoms(pts, cols, centres, gt_boxes):
    import open3d as o3d
    geoms = []
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(cols / 255.0)
    geoms.append(pc)
    for c in centres:
        if np.all(np.isfinite(c)):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            s.translate(c)
            s.paint_uniform_color([1, 0, 0])
            geoms.append(s)
    for corners, _ in gt_boxes:
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(corners), o3d.utility.Vector2iVector(EDGES))
        ls.paint_uniform_color([0, 0.7, 0])
        geoms.append(ls)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
    return geoms


def show_open3d(pts, cols, centres, gt_boxes) -> None:
    import open3d as o3d
    o3d.visualization.draw_geometries(_o3d_geoms(pts, cols, centres, gt_boxes), window_name="Project 5 point cloud")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sun", metavar="IMAGE_ID")
    g.add_argument("--tum", nargs=2, metavar=("SEQUENCE", "FRAME_INDEX"))
    g.add_argument("--suggest-sun", action="store_true")
    ap.add_argument("--name", help="output basename (default derived from the frame)")
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--show", action="store_true", help="open an interactive open3d window")
    args = ap.parse_args()

    dcfg, gcfg = load_config("dataset"), load_config("depth")
    conf = float(load_config("detector")["conf_operational"])
    if args.suggest_sun:
        suggest_sun(dcfg, conf)
        return 0
    p = AssocParams.from_config(gcfg)
    k, floor = float(gcfg["centre"]["k"]), float(gcfg["centre"]["sigma_floor_m"])

    gt, Rtilt = [], None
    if args.sun:
        image_id = f"{int(args.sun):06d}"
        fr, ann = load_sun(image_id, dcfg)
        Rtilt = ann.Rtilt_np
        gt = [(b.corners_cam(Rtilt), b) for b in ann.boxes3d]
        boxes, scores, cls, run = find_detections("sun_*_full.jsonl", image_id, conf)
        name = args.name or f"sun_{image_id}"
        title = f"SUN {image_id} ({ann.sensor})"
    else:
        seq = TUMSequence.from_config(args.tum[0], dcfg)
        fr = seq[int(args.tum[1])]
        boxes, scores, cls, run = find_detections("tum_*_full.jsonl", fr.frame_id, conf)
        name = args.name or f"tum_{seq.name.replace('rgbd_dataset_', '')}_{int(args.tum[1]):04d}"
        title = f"TUM {seq.name} frame {args.tum[1]} (t={fr.frame_id}, undistort={'on' if fr.dist is not None else 'off'})"
    if run is None:
        print("warning: no stored detections for this frame; showing the cloud and GT only")

    pts, cols = frame_cloud(fr.depth_m, fr.valid, fr.K, fr.rgb, stride=1, dist=fr.dist)
    pts, cols = voxel_downsample(pts, args.voxel, cols)
    ests = estimate_boxes(fr.depth_m, fr.valid, boxes, p)
    objs = [object_centre(fr.depth_m, fr.valid, fr.K, e, k, floor, fr.dist) for e in ests]
    centres = np.array([o.centre for o in objs], np.float64).reshape(-1, 3)
    labels = [f"{COCO_NAMES[c]} {s:.2f}" for c, s in zip(cls, scores)]

    rows = []
    for o, e, lab, c in zip(objs, ests, labels, centres):
        row = {"det": lab, "status": e.status, "z_m": round(e.z, 3) if np.isfinite(e.z) else None,
               "centre_cam_m": [round(v, 3) for v in o.centre] if np.all(np.isfinite(c)) else None}
        if gt and np.all(np.isfinite(c)):
            d = [distance_to_box(c, b, Rtilt) for _, b in gt]
            j = int(np.argmin(d))
            row.update(nearest_gt=gt[j][1].cls, dist_to_gt_box_m=round(d[j], 3))
        rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    render_png(OUT / f"{name}.png", pts, cols, centres, labels, gt, title)
    o3d_png = capture_open3d(OUT / f"{name}_o3d.png", pts, cols, centres, gt)
    write_ply(OUT / f"{name}_cloud.ply", pts, cols)
    finite = centres[np.all(np.isfinite(centres), axis=1)]
    write_ply(OUT / f"{name}_centres.ply", finite, np.tile(np.array([[255, 0, 0]], np.uint8), (len(finite), 1)))
    summary = {"frame": title, "run": run, "conf_operational": conf, "n_cloud_points": int(len(pts)),
               "voxel_m": args.voxel, "open3d_png": o3d_png, "gt_boxes": [b.cls for _, b in gt], "objects": rows}
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT / name}.png/.json/_cloud.ply/_centres.ply")
    if args.show:
        show_open3d(pts, cols, centres, gt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
