"""Spec 03.1 R9 / AC3 and spec 03.2 R7 / AC5: CPU cost of depth association, object centres, and frame clouds.

Protocol (same as the detector latency protocol, docs/evaluation_protocol.md):
- Frames are the fixed TUM fr1 200-frame benchmark list. Detections are the stored ones
  (benchmarks/detection/tum_fr1_desk_full.jsonl), kept at conf >= conf_operational (configs/detector.yaml),
  then the top 50 by score if there are more (the spec bound is <= 50 boxes). Real frames carry far fewer boxes,
  so a stress stage also runs 50 real boxes (seeded sample from all kept boxes) on every frame.
- Frame decoding is NOT timed. 10 warm-up frames are run and excluded; then every frame is timed.
- Reported: median and p95 per stage, box counts, machine, and numpy threads.

Usage:  python -I scripts/time_geometry.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import load_config  # noqa: E402
from perception.data.sampling import read_ids  # noqa: E402
from perception.data.tum import TUMSequence  # noqa: E402
from perception.depth.estimate import AssocParams, estimate_boxes  # noqa: E402
from perception.detection.runio import latency_stats, read_run  # noqa: E402
from perception.geometry.object3d import object_centre  # noqa: E402
from perception.geometry.pointcloud import frame_cloud  # noqa: E402

SEQ = "rgbd_dataset_freiburg1_desk"
LIST = REPO / "configs" / "benchmark_frames" / f"tum_{SEQ}_200.txt"
RUN = REPO / "benchmarks" / "detection" / "tum_fr1_desk_full.jsonl"
WARMUP, MAX_BOXES = 10, 50


def numpy_threads() -> dict:
    try:
        from threadpoolctl import threadpool_info
        return {f"{d['internal_api']}": d["num_threads"] for d in threadpool_info()}
    except ImportError:
        return {"note": "threadpoolctl not installed",
                **{k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def main() -> int:
    dcfg, gcfg = load_config("dataset"), load_config("depth")
    conf_op = float(load_config("detector")["conf_operational"])
    p = AssocParams.from_config(gcfg)
    k, floor = float(gcfg["centre"]["k"]), float(gcfg["centre"]["sigma_floor_m"])

    seq = TUMSequence.from_config(SEQ, dcfg)
    index = {f"{t:.6f}": i for i, (t, _, _) in enumerate(seq.items)}
    ids = read_ids(LIST)
    _, recs = read_run(RUN)
    dets = {r["frame_id"]: r for r in recs}
    missing = [f for f in ids if f not in index or f not in dets]
    if missing:
        raise SystemExit(f"{len(missing)} benchmark frames missing from sequence or run, e.g. {missing[:3]}")

    work = []
    for fid in ids:
        fr = seq[index[fid]]
        r = dets[fid]
        b, s = np.asarray(r["boxes"], np.float64).reshape(-1, 4), np.asarray(r["scores"])
        keep = np.argsort(-s[s >= conf_op], kind="stable")[:MAX_BOXES]
        work.append((fr, b[s >= conf_op][keep]))

    def run(fr, boxes):
        t0 = time.perf_counter()
        ests = estimate_boxes(fr.depth_m, fr.valid, boxes, p)
        t1 = time.perf_counter()
        objs = [object_centre(fr.depth_m, fr.valid, fr.K, e, k, floor, fr.dist) for e in ests]
        t2 = time.perf_counter()
        frame_cloud(fr.depth_m, fr.valid, fr.K, fr.rgb, stride=2, dist=fr.dist)
        t3 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, ests, objs

    for fr, boxes in work[:WARMUP]:
        run(fr, boxes)
    rows, statuses = [], {"valid": 0, "low_coverage": 0, "no_depth": 0}
    for fr, boxes in work:
        a, c, cl, ests, _ = run(fr, boxes)
        rows.append((a, c, cl, len(boxes)))
        for e in ests:
            statuses[e.status] += 1
    a, c, cl, n = (np.array(x) for x in zip(*rows))

    # Stress case at the spec bound: 50 real boxes (seeded sample from the pool of kept boxes) on every frame.
    pool = np.concatenate([b for _, b in work])
    stress = pool[np.random.default_rng(0).choice(len(pool), MAX_BOXES, replace=len(pool) < MAX_BOXES)]
    for fr, _ in work[:WARMUP]:
        run(fr, stress)
    sa, sc = (np.array(x) for x in zip(*[run(fr, stress)[:2] for fr, _ in work]))
    stages = {"association": latency_stats(a), "centres": latency_stats(c),
              "association_plus_centres": latency_stats(a + c), "frame_cloud_stride2_rgb": latency_stats(cl),
              "stress50_association": latency_stats(sa), "stress50_centres": latency_stats(sc)}
    targets = {"association_median_lt_ms": 5.0, "centres_median_lt_ms": 5.0, "frame_cloud_stride2_median_lt_ms": 30.0}
    report = {
        "spec": "03.1 R9 / AC3; 03.2 R7 / AC5",
        "label": "Measured: numpy CPU, local laptop (decode excluded)",
        "machine": {"cpu": "Intel Core i3-1115G4 (2C/4T)", "platform": platform.platform(),
                    "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)},
        "threads": numpy_threads(), "python": platform.python_version(), "numpy": np.__version__,
        "frames": {"list": str(LIST.relative_to(REPO)).replace("\\", "/"), "n": len(rows), "warmup": WARMUP},
        "detections": {"run": str(RUN.relative_to(REPO)).replace("\\", "/"), "conf_operational": conf_op,
                       "max_boxes": MAX_BOXES, "boxes_per_frame": {"median": float(np.median(n)), "max": int(n.max()),
                                                                   "total": int(n.sum())}},
        "params": {**p.__dict__, "centre_k": k, "sigma_floor_m": floor},
        "status_counts": statuses,
        "per_box_ms_stress50": {"association": round(float(np.median(sa)) / MAX_BOXES, 3),
                                "centres": round(float(np.median(sc)) / MAX_BOXES, 3)},
        "stages_ms": stages,
        "targets": targets,
        "pass": {"association": stages["association"]["median"] < 5.0,
                 "centres": stages["centres"]["median"] < 5.0,
                 "stress50_association": stages["stress50_association"]["median"] < 5.0,
                 "stress50_centres": stages["stress50_centres"]["median"] < 5.0,
                 "frame_cloud_stride2": stages["frame_cloud_stride2_rgb"]["median"] < 30.0},
    }
    out = REPO / "benchmarks" / "geometry" / "geometry_timing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, s in stages.items():
        print(f"{name}: median {s['median']:.2f} ms, p95 {s['p95']:.2f} ms")
    print(f"boxes/frame median {np.median(n):.0f}, max {n.max()}; statuses {statuses}; pass {report['pass']}")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
