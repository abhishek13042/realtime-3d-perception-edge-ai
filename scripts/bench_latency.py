"""Spec 02.2 R3 / AC3: CPU latency of the PyTorch detector, per stage, on the fixed benchmark frame lists.

Protocol (docs/evaluation_protocol.md):
- 10 warm-up frames are run and excluded.
- Every frame of each list is then timed. Nothing is written to disk inside the timed loop.
- Stages are decode (imread + BGR->RGB, reported separately), preprocess, infer, and postprocess.
  "detector" = pre + infer + post; "end_to_end_with_decode" = decode + detector.
- The report gives median and p95, together with the machine, threads, input size, power state, and peak RSS.

Usage:  python -I scripts/bench_latency.py [--conf 0.25] [--head one2one] [--imgsz 640] [--out NAME]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import psutil

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import load_config  # noqa: E402
from perception.data.rgb_sources import sun_items, tum_items  # noqa: E402
from perception.detection.runio import latency_stats, provenance  # noqa: E402
from perception.detection.torch_backend import build_detector  # noqa: E402

BF = "configs/benchmark_frames"
LISTS = {
    "sunrgbd_test_500": lambda: sun_items(f"{BF}/sunrgbd_test_500.txt"),
    "tum_fr1_desk_200": lambda: tum_items("rgbd_dataset_freiburg1_desk",
                                          f"{BF}/tum_rgbd_dataset_freiburg1_desk_200.txt"),
    "tum_fr3_long_office_200": lambda: tum_items("rgbd_dataset_freiburg3_long_office_household",
                                                 f"{BF}/tum_rgbd_dataset_freiburg3_long_office_household_200.txt"),
}
WARMUP = 10


def power_state() -> dict:
    out = {}
    try:
        b = psutil.sensors_battery()
        out["plugged_in"] = None if b is None else bool(b.power_plugged)
        out["battery_percent"] = None if b is None else b.percent
    except (AttributeError, RuntimeError):
        out["plugged_in"] = None
    try:
        r = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10)
        out["power_scheme"] = r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        out["power_scheme"] = None
    return out


def load(it):
    t = time.perf_counter()
    bgr = cv2.imread(str(it.path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"unreadable {it.path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, (time.perf_counter() - t) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--head", choices=["one2one", "one2many_nms"])
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--lists", nargs="*", default=list(LISTS))
    ap.add_argument("--out", default="latency_cpu_torch")
    args = ap.parse_args()

    cfg = load_config("detector")
    over = {k: v for k, v in (("conf", args.conf), ("head", args.head), ("imgsz", args.imgsz)) if v is not None}
    det = build_detector(cfg, **over)
    proc = psutil.Process()
    power_before = power_state()
    results = {}
    for name in args.lists:
        items = LISTS[name]()
        for it in items[:WARMUP]:
            det.detect(load(it)[0])
        rows = []
        for it in items:
            rgb, dec = load(it)
            d = det.detect(rgb)
            rows.append({"decode": dec, **d.timings_ms})
        stages = {k: latency_stats([r[k] for r in rows]) for k in ("decode", "preprocess", "infer", "postprocess")}
        detector = [r["preprocess"] + r["infer"] + r["postprocess"] for r in rows]
        stages["detector"] = latency_stats(detector)
        stages["end_to_end_with_decode"] = latency_stats([r["decode"] + x for r, x in zip(rows, detector)])
        results[name] = {"n_frames": len(rows), "warmup_frames": WARMUP, "stages_ms": stages,
                         "detector_fps_from_median": round(1000 / stages["detector"]["median"], 2)}
        s = stages["detector"]
        print(f"{name}: detector median {s['median']:.1f} ms, p95 {s['p95']:.1f} ms "
              f"(infer median {stages['infer']['median']:.1f}); decode median {stages['decode']['median']:.1f} ms",
              flush=True)

    mem = proc.memory_info()
    report = {
        "spec": "02.2 R3", "label": "Measured: PyTorch CPU, local laptop", "backend": "pytorch-cpu",
        "machine": {"cpu": "Intel Core i3-1115G4 (2C/4T)", "platform": platform.platform(),
                    "processor": platform.processor(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)},
        "power_before": power_before, "power_after": power_state(),
        "conf_note": "conf is provisional until spec 02.3 sets conf_operational. With the one2one head, conf only "
                     "affects the postprocess filter.",
        "peak_rss_mb": round(getattr(mem, "peak_wset", mem.rss) / 2**20, 1),
        "provenance": provenance(cfg, REPO / "configs" / "detector.yaml", det),
        "lists": results,
    }
    out = REPO / "benchmarks" / "detection" / f"{args.out}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}; peak RSS {report['peak_rss_mb']} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
