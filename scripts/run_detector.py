"""Spec 02.2 R1/R2: run the detector over a frame list and store every detection with provenance (JSONL).

Examples:
  python -I scripts/run_detector.py --sun sunrgbd_test --name sun_test_full
  python -I scripts/run_detector.py --sun sunrgbd_val --name sun_val_full
  python -I scripts/run_detector.py --tum rgbd_dataset_freiburg1_desk --name tum_fr1_desk_full
Interrupted runs continue with --resume, which skips frames already stored.
Timings in these files are logged only for reference. Latency is measured by scripts/bench_latency.py (R3).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import load_config  # noqa: E402
from perception.data.rgb_sources import sun_items, tum_items  # noqa: E402
from perception.detection.runio import RunWriter, completed_ids, frame_record, provenance  # noqa: E402
from perception.detection.torch_backend import build_detector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sun", help="SUN split name (sunrgbd_test) or a list file")
    src.add_argument("--tum", help="TUM sequence directory name")
    ap.add_argument("--list", help="for --tum: restrict to the stamps in this list file")
    ap.add_argument("--name", required=True, help="run name -> benchmarks/detection/<name>.jsonl")
    ap.add_argument("--conf", type=float, help="override configs/detector.yaml conf")
    ap.add_argument("--head", choices=["one2one", "one2many_nms"])
    ap.add_argument("--limit", type=int, help="first N frames only (smoke tests)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config("detector")
    over = {k: v for k, v in (("conf", args.conf), ("head", args.head)) if v is not None}
    items = sun_items(args.sun) if args.sun else tum_items(args.tum, args.list)
    if args.limit:
        items = items[: args.limit]
    out = REPO / "benchmarks" / "detection" / f"{args.name}.jsonl"
    done = completed_ids(out) if args.resume else set()
    todo = [it for it in items if it.frame_id not in done]
    print(f"{args.name}: {len(items)} frames, {len(done)} already done, {len(todo)} to run", flush=True)

    det = build_detector(cfg, **over)
    header = provenance(cfg, REPO / "configs" / "detector.yaml", det,
                        {"run": args.name, "source": args.sun or args.tum, "list": args.list, "n_frames": len(items)})
    failures, t0 = [], time.perf_counter()
    with RunWriter(out, header, resume=args.resume) as w:
        for k, it in enumerate(todo, 1):
            ts = time.perf_counter()
            bgr = cv2.imread(str(it.path), cv2.IMREAD_COLOR)
            if bgr is None:
                failures.append((it.frame_id, f"unreadable image {it.path}"))
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            decode_ms = (time.perf_counter() - ts) * 1e3
            d = det.detect(rgb)
            w.write(frame_record(d, it.frame_id, it.dataset, it.scene_id, it.sensor, rgb.shape[:2],
                                 {"decode": decode_ms, **d.timings_ms}))
            if k % 100 == 0 or k == len(todo):
                el = time.perf_counter() - t0
                print(f"  {k}/{len(todo)}  {el / k * 1e3:.0f} ms/frame  eta {(len(todo) - k) * el / k / 60:.1f} min",
                      flush=True)

    n_lines = len(completed_ids(out))
    print(f"wrote {out.relative_to(REPO)}: {n_lines}/{len(items)} frames")
    for fid, why in failures:
        print(f"  FAILED {fid}: {why}")
    return 0 if n_lines == len(items) and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
