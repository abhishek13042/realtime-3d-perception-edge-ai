"""Stored TUM detections -> ByteTrack tracks + 3D centres (spec 04.1 R4-R8, AC2-AC4).

    python scripts/run_tracking.py --sequence fr1 --stride 1
    python scripts/run_tracking.py --sequence fr3 --stride 6

Writes benchmarks/tracking/tum_<seq>_s<stride>.jsonl (frame lines, protocol §7 header) and a summary JSON beside it
with the tracker timing (R8), the determinism re-run (R7/AC2) and the 3D spot check (AC3).
Tracker settings come from configs/tracker.yaml (per-stride values when 04.2 has written them).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from perception.config import CONFIG_DIR, load_config  # noqa: E402
from perception.data.tum import TUMSequence  # noqa: E402
from perception.depth.estimate import AssocParams, estimate_box  # noqa: E402
from perception.detection.runio import latency_stats, read_run  # noqa: E402
from perception.geometry.object3d import object_centre, to_world  # noqa: E402
from perception.tracking.bytetrack import TrackerParams  # noqa: E402
from perception.tracking.runner import Centres3D, align_records, track_sequence  # noqa: E402
from perception.tracking.track_history import TrackHistory, read_tracks, run_header, write_tracks  # noqa: E402

SEQUENCES = {
    "fr1": ("rgbd_dataset_freiburg1_desk", REPO / "benchmarks" / "detection" / "tum_fr1_desk_full.jsonl"),
    "fr3": ("rgbd_dataset_freiburg3_long_office_household",
            REPO / "benchmarks" / "detection" / "tum_fr3_long_office_full.jsonl"),
}
OUT = REPO / "benchmarks" / "tracking"
SPOT_N, SPOT_SEED = 20, 0
TIMING_PASSES = 3


def spot_check(seq, lines: list[dict], depth_cfg: dict) -> dict:
    """AC3: for 20 random frames with tracks, recompute every box's 3D directly and compare with the file."""
    p = AssocParams.from_config(depth_cfg)
    k, floor = float(depth_cfg["centre"]["k"]), float(depth_cfg["centre"]["sigma_floor_m"])
    cand = [ln for ln in lines if ln["tracks"]]
    pick = random.Random(SPOT_SEED).sample(cand, min(SPOT_N, len(cand)))
    n_boxes, worst, status_mismatch = 0, 0.0, 0
    for ln in pick:
        fr = seq[ln["seq_index"]]
        for d in ln["tracks"]:
            est = estimate_box(fr.depth_m, fr.valid, d["box"], p)
            o = object_centre(fr.depth_m, fr.valid, fr.K, est, k, floor, fr.dist)
            n_boxes += 1
            status_mismatch += int(o.status != d["status"])
            if all(math.isfinite(x) for x in o.centre):
                if d["centre_cam"] is None:
                    status_mismatch += 1
                    continue
                worst = max(worst, float(np.max(np.abs(np.subtract(o.centre, d["centre_cam"])))))
                if fr.T_world_cam is not None:
                    w = to_world(np.asarray(o.centre), fr.T_world_cam)[0]
                    worst = max(worst, float(np.max(np.abs(w - np.asarray(d["centre_world"])))))
            elif d["centre_cam"] is not None:
                status_mismatch += 1
    return {"frames": len(pick), "boxes": n_boxes, "max_abs_diff_m": worst, "mismatches": status_mismatch,
            "pass": status_mismatch == 0 and worst <= 1e-5}  # file stores 6 decimals


def tracker_only_timing(records, params, stride, passes: int = TIMING_PASSES) -> dict:
    meds, p95s = [], []
    for _ in range(passes):
        _, ms = track_sequence(records, params, stride, lambda fi, si, tracked: [])
        st = latency_stats(ms[10:])
        meds.append(st["median"]); p95s.append(st["p95"])
    return {"passes": passes, "median": float(np.median(meds)), "p95": float(np.median(p95s)),
            "per_pass_median": meds, "target_ms": 2.0, "meets_target": float(np.median(meds)) < 2.0}


def strip_timings(lines):
    return [{k: v for k, v in ln.items() if k != "timings_ms"} for ln in lines]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", choices=sorted(SEQUENCES), required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-checks", action="store_true", help="skip the determinism re-run and the 3D spot check")
    ap.add_argument("--timing-only", action="store_true",
                    help="R8 only: tracker-only timing in a fresh process (no frames, no depth), 5 passes")
    a = ap.parse_args()

    seq_name, det_path = SEQUENCES[a.sequence]
    dcfg, gcfg, tcfg = load_config("dataset"), load_config("depth"), load_config("tracker")
    params = TrackerParams.from_config(tcfg, stride=a.stride)
    seq = TUMSequence.from_config(seq_name, dcfg)
    det_header, recs = read_run(det_path)
    records = align_records(seq, recs)

    if a.timing_only:
        res = {"run": f"tum_{a.sequence}_s{a.stride}", "params": params.__dict__,
               "tracker_update_ms": tracker_only_timing(records, params, a.stride, 5),
               "note": "fresh process, tracker only, no image or depth work before or between passes",
               "created_utc": run_header(REPO, {"tracker": CONFIG_DIR / "tracker.yaml"}, {})["created_utc"]}
        (OUT / f"tum_{a.sequence}_s{a.stride}_timing.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res["tracker_update_ms"]))
        return 0

    centres = Centres3D(seq, gcfg)
    lines, upd_ms = track_sequence(records, params, a.stride, centres)

    run = f"tum_{a.sequence}_s{a.stride}"
    header = run_header(REPO, {"tracker": CONFIG_DIR / "tracker.yaml", "depth": CONFIG_DIR / "depth.yaml",
                               "dataset": CONFIG_DIR / "dataset.yaml"}, {"detections": det_path},
                        {"run": run, "sequence": seq_name, "stride": a.stride, "n_processed": len(lines),
                         "params": params.__dict__, "detection_run": det_path.name,
                         "detector_weights_sha256": det_header.get("weights_sha256")})
    path = OUT / f"{run}.jsonl"
    write_tracks(path, header, lines)

    hist = TrackHistory.from_frames(lines)
    n_obs = sum(len(v) for v in hist.tracks.values())
    statuses = {}
    for v in hist.tracks.values():
        for o in v:
            statuses[o.status] = statuses.get(o.status, 0) + 1
    summary = {
        "run": run, "file": path.name, "sequence": seq_name, "stride": a.stride, "params": params.__dict__,
        "n_processed_frames": len(lines), "n_tracks": len(hist.tracks),
        "n_confirmed_tracks": len(hist.confirmed(int(tcfg["min_track_len"]))), "min_track_len": tcfg["min_track_len"],
        "n_tracked_boxes": n_obs, "status_counts": statuses,
        "frames_without_tracks": sum(1 for ln in lines if not ln["tracks"]),
        "tracker_update_ms_in_run": latency_stats(upd_ms[10:]),  # first 10 frames = warm-up (03.1 R9 convention)
        "tracker_update_ms": tracker_only_timing(records, params, a.stride),
        "tracker_update_note": ("Tracker.update only (2D association), excluding frame loading, depth and 3D. "
                                "R8 figure = tracker_update_ms: tracker-only passes, median of the per-pass medians. "
                                "_in_run is interleaved with image decoding and depth work (cache effects)."),
        "depth_3d_total_ms": round(centres.ms, 1),
        "created_utc": header["created_utc"], "git_commit": header["git_commit"], "git_dirty": header["git_dirty"],
    }
    if not a.no_checks:
        lines2, _ = track_sequence(records, params, a.stride, centres)
        _, reread = read_tracks(path)
        summary["determinism"] = {"identical": strip_timings(lines2) == strip_timings(lines),
                                  "file_roundtrip_identical": strip_timings(reread) == strip_timings(
                                      json.loads(json.dumps(lines)))}
        summary["spot_check_3d"] = spot_check(seq, lines, gcfg)
    (OUT / f"{run}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("run", "n_processed_frames", "n_tracks", "n_confirmed_tracks",
                                              "status_counts", "tracker_update_ms")
                      } | {k: summary.get(k) for k in ("determinism", "spot_check_3d")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
