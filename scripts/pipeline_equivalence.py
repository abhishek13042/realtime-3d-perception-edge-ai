"""Spec 05.3 R2 / AC2: the live pipeline equals the offline Phase 04 run on fr1_desk (all frames, stride 1).

The pipeline runs the detector live; the offline run (benchmarks/tracking/tum_fr1_s1.jsonl) used the stored
detections (benchmarks/detection/tum_fr1_desk_full.jsonl). Two comparisons per frame:
1. live detections vs stored detections (exact), which isolates detector non-determinism;
2. pipeline tracked objects (frame_line form: IDs, det indices, boxes, statuses, 3D centres) vs the offline tracks.
Not a timing run. Writes benchmarks/results/pipeline_equivalence_fr1.json.

Usage:  python -I scripts/pipeline_equivalence.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import CONFIG_DIR, load_config  # noqa: E402
from perception.data.tum import TUMSequence  # noqa: E402
from perception.detection.runio import read_run  # noqa: E402
from perception.detection.torch_backend import build_detector  # noqa: E402
from perception.eval.results_index import git_state, sha256_file  # noqa: E402
from perception.pipeline import Pipeline, validate_result  # noqa: E402
from perception.tracking.runner import align_records  # noqa: E402
from perception.tracking.track_history import frame_line, read_tracks  # noqa: E402

SEQ = "rgbd_dataset_freiburg1_desk"
DET_RUN = REPO / "benchmarks" / "detection" / "tum_fr1_desk_full.jsonl"
OFFLINE = REPO / "benchmarks" / "tracking" / "tum_fr1_s1.jsonl"
OUT = REPO / "benchmarks" / "results" / "pipeline_equivalence_fr1.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    dcfg, gcfg, tcfg, detcfg = (load_config(n) for n in ("dataset", "depth", "tracker", "detector"))
    seq = TUMSequence.from_config(SEQ, dcfg)
    records = align_records(seq, read_run(DET_RUN)[1])
    off_header, offline = read_tracks(OFFLINE)
    n = len(offline) if a.limit is None else min(a.limit, len(offline))
    det = build_detector(detcfg)
    pipe = Pipeline(det, gcfg, tcfg, 1, conf_operational=detcfg["conf_operational"])

    det_diff, trk_diff, schema_err, max_box = [], [], 0, 0.0
    n_tracked = n_objects = 0
    t0 = time.perf_counter()
    for i in range(n):
        frame = seq[i]
        # Capture the live detections the pipeline uses (wrap detect once; no change to its behaviour).
        live = {}
        orig = det.detect

        def capture(rgb, _orig=orig):
            live["d"] = _orig(rgb)
            return live["d"]
        det.detect = capture
        r = pipe.process(frame, seq_index=i)
        det.detect = orig
        d, rec = live["d"], records[i]
        sb = np.asarray(rec["boxes"], np.float64).reshape(-1, 4)
        same_det = (len(d) == len(sb) and np.array_equal(d.boxes.astype(np.float64), sb)
                    and np.array_equal(d.scores.astype(np.float64), np.asarray(rec["scores"], np.float64))
                    and np.array_equal(d.class_ids, np.asarray(rec["class_ids"], np.int64)))
        if not same_det:
            det_diff.append(i)
            if len(d) == len(sb):
                max_box = max(max_box, float(np.abs(d.boxes.astype(np.float64) - sb).max(initial=0.0)))
        line = frame_line(r.frame_index, r.seq_index, r.frame_id, float(r.frame_id), r.n_dets, r.observations(), {})
        ref = offline[i]
        if line["tracks"] != ref["tracks"] or line["n_dets"] != ref["n_dets"] or line["frame_id"] != ref["frame_id"]:
            trk_diff.append(i)
        try:
            validate_result(json.loads(json.dumps(r.to_dict(), allow_nan=False)))
        except ValueError:
            schema_err += 1
        n_tracked += len(ref["tracks"])
        n_objects += len(r.objects)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{n} frames; detection diffs {len(det_diff)}, track diffs {len(trk_diff)}", flush=True)
    wall = time.perf_counter() - t0

    res = {
        "spec": "05.3 R2", "label": "Measured: correctness check (not a timing run)",
        "sequence": SEQ, "stride": 1, "n_frames": n, "n_frames_in_sequence": len(offline),
        "detections_identical_frames": n - len(det_diff), "detection_diff_frames": det_diff[:50],
        "max_box_abs_diff_px": max_box,
        "tracks_identical_frames": n - len(trk_diff), "track_diff_frames": trk_diff[:50],
        "tracked_boxes_offline": n_tracked, "pipeline_objects": n_objects,
        "schema_errors": schema_err,
        "equal": not trk_diff and not schema_err,
        "wall_s_including_io": round(wall, 1),
        "header": {**git_state(REPO), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "config_sha256": {k: sha256_file(CONFIG_DIR / f"{k}.yaml") for k in ("depth", "tracker", "detector")},
                   "input_sha256": {"offline_tracks": sha256_file(OFFLINE), "detections": sha256_file(DET_RUN)},
                   "detector_weights_sha256": det.weights_sha256, "torch_threads": detcfg.get("threads")},
    }
    if a.limit is None:
        OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
    print(json.dumps({k: res[k] for k in ("n_frames", "detections_identical_frames", "tracks_identical_frames",
                                          "schema_errors", "equal", "wall_s_including_io")}))
    return 0 if res["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
