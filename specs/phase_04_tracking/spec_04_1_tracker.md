# Spec 04.1 — ByteTrack adapter, 3D attachment, track history

| Field | Value |
|---|---|
| Phase | 04 Tracking |
| Status | ✅ Approved 2026-11-11 |

## Objective
Give every detection a persistent track ID over a TUM sequence, using ByteTrack on 2D boxes (architecture §3 [5]). Attach the Phase 03 depth estimate, status and 3D centre to each tracked box, and keep a per-track history in both the camera and the world frame. The tracker itself does not use 3D.

## Requirements
- **R1. Adapter.**
  - `perception/tracking/bytetrack.py` wraps `BYTETracker` behind `Tracker.update(boxes_xyxy, scores, classes) -> list[TrackedBox]`. The input is plain numpy, with no Ultralytics `Results` objects in the interface.
  - `TrackedBox` carries: `track_id`, `box_xyxy`, `score`, `cls`, and `det_index`, the index of the source detection.
  - Only boxes that matched a detection in this frame are output. Coasting (predicted-only) tracks are not output, so no 3D is made up from a Kalman prediction.
- **R2. Class handling.** By default, association is class-aware: a track keeps its first class, and a detection of another class never continues it. Ultralytics' ByteTrack ignores class when matching, so the adapter runs one tracker per class, with IDs kept unique across classes. This choice is recorded, and the class-agnostic variant is kept as a config switch for the ablation in 04.2.
- **R3. Config.** All thresholds live in `configs/tracker.yaml`:
  - `track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `match_thresh`, `track_buffer` (in frames *at the processed rate*), `fuse_score`;
  - `min_track_len` (frames before a track counts as confirmed);
  - `excluded_static_classes`.
  - Starting values: the Ultralytics defaults, except `track_high_thresh = new_track_thresh = conf_operational` (0.2411, chosen on SUN val in 02.3). This way the operating point is not tuned again here.
- **R4. 3D attachment.** For each output box, call `estimate_box` (03.1) and `object_centre` (03.2) with `configs/depth.yaml`. When `T_world_cam` is available, store `centre_world = T_world_cam · centre_cam`. Only `status == valid` centres go into the world-frame metrics. The other statuses are kept and counted.
- **R5. History.** `track_history.py` keeps the per-track lists of frame index, timestamp, box, score, status, `centre_cam` and `centre_world`. It exports to JSON lines (`benchmarks/tracking/<run>.jsonl`), one line per frame, each carrying the header and hash rules of protocol §7.
- **R6. Stride.** `scripts/run_tracking.py --sequence … --stride N` feeds every N-th frame, and `track_buffer` is in processed frames. Stride 1 and 6 are the two reported settings.
- **R7. Determinism.** Two runs of the same input give identical track IDs and boxes.
- **R8. Cost.** The tracker update (excluding depth and 3D) is < 2 ms per frame median on fr1 at stride 1, measured as in 03.1 R9. Depth and 3D cost is already known from Phase 03.

## Tasks
- [x] T1. `tracking/bytetrack.py` (adapter, per-class wrapper), `configs/tracker.yaml`.
- [x] T2. `tracking/track_history.py` + JSONL export.
- [x] T3. `scripts/run_tracking.py` (stored detections → tracks + 3D), at strides 1 and 6, on fr1 and fr3.
- [x] T4. Tests (below); timing (R8).

## Acceptance criteria
- **AC1.** On the synthetic tests, IDs persist through a short occlusion of up to `track_buffer` frames, and a new ID starts after that.
- **AC2.** Determinism (R7) holds on a real fr1 run.
- **AC3.** Each output box's 3D fields equal a direct call to `estimate_box`/`object_centre` on that frame (spot check on 20 frames).
- **AC4.** R8 timing is measured and reported.

## Tests (`tests/test_tracking.py`)
- A synthetic box moving at a constant velocity keeps one ID over 50 frames.
- An occlusion gap shorter than the buffer keeps the ID; a longer gap gets a new ID.
- Two classes with overlapping boxes do not swap IDs (class-aware mode).
- Coasting tracks are not output (R1).
- A detection with conf below `track_low_thresh` is ignored.
- World transform: with a known `T_world_cam`, `centre_world` equals the analytic value.
- The JSONL round trip preserves IDs and centres.

## Dependencies
ultralytics 8.4.208 (installed); 02.2 stored TUM runs; 03.1, 03.2; `configs/depth.yaml` (frozen).

## Results (2026-11-11, measured)
- **AC1 ✅** Synthetic tests (18 in `tests/test_tracking.py`) pass. Measured on Ultralytics 8.4.208: a lost track survives `track_buffer + 1` missed frames; a new track needs one confirming frame except on frame 1.
- **AC2 ✅** Two fr1 runs give identical IDs, boxes and 3D fields; the same holds for all four frozen runs.
- **AC3 ✅** 3D fields match direct `estimate_box`/`object_centre` calls on the 20-frame spot check, for every run.
- **AC4 ✅ measured; R8 met on fr1 only.** Tracker.update median, isolated process, 5 passes (`run_tracking.py --timing-only`):

  | sequence | stride | isolated median (ms) | in-run median (ms) | < 2 ms |
  |---|---|---|---|---|
  | fr1 (R8 target) | 1 | 1.70 | 2.96 | yes (isolated) |
  | fr1 | 6 | 1.39 | 2.78 | yes (isolated) |
  | fr3 | 1 | 2.60 | 2.46 | no |
  | fr3 | 6 | 2.39 | 2.15 | no |

  The in-run figure is higher on fr1, and the code measured 10–12 ms under concurrent load, so the number is load-sensitive. Class-aware mode (one tracker per class) is the cost: class-agnostic takes about 0.8 ms on fr1. fr3 has more classes and tracks, and misses the 2 ms budget by 0.4–0.6 ms.
- Frozen runs: fr1 s1 97 tracks (88 confirmed), fr1 s6 48 (18), fr3 s1 275 (252), fr3 s6 290 (143). Per-stride parameters in `configs/tracker.yaml` (chosen in 04.2).
