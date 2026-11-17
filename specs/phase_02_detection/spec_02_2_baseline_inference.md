# Spec 02.2 — Baseline Inference Runs and Latency

| Field | Value |
|---|---|
| Phase | 02 — Detection |
| Status | ✅ Done 2026-11-10 |
| Depends on | 02.1, 01.4 (benchmark frame lists) |
| Blocks | 02.3, 04.x, 06.x (baseline latency to compare against) |

## Objective
Run the detector over fixed frame lists, store every detection with full provenance, and measure CPU latency per stage. The stored detections are the input to evaluation (02.3) and tracking (Phase 04). Nothing downstream re-runs the model just to get boxes.

## Requirements
- **R1.** `scripts/run_detector.py --list <split or benchmark list> [--dataset sunrgbd|tum --sequence ...]` writes `benchmarks/detection/<run_name>.jsonl`, with one line per frame: `{frame_id, dataset, scene_id, sensor, image_hw, boxes, scores, class_ids, timings_ms}`. A header line records the model SHA-256, the input size, thresholds, thread count, package versions, the git commit, and the config hash.
- **R2. Runs.**
  | Run | Frames | Threshold | Used for |
  |---|---|---|---|
  | `sun_test_full` | all 5,050 test images | 0.001 | mAP (02.3) |
  | `sun_val_full` | all 528 val images | 0.001 | Choosing the operational threshold (02.3) |
  | `tum_<seq>_full` | every associated frame of both sequences | 0.001 | Tracking (Phase 04) and the measured object-presence check |
  | `latency_*` | `configs/benchmark_frames/*_200.txt` / `sunrgbd_test_500.txt` | operational | Latency only |

  The estimated CPU time is about 1.5–3 h in total [Target, estimated at ~100–200 ms per image]. Runs are resumable: frames already in the JSONL are skipped.
- **R3. Latency protocol** (as in docs/evaluation_protocol.md):
  - 10 warm-up frames are excluded.
  - Report the median and p95 per stage (decode / preprocess / infer / postprocess) and end to end.
  - Image loading from disk is reported **separately**, not inside "detector latency".
  - Record the thread count, power plan / plugged-in state if available, and the machine (i3-1115G4, PyTorch CPU).
  - Also record peak RSS.
- **R4. Measured object presence (closes 01.4 R5).** For each TUM sequence, report the share of frames with at least one detection at the operational threshold, and the per-class counts.
- **R5. Gallery.** Save 24 overlays to `benchmarks/detection/gallery/`: 4 per SUN sensor plus 4 per TUM sequence. Each shows the detections at the operational threshold, together with the GT 2D boxes (SUN only).
- **R6. RAM.** Frames are streamed and nothing accumulates except the result lines. This must work within 7.7 GB alongside the OS.

## Tasks
- [x] T1. The run script, with resume support and the header line.
- [x] T2. A latency harness, kept separate from the full runs so that its timings are not disturbed by JSONL writes.
- [x] T3. Do the four runs from R2.
- [x] T4. Object presence and gallery.

## Acceptance criteria
- **AC1.** All R2 runs are complete: line counts equal the list sizes, and no frame errors were swallowed. Any failed frame is listed with its reason.
- **AC2.** A re-run of 50 frames from `sun_test_full` reproduces the stored detections exactly (this follows from 02.1 AC2).
- **AC3.** The latency report has the median and p95 per stage, plus the machine, threads, input size, and warm-up count. It is written to `benchmarks/detection/latency_cpu_torch.json`, and no number is presented without those labels.
- **AC4.** Object presence per TUM sequence is measured and added to `docs/data_quality_report.md` (replacing the qualitative note).
- **AC5.** The gallery exists, and I review it visually. Systematic failure patterns are written down, e.g. "kv2 dark images: missed chairs".

## Results (measured 2026-11-10)
| AC | Result |
|---|---|
| AC1 | ✅ sun_val_full 528/528, tum_fr1_desk_full 573/573, tum_fr3_long_office_full 2485/2485, sun_test_full 5050/5050 (= list sizes). No failed frames; no error lines in `full_runs.log`. |
| AC2 | ✅ `repro_check_50` (fresh process) vs `sun_test_full`: 50/50 frames, 9,204 detections, boxes/scores/classes **exactly** equal (`benchmarks/detection/repro_check.json`). `config_sha256` differs only because `conf_operational` was written to `configs/detector.yaml` after the full run; it does not affect inference. |
| AC3 | ✅ `latency_cpu_torch.json` (run 1) and `latency_cpu_torch_run2.json` (run 2), with machine, threads, imgsz, warm-up, power state. Detector median, run 2: SUN 180 ms, TUM fr1 163 ms, fr3 151 ms (p95 296–370). Run 1 was under memory pressure, cold disk (SUN 228 / p95 625). Both kept. |
| AC4 | ✅ `tum_presence.json`; measured table in `docs/data_quality_report.md`: fr1 572/573, fr3 2484/2485 frames with ≥ 1 detection. |
| AC5 | ✅ 24 overlays in `benchmarks/detection/gallery/`, reviewed; patterns in `docs/detection_baseline.md` (missed dense chairs, armchair→chair vs SUN couch, unlabelled-correct FPs, edge-cut misses, one2one near-duplicates). |

## Tests
- JSONL round trip: write and read back one frame result, and the arrays are equal.
- Resume: a JSONL that already holds frames 1–3 makes the runner start at frame 4.
- The latency statistics helper gives the correct median and p95 on a known array.

## Dependencies
02.1; psutil (for RSS).
