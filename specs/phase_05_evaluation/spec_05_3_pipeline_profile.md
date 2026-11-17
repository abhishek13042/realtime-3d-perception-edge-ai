# Spec 05.3 — End-to-end pipeline and stage profile

| Field | Value |
|---|---|
| Phase | 05 Evaluation |
| Status | 🔧 Approved 2026-11-11; in progress |

## Objective
Assemble the stages built in Phases 02–04 into one `Pipeline` that turns a `Frame` into a `FrameResult` (problem statement §4). Then measure its per-stage latency on CPU with the protocol §5 procedure. This is the PyTorch-CPU baseline that Phase 06 optimises against, and the first honest end-to-end FPS number.

## Requirements
- **R1. `perception/pipeline.py`.**
  - `Pipeline(detector, depth_cfg, tracker_cfg, stride)`.
  - `.process(frame) -> FrameResult` runs validate → detect → depth association → 3D centre → track, and fills `timing_ms` per stage.
  - The stages are the existing functions. No logic is copied: the pipeline only calls them.
  - `FrameResult` serialises to the §4 JSON contract. `track_id` is null for unconfirmed tracks. Fields that are null when there is no depth stay null.
- **R2. Equivalence.**
  - On fr1 (all 573 frames, stride 1), the pipeline's objects equal the offline Phase 04 run (`benchmarks/tracking/tum_fr1_s1.jsonl`): same IDs, boxes and 3D centres.
  - The pipeline runs the detector live; the offline run used stored detections. Any difference is therefore detector non-determinism, and is reported. Phase 02 measured it as deterministic.
- **R3. Profile procedure (protocol §5).**
  - Frames: the fixed 200-frame lists in `configs/benchmark_frames/`: fr1 and fr3 with tracking, and the first 200 of `sunrgbd_test_500.txt` in single-image mode, with no tracking.
  - 20 warm-up iterations, discarded. 3 repeated runs, each in a fresh process.
  - Statistics: median, p95 and mean per stage, and `pipeline` FPS = 1000 / mean.
  - Boundaries exactly as in §5: `detector`, `detect_total`, `pipeline`. `io` (decode) is reported separately and never included in the pipeline FPS.
- **R4. Disclosure.** CPU, RAM, OS, power state before and after, thread counts (torch, OpenBLAS), library versions, peak RSS. The machine must be idle: the script refuses to start while another heavy process from this project is running (e.g. the Kaggle upload), unless `--allow-load` is given. In that case the result is labelled "under load".
- **R5. Reliability (protocol §6).** Per run: failed frames, frames with zero detections, the mean invalid-depth rate, and the recovery fraction (from 04.2).
- **R6. Bottleneck statement.** Report each stage's share of the pipeline median and compare the result with the 10 FPS soft target. If the target is missed, name the dominant stage and the Phase 06 levers. No boundary is moved.

## Tasks
- [x] T1. `perception/pipeline.py`, `FrameResult` (R1).
- [x] T2. Equivalence run on fr1 (R2).
- [x] T3. `scripts/profile_pipeline.py` (R3–R5); `benchmarks/results/pipeline_profile.json`.
- [x] T4. Stage-profile table and stacked-bar chart in `docs/results_summary.md` (05.4) (R6).
- [x] T5. Tests.

## Acceptance criteria
- **AC1.** `FrameResult` matches the §4 contract (schema test), including the null rules.
- **AC2.** The pipeline equals the offline Phase 04 run on fr1 (R2), or any difference is explained.
- **AC3.** The stage profile is reported for 3 runs with median, p95, mean and FPS, and with the run-to-run spread.
- **AC4.** The bottleneck and the 10 FPS comparison are stated.

## Tests (`tests/test_pipeline.py`)
- A synthetic frame with a stub detector produces a valid `FrameResult`; JSON round trip.
- A box with no valid depth gives null `depth_m` and `center_cam_m` with status `no_depth`.
- `timing_ms.total` ≥ the sum of the stage timings.
- With a stub detector, the pipeline output equals the offline runner's on 20 fr1 frames.

## Dependencies
02.1 detector; 03.1, 03.2; 04.1; `configs/benchmark_frames/`.

## Results (2026-11-11, Measured)
- **AC1.** `FrameResult` schema and null rules are covered by `tests/test_pipeline.py` (5 tests, passing).
- **AC2. Equivalence (fr1, stride 1, 573 frames).** Detections are identical on 573 / 573 frames (max box difference 0.0 px), tracks are identical on 573 / 573 frames, and there are 0 schema errors. Source: `benchmarks/results/pipeline_equivalence_fr1.json`. This is a correctness run, not a timing run. The offline reference file is `benchmarks/tracking/tum_fr1_s1.jsonl`; the name `tum_fr1_desk_s1.jsonl` in the earlier draft of this spec was wrong.
- **AC3. Stage profile (Measured: PyTorch CPU, local laptop, 3 runs x 200 frames per list, io excluded).** Source: `benchmarks/results/pipeline_profile.json`. The runs started only once the idle check passed (attempt 5 of the retry loop; CPU 15.2% before, 20.3% after; plugged in; Balanced power plan; `under_load` false).

| List | Pipeline median | p95 | FPS pooled | FPS per run | Detector median | Per-run median range |
|---|---|---|---|---|---|---|
| tum_fr1_desk_200 | 119.4 ms | 212.5 ms | 7.19 | 7.29, 8.07, 6.40 | 103.0 ms | 114.8–139.9 ms |
| tum_fr3_long_office_200 | 121.9 ms | 202.7 ms | 7.27 | 6.76, 8.33, 6.91 | 103.1 ms | 115.5–134.5 ms |
| sunrgbd_test_200 | 125.6 ms | 208.9 ms | 6.97 | 6.56, 7.74, 6.71 | 106.9 ms | 118.4–132.0 ms |

  Run-to-run spread is up to about 20% of the median on this laptop (fr1). The stacked-bar chart of mean stage times is `benchmarks/results/pipeline_stage_profile.png`.
- **AC4. Bottleneck and 10 FPS.** Lowest pipeline FPS is 6.97 against the 10 FPS Target (soft): **missed**. Detection (preprocess + model + postprocess) is 87.5–88.5% of the mean pipeline time per list, and preprocess + detect together reach up to 93%. Depth, geometry and tracking are the rest. The Phase 06 levers, in protocol order: ONNX Runtime FP32, ONNX Runtime INT8, the input-size sweep (640 -> 512 / 416, with the mAP change), and optionally OpenVINO. No measurement boundary was moved.

