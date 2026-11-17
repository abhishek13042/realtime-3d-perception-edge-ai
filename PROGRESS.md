# Project Progress Log

**Project:** Real-Time 3D Perception and Edge Intelligence Platform
**Current phase:** 05 — Evaluation (specs approved 2026-11-11; 05.1–05.4 done, awaiting your approval; repro pipeline_median failed, cause reported). Phase 04 approved 2026-11-11.
**Last updated:** 2026-11-11

## Spec status

| Spec | Title | Doc | Status |
|---|---|---|---|
| 00.1 | Problem statement and goals | [docs/problem_statement.md](docs/problem_statement.md) | ✅ Accepted 2026-11-07 |
| 00.2 | Literature review | [docs/literature_review.md](docs/literature_review.md) | ✅ Accepted; user reading notes still pending |
| 00.3 | Dataset and license audit | [docs/dataset_audit.md](docs/dataset_audit.md) | ✅ Accepted; [To verify] items moved to Phase 01 |
| 00.4 | Architecture, risks, compute | [docs/architecture.md](docs/architecture.md), [docs/evaluation_protocol.md](docs/evaluation_protocol.md), [docs/risks_and_limitations.md](docs/risks_and_limitations.md) | ✅ Accepted 2026-11-07 |
| 01.1 | Dataset downloader | [spec](specs/phase_01_data/spec_01_1_downloader.md), [docs/data_layout.md](docs/data_layout.md) | ✅ Accepted 2026-11-09. Kaggle mirrors moved to account `abhishekgupta0220` on 2026-11-10 (see Next actions) |
| 01.2 | Calibration and alignment | [spec](specs/phase_01_data/spec_01_2_calibration_alignment.md), [docs/calibration.md](docs/calibration.md) | ✅ Accepted 2026-11-09 |
| 01.3 | Annotations and splits | [spec](specs/phase_01_data/spec_01_3_annotations_splits.md), [docs/annotations.md](docs/annotations.md) | ✅ Accepted 2026-11-09 (stricter AC2 approved and met) |
| 01.4 | Data quality report | [spec](specs/phase_01_data/spec_01_4_quality_report.md), [docs/data_quality_report.md](docs/data_quality_report.md) | ✅ Accepted 2026-11-09 |
| 02.1 | Detector interface, letterbox, PyTorch backend | [spec](specs/phase_02_detection/spec_02_1_detector_interface.md) | ✅ Done 2026-11-10: parity with Ultralytics (IoU ≥ 0.99999, \|Δscore\| ≤ 1.2e-4), deterministic across processes |
| 02.2 | Baseline inference runs and latency | [spec](specs/phase_02_detection/spec_02_2_baseline_inference.md) | ✅ Done 2026-11-10: 4 full runs complete, 50-frame re-run exact, CPU detector median 151–180 ms (run 2) |
| 02.3 | Detection evaluation (G2) | [spec](specs/phase_02_detection/spec_02_3_detection_evaluation.md), [docs/detection_baseline.md](docs/detection_baseline.md) | ✅ Done 2026-11-10: SUN test mAP 0.228, AP50 0.322; P/R 0.544/0.485 @ conf 0.2411 (val-chosen) |
| 03.1–03.3 | Depth association, 3D centres & point clouds, depth parameters on val | [specs/phase_03_geometry/](specs/phase_03_geometry/README.md) | ✅ Done 2026-11-11 (val only). Frozen: estimator A, shrink 0.7, k 3.0, n_min 20, tau_cov 0.3, tau_spread 0.5 m. Oracle valid median 3D centre error 28.2 cm (valid 0.81), detected 24.6 cm; low_coverage/valid ratio 1.45; see [docs/depth_tuning.md](docs/depth_tuning.md). R5 ran twice by review decision; the cost bound was restated to real frames + per-box cost |
| 04.1–04.2 | Tracking (ByteTrack + 3D centres), label-free evaluation on TUM | [specs/phase_04_tracking/](specs/phase_04_tracking/README.md), [docs/tracking_results.md](docs/tracking_results.md) | ✅ Approved 2026-11-11. Tuned on fr1 only, frozen per stride (s1: 60/0.9, s6: 15/0.7, class-aware). fr3 (one-shot): stability median 2.9 / 4.3 cm, fragmentation 5.0 / 3.16 tracks per object, jumps > 0.75 m 0.14 / 0.29 per 100, recovery 0.69 / 0.47 (stride 1 / 6). Motion trigger does not fire. R8 met on fr1 (1.70 ms isolated), missed on fr3 (2.4–2.6 ms) |
| 05.1–05.4 | Protocol freeze, SUN test 3D (G3/G4), pipeline profile, failures + repro | [specs/phase_05_evaluation/](specs/phase_05_evaluation/README.md) | ✅ Done 2026-11-11, awaiting approval. SUN test oracle median 28.8 cm, detected 24.3 cm; pipeline 6.97–7.27 FPS (10 FPS Target missed, detection ≈88% of mean time); repro 3/4 pass, pipeline_median fails (laptop timing spread) |
| 06.x – 07.x | — | — | ⬜ Not started |

Legend: ⬜ not started · 🔧 in progress · 📝 drafted / awaiting review · ✅ accepted (all acceptance criteria pass)

## Data inventory (measured)

| Dataset | Local | Content | Kaggle (private) |
|---|---|---|---|
| TUM fr1_desk | ✅ | 573 associated RGB-D frames with pose, 20 s | ✅ `abhishekgupta0220/tum-rgbd-p5` (private, verified 2026-11-11) |
| TUM fr3_long_office_household | ✅ | 2,485 associated frames with pose, 87 s | 🔧 same dataset |
| SUN RGB-D | ✅ | 10,335 RGB-D images (kv1 2,003 · kv2 3,784 · realsense 1,159 · xtion 3,389) + 3D/2D metadata + toolbox | 🔧 `abhishekgupta0220/sunrgbd-p5`: all 4 files uploaded 2026-11-11, created as private; Kaggle still processing, listing not yet verified |

## Decisions pending (user)

| ID | Decision | Status |
|---|---|---|
| D2 | ScanNet access | User's action; outcome unknown. Optional; the MVP does not need it |

## Decisions made

| Date | Decision | Rationale |
|---|---|---|
| 2026-11-07 | D1: SUN RGB-D (primary) + TUM RGB-D (sequences) | Both open; cover detection, depth, and sequences |
| 2026-11-07 | D3: Ultralytics YOLO26-n (AGPL-3.0) | Best maintained export path; the repo is AGPL |
| 2026-11-07 | D4: CPU / Intel iGPU locally; TensorRT on Kaggle only, labelled separately | Matches the hardware |
| 2026-11-07 | D5: Offline only | No RGB-D sensor |
| 2026-11-07 | D6: Python 3.14 venv (3.12 fallback); git initialised | Core wheels verified on PyPI |
| 2026-11-07 | Kaggle mirrors are private (both datasets) | SUN RGB-D has no redistribution licence; private is all the GPU phases need |
| 2026-11-08 | TUM uses calibrated per-camera intrinsics, not the ROS default | fr1 principal point differs by ~16 px; see docs/calibration.md |
| 2026-11-08 | SUN RGB-D uses raw `depth/`, not hole-filled `depth_bfx/` | Honest valid-depth fraction |
| 2026-11-08 | Alignment is judged visually; the edge-agreement score is diagnostic only | The automatic score is confounded by texture |
| 2026-11-09 | Phase 01 accepted. Spec 01.3 AC2 replaced: median depth error ≤ 0.3 × median box extent **and** lower than every wrong-transform control | The old AC2 also passed with wrong transforms |
| 2026-11-09 | `configs/class_map.yaml` approved as the Phase 02 class mapping (desk unmapped) | Only unambiguous COCO counterparts |
| 2026-10-21 | No local training | No NVIDIA GPU; i3 CPU |
| 2026-10-21 | MVP must not depend on gated data | ScanNet access is uncertain |
| 2026-11-07 | Camera frame = OpenCV convention; depth = planar Z in m | One convention prevents silent axis or unit bugs |
| 2026-11-07 | Track in 2D, attach 3D | Robust to missing depth; switch only if measured camera-motion ID switches |
| 2026-11-11 | 2D association kept (spec 04.2 R5) | Camera-motion trigger did not fire on fr3 at either stride |
| 2026-11-11 | `configs/tracker.yaml` per-stride values FROZEN (tuned on fr1 only) | Any change must be logged here and the fr3 report re-run in full (R7) |
| 2026-11-11 | Protocol deviation D1: depth tuning on SUN val (528 scene-disjoint images), not a held-out train subset | Same guarantee (test never seen); a named split is guarded in code |
| 2026-11-11 | Protocol deviation D2: GT 2D-3D linkage by projection + Hungarian IoU, not `has3d` order | `has3d` order measured unreliable on val (spec 03.3 R1) |
| 2026-11-11 | Protocol deviation D3: TUM tracking metrics = spec 04.2 label-free set | Fixed before any run; richer than the draft's std-dev only |
| 2026-11-11 | Protocol deviation D4: Phase 01-04 results stay in `benchmarks/<area>/`; Phase 05 in `benchmarks/results/`; one index covers both | Earlier files existed; `benchmarks/results/index.json` traces all |
| 2026-11-11 | Protocol deviation D5: runtime = fixed 200-frame lists, 20 warm-up, 3 fresh-process runs, idle machine | Phase 04 timings were 4-5x slower under concurrent load |
| 2026-11-11 | `docs/evaluation_protocol.md` FROZEN (sha256 58e3b4f7b07e…, spec 05.1) | Changes after this follow the protocol's own rule and are logged here |

## Issues found and fixed
- 2026-11-08: the downloader truncated the partial file if the server ignored Range. It now keeps the file and retries. The SUN download then survived 6 connection drops.
- 2026-11-08: the first Kaggle upload failed. TUM got `400 Bad Request` on create, and SUN hit a connection reset mid-upload; neither dataset was created. The metadata now uses the same fields as earlier accepted uploads on this account (`subtitle` and `isPrivate` removed, SUN licence `other`), and the upload was re-run. Datasets are private because the CLI was not given `--public`.
- 2026-11-08: `.gitignore` rule `data/` also ignored the code package `src/perception/data/`. It is now anchored to the repo root (`/data/`).

- 2026-11-08: spec 01.3 AC2 ("median depth error < median box extent") is too weak: the wrong-tilt controls also pass it. The correct transform clearly wins (0.19 m error and 92.5% of boxes within extent, against ≥ 0.46 m and 66% for the controls). A stricter AC2 is proposed in docs/annotations.md.

## Test status
175 tests pass (`pytest`, 2026-11-11; Phase 05 added 6 results-index, 6 SUN-test eval, 5 pipeline and 12 failures/repro tests). Earlier: 146 tests passed (`pytest`, 2026-11-11; 18 tracker + 9 tracking-eval added in Phase 04). The 2026-11-08 set of 31 covered download safety, depth decode, camera geometry, TUM association and pose interpolation, the Frame contract, SUN paths, upright→camera transform and box corners, the split builder, quality statistics, and the sampler.

## Benchmark results
**SUN RGB-D test 3D localization (Measured, one-shot, spec 05.2):** oracle median 3D centre error 28.8 cm (95% CI 28.2–29.4), p90 84.3 cm; detected 24.3 cm (23.5–25.0), p90 64.2 cm; valid fraction 0.82 / 0.84; low/valid error ratio 1.35 / 1.44; 100% status coverage. Matches val (28.2 / 24.6 cm). [docs/localization_results.md](docs/localization_results.md).
Data checks: [docs/calibration.md](docs/calibration.md). 3D localization on SUN **val** (Measured, used for tuning, not a headline): [docs/depth_tuning.md](docs/depth_tuning.md). SUN test is untouched: the tuning script refuses any split but `sunrgbd_val`; test numbers come in Phase 05 with configs/depth.yaml frozen. Tracking on TUM (label-free proxies, Measured): [docs/tracking_results.md](docs/tracking_results.md).

## Known limitations
See [docs/risks_and_limitations.md](docs/risks_and_limitations.md). fr1/fr2 lens distortion is now corrected when back-projecting (spec 03.2 R6, p90 effect 2.10 cm). Projecting results back onto images with distortion is still open (Phase 07).

## Next actions
1. Phase 05 done, awaiting your approval. Summary: [docs/results_summary.md](docs/results_summary.md) (generated from the results index).
   - 05.3 profile (Measured, PyTorch CPU, this laptop, 3 x 200 frames per list, idle check passed on attempt 5): pipeline median 119.4 / 121.9 / 125.6 ms (fr1 / fr3 / SUN), FPS 7.19 / 7.27 / 6.97. The 10 FPS Target is **missed**; detection is 87.5–88.5% of mean pipeline time (preprocess + detect up to 93%). Chart: `benchmarks/results/pipeline_stage_profile.png`.
   - 05.4 repro: sun_test_200, fr3_s1_metrics and detection_map (mAP 0.22791) reproduce exactly. pipeline_median **fails** the ±15% band in both attempts (latest: +31.4% fr1, +37.5% fr3, +7.1% SUN). Measured cause: this laptop's run-to-run spread is about as wide as the band (per-run medians 114.8–139.9 ms within the profile itself). The tolerance is unchanged and no retry-until-pass. `scripts/repro_check.py` now refuses to time when the machine is not idle and keeps earlier attempts in `history`.
   - Failure gallery: background bleed 1 of 12 large errors (does not dominate); GT-box linkage problems 8 of 12.
2. Git hygiene before the first commit (your call): `benchmarks/detection/*.jsonl` total about 245 MB and are **not** git-ignored; the tracking `*.jsonl` files are up to 4.7 MB each. No commit has been made.
3. R7 log, 2026-11-11: after the fr1 tuning, the `configs/tracker.yaml` comments were edited to mark the freeze, with no parameter change. As a result, the sha in `eval_fr1_tune.json` differs from the sha of all tracking runs and reports, which were produced after the edit.
4. Kaggle mirrors (account `abhishekgupta0220`, private): TUM verified 2026-11-11 (fr1 desk 344 MB, fr3 long office 1.48 GB, isPrivate true). SUN re-upload finished 2026-11-11 14:27 (exit 0): SUNRGBD.zip 6.9 GB, both metadata .mat files and SUNRGBDtoolbox.zip all uploaded, and the CLI reported "Your private Dataset is being created". Still to do: at 15:00 (33 min after upload) it is not yet in `kaggle datasets list --mine` and the status API returns 403, so Kaggle is presumably still processing; confirm later, and check that isPrivate is true.
