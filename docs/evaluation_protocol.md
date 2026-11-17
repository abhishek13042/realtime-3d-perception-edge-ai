# Evaluation Protocol

**Spec:** 00.4, finalized in 05.1 · **Status:** Frozen 2026-11-11 (spec 05.1; sha256 in `benchmarks/results/index.json`) · **Last updated:** 2026-11-11

**Rule:** this protocol is fixed **before** any result is computed. If it changes later, the change and the reason are logged in `PROGRESS.md`, and all affected results are re-run.

## 1. Data splits
- **SUN RGB-D:** the official test split (5,050 images, `configs/splits/sunrgbd_test.txt`) is used for all reported numbers. The official train split (5,285 images) is divided at the scene level into train (4,757) and **val** (528, `configs/splits/sunrgbd_val.txt`). All threshold tuning (detector operating point, depth thresholds, the shrink factor) used val, never test (deviation D1).
- **TUM:** `rgbd_dataset_freiburg1_desk` is used for tuning tracker thresholds, and `rgbd_dataset_freiburg3_long_office_household` for reporting (spec 04.2).
- **Splits are made at the scene level only.** A test asserts there is no scene overlap.

## 2. Detection (G2)
- **Classes:** COCO ↔ SUN RGB-D shared classes (see `dataset_audit.md` §4). Excluded classes are listed in the report.
- **Ground truth:** SUN RGB-D 2D boxes (`SUNRGBDMeta2DBB_v2`). Ground-truth boxes for unmapped classes are ignored, not counted as misses.
- **Metrics:** COCO-style mAP@50, mAP@50–95, and precision and recall at a fixed score threshold (pycocotools). Reported per class and as a mean.
- **Settings recorded:** model, weights hash, input size, confidence threshold, NMS IoU.

## 3. Depth and 3D localization (G3, G4)
- **GT linkage:** each GT 3D box is linked to a GT 2D box of the same class by projecting the 3D box and maximising total IoU (Hungarian), not by the `has3d` order (deviation D2).
- **Matching:** a prediction is matched to the ground-truth object of the same class with the highest 2D IoU, if IoU ≥ 0.5 (greedy, by score).
- **Ground-truth 3D center:** the center of the SUN RGB-D 3D box, converted into the OpenCV camera frame.
- **Metrics, over matched objects with status `valid`:**
  - Depth error `|Ẑ − Z_gt|`: MAE and RMSE.
  - 3D center error `‖p̂ − p_gt‖₂`: median, mean, and p90.
  - Everything is broken down by distance bucket: 0–2, 2–4, and >4 m.
- **Reported separately:**
  - the fraction of matched objects by status (`valid` / `low_coverage` / `no_depth`),
  - the error for `low_coverage` objects. This shows whether the flag is meaningful: their error should be higher.
- **Known bias (reported, not hidden):** the ground-truth center is a *volume* center, while our estimate is on the *visible surface*. We report the signed mean of `Ẑ − Z_gt`. As an analysis only, we also report a "surface-corrected" variant that adds half the GT box depth extent. That variant is never used as the headline number.
- **Sensor-depth honesty:** only raw sensor depth is used. NYU inpainted depth is not used.

## 4. Tracking (G5)
- **With identity labels (ScanNet, if available):** HOTA, IDF1, ID switches, and fragmentation via TrackEval. ScanNet was not used (licence-gated, not downloaded).
- **Without labels (TUM):** the label-free metrics defined in spec 04.2 (deviation D3):
  - stability: per-track 3D RMS of `valid` world-frame centres about their median, reported as the median and p90 over static tracks;
  - fragmentation proxy: confirmed static tracks per object of a manual inventory (`docs/tum_object_inventory.yaml`), with a range;
  - ID-switch proxy: jumps > 0.75 m between consecutive valid centres, per 100 valid pairs, with a sensitivity check at 0.5 and 1.0 m;
  - recovery: resumed gaps / (resumed gaps + losses);
  - camera-motion check: speed at track losses vs overall.
- **Assumption:** the objects used are static. Movable classes (`configs/tracker.yaml` `excluded_static_classes`) are excluded.
- **Report wording:** the TUM metric is called "label-free stability". It is not presented as HOTA.

## 5. Runtime (G6)
- **Hardware disclosure:** CPU model, RAM, OS, power mode (plugged in, high-performance plan), thread count, library versions.
- **Boundaries** (each reported separately):

  | Name | Starts | Ends |
  |---|---|---|
  | `detector` | Preprocessed tensor | Raw model output |
  | `detect_total` | Original image in memory | Boxes in original pixels (pre + infer + post + NMS) |
  | `pipeline` | `Frame` in memory | `FrameResult` (detect + depth + geometry + track) |
  | `io` / `render` | — | Reported separately; **never included in pipeline FPS** |

- **Procedure:** the fixed 200-frame lists in `configs/benchmark_frames/`; 20 warm-up iterations, discarded; 3 repeated runs, each in a fresh process, on an otherwise idle machine. A run made under load is labelled so.
- **Statistics reported:** median, p95, and mean latency, plus FPS = 1000 / mean_ms. We note that mean-based FPS is affected by outliers.
- **Comparison fairness:** identical frames, input size, thresholds, thread count, and machine state for every backend.
- **Optimization trade-off:** for each configuration (PyTorch FP32, ORT FP32, ORT INT8, and optionally OpenVINO), report latency, model file size, peak RSS memory, and the **change in mAP** on the same eval set.
- **TensorRT** (optional, Kaggle only) is reported in a separate table labelled with its GPU model. It is never compared directly with local CPU numbers.

## 6. Reliability
- Per run, we report:
  - failed frames (exceptions),
  - frames with zero detections,
  - the mean invalid-depth rate,
  - the fraction of tracks recovered after 1–N missed frames.

## 7. Result hygiene
- Every result file stores the git commit, the config hash, the dataset split file hash, and a timestamp.
- Results are JSON files under `benchmarks/<area>/` (Phases 01–04: `data_quality`, `detection`, `geometry`, `tracking`) and `benchmarks/results/` (Phase 05 onward). `benchmarks/results/index.json` lists every one with its provenance fields (deviation D4). Tables in docs are **generated** from those files, never typed by hand.
- Until the first git commit, `git_commit` is null and `git_dirty` is true in every file; the config and input sha256 values are the binding provenance.

## 8. Deviations from the 00.4 draft (logged 2026-11-11, spec 05.1)

| ID | Draft said | What was done | Reason |
|---|---|---|---|
| D1 | Tune on a held-out subset of train | Tuned on SUN **val**, a scene-disjoint 528-image split of official train | Same guarantee (test never seen); a fixed named split is simpler to guard (`require_tuning_split`) |
| D2 | GT 2D↔3D pairing implicit (`has3d` order) | Projection + Hungarian IoU linkage | Measured on val: the `has3d` order is not reliable (spec 03.3 R1, `docs/depth_tuning.md`) |
| D3 | TUM: per-track std-dev; fragmentation "if time allows" | The full label-free metric set of spec 04.2 | Fixed in 04.2 before any run; adds jump, recovery and motion checks |
| D4 | All results in `benchmarks/results/` | Per-area folders for Phases 01–04, plus an index | The earlier files already existed; the index gives one place to trace them |
| D5 | Runtime: N ≥ 200 frames, 3 runs | 200-frame fixed lists, fresh process per run, idle-machine rule | Phase 04 measured 4-5x slower timings under concurrent load |

## 9. Frozen inputs for Phase 05 (sha256)

| File | sha256 |
|---|---|
| `configs/depth.yaml` | `9e51c8b6b95c3dce536b1d9daac99ead022789c8fca69cf95a330a2e517db26f` |
| `configs/detector.yaml` | `05c49ce46d9f16153d678b3650c42f6cf36d6429ac4a148bad57c4c579b4e09f` |
| `configs/tracker.yaml` | `8832026648357df7e38285345124f605ee44e4080a1fb287cefaa8ae3b59e818` |
| `configs/class_map.yaml` | `2900a9d8a812d2e044f4c4ab46b6839c2267795c69aa504430bde219dec0ef4b` |
| `configs/splits/sunrgbd_test.txt` | `adadcd59ea80a8252aefaeeb40639d44e9547c8547f4932f8df21f1fa82a6fd4` |
| `configs/splits/sunrgbd_val.txt` | `81c57c936154ab25f25396be9b1fd4a01bd0ac2009b51f810b92d5eea2195416` |

Phase 05 scripts refuse to run when one of these differs.
