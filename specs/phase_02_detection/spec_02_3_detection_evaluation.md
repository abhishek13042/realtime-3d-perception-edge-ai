# Spec 02.3 — Detection Evaluation on COCO-Overlap Classes (G2)

| Field | Value |
|---|---|
| Phase | 02 — Detection |
| Status | ✅ Done 2026-11-10 |
| Depends on | 02.2, 01.3 (splits, 2D boxes), 01.4 (`configs/class_map.yaml`) |
| Blocks | 05.x (end-to-end evaluation), 06.x (mAP change after optimization) |

## Objective
Measure how well a COCO-pretrained detector, with no fine-tuning, finds SUN RGB-D objects on the classes both label sets share. Report the result per class, per sensor, and per object size, with the caveats that make the numbers interpretable.

## Known validity issues (to be stated next to every number)
1. **Incomplete GT.** SUN 2D labels were not made for COCO. A correct detection of an object nobody labelled (for example a cup) counts as a false positive, so precision and AP are **lower bounds** for those classes.
2. **Class mismatch.** The mapping is not exact. SUN `table` → COCO `dining table` includes coffee and side tables. SUN `desk` is unmapped, but YOLO often calls a desk a `dining table`.
3. **Domain shift.** COCO is mostly internet photos, while SUN images are indoor shots from 4 depth sensors with different image quality.

## Requirements
- **R1. GT conversion.** SUN 2D boxes (0-based xyxy) are mapped with `configs/class_map.yaml` into COCO-format JSON per split. Unmapped GT classes are dropped. Boxes with zero area or outside the image are dropped and counted.
- **R2. Prediction conversion.** Detections from 02.2, kept only when their COCO class is in the mapped set. Evaluation uses threshold 0.001 and `max_det` 100 (COCO standard).
- **R3. Metrics.** We use pycocotools `COCOeval` (bbox) to compute:
  - mAP@50–95, AP50, and AP75;
  - AR@100;
  - AP by size (small / medium / large);
  - per-class AP50 and AP50–95, with the GT instance count.
- **R4. Breakdowns.** Per sensor (kv1 / kv2 / realsense / xtion), and per class. Classes with fewer than 50 test GT instances are marked **low-n**.
- **R5. Sensitivity row.** Re-run `dining table` with SUN `desk` GT boxes treated as an *ignore region* (`iscrowd`-style), so a prediction there is neither a TP nor an FP. This shows how much of the error comes from the class mismatch. The strict number stays the headline.
- **R6. Operational threshold.** On **val**, at IoU 0.5 over the mapped classes, choose the global confidence threshold that maximizes F1. Write it to `configs/detector.yaml` as `conf_operational`, and report precision and recall at that threshold on **test**.
- **R7. Evaluator validation.** We check the evaluator before trusting it:
  - GT used as predictions gives AP = 1.0;
  - an empty prediction set gives 0;
  - an exact hand-computed AP on a toy case.
- **R8. Report.** `docs/detection_baseline.md` is generated from `benchmarks/detection/eval_*.json`, in the same "do not edit by hand" pattern as the data quality report.

## Tasks
- [x] T1. `perception/eval/detection.py`: GT and prediction conversion, a COCOeval wrapper, per-class and per-sensor slicing, and threshold selection.
- [x] T2. Evaluator validation tests (R7).
- [x] T3. `scripts/eval_detection.py`: val first (threshold), then test.
- [x] T4. Report generator and a review of the results.

## Acceptance criteria
- **AC1. G2 met.** A test-split table with mAP@50–95, AP50, precision, and recall at the operational threshold, on the shared classes. It also includes the per-class table with instance counts and the per-sensor table.
- **AC2.** The evaluator validation tests pass (R7).
- **AC3.** The threshold was chosen on val only. The report states the val F1 curve source and the chosen value.
- **AC4.** The report states the three validity issues next to the headline table, and includes the R5 sensitivity row.
- **AC5. Phase 02 exit.** All tests pass; 02.1 AC1–AC2 hold (letterbox ≤ 1 px, deterministic); this baseline table exists. **No target mAP is set**, because the baseline is a measurement. The only expectation is that AP50 for `chair` is clearly above 0, as a sanity check.

## Results (measured 2026-11-10; full report `docs/detection_baseline.md`)
| AC | Result |
|---|---|
| AC1 | ✅ Test, 5,050 images, 20,542 GT, 23 classes with GT: mAP 0.228, AP50 0.322, P 0.544 / R 0.485 / F1 0.513 at conf 0.2411. Per-class (low-n flag < 50) and per-sensor tables in the report. |
| AC2 | ✅ `tests/test_eval_detection.py` 9/9 (suite 82/82). |
| AC3 | ✅ conf_operational 0.2411 = max F1 on val (P 0.527, R 0.459, F1 0.491); source `eval_baseline.json` → `val`; written to `configs/detector.yaml`. |
| AC4 | ✅ Three validity issues stated above the headline table; desk-as-ignore row: dining table AP50 0.373 → 0.424, overall mAP 0.228 → 0.229. |
| AC5 | ✅ 82/82 tests; 02.1 AC1–AC2 hold; chair AP50 0.587 > 0. |

Supplementary ablation (`eval_nms_ablation.json`): one2many+NMS gives test mAP 0.226 / AP50 0.320 / AR100 0.562, F1 0.515 at its own val threshold 0.2338, vs one2one 0.228 / 0.322 / 0.615 / 0.513. one2one stays the default.

Finding for the user (not acted on): SUN `sofa_chair` is predicted as `chair` 404/765 vs `couch` 155/765 on test. Changing the class map would need justification on val, not test. **Decision (2026-11-11): keep `sofa_chair` → couch.** The mapping was fixed before any results. The confusion was found on test, and changing the map after seeing test numbers would be tuning on test. It is reported as a known limitation of the SUN→COCO mapping.

## Tests (`tests/test_eval_detection.py`)
- Perfect predictions → AP 1.0.
- No predictions → AP 0.
- A toy case with 2 GT and 3 ranked predictions gives the hand-computed AP50.
- Class mapping: an unmapped SUN class produces no GT, and a mapped synonym (`sofa_chair` → couch) is merged.
- The F1 threshold picker selects the known optimum on a synthetic score/label set.

## Dependencies
pycocotools, numpy, PyYAML; 02.2 outputs.
