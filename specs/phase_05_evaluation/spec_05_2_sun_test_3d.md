# Spec 05.2 — 3D localization on SUN RGB-D test (G3, G4)

| Field | Value |
|---|---|
| Phase | 05 Evaluation |
| Status | ✅ Done 2026-11-11 |

## Objective
Report the headline 3D numbers (G3, G4) on the SUN RGB-D **test** split, once. Use the depth settings frozen on val in 03.3 and the stored YOLO26-n test run from 02.2. The protocol is followed exactly (evaluation protocol §3); nothing is chosen here.

## Requirements
- **R1. Inputs, frozen.**
  - Detections: `benchmarks/detection/sun_test_full.jsonl`, filtered at `conf_operational` 0.2411.
  - Depth and centre settings: `configs/depth.yaml`, frozen in 03.3. Its sha256 is recorded in the 05.1 index at the protocol freeze (03.3 stored the parameters but not a file hash). The script refuses to run if the sha differs.
  - GT and linkage: the 03.3 code (`perception/eval/localization.py`), unchanged.
- **R2. Two modes**, the same as on val:
  - **Oracle boxes:** GT 2D boxes drive depth association. This isolates the depth and geometry stage.
  - **Detected boxes:** the full system. Matching follows protocol §3: same class, IoU ≥ 0.5, greedy by score.
- **R3. Metrics, per mode** (protocol §3):
  - Over `valid` objects: 3D centre error median, mean and p90; depth error MAE and RMSE; and signed mean `Ẑ − Z_gt` (the visible-surface vs volume-centre bias).
  - All of it by distance bucket (0–2, 2–4, > 4 m) and by sensor (kv1, kv2, realsense, xtion), with n in every cell.
  - Status fractions: `valid` / `low_coverage` / `no_depth`. The error of `low_coverage` objects and the low/valid median-error ratio (G4: is the flag meaningful?).
  - Surface-corrected variant: analysis only, never the headline.
  - Bootstrap 95% CIs (1000 resamples, by image, fixed seed) on the oracle and detected medians.
- **R4. G4 completeness.** 100% of output boxes carry a `depth_status`. The script asserts this; a missing status fails the run.
- **R5. Val → test comparison.** Put the val numbers from 03.3 next to the test numbers. A large gap is reported as a finding (possible val overfit, or a sensor mix difference), not fixed.
- **R6. One-shot guard.** The script writes `benchmarks/results/localization_sun_test.json` once. A second run needs `--rerun-reason`, which is logged in the file and in `PROGRESS.md`. A test covers this.
- **R7. Report.** `docs/localization_results.md` is generated from the JSON (protocol §7), with:
  - an error-vs-distance plot (median and p90 per bucket, both modes);
  - an error CDF;
  - the status table and the limitations.

## Tasks
- [x] T1. `scripts/eval_localization.py --split sunrgbd_test` with the freeze checks and the one-shot guard (R1, R6).
- [x] T2. Metrics, breakdowns and bootstrap (R3–R5), reusing `perception/eval/localization.py`.
- [x] T3. Run once; generate the doc (R7).
- [x] T4. Tests.

## Acceptance criteria
- **AC1.** G3: median and p90 3D centre error, by distance bucket, with valid-depth coverage, on test, both modes.
- **AC2.** G4: 100% status coverage asserted, and the low_coverage vs valid error ratio is reported.
- **AC3.** The run used the frozen `depth.yaml`, the frozen detector config and the frozen protocol (shas in the header).
- **AC4.** The val vs test table is present.

## Tests (`tests/test_eval_localization.py`)
- The bootstrap CI contains the true median on a synthetic sample with a known distribution.
- The one-shot guard refuses a second run without `--rerun-reason`.
- The freeze check refuses a modified `depth.yaml`.
- The bucket and sensor breakdown counts sum to the total.

## Dependencies
02.2 stored test run; 03.1–03.3; 05.1 (frozen protocol).

## Results (2026-11-11, Measured, one-shot)
Files:
- `benchmarks/results/localization_sun_test.json` (headline; index row complete) and `_objects.npz` (per-object data for 05.4);
- `docs/localization_results.md` (generated), with the error-vs-distance and CDF plots.

**Code check before the test run.** The same script on `sunrgbd_val` (`localization_sun_val_check.json`) reproduces the 03.3 `tune_val.json` final numbers **exactly** in both modes (n, all statistics, status fractions). The test run therefore uses the same code path as the tuning.

Test run:
- 5,050 images, 144 s, no image without a detection record.
- Freeze checks passed: the protocol sha and the six §9 inputs (depth, detector, class map, split).
- Oracle set: 17,039 linked objects in mapped classes.
- Detected set: 9,024 matched detections. That is 0.53 of the oracle set, out of 18,749 output boxes at conf 0.2411.

| mode | valid fraction | median (95% CI) | p90 (95% CI) | low/valid ratio (95% CI) |
|---|---|---|---|---|
| oracle | 0.82 | **28.8 cm** (28.2–29.4) | **84.3 cm** (81.9–86.9) | 1.35 (1.30–1.42) |
| detected | 0.84 | **24.3 cm** (23.5–25.0) | **64.2 cm** (61.0–67.9) | 1.44 (1.36–1.56) |

- **AC1 (G3):**
  - The error grows with distance. Oracle medians are 22.2 / 31.9 / 38.6 cm and p90 52.8 / 93.7 / 132.2 cm for 0–2 / 2–4 / > 4 m.
  - Valid coverage falls with distance: 0.92 / 0.78 / 0.72.
  - The per-sensor tables are in the doc.
- **AC2 (G4):**
  - 44,812 of 44,812 boxes carry a status (asserted).
  - low_coverage objects have 1.35× (oracle) and 1.44× (detected) the median error of valid ones. The CIs exclude 1, so the flag is meaningful, though the separation is modest.
- **AC3:** the shas of the protocol, depth.yaml, detector.yaml, class_map and the split are in the JSON header. `git_commit` is null because there are no commits yet.
  - The stored run's header `config_sha256` (4d2b70f2…) differs from the frozen `detector.yaml` (05c49ce4…). The fields that affect the stored detections do match field by field: weights sha, head one2one, imgsz 640, conf 0.001, max_det 300.
- **AC4:** the val vs test table is present.

Findings, reported and not fixed:
1. **No val → test gap.**
   - Oracle median: 28.2 cm on val vs 28.8 cm on test. Detected: 24.6 vs 24.3 cm. p90 agrees within 2 cm.
   - The sensor mix differs: kv2 is 49% of valid oracle objects on test and 32% on val. Even so, the overall numbers agree.
2. **Detected < oracle error is a selection effect, not a better box.**
   - Detected objects are easier: 39% are in the 0–2 m bucket, against 33% of the oracle set (computed from the n's in the table).
   - Within every distance bucket the gap is smaller (22.2 vs 19.1, 31.9 vs 27.4, 38.6 vs 32.8 cm). That still favours detected boxes, plausibly because a detector box covers the visible extent.
   - This was not tested further.
3. **Bias.**
   - The signed Z error is −10.9 cm (oracle), meaning the estimate is nearer than the volume centre, as expected. It grows with distance: −1.6 / −15.2 / −22.8 cm.
   - The surface-corrected variant (analysis only) gives +40 cm, so moving the GT half a box depth toward the camera over-corrects. The volume-centre headline stands.
4. **Linkage.** 19% of oracle links have projection IoU < 0.3. Restricting to IoU ≥ 0.3 lowers the oracle median to 25.4 cm, so weak links inflate the headline by at most ~3 cm.
