# Spec 05.4 — Failure gallery, reproducibility, results summary

| Field | Value |
|---|---|
| Phase | 05 Evaluation |
| Status | 🔧 Approved 2026-11-11; in progress |

## Objective
Show where the system fails, with categories counted by eye. Check that the headline results reproduce within a tolerance stated in advance. Collect the G2–G5 headline numbers into one generated summary.

## Requirements
- **R1. Failure gallery (SUN test, detected mode).** Pick these cases by rule, not by eye:
  - the 12 largest 3D errors among `valid` objects;
  - 8 `no_depth` matched objects;
  - 8 missed GT objects (no match at IoU 0.5) from the most frequent classes;
  - 8 high-score false positives.
- **R2. Categories.** Each gallery item gets exactly one category, recorded in `docs/failure_review.yaml` (same pattern as the 03.3 and 04.2 reviews):
  - depth: background bleed, occluder, missing or invalid depth, sensor edge;
  - GT: GT-box problem;
  - detection: localisation, misclassification, small object, truncation.

  Counts per category are reported. The gallery explicitly tests the architecture's open question: does background bleed dominate the large errors? That would justify the instance-mask extension (architecture §3).
- **R3. Reproducibility.** Tolerances are fixed now, before any re-run. `scripts/repro_check.py` re-runs from scratch in a fresh process:
  - 05.2 on a fixed 200-image test subset: **exact** match of the per-object errors (deterministic code; tolerance 1e-9 m);
  - the 04.2 fr3 stride-1 metrics: **exact**;
  - the 02.3 detection mAP: within **±0.001**;
  - the 05.3 pipeline median: within **±15%** of the reported value (timing is load-sensitive; 04.1 measured up to 5× under load).

  Results go to `benchmarks/results/repro_check.json` with pass/fail per item.
- **R4. Results summary.** `docs/results_summary.md` is generated from `benchmarks/results/index.json` (05.1). It contains:
  - one table of G1–G7 status: goal, metric, Measured value, source file, met / not met / pending (G6 and G7 are pending until Phases 06–07);
  - the stage profile (05.3);
  - the failure-category counts;
  - the repro results;
  - the top limitations, linked to `docs/risks_and_limitations.md`.

## Tasks
- [x] T1. Gallery selection and rendering (R1); review by eye (R2).
- [x] T2. `scripts/repro_check.py` (R3). All 4 items run: 3 pass, pipeline_median fails and its cause is reported (AC2 allows this).
- [x] T3. `scripts/make_results_summary.py` (R4).
- [x] T4. Tests.

## Acceptance criteria
- **AC1.** Every gallery item is categorised; counts are reported, and the background-bleed question is answered.
- **AC2.** All R3 items are within tolerance, or the failure is reported with its cause.
- **AC3.** The summary is generated, contains no hand-typed numbers, and every value links to its source file.
- **AC4. Phase 05 exit.** Protocol followed (05.1), 3D error by bucket (05.2), stage profile (05.3), failure gallery and repro (05.4); all tests pass.

## Tests (`tests/test_failures_repro.py`)
- The gallery selection rule is deterministic: same input, same items, in the same order.
- The repro comparator passes on identical inputs, fails just outside each tolerance, and passes just inside it.
- The summary generator refuses a value whose source file is missing from the index.

## Dependencies
05.1–05.3; 02.3; 04.2.

## Results (2026-11-11, Measured)
**R1–R2. Failure gallery.** 34 rule-selected items, all categorised by eye in `docs/failure_review.yaml`. Source: `benchmarks/results/failure_gallery.json` (+ 4 PNGs).

| Category | large_error (12) | no_depth (8) | missed (8) | false_positive (8) | total |
|---|---|---|---|---|---|
| GT-box problem | 8 | 0 | 2 | 7 | 17 |
| missing or invalid depth | 0 | 8 | 0 | 0 | 8 |
| truncation | 0 | 0 | 6 | 0 | 6 |
| occluder | 2 | 0 | 0 | 0 | 2 |
| localisation | 1 | 0 | 0 | 0 | 1 |
| background bleed | 1 | 0 | 0 | 0 | 1 |
| misclassification | 0 | 0 | 0 | 1 | 1 |

- **Does background bleed dominate the large errors?** No: 1 of 12 (8.3%). The instance-mask extension is not justified by this evidence.
- Most of the largest errors (8 of 12) are **GT-box problems**: the protocol's Hungarian linkage (no minimum IoU, D2) pairs the box with the wrong GT 3D box (link IoU < 0.3). 05.2 already quantifies this as diagnostic only: 14.8% of detected objects have link IoU < 0.3, and without them the median is 0.217 m (headline 0.243 m). The frozen protocol is unchanged; this is reported as a limitation.
- Missed objects are mostly truncated at the image edge. Most high-score false positives are real objects without a SUN label (person and bicycle are not in the class map).

**R3. Reproducibility** (`benchmarks/results/repro_check.json`; tolerances fixed in `perception/eval/repro.py` before any re-run):
- sun_test_200: **pass**. 925 objects, max abs difference 0.0 m; rows, statuses and output boxes identical.
- fr3_s1_metrics: **pass**. 0 differing values.
- detection_map: **pass**. Fresh detector re-run on all 5,050 test images (1,069 s, CPU shared with an upload). mAP 0.22791 vs reported 0.22791 (diff 0.0), AP50 0.32217 vs 0.32217; weights sha, head, imgsz and conf all match.
- pipeline_median: **FAIL** (tolerance ±15% relative, unchanged). Two attempts, both kept in the file (`history`):
  - attempt 1 (CPU 39.6% at the start, not idle): +21.6% fr1, +47.5% fr3, +14.0% SUN;
  - attempt 2 (idle at the start, CPU 20%): fr1 156.9 vs 119.4 ms (+31.4%), fr3 167.6 vs 121.9 ms (+37.5%), SUN 134.5 vs 125.6 ms (+7.1%).
  - **Measured cause:** one fresh run is compared against a ±15% band, but this laptop's own run-to-run spread is about that wide: within the 05.3 profile the per-run medians already span fr1 114.8–139.9, fr3 115.5–134.5 and SUN 118.4–132.0 ms. The fresh runs came in slower still. Thermal throttling or background load may contribute but was not measured, so it is not claimed. The correctness items (outputs and mAP) all reproduce exactly; only wall-clock time does not. No further attempts are made: retrying until one passes would be cherry-picking.
  - `scripts/repro_check.py` now refuses to time pipeline_median when the machine is not idle (`not_idle`, no pass/fail), and keeps an earlier result in `history` before overwriting it.

**R4. Summary.** `docs/results_summary.md` is generated by `scripts/make_results_summary.py` after `scripts/results_index.py`. Every value is read through `perception.eval.summary.Sources`, which refuses files not in the index or changed since it was written. G1–G5 are met (G1 by tests); the speed goal is **not met** (6.97 FPS vs 10 FPS Target); G6 and G7 are pending until Phases 06–07.

**Tests.** `tests/test_failures_repro.py`: 11 tests (deterministic and row-order-invariant selection, the tolerance edges for every comparator, the source guard). Full suite: 174 passed.

**AC4 (phase exit).** Protocol followed and frozen (05.1); 3D error by bucket (05.2); stage profile (05.3); failure gallery and all four repro items (05.4), with pipeline_median reported as a failure with its cause per AC2. Full test suite passes (see PROGRESS.md for the count).
