# Calibration and Alignment Notes (spec 01.2)

## TUM intrinsics choice (AC4)
**Decision:** use the per-camera calibrated intrinsics from the TUM dataset page (`intrinsics: calibrated` in [configs/dataset.yaml](../configs/dataset.yaml)), chosen by sequence prefix:

| Camera | fx | fy | cx | cy |
|---|---|---|---|---|
| freiburg1 | 517.3 | 516.5 | 318.6 | 255.3 |
| freiburg2 | 520.9 | 521.0 | 325.1 | 249.7 |
| freiburg3 | 535.4 | 539.2 | 320.1 | 247.6 |

**Why not the ROS default (525, 525, 319.5, 239.5):** the default is a generic Kinect value. For fr1 its principal point is ~16 px off in *y* from the calibrated one, which shifts every back-projected point by about 3% of depth vertically. The ROS default stays available (`intrinsics: ros_default`) for comparison.

**Depth registration:** TUM depth is already registered to the RGB camera, so the RGB intrinsics are used for depth too.

## fr1/fr2 lens distortion: measured, and now corrected (spec 03.2 R6, 2026-11-10)
fr3 images are published undistorted. fr1 and fr2 are **not**: TUM lists non-zero radial distortion for them
(fr1 OpenCV k1, k2, p1, p2, k3 = 0.2624, -0.9531, -0.0054, 0.0026, 1.1633).

**Measurement** (`scripts/fr1_distortion.py` -> `benchmarks/geometry/fr1_distortion.json`). These figures are measured without
ground truth, so they show the effect of distortion, not error against GT. For every fr1_desk detection at the operational
threshold, each object centre was computed twice from the same inlier pixels: once with raw pixels and once with
undistorted pixels. Sample: 572 frames, 2,533 boxes.

| Box-centre radius | p90 displacement |
|---|---|
| 0-100 px | 0.19 cm |
| 100-200 px | 1.29 cm |
| 200-300 px | 2.53 cm |
| >300 px | 3.44 cm |
| **all** | **2.10 cm** (median 0.86 cm, max 6.15 cm; p90 = 1.5 % of depth) |

**Decision**, from the rule fixed before measuring: p90 2.10 cm > 2 cm, so **undistort fr1/fr2**.
- `configs/dataset.yaml` sets `tum.undistort: true`. `TUMSequence` then attaches `Frame.dist`.
- `unproject_pixels` maps each pixel through a cached per-image look-up table of `cv2.undistortPoints` before
  computing `X = x_n * Z`. Images are not resampled, so detection and depth association still see raw pixels.
- Measured cost: none visible in `scripts/time_geometry.py` (association + centres median 3.4 ms per frame either way).

**Still open:** drawing 3D results back onto the image (overlays) must project *with* distortion
(`cv2.projectPoints`). This is not needed until Phase 07.
Depth-to-colour registration is TUM's own and is not affected by this.

## Alignment check results (measured 2026-11-08, `scripts/check_alignment.py --dataset tum --n 10`)

| Sequence | RGB | Depth | Associated (≤ 20 ms) | With pose | Median depth in 0.5–6 m |
|---|---|---|---|---|---|
| fr1_desk | 613 | 595 | 573 | 573 | 10/10 frames (0.58–1.66 m) |
| fr3_long_office_household | 2,585 | 2,509 | 2,488 | 2,485 | 10/10 frames (1.04–2.41 m) |

Valid-depth fraction per frame: 0.70–0.83 (fr1), 0.80–0.86 (fr3).

**Visual review (AC2):** depth discontinuities sit on object boundaries (globe, box, dice, pole in fr3; cereal box, books, desk edge in fr1). No systematic offset was seen. The overlays are in `benchmarks/data_quality/alignment/`; raw numbers are in `benchmarks/data_quality/alignment_tum.json`.

**Observations:**
- A ~30–40 px strip at the right and top edges has no depth. This is the registration border, where the depth sensor's view does not cover the RGB view.
- Far walls (> 4 m) produce speckled "edges", because depth noise there exceeds the fixed 10 cm jump threshold. That is noise, not misalignment.
- The automatic edge-agreement score (0.17–0.85) is **diagnostic only**: Canny also fires on texture such as book covers and wood grain, and a frame with few depth edges scores low regardless of alignment. A shift search based on it gave inconsistent answers (−6 to +6 px with no pattern), so it was removed and alignment is judged visually.

## SUN RGB-D (measured 2026-11-08, `scripts/check_alignment.py --dataset sun --n 10`)

**Source of truth:** `SUNRGBDMeta3DBB_v2.mat`, which has 10,335 records, each with its own `K` and `Rtilt`. Image paths are the stored `rgbpath`/`depthpath` with the `/n/fs/sun3d/data/` prefix removed. All 10,335 `image/` JPGs are present on disk.

**Depth folder:** we use raw `depth/`, not `depth_bfx/`. The `bfx` version is hole-filled, so it would invent depth where the sensor saw nothing and inflate coverage.

The check used 200 frames, sampled in proportion to each sensor with seed 0, plus 10 overlays.

| Sensor | Records | Image size (H×W) | Median depth in 0.5–6 m | Median of medians | Median valid fraction |
|---|---|---|---|---|---|
| kv1 | 2,003 | 427×561 | 100% | 2.32 m | 0.95 |
| kv2 | 3,784 | 530×730 | 100% | 2.15 m | 0.79 |
| realsense | 1,159 | 531×681 | 100% | 2.00 m | 0.68 |
| xtion | 3,389 | 441×591 | 98% | 1.99 m | 0.99 |
| **All** | **10,335** | | **99.5%** (AC3 needs ≥ 95%) | | |

**Visual review (AC2):** depth edges lie on object boundaries for all four sensors. No offset was seen.

**Sensor differences that matter later:**
- kv2 drops pixels along occlusion edges and on dark or distant surfaces.
- RealSense depth is the noisiest.
- Xtion and kv1 are nearly hole-free (median valid fraction 0.98 and 0.94 over all images; see docs/data_quality_report.md). Whether this raw depth was already processed upstream is **unresolved**. Comparing it with `depth_bfx/` cannot answer the question, because `depth_bfx` differs from raw depth in ~93% of pixels, including valid ones. So `depth_bfx` is not "raw plus filled holes", and raw = bfx in only 1–5% of frames.

Per-detection depth statistics (Phase 03) should therefore be reported **per sensor**.
