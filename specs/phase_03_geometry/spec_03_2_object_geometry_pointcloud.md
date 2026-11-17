# Spec 03.2 — Object 3D Centres, Point Clouds, and Viewer

| Field | Value |
|---|---|
| Phase | 03 — Geometry |
| Status | ✅ Done 2026-11-10 (AC1–AC4 met; AC5: real frames met, the 50-box bound not met, bound restated 2026-11-11, see 03.1 R9) |
| Depends on | 01.2 (`geometry/camera.py`, intrinsics), 03.1 |
| Blocks | 03.3, 04.x (world-frame stability), 05.x, 07.x (3D view in the demo) |

## Objective
Turn a depth estimate into a 3D object centre in the OpenCV camera frame (metres), build frame and per-object point clouds, and provide a viewer. All maths reuses `geometry/camera.py` (`unproject`, `unproject_depth_map`) from Phase 01.

## Requirements
- **R1. Object centre (architecture §3 [4]).** Take the valid pixels of the 03.1 region whose depth is within `±k · max(spread, σ_floor)` of the estimate `z` (k default 2.0, σ_floor 2 cm, both tuned in 03.3). Back-project them with K and return the **per-axis median** as the centre. Status `no_depth` → no centre (NaN), never a guessed one.
- **R2. Inlier record.** Return the inlier count and the centre's 3D spread (per-axis 1.4826·MAD), so later phases can weight or flag centres.
- **R3. World frame.** A helper maps camera-frame centres to world with `T_world_cam` (TUM ground-truth poses); identity convention documented. Used by Phase 04's label-free stability metric.
- **R4. Point clouds.** (a) Frame cloud from valid pixels with RGB colours, with an optional stride; (b) per-object cloud from the R1 inliers; (c) voxel downsampling implemented in numpy (voxel centroid), so core code does not need open3d; (d) `.ply` export (ASCII, xyz + rgb).
- **R5. Viewer.** `scripts/view_pointcloud.py --sun <id> | --tum <seq> <frame>`: shows the frame cloud, object centres as spheres, and (SUN) the GT 3D box wireframes from `Box3D.corners_cam`. Uses open3d if available; otherwise writes `.ply` files and a matplotlib 3D scatter PNG. It must also save a screenshot/PNG to `benchmarks/geometry/viewer/` for the record.
- **R6. TUM fr1 distortion (docs/calibration.md).** fr1 images are distorted (k1 ≈ 0.26) and we currently ignore it. Measure, without GT: for every fr1 detection (stored run, conf_operational), compute the centre once with raw pixels and once with undistorted pixel coordinates (`cv2.undistortPoints`), and report the distribution of the 3D displacement (median, p90, max) by image radius. **Decision rule:** if the p90 displacement is > 2 cm, enable undistortion for fr1 in the pipeline; otherwise document it as a known small bias. The rule is fixed now, before the measurement.
- **R7. Cost.** Centre computation for ≤ 50 boxes < 5 ms per frame (measured as in 03.1 R9). Frame cloud at stride 2 < 30 ms.

## Tasks
- [x] T1. `perception/geometry/object3d.py`: centre, inliers, world mapping.
- [x] T2. `perception/geometry/pointcloud.py`: frame cloud, object cloud, voxel downsample, PLY writer.
- [x] T3. Install open3d in the venv (pin the version in `pyproject.toml` as an optional `viz` extra); `scripts/view_pointcloud.py` with fallback.
- [x] T4. Synthetic and unit tests (below).
- [x] T5. fr1 distortion analysis → `benchmarks/geometry/fr1_distortion.json` and a section in `docs/calibration.md`.
- [x] T6. Viewer screenshots: 3 SUN (one per depth regime: near, mid, far) and 2 TUM frames, reviewed by eye for alignment of cloud, centres, and GT boxes.

## Acceptance criteria
- **AC1. Synthetic tests pass:** project → unproject round trip < 1e-6 m; a tilted plane depth map is recovered (normal within 0.1°, offset within 1 mm) by a least-squares fit to the frame cloud.
- **AC2.** Unit/invalid-depth tests pass (below).
- **AC3. Viewer works:** the 5 screenshots exist, and on the SUN ones the object centres lie inside or on the visible face of the matching GT 3D boxes (checked by eye, written down per image).
- **AC4.** R6 measured, and the decision applied as per the fixed rule.
- **AC5.** R7 timings measured with machine labels.

## Tests (`tests/test_object3d.py`, `tests/test_pointcloud.py`)
- Synthetic box object at known (X, Y, Z) on a far background → centre within 1 mm of the known front-surface centre.
- `no_depth` → NaN centre; inlier band excludes background pixels.
- World mapping with a known `T_world_cam` (rotation + translation) and its inverse round trip.
- Frame cloud: point count equals valid count (stride 1); invalid pixels never appear; colours align with pixels.
- Voxel downsample: points in one voxel collapse to their centroid; empty input → empty output.
- PLY writer → reader round trip (our own minimal reader, or open3d if present).

## Dependencies
numpy, opencv (undistortPoints); open3d optional (`viz` extra); matplotlib fallback.

## Results (measured 2026-11-10, local laptop: i3-1115G4 2C/4T, 7.7 GB, Windows 11, Python 3.14, numpy 2.5.4, OpenBLAS 4 threads)

**AC1, AC2: met.** The unit tests are in `tests/test_object3d.py`; point-cloud tests are there too rather than in a
separate `test_pointcloud.py`. The undistortion tests are in `tests/test_geometry.py`. Full suite: 106 passed.
- Round trip error < 1e-9 m. On a tilted plane, the normal is recovered within 0.1° and the offset within 1 mm.
- The centre of a synthetic box object is within 1 mm. `no_depth` gives NaN, and the inlier band excludes the background.
- The world-frame round trip works. The frame cloud count equals the valid count, and the colours align with the pixels.
- Voxel centroids are correct, and the PLY round trip works.

**AC4: met.** Measured without GT in `benchmarks/geometry/fr1_distortion.json`.
- p90 centre displacement is **2.10 cm**, which is above the 2 cm rule, so undistortion is **enabled for fr1/fr2**.
- It is set by `configs/dataset.yaml` `tum.undistort: true`, which makes `Frame.dist` drive `unproject_pixels`, a cached
  `cv2.undistortPoints` look-up table. Images are not resampled.
- The breakdown by image radius is in `docs/calibration.md`.
- Still open: projecting 3D results *back* onto images must apply the distortion. This is needed for Phase 07 overlays.

**AC5** (`scripts/time_geometry.py` → `benchmarks/geometry/geometry_timing.json`). Setup: 200 fr1 frames after 10 warm-up
frames, stored detections at conf ≥ 0.2411 (median 4, max 11 boxes per frame), undistortion on.

| Stage | Median | p95 | Bound | |
|---|---|---|---|---|
| Centres, real frames | 2.28 ms | 4.24 ms | < 5 ms | met |
| Centres, 50 boxes per frame (stress) | 17.7 ms | 22.7 ms | < 5 ms | **not met** (0.35 ms per box) |
| Frame cloud, stride 2, with RGB | 5.04 ms | 6.98 ms | < 30 ms | met |

The 50-box bound fails for the same reason as 03.1 R9: per-box Python and numpy overhead. Turning on the undistortion
look-up table made no measurable difference. Whether to restate the bound as "real frames + per-box cost" or to
vectorise across boxes in Phase 06 is the same open decision as in 03.1.

**AC3: met.** Viewer screenshots are in `benchmarks/geometry/viewer/`. The SUN images were chosen by a fixed rule: the
middle example of each tercile of median depth among val images with ≥ 2 detections and 3D GT
(`--suggest-sun`; terciles 0.79–1.93 / 1.93–2.44 / 2.44–5.83 m). Each frame has three files:
- `.png`: matplotlib top-down and camera views;
- `_o3d.png`: an open3d offscreen render;
- `.json`: per-object status, centre, and distance to the nearest GT box.

The checks below were done by eye on both renders. The distances come from the `.json`; 0 means inside the box.
- **SUN 007753, near (kv1, 1.66 m).** The GT boxes sit on the cloud: the table top and the chairs line up.
  - The 6 chair centres, the dining table and the two bottles on the tray are all inside their matching or containing GT boxes (0.00 m).
  - Two `person` detections (4.4 m) and a `refrigerator` on the left wall have no GT box. They are detector false positives or
    unlabelled objects, not geometry errors; the centres sit on the cloud surfaces they were detected on.
- **SUN 008574, mid (xtion, 2.14 m).** This is a dark scene.
  - The chair centre (`low_coverage`) is inside the chair GT box, which overlaps the desk box.
  - `tv 0.45` is `no_depth`: the screen returns no depth, so no centre is given, as required.
  - `tv 0.30` (5 m) is 0.33 m from the TV box. It is a far, low-confidence box on a shelf area whose cloud has
    structured-light striping. This is the weakest case and is recorded as such.
- **SUN 009543, far (xtion, 2.91 m).**
  - The chair centre is inside the right sofa_chair box, on its visible front face.
  - The couch centre is 1 cm outside the left sofa_chair box's front edge, which counts as the visible face.
  - The TV centre is 10 cm in front of the TV box. The screen itself has almost no depth, so the estimate (`low_coverage`) comes from the bezel and wall
    pixels in the box, and the GT box is set slightly behind the visible surface.
- **TUM fr1_desk frame 286 (undistortion on).** The mouse, laptop and book centres lie on the desk-top cloud at the right
  places. The cloud looks plausible: a flat desk plane, monitor edges, and no doubled surfaces.
- **TUM fr3_long_office frame 1242.** The keyboard, mouse, book and monitors lie on the desk and monitor surfaces, and the chair is in front of the desk.
  There are two `chair` boxes (0.33 / 0.32) on the same chair at 0.75 m, with centres 3 cm apart. This is a detector duplicate, which the one2one
  head does not always remove, and tracking will have to absorb it (Phase 04). A few far returns through the door at 6–7 m are real.

**Decision (2026-11-11).** The cost bound is restated: < 5 ms per frame on real frames at `conf_operational` (met: fr1 has at most 11 boxes, about 4 ms), with per-box cost recorded (03.1 about 0.14 ms per box; 03.2 about 0.35 ms per box). The 50-box figure stays as a recorded stress number, not a pass/fail bound. Vectorising across boxes is an optional Phase 06 optimisation if the end-to-end budget needs it.
