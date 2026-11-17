# Spec 01.3 — Annotation Parser, Split Validation, and Data Visualization

| Field | Value |
|---|---|
| Phase | 01 — Data |
| Status | ✅ Accepted 2026-11-09 (AC1–AC4 met; AC2 replaced by the stricter version below, approved by the user) |
| Depends on | 01.2 |
| Blocks | 02.3 (detection evaluation), 05.2 (3D evaluation) |

## Objective
Load the SUN RGB-D 2D and 3D box annotations, **convert the 3D boxes into the OpenCV camera frame**, build scene-level splits, and prove visually that the annotations line up with the images.

## Background: why this spec carries the highest risk (R2)
SUN RGB-D stores each 3D box in an **upright, gravity-aligned frame**. A per-image `Rtilt` rotates camera-frame points into that frame, which uses the axes x right, y forward, z up. Our plan for the conversion, to be confirmed on the data:

```
p_cam_flipped = Rtiltᵀ · p_upright               (undo the tilt)
p_opencv      = (x, −z, y) of p_cam_flipped       (z-up/y-forward → y-down/z-forward)
```

This matches the convention used in public SUN RGB-D tooling (e.g. VoteNet's `sunrgbd_utils`) [To verify]. Every G3 number depends on getting it right, so the acceptance criteria use a visual check *and* a quantitative one.

## Requirements
- **R1.** Parse `SUNRGBDMeta3DBB_v2.mat`. For each image, read the image paths, K, Rtilt, and the 3D boxes (class, centroid, basis, coeffs). Parse `SUNRGBDMeta2DBB_v2.mat` for the 2D boxes.
- **R2.** Compute the 8 corners of each 3D box in the upright frame, then convert the corners and the centroid to the OpenCV camera frame.
- **R3.** Normalize class names (lower case, strip whitespace), and keep the raw name too.
- **R4.** Splits:
  - use the official train/test split from the toolbox;
  - carve a **validation** subset out of train *by scene folder* with a fixed seed;
  - save the splits as text files of image IDs under `configs/splits/`.
- **R5.** A split-overlap check asserts that no scene ID appears in two splits.
- **R6.** Visualization: draw the 2D boxes, and the projected 3D box wireframes, on the RGB image.
- **R7.** The parsed metadata is cached as compact JSON (R9 RAM policy).

## Tasks
- [x] T1. Inspect the `.mat` structure, and document the field names in `docs/data_layout.md`.
- [x] T2. Annotation dataclasses and the parser, with a JSON cache.
- [x] T3. Upright → camera transform in `perception/geometry/transforms.py`.
- [x] T4. Split builder and overlap check.
- [x] T5. `scripts/visualize_annotations.py`.
- [x] T6. Visual check on 20 random images.

## Acceptance criteria
- **AC1.** On 20 random test images, the projected 3D wireframes visibly enclose their objects.
- **AC2 (revised 2026-11-09, approved by the user).** **Quantitative check:** for each GT 3D box, take the valid depth pixels inside its projected 2D footprint and compute their median Z. Across all boxes:
  - the median of `|median_depth − box_center_Z|` is **≤ 0.3 × the median box depth extent**; and
  - the correct transform has a lower median error than **every** negative control (wrong tilt, tilt ignored).

  **Met:** correct 0.19 m (92.5% of boxes within extent) against ≥ 0.46 m (66%) for the controls; see `benchmarks/data_quality/annotations_check.json`.
  _Superseded wording:_ "median error smaller than the median box depth extent". The wrong-transform controls also passed it, so it could not catch the bug it was written for.
- **AC3.** The split files exist; the train/test counts match the official split; the overlap test passes.
- **AC4.** Class frequency tables are generated.

## Tests (`tests/test_transforms.py`, `tests/test_splits.py`)
- With the identity Rtilt, an upright point (0, 2, 0) (2 m forward) maps to OpenCV (0, 0, 2).
- An upright point (0, 0, 1) (up) maps to OpenCV Y = −1 (up is −Y).
- The transform is a rotation: lengths are preserved and the determinant is +1.
- Box corner generation: 8 corners, and the centroid equals the corner mean.
- The split builder is deterministic for a given seed, with no scene overlap.

## Dependencies
scipy.io / h5py (MAT files), numpy, opencv-python.

Results: [docs/annotations.md](../../docs/annotations.md).
