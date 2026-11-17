# Spec 01.2 — RGB/Depth Alignment and Camera-Calibration Reader

| Field | Value |
|---|---|
| Phase | 01 — Data |
| Status | ✅ Accepted 2026-11-09 (all ACs met 2026-11-08; overlays reviewed by the user) |
| Depends on | 01.1 |
| Blocks | 01.3, 01.4, and Phases 02–03 |

## Objective
Provide one `Frame` contract for every dataset: RGB, depth in **metres (planar Z)**, a valid mask, intrinsics K, an optional camera pose, and a timestamp. Every reader must produce it correctly. All unit and encoding quirks stay inside the readers.

## The `Frame` contract

| Field | Type | Meaning |
|---|---|---|
| `rgb` | uint8 H×W×3, **RGB order** | The colour image (OpenCV loads BGR, so the reader converts it) |
| `depth_m` | float32 H×W | Planar Z in metres; `0.0` where invalid |
| `valid` | bool H×W | `True` where the depth is usable |
| `K` | float64 3×3 | Intrinsics for *this* image size |
| `T_world_cam` | float64 4×4 or None | Camera-to-world pose (TUM ground truth) |
| `timestamp` | float or None | Seconds |
| `frame_id`, `dataset`, `scene_id` | str | Identification; `scene_id` drives the splits |

## Requirements
- **R1.** `perception.depth.decode`: per-dataset raw → metres conversion.
  - TUM: `raw / 5000`.
  - SUN RGB-D: `((raw >> 3) | (raw << 13)) & 0xFFFF / 1000`.
  - `0 → invalid` in both cases.
- **R2.** The valid mask comes from the configurable range `[min_m, max_m]` and is finite.
- **R3.** `perception.geometry.camera`: `project`, `unproject`, `unproject_depth_map`, `scale_intrinsics`, using the OpenCV convention.
- **R4.** TUM reader:
  - parse `rgb.txt`, `depth.txt`, `groundtruth.txt`;
  - associate RGB↔depth by nearest timestamp within `max_dt` (default 0.02 s);
  - interpolate the pose at the RGB timestamp (linear for position, SLERP for rotation), dropping frames outside the ground-truth time range;
  - choose intrinsics by sequence prefix (fr1/fr2/fr3) or the ROS default, as configured.
- **R5.** SUN RGB-D reader: read the RGB, depth, and K for each image from the dataset folders. The image list comes from the 3D metadata.
- **R6.** Readers are lazy: indexing loads one frame. Nothing loads all frames at once.

## Tasks
- [x] T1. `Frame` dataclass, with validation of shapes and dtypes.
- [x] T2. Depth decoding and the valid mask.
- [x] T3. Camera model functions.
- [x] T4. TUM reader (association and pose interpolation).
- [x] T5. SUN RGB-D reader (after inspecting the real file layout).
- [x] T6. `scripts/check_alignment.py`: overlay depth edges on RGB for N sampled frames and save PNGs.
- [x] T7. **Visual check:** depth edges line up with RGB edges on 10 frames per dataset.

Results: [docs/calibration.md](../../docs/calibration.md).

## Acceptance criteria
- **AC1.** All unit tests pass.
- **AC2.** The alignment overlays for 10 TUM and 10 SUN frames show depth edges on the object boundaries (reviewed by eye, and saved under `benchmarks/data_quality/alignment/`).
- **AC3.** The median valid depth per frame falls in the plausible indoor range of 0.5–6 m on ≥ 95% of sampled frames.
- **AC4.** The TUM intrinsics choice is documented, with the reason.

## Tests (`tests/test_geometry.py`, `tests/test_depth.py`, `tests/test_data_tum.py`)
- Project → unproject round trip error < 1e-9 m on random points.
- `unproject_depth_map` on a synthetic fronto-parallel plane at Z = 2 m gives every point with Z = 2 and the correct X range.
- `scale_intrinsics(K, 0.5)` halves fx, fy, cx, cy, using the pixel-center convention.
- TUM decode: 5000 → 1.0 m, 0 → invalid.
- SUN decode: a bit-rotated encoding of 1234 mm decodes to 1.234 m.
- Out-of-range and NaN depth are marked invalid.
- Timestamp association on synthetic lists: nearest match, unmatched frames dropped beyond `max_dt`.
- Pose interpolation halfway between two poses gives the midpoint translation, and a rotation at half the angle.

## Dependencies
numpy, opencv-python, scipy (`Rotation`, `Slerp`), pyyaml.
