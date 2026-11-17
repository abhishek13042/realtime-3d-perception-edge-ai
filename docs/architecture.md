# Architecture

**Spec:** 00.4 · **Status:** Drafted · **Last updated:** 2026-11-07

## 1. Pipeline

```
 DatasetReader (SUN RGB-D | TUM | ScanNet adapter)
     │  Frame{rgb uint8 HxWx3, depth_m float32 HxW, K 3x3, T_world_cam 4x4|None, t}
     ▼
 [1] Validate   → depth to metres, valid_mask (0 / NaN / <min / >max), registration check
     ▼
 [2] Detect     → Detection{bbox_xyxy px (original image), class, score}   ← ONNX/PyTorch backend
     ▼
 [3] DepthAssoc → DepthEstimate{depth_m, spread_m, valid_fraction, status, sample_uv}
     ▼
 [4] Geometry   → center_cam_m (X,Y,Z), optional scene point cloud (N×3, + colour)
     ▼
 [5] Track      → ByteTrack on 2D boxes → track_id;  Track.history[(t, center_cam, center_world?)]
     ▼
 [6] Output     → FrameResult (JSON contract, docs/problem_statement.md §4) + per-stage timings
     ▼
 [7] Present    → Streamlit viewer · FastAPI /infer · benchmark reports
```

**Design rule:** stages 1, 3, 4, and 5 are **pure functions** of their inputs plus a config. They do no I/O and have no global state (the tracker keeps its state in an explicit object). This makes each stage testable with synthetic data.

## 2. Coordinate frames and units (single source of truth)

| Frame | Axes | Unit | Used for |
|---|---|---|---|
| **Image** | u → right, v → down, origin at the top-left pixel corner; pixel *centers* at integer coordinates | px | Boxes, depth lookup |
| **Camera (OpenCV)** | +X right, +Y down, +Z forward | m | All 3D outputs |
| **World** | Dataset-defined (TUM: mocap frame; ScanNet: scan frame) | m | Stability metric only |
| **SUN upright** | Gravity-aligned, via `Rtilt` | m | GT 3D boxes. **Converted to the camera frame in the reader**, never used downstream |

- **Depth convention:** `depth_m` is always **planar Z** in metres, never the ray distance.
- **Invalid depth:** stored as `0.0`, with `valid_mask == False`.
- **Intrinsics:** always describe the image *as delivered by the reader*. If an image is resized, K is rescaled in the same function.

## 3. Stage details and choices

### [1] Validate
- Converts raw depth to metres using a per-dataset scale: SUN = bit-rotation then /1000; TUM = /5000.
- Builds the valid mask with a configurable range. The default is 0.3–8.0 m [Target, to be set from sensor specs and data].
- Records the per-frame **valid-depth fraction**, which is reported in the evaluation.

### [2] Detect
- A pretrained COCO detector behind a `Detector` interface with two backends: `torch` and `onnxruntime`.
- Preprocessing: letterbox resize to the model input size, keeping a record of the scale and padding. Postprocessing: inverts the letterbox, so boxes come out in **original-image pixels**.
- Candidate models: **Ultralytics YOLO26-n** (the current family, AGPL-3.0) or **YOLOX-Nano/Tiny** (Apache-2.0). **Decision D3 is pending.**

### [3] Depth association (the core heuristic)
1. **Shrink the box** about its center by a factor `s` (default 0.5 [Target]). The edges of a box mostly contain background.
2. Collect the valid depth pixels in the shrunken region.
3. **Estimator A (baseline):** median of those pixels.
   **Estimator B:** nearest significant mode of the depth histogram, which targets the foreground object.
4. **Spread:** `1.4826 · MAD`. This is a robust stand-in for the standard deviation, and it measures how mixed the depths are.
5. **Status:**
   - `no_depth` if there are fewer than N_min valid pixels.
   - `low_coverage` if the valid fraction is below τ_cov, or the spread is above τ_spread.
   - `valid` otherwise.

   The thresholds are chosen in Phase 03 from data.
- **Alternative (extension):** sample only inside an instance-segmentation mask. This adds model latency, so we add it only if Phase 05 shows that background bleed dominates the error.

### [4] Geometry
- Back-projection is vectorized with NumPy. For the object center, we back-project the **pixels whose depth lies within ±k·spread of the estimate**, then take their 3D median. This is more robust than back-projecting the box center pixel, which may lie on background.
- Point cloud: the whole frame, or only the valid pixels, optionally voxel-downsampled with Open3D for display.

### [5] Track
- ByteTrack (two-stage association) on 2D boxes and scores. 3D centers are attached to tracks; the tracker itself does not use them.
- **Alternative:** a 3D Kalman tracker (AB3DMOT-style). It is more robust to camera motion, but it fails whenever depth is invalid. **Trigger for switching:** Phase 04 finds that ID switches correlate with camera motion.

### [6–7] Output and presentation
- `FrameResult` is serialized to JSON lines.
- The Streamlit app reads results or runs the pipeline.
- FastAPI wraps the same `Pipeline` object, so there is no duplicated logic.

## 4. Module map

```
src/perception/
  data/           readers: base.py, sunrgbd.py, tum.py, (scannet.py); Frame dataclass
  depth/          validate.py (units, masks), estimate.py (estimators A/B, status)
  geometry/       camera.py (K, project, unproject, rescale), pointcloud.py, transforms.py
  detection/      base.py (Detector), torch_backend.py, onnx_backend.py, letterbox.py, classes.py
  tracking/       bytetrack adapter, track_history.py
  inference/      pipeline.py (stage orchestration + timing)
  visualization/  overlay.py, depth_vis.py, o3d_view.py
  api/            FastAPI app
configs/          dataset.yaml, detector.yaml, runtime.yaml   (all thresholds live here)
scripts/          inspect_dataset.py, run_sequence.py, benchmark.py, export_onnx.py
tests/            test_geometry.py, test_depth.py, test_tracking.py, test_inference.py, test_data.py
app/              Streamlit
```

## 5. Phase exit criteria

| Phase | Exit criteria |
|---|---|
| 01 Data | Readers return the `Frame` contract; depth reprojection looks aligned on 10 sampled frames per dataset; GT 3D boxes project onto the image correctly; the scene-split test shows no overlap; quality report generated |
| 02 Detection | Deterministic output; letterbox round trip ≤ 1 px; baseline mAP table on the shared classes |
| 03 Geometry | Synthetic tests pass (round trip < 1e-6 m, plane recovery); unit and invalid-depth tests pass; point cloud viewer works |
| 04 Tracking | IDs persist on TUM sequences; ID switches and fragmentation computed; limitations documented |
| 05 Evaluation | Protocol followed; 3D error by distance bucket; stage profile; failure gallery; results re-run within a stated tolerance |
| 06 Optimization | ONNX output matches PyTorch within tolerance; FP32/INT8 benchmarked on the same machine; trade-off table |
| 07 Product | A fresh setup from the README works; `pytest` green; demo video; limitations published |

## 6. Dependencies (versions pinned in Phase 01)

| Package | Purpose | License note |
|---|---|---|
| numpy, opencv-python | Arrays, image I/O, geometry | BSD / Apache-2.0 |
| torch | Detector reference backend | BSD-style |
| ultralytics **or** yolox | Pretrained detector and export | **AGPL-3.0** / Apache-2.0 (D3) |
| onnx, onnxruntime | Export and optimized inference | MIT |
| open3d | Point clouds and 3D view | MIT |
| ByteTrack implementation | Tracking | MIT (reference code) |
| pycocotools / TrackEval | mAP / HOTA, IDF1 | BSD / MIT |
| scipy | Hungarian matching, MAT-file reading | BSD |
| fastapi, uvicorn, streamlit | API and demo | MIT / BSD / Apache-2.0 |
| pytest | Tests | MIT |

**Python version.** As of 2026-11-07, torch 2.15, onnxruntime 1.31.1, open3d 0.20.0, and ultralytics 8.4 all publish Windows wheels for **Python 3.14** [Verified on PyPI]. Secondary packages (pycocotools, `lap`, TrackEval, onnxruntime-openvino) are not yet checked.
**Plan:** create a project venv on 3.14. If any secondary package fails to install, fall back to 3.12. The system interpreter is never used directly.

## 7. Compute and storage budget

**Development machine (measured 2026-10-21)**
- CPU: Intel Core i3-1115G4 (2C/4T).
- GPU: Intel UHD (no NVIDIA).
- RAM: 7.7 GB.
- Free disk on D: about 102 GB.

| Item | Disk |
|---|---|
| SUN RGB-D (archive + extracted) | ~20 GB |
| TUM RGB-D, 3–5 sequences | ~3–8 GB |
| ScanNet subset (optional) | ~10–25 GB |
| venv + models + outputs | ~5–8 GB |
| **Total** | **~40–60 GB** ✅ fits |

**Compute policy**
- Local CPU only: data inspection, tests, inference, benchmarks.
- **No local training.**
- Kaggle GPU: only for optional fine-tuning, and an optional TensorRT comparison that is clearly labelled as other hardware.

**RAM policy**
- Readers yield frames lazily. A 640×480 frame is about 2.2 MB for RGB plus depth, and a full-frame cloud is about 307k points (~3.7 MB as float32).
- MATLAB metadata files are parsed once and cached as compact JSON/NPZ.
