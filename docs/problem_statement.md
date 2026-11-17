# Problem Statement

**Spec:** 00.1 · **Status:** Drafted, awaiting approval · **Last updated:** 2026-11-07

> Labels: **[Target]** is a planning goal, not a result. No results exist yet.

## 1. The problem

A robot, drone, AR device, or assistive device that sees the world through an RGB-D camera needs four answers for every relevant object, many times per second:

1. **What is it?** Its category, such as chair, bed, TV, or bottle.
2. **Where is it?** Its position in metres, relative to the camera.
3. **Is it the same object as before?** Its identity across frames.
4. **Did the answer arrive in time?** The latency on the device's own hardware.

The system must also know when it does **not** know. Depth sensors return holes, noise, and out-of-range values on shiny, dark, transparent, or distant surfaces. A system that silently turns a bad depth reading into a confident distance is dangerous for anything that moves or grasps.

**Problem statement.**
> Given synchronized RGB images, registered metric depth images, and camera intrinsics, produce for each frame a set of detected objects. Each object has a class, a 2D box, a 3D position in the camera frame in metres, a depth-validity status, and a persistent track ID. Measure the accuracy of every output against ground truth where it exists, and measure the processing latency on a stated piece of hardware.

## 2. Input assumption (scope boundary)

| Required input | Why |
|---|---|
| RGB image | For object detection |
| Depth image **registered to the RGB image** (same pixel grid) | So the depth at pixel (u, v) belongs to the RGB pixel (u, v) |
| Depth **units** known (e.g. mm, or ×1/5000) | To get metres |
| Camera intrinsics `fx, fy, cx, cy` | For back-projection to 3D |
| (Optional) Camera pose per frame | Only to express positions in a world frame, used for the stability metric |

**Out of scope:** metric 3D from RGB alone, i.e. monocular depth estimation. Without real depth, the scale of the scene cannot be recovered reliably, and we will not claim metric accuracy we cannot measure.

## 3. Users

| User | Need |
|---|---|
| **Perception or robotics engineer** (primary) | A tested module that turns detections into metric 3D positions with honest validity flags |
| **Edge deployment engineer** | Latency per stage, and evidence of whether ONNX or INT8 actually help on CPU hardware |
| **Portfolio reviewer / interviewer** | Reproducible results, clear limitations, and design choices the author can defend |

## 4. Per-frame output contract

Each processed frame produces one record:

```json
{
  "frame_id": "string",
  "timestamp_s": 0.0,
  "objects": [
    {
      "track_id": 3,                       // null if the tracker has not confirmed it
      "class_name": "chair",
      "score": 0.87,                       // detector confidence, 0–1
      "bbox_xyxy": [x1, y1, x2, y2],       // pixels, in the ORIGINAL image
      "depth_status": "valid",             // see the table below
      "depth_m": 2.41,                     // null unless the status is valid or low_coverage
      "depth_spread_m": 0.06,              // robust spread (MAD-based); null if no depth
      "valid_depth_fraction": 0.78,        // valid pixels / pixels in the sampling region
      "center_cam_m": [X, Y, Z]            // camera frame in metres; null if no depth
    }
  ],
  "timing_ms": { "preprocess": 0, "detect": 0, "depth": 0, "geometry": 0, "track": 0, "total": 0 }
}
```

| `depth_status` | Meaning | Is a 3D position reported? |
|---|---|---|
| `valid` | Enough valid depth pixels, and their spread is below the threshold | Yes |
| `low_coverage` | Some valid depth, but below the coverage threshold or with high spread | Yes, **flagged** as unreliable |
| `no_depth` | No usable depth pixels in the region | **No** (null) |

The thresholds are set in Phase 03 from data and recorded in `configs/`.

**Camera frame convention (OpenCV):** +X points right, +Y points down, +Z points forward out of the lens, in metres. Dataset-specific frames, such as SUN RGB-D's upright frame, are converted to this convention in Phase 01.

## 5. Measurable goals

| ID | Goal [Target] | Metric | Data | Measured in |
|---|---|---|---|---|
| **G1** | Back-projection is exact | Max round-trip error < 1e-6 m; a known synthetic plane is recovered | Synthetic depth maps we generate | Spec 03.3 |
| **G2** | Detection baseline is documented | mAP@50, mAP@50–95, precision, recall on classes shared by COCO and the dataset | SUN RGB-D test split | Spec 02.3 |
| **G3** | 3D center error is measured | Median and p90 3D center error in m, broken down by distance; valid-depth coverage | SUN RGB-D objects with 3D box ground truth | Spec 05.2 |
| **G4** | Invalid depth is never silently used | 100% of detections carry a `depth_status`; unit tests for zero, NaN, and out-of-range depth | Synthetic data and SUN RGB-D | Specs 03.1, 03.2 |
| **G5** | Tracking stability is measured | ID switches, fragmentation; IDF1/HOTA if labels exist; otherwise the world-frame position std-dev of static objects | TUM RGB-D (ScanNet if access is granted) | Spec 04.3 |
| **G6** | Optimization is measured fairly | Median and p95 latency, FPS, model size, memory, and mAP change for PyTorch vs ONNX Runtime FP32 vs INT8, on the same CPU | Fixed benchmark subset | Specs 06.3, 06.4 |
| **G7** | The demo is reproducible | A fresh clone plus the README steps runs the demo and `pytest` passes | Sample sequence | Spec 07.3 |

**Speed goal (soft) [Target].** Ideally 10+ FPS end-to-end on the development CPU at a reduced input size. On an i3-1115G4 this is uncertain. If we miss it, we report the bottleneck and the optimization steps we tried. We do not change the measurement boundary to make the number look better.

## 6. Scope

**MVP (must have)**
1. Offline processing of recorded RGB-D frames and sequences.
2. A pretrained 2D detector (no training).
3. Robust per-object depth, with the validity statuses above.
4. 3D centers in the camera frame, and a scene point cloud.
5. ByteTrack track IDs, with a 3D position history for each track.
6. Quantitative evaluation (G2–G5).
7. ONNX export, and a CPU benchmark against PyTorch (G6).
8. A Streamlit viewer: RGB overlay, depth view, point cloud with object centers.

**Extensions (only if time allows and a measured need exists)**
- FastAPI service. This is planned for Phase 07, but the MVP does not depend on it.
- Live RGB-D camera input, if the user has a sensor.
- Oriented 3D boxes.
- Segmentation-based depth sampling.
- Fine-tuning on Kaggle GPU.
- TensorRT benchmark on Kaggle.
- OpenVINO execution provider.
- A learned point-cloud model.

**Non-goals**
- Training any model on the local CPU.
- Monocular metric depth.
- SLAM or mapping.
- ROS integration.
- A production frontend.
- Commercial use of non-commercial datasets.
