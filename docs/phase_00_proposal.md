# Phase 00 — Project Definition, Research, and Proposal

**Project:** Real-Time 3D Perception and Edge Intelligence Platform
**Status:** DRAFT. Waiting for approval. No implementation has started.

> **Superseded in detail (2026-11-07).** The per-spec documents (`problem_statement.md`, `literature_review.md`, `dataset_audit.md`, `architecture.md`, `evaluation_protocol.md`, `risks_and_limitations.md`) are now authoritative. Updates since this draft:
> - Python 3.14 is now supported by torch, onnxruntime, open3d, and ultralytics.
> - The current Ultralytics model family is YOLO26.
> - TUM depth scale (×5000, pre-registered) and intrinsics are now [Verified].
**Date:** 2026-10-21

> Conventions in this document
> - **[Verified]**: checked against an official source during Phase 00 (links given).
> - **[To verify]**: plausible but not yet confirmed. Phase 01 must confirm it before we rely on it.
> - **[Target]**: a planning goal, not a result. No results exist yet.

---

## 0. Development machine (measured, 2026-10-21)

| Item | Value | Consequence |
|---|---|---|
| CPU | Intel Core i3-1115G4 (2 cores / 4 threads, 3.0 GHz) | Real-time rates are hard to reach here. All runtime numbers must name this CPU. |
| GPU | Intel UHD Graphics (integrated). **No NVIDIA GPU.** | We cannot run TensorRT or CUDA locally. The OpenVINO execution provider for ONNX Runtime is the natural accelerator to test on this machine. |
| RAM | 7.7 GB | Load data lazily and stream frames. Never load a whole dataset into memory. |
| Free disk (D:) | ~102 GB | Enough for the recommended datasets, if we download subsets. |
| Python | 3.14.0 system install | This is too new for some scientific wheels (Open3D, onnxruntime). **[To verify]** We will use a separate 3.11 or 3.12 virtual environment. |

This changes the plan in the pasted brief in two ways:
1. **TensorRT moves from "optional" to "remote-only".** If we test it at all, we test it on a Kaggle GPU (T4 or P100) and disclose that the hardware is different.
2. **The local edge story becomes CPU and Intel iGPU deployment.** That means ONNX Runtime on CPU, INT8 quantization, a smaller input resolution, and possibly OpenVINO. This is a real edge scenario, and it is arguably more honest than a desktop GPU.

---

## 1. Problem statement and MVP

### Problem statement
A robot or smart device that sees through an RGB-D camera needs four things for each object it cares about:
- what the object is,
- where it is in metric 3D space relative to the camera,
- whether it is the same object it saw a moment ago,
- whether that answer arrived quickly enough to act on.

It must also say "I don't know" when the depth is missing or unreliable, rather than inventing a distance.

### Users
- **Primary:** a perception or robotics engineer who needs object positions in the camera frame from a commodity RGB-D sensor.
- **Portfolio audience:** interviewers who will probe the geometry, the metrics, and the deployment trade-offs.

### Measurable goals (all [Target])
| ID | Goal | How it is measured |
|---|---|---|
| G1 | 3D back-projection is exact on synthetic data | Unit tests: max error < 1e-6 m on planes and points we generate ourselves |
| G2 | Detection baseline is documented | mAP@50 and mAP@50–95 on the classes the detector and dataset share, with a fixed evaluation split |
| G3 | 3D object center error is measured | Median and p90 error in metres against ground-truth 3D box centers; valid-depth coverage is reported alongside |
| G4 | Invalid depth is never silently used | Every detection is labelled `valid`, `low_coverage`, or `no_depth`. Tests cover each case. |
| G5 | Tracking stability is measured | IDF1 and HOTA where labels exist. Otherwise, world-frame jitter of static objects (see §6). |
| G6 | Optimization is measured fairly | PyTorch vs ONNX Runtime FP32 vs INT8 on the same CPU, same inputs, with warm-up and repetitions defined. We report median and p95 latency, and the change in mAP. |
| G7 | The demo is reproducible | A fresh clone, documented setup, and one command run the demo on a sample sequence |

### MVP (minimum version that counts as "done")
1. Offline pipeline over recorded RGB-D frames: detect → robust depth per box → 3D center in the camera frame → track IDs.
2. Quantitative 2D and 3D evaluation on one annotated RGB-D dataset.
3. Tracking demo on one recorded sequence, with a stability metric.
4. ONNX export, plus a CPU latency comparison against PyTorch.
5. Streamlit viewer showing the RGB overlay, the depth view, and a point cloud with object centers.

**Out of MVP (extensions):** live camera input, 3D oriented boxes, FastAPI service, TensorRT, a learned 3D model (PointNet-style), and fine-tuning.

---

## 2. Dataset comparison

| | **SUN RGB-D** | **NYU Depth V2** | **ScanNet v2** | **Hypersim** | **TUM RGB-D** |
|---|---|---|---|---|---|
| Source | [rgbd.cs.princeton.edu](https://rgbd.cs.princeton.edu/) | [cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) | [github.com/ScanNet/ScanNet](https://github.com/ScanNet/ScanNet) | [github.com/apple/ml-hypersim](https://github.com/apple/ml-hypersim) | [cvg.cit.tum.de/data/datasets/rgbd-dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) |
| Access | Direct download | Direct download | **Signed Terms of Use with an institutional email, sent by email; the reply may take about a week** [Verified] | Direct download | Direct download |
| License | No formal license stated. The site asks for citation. [Verified] | No license stated. The site asks for citation. [Verified] | ScanNet Terms of Use (non-commercial research). Code is MIT. [Verified] | CC BY-SA 3.0. Ground-truth meshes and source assets must be bought. [Verified] | CC BY 4.0 [Verified] |
| Real or synthetic | Real, from 4 sensors | Real (Kinect v1) | Real (Structure sensor) | Synthetic | Real (Kinect) |
| Size | 10,335 RGB-D frames | 1,449 labelled frames (2.8 GB); raw set 428 GB | 1,513 scans, 2.5M frames | 74,619 images, ~1.9 TB | Dozens of sequences, each 0.5–2 GB |
| Sequences (for tracking) | No (single frames) | Raw set only, unlabelled | **Yes, with per-frame instance IDs** (2D label projections) | Short camera trajectories | **Yes, with ground-truth camera poses**, but no object labels |
| Object annotations | 2D polygons, **64,595 3D oriented boxes** [Verified, paper] | Per-pixel class and instance masks (boxes can be derived) | 3D instance segmentation and per-frame 2D instance PNGs | 9-DOF 3D boxes, instance masks, NYU40 labels | None |
| Intrinsics | Per image, in the toolbox | In `camera_params.m` in the toolbox | Per scan, in `.sens` | Per scene. Some scenes use tilt-shift, so a modified projection matrix is needed. | Published Kinect calibration |
| Depth gotchas | Depth PNGs are bit-rotated in the toolbox [To verify] | The labelled `depths` are **inpainted**, so `rawDepths` must be used for evaluation | Depth is 640×480 and RGB is 1296×968, so the streams must be registered | **`depth_meters` is Euclidean distance from the optical center, not planar Z** [Verified] | Depth is scaled by 5000 [To verify] |
| Fit for our project | **Best for single-frame 2D detection and 3D localization evaluation** | Good for geometry; a subset of it is included in SUN RGB-D | **Best overall, but access is gated** | Good fallback; synthetic-to-real gap; very large | **Best ungated source for tracking stability** |

### Points to raise in interviews
- **The NYU inpainting trap.** If you evaluate depth on inpainted maps, you are scoring a colorization algorithm, not the sensor.
- **The Hypersim distance trap.** If you plug Euclidean ray distance into `X = (u−cx)Z/fx`, objects near the image edges come out too far away. You must first convert: `Z = d · f / sqrt((u−cx)² + (v−cy)² + f²)`.
- **Split leakage.** Frames from the same room are near-duplicates. Every split must be made at the scene level, never the frame level.

### Licensing posture for a public GitHub repo
We commit **no dataset files** and no derived images except a few small figures, each credited to its source. We publish download scripts and instructions only. We do not use ScanNet for anything commercial.

---

## 3. Literature review and reading order

| # | Paper | Why it matters here | How we use it |
|---|---|---|---|
| 0 | Pinhole camera model: [OpenCV camera calibration docs](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html); Hartley & Zisserman, *Multiple View Geometry*, ch. 6 | This is the projection and back-projection we implement | **Implemented directly** |
| 1 | Qi et al., **PointNet**, CVPR 2017 ([arXiv:1612.00593](https://arxiv.org/abs/1612.00593)) | Point sets are unordered. A symmetric function (max-pool) gives permutation invariance. | Design reference only |
| 2 | Qi et al., **PointNet++**, NeurIPS 2017 ([arXiv:1706.02413](https://arxiv.org/abs/1706.02413)) | Adds local neighbourhoods (farthest-point sampling and ball grouping), which matters when point density varies with depth | Design reference only |
| 3 | Qi et al., **Frustum PointNets**, CVPR 2018 ([arXiv:1711.08488](https://arxiv.org/abs/1711.08488)) | The closest relative of our pipeline: a 2D box becomes a 3D frustum, which becomes a 3D localization. **It was evaluated on SUN RGB-D.** | Our "median depth inside the box" is a **non-learned baseline** of this idea. The paper's segmentation step explains why our baseline fails on occluded objects and background bleed. |
| 4 | Bewley et al., **SORT**, ICIP 2016 ([arXiv:1602.00763](https://arxiv.org/abs/1602.00763)) | Kalman filter plus Hungarian matching: the skeleton ByteTrack builds on | Background |
| 5 | Zhang et al., **ByteTrack**, ECCV 2022 ([arXiv:2110.06864](https://arxiv.org/abs/2110.06864)) | Associates low-score boxes in a second pass, which recovers occluded objects | **Integrated directly** (MIT-licensed reference implementation) |
| 6 | Luiten et al., **HOTA**, IJCV 2021 ([arXiv:2009.07736](https://arxiv.org/abs/2009.07736)) | A tracking metric that balances detection and association | Evaluation |

Reading order: 0 → 1 → 2 → 3 → 4 → 5 → 6. Read 3 carefully, and skim 1 and 2 only for intuition.

We will claim to have *implemented* only the geometry, the depth estimator, the tracker integration, and the benchmarks. We will not claim to have implemented PointNet, PointNet++, or F-PointNets.

---

## 4. Architecture

```
RGB + depth + intrinsics (dataset reader: one interface, one adapter per dataset)
  → validation: depth unit → metres, invalid mask (0 / NaN / out of range), registration check
  → 2D detector (pretrained, COCO classes) → boxes in source-image pixels
  → depth association: shrink box → valid pixels → robust Z (median) + spread (MAD) + coverage
  → back-projection: (u, v, Z) → (X, Y, Z) in the camera frame; status = valid | low_coverage | no_depth
  → tracker: ByteTrack on 2D boxes; each track carries a history of 3D centers
  → (optional) camera pose → world frame, used for the stability metric
  → outputs: JSON per frame, overlays, point cloud, timing per stage
```

**Key decision.** Tracking runs in 2D image space first, and 3D positions are attached to tracks. *Alternative:* track in 3D with a 3D Kalman filter. This is more robust when the camera moves, but it fails whenever depth is invalid, which happens often. We start with 2D and measure whether camera motion causes ID switches. We switch to 3D tracking only if the measurements show 2D is the problem.

**Depth estimator alternatives** (we compare them in Phase 05):
- (a) Median of the central 50% of the box. This is the baseline.
- (b) The nearest mode of the depth histogram. This targets the foreground object and rejects the background behind it.
- (c) Restrict to pixels inside a segmentation mask. This requires a segmentation model, so it is an extension.

**Detector choice** (needs your decision, see §8):

| Option | License | Pros | Cons |
|---|---|---|---|
| Ultralytics YOLO nano | **AGPL-3.0** | Best tooling, one-line ONNX export, maintained | The repo must be AGPL-compatible. That is fine for a public portfolio. |
| YOLOX-Nano / Tiny | Apache-2.0 | Permissive license | Older, and the tooling is rougher |

---

## 5. Phase and specification breakdown

Every specification gets a file in `specs/phase_XX/` with five sections: **Requirements · Tasks · Acceptance criteria · Tests · Dependencies**. The acceptance criteria are summarized below.

| Spec | Deliverable | Acceptance criteria (summary) |
|---|---|---|
| **00.1** | Problem statement | §1 approved |
| **00.2** | Literature review | §3 approved; reading notes written by you |
| **00.3** | Dataset audit | Licenses, sizes, and intrinsics for the chosen datasets verified; [To verify] items resolved |
| **00.4** | Architecture, risks, budget | §4, §6, and §7 approved |
| **01.1** | Downloader / importer | Script fetches a documented subset; checksums recorded; data lives outside Git |
| **01.2** | Calibration and alignment reader | Reprojecting depth onto RGB lines up visually on 10 sampled frames; intrinsics unit-tested against known values |
| **01.3** | Annotation parser and splits | Ground-truth 2D boxes drawn correctly; 3D boxes converted to the camera frame and projected onto the image as a visual check; splits are made by scene, and a test asserts there is no overlap |
| **01.4** | Quality report and sample loader | Per-dataset report covering depth validity %, depth range, and class counts; loader is deterministic given a seed |
| **02.1** | Detector integration | Same input gives the same output; versions are pinned |
| **02.2** | Coordinate mapping | Letterbox inverse is tested: a known box survives the resize round trip within 1 px |
| **02.3** | Baseline metrics | mAP@50 and mAP@50–95 on the shared classes, using pycocotools; failure gallery written |
| **03.1** | Depth units and invalid depth | Tests for 0, NaN, saturation, and unit scaling (mm → m, ×1/5000) |
| **03.2** | Robust per-box depth | Estimators (a) and (b) implemented; status flags tested; synthetic box-over-background test passes |
| **03.3** | Back-projection | Synthetic round-trip project → unproject error < 1e-6; known plane recovered |
| **03.4** | Point cloud and visualization | Vectorized depth → points; Open3D view with axes and centers; downsampling documented |
| **04.1** | Tracker | ByteTrack integrated; tracker is a pure function of the detections, so it can be tested |
| **04.2** | Track history | 3D history per ID; handles frames with missing depth |
| **04.3** | Tracking evaluation | ID switches, fragmentation, IDF1/HOTA (TrackEval) where labels exist; otherwise world-frame jitter |
| **05.1** | Benchmark protocol | Hardware, warm-up, repetitions, boundaries, and data are written down **before** running |
| **05.2** | 3D error | Center error vs ground truth, broken down by distance bucket and coverage |
| **05.3** | Per-stage profiling | Time for read / pre / infer / post / depth / track / visualize, measured separately |
| **05.4** | Failure analysis | Occlusion, reflective and dark surfaces, thin objects, out-of-range depth |
| **06.1** | ONNX export | Model passes the ONNX checker; opset recorded |
| **06.2** | Consistency | Max absolute output difference vs PyTorch below a stated tolerance; mAP difference reported |
| **06.3** | ORT benchmarks | CPU FP32, INT8 (static, calibrated), and optionally OpenVINO EP. Kaggle GPU CUDA/TRT only if you approve. |
| **06.4** | Trade-off report | Latency, throughput, memory, model size, and mAP change in one table |
| **07.1** | Inference interface and FastAPI | `/infer` and `/health`; tested with the FastAPI TestClient |
| **07.2** | Streamlit demo | RGB, depth, tracks, and point cloud |
| **07.3** | Tests, configs, setup | A fresh environment passes `pytest`; README steps checked by following them exactly |
| **07.4** | Documentation | Architecture diagram, results, limitations, demo video |

---

## 6. Evaluation methodology

- **Detection.** Evaluate only on classes shared by COCO and the dataset. For SUN RGB-D the likely shared classes are *chair, sofa→couch, bed, toilet, table→dining table* [To verify mapping quality]. Desk, dresser, and similar classes have no COCO equivalent, so they are excluded and the exclusion is disclosed.
- **Depth.** For each detection, compare the estimated Z with the Z of the ground-truth 3D box center. Report MAE, RMSE, and coverage. Only raw sensor depth counts as ground truth. Inpainted depth is never used.
- **3D localization.** Match predicted boxes to ground-truth boxes by 2D IoU ≥ 0.5, then compute the Euclidean distance between the 3D centers in metres. Report it by distance bucket (0–2, 2–4, >4 m).
  - **Caveat:** the ground-truth box center is the center of the object's *volume*. Our estimate is a center on the *visible surface*. This creates a systematic bias of roughly half the object's depth. We report the bias explicitly; it is not a bug.
- **Tracking.**
  - *With ScanNet:* IDF1, HOTA, ID switches, and fragmentation using TrackEval.
  - *With TUM (no labels):* most objects in the scene are static. Using the ground-truth camera poses, transform each track's 3D centers into the world frame. Their standard deviation is a **jitter metric that needs no object labels**.
- **Runtime.**
  - Measurement boundary: from the decoded frame in memory to the per-frame result object. Disk I/O and rendering are reported separately.
  - Procedure: 20 warm-up iterations, then at least 200 timed iterations.
  - Report median and p95 latency, plus the number of threads, the power plan, and whether the laptop is plugged in.
  - Detector-only FPS and end-to-end FPS are always reported as **separate** numbers.

### Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| ScanNet access denied or delayed (requires an institutional email) | Medium–High | MVP does not depend on it; TUM is the fallback for tracking |
| Poor COCO↔dataset class overlap | Medium | Evaluate only the shared classes; fine-tuning on Kaggle is a later, evidence-driven option |
| CPU too slow for "real-time" | High | Reduce input size (320/416), use INT8, and test OpenVINO EP. Report the bottleneck honestly. |
| Python 3.14 wheel gaps | High | Pin a 3.11/3.12 virtual environment |
| 7.7 GB RAM limits | Medium | Stream frames, downsample point clouds, avoid caching full datasets |
| Bias from volume center vs visible surface | Certain | Document the bias, measure it, and optionally correct it using the class's typical size |
| Camera motion breaks the 2D Kalman model | Medium | Measure ID switches; fall back to 3D tracking or camera-motion compensation |

---

## 7. Compute and storage

| Item | Disk | Notes |
|---|---|---|
| SUN RGB-D (images + toolbox + updated boxes) | ~7 GB archive, budget ~20 GB extracted | Sources disagree on the exact size [To verify] |
| TUM RGB-D (3–5 sequences) | ~3–8 GB | Pick static-scene sequences with objects (e.g. the `fr3` office/desk sequences) [To verify] |
| ScanNet (5–10 val scenes, if granted) | ~10–25 GB | `.sens` files plus 2D instance archives [To verify] |
| Python environment, models, outputs | ~5–8 GB | |
| **Total** | **~40–60 GB** of the 102 GB free | |

**Compute.** All development runs locally on the CPU. **No training happens locally.** Kaggle GPU would be used only for optional fine-tuning or a remote TensorRT comparison, and only if the measurements justify it.

---

## 8. Decisions needed from you

1. **Dataset plan.** Approve the recommendation below, or choose another.
2. **ScanNet.** Do you have an institutional (university or company) email to sign the Terms of Use? If yes, send the request **now**, because it takes days.
3. **Detector license.** Ultralytics (AGPL-3.0, repo becomes AGPL) or YOLOX (Apache-2.0)?
4. **Edge target.** Accept that local benchmarks run on CPU and Intel iGPU, with TensorRT only on Kaggle and clearly labelled? Or drop TensorRT?
5. **Live camera.** Do you own an RGB-D camera (RealSense, Kinect, OAK-D)? If not, the project is offline-only, and that is fine.
6. **Python version.** OK to create a 3.11 or 3.12 virtual environment for the project?
7. **Git.** The folder is not a Git repository yet. OK to `git init` at the start of Phase 01?

---

## Recommendation

**Datasets:**
- **SUN RGB-D** is the primary evaluation dataset. It is ungated, has real sensor data, includes per-image intrinsics, has 3D boxes for ground-truth centers, and is the dataset F-PointNets was evaluated on, which is a good interview anchor.
- **TUM RGB-D** (CC BY 4.0) is for sequences. It provides tracking and stability measurement through its ground-truth camera poses.
- **ScanNet** is an upgrade to pursue in parallel. It adds labelled tracking metrics (IDF1/HOTA) if access is granted.

This combination covers every MVP metric without depending on a gated dataset.

**MVP architecture:** pretrained COCO nano detector → median-in-shrunk-box depth with coverage and MAD flags → pinhole back-projection → ByteTrack carrying 3D histories → ONNX Runtime CPU FP32 and INT8 benchmark.

It is simple and fully explainable. Each stage fails in a way we can name and measure. The F-PointNets comparison gives a principled answer to "what would you do next?": replace the median-in-box with learned frustum segmentation, and back it with our own measured failure cases.
