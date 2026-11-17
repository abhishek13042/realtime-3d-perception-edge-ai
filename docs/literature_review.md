# Literature Review

**Spec:** 00.2 · **Status:** Drafted · **Last updated:** 2026-11-07

Each entry has a **role** in this project:
- **IMPLEMENT** — we write this ourselves.
- **INTEGRATE** — we use the authors' or an open implementation.
- **REFERENCE** — it shapes our design, but we do not implement it.
- **EVALUATION** — we use it to measure results.

---

## 0. Pinhole camera model and back-projection — IMPLEMENT
**Sources:** R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2004, ch. 6. [OpenCV calib3d documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

**Projection.** A 3D point in the camera frame, (X, Y, Z), lands on pixel (u, v):

```
u = fx · X / Z + cx
v = fy · Y / Z + cy        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
```

**Back-projection.** This is the inverse, and it needs the depth Z:

```
X = (u − cx) · Z / fx
Y = (v − cy) · Z / fy
Z = Z
```

**What the symbols mean**
- `fx`, `fy` are the focal lengths in *pixels*: the focal length in mm divided by the pixel size.
- `(cx, cy)` is the principal point, where the optical axis hits the image.
- Without Z, a pixel defines only a **ray**, not a point. This is why RGB alone cannot give metric 3D.

**Traps we must handle**
1. **Planar Z vs ray distance.** Some data, such as Hypersim, stores the distance along the ray, `d`, rather than Z. The conversion is `Z = d · f / √((u−cx)² + (v−cy)² + f²)`.
2. **Registration.** Depth and RGB come from different sensors. The depth must be warped onto the RGB pixel grid before (u, v) means the same pixel in both images.
3. **Resized images.** If an image is resized by a factor s, then fx, fy, cx and cy also scale by s.
4. **Frame conventions.** OpenCV uses +Y down. Many 3D tools use +Y or +Z up. Mixing these silently flips results.

**Limitation.** The model ignores lens distortion. That is acceptable for undistorted dataset images (TUM fr3 is already undistorted [Verified]).

---

## 1. PointNet — REFERENCE
**Citation.** C. R. Qi, H. Su, K. Mo, L. J. Guibas. *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* CVPR 2017. [arXiv:1612.00593](https://arxiv.org/abs/1612.00593)

**Problem.** A point cloud is an **unordered set**. A CNN expects an ordered grid, and voxelizing a cloud to make one wastes memory.

**Core idea**
- Apply the same small MLP to each point on its own, producing a per-point feature h(xᵢ).
- Combine all points with a **symmetric function**, max-pooling: `f({x₁…xₙ}) ≈ γ( max_i h(xᵢ) )`.
- Because max does not care about order, the network is **permutation invariant**.
- A small "T-Net" predicts an alignment transform, for some robustness to rotation.

**Key insight.** The max-pool keeps a small "critical set" of points. This makes the network robust to missing points, but it **ignores local neighbourhood structure**.

**Limitation.** It captures no local geometric context. That motivates PointNet++.

**Relevance.** It explains why point clouds need different networks from images, and it is the building block of F-PointNets. **We do not implement it.**

---

## 2. PointNet++ — REFERENCE
**Citation.** C. R. Qi, L. Yi, H. Su, L. J. Guibas. *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.* NeurIPS 2017. [arXiv:1706.02413](https://arxiv.org/abs/1706.02413)

**Core idea.** Apply PointNet hierarchically, the way a CNN applies filters to local patches. Each "set abstraction" layer does three things:
1. **Sampling.** Farthest point sampling (FPS) picks well-spread centroids.
2. **Grouping.** A ball query collects the neighbours within a radius r of each centroid.
3. **Feature extraction.** A mini-PointNet runs on each local group.

**Density problem.** RGB-D clouds are **dense near the camera and sparse far away**. PointNet++ handles this with multi-scale and multi-resolution grouping.

**Relevance.** It explains why far objects are harder: there are fewer points per object. **We will measure this effect directly** in Phase 05 by bucketing errors by distance. **We do not implement it.**

---

## 3. Frustum PointNets — REFERENCE (closest relative of our pipeline)
**Citation.** C. R. Qi, W. Liu, C. Wu, H. Su, L. J. Guibas. *Frustum PointNets for 3D Object Detection from RGB-D Data.* CVPR 2018. [arXiv:1711.08488](https://arxiv.org/abs/1711.08488). It is evaluated on KITTI and **SUN RGB-D**.

**Pipeline**
1. A **2D detector** proposes boxes in the RGB image.
2. Each box is lifted into a **3D frustum**: all points whose projection falls inside the box.
3. **3D instance segmentation**, using PointNet, separates object points from clutter and background inside the frustum.
4. **Amodal 3D box estimation** regresses the full box, including the parts you cannot see.

**Key insight.** Mature 2D detectors reduce the 3D search space, so the 3D network only has to look inside a frustum.

**Relation to our design**

| F-PointNets step | Our MVP | Gap |
|---|---|---|
| 2D detector | Pretrained YOLO | Same idea |
| Frustum from the box | Depth pixels inside the box | Same idea |
| Learned 3D segmentation | **Median of the depth in a shrunken box** (no learning) | We may include background and occluders |
| Amodal 3D box | Center on the **visible surface** only | We are biased toward the camera by about half the object's depth |

**What this means for us**
- Our method is a **non-learned baseline of the frustum idea**.
- Its failure modes (background bleed, occlusion, surface-vs-center bias) are exactly what step 3 and step 4 of the paper exist to solve.
- This is our principled "next step".
- **We do not implement F-PointNets** unless it becomes a justified extension.

---

## 4. SORT — REFERENCE (background for ByteTrack)
**Citation.** A. Bewley, Z. Ge, L. Ott, F. Ramos, B. Upcroft. *Simple Online and Realtime Tracking.* ICIP 2016. [arXiv:1602.00763](https://arxiv.org/abs/1602.00763)

**Core idea: tracking-by-detection**
- Each track has a **Kalman filter** over its box state (center, scale, aspect ratio, and their velocities) with a constant-velocity motion model.
- Each frame, the filter **predicts** where every track's box will be.
- The predictions are **matched** to new detections with the **Hungarian algorithm** on an IoU cost.
- Unmatched detections start new tracks. Tracks unmatched for too long are deleted.

**Limitations**
- The constant-velocity model breaks under **camera motion**. This is relevant to us, because TUM sequences use a hand-held camera.
- There is no re-identification after a long occlusion.
- Low-score detections are thrown away.

---

## 5. ByteTrack — INTEGRATE
**Citation.** Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, X. Wang. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022. [arXiv:2110.06864](https://arxiv.org/abs/2110.06864). The reference code is MIT-licensed.

**Core idea: associate *every* box, in two stages**
1. Split the detections into **high-score** and **low-score** sets.
2. **First association:** match high-score boxes to all tracks (Kalman prediction plus IoU and Hungarian).
3. **Second association:** match the *low-score* boxes to the tracks left unmatched. Low scores often mean an occluded but real object, and IoU with a predicted track filters out background false positives.
4. Unmatched high-score boxes start new tracks. Lost tracks are kept for a buffer period before deletion.

**Why it fits.** It needs no appearance or re-ID network, so it costs very little CPU, which matters on our hardware. It also works with any detector that outputs boxes and scores.

**Our use**
- We integrate it as the 2D tracker. Each track carries the history of its 3D centers.
- We tune its thresholds on a held-out sequence and evaluate them on a different one.

---

## 6. HOTA — EVALUATION
**Citation.** J. Luiten, A. Ošep, P. Dendorfer, P. Torr, A. Geiger, L. Leal-Taixé, B. Leibe. *HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking.* IJCV 2021. [arXiv:2009.07736](https://arxiv.org/abs/2009.07736)

**Core idea.** `HOTA = √(DetA · AssA)`, averaged over localization thresholds α. It balances two things:
- **detection accuracy** (did you find the objects?), and
- **association accuracy** (did you keep their identities?).

**Comparison with older metrics**
- **MOTA** is dominated by detection errors.
- **IDF1** is dominated by association errors.

**Our use.** We use HOTA and IDF1 via [TrackEval](https://github.com/JonathonLuiten/TrackEval) **only where per-frame identity labels exist**, i.e. ScanNet. TUM has no object labels, so there we use a label-free world-frame jitter metric (see `evaluation_protocol.md`) and report clearly that it is a different kind of metric.

---

## 7. Baseline comparison

| Task | Our MVP choice | Published alternative | Why the MVP choice first |
|---|---|---|---|
| 2D detection | Pretrained Ultralytics YOLO26-n (or YOLOX) on COCO | Fine-tuned detector; DETR family | No training, CPU-friendly, mature ONNX export |
| 3D localization | Median depth in a shrunken box → back-project | F-PointNets; VoteNet (Qi et al., ICCV 2019, [arXiv:1904.09664](https://arxiv.org/abs/1904.09664)) | Explainable, needs no GPU, and its failures can be measured |
| Depth sampling | Box-center region; nearest-mode variant | Instance-mask sampling (segmentation model) | The mask costs extra latency; we add it only if the measured error justifies it |
| Tracking | ByteTrack (2D IoU + Kalman) | 3D Kalman tracking (e.g. AB3DMOT, Weng et al., IROS 2020, [arXiv:1907.03961](https://arxiv.org/abs/1907.03961)) | 2D tracking still works when depth is missing |
| Tracking metric | HOTA / IDF1 (ScanNet), world jitter (TUM) | MOTA | MOTA hides association errors |

---

## 8. Reading order and self-check

| # | Paper | Depth | Self-check question |
|---|---|---|---|
| 1 | Pinhole model | **Full**, be able to derive it | If an image is resized to half width, what happens to fx and cx? |
| 2 | PointNet | Skim | Why does max-pooling make the network permutation invariant? |
| 3 | PointNet++ | Skim | Why are far objects harder for point-based methods? |
| 4 | F-PointNets | **Careful** | Which F-PointNets step fixes our background-bleed failure, and how? |
| 5 | SORT | Moderate | What do the Kalman filter and the Hungarian algorithm each contribute? |
| 6 | ByteTrack | **Careful** | Why does using low-score boxes reduce ID switches but not add false tracks? |
| 7 | HOTA | Metric definitions | Why can MOTA be high while association is poor? |

User reading notes go in `docs/reading_notes/<paper>.md`, half a page each, in your own words.
