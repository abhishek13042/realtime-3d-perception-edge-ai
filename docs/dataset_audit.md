# Dataset Audit

**Spec:** 00.3 · **Status:** Drafted · **Last updated:** 2026-11-07

**Download status: NOTHING DOWNLOADED YET.** Downloads start in Phase 01, after approval.

> **[Verified]** means checked against the official page or paper (linked). **[To verify]** means it will be confirmed on the downloaded files in Phase 01. The owner spec for each is given in §6.

---

## 1. Requirements the data must meet (from spec 00.1)

| Need | Required for |
|---|---|
| RGB + **registered** metric depth | Everything |
| Known intrinsics and depth units | Back-projection (G1, G3) |
| 2D box (or mask) labels on COCO-overlapping classes | Detection mAP (G2) |
| 3D object location ground truth | 3D center error (G3) |
| Sequences (video), ideally with poses or identity labels | Tracking (G5) |
| Scene-level splits | No leakage |
| License compatible with a public research repo | Publishing |

---

## 2. Candidates

### 2.1 SUN RGB-D — ✅ SELECTED (primary: detection + 3D)
- **Source:** [rgbd.cs.princeton.edu](https://rgbd.cs.princeton.edu/). Paper: Song, Lichtenberg, Xiao, *SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite*, CVPR 2015.
- **Access:** direct download, no registration [Verified].
- **License:** **no formal license is stated.** Users are asked to cite the paper(s) [Verified]. Treat it as research use only, and do not redistribute.
- **Content:**
  - 10,335 RGB-D images from 4 sensors: Kinect v1, Kinect v2, RealSense, Xtion. Part of NYU V2 is included.
  - 146,617 2D polygons.
  - **64,595 3D oriented boxes** [Verified, paper].
- **Files:** `SUNRGBD.zip` (images), `SUNRGBDtoolbox.zip` (annotations and MATLAB code), `SUNRGBDMeta2DBB_v2.mat`, `SUNRGBDMeta3DBB_v2.mat` (updated boxes) [Verified].
- **Depth encoding:** 16-bit PNG, **bit-rotated**. Decoding is `d = ((raw >> 3) | (raw << 13)) / 1000` → metres.
  - This was confirmed in two independent implementations: [cooperative_scene_parsing parser](https://github.com/thusiyuan/cooperative_scene_parsing/blob/8605cf02aaba3fd2af3eb7e285bcea9966158984/preprocess/sunrgbd/sunrgbd_parser.py) and an [Open3D discussion](https://github.com/isl-org/Open3D/pull/42) citing the SUN3D reader.
  - Open3D also has `create_from_sun_format`.
  - Status: **[To verify] on our files.**
- **Intrinsics:** a per-image `intrinsics.txt` (K) [To verify: file name and layout].
- **Extrinsics:** per image, `Rtilt` rotates the camera frame into a **gravity-aligned upright frame**. The 3D boxes are stored in that upright frame, which uses a different axis convention (x right, y forward, z up) [To verify]. We must convert boxes to the OpenCV camera frame. **This is the main conversion risk for G3.**
- **Splits:** the toolbox has an official train/test split, commonly cited as 5,285 train / 5,050 test [To verify in the toolbox `allsplit.mat`].
- **Size:** about 6.5–7.2 GB for the archive; sources disagree [To verify]. Budget about 20 GB once extracted.
- **Fit:** ✅ G2 (2D boxes) · ✅ G3 (3D box centers) · ❌ G5 (single frames, no sequences).

### 2.2 TUM RGB-D — ✅ SELECTED (sequences: tracking + stability)
- **Source:** [cvg.cit.tum.de/data/datasets/rgbd-dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset). Paper: Sturm et al., *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012.
- **Access:** direct download [Verified].
- **License:** **CC BY 4.0** (data); the code is BSD-2-Clause [Verified]. This is the clearest license of all the candidates.
- **Sensor:** Microsoft Kinect, 640×480 [Verified].
- **Depth:** 16-bit PNG, **scaled by 5000** (5000 = 1 m), and **0 = no data**. Depth is **pre-registered to the RGB image** [Verified, [file formats page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)].
- **Intrinsics** [Verified, file formats page]:

  | Set | fx | fy | cx | cy |
  |---|---|---|---|---|
  | fr1 | 517.3 | 516.5 | 318.6 | 255.3 |
  | fr2 | 520.9 | 521.0 | 325.1 | 249.7 |
  | fr3 | 535.4 | 539.2 | 320.1 | 247.6 |

  - fr3 images are **already undistorted** [Verified].
  - For registered depth, TUM itself recommends the ROS default (fx = fy = 525.0, cx = 319.5, cy = 239.5). **Decision for Phase 01:** test both and document which one we use.
- **Ground truth:** motion-capture camera poses, `timestamp tx ty tz qx qy qz qw` [Verified]. **There are no object annotations.**
- **Synchronization:** RGB and depth timestamps differ slightly, so they must be matched by nearest timestamp (TUM's `associate.py` approach) [To verify tolerance].
- **Candidate sequences:** fr3 office/desk-type sequences with furniture and monitors [To verify which sequences contain enough COCO objects].
- **Size:** about 0.5–2 GB per sequence [To verify].
- **Fit:** ❌ G2/G3 (no labels) · ✅ G5 via the label-free world-frame jitter metric (static scene + GT poses).

### 2.3 ScanNet v2 — ⭐ OPTIONAL UPGRADE (labelled tracking)
- **Source:** [github.com/ScanNet/ScanNet](https://github.com/ScanNet/ScanNet). Paper: Dai et al., CVPR 2017.
- **Access:** **gated.** You sign the Terms of Use **with an institutional email** and send it to the team by email. If there is no reply within a week, check the bounce. [Verified]
- **License:** ScanNet Terms of Use (non-commercial research); the code is MIT [Verified].
- **Content:**
  - 1,500+ scans, 2.5M frames [Verified].
  - `.sens` per scan (color, depth, poses, intrinsics) [Verified].
  - 3D instance annotations [Verified].
  - **Per-frame 2D label and instance PNG projections** [Verified]. These give per-frame object IDs, so **real IDF1/HOTA** can be computed.
- **Depth vs RGB resolution:** 640×480 vs 1296×968, so registration is needed [To verify].
- **Splits:** official train/val/test lists (commonly 1201/312/100 scans) [To verify].
- **Size:** the full dataset is TB-scale. We would take only 5–10 val scenes, about 10–25 GB [To verify].
- **Fit:** ✅ G2 · ✅ G3 · ✅ G5 with labels. **But it is not required by the MVP.**

### 2.4 NYU Depth V2 — ❌ NOT SELECTED (overlaps SUN RGB-D)
- **Source:** [cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).
- **Access:** direct [Verified].
- **License:** none stated; cite the ECCV 2012 paper [Verified].
- **Content:**
  - 1,449 labelled frames in a 2.8 GB `.mat` file, with class and instance masks.
  - A 428 GB unlabelled raw set [Verified].
- **Trap:** the labelled `depths` are **inpainted**. Evaluation must use `rawDepths` [Verified that both exist].
- **Why not:** SUN RGB-D already contains NYU frames *and* adds 3D boxes.

### 2.5 Hypersim — ❌ NOT SELECTED (fallback only)
- **Source:** [github.com/apple/ml-hypersim](https://github.com/apple/ml-hypersim).
- **License:** CC BY-SA 3.0. Ground-truth meshes and source assets must be bought [Verified].
- **Content:** 74,619 synthetic images, 461 scenes, 9-DOF 3D boxes, instance masks [Verified].
- **Traps:**
  - `depth_meters` is **Euclidean ray distance, not planar Z** [Verified].
  - Some scenes use tilt-shift intrinsics [Verified].
- **Why not:** it is about 1.9 TB, it is synthetic (so there is a domain gap to real sensors), and it has no real sensor noise, which is exactly what G4 must handle.

---

## 3. Goal coverage matrix

| Goal | SUN RGB-D | TUM RGB-D | ScanNet (optional) |
|---|---|---|---|
| G1 geometry | — (synthetic tests) | — | — |
| G2 detection mAP | ✅ **primary** | ❌ | ✅ |
| G3 3D center error | ✅ **primary** | ❌ | ✅ |
| G4 invalid depth | ✅ real sensor holes | ✅ | ✅ |
| G5 tracking | ❌ | ✅ **label-free jitter** | ✅ IDF1/HOTA |
| G6 benchmark | ✅ fixed subset | ✅ sequence | — |

**No MVP gap. The MVP needs no gated data.** ✅

---

## 4. Class mapping, COCO → SUN RGB-D [To verify in spec 02.3]
SUN RGB-D's common 3D benchmark uses 10 classes. Only some of them have a COCO equivalent:

| SUN RGB-D | COCO | Use |
|---|---|---|
| chair | chair | ✅ |
| sofa | couch | ✅ |
| bed | bed | ✅ |
| toilet | toilet | ✅ |
| table | dining table | ⚠️ partial semantic match; reported separately |
| desk, dresser, night_stand, bookshelf, bathtub | — | ❌ excluded and disclosed |

Classes outside this 10-class set may add more overlap, such as tv/monitor, sink, and refrigerator [To verify the label names in the toolbox].

---

## 5. Repository data policy
- **Never committed:** dataset files, extracted frames, model weights, point-cloud dumps, generated videos.
- **Committed:** download scripts, checksums, split-list files *we* generate (scene IDs only), and small figures (<500 KB each) with a source credit.
- **ScanNet:** nothing derived from it is published beyond aggregate metrics. Non-commercial use only.
- **Citations:** each dataset's paper is cited in the README.
- **Data root:** configured in `configs/dataset.yaml`, outside the repo (e.g. `D:/datasets/`).

---

## 6. Open items handed to Phase 01

| Item | Owner spec |
|---|---|
| SUN RGB-D archive sizes and file layout | 01.1 |
| SUN RGB-D depth decoding matches our reader (e.g. median depth of a known wall ≈ plausible) | 01.2 |
| SUN RGB-D intrinsics file format | 01.2 |
| SUN RGB-D `Rtilt` and upright-frame axis convention → OpenCV camera frame | 01.3 |
| SUN RGB-D official split counts | 01.3 |
| TUM intrinsics choice (fr3 calibrated vs ROS default) | 01.2 |
| TUM RGB↔depth timestamp association tolerance | 01.2 |
| TUM sequence selection (enough COCO objects) | 01.1 |
| COCO ↔ SUN RGB-D class mapping | 02.3 |
| ScanNet access outcome | User |
