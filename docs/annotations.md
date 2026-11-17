# SUN RGB-D Annotations and Splits (spec 01.3)

Measured 2026-11-08. Scripts: `scripts/build_sun_annotations.py` and `scripts/visualize_annotations.py`. Raw numbers are in `benchmarks/data_quality/sunrgbd_annotations.json` and `benchmarks/data_quality/annotations_check.json`.

## Upright → camera transform (risk R2)
The SUN RGB-D toolbox (`read3dPoints.m`) builds upright points as `p_up = Rtilt · (x, z, −y)` from OpenCV camera points. The inverse, implemented in `perception.geometry.transforms`, is:

```
p_cam = U · Rtiltᵀ · p_up,   U = [[1,0,0],[0,0,−1],[0,1,0]]   i.e. (a, b, c) → (a, −c, b)
```

Box corners come from `centroid + Σ ±coeffs_i · basis_i` (the rows of `basis` are the box axes).

### Visual check (AC1): passed
We drew 20 random **test** images with 3D boxes (11 kv2, 7 xtion, 1 kv1, 1 realsense), with the wireframes projected through K, in `benchmarks/data_quality/annotations/`.
- Wireframes enclose their objects: sofas, beds, chairs, a sink, a whiteboard, shelves.
- Vertical box edges are vertical in the image, so the tilt is applied in the right direction.
- The 2D boxes (green) sit around the same objects.

### Quantitative check (AC2), with negative controls
We used 600 random test images with ≥ 1 3D box (seed 0). For each box fully in front of the camera, we filled the convex hull of its projected corners and took the median valid depth inside it. The error is `|median depth − box centre Z|`.

| Transform | Boxes used | Median error | Median box Z extent | Boxes with error < extent |
|---|---|---|---|---|
| **Correct: `U · Rtiltᵀ`** | **3,843** | **0.19 m** | 0.97 m | **92.5%** |
| Control: `U · Rtilt` (tilt the wrong way) | 1,361 | 0.53 m | 0.90 m | 65.8% |
| Control: `U` only (tilt ignored) | 3,453 | 0.46 m | 0.75 m | 65.9% |
| Control: no axis swap | 1 | n/a | n/a | n/a (almost every box ends up behind the camera) |

Results for the correct transform, by sensor:

| Sensor | Boxes | Median error | Error < extent |
|---|---|---|---|
| kv1 | 617 | 0.19 m | 93% |
| kv2 | 1,727 | 0.22 m | 91% |
| realsense | 365 | 0.22 m | 93% |
| xtion | 1,134 | 0.15 m | 95% |

**AC2 as written** ("median error < median extent") **passes. However, the wrong-tilt controls pass it too**, so this criterion alone cannot catch a subtly wrong tilt. What separates the correct transform from the controls:
- error / extent ratio: 0.20, against 0.59 and 0.61;
- fraction of boxes within their extent: 92.5%, against 66%;
- number of boxes left in front of the camera: 3,843, against 1,361 when the tilt is applied the wrong way.

**Proposed stricter AC2 (needs user approval):** median error ≤ 0.3 × median extent, **and** the correct transform beats every control on both the ratio and the in-extent fraction. The current results already meet it.

The remaining 0.19 m error is expected and is not a transform bug. The footprint median mixes the object's front surface with the background and any occluders it covers, while the centre Z sits inside the object.

## Splits (AC3)
| Split | Images | Images with 3D boxes | 3D boxes | 2D boxes | 3D classes |
|---|---|---|---|---|---|
| train | 4,757 | 4,738 | 29,370 | 37,603 | 724 |
| val | 528 | 525 | 3,117 | 3,970 | 267 |
| test | 5,050 | 5,028 | 32,170 | 38,647 | 825 |

- **train + val = 5,285 = official `alltrain`; test = 5,050 = official `alltest`.** Every one of the 10,335 images is assigned to exactly one split.
- Val is 10% of the official train, chosen by scene group with seed 0. The overlap check (no shared image id and no shared scene group) passes.
- Files: `configs/splits/sunrgbd_{train,val,test}.txt`, holding 6-digit image ids (= metadata index + 1, the same as `Frame.frame_id`).

**Limitation:** SUN RGB-D has one image per `sequenceName` folder, so "scene group" equals the image. Two different photos of the same room (e.g. consecutive NYU frames in kv1) can still land in train and val. This affects only our val; the official train/test split is used unchanged.

## Class frequencies (AC4)
The full tables are in `benchmarks/data_quality/sunrgbd_annotations.json`: the top 50 classes per split, plus the 10 classes of the standard SUN RGB-D 3D benchmark.

Test-split 3D boxes for the benchmark-10 classes:

| chair | table | desk | sofa | bed | bookshelf | night_stand | dresser | toilet | bathtub |
|---|---|---|---|---|---|---|---|---|---|
| 10,429 | 2,452 | 2,155 | 640 | 545 | 302 | 263 | 223 | 151 | 52 |

The labels are long-tailed: there are 825 raw 3D class names in test, and `chair` alone is about a third of all boxes. A detector evaluated only on the head classes would overstate its quality, so Phase 02 should report per-class metrics.
