# Data Layout (spec 01.1)

Data lives **outside the repo** at `data_root` (set in [configs/dataset.yaml](../configs/dataset.yaml), default `D:/datasets`). Nothing under it is committed.

```
D:/datasets/
├── _manifest_lock.json          # url, bytes, sha256, UTC time per downloaded entry
├── _archives/                   # raw downloads (.tgz/.zip/.mat), kept so extraction can be redone
├── tum/
│   ├── rgbd_dataset_freiburg1_desk/
│   │   ├── rgb/  depth/  rgb.txt  depth.txt  groundtruth.txt  accelerometer.txt
│   └── rgbd_dataset_freiburg3_long_office_household/
└── sunrgbd/
    ├── SUNRGBDMeta3DBB_v2.mat
    ├── SUNRGBDMeta2DBB_v2.mat
    ├── SUNRGBDtoolbox/          # traintestSUNRGBD/allsplit.mat = official split
    ├── SUNRGBD/                 # kv1/ kv2/ realsense/ xtion/ → per-image dirs
    ├── _cache/annotations_v1.json   # parsed boxes (built by scripts/build_sun_annotations.py)
    └── __MACOSX/                # junk from the zip; unused
```

## SUN RGB-D files (inspected 2026-11-08)

**Per-image folder** (`SUNRGBD/<sensor>/.../<sequenceName>/`): `image/` (RGB jpg), `depth/` (raw 16-bit png, bit-rotated mm; this is what we use), `depth_bfx/` (hole-filled; not used), `fullres/`, `intrinsics.txt`, `extrinsics/`, `annotation*/index.json`, `seg.mat`, `scene.txt` (room type). Each `sequenceName` folder holds **one** image.

**`SUNRGBDMeta3DBB_v2.mat` → `SUNRGBDMeta`** (10,335 records, same order as the 2D file):

| Field | Meaning |
|---|---|
| `sequenceName` | Folder relative to the data root; used as `scene_id` |
| `rgbpath`, `depthpath` | Absolute paths with the prefix `/n/fs/sun3d/data/`, which we strip |
| `K` | 3×3 intrinsics for this image |
| `Rtilt` | 3×3; `p_upright = Rtilt · (x, z, −y)` of an OpenCV camera point |
| `sensorType` | kv1 / kv2 / realsense / xtion |
| `groundtruth3DBB[]` | `centroid` (upright frame: x right, y forward, z up), `basis` (rows = box axes), `coeffs` (half sizes, m), `orientation`, `classname`, `label` |

**`SUNRGBDMeta2DBB_v2.mat` → `SUNRGBDMeta2DBB`**: `groundtruth2DBB[]` with `gtBb2D` = MATLAB `[x, y, w, h]` (1-based; we convert it to 0-based `[x1, y1, x2, y2]`), `classname`, `objid`, `has3dbox`.

**`traintestSUNRGBD/allsplit.mat`**: `alltrain` (5,285), `alltest` (5,050), and `trainvalsplit.{train, val}` (2,666 / 2,619). We use `alltrain`/`alltest`, and carve our own smaller val; see [annotations.md](annotations.md).

## Conventions (used by every reader and every later phase)
| Item | Convention |
|---|---|
| Camera frame | OpenCV: +X right, +Y down, +Z forward; metres |
| Depth | Planar Z in metres (float32); `0.0` where invalid; `valid` = finite and within `[min_m, max_m]` (0.1–10 m) |
| RGB | uint8, RGB channel order (readers convert from OpenCV's BGR) |
| Intrinsics | Per frame `K` (3×3); TUM uses calibrated per-camera values; SUN uses the per-image `K` from the metadata |
| Pose | `T_world_cam` 4×4 camera-to-world (TUM only) |
| SUN 3D boxes | Stored in the upright frame; converted with `p_cam = U · Rtiltᵀ · p_up` ([annotations.md](annotations.md)) |
| SUN 2D boxes | 0-based `[x1, y1, x2, y2]` pixels |
| Image ids | SUN: `f"{index+1:06d}"`; TUM: RGB timestamp `f"{t:.6f}"` |
| Splits | `configs/splits/sunrgbd_{train,val,test}.txt`; benchmark lists in `configs/benchmark_frames/` |

## Usage

```bash
python scripts/download_data.py --dry-run
```

```bash
python scripts/download_data.py tum_fr1_desk
```

Downloads resume from a `.part` file using an HTTP Range request. Extraction refuses archives with absolute paths, `..` components, drive letters, or tar links.

## Reproducibility
`_manifest_lock.json` records the SHA-256 of every archive. Anyone re-downloading can compare hashes to confirm they have the same bytes. The upstream sites publish no checksums, so the first download becomes the reference.
