# Phase 03 — Depth Association and 3D Geometry

**Status:** ✅ Approved 2026-11-10.

**Exit condition (docs/architecture.md §5):** synthetic tests pass (round trip < 1e-6 m, plane recovery); unit and invalid-depth tests pass; the point cloud viewer works. Also, per architecture §3 [3], the depth thresholds and the shrink factor are **chosen from data** in this phase and recorded in `configs/`.

| Spec | File | Main outputs |
|---|---|---|
| 03.1 | [spec_03_1_depth_association.md](spec_03_1_depth_association.md) | `perception/depth/estimate.py` (box → depth estimate, spread, status) |
| 03.2 | [spec_03_2_object_geometry_pointcloud.md](spec_03_2_object_geometry_pointcloud.md) | `perception/geometry/{object3d,pointcloud}.py`, `scripts/view_pointcloud.py`, fr1 distortion analysis |
| 03.3 | [spec_03_3_depth_tuning_val.md](spec_03_3_depth_tuning_val.md) | `perception/eval/localization.py`, `scripts/tune_depth.py`, `configs/depth.yaml`, `docs/depth_tuning.md` |

**Scope rules for this phase**
- **Val only.** All choices (estimator A vs B, shrink factor, N_min, τ_cov, τ_spread, inlier k) are made on SUN **val** (`configs/splits/sunrgbd_val.txt`, 528 images, scene-disjoint from test). The **test** 3D numbers (G3/G4 headline) belong to Phase 05 and are computed once, with the choices frozen.
- **No training.** Everything here is classical geometry on CPU.
- Raw sensor depth only (no NYU inpainted depth), per `docs/evaluation_protocol.md` §3.
- The stored Phase 02 runs (`benchmarks/detection/sun_val_full.jsonl`, TUM `*_full.jsonl`) are reused. No detector re-runs are needed.
- New dependency: **open3d** (MIT) for the viewer only. If the Python 3.14 wheel fails to install, the viewer falls back to writing `.ply` files plus a matplotlib 3D scatter; core code never imports open3d.

**Template:** Objective · Requirements · Tasks · Acceptance criteria · Tests · Dependencies.
