# Phase 01 — Data Foundation and Geometry

**Exit condition:** verified samples load correctly, and the coordinate conventions are documented.

| Spec | File | Main outputs |
|---|---|---|
| 01.1 | [spec_01_1_downloader.md](spec_01_1_downloader.md) | `scripts/download_data.py`, `configs/dataset.yaml`, `docs/data_layout.md` |
| 01.2 | [spec_01_2_calibration_alignment.md](spec_01_2_calibration_alignment.md) | `perception/data/{frame,tum,sunrgbd}.py`, `perception/depth/decode.py`, `perception/geometry/camera.py` |
| 01.3 | [spec_01_3_annotations_splits.md](spec_01_3_annotations_splits.md) | `perception/data/annotations.py`, `perception/data/splits.py`, `perception/geometry/transforms.py` |
| 01.4 | [spec_01_4_quality_report.md](spec_01_4_quality_report.md) | `scripts/inspect_dataset.py`, `benchmarks/data_quality/*.json`, `docs/data_quality_report.md` |

**Template:** Objective · Requirements · Tasks · Acceptance criteria · Tests · Dependencies.
