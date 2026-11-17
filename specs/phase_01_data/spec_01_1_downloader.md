# Spec 01.1 — Dataset Downloader / Importer and Directory Conventions

| Field | Value |
|---|---|
| Phase | 01 — Data |
| Status | ✅ Accepted 2026-11-09 (all 6 files downloaded, SHA-256 locked, extracted; private Kaggle mirrors) |
| Depends on | 00.3 (dataset selection: SUN RGB-D + TUM) |
| Blocks | 01.2, 01.3, 01.4 |

## Objective
Fetch exactly the approved files, record what was fetched, and place them in a fixed layout **outside the Git repo**. Every later script finds the data through one config file.

## Requirements
- **R1.** The data root is configurable in `configs/dataset.yaml`. The default is `D:/datasets`, outside the repo.
- **R2.** A manifest lists each file's URL, expected size, and destination. Only manifest entries can be downloaded.
- **R3.** Downloads are resumable (HTTP Range) and skipped if already complete.
- **R4.** A SHA-256 is computed after each download and stored in `data_root/_manifest_lock.json`. That file is the reproducibility record.
- **R5.** Archives are extracted into `data_root/<dataset>/`, with a guard against path traversal (no `..` and no absolute paths in members).
- **R6.** `--dry-run` prints what would be downloaded and the total size, without downloading.
- **R7.** Nothing under the data root is ever committed (`.gitignore` also blocks common data extensions inside the repo).

## Manifest (sizes measured with HTTP HEAD on 2026-11-07)

| Key | File | Size | Purpose |
|---|---|---|---|
| `tum_fr1_desk` | rgbd_dataset_freiburg1_desk.tgz | 328 MB | Short tracking sequence (desk, monitors) |
| `tum_fr3_long_office` | rgbd_dataset_freiburg3_long_office_household.tgz | 1,415 MB | Longer static office scene; fr3 images are already undistorted |
| `sunrgbd_meta3d` | SUNRGBDMeta3DBB_v2.mat | 9 MB | 3D boxes, K, Rtilt per image |
| `sunrgbd_meta2d` | SUNRGBDMeta2DBB_v2.mat | 4 MB | 2D boxes |
| `sunrgbd_toolbox` | SUNRGBDtoolbox.zip | 544 MB | Official splits and reference code |
| `sunrgbd_images` | SUNRGBD.zip | 6,567 MB | RGB + depth images |

**Total: about 8.9 GB.**

## Tasks
- [ ] T1. `configs/dataset.yaml` with the data root and the manifest.
- [ ] T2. `scripts/download_data.py` (dry-run, resume, checksum, extract, traversal guard).
- [ ] T3. `docs/data_layout.md` describing the on-disk layout after extraction.
- [ ] T4. Download with **user approval** (small TUM sequence first, then SUN RGB-D).
- [ ] T5. Record the checksums in the lock file.

## Acceptance criteria
- **AC1.** `--dry-run` lists the files and the total size.
- **AC2.** Re-running after completion downloads nothing.
- **AC3.** The lock file contains the SHA-256 and byte size for every downloaded file.
- **AC4.** `git status` shows no data files.

## Tests
- `test_download.py::test_safe_extract_rejects_traversal`: a crafted tar or zip with a `../evil` member is rejected.
- `test_download.py::test_manifest_keys_unique`.
- `test_download.py::test_dry_run_makes_no_files`.

## Dependencies
Python standard library only (`urllib`, `tarfile`, `zipfile`, `hashlib`), plus PyYAML.
