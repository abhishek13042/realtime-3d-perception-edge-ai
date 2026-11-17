# Spec 05.1 — Protocol freeze, deviations, results index

| Field | Value |
|---|---|
| Phase | 05 Evaluation |
| Status | ✅ Done 2026-11-11 |

## Objective
`docs/evaluation_protocol.md` was drafted in 00.4 "to be finalized in 05.1". Bring it in line with what Phases 01–04 actually did, log every deviation with its reason, and freeze it **before** the 05.2 test run. Add one index of all result files so the headline numbers can be traced to a file, a commit and a config hash.

## Requirements
- **R1. Deviations, logged.** Each one is written in the protocol's new "Deviations" section and in `PROGRESS.md`, with its reason. Known so far:
  - Tuning used the scene-disjoint **SUN val** split (`configs/splits/sunrgbd_val.txt`, 528 images) rather than "a held-out subset of train" (§1). The effect is the same: test is never seen.
  - 2D↔3D GT linkage uses projected 3D boxes with Hungarian IoU, not the `has3d` order (spec 03.3 R1, evidence in `docs/depth_tuning.md`).
  - TUM tracking: the fragmentation proxy uses a manual inventory, and the ID-switch and recovery proxies come from spec 04.2. The §4 text is replaced by a pointer to it.
  - Result files live in `benchmarks/<area>/` (detection, geometry, tracking) from earlier phases. §7's `benchmarks/results/` is used for the Phase 05 files, and the index (R3) covers both.
- **R2. Open items resolved.**
  - Replace "[To verify the counts]" with the measured split sizes from `configs/splits/`.
  - State the runtime procedure actually used: N = 200 frames, warm-up, repeats (05.3).
- **R3. Results index.** `scripts/results_index.py` walks `benchmarks/**/*.json` and writes `benchmarks/results/index.json`. Each file gets one row: path, spec, label (Measured / Target), git commit, dirty flag, config hashes, split hash (when present), timestamp, and whether it is a headline file. Files missing a §7 field are listed as such, not silently passed.
- **R4. Freeze.** The protocol gets status "Frozen 2026-11-xx" and its sha256 is stored in the index. 05.2–05.4 headers record that sha. Changes after this follow the protocol's own rule.

## Tasks
- [x] T1. Update `docs/evaluation_protocol.md` (R1, R2) and freeze it (R4).
- [x] T2. `scripts/results_index.py` (R3).
- [x] T3. Tests.

## Acceptance criteria
- **AC1.** Every deviation in R1 appears in the protocol and in `PROGRESS.md`.
- **AC2.** The index lists every JSON result file. Every headline number in `docs/*.md` points to a file in the index.
- **AC3.** The frozen protocol sha is in the index and is checked by 05.2.

## Tests (`tests/test_results_index.py`)
- A JSON file with all §7 fields gets a complete row. A file missing the commit is flagged.
- The protocol sha check fails when the protocol text changes.

## Dependencies
Phases 01–04 outputs.

## Results (2026-11-11)
- **T1 (R1, R2, R4).** `docs/evaluation_protocol.md` is updated and frozen.
  - §1: measured split sizes (test 5,050; official train 5,285 → train 4,757 + val 528).
  - §5: the N = 200 / 20 warm-up / 3 fresh-process procedure.
  - New §8 logs deviations D1–D5. New §9 lists the six frozen inputs with sha256 (`depth.yaml`, `detector.yaml`, `tracker.yaml`, `class_map.yaml`, `sunrgbd_test.txt`, `sunrgbd_val.txt`).
  - Frozen protocol sha256 `58e3b4f7b07e…dde60`, recorded in `benchmarks/results/index.json` (frozen 2026-11-11T08:00:08Z).
- **T2 (R3).** `scripts/results_index.py` and `perception/eval/results_index.py`.
  - The index lists **34** JSON result files. **31** have at least one missing §7 field. They are flagged, not filled in after the fact:

    | Group | Files | Missing |
    |---|---|---|
    | Phase 01 data-quality, Phase 03 geometry/viewer, detection side checks | 16 | commit, config/input sha, timestamp |
    | `detector_check.json` | 1 | commit, timestamp |
    | Detection eval/latency, `sunrgbd.json`, `tum.json` | 6 | timestamp |
    | Tracking `*_summary.json` | 4 | config sha |
    | Tracking `*_timing.json` | 4 | commit, config sha |

  - Only the three tracking eval reports are complete. No commit exists yet, so `git_commit` is null wherever it is recorded.
  - Phase 05 files must be complete (05.4 repro check verifies this).
- **T3.** `tests/test_results_index.py`: 6 tests pass.
- **AC1** met: D1–D5 are in protocol §8 and in `PROGRESS.md`.
- **AC2**: every JSON file is indexed. Headline links from `docs/*.md` are checked when `docs/results_summary.md` is generated (05.4). Today the headline files are the detection baseline and the fr3 report, and both are in the index.
- **AC3** met: the sha is in the index. `check_protocol` and `check_frozen` are called by 05.2–05.4.
