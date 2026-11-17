# Spec 01.4 — Dataset Quality Report and Reproducible Sample Loader

| Field | Value |
|---|---|
| Phase | 01 — Data |
| Status | ✅ Accepted 2026-11-09 (AC1–AC4 met 2026-11-08) |
| Depends on | 01.2, 01.3 |
| Blocks | Phase 02 |

## Objective
Measure the data before modelling it. Produce the depth-validity statistics, depth range, class counts, and sensor breakdown that later results will be interpreted against. Provide a deterministic sample loader for development and benchmarks.

## Requirements
- **R1.** `scripts/inspect_dataset.py --dataset {tum,sunrgbd} --n N --seed S` computes, per frame:
  - the valid-depth fraction,
  - the depth percentiles (5/50/95),
  - the image size,
  - and, for SUN RGB-D, the sensor type.
- **R2.** Aggregates: distributions (histograms saved as PNG), per-sensor breakdown (SUN RGB-D), and class counts for the COCO-overlap classes.
- **R3.** The results are written to `benchmarks/data_quality/<dataset>.json`, including the git commit and config hash. `docs/data_quality_report.md` is **generated** from that JSON.
- **R4.** `perception.data.sampling.sample_ids(split, n, seed)` returns a deterministic list of IDs. The benchmark frame lists are saved to `configs/benchmark_frames/`.
- **R5.** The TUM sequence choice (R7) is justified by an object-presence check. This check can only be qualitative until the detector exists in Phase 02.

## Tasks
- [x] T1. Statistics functions (pure, tested).
- [x] T2. Inspection script, with JSON and plot outputs.
- [x] T3. Report generator.
- [x] T4. Deterministic sampler and the saved benchmark lists.
- [x] T5. Run on TUM and SUN RGB-D, and review the report.

## Acceptance criteria
- **AC1.** Running it twice with the same seed gives byte-identical JSON, apart from the timestamp field.
- **AC2.** The report states the valid-depth fraction for each dataset (and per sensor for SUN).
- **AC3.** The report lists any anomalies found (e.g. sensors with unusual depth ranges) and how each one is handled.
- **AC4.** **Phase 01 exit:** all Phase 01 tests pass; the alignment and annotation visual checks are done; conventions are documented in `docs/data_layout.md`.

## Tests (`tests/test_quality.py`)
- The valid fraction of a synthetic mask is exact.
- Percentiles ignore invalid pixels.
- The sampler is deterministic, and different seeds give different lists.

## Dependencies
numpy, matplotlib.

Results: [docs/data_quality_report.md](../../docs/data_quality_report.md).
