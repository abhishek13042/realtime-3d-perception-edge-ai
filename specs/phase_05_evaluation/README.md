# Phase 05 — Evaluation

**Status:** ✅ Done 2026-11-11, awaiting approval. 05.1 protocol frozen; 05.2 SUN test 3D error by bucket; 05.3 equivalence 573/573 and the timed profile (lowest 6.97 FPS, 10 FPS Target missed, detection is the bottleneck); 05.4 gallery reviewed, summary generated, repro 3 of 4 pass and pipeline_median fails with its measured cause (laptop run-to-run timing spread vs the ±15% band).

**Exit condition (docs/architecture.md §5):**
- the protocol is followed;
- 3D error is reported by distance bucket;
- a stage profile is reported;
- a failure gallery exists;
- the results are re-run within a stated tolerance.

| Spec | File | Main outputs |
|---|---|---|
| 05.1 | [spec_05_1_protocol_freeze.md](spec_05_1_protocol_freeze.md) | final `docs/evaluation_protocol.md` (deviations logged), `scripts/results_index.py`, `benchmarks/results/index.json` |
| 05.2 | [spec_05_2_sun_test_3d.md](spec_05_2_sun_test_3d.md) | `scripts/eval_localization.py`, `benchmarks/results/localization_sun_test.json`, `docs/localization_results.md` (G3, G4) |
| 05.3 | [spec_05_3_pipeline_profile.md](spec_05_3_pipeline_profile.md) | `perception/pipeline.py` (`FrameResult`), `scripts/profile_pipeline.py`, `benchmarks/results/pipeline_profile.json` |
| 05.4 | [spec_05_4_failures_repro.md](spec_05_4_failures_repro.md) | failure gallery, `scripts/repro_check.py`, `docs/results_summary.md` |

**Scope rules for this phase**
- **Test split, once.** SUN RGB-D **test** 3D numbers are computed exactly once, with `configs/depth.yaml` and `configs/detector.yaml` frozen. Nothing is tuned in this phase. If a bug forces a re-run, the reason is logged in `PROGRESS.md` and the run is repeated in full (same rule as 04.2 R7).
- **Stored detections first.** Accuracy numbers use the stored Phase 02 runs (`sun_test_full.jsonl`), so they have no detector re-runs. The pipeline profile (05.3) is the only place the detector runs live, because latency needs it.
- **Measured vs target.** Every number in the docs is labelled Measured; the 10 FPS goal stays a soft Target (problem statement §5) and the bottleneck is reported if it is missed. The measurement boundary is not moved.
- **No training, CPU only.** Optimisation (ONNX, INT8) belongs to Phase 06; this phase is the PyTorch-CPU baseline it is compared against.
- **No new dependencies.**

**Template:** Objective · Requirements · Tasks · Acceptance criteria · Tests · Dependencies.
