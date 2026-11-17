# Spec 03.1 — Depth Association (box → depth, spread, status)

| Field | Value |
|---|---|
| Phase | 03 — Geometry |
| Status | ✅ Done 2026-11-10 (R9 met on real frames; 50-box bound not met, see Results — bound restated 2026-11-11) |
| Depends on | 01.2 (`Frame`, depth decode, valid mask), 02.1 (`Detections`, boxes in original px) |
| Blocks | 03.2, 03.3, 04.x (3D centres attached to tracks) |

## Objective
Given a frame's metric depth map, its valid mask, and one 2D box, return a single depth estimate for the object, a robust measure of how mixed the depths inside the box are, and a status that says whether the estimate can be trusted. This is the core heuristic of the system (architecture §3 [3]).

## Requirements
- **R1. Sampling region.** Shrink the box about its centre by factor `s` (width and height both × s; default 0.5, tuned in 03.3). Convert to integer pixel ranges with one documented rule (`floor(x1)…ceil(x2)-1`, clipped to the image). A box that becomes empty after clipping gives `no_depth`.
- **R2. Valid pixels only.** Use only pixels where `valid` is True. Never fill, interpolate, or guess missing depth.
- **R3. Estimator A (baseline):** the median of the valid depths.
- **R4. Estimator B (foreground mode):** histogram the valid depths with a fixed bin width (default 5 cm, configurable). A bin is *significant* if it holds ≥ `m_frac` of the valid pixels (default 0.15). Take the **nearest** significant bin, then return the median of the valid depths within ±1 bin of it. If no bin is significant, fall back to estimator A and record `fallback=True`.
- **R5. Spread:** `1.4826 · MAD` of the valid depths (MAD about the median), in metres.
- **R6. Status:**
  - `no_depth` if valid pixel count < `N_min`;
  - `low_coverage` if valid fraction (valid / pixels in region) < `τ_cov` **or** spread > `τ_spread`;
  - `valid` otherwise.
  Defaults: `N_min = 20`, `τ_cov = 0.3`, `τ_spread = 0.25 m`. These are placeholders labelled **[Target, to be set in 03.3]**.
- **R7. Output:** a `DepthEstimate` dataclass per box: `z` (m, NaN if `no_depth`), `estimator`, `spread`, `n_valid`, `valid_frac`, `status`, `fallback`, and the integer region used. Vectorized over all boxes of a frame where practical.
- **R8. Config.** All parameters live in a new `configs/depth.yaml` and are passed in explicitly (no hidden globals), so 03.3 can sweep them.
- **R9. Cost.** The association for one frame with ≤ 50 boxes takes < 5 ms on the dev CPU (measured, median over the 200-frame TUM fr1 list). This keeps it negligible next to the ~150–180 ms detector.

## Tasks
- [x] T1. `perception/depth/estimate.py`: region, estimators A and B, spread, status, `DepthEstimate`.
- [x] T2. `configs/depth.yaml` with the placeholder defaults, each marked as a target.
- [x] T3. Unit tests (below).
- [x] T4. Timing measurement on the TUM fr1 200-frame list, using the stored detections.

## Acceptance criteria
- **AC1.** All tests below pass.
- **AC2.** On a synthetic frame with a 1.0 m object covering 60% of the box over a 3.0 m background, estimator B returns 1.0 m ± 1 bin and estimator A is reported as is (whatever it is). The test documents the case where A fails, so the difference is shown, not assumed.
- **AC3.** R9 timing measured and recorded with machine and thread labels.

## Tests (`tests/test_depth_estimate.py`)
- Constant plane at 2.0 m → A = B = 2.0, spread 0, `valid`.
- All pixels invalid → `no_depth`, z NaN; too few valid pixels → `no_depth`.
- 50% invalid with a uniform plane → `low_coverage` only when `τ_cov` > 0.5.
- Foreground/background mix (AC2), including a case with background as the majority, where B must still pick the nearer significant mode.
- Noise-only bins below `m_frac` are ignored (a few flying pixels at 0.4 m do not become the "nearest mode").
- No significant bin → fallback to A, `fallback=True`.
- Box partly outside the image → clipped; box fully outside → `no_depth`.
- Region rule: a box [10, 10, 20, 20] with s = 1 samples exactly rows/cols 10…19.

## Dependencies
numpy; 01.2 decode; 02.1 detections.

## Results (2026-11-10)

**AC1.** `tests/test_depth_estimate.py`: 12 tests pass (full suite 103 pass).

**AC2.** With a 60/40 object/background mix, B = 1.0 m ± 1 bin and A also lands on the object, because the object is the majority. With the background as the majority (30/70), B = 1.0 m and A = 3.0 m. That is the documented failure of A, and the test shows it.

**AC3 / R9.** Measured on an i3-1115G4 (2C/4T), OpenBLAS 4 threads, using numpy CPU. Decode is excluded. There were 10 warm-up frames, then the 200 TUM fr1 benchmark frames. Boxes are the stored detections at `conf ≥ conf_operational` (0.2411). A Kaggle upload was running in the background during the measurement, and across runs the medians varied by up to about 1.6×. Source: `benchmarks/geometry/geometry_timing.json` (`scripts/time_geometry.py`).

| Stage | Median (ms) | p95 (ms) | Target | |
|---|---|---|---|---|
| Association, real frames (median 4 boxes, max 11) | 1.28 | 2.66 | < 5 | Met |
| Association, stress: 50 real boxes on every frame | 10.75 | 24.08 | < 5 | **Not met** |

- **Per-box cost** is about 0.2–0.4 ms. It is mostly fixed numpy call overhead (several medians per box), not pixel count. A capped pixel sample (at most 1,000–2,000 px per box) was measured. It cut the 50-box time by only about 20% and changed 14–25 of 936 statuses, so it was not adopted.
- **Kept speedups.** These give identical results: a bincount histogram replaces `np.unique`, and one median is shared by the spread, estimator A and the fallback.
- **Decision pending.** One option is to restate R9 as "real frames at `conf_operational`, with per-box cost recorded". On fr1 the maximum is 11 boxes, which comes to about 4 ms. The other is to keep the 50-box bound and vectorise across boxes in Phase 06. Either way, at 50 boxes it is about 7% of the ~150–180 ms detector.

**Note: MAD breakdown limit (found by the AC2 test).** The spread is MAD-based, and MAD has a 50% breakdown point.
- A box where one depth holds the majority of the pixels has a small spread, even if the rest is background. A 60/40 mix gets status `valid`.
- With a 30/70 object/background mix, B correctly picks the object, but the spread describes the background. The status is still `valid`.
- Only mixes with no majority depth, roughly 50/50, inflate the spread past `τ_spread`.
So the spread is not a reliable "mixed box" detector on its own. Spec 03.3 should consider an extra signal, such as `|B − A|` or the fraction of pixels in B's ±1-bin window, and choose it on val only.


**Decision (2026-11-11).** The cost bound is restated: < 5 ms per frame on real frames at `conf_operational` (met: fr1 has at most 11 boxes, about 4 ms), with per-box cost recorded (03.1 about 0.14 ms per box; 03.2 about 0.35 ms per box). The 50-box figure stays as a recorded stress number, not a pass/fail bound. Vectorising across boxes is an optional Phase 06 optimisation if the end-to-end budget needs it.
