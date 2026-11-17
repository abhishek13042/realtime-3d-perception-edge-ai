# Spec 04.2 — Label-free tracking evaluation on TUM

| Field | Value |
|---|---|
| Phase | 04 Tracking |
| Status | ✅ Approved 2026-11-11 |

## Objective
Measure how well identities persist without identity labels (evaluation protocol §4). Choose the tracker's free parameters on **fr1_desk** only, then report on **fr3_long_office_household** once, with the parameters frozen. Check the architecture's switching trigger: whether ID breaks correlate with camera motion.

## Requirements
- **R1. Object inventory (pseudo ground truth for counting only).**
  - I count the distinct static objects per class by eye from a contact sheet of every 30th frame of each sequence, and record them in `docs/tum_object_inventory.yaml` (class → count, with a note).
  - Only classes the detector actually reports are counted (from `tum_presence.json`). This is the protocol's "small annotated subset".
  - It is labelled as a manual count with its own uncertainty, and it is never used to tune anything except the fragmentation proxy on fr1.
- **R2. Metrics.** All of them are per sequence and per stride (1 and 6), over confirmed tracks (≥ `min_track_len`) of static classes.
  - **Label-free stability:** per track, the std-dev of the `valid` world-frame centres, as a 3D RMS about the median (m). Report the median and p90 over tracks with ≥ 5 valid centres.
  - **Fragmentation proxy:** confirmed tracks per inventory object, per class and overall.
  - **ID-switch proxy:** within a track, a jump in world position larger than `d_jump` between consecutive valid centres. A static object cannot move, so such a jump means the track moved to another object, or a depth outlier. `d_jump = 0.75 m`, about 3× the 03.3 valid median error; this is fixed now, before any run. A sensitivity check at 0.5 and 1.0 m is reported. The metric is jumps per 100 valid track-frames, and the gallery shows the 8 largest jumps so they can be sorted by eye into switch vs depth outlier.
  - **Recovery:** the fraction of gaps of 1–N processed frames (N = `track_buffer`) after which the same ID resumes (protocol §6).
  - **Coverage:** the share of `valid` 3D statuses among tracked boxes, and the frames with zero tracks.
- **R3. Tuning on fr1 only.**
  - Grid: `track_buffer` ∈ {15, 30, 60} (processed frames) × `match_thresh` ∈ {0.7, 0.8, 0.9} × class-aware/agnostic, at both strides.
  - The confidence thresholds stay fixed at 04.1 R3 (conf_operational).
  - Objective: minimise the fragmentation proxy, subject to an ID-switch proxy no higher than the Ultralytics-default configuration's (same stride). Ties within 0.05 tracks per object go to the default.
  - Each stride gets its own frozen values in `configs/tracker.yaml`.
- **R4. Split guard.** The tuning script refuses any sequence other than fr1_desk, and the reporting script refuses to tune. A test covers both.
- **R5. Camera-motion check (architecture switching trigger).**
  - Camera linear and angular speed per frame come from the GT poses.
  - Compare the speed at **track-loss events** (a confirmed track's last frame before a gap longer than the buffer) with the speed over all frames.
  - The trigger fires if the median speed at losses is > 1.5× the overall median, there are ≥ 20 loss events, and a Mann–Whitney p < 0.01.
  - If it fires, it is reported as a finding with the 3D-Kalman alternative. It is not built in this phase without approval.
- **R6. Report.** `docs/tracking_results.md` is generated from `benchmarks/tracking/*.json` and contains:
  - tables per sequence × stride;
  - the tuning table;
  - a stability histogram;
  - the jump gallery;
  - a camera-speed plot with the loss events marked;
  - a limitations section: no identity labels; the static assumption; the inventory is manual; jumps mix depth outliers with switches.
- **R7. Reporting run.** Exactly one run on fr3 with the frozen config. If anything is changed after it, the change is logged in `PROGRESS.md` and the run is repeated in full.

## Tasks
- [x] T1. Contact sheets and `docs/tum_object_inventory.yaml` (R1).
- [x] T2. `perception/eval/tracking.py` (metrics R2, motion check R5).
- [x] T3. `scripts/eval_tracking.py --stage tune|report` with the split guard (R3, R4, R7).
- [x] T4. Tune on fr1; freeze; report on fr3; generate the doc (R6).

## Acceptance criteria
- **AC1.** Tuning is done on fr1 only (guard plus test), and the chosen values and the reason are stated.
- **AC2.** fr3 results are reported for both strides: stability (median and p90), fragmentation, ID-switch proxy (with the d_jump sensitivity), recovery and coverage.
- **AC3.** The 8 largest jumps are categorised by eye.
- **AC4.** The camera-motion trigger is evaluated, with the outcome stated.
- **AC5. Phase 04 exit.** IDs persist (fragmentation proxy reported, not hidden); limitations documented; all tests pass.

## Tests (`tests/test_tracking_eval.py`)
- Stability of a synthetic static track with known noise σ: the measured RMS ≈ √3·σ, within the sampling tolerance.
- The jump detector fires on an injected 1 m jump and not on 0.2 m noise.
- Recovery counting on a hand-made gap pattern.
- The motion-trigger rule on synthetic speeds (fires and does not fire).
- The split guard rejects fr3 for tuning.

## Dependencies
04.1; TUM GT poses (01.2 `PoseInterpolator`); scipy (Mann–Whitney); matplotlib.

## Results (2026-11-11, measured; full report in [docs/tracking_results.md](../../docs/tracking_results.md))
- **AC1 ✅** Tuning ran on fr1 only (`split_guard`, plus a subprocess test that the script refuses fr3).
  - Stride 1: buffer 60, match 0.9, class-aware. Fragmentation 6.50 → 4.72 per object; jump rate 0.100 vs the reference's 0.117.
  - Stride 6: buffer 15, match 0.7, class-aware. Fragmentation 1.28 → 1.00; 0 jumps. **Fragile choice:** buffer 30 / 0.7 scores 1.06, so the winner is decided by one track on 96 frames.
  - Class-agnostic at stride 6 had 3–5 jumps per 100 and a stability median of 17–30 cm, so class-aware won at both strides.
- **AC2 ✅** fr3, reported once with the frozen config:

  | fr3 | stride 1 | stride 6 |
  |---|---|---|
  | static confirmed tracks | 237 | 139 |
  | stability median / p90 | 2.9 / 14.8 cm | 4.3 / 18.8 cm |
  | fragmentation (tracks / object) [inventory range] | 5.00 [4.00, 6.67] | 3.16 [2.53, 4.21] |
  | jumps per 100 valid pairs, d_jump 0.5 / **0.75** / 1.0 m | 0.18 / **0.14** / 0.04 | 0.76 / **0.29** / 0.00 |
  | recovery | 0.69 (482 resumed / 216 lost) | 0.47 (104 / 119) |
  | valid 3D share | 0.965 | 0.971 |

- **AC3 ✅** The 16 largest jumps were reviewed by eye (`docs/tracking_jump_review.yaml`): 13 depth outliers, 3 switches.
  - Depth outliers: chairs truncated at the image edge, thin chair backs and bottles with background bleed.
  - Switches: all 3 are at stride 6 (a monitor after a camera turn; two bottle-to-bottle switches after gaps).
  - Two of the three switches are **below** d_jump (0.70 and 0.51 m). The 0.75 m proxy undercounts same-class switches between nearby objects, as the limitations section says.
- **AC4 ✅** The camera-motion trigger does **not** fire at either stride.
  - Stride 1, angular speed: ratio 1.16, p 0.0005.
  - Stride 6, linear speed: ratio 1.14, p 0.0003.
  - These are significant but well under 1.5×. 2D association stays.
- **AC5 ✅** Fragmentation is reported prominently and is high (5.0 tracks per object at stride 1). Limitations are documented. 146 tests pass.
- R7 note: after the tuning run, the tracker.yaml *comments* were edited to mark the freeze. Because of that, the sha in `eval_fr1_tune.json` differs from the sha of the reporting runs. The parameters did not change. All fr1/fr3 tracking runs and both reports were produced after that edit and match its sha.
