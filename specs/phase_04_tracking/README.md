# Phase 04 — Tracking

**Status:** ✅ Approved 2026-11-11. Results: [docs/tracking_results.md](../../docs/tracking_results.md).

**Exit condition (docs/architecture.md §5):** IDs persist on TUM sequences; ID switches and fragmentation are computed; limitations are documented.

| Spec | File | Main outputs |
|---|---|---|
| 04.1 | [spec_04_1_tracker.md](spec_04_1_tracker.md) | `perception/tracking/{bytetrack.py,track_history.py}`, `configs/tracker.yaml`, `scripts/run_tracking.py` |
| 04.2 | [spec_04_2_tracking_eval_tum.md](spec_04_2_tracking_eval_tum.md) | `perception/eval/tracking.py`, `scripts/eval_tracking.py`, `docs/tracking_results.md`, `docs/tum_object_inventory.yaml` |

**Scope rules for this phase**
- **Offline, on stored detections.** The Phase 02 TUM runs (`benchmarks/detection/tum_fr1_desk_full.jsonl`, 573 frames; `tum_fr3_long_office_full.jsonl`, 2485 frames; conf ≥ 0.001) are reused, with no detector re-runs. Depth and 3D centres come from the Phase 03 code with `configs/depth.yaml` frozen.
- **Tuning vs reporting split (evaluation protocol §1):**
  - **fr1_desk** is the tuning sequence;
  - **fr3_long_office_household** is the reporting sequence.
  - Nothing is tuned on fr3. The tuning script refuses it, like the 03.3 split guard.
- **No identity labels.** TUM has none, so G5 is reported as **"label-free stability"**, never as HOTA/IDF1 (protocol §4). ScanNet is out of scope (not downloaded; licence-gated).
- **Two frame rates.** The sensor runs at 30 Hz, but the CPU pipeline runs at about 5–6 FPS (detector 151–180 ms). Every metric is reported at **stride 1** (all frames) and at **stride 6** (≈ 5 Hz, the frames a real-time CPU run would see). Larger gaps between frames are what break IoU association under camera motion.
- **Static-object assumption.** `person` and other movable classes listed in `configs/tracker.yaml` are excluded from the world-frame metrics.
- **No training.** Everything runs on CPU and is classical.
- **One new dependency.** ByteTrack comes from the installed `ultralytics` 8.4.208 (`ultralytics.trackers.byte_tracker.BYTETracker`, AGPL-3.0, the same licence the project already carries through YOLO26). Its matching module needs `lap` and pip-installs it at import time when missing. To avoid that hidden install, `lap==0.5.14` (BSD-2, cp314 wheel from PyPI) was installed explicitly and pinned in `pyproject.toml` (2026-11-11). The adapter checks that `lap` imports before it imports the tracker, so a missing package fails loudly instead of being installed silently.

**Template:** Objective · Requirements · Tasks · Acceptance criteria · Tests · Dependencies.
