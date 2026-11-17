# Spec 00.3 — Dataset, License, Annotation, and Calibration Audit

| Field | Value |
|---|---|
| Phase | 00 — Foundation |
| Status | **Drafted; final checks deferred to Phase 01 (after download)** |
| Output document | [docs/dataset_audit.md](../../docs/dataset_audit.md) |
| Depends on | 00.1 (the goals define the data requirements) |
| Blocks | 00.4, and all of Phase 01 |

## Objective
Choose datasets that legally and technically support every MVP goal (G2–G5). Document the access, license, depth units, calibration, annotations, splits, and size of each one. Separate facts verified from official sources from facts still to be verified after download.

## Requirements
- **R1.** Evaluate at least NYU Depth V2, ScanNet, Hypersim, and two alternatives.
- **R2.** For each dataset, record:
  - the official source,
  - the access process,
  - the license or terms,
  - RGB and depth availability,
  - registration,
  - intrinsics,
  - depth units,
  - annotations,
  - sequences,
  - splits,
  - size,
  - suitability for each goal.
- **R3.** Every fact is labelled [Verified] (with a source) or [To verify].
- **R4.** Any license restriction that affects a public GitHub repo is flagged.
- **R5.** A coverage matrix maps each goal (G2–G5) to the dataset that measures it.
- **R6.** The MVP must not depend on a gated dataset.

## Tasks
- [x] T1. Check the official pages for all candidates.
- [x] T2. Record the depth encoding for TUM (×5000, registered) [Verified].
- [x] T3. Record the SUN RGB-D depth decoding (bit rotation, /1000) from two independent implementations.
- [x] T4. Build the goal coverage matrix.
- [x] T5. Write the repo data policy (what may and may not be committed).
- [ ] T6. **Phase 01:** verify the remaining [To verify] items on the downloaded files.
- [ ] T7. User decision on ScanNet (institutional email).
- [ ] T8. User approval of the dataset selection.

## Acceptance criteria
- **AC1.** Each goal G2–G5 is covered by at least one ungated dataset.
- **AC2.** Each selected dataset has a documented depth unit, an intrinsics source, and a split strategy.
- **AC3.** The license and terms of each selected dataset are recorded, with their consequence for the repo.
- **AC4.** All [To verify] items have an owner spec in Phase 01.
- **AC5.** The user has approved the selection.

## Verification
- [x] V1. Every [Verified] fact links to an official page or a paper.
- [x] V2. The coverage matrix has no MVP gap.
- [ ] V3. (Phase 01) Each [To verify] item is converted to [Verified], or corrected.

## Dependencies
- Internet access. Downloads begin in Phase 01, only with user approval for each file.
