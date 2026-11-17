# Spec 00.4 — Architecture, Risks, Compute Budget, and Acceptance Criteria

| Field | Value |
|---|---|
| Phase | 00 — Foundation |
| Status | **Drafted, awaiting approval** |
| Output documents | [docs/architecture.md](../../docs/architecture.md), [docs/evaluation_protocol.md](../../docs/evaluation_protocol.md), [docs/risks_and_limitations.md](../../docs/risks_and_limitations.md) |
| Depends on | 00.1, 00.2, 00.3 |
| Blocks | Phase 01 start |

## Objective
Fix the system design, the module interfaces, the coordinate conventions, the evaluation protocol, the risk register, and the compute and storage budget, so that implementation can start without open architectural questions.

## Requirements
- **R1.** Pipeline stages with the input/output types between them.
- **R2.** Coordinate frames and units, defined once and used everywhere.
- **R3.** Module layout matching the repo structure (`src/perception/...`).
- **R4.** Alternatives considered for each major choice, with the reason for the choice.
- **R5.** An evaluation protocol for every metric: data, matching rule, aggregation, and benchmark boundaries.
- **R6.** A risk register with likelihood, impact, and mitigation.
- **R7.** A compute and storage budget for the measured development machine.
- **R8.** Phase exit criteria for every phase 01–07.
- **R9.** A dependency list with license notes.

## Tasks
- [x] T1. Pipeline diagram and stage interfaces.
- [x] T2. Coordinate convention section.
- [x] T3. Module map, and the configuration files.
- [x] T4. Design decisions and the alternatives table.
- [x] T5. Evaluation protocol document.
- [x] T6. Risks and limitations document.
- [x] T7. Compute and storage budget.
- [x] T8. Phase exit criteria table.
- [ ] T9. User decisions D1–D6 (see `PROGRESS.md`).
- [ ] T10. User approval.

## Acceptance criteria
- **AC1.** Every stage interface names its data types, units, and frame.
- **AC2.** Every metric in the evaluation protocol names its ground-truth source and matching rule.
- **AC3.** Every benchmark states its timing boundary, warm-up, repetitions, and hardware.
- **AC4.** Every high-likelihood risk has a mitigation that does not depend on the user buying hardware.
- **AC5.** The storage budget fits in the measured free disk space (~102 GB).
- **AC6.** The user has approved the document and resolved D1–D6.

## Verification
- [x] V1. Each stage in `architecture.md` maps to a package in the module map.
- [x] V2. Each goal G1–G7 maps to a protocol section.
- [ ] V3. User sign-off is recorded in `PROGRESS.md`.

## Dependencies
- Specs 00.1–00.3.
