# Spec 00.1 — Problem Statement, Users, and Measurable Goals

| Field | Value |
|---|---|
| Phase | 00 — Foundation |
| Status | **Drafted, awaiting approval** |
| Output document | [docs/problem_statement.md](../../docs/problem_statement.md) |
| Depends on | Nothing; this is the first spec |
| Blocks | 00.4 (architecture must serve these goals), all later phases |

## Objective
Define what the system must do, who it is for, what is in and out of scope, and how success is measured. The goals must be stated so that a number or a test can confirm them later. Vague goals like "accurate" or "fast" are not allowed.

## Requirements
- **R1.** State the problem in terms of system inputs and outputs.
- **R2.** Identify the primary users and the portfolio audience.
- **R3.** Every goal must be tied to a metric or a test, and must say what data it is measured on.
- **R4.** Separate the MVP scope from the extensions.
- **R5.** Mark every number as a [Target]. No results may be presented as achieved.
- **R6.** State the input assumption: RGB with *registered metric depth* and *known intrinsics*. Metric 3D from RGB alone is out of scope.
- **R7.** State the honest-uncertainty requirement: the system must report invalid depth, never a fabricated distance.

## Tasks
- [x] T1. Write the problem statement and input/output contract.
- [x] T2. Write user and stakeholder descriptions.
- [x] T3. Write goals G1–G7, each with a metric, data, and phase.
- [x] T4. Write the MVP and extension lists, and the explicit non-goals.
- [x] T5. Write the per-frame output contract (fields and status values).
- [ ] T6. User review and approval.

## Acceptance criteria
- **AC1.** Every goal has a metric, a data source, and the phase where it is measured.
- **AC2.** The MVP can be built without any gated dataset and without an NVIDIA GPU.
- **AC3.** No numeric value is stated as achieved.
- **AC4.** The output contract defines the `depth_status` values and what each one means.
- **AC5.** The user has approved the document.

## Verification (review checklist; this is a documentation spec, so there are no code tests)
- [x] V1. Search the document for numbers. Each one is labelled [Target] or [Verified].
- [x] V2. Each goal G1–G7 maps to at least one later spec.
- [x] V3. The MVP list contains no item that requires ScanNet or TensorRT.
- [ ] V4. User sign-off is recorded in `PROGRESS.md`.

## Dependencies
- Hardware facts measured on 2026-10-21 (see `docs/architecture.md` §7).
